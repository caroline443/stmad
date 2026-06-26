"""
时空图模块（论文 Section 3.2.2，公式 13-24）

包含两个子算子：
G_Θ2^(l) = G_attn ∘ G_graph

1. G_graph: 动态时空图构建（公式 15-18）
   - 多头（H=4）动态邻接矩阵学习
   - Top-k 稀疏化 + Softmax 归一化 + Dropout

2. G_attn: 结构引导的图注意力（公式 19-24）
   - 改进 GATv2：直接用邻接矩阵调制注意力分数
   - 多头 → 拼接 → W_O 投影 → 残差 + LayerNorm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  动态图构建（G_graph）
# ─────────────────────────────────────────────────────────────────────────────

class DynamicGraphLearner(nn.Module):
    """
    多头动态邻接矩阵学习（公式 15-18）

    输入：Z ∈ R^(B, n, D)
    输出：A_final ∈ R^(B, H, n, n)

    步骤：
    1. 将 Z 按头分割 → H 份 Z^(h) ∈ R^(B, n, D/H)
    2. 共享线性变换 W1, W2 → E1, E2 ∈ R^(B, n, D/H)
    3. A_dense^(h) = ReLU(E1 @ E2^T)
    4. Top-k 稀疏化（γ=0.1，保留 k=ceil(γ·n) 个连接）
    5. Softmax 归一化
    6. Dropout
    """

    def __init__(
        self,
        d_model: int,       # D = 512
        num_heads: int,     # H = 4
        top_k: int,         # k = ceil(γ·n) = 6
        dropout: float = 0.1,
    ):
        super().__init__()
        self.H = num_heads
        self.head_dim = d_model // num_heads   # D/H = 128
        self.top_k = top_k

        # 共享权重 W1, W2 ∈ R^(D/H × D/H)（论文：两矩阵在所有头间共享）
        self.W1 = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W2 = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, n, D]
        Returns:
            A_final: [B, H, n, n]
        """
        B, n, D = z.shape

        # 1. 分成 H 份：[B, n, H, D/H] → [B, H, n, D/H]
        z_heads = z.reshape(B, n, self.H, self.head_dim).permute(0, 2, 1, 3)
        # z_heads: [B, H, n, D/H]

        # 2. 线性变换（公式 16）
        E1 = self.W1(z_heads)   # [B, H, n, D/H]
        E2 = self.W2(z_heads)   # [B, H, n, D/H]

        # 3. 密集邻接矩阵（公式 17）
        # A_dense = ReLU(E1 @ E2^T)
        A_dense = torch.matmul(E1, E2.transpose(-2, -1))   # [B, H, n, n]
        A_dense = F.relu(A_dense)

        # 4. Top-k 稀疏化：对每行保留 top-k 值，其余设为 -inf
        k = min(self.top_k, n)
        if k < n:
            # topk 返回 values 和 indices
            topk_vals, topk_idx = torch.topk(A_dense, k, dim=-1)  # [B, H, n, k]
            # 构造 -inf 掩码
            A_mask = torch.full_like(A_dense, float("-inf"))
            A_mask.scatter_(-1, topk_idx, topk_vals)
        else:
            A_mask = A_dense

        # 5. Softmax 归一化（公式 18）
        # 注意：-inf 位置经 softmax 后变为 0
        A_norm = F.softmax(A_mask, dim=-1)  # [B, H, n, n]
        # 将 nan 替换为 0（若整行都是 -inf 则 softmax 输出 nan）
        A_norm = torch.nan_to_num(A_norm, nan=0.0)

        # 6. Dropout
        A_final = self.dropout(A_norm)      # [B, H, n, n]

        return A_final


# ─────────────────────────────────────────────────────────────────────────────
#  结构引导的图注意力（G_attn）
# ─────────────────────────────────────────────────────────────────────────────

class StructureGuidedGraphAttention(nn.Module):
    """
    改进的 GATv2：用学得的邻接矩阵直接调制注意力分数（公式 19-24）

    核心区别：
    - 标准 GATv2：attention score = LeakyReLU(w_A · Linear([Q||K]))
    - 本文改进：score = A_final_ij × LeakyReLU(w_A · [Q_i||K_j])
      即邻接矩阵作为 inductive bias 乘入注意力分数

    输入：Z ∈ R^(B, n, D)，A_final ∈ R^(B, H, n, n)
    输出：Z_out ∈ R^(B, n, D)
    """

    def __init__(
        self,
        d_model: int,       # D = 512
        num_heads: int,     # H = 4
        dropout: float = 0.1,
    ):
        super().__init__()
        self.H = num_heads
        self.head_dim = d_model // num_heads   # D/H = 128

        # Q, K, V 投影（三个线性层，公式 19 中的 W_Q, W_K, W_V）
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # 输出投影 W_O ∈ R^(D×D)（公式 23）
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # 每个头的注意力向量 w_A ∈ R^(2·D/H)（公式 20，跨头共享）
        self.w_A = nn.Parameter(torch.empty(1, 1, 1, 1, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.w_A)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        z: torch.Tensor,          # [B, n, D]（即 H^[l-1]）
        A_final: torch.Tensor,    # [B, H, n, n]
    ) -> torch.Tensor:
        """
        Returns:
            H_out: [B, n, D]（即 H^[l]）
        """
        B, n, D = z.shape

        # 1. 线性投影 Q, K, V
        Q = self.W_Q(z)   # [B, n, D]
        K = self.W_K(z)   # [B, n, D]
        V = self.W_V(z)   # [B, n, D]

        # 2. 按头分割：[B, H, n, D/H]
        def split_heads(t):
            return t.reshape(B, n, self.H, self.head_dim).permute(0, 2, 1, 3)

        Q = split_heads(Q)   # [B, H, n, D/H]
        K = split_heads(K)   # [B, H, n, D/H]
        V = split_heads(V)   # [B, H, n, D/H]

        # 3. 计算结构引导的注意力分数（公式 20）
        # 对所有 (i,j) 对：[Q_i || K_j] ∈ R^(2·D/H)
        # Q: [B, H, n, D/H] → Q_i broadcast: [B, H, n, 1, D/H]
        # K: [B, H, n, D/H] → K_j broadcast: [B, H, 1, n, D/H]
        Q_i = Q.unsqueeze(-2)                              # [B, H, n, 1, D/H]
        K_j = K.unsqueeze(-3)                              # [B, H, 1, n, D/H]
        QK_cat = torch.cat([Q_i.expand(-1,-1,n,n,-1),
                            K_j.expand(-1,-1,n,n,-1)], dim=-1)  # [B, H, n, n, 2·D/H]

        # 点积：w_A ∈ [1,1,1,1,2·D/H] → 标量分数
        raw_score = (self.w_A * QK_cat).sum(dim=-1)        # [B, H, n, n]
        raw_score = self.leaky_relu(raw_score)

        # 用邻接矩阵调制（公式 20）
        e = A_final * raw_score                             # [B, H, n, n]

        # 4. 邻邻居 mask：A_final 中为 0 的位置对应不存在的边，设为 -inf
        # 这样 softmax 后这些位置权重为 0
        mask = (A_final == 0)
        e = e.masked_fill(mask, float("-inf"))

        # 5. Softmax 归一化（公式 21）
        alpha = F.softmax(e, dim=-1)                        # [B, H, n, n]
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.attn_dropout(alpha)

        # 6. 消息聚合（公式 22）
        M = torch.matmul(alpha, V)                          # [B, H, n, D/H]

        # 7. 多头拼接（公式 23）
        M = M.permute(0, 2, 1, 3).reshape(B, n, D)         # [B, n, D]
        M = self.W_O(M)                                     # [B, n, D]

        # 8. 残差 + LayerNorm（公式 24）
        out = self.layer_norm(z + M)                        # [B, n, D]

        return out


# ─────────────────────────────────────────────────────────────────────────────
#  单层时空图推理（G_Θ2^(l)）
# ─────────────────────────────────────────────────────────────────────────────

class SpatioTemporalGraphLayer(nn.Module):
    """
    一个 Progressive 层 G_Θ2^(l) = G_attn ∘ G_graph（公式 13-14）

    先动态构建邻接矩阵，再做结构引导的图注意力。
    """

    def __init__(self, d_model: int, num_heads: int, top_k: int, dropout: float = 0.1):
        super().__init__()
        self.graph_learner = DynamicGraphLearner(d_model, num_heads, top_k, dropout)
        self.graph_attn = StructureGuidedGraphAttention(d_model, num_heads, dropout)

    def forward(self, z: torch.Tensor) -> tuple:
        """
        Args:
            z: [B, n, D]
        Returns:
            (z_out, A_final): [B,n,D], [B,H,n,n]
        """
        A_final = self.graph_learner(z)    # [B, H, n, n]
        z_out = self.graph_attn(z, A_final)
        return z_out, A_final
