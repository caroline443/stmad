"""
PSTG — Progressive Spatiotemporal Graph Modelling for Spacecraft Anomaly Detection
Chen et al., Entropy 2026 (entropy-28-00426)

严格按照论文公式实现，用于复现基线结果。

架构：
  1. Multi-scale Patch Embedding + Gated Attention Fusion   (公式 3-12)
  2. Progressive Spatiotemporal Graph Reasoning × n_L=2    (公式 13-24)
     G_graph: 学习稀疏邻接矩阵 A^h = ReLU(E1 @ E2^T)      (公式 16-18)
     G_atm:   邻接引导的图注意力                           (公式 19-24)
  3. Forecast Head: 线性投影 → 预测未来 F=10 步            (公式 2)

节点定义：n = C × N（C 个通道 × N=L/p_main 个 patch token）
         e.g. C=6, N=10 → n=60 个时空节点

Loss: MSE + λ_shape * shape_loss                          (公式 25)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Multi-scale Patch Embedding with Gated Attention Fusion ───────────────

class PSTGPatchEmbedding(nn.Module):
    """公式 3-12：多尺度 patch embedding + gated attention fusion。

    输入: X ∈ R^(B, L, C)
    输出: Z_fused ∈ R^(B, N, C, D)  →  reshape → (B, n, D), n = N*C
    """

    def __init__(
        self,
        window_size: int,
        patch_sizes: list[int],
        n_sensors: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_sizes = patch_sizes
        self.p_main  = patch_sizes[0]
        self.N       = window_size // self.p_main   # 主尺度 patch 数 = L/p_main = 10
        self.C       = n_sensors
        self.D       = d_model
        self.n       = self.N * n_sensors           # 时空节点总数 = 60

        # 每个尺度的 Linear 投影 (公式 8)
        self.patch_projectors = nn.ModuleList([
            nn.Linear(p, d_model) for p in patch_sizes
        ])

        # 位置编码（固定 sinusoidal，公式 9）
        self.register_buffer("pos_enc", self._make_pos_enc(self.N, d_model))

        # Gated Attention Fusion：把 K 个尺度融合为一个 (公式 10-12)
        K = len(patch_sizes)
        self.gate_linear = nn.Linear(K * d_model, K)  # → softmax attention

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _make_pos_enc(N: int, D: int) -> torch.Tensor:
        """正弦位置编码 (公式 9)。"""
        pe  = torch.zeros(N, D)
        pos = torch.arange(N).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:D//2])
        return pe   # (N, D)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, L, C)
        返回: Z ∈ (B, n, D)  n = N*C
        """
        B, L, C = X.shape
        scale_embeds = []

        for p, proj in zip(self.patch_sizes, self.patch_projectors):
            n_p = L // p
            # (B, L, C) → (B, n_p, p, C) → (B, n_p, C, p)
            x_p = X[:, :n_p*p, :].reshape(B, n_p, p, C).permute(0, 1, 3, 2)
            h   = proj(x_p)              # (B, n_p, C, D)

            # 上采样到 N
            if n_p != self.N:
                rf = self.N // n_p
                h  = h.repeat_interleave(rf, dim=1)   # (B, N, C, D)

            # 加位置编码 (公式 8)
            h = h + self.pos_enc.unsqueeze(1)         # (B, N, C, D)
            scale_embeds.append(h)                    # K × (B, N, C, D)

        # Gated Attention Fusion (公式 10-12)
        # 把 K 个 scale 的 embed 拼接后算 attention weight
        stacked = torch.stack(scale_embeds, dim=-2)           # (B, N, C, K, D)
        concat  = stacked.reshape(B, self.N, C, -1)           # (B, N, C, K*D)
        alpha   = F.softmax(self.gate_linear(concat), dim=-1) # (B, N, C, K)
        alpha   = alpha.unsqueeze(-1)                          # (B, N, C, K, 1)

        Z_fused = (stacked * alpha).sum(dim=-2)               # (B, N, C, D)
        Z_fused = self.norm(self.dropout(Z_fused))

        # reshape 成节点序列
        Z = Z_fused.permute(0, 2, 1, 3).reshape(B, self.n, self.D)  # (B, n, D)
        return Z


# ── 2. G_graph: 学习稀疏邻接矩阵 ──────────────────────────────────────────────

class GraphStructureLearner(nn.Module):
    """公式 16-18：对每个 head 生成稀疏行随机邻接矩阵。

    A^h_dense = ReLU(E1^h @ (E2^h)^T)
    A^h_final = softmax(top-γ sparse A^h_dense) + dropout
    """

    def __init__(self, d_model: int, n_heads: int = 4, gamma: float = 0.1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.gamma   = gamma
        d_head = d_model // n_heads

        self.W1 = nn.ModuleList([nn.Linear(d_model, d_head, bias=False) for _ in range(n_heads)])
        self.W2 = nn.ModuleList([nn.Linear(d_model, d_head, bias=False) for _ in range(n_heads)])
        self.drop = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor) -> list[torch.Tensor]:
        """
        H: (B, n, D)
        返回: list of H 个 A^h ∈ (B, n, n)
        """
        n = H.size(1)
        k = max(1, int(self.gamma * n))
        adj_list = []

        for w1, w2 in zip(self.W1, self.W2):
            E1 = w1(H)                                     # (B, n, d_head)
            E2 = w2(H)
            A  = F.relu(torch.bmm(E1, E2.transpose(1, 2))) # (B, n, n) 公式 17

            # Top-γ 掩码 (公式 18)
            if k < n:
                thr = A.topk(k, dim=-1).values[..., -1, None]
                A   = A.masked_fill(A < thr, 0.0)

            # 行归一化（避免全零行）
            row_sum = A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            A       = A / row_sum
            A       = self.drop(A)
            adj_list.append(A)

        return adj_list


# ── 3. G_atm: 邻接引导的多头图注意力 ─────────────────────────────────────────

class AdjacencyGuidedAttention(nn.Module):
    """公式 19-24：以学到的 A^h 作为结构先验的图注意力。

    e^h_ij = A^h_ij * LeakyReLU(w_A * (Q^h_i || K^h_j))
    a^h_ij = softmax_j(e^h_ij)
    M^h_i  = Σ_j a^h_ij * V^h_j
    Z_out  = LayerNorm(H + concat_h(M^h) @ W_O)
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.W_Q = nn.ModuleList([nn.Linear(d_model, self.d_head, bias=False) for _ in range(n_heads)])
        self.W_K = nn.ModuleList([nn.Linear(d_model, self.d_head, bias=False) for _ in range(n_heads)])
        self.W_V = nn.ModuleList([nn.Linear(d_model, self.d_head, bias=False) for _ in range(n_heads)])
        self.w_A  = nn.ParameterList([nn.Parameter(torch.randn(2 * self.d_head)) for _ in range(n_heads)])
        self.W_O  = nn.Linear(d_model, d_model, bias=False)

        self.leaky = nn.LeakyReLU(0.2)
        self.drop  = nn.Dropout(dropout)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, H: torch.Tensor, adj_list: list[torch.Tensor]) -> torch.Tensor:
        """
        H:        (B, n, D)
        adj_list: H 个 (B, n, n)
        返回:     (B, n, D)
        """
        B, n, D = H.shape
        heads_out = []

        for h_idx, (wq, wk, wv, wa, A) in enumerate(
            zip(self.W_Q, self.W_K, self.W_V, self.w_A, adj_list)
        ):
            Q = wq(H)   # (B, n, d_head)
            K = wk(H)
            V = wv(H)

            # e_ij = A_ij * LeakyReLU(w_A · (Q_i || K_j))  公式 20
            # 构造所有 (i,j) 对
            Q_exp = Q.unsqueeze(2).expand(B, n, n, self.d_head)  # (B, n, n, d_head)
            K_exp = K.unsqueeze(1).expand(B, n, n, self.d_head)
            QK    = torch.cat([Q_exp, K_exp], dim=-1)             # (B, n, n, 2*d_head)
            e     = self.leaky((QK * wa).sum(dim=-1))             # (B, n, n)
            e     = A * e                                          # 公式 20：邻接加权

            # softmax（只在邻居上）公式 21
            # 把零邻接变成 -inf 避免影响 softmax
            mask   = (A == 0)
            e      = e.masked_fill(mask, float("-inf"))
            # 处理全零行
            all_inf = mask.all(dim=-1, keepdim=True)
            e       = e.masked_fill(all_inf, 0.0)
            attn    = F.softmax(e, dim=-1)
            attn    = self.drop(attn)

            # 聚合 公式 22
            m = torch.bmm(attn, V)    # (B, n, d_head)
            heads_out.append(m)

        M     = torch.cat(heads_out, dim=-1)   # (B, n, D)  公式 23
        M     = self.W_O(M)
        Z_out = self.norm(H + M)               # 公式 24
        return Z_out


# ── 4. PSTG Block = G_graph + G_atm ──────────────────────────────────────────

class PSTGBlock(nn.Module):
    """一层 Progressive Spatiotemporal Graph Reasoning (公式 13-24)。"""

    def __init__(self, d_model: int, n_heads: int = 4, gamma: float = 0.1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.g_graph = GraphStructureLearner(d_model, n_heads, gamma, dropout)
        self.g_atm   = AdjacencyGuidedAttention(d_model, n_heads, dropout)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        adj_list = self.g_graph(H)
        H        = self.g_atm(H, adj_list)
        return H


# ── 5. Forecast Head ──────────────────────────────────────────────────────────

class PSTGForecastHead(nn.Module):
    """从时空节点表示预测未来 F 步 (公式 2 中的 T_Θ)。

    H_final: (B, n, D) = (B, N*C, D)
    → reshape: (B, C, N, D)
    → 对每个 channel 的 N 个 token 做全局池化
    → Linear: D → F
    → 输出: (B, F, C)
    """

    def __init__(self, d_model: int, n_patch: int, n_sensors: int,
                 forecast_horizon: int) -> None:
        super().__init__()
        self.N = n_patch
        self.C = n_sensors
        self.F = forecast_horizon
        # 用最后一个 patch token 预测，对应 τ=1（预测紧接的未来步）
        self.proj = nn.Linear(d_model, forecast_horizon)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """H: (B, n, D) = (B, N*C, D)"""
        B, n, D = H.shape
        # reshape: (B, C, N, D)
        H_r = H.reshape(B, self.C, self.N, D)
        # 取最后一个 patch token（对应 τ=1 原则）
        h_last = H_r[:, :, -1, :]          # (B, C, D)
        out    = self.proj(h_last)          # (B, C, F)
        out    = out.permute(0, 2, 1)       # (B, F, C)
        return out


# ── 6. 完整 PSTG 模型 ─────────────────────────────────────────────────────────

class PSTG(nn.Module):
    """完整 PSTG 模型（严格按论文公式实现）。

    输入:  X_ctx ∈ R^(B, L, C)
    输出:  X̂_fut ∈ R^(B, F, C)
    """

    def __init__(
        self,
        n_sensors:        int,
        window_size:      int   = 250,
        forecast_horizon: int   = 10,
        patch_sizes:      list  | None = None,
        d_model:          int   = 512,
        n_heads:          int   = 4,
        n_layers:         int   = 2,
        gamma:            float = 0.1,
        dropout:          float = 0.1,
    ) -> None:
        super().__init__()
        if patch_sizes is None:
            patch_sizes = [25, 50, 125]

        p_main  = patch_sizes[0]
        N       = window_size // p_main   # 10

        self.patch_embed = PSTGPatchEmbedding(
            window_size=window_size,
            patch_sizes=patch_sizes,
            n_sensors=n_sensors,
            d_model=d_model,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList([
            PSTGBlock(d_model, n_heads, gamma, dropout)
            for _ in range(n_layers)
        ])
        self.forecast_head = PSTGForecastHead(
            d_model=d_model,
            n_patch=N,
            n_sensors=n_sensors,
            forecast_horizon=forecast_horizon,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Z = self.patch_embed(X)        # (B, n, D)
        for block in self.blocks:
            Z = block(Z)               # (B, n, D)
        X_hat = self.forecast_head(Z)  # (B, F, C)
        return X_hat

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 7. Loss 函数（公式 25）────────────────────────────────────────────────────

def pstg_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    lambda_shape: float = 0.05,
) -> torch.Tensor:
    """公式 25: L = MSE + λ_shape * shape_loss

    pred / target: (B, F, C)
    shape_loss: 时间梯度误差，保留预测的动态属性
    """
    mse = F.mse_loss(pred, target)

    # 时间梯度损失（公式 25 第三项，λ_1=0 → 不用频域损失）
    if lambda_shape > 0 and target.size(1) > 1:
        grad_pred   = pred[:, 1:, :] - pred[:, :-1, :]
        grad_target = target[:, 1:, :] - target[:, :-1, :]
        shape_loss  = F.mse_loss(grad_pred, grad_target)
        return mse + lambda_shape * shape_loss

    return mse


# ── 8. build_pstg factory ──────────────────────────────────────────────────────

def build_pstg(config: dict) -> PSTG:
    return PSTG(
        n_sensors        = config["n_sensors"],
        window_size      = config.get("window_size", 250),
        forecast_horizon = config.get("forecast_horizon", 10),
        patch_sizes      = config.get("patch_sizes", [25, 50, 125]),
        d_model          = config.get("d_model", 512),
        n_heads          = config.get("n_heads", 4),
        n_layers         = config.get("n_layers", 2),
        gamma            = config.get("gamma", 0.1),
        dropout          = config.get("dropout", 0.1),
    )
