"""
多尺度 Patch 嵌入模块（论文 Section 3.2.1，公式 5-12）

算子 P_Θ1 = Fuse ∘ Embed ∘ Π : R^(C×L) → R^(C×N×D)

三个子步骤：
1. Π_P(·)      —— 多尺度 Patch 分割（公式 6）
2. Embed_Θ(·)  —— 线性投影 + 正弦位置编码（公式 7-9）
3. Fuse_gate(·)—— 门控注意力融合（公式 10-12）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEncoding(nn.Module):
    """
    固定正弦位置编码（公式 9）：
      (p_i)_j = sin(i / θ^(2j/D))  if j is even
              = cos(i / θ^(j-1)/D) if j is odd
    其中 θ=10000（Transformer 默认值）
    """

    def __init__(self, d_model: int, max_len: int = 1000, theta: float = 10000.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, D]
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(theta) / d_model)
        )  # [D/2]
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., N, D] → 加上前 N 个位置编码"""
        n = x.size(-2)
        return x + self.pe[:n]


class ScalePatchEmbedding(nn.Module):
    """
    单个尺度的 Patch 嵌入：
      1. 用滑动窗口提取 N 个 patch（每个长度为 patch_size）
      2. 线性投影到 d_model 维
      3. 加正弦位置编码
    """

    def __init__(self, patch_size: int, d_model: int, n_patches: int, context_len: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = n_patches
        # stride h_k = floor((L - p_k) / (N - 1))
        if n_patches > 1:
            self.stride = (context_len - patch_size) // (n_patches - 1)
        else:
            self.stride = context_len
        self.stride = max(1, self.stride)

        # 线性投影 W_k: p_k → D（公式 8）
        self.proj = nn.Linear(patch_size, d_model)
        # 正弦位置编码（公式 9）
        self.pos_enc = SinusoidalPositionEncoding(d_model, max_len=n_patches + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]
        Returns:
            z: [B, C, N, D]
        """
        B, C, L = x.shape
        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            end = start + self.patch_size
            # 防止越界：末尾做 padding
            if end <= L:
                patch = x[:, :, start:end]           # [B, C, p]
            else:
                pad_len = end - L
                patch = F.pad(x[:, :, start:], (0, pad_len))  # [B, C, p]
            patches.append(patch)

        # [B, C, N, p]
        patches = torch.stack(patches, dim=2)
        # 线性投影：[B, C, N, p] → [B, C, N, D]
        z = self.proj(patches)
        # 位置编码：沿 N 维
        z = self.pos_enc(z)
        return z


class MultiScalePatchEmbedding(nn.Module):
    """
    多尺度 Patch 嵌入 + 门控注意力融合（公式 5-12）

    输入：X ∈ R^(B, C, L)
    输出：Z_fused ∈ R^(B, n, D)，其中 n = C × N
    """

    def __init__(
        self,
        patch_sizes: list,      # P = [25, 50, 125]
        patch_main: int,        # p_main = 25
        d_model: int,           # D = 512
        context_len: int,       # L = 250
        n_channels: int,        # C = 6
    ):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.K = len(patch_sizes)
        self.d_model = d_model
        self.n_patches = context_len // patch_main   # N = 10
        self.n_channels = n_channels

        # 每个尺度的 Patch 嵌入
        self.scale_embeds = nn.ModuleList([
            ScalePatchEmbedding(p, d_model, self.n_patches, context_len)
            for p in patch_sizes
        ])

        # 门控注意力融合（公式 11-12）
        # 输入：拼接 K 个尺度 → [B, C, N, K*D]
        # 输出：K 个权重（softmax 后加权求和）
        self.gate_proj = nn.Linear(self.K * d_model, self.K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]
        Returns:
            z_fused: [B, n, D]，n = C × N
        """
        # 1. 对每个尺度提取嵌入
        scale_zs = []
        for embed in self.scale_embeds:
            z = embed(x)                 # [B, C, N, D]
            scale_zs.append(z)

        # 2. 拼接所有尺度：[B, C, N, K*D]
        z_cat = torch.cat(scale_zs, dim=-1)

        # 3. 计算门控权重（公式 12）
        alpha = self.gate_proj(z_cat)    # [B, C, N, K]
        alpha = F.softmax(alpha, dim=-1) # [B, C, N, K]

        # 4. 加权求和（公式 11）
        z_stack = torch.stack(scale_zs, dim=-1)   # [B, C, N, D, K]
        # alpha: [B, C, N, K] → [B, C, N, 1, K]
        alpha = alpha.unsqueeze(-2)
        z_fused = (z_stack * alpha).sum(dim=-1)   # [B, C, N, D]

        # 5. reshape 为 [B, n, D]，n = C × N
        B, C, N, D = z_fused.shape
        z_fused = z_fused.reshape(B, C * N, D)

        return z_fused
