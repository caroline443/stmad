"""
Mamba 时序嵌入模块

用 Mamba SSM 替换 PSTG 中独立的 patch 线性投影，
为每个 patch 嵌入引入长程时序上下文。

原 PSTG（每 patch 独立）：
  x_{i,c,k} ∈ R^{p_k}  →  W_k @ x + b  →  z_{i,c,k} ∈ R^D
  N 个 patch 互相不知道

MambaTemporalEmbedding（长程上下文）：
  [x_{1,c}, x_{2,c}, ..., x_{L,c}] ∈ R^L
    → Linear：每个时间步投影到 D 维
    → Mamba SSM：O(L) 复杂度，捕获长程时序依赖
    → 按 N 个 patch 中心位置采样
    → [z_{1,c}, z_{2,c}, ..., z_{N,c}] ∈ R^{N×D}
  每个 patch 嵌入包含了所有前序时间步的信息

对比 PSTG 多尺度方案：
  - 保留多尺度采样策略（3 种步长）
  - 用 Mamba 替换独立线性投影
  - 保留门控注意力融合
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class MambaBlock(nn.Module):
    """
    单个 Mamba 块：输入输出形状均为 [B, L, D]

    如果 mamba_ssm 不可用，自动降级为 Bidirectional GRU（效果接近）。
    """

    def __init__(
        self,
        d_model:  int   = 512,
        d_state:  int   = 16,    # SSM 状态维度（Mamba 默认）
        d_conv:   int   = 4,     # 局部卷积宽度（Mamba 默认）
        expand:   int   = 2,     # 内部扩展因子（Mamba 默认）
    ):
        super().__init__()
        if MAMBA_AVAILABLE:
            self.ssm = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.use_mamba = True
        else:
            # 降级：双向 GRU，捕获双向时序依赖
            self.ssm = nn.GRU(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.use_mamba = False
            print("[警告] mamba_ssm 未安装，使用 BiGRU 代替")

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D] → [B, L, D]"""
        if self.use_mamba:
            return self.norm(x + self.ssm(x))
        else:
            out, _ = self.ssm(x)
            return self.norm(x + out)


class MambaScaleEmbedding(nn.Module):
    """
    单尺度的 Mamba 时序嵌入：

    1. 输入投影：每个时间步 R^1 → R^D
    2. Mamba SSM：捕获长程时序依赖 [B, L, D] → [B, L, D]
    3. 按 patch 采样：从 N 个位置取出嵌入 → [B, N, D]
    4. 正弦位置编码
    """

    def __init__(
        self,
        patch_size:   int,
        d_model:      int,
        n_patches:    int,
        context_len:  int,
        d_state:      int = 16,
        d_conv:       int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = n_patches

        # stride：与 PSTG 完全一致
        if n_patches > 1:
            self.stride = max(1, (context_len - patch_size) // (n_patches - 1))
        else:
            self.stride = context_len

        # 输入投影：patch_size → D（与 PSTG 保持可比性）
        self.input_proj = nn.Linear(1, d_model)  # 逐时间步投影

        # Mamba 时序建模
        self.mamba = MambaBlock(d_model, d_state, d_conv)

        # 正弦位置编码（与 PSTG 一致）
        pe = torch.zeros(n_patches, d_model)
        pos = torch.arange(0, n_patches, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)   # [N, D]

    def forward(self, x_c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_c: [B, L] 单通道时序
        Returns:
            z: [B, N, D]
        """
        B, L = x_c.shape

        # 1. 逐时间步投影：[B, L] → [B, L, D]
        h = self.input_proj(x_c.unsqueeze(-1))   # [B, L, 1] → [B, L, D]

        # 2. Mamba 时序建模
        h = self.mamba(h)   # [B, L, D]

        # 3. 按 patch 起始位置采样（取 patch 中心时间步的嵌入）
        patch_centers = [min(i * self.stride + self.patch_size // 2, L - 1)
                         for i in range(self.n_patches)]
        z = torch.stack([h[:, t, :] for t in patch_centers], dim=1)   # [B, N, D]

        # 4. 位置编码
        z = z + self.pe.unsqueeze(0)   # [B, N, D]

        return z


class MambaMultiScaleEmbedding(nn.Module):
    """
    多尺度 Mamba 时序嵌入（替换 PSTG 的 MultiScalePatchEmbedding）

    对每个通道独立跑 Mamba，然后按 K 种尺度采样 N 个 patch，门控融合。

    输入：X ∈ R^(B, C, L)
    输出：Z_fused ∈ R^(B, n, D)，n = C × N
    """

    def __init__(
        self,
        patch_sizes:  list,         # [25, 50, 125]
        patch_main:   int,          # 25
        d_model:      int,          # 512
        context_len:  int,          # 250
        n_channels:   int,          # 6
        d_state:      int = 16,
        d_conv:       int = 4,
    ):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.K           = len(patch_sizes)
        self.d_model     = d_model
        self.n_patches   = context_len // patch_main
        self.n_channels  = n_channels

        # 共享一个 Mamba 块（所有尺度共用同一个时序建模器）
        # 分尺度只在采样位置上不同
        self.mamba_shared = MambaBlock(d_model, d_state, d_conv)
        self.input_proj   = nn.Linear(1, d_model)   # 逐时间步投影

        # 每尺度独立的采样 stride 和位置编码
        self.strides = []
        for p in patch_sizes:
            s = max(1, (context_len - p) // (self.n_patches - 1)) if self.n_patches > 1 else context_len
            self.strides.append(s)

        # 正弦位置编码（各尺度共享）
        pe = torch.zeros(self.n_patches, d_model)
        pos = torch.arange(0, self.n_patches, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

        # 门控注意力融合（与 PSTG 完全相同）
        self.gate_proj = nn.Linear(self.K * d_model, self.K)

    def _sample_patches(
        self, h: torch.Tensor, patch_size: int, stride: int
    ) -> torch.Tensor:
        """从 Mamba 输出 h [B×C, L, D] 中按 patch 中心采样 → [B×C, N, D]"""
        L = h.shape[1]
        centers = [min(i * stride + patch_size // 2, L - 1)
                   for i in range(self.n_patches)]
        z = torch.stack([h[:, t, :] for t in centers], dim=1)   # [B×C, N, D]
        return z + self.pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]
        Returns:
            z_fused: [B, n, D]，n = C × N
        """
        B, C, L = x.shape

        # 1. 将所有通道展开为 [B×C, L, 1]，一次性处理
        x_flat = x.reshape(B * C, L)
        h = self.input_proj(x_flat.unsqueeze(-1))   # [B×C, L, D]

        # 2. 共享 Mamba 时序建模（一次 SSM 搞定所有通道）
        h = self.mamba_shared(h)   # [B×C, L, D]

        # 3. 按 K 种尺度采样
        scale_zs = []
        for p, s in zip(self.patch_sizes, self.strides):
            z_k = self._sample_patches(h, p, s)    # [B×C, N, D]
            z_k = z_k.reshape(B, C, self.n_patches, self.d_model)
            scale_zs.append(z_k)

        # 4. 门控注意力融合（与 PSTG 完全相同）
        z_cat = torch.cat(scale_zs, dim=-1)         # [B, C, N, K×D]
        alpha = F.softmax(self.gate_proj(z_cat), dim=-1).unsqueeze(-2)   # [B, C, N, 1, K]
        z_stack = torch.stack(scale_zs, dim=-1)    # [B, C, N, D, K]
        z_fused = (z_stack * alpha).sum(dim=-1)    # [B, C, N, D]

        # 5. reshape → [B, n, D]
        z_fused = z_fused.reshape(B, C * self.n_patches, self.d_model)
        return z_fused
