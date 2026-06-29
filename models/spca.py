"""
SpCA: Spectral Channel Attention Network for Spacecraft Anomaly Detection

架构思路（受 PSTG 启发，但完全独立实现）：
  PSTG 做多尺度时序建模用的是 patch 切分，SpCA 改用 FFT 频段分解——
  物理上更直觉：航天器遥测信号高度周期性，频域天然捕捉轨道周期、热循环等模式。
  PSTG 做通道关系建模用的是动态图+GATv2，SpCA 改用纯 Cross-Channel Attention——
  对于 C=6 的小通道数，注意力机制足够且更轻量。

整体流程：
  X[B, C, L]
    → FFT 分解为 n_bands 个频段 → IFFT 还原为时域信号
    → 每个频段独立：线性投影 + Cross-Channel Attention
    → 可学习权重融合各频段
    → 全局 Cross-Channel Attention 精炼
    → 线性预测头 → X̂[B, C, F]

与 PSTG 关键区别：
  - 无 patch 操作：频域分解替代多尺度 patch
  - 无图结构：Cross-Channel Attention 替代动态图 + GATv2
  - 无动态邻接矩阵：注意力权重直接学习通道关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  频段分解
# ─────────────────────────────────────────────────────────────────────────────

class SpectralBandDecomposer(nn.Module):
    """
    把时域信号分解为 n_bands 个频段，每个频段独立保留对应的频率分量。

    固定三段划分（可扩展为可学习）：
      低频段  [0,   lo_rate)  : 长周期趋势（轨道、热循环等）
      中频段  [lo_rate, hi_rate) : 操作周期性变化
      高频段  [hi_rate, Nyquist] : 短时波动与瞬态

    输入：x [B, C, L]
    输出：List[Tensor]，长度 = n_bands，每个元素形状 [B, C, L]
    """

    def __init__(
        self,
        context_len: int,
        n_bands: int = 3,
        band_splits: tuple = (0.1, 0.4),  # 相对于 Nyquist 的分割点
    ):
        super().__init__()
        self.L = context_len
        self.n_bands = n_bands
        N = context_len // 2 + 1  # rfft 输出长度

        # 计算各频段的 bin 边界（固定，不可学习）
        splits = [0] + [max(1, int(s * N)) for s in band_splits] + [N]
        # 确保单调递增且不重叠
        for i in range(1, len(splits)):
            splits[i] = max(splits[i], splits[i - 1] + 1)
        splits[-1] = N
        self.register_buffer(
            "splits", torch.tensor(splits, dtype=torch.long)
        )

    def forward(self, x: torch.Tensor) -> list:
        """x: [B, C, L] → List of [B, C, L]（频段数 = n_bands）"""
        X_freq = torch.fft.rfft(x, dim=-1)  # [B, C, N]

        bands = []
        for k in range(self.n_bands):
            lo = self.splits[k].item()
            hi = self.splits[k + 1].item()
            # 零化其他频率分量
            mask = torch.zeros_like(X_freq)
            mask[:, :, lo:hi] = 1.0
            band_time = torch.fft.irfft(X_freq * mask, n=self.L, dim=-1)  # [B, C, L]
            bands.append(band_time)

        return bands


# ─────────────────────────────────────────────────────────────────────────────
#  单频段编码器（带时序注意力）
# ─────────────────────────────────────────────────────────────────────────────

class BandTemporalEncoder(nn.Module):
    """
    对单频段信号做时序感知编码：[B, C, L] → [B, C, D]

    改进点（相对于直接线性投影 Linear(L→D)）：
      - 先把时序切成 n_patches 个 patch，每个 patch 独立投影到 D 维
      - 加可学习位置编码，保留 patch 的时序顺序信息
      - 对 n_patches 个 token 做时序自注意力（Temporal Self-Attention）
      - 对所有 patch 的输出做均值池化，得到 [B, C, D]

    这样模型能感知"频段信号在时间轴上哪段更重要"，
    而不是把所有时间步等权叠加（Linear(L→D) 的局限）。
    """

    def __init__(
        self,
        context_len: int,
        d_model:     int,
        n_patches:   int   = 10,
        n_heads:     int   = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.n_patches  = n_patches
        self.patch_size = context_len // n_patches   # 250//10 = 25

        # patch 投影：每个 patch_size 长的片段 → D
        self.patch_proj = nn.Linear(self.patch_size, d_model)
        # 可学习位置编码（每个 patch 一个）
        self.pos_emb    = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # 时序自注意力（在 n_patches 个 token 之间）
        self.temporal_attn = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN，训练更稳定
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        → 切 patch → 投影 → 时序注意力 → 均值池化
        → [B, C, D]
        """
        B, C, L = x.shape
        p = self.patch_size

        # 截断确保整除（避免 L 不整除时越界）
        x = x[:, :, : self.n_patches * p]                    # [B, C, N*p]

        # 切 patch：[B, C, N, p]
        x = x.reshape(B, C, self.n_patches, p)

        # 投影：[B, C, N, D]
        x = self.patch_proj(x)

        # 加位置编码（广播到 B 和 C）
        x = x + self.pos_emb.unsqueeze(0)                    # [B, C, N, D]

        # 时序自注意力：需要把 B 和 C 合并为 batch 维
        x = x.reshape(B * C, self.n_patches, -1)             # [B*C, N, D]
        x = self.temporal_attn(x)                            # [B*C, N, D]

        # 均值池化：[B*C, D]
        x = x.mean(dim=1)

        # 还原：[B, C, D]
        x = self.out_norm(x.reshape(B, C, -1))
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  跨通道注意力块
# ─────────────────────────────────────────────────────────────────────────────

class CrossChannelAttention(nn.Module):
    """
    跨通道注意力：把通道维 C 当作序列长度，D 当作特征维度。

    输入 [B, C, D]：C=6 个通道各自有一个 D 维嵌入
    每个通道作为一个"token"，通过 Multi-Head Self-Attention 相互交换信息。
    包含 Post-LN 和 FFN（与标准 Transformer block 一致）。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, D] → [B, C, D]"""
        # Self-attention：每个通道关注其他所有通道
        x2, _ = self.attn(x, x, x)
        x = self.norm1(x + x2)
        x = self.norm2(x + self.ffn(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  频段融合
# ─────────────────────────────────────────────────────────────────────────────

class SpectralFusion(nn.Module):
    """
    可学习的频段加权融合：每个频段有一个标量权重，经 Softmax 归一化后加权求和。

    设计动机：不同频段对不同类型的异常贡献不同，
    让模型自适应地学习低/中/高频的重要性。
    """

    def __init__(self, n_bands: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_bands))

    def forward(self, band_features: list) -> torch.Tensor:
        """
        band_features: List of [B, C, D]，长度 = n_bands
        返回：[B, C, D]
        """
        w = F.softmax(self.weights, dim=0)  # [n_bands]
        fused = sum(w[i] * band_features[i] for i in range(len(band_features)))
        return fused


# ─────────────────────────────────────────────────────────────────────────────
#  预测头
# ─────────────────────────────────────────────────────────────────────────────

class ForecastHead(nn.Module):
    """[B, C, D] → [B, C, F]"""

    def __init__(self, d_model: int, forecast_len: int):
        super().__init__()
        self.proj = nn.Linear(d_model, forecast_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
#  主模型
# ─────────────────────────────────────────────────────────────────────────────

class SpCA(nn.Module):
    """
    Spectral Channel Attention Network (SpCA)

    设计原则：
      1. 用 FFT 频段分解替代 PSTG 的多尺度 patch 嵌入
         → 频域分解对周期性遥测信号更自然
      2. 用 Cross-Channel Attention 替代 PSTG 的动态图 + GATv2
         → C=6 小通道数下，注意力机制足够且更简洁
      3. 可学习频段融合权重
         → 模型自适应决定低/中/高频的权重
      4. 预测残差范式（同 PSTG）
         → 预测正常行为，用误差标记异常

    Args:
        n_channels:      通道数，默认 6
        context_len:     输入序列长度，默认 250
        forecast_len:    预测步长，默认 10
        d_model:         嵌入维度，默认 256
        n_heads:         注意力头数，默认 4
        n_bands:         频段数，默认 3（低/中/高）
        band_splits:     频段分割点（相对于 Nyquist），默认 (0.1, 0.4)
        n_layers_band:   每频段注意力层数，默认 1
        n_layers_global: 融合后全局注意力层数，默认 2
        dropout:         Dropout 率，默认 0.1
    """

    def __init__(
        self,
        n_channels:      int   = 6,
        context_len:     int   = 250,
        forecast_len:    int   = 10,
        d_model:         int   = 256,
        n_heads:         int   = 4,
        n_bands:         int   = 3,
        band_splits:     tuple = (0.1, 0.4),
        n_patches:       int   = 10,    # 每频段切多少个 patch（时序编码用）
        n_layers_band:   int   = 1,
        n_layers_global: int   = 2,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.n_channels  = n_channels
        self.n_bands     = n_bands
        self.context_len = context_len

        # ── 频段分解 ─────────────────────────────────────────────────────────
        self.decomposer = SpectralBandDecomposer(
            context_len=context_len,
            n_bands=n_bands,
            band_splits=band_splits,
        )

        # ── 每频段独立分支 ────────────────────────────────────────────────────
        # use_temporal=True → BandTemporalEncoder（时序注意力，参数更多）
        # use_temporal=False → BandProjection（简单线性，v1 原版，参数少）
        self.use_temporal = n_patches > 0
        if self.use_temporal:
            self.band_projs = nn.ModuleList([
                BandTemporalEncoder(
                    context_len=context_len,
                    d_model=d_model,
                    n_patches=n_patches,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_bands)
            ])
        else:
            self.band_projs = nn.ModuleList([
                BandProjection(context_len, d_model) for _ in range(n_bands)
            ])
        # 跨通道注意力（每频段 n_layers_band 层）
        self.band_attns = nn.ModuleList([
            nn.ModuleList([
                CrossChannelAttention(d_model, n_heads, dropout)
                for _ in range(n_layers_band)
            ])
            for _ in range(n_bands)
        ])

        # ── 频段融合 ──────────────────────────────────────────────────────────
        self.fusion = SpectralFusion(n_bands)

        # ── 全局精炼（融合后跨通道注意力） ─────────────────────────────────────
        self.global_attns = nn.ModuleList([
            CrossChannelAttention(d_model, n_heads, dropout)
            for _ in range(n_layers_global)
        ])

        # ── 预测头 ────────────────────────────────────────────────────────────
        self.forecast_head = ForecastHead(d_model, forecast_len)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]  输入时序（已归一化）
        Returns:
            x_hat: [B, C, F]  预测的未来 F 步
        """
        # 1. 频段分解
        bands = self.decomposer(x)           # List of n_bands × [B, C, L]

        # 2. 每频段独立处理
        band_feats = []
        for k in range(self.n_bands):
            z = self.band_projs[k](bands[k])  # [B, C, D]
            for attn in self.band_attns[k]:
                z = attn(z)                   # [B, C, D]
            band_feats.append(z)

        # 3. 频段融合
        z = self.fusion(band_feats)           # [B, C, D]

        # 4. 全局跨通道精炼
        for attn in self.global_attns:
            z = attn(z)                       # [B, C, D]

        # 5. 预测
        x_hat = self.forecast_head(z)         # [B, C, F]
        return x_hat

    @classmethod
    def from_config(cls, cfg):
        return cls(
            n_channels      = cfg.NUM_CHANNELS,
            context_len     = cfg.CONTEXT_LEN,
            forecast_len    = cfg.FORECAST_LEN,
            d_model         = cfg.D_MODEL,
            n_heads         = cfg.NUM_HEADS,
            n_bands         = cfg.N_BANDS,
            band_splits     = cfg.BAND_SPLITS,
            n_patches       = cfg.N_PATCHES,
            n_layers_band   = cfg.N_LAYERS_BAND,
            n_layers_global = cfg.N_LAYERS_GLOBAL,
            dropout         = cfg.P_DROPOUT,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
