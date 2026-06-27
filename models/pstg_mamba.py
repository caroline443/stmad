"""
PSTG-Mamba：Mamba 时序编码 + Progressive 时空图

核心创新：
  用 Mamba SSM 替换 PSTG 的多尺度 patch 线性投影，
  为 patch 嵌入引入长程时序上下文，
  然后保持 PSTG 的渐进式时空图推理不变。

与 STMAD 的区别（同样用了 Mamba）：
  STMAD：Mamba（时序） → 单层 Dynamic GAT（空间），两步串行
  PSTG-Mamba：Mamba（通道内时序）→ 渐进式时空图（跨通道，2层精炼），
               图推理同时建模空间和时空依赖，且逐层迭代细化

架构：
  Input X [B, C, L]
    ↓ MambaMultiScaleEmbedding（共享 Mamba + 多尺度采样 + 门控融合）
  Z_fused [B, n, D]
    ↓ SpatioTemporalGraphLayer × n_L（与 PSTG 完全相同）
  H^[nL] [B, n, D]
    ↓ ForecastHead
  X̂ [B, C, F]
"""

import torch
import torch.nn as nn

from .mamba_temporal import MambaMultiScaleEmbedding
from .graph_module import SpatioTemporalGraphLayer
from .pstg import ForecastHead


class PSTG_Mamba(nn.Module):
    """
    PSTG with Mamba Temporal Encoding

    相比 PSTG 的改动：
      - MultiScalePatchEmbedding → MambaMultiScaleEmbedding
      - 新增 d_state / d_conv 超参（Mamba SSM 参数）
      - 其余（图推理、预测头、评估）完全一致，可直接用 evaluate.py

    新增参数量：约 D × d_state × 4（Mamba SSM 内部矩阵），远少于图层参数。
    """

    def __init__(
        self,
        # ── PSTG 原有参数 ──────────────────────────────────────────────────
        patch_sizes:  list  = None,
        patch_main:   int   = 25,
        d_model:      int   = 512,
        num_heads:    int   = 4,
        num_layers:   int   = 2,
        top_k:        int   = 6,
        n_channels:   int   = 6,
        context_len:  int   = 250,
        forecast_len: int   = 10,
        dropout:      float = 0.1,
        # ── Mamba 新增参数 ──────────────────────────────────────────────────
        d_state:      int   = 16,   # SSM 状态维度（越大表达力越强，内存越多）
        d_conv:       int   = 4,    # 局部卷积宽度（Mamba 默认 4）
    ):
        super().__init__()
        if patch_sizes is None:
            patch_sizes = [25, 50, 125]

        self.n_channels  = n_channels
        self.n_patches   = context_len // patch_main
        self.n_nodes     = n_channels * self.n_patches
        self.num_layers  = num_layers

        # ── Mamba 时序嵌入（替换 PSTG 的多尺度 patch 线性投影）────────────
        self.patch_embed = MambaMultiScaleEmbedding(
            patch_sizes=patch_sizes,
            patch_main=patch_main,
            d_model=d_model,
            context_len=context_len,
            n_channels=n_channels,
            d_state=d_state,
            d_conv=d_conv,
        )

        # ── 以下与 PSTG 完全相同 ─────────────────────────────────────────────
        self.graph_layers = nn.ModuleList([
            SpatioTemporalGraphLayer(
                d_model=d_model, num_heads=num_heads,
                top_k=top_k, dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.forecast_head = ForecastHead(
            n_channels=n_channels, n_patches=self.n_patches,
            d_model=d_model, forecast_len=forecast_len,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_adj: bool = False):
        """与 PSTG.forward 接口完全一致。"""
        # 1. Mamba 时序嵌入
        h = self.patch_embed(x)        # [B, n, D]

        # 2. Progressive 时空图推理（不变）
        adj_list = []
        for layer in self.graph_layers:
            h, A = layer(h)
            if return_adj:
                adj_list.append(A.detach().cpu())

        # 3. 预测头（不变）
        x_hat = self.forecast_head(h)  # [B, C, F]

        if return_adj:
            return x_hat, adj_list
        return x_hat

    @classmethod
    def from_config(cls, cfg):
        return cls(
            patch_sizes=cfg.PATCH_SIZES,  patch_main=cfg.PATCH_MAIN,
            d_model=cfg.D_MODEL,          num_heads=cfg.NUM_HEADS,
            num_layers=cfg.NUM_LAYERS,    top_k=cfg.top_k,
            n_channels=cfg.NUM_CHANNELS,  context_len=cfg.CONTEXT_LEN,
            forecast_len=cfg.FORECAST_LEN, dropout=cfg.P_DROPOUT,
            d_state=cfg.MAMBA_D_STATE,    d_conv=cfg.MAMBA_D_CONV,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def mamba_param_count(self) -> int:
        return sum(p.numel() for p in self.patch_embed.parameters() if p.requires_grad)

    def graph_param_count(self) -> int:
        total = sum(p.numel() for p in self.graph_layers.parameters() if p.requires_grad)
        total += sum(p.numel() for p in self.forecast_head.parameters() if p.requires_grad)
        return total
