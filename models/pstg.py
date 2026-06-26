"""
PSTG 完整模型（论文 Section 3.2，公式 2-4）

整体流程（公式 2）：
  X̂_{t+1:t+F} = T_Θ3( G^(n_L) ∘ G^(n_L-1) ∘ ... ∘ G^(1) ( P_Θ1(X_{t-L+1:t}) ) )

三个算子：
  P: Multi-scale temporal patching → Z_fused ∈ R^(B, n, D)
  G: Progressive Spatiotemporal Graph Reasoning（n_L=2 层）
  T: Forecast head → X̂ ∈ R^(B, C, F)
"""

import torch
import torch.nn as nn

from .patch_embedding import MultiScalePatchEmbedding
from .graph_module import SpatioTemporalGraphLayer


class ForecastHead(nn.Module):
    """
    预测头 T_Θ3（公式 2）

    H^[n_L] ∈ R^(B, n, D) → X̂ ∈ R^(B, C, F)

    实现：reshape → [B, C, N, D] → flatten → [B, C, N*D] → Linear → [B, C, F]
    """

    def __init__(self, n_channels: int, n_patches: int, d_model: int, forecast_len: int):
        super().__init__()
        self.C = n_channels
        self.N = n_patches
        self.D = d_model
        self.F = forecast_len
        # 简单线性投影（论文 Figure 1 中提及 "Three Distinct Linear Layers" 对应 W_Q,W_K,W_V，
        # 预测头用一个线性层）
        self.proj = nn.Linear(n_patches * d_model, forecast_len)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, n, D]，其中 n = C × N
        Returns:
            x_hat: [B, C, F]
        """
        B, n, D = h.shape
        # reshape: [B, C, N, D]
        h = h.reshape(B, self.C, self.N, D)
        # flatten 最后两维: [B, C, N*D]
        h = h.reshape(B, self.C, self.N * D)
        # 线性投影: [B, C, F]
        x_hat = self.proj(h)
        return x_hat


class PSTG(nn.Module):
    """
    Progressive Spatiotemporal Graph（完整模型）

    Args:
        patch_sizes:  多尺度 patch 尺寸，默认 [25, 50, 125]
        patch_main:   计算 N 的基准 patch 尺寸，默认 25
        d_model:      嵌入维度，默认 512
        num_heads:    图注意力头数，默认 4
        num_layers:   Progressive 层数（n_L），默认 2
        top_k:        稀疏化 top-k 值，默认 6
        n_channels:   输入通道数，默认 6
        context_len:  输入序列长度，默认 250
        forecast_len: 预测步长，默认 10
        dropout:      Dropout 率，默认 0.1
    """

    def __init__(
        self,
        patch_sizes: list = None,
        patch_main: int = 25,
        d_model: int = 512,
        num_heads: int = 4,
        num_layers: int = 2,
        top_k: int = 6,
        n_channels: int = 6,
        context_len: int = 250,
        forecast_len: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        if patch_sizes is None:
            patch_sizes = [25, 50, 125]

        self.n_channels = n_channels
        self.n_patches = context_len // patch_main  # N = 10
        self.n_nodes = n_channels * self.n_patches   # n = 60
        self.num_layers = num_layers

        # ── 算子 P：多尺度 Patch 嵌入 ────────────────────────────────────
        self.patch_embed = MultiScalePatchEmbedding(
            patch_sizes=patch_sizes,
            patch_main=patch_main,
            d_model=d_model,
            context_len=context_len,
            n_channels=n_channels,
        )

        # ── 算子 G：Progressive 时空图推理（n_L=2 层） ────────────────────
        self.graph_layers = nn.ModuleList([
            SpatioTemporalGraphLayer(
                d_model=d_model,
                num_heads=num_heads,
                top_k=top_k,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── 算子 T：预测头 ───────────────────────────────────────────────
        self.forecast_head = ForecastHead(
            n_channels=n_channels,
            n_patches=self.n_patches,
            d_model=d_model,
            forecast_len=forecast_len,
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化所有线性层"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,               # [B, C, L]
        return_adj: bool = False,       # 是否返回邻接矩阵（用于可视化）
    ):
        """
        Args:
            x: [B, C, L] 输入时序（已归一化）
            return_adj: 若 True，同时返回每层的邻接矩阵和注意力矩阵
        Returns:
            x_hat: [B, C, F]
            (可选) adj_list: 每层的 A_final，列表长度 = n_L
        """
        # 1. 多尺度 Patch 嵌入（公式 3）
        h = self.patch_embed(x)    # [B, n, D]，H^[0] = Z_fused

        # 2. Progressive 时空图推理（公式 4）
        adj_list = []
        for layer in self.graph_layers:
            h, A_final = layer(h)   # [B, n, D]，[B, H, n, n]
            if return_adj:
                adj_list.append(A_final.detach().cpu())

        # 3. 预测头
        x_hat = self.forecast_head(h)  # [B, C, F]

        if return_adj:
            return x_hat, adj_list
        return x_hat

    @classmethod
    def from_config(cls, cfg):
        """从 Config 对象构建 PSTG 模型"""
        return cls(
            patch_sizes=cfg.PATCH_SIZES,
            patch_main=cfg.PATCH_MAIN,
            d_model=cfg.D_MODEL,
            num_heads=cfg.NUM_HEADS,
            num_layers=cfg.NUM_LAYERS,
            top_k=cfg.top_k,
            n_channels=cfg.NUM_CHANNELS,
            context_len=cfg.CONTEXT_LEN,
            forecast_len=cfg.FORECAST_LEN,
            dropout=cfg.P_DROPOUT,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
