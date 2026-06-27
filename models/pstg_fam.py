"""
PSTG-FAM：频率感知改进版 PSTG

在 PSTG 的 patch embedding 之前，先对原始时序做频率感知滤波（FAM），
去除高频噪声，使模型学到更干净的正常模式。

架构（相比 PSTG 只新增一步）：
  Input X [B, C, L]
    ↓ FrequencyAwareFilter（Top-K FFT，零参数）← 新增
  X_filtered [B, C, L]
    ↓ MultiScalePatchEmbedding（不变）
  Z_fused [B, n, D]
    ↓ SpatioTemporalGraphLayer × n_L（不变）
  H^[nL] [B, n, D]
    ↓ ForecastHead（不变）
  X̂ [B, C, F]

优势：
  - 零新增参数（FAM 是纯 FFT 运算）
  - 可直接从 PSTG checkpoint 热启动（参数完全兼容）
  - 改善正常数据的预测精度 → 降低 R_pred_normal → 更好的异常分离
  - 评估协议与 PSTG 完全一致（直接用 detect_anomalies）
"""

import torch
import torch.nn as nn

from .patch_embedding import MultiScalePatchEmbedding
from .graph_module import SpatioTemporalGraphLayer
from .pstg import ForecastHead
from .frequency_module import FrequencyAwareFilter


class PSTG_FAM(nn.Module):
    """
    Frequency-Aware PSTG

    新增超参：
        top_k_rate: FAM 保留的频率分量比例（默认 0.5）
    """

    def __init__(
        self,
        # ── PSTG 原有参数（不变）──────────────────────────────────────────
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
        # ── FAM 新增参数 ────────────────────────────────────────────────
        top_k_rate:   float = 0.5,
    ):
        super().__init__()
        if patch_sizes is None:
            patch_sizes = [25, 50, 125]

        self.n_channels  = n_channels
        self.n_patches   = context_len // patch_main
        self.n_nodes     = n_channels * self.n_patches
        self.num_layers  = num_layers

        # ── FAM：频率感知滤波（零参数，只在 forward 中执行 FFT）─────────
        self.fam = FrequencyAwareFilter(top_k_rate=top_k_rate)

        # ── 以下与 PSTG 完全相同 ─────────────────────────────────────────
        self.patch_embed = MultiScalePatchEmbedding(
            patch_sizes=patch_sizes, patch_main=patch_main,
            d_model=d_model, context_len=context_len, n_channels=n_channels,
        )
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

    # ── 前向传播 ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, return_adj: bool = False):
        """
        与 PSTG.forward 接口完全一致，直接替换使用。
        """
        # 1. FAM 频率过滤（新增，零参数）
        x_filtered = self.fam(x)          # [B, C, L] → [B, C, L]

        # 2-4. 标准 PSTG 流程（不变）
        h = self.patch_embed(x_filtered)  # [B, n, D]

        adj_list = []
        for layer in self.graph_layers:
            h, A = layer(h)
            if return_adj:
                adj_list.append(A.detach().cpu())

        x_hat = self.forecast_head(h)     # [B, C, F]

        if return_adj:
            return x_hat, adj_list
        return x_hat

    # ── 工厂方法 ─────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg):
        return cls(
            patch_sizes=cfg.PATCH_SIZES,
            patch_main= cfg.PATCH_MAIN,
            d_model=    cfg.D_MODEL,
            num_heads=  cfg.NUM_HEADS,
            num_layers= cfg.NUM_LAYERS,
            top_k=      cfg.top_k,
            n_channels= cfg.NUM_CHANNELS,
            context_len=cfg.CONTEXT_LEN,
            forecast_len=cfg.FORECAST_LEN,
            dropout=    cfg.P_DROPOUT,
            top_k_rate= cfg.FAM_TOP_K_RATE,
        )

    @classmethod
    def from_pstg_checkpoint(cls, ckpt_path: str, cfg, device: str = "cpu"):
        """
        从 PSTG checkpoint 热启动。

        FAM 是零参数模块，所有 PSTG 参数（patch_embed / graph_layers / forecast_head）
        完全兼容，直接加载，无需任何参数映射。
        """
        ckpt     = torch.load(ckpt_path, map_location=device)
        ckpt_cfg = ckpt.get("config", {})

        model = cls(
            patch_sizes=  ckpt_cfg.get("patch_sizes",  cfg.PATCH_SIZES),
            d_model=      ckpt_cfg.get("d_model",       cfg.D_MODEL),
            num_heads=    ckpt_cfg.get("num_heads",     cfg.NUM_HEADS),
            num_layers=   ckpt_cfg.get("num_layers",    cfg.NUM_LAYERS),
            n_channels=   ckpt_cfg.get("n_channels",    cfg.NUM_CHANNELS),
            context_len=  ckpt_cfg.get("context_len",   cfg.CONTEXT_LEN),
            forecast_len= ckpt_cfg.get("forecast_len",  cfg.FORECAST_LEN),
            top_k=cfg.top_k, dropout=cfg.P_DROPOUT,
            top_k_rate=cfg.FAM_TOP_K_RATE,
        )

        # FAM 无参数，PSTG 的其余参数全部兼容，直接 strict=True 加载
        model.load_state_dict(ckpt["model"], strict=True)

        print(f"[热启动] PSTG-FAM 从 PSTG checkpoint 加载全部参数")
        print(f"  epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss','?')}")
        print(f"  FAM top_k_rate={cfg.FAM_TOP_K_RATE}（保留前 {cfg.FAM_TOP_K_RATE*100:.0f}% 频率分量）")
        return model.to(device)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_new_parameters(self) -> int:
        """相比 PSTG 新增的参数量（FAM 零参数，所以是 0）"""
        return 0
