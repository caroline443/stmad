"""
STMAD — Spatiotemporal Mamba with Dynamic GAT for Anomaly Detection.

Top-level model combining:
    1. Multi-scale Patch Embedding  (temporal tokenisation)
    2. Stack of STMAD Blocks        (Mamba temporal + Dynamic GAT spatial)
    3. Reconstruction Decoder       (patch tokens → raw signal)

The model is trained with an MSE reconstruction loss.  Anomaly scores
at inference time are derived from per-timestep reconstruction errors.

Input:  X  ∈ R^{B × T × N}
Output: X̂ ∈ R^{B × T × N}
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .patch_embed import PatchEmbedding
from .stmad_block import STMADBlock
from .decoder     import ReconstructionDecoder, ForecastDecoder


class STMAD(nn.Module):
    """Spatiotemporal Mamba + Dynamic GAT Anomaly Detector.

    Args:
        n_sensors:    N — number of input sensor channels
        window_size:  T — input time steps per window
        d_model:      embedding dimension (default 64)
        d_state:      Mamba SSM state size (default 16)
        d_conv:       Mamba conv width (default 4)
        expand:       Mamba expansion factor (default 2)
        n_heads:      GAT attention heads (default 4)
        n_layers:     number of STMAD blocks (default 2)
        patch_sizes:  list of patch sizes; first is finest (default [25, 50, 125])
        top_k:        GAT top-k neighbours (default 5)
        dropout:      dropout probability (default 0.1)
    """

    def __init__(
        self,
        n_sensors: int,
        window_size: int,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 4,
        n_layers: int = 2,
        patch_sizes: list[int] | None = None,
        top_k: int = 5,
        dropout: float = 0.1,
        temporal_encoder_type: str = "mamba",   # "mamba" | "transformer"
        forecast_horizon: int = 0,              # 0=重建, >0=预测未来F步
    ) -> None:
        super().__init__()

        if patch_sizes is None:
            patch_sizes = [25, 50, 125]

        p_main = patch_sizes[0]
        L      = window_size // p_main

        self.n_sensors   = n_sensors
        self.window_size = window_size
        self.d_model     = d_model
        self.p_main      = p_main
        self.L           = L

        # ── Modules ───────────────────────────────────────────────────────
        self.patch_embed = PatchEmbedding(
            window_size=window_size,
            patch_sizes=patch_sizes,
            d_model=d_model,
            dropout=dropout,
        )

        self.blocks = nn.ModuleList([
            STMADBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                n_heads=n_heads,
                top_k=top_k,
                dropout=dropout,
                temporal_encoder_type=temporal_encoder_type,
            )
            for _ in range(n_layers)
        ])

        self.forecast_horizon = forecast_horizon
        if forecast_horizon > 0:
            self.decoder = ForecastDecoder(
                d_model=d_model,
                forecast_horizon=forecast_horizon,
                n_sensors=n_sensors,
            )
        else:
            self.decoder = ReconstructionDecoder(
                d_model=d_model,
                p_main=p_main,
                L=L,
            )

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N)

        Returns:
            x_hat: (B, T, N)   — reconstructed time series
        """
        H = self.patch_embed(x)            # (B, L, N, d)
        for block in self.blocks:
            H = block(H)                   # (B, L, N, d)
        x_hat = self.decoder(H)            # (B, T, N)
        return x_hat

    # ── helpers ───────────────────────────────────────────────────────────────

    def get_attn_weights(self) -> torch.Tensor | None:
        """Return the dynamic attention matrix from the last block's GAT.

        Shape: (B, L, N, N)  — averaged over attention heads.
        Call after a forward() pass.
        """
        return self.blocks[-1].last_attn_weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(config: dict) -> STMAD:
    """Instantiate STMAD from a config dictionary.

    Expected keys (with defaults from base.yaml):
        n_sensors, window_size, d_model, d_state, d_conv, expand,
        n_heads, n_layers, patch_sizes, top_k, dropout
    """
    # model_type: "mamba"（STMAD，默认）或 "transformer"（PSTG baseline）
    model_type = config.get("model_type", "mamba")

    model = STMAD(
        n_sensors              = config["n_sensors"],
        window_size            = config["window_size"],
        d_model                = config.get("d_model",    512),
        d_state                = config.get("d_state",    64),
        d_conv                 = config.get("d_conv",     4),
        expand                 = config.get("expand",     2),
        n_heads                = config.get("n_heads",    4),
        n_layers               = config.get("n_layers",   2),
        patch_sizes            = config.get("patch_sizes", [25, 50, 125]),
        top_k                  = config.get("top_k",      5),
        dropout                = config.get("dropout",    0.1),
        temporal_encoder_type  = model_type,
        forecast_horizon       = config.get("forecast_horizon", 0),
    )
    return model
