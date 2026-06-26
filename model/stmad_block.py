"""
STMAD Block — one Mamba→GAT→Gated-Fusion residual unit.

Each block:
    1. MambaTemporalEncoder  : update node features along the time axis
    2. DynamicGATLayer       : update node features across the sensor graph
    3. Gated Fusion          : learn a per-element gate to blend both streams
    4. Layer Norm            : stabilise gradients

Stacking n_layers of these blocks progressively refines both temporal
and spatial representations in an interleaved fashion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .mamba_encoder       import MambaTemporalEncoder
from .transformer_encoder import TransformerTemporalEncoder
from .dynamic_gat         import DynamicGATLayer


class STMADBlock(nn.Module):
    """Single temporal-encoder + Dynamic-GAT block with gated fusion.

    temporal_encoder_type:
        'mamba'       — Mamba SSM (STMAD，我们的模型，线性复杂度)
        'transformer' — Multi-Head Self-Attention (PSTG baseline，二次复杂度)

    Input / output shape: (B, L, N, d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 4,
        top_k: int = 5,
        dropout: float = 0.1,
        temporal_encoder_type: str = "mamba",
    ) -> None:
        super().__init__()

        if temporal_encoder_type == "transformer":
            self.temporal_enc = TransformerTemporalEncoder(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )
        else:  # mamba（默认）
            self.temporal_enc = MambaTemporalEncoder(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )

        self.gat_enc = DynamicGATLayer(
            d_model=d_model,
            n_heads=n_heads,
            top_k=top_k,
            dropout=dropout,
        )

        # Gated fusion: learns a per-element blending weight ∈ (0, 1)
        # Input: concatenation of temporal and spatial representations
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, L, N, d_model)

        Returns:
            H_out: (B, L, N, d_model)
        """
        H_time  = self.temporal_enc(H)                 # (B, L, N, d)
        H_space = self.gat_enc(H)                     # (B, L, N, d)

        # Gated fusion: g ∈ (0,1) weights the temporal stream
        g     = self.gate(torch.cat([H_time, H_space], dim=-1))   # (B, L, N, d)
        H_out = g * H_time + (1.0 - g) * H_space                 # (B, L, N, d)

        return self.norm(self.dropout(H_out))

    @property
    def last_attn_weights(self) -> torch.Tensor | None:
        """Convenience accessor to the GAT attention matrix."""
        return self.gat_enc.last_attn_weights
