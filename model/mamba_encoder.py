"""
Mamba Temporal Encoder.

Applies a Mamba SSM independently to each sensor's token sequence
(channel-independent design).  This gives linear O(L) complexity
and avoids the O(L²) cost of Transformer-based temporal encoders used
in PSTG.

Mamba's selective-scan mechanism naturally amplifies state changes at
anomalous time steps, making the hidden-state residual a useful anomaly
signal on its own.

Dependency: mamba-ssm  (pip install mamba-ssm causal-conv1d)
            Requires CUDA.

If mamba-ssm is not available (e.g., CPU-only testing), a fallback
GRU-based encoder is used automatically.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Try to import the CUDA Mamba kernel ───────────────────────────────────────
try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
except ImportError:
    logger.error(
        "mamba-ssm not found — falling back to GRU (results will differ from paper). "
        "Install on the GPU server with: pip install mamba-ssm causal-conv1d"
    )
    _MAMBA_AVAILABLE = False


# ── GRU fallback (CPU / debugging) ───────────────────────────────────────────

class _GRUFallback(nn.Module):
    """Bidirectional GRU as a drop-in replacement for Mamba in CPU environments."""

    def __init__(self, d_model: int, **kwargs) -> None:
        super().__init__()
        self.gru = nn.GRU(
            d_model, d_model // 2, batch_first=True, bidirectional=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out


# ── Mamba Temporal Encoder ────────────────────────────────────────────────────

class MambaTemporalEncoder(nn.Module):
    """Per-sensor Mamba (or GRU fallback) temporal encoder.

    The L patch tokens of each sensor are treated as an independent
    sequence and passed through a Mamba SSM.  Batch and sensor dimensions
    are merged so the operation is fully vectorised.

    Args:
        d_model:  embedding dimension (must match Mamba d_model)
        d_state:  Mamba SSM state size (default 16)
        d_conv:   Mamba conv kernel width (default 4)
        expand:   Mamba inner-dimension expansion factor (default 2)
        dropout:  dropout on the residual output

    Input / output shape: (B, L, N, d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if _MAMBA_AVAILABLE:
            self.ssm = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.ssm = _GRUFallback(d_model)

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, L, N, d_model)

        Returns:
            H_out: (B, L, N, d_model)  — residual connection included
        """
        B, L, N, d = H.shape

        # Merge batch × sensor → process each sensor's L-step sequence
        H_flat = H.permute(0, 2, 1, 3).reshape(B * N, L, d)   # (B*N, L, d)
        out    = self.ssm(H_flat)                               # (B*N, L, d)
        out    = out.reshape(B, N, L, d).permute(0, 2, 1, 3)  # (B, L, N, d)

        # Pre-norm residual
        return self.norm(H + self.dropout(out))                 # (B, L, N, d)
