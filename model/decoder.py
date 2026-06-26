"""
Reconstruction Decoder.

Projects the L patch-level representations back to the original T time
steps by predicting each patch's raw values.

Architecture: a two-layer MLP applied per-patch, per-sensor:
    d_model  →  2*d_model (GELU)  →  p_main

The output is then rearranged from (B, L, N, p_main) to (B, T, N).

This is a deliberate design choice: using a simple MLP decoder forces
the encoder (Mamba + GAT stack) to learn all the structure, and keeps
the anomaly signal localised in the encoder residuals rather than being
absorbed by a powerful decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionDecoder(nn.Module):
    """MLP decoder: patch embeddings → reconstructed time series.

    Args:
        d_model:  encoder embedding dimension
        p_main:   finest patch size (determines output stride)
        L:        number of patch tokens = window_size // p_main

    Input:  (B, L, N, d_model)
    Output: (B, T, N)   where T = L * p_main = window_size
    """

    def __init__(self, d_model: int, p_main: int, L: int) -> None:
        super().__init__()
        self.p_main = p_main
        self.L      = L

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, p_main),
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, L, N, d_model)

        Returns:
            x_hat: (B, T, N)
        """
        B, L, N, d = H.shape

        out = self.mlp(H)                          # (B, L, N, p_main)
        # Rearrange patches back to a contiguous time series
        # (B, L, N, p_main) → (B, L, p_main, N) → (B, L*p_main, N)
        out = out.permute(0, 1, 3, 2)             # (B, L, p_main, N)
        out = out.reshape(B, L * self.p_main, N)  # (B, T, N)

        return out
