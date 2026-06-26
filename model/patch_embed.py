"""
Multi-scale Patch Embedding.

Splits the input time series into patches at multiple temporal scales,
projects each patch to d_model, and sums the contributions.

Reference: adapted from PSTG (Chen et al., 2026) multi-scale patch design.

Shape convention throughout this project:
    B  — batch size
    T  — window size (raw time steps)
    N  — number of sensors
    L  — number of patch tokens = T // patch_sizes[0]  (finest scale)
    d  — d_model (embedding dimension)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Multi-scale patch embedding.

    For each patch size ``p`` in ``patch_sizes``:
        1. Split the T-length series into ``n_p = T // p`` non-overlapping patches.
        2. Project each patch (size p) to d_model via a Linear layer.
        3. Upsample the ``n_p`` tokens to L = T // patch_sizes[0] by repeating
           (``repeat_interleave``), so all scales produce the same L tokens.

    The contributions from all scales are summed element-wise, then normalised.

    Args:
        window_size:  T — raw time steps per window
        patch_sizes:  list of patch sizes in ascending or any order;
                      the *first* element is the finest (smallest) patch that
                      defines L.  Recommended: [25, 50, 125].
        d_model:      output embedding dimension
        dropout:      dropout applied after normalisation
    """

    def __init__(
        self,
        window_size: int,
        patch_sizes: list[int],
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.patch_sizes = patch_sizes
        self.p_main = patch_sizes[0]    # finest scale defines L
        self.L = window_size // self.p_main
        self.d_model = d_model

        # Validate that all patch sizes divide window_size evenly
        for p in patch_sizes:
            assert window_size % p == 0, (
                f"window_size={window_size} must be divisible by patch_size={p}"
            )
            assert self.L % (window_size // p) == 0, (
                f"L={self.L} must be divisible by n_patches={window_size // p} "
                f"for patch_size={p} (so repeat_interleave is exact)"
            )

        # One linear projector per scale: maps (batch, n_p, N, p) → (…, d_model)
        self.projectors = nn.ModuleList([
            nn.Linear(p, d_model) for p in patch_sizes
        ])

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N)

        Returns:
            H: (B, L, N, d_model)
        """
        B, T, N = x.shape
        assert T == self.window_size, (
            f"Expected window_size={self.window_size}, got T={T}"
        )

        out = x.new_zeros(B, self.L, N, self.d_model)

        for p, proj in zip(self.patch_sizes, self.projectors):
            n_p = T // p                         # number of patches at this scale
            rf  = self.L // n_p                  # repeat factor to reach L

            # Reshape: (B, T, N) → (B, n_p, p, N) → (B, n_p, N, p)
            x_p = x[:, :n_p * p, :].reshape(B, n_p, p, N).permute(0, 1, 3, 2)

            h = proj(x_p)                        # (B, n_p, N, d_model)

            if rf > 1:
                # Upsample by repeating each token rf times along dim=1
                h = h.repeat_interleave(rf, dim=1)   # (B, L, N, d_model)

            out = out + h

        return self.norm(self.dropout(out))      # (B, L, N, d_model)
