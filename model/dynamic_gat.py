"""
Dynamic Graph Attention Layer.

Key idea: instead of fixing a sensor-relationship graph at training time
(as in GDN / FuSAGNet) or using coarse temporal snapshots (ContrastAD),
we recompute multi-head attention coefficients α_ij(t) *for every patch
token* from the current node features.

This yields a continuously time-varying adjacency matrix A(t) ∈ R^{N×N}
that adapts to the evolving state of the spacecraft, and is directly
interpretable as an XAI artefact.

Implementation trick: flatten (B, L) → (B*L) so all L timesteps are
computed in a single batched matrix multiply — no Python loop over time.

Top-K sparsification: only the K strongest edges per node are kept
(negative-infinity mask on the rest), which reduces noise and can be
tuned via the `top_k` config parameter.

Stores `self.last_attn_weights` after each forward pass for visualisation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGATLayer(nn.Module):
    """Multi-head dynamic graph attention over sensors.

    Args:
        d_model:  node feature dimension
        n_heads:  number of attention heads (d_model must be divisible by n_heads)
        top_k:    keep only the top-k attention weights per node row (sparse GAT)
        dropout:  dropout on attention weights

    Input / output shape: (B, L, N, d_model)

    Stores:
        last_attn_weights: Tensor (B, L, N, N) averaged over heads — available
                           after every forward() call for visualisation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        top_k: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model={d_model} must be divisible by n_heads={n_heads}"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.top_k   = top_k
        self.scale   = math.sqrt(self.d_head)

        # Projections (no bias following common GAT convention)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.norm      = nn.LayerNorm(d_model)

        # Populated after each forward() — shape (B, L, N, N)
        self.last_attn_weights: torch.Tensor | None = None

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, L, N, d_model)

        Returns:
            H_out: (B, L, N, d_model)  — residual connection included
        """
        B, L, N, d = H.shape
        BL = B * L

        # Flatten time into batch dimension for vectorised computation
        H_flat = H.reshape(BL, N, d)            # (BL, N, d)

        # Multi-head projections → (BL, H, N, d_head)
        def proj_mh(W: nn.Linear) -> torch.Tensor:
            return W(H_flat).reshape(BL, N, self.n_heads, self.d_head).transpose(1, 2)

        Q = proj_mh(self.W_q)    # (BL, H, N, d_head)
        K = proj_mh(self.W_k)
        V = proj_mh(self.W_v)

        # Scaled dot-product attention scores: (BL, H, N, N)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Top-K sparsification: zero out all but the K largest scores per row
        k = min(self.top_k, N)
        if k < N:
            # kth value per row (threshold): (BL, H, N, 1)
            threshold = attn.topk(k, dim=-1).values[..., -1, None]
            attn = attn.masked_fill(attn < threshold, float("-inf"))

        attn = F.softmax(attn, dim=-1)          # (BL, H, N, N)
        attn = self.attn_drop(attn)

        # Store for visualisation (average over heads, detach)
        with torch.no_grad():
            self.last_attn_weights = (
                attn.mean(dim=1)                 # (BL, N, N)
                .reshape(B, L, N, N)
                .detach()
            )

        # Weighted aggregation: (BL, H, N, d_head) → (BL, N, d)
        out = torch.matmul(attn, V)              # (BL, H, N, d_head)
        out = out.transpose(1, 2).reshape(BL, N, d)
        out = self.W_o(out)

        # Restore time dimension + residual + norm
        out = out.reshape(B, L, N, d)
        return self.norm(H + out)                # (B, L, N, d)
