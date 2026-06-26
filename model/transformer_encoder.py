"""
Transformer Temporal Encoder — PSTG baseline 对应模块。

用标准 Multi-Head Self-Attention 替代 Mamba，复现 PSTG 的时序建模方式。
复杂度 O(L²)，与 Mamba 的 O(L) 形成对比。

Input / output: (B, L, N, d_model)  — 与 MambaTemporalEncoder 接口完全一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerTemporalEncoder(nn.Module):
    """Per-sensor Transformer temporal encoder (PSTG-style).

    对每个传感器的 L 个 patch token 独立做 Multi-Head Self-Attention，
    与 MambaTemporalEncoder 的 channel-independent 设计一致。

    Args:
        d_model:  embedding dimension
        n_heads:  attention heads (d_model 必须被 n_heads 整除)
        ff_mult:  FFN hidden dim 倍数（标准 Transformer 用 4）
        dropout:  dropout rate
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ff_mult:  int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, L, N, d_model)
        Returns:
            H_out: (B, L, N, d_model)
        """
        B, L, N, d = H.shape

        # 对每个传感器独立做 self-attention：合并 B 和 N 维度
        H_flat = H.permute(0, 2, 1, 3).reshape(B * N, L, d)  # (B*N, L, d)

        # MHA + residual + norm
        attn_out, _ = self.attn(H_flat, H_flat, H_flat)
        H_flat = self.norm1(H_flat + self.dropout(attn_out))

        # FFN + residual + norm
        H_flat = self.norm2(H_flat + self.dropout(self.ff(H_flat)))

        # 还原维度
        H_out = H_flat.reshape(B, N, L, d).permute(0, 2, 1, 3)  # (B, L, N, d)
        return H_out
