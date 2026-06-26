"""
Decoder — 支持重建（F=0）和预测（F>0）两种输出模式。

重建模式：输入 (B, L, N, d) → 输出 (B, T, N)，T = L × p_main
预测模式：输入 (B, L, N, d) → 输出 (B, F, N)，F = forecast_horizon
          对应 PSTG：用 L=10 个 patch token 预测未来 F=10 步
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionDecoder(nn.Module):
    """重建解码器：patch 表示 → 原始时间序列。

    Input:  (B, L, N, d_model)
    Output: (B, T, N)  where T = L * p_main
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
        # H: (B, L, N, d)
        B, L, N, d = H.shape
        out = self.mlp(H)                                    # (B, L, N, p_main)
        out = out.permute(0, 1, 3, 2).contiguous()          # (B, L, p_main, N)
        out = out.reshape(B, L * self.p_main, N)            # (B, T, N)
        return out


class ForecastDecoder(nn.Module):
    """预测解码器：patch 表示 → 未来 F 步（PSTG 范式）。

    用所有 L 个 patch token 的全局表示（取最后一个或平均）
    预测未来连续 F 个时间步的值。

    Input:  (B, L, N, d_model)
    Output: (B, F, N)
    """

    def __init__(self, d_model: int, forecast_horizon: int, n_sensors: int) -> None:
        super().__init__()
        self.F = forecast_horizon
        self.N = n_sensors

        # 对每个传感器独立预测 F 步：(B, N, d) → (B, N, F)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, forecast_horizon),
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (B, L, N, d)
        # 取最后一个 patch token 作为预测依据（对应 PSTG 的 τ=1 设计）
        h_last = H[:, -1, :, :]              # (B, N, d)
        out    = self.mlp(h_last)            # (B, N, F)
        out    = out.permute(0, 2, 1)        # (B, F, N)
        return out
