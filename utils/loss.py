"""
复合损失函数（论文 Section 3.2.3，公式 25）

L = L_MSE + λ1·L_freq + λ2·L_shape

L_MSE  = ||X_future - X̂_future||_F²
L_freq = ||F(X_future) - F(X̂_future)||_F²   （沿时间轴的 DFT）
L_shape = ||∇_t X_future - ∇_t X̂_future||_F²（时间差分）
"""

import torch
import torch.nn as nn


class PSTGLoss(nn.Module):
    """
    PSTG 的三项复合预测损失。

    Args:
        lambda1: 频域损失权重 λ1
        lambda2: 形态损失权重 λ2
    """

    def __init__(self, lambda1: float = 0.1, lambda2: float = 0.1):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(
        self,
        pred: torch.Tensor,   # X̂_future [B, C, F]
        target: torch.Tensor, # X_future  [B, C, F]
    ) -> tuple:
        """
        Returns:
            total_loss, (mse, freq_loss, shape_loss)
        """
        # ── L_MSE：逐点均方误差 ───────────────────────────────────────
        mse = torch.mean((pred - target) ** 2)

        # ── L_freq：频域损失（沿 F 轴做 DFT）────────────────────────
        # rfft 在最后一维（时间轴）做实数 FFT，返回复数张量
        pred_fft = torch.fft.rfft(pred, dim=-1)    # [B, C, F//2+1]
        tgt_fft  = torch.fft.rfft(target, dim=-1)  # [B, C, F//2+1]
        # Frobenius norm² = 实部差² + 虚部差²
        diff_fft = pred_fft - tgt_fft
        freq_loss = torch.mean(diff_fft.real ** 2 + diff_fft.imag ** 2)

        # ── L_shape：时间梯度损失（相邻时间步差分）──────────────────
        # ∇_t X = X[:,:,1:] - X[:,:,:-1]  → [B, C, F-1]
        pred_grad = pred[..., 1:] - pred[..., :-1]
        tgt_grad  = target[..., 1:] - target[..., :-1]
        shape_loss = torch.mean((pred_grad - tgt_grad) ** 2)

        # ── 加权求和 ───────────────────────────────────────────────────
        total = mse + self.lambda1 * freq_loss + self.lambda2 * shape_loss

        return total, (mse.item(), freq_loss.item(), shape_loss.item())
