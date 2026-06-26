"""
Anomaly Scorer — 支持重建（F=0）和预测（F>0）两种模式。

重建模式：score(t) = ||x(t) - x̂(t)||²，按滑窗平均聚合
预测模式：score(t) = ||x_fut(t) - x̂_fut(t)||²
          每个窗口预测未来 F 步，score 位于未来窗口区间 [s+T, s+T+F)
          对应 PSTG 的预测误差评分
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_anomaly_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    window_size: int,
    stride: int = 1,
    total_T: int | None = None,
    forecast_horizon: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """计算每个时间步的异常分数。

    重建模式 (forecast_horizon=0):
        每个窗口 [s, s+T) 的重建误差聚合到 [s, s+T)

    预测模式 (forecast_horizon=F):
        每个窗口 [s, s+T) 的预测误差聚合到未来区间 [s+T, s+T+F)
        → 分数时间轴对应实际观测时间

    Returns:
        scores: (total_T,) 异常分
        labels: (total_T,) 或 None
    """
    model.eval()

    is_forecast = forecast_horizon > 0
    n_windows   = len(loader.dataset)

    if total_T is None:
        if is_forecast:
            # 每个窗口覆盖 [s+T, s+T+F)，最后一个窗口结束位置
            last_start = (n_windows - 1) * stride
            total_T    = last_start + window_size + forecast_horizon
        else:
            total_T = (n_windows - 1) * stride + window_size

    score_sum  = np.zeros(total_T, dtype=np.float64)
    count      = np.zeros(total_T, dtype=np.float64)
    label_buf  = np.zeros(total_T, dtype=np.float32)
    has_labels = False
    window_idx = 0

    for batch in tqdm(loader, desc="Scoring", leave=False):
        # 解包
        if is_forecast:
            # batch: (x_ctx, x_fut) 或 (x_ctx, x_fut, label)
            if len(batch) == 3:
                x_ctx, x_fut, lbl = batch
                has_labels = True
            else:
                x_ctx, x_fut = batch
                lbl = None
            x_ctx = x_ctx.to(device)
            x_fut = x_fut.to(device)
            pred  = model(x_ctx)                          # (B, F, N)
            err   = (x_fut - pred).pow(2).mean(dim=-1)   # (B, F)
        else:
            # batch: x 或 (x, label)
            if isinstance(batch, (list, tuple)):
                if batch[1].ndim > 1:                     # (x, label_2d) 不存在，但防御
                    x, lbl = batch
                    has_labels = True
                elif batch[1].shape[-1] == window_size:   # label 形状与 x 一致时
                    x, lbl = batch
                    has_labels = True
                else:
                    x   = batch[0]
                    lbl = batch[1] if len(batch) > 1 else None
                    if lbl is not None:
                        has_labels = True
            else:
                x   = batch
                lbl = None
            x     = x.to(device)
            x_hat = model(x)
            err   = (x - x_hat).pow(2).mean(dim=-1)      # (B, T)

        err_cpu = err.cpu().numpy()
        B = err_cpu.shape[0]

        for i in range(B):
            s = window_idx * stride

            if is_forecast:
                # 分数归属：未来窗口 [s+T, s+T+F)
                start = s + window_size
                end   = start + forecast_horizon
                valid = min(end, total_T) - start
                if valid <= 0:
                    window_idx += 1
                    continue
                score_sum[start:start+valid] += err_cpu[i, :valid]
                count[start:start+valid]     += 1
                if has_labels and lbl is not None:
                    label_buf[start:start+valid] = np.maximum(
                        label_buf[start:start+valid],
                        lbl[i, :valid].cpu().numpy(),
                    )
            else:
                # 分数归属：当前窗口 [s, s+T)
                end   = min(s + window_size, total_T)
                valid = end - s
                score_sum[s:end] += err_cpu[i, :valid]
                count[s:end]     += 1
                if has_labels and lbl is not None:
                    label_buf[s:end] = np.maximum(
                        label_buf[s:end],
                        lbl[i, :valid].cpu().numpy(),
                    )

            window_idx += 1

    uncovered = int((count == 0).sum())
    if uncovered > 0:
        logger.warning(f"{uncovered}/{total_T} 时间步 count=0（未被任何窗口覆盖）")

    with np.errstate(invalid="ignore"):
        scores = np.where(count > 0, score_sum / count, np.nan).astype(np.float32)

    valid_mask = ~np.isnan(scores)
    if not valid_mask.all():
        scores = np.where(valid_mask, scores, float(np.nanmedian(scores)))

    return scores, (label_buf if has_labels else None)
