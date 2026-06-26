"""
Anomaly Scorer.

Runs the trained STMAD model over a DataLoader and aggregates per-timestep
reconstruction errors into a single anomaly score time series.

Because consecutive windows overlap (stride=1 by default), multiple
windows contribute a score to each time step.  We aggregate by *mean*
over all windows that cover a given time step.

Output
------
scores : np.ndarray of shape (T_total,)  — higher = more anomalous
labels : np.ndarray of shape (T_total,)  — ground-truth binary labels
                                            (None if DataLoader has no labels)
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
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute per-timestep anomaly scores via sliding-window reconstruction.

    Args:
        model:       trained STMAD (in eval mode)
        loader:      DataLoader whose dataset is SlidingWindowDataset
        device:      torch device
        window_size: T — length of each window
        stride:      stride used when building the DataLoader
        total_T:     total number of time steps; inferred from loader if None

    Returns:
        scores : (T_total,) float32  — mean squared reconstruction error per step
        labels : (T_total,) float32 | None
    """
    model.eval()

    # Derive total_T from the dataset to guarantee full coverage.
    # Do NOT derive from len(loader) * batch_size: that undercounts if
    # the DataLoader silently dropped the last batch (drop_last=True).
    n_windows = len(loader.dataset)          # always the full window count
    if total_T is None:
        total_T = (n_windows - 1) * stride + window_size

    score_sum  = np.zeros(total_T, dtype=np.float64)
    count      = np.zeros(total_T, dtype=np.float64)
    label_buf  = np.zeros(total_T, dtype=np.float32)
    has_labels = False
    window_idx = 0   # global window index (tracks position in the time series)

    for batch in tqdm(loader, desc="Scoring", leave=False):
        if isinstance(batch, (list, tuple)):
            x, y = batch
            has_labels = True
        else:
            x = batch
            y = None

        x = x.to(device)                            # (B, T, N)
        B = x.size(0)

        x_hat   = model(x)                          # (B, T, N)
        err_cpu = (x - x_hat).pow(2).mean(dim=-1).cpu().numpy()  # (B, T)

        for i in range(B):
            start = window_idx * stride
            end   = start + window_size
            end   = min(end, total_T)               # clamp to buffer boundary
            valid = end - start

            score_sum[start:end] += err_cpu[i, :valid]
            count[start:end]     += 1

            if has_labels and y is not None:
                label_buf[start:end] = np.maximum(
                    label_buf[start:end],
                    y[i, :valid].cpu().numpy(),
                )
            window_idx += 1

    # Warn if any timestep was never covered (indicates drop_last or stride issue)
    uncovered = int((count == 0).sum())
    if uncovered > 0:
        logger.warning(
            f"{uncovered}/{total_T} timesteps have count=0 "
            f"(likely due to drop_last=True in DataLoader). "
            f"Set drop_last=False in build_dataloaders to avoid this."
        )

    # Safe mean: use NaN for uncovered steps so they don't corrupt thresholding
    with np.errstate(invalid="ignore"):
        scores = np.where(count > 0, score_sum / count, np.nan).astype(np.float32)

    # For threshold fitting: drop NaN entries (they should not exist with drop_last=False)
    valid_mask = ~np.isnan(scores)
    if not valid_mask.all():
        scores = np.where(valid_mask, scores, float(np.nanmedian(scores)))

    return scores, (label_buf if has_labels else None)
