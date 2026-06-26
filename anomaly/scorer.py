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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


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

    # We need total_T to pre-allocate accumulation buffers
    if total_T is None:
        n_windows = len(loader.dataset)
        total_T   = (n_windows - 1) * stride + window_size

    score_sum = np.zeros(total_T, dtype=np.float64)
    count     = np.zeros(total_T, dtype=np.float64)
    label_buf = np.zeros(total_T, dtype=np.float32)
    has_labels = False

    window_idx = 0   # linear index over windows

    for batch in tqdm(loader, desc="Scoring", leave=False):
        if isinstance(batch, (list, tuple)):
            x, y = batch
            has_labels = True
        else:
            x = batch
            y = None

        x = x.to(device)                  # (B, T, N)
        B = x.size(0)

        x_hat   = model(x)                # (B, T, N)
        err     = (x - x_hat).pow(2)     # (B, T, N)
        err_cpu = err.mean(dim=-1).cpu().numpy()   # (B, T)  — mean over sensors

        for i in range(B):
            start = window_idx * stride
            end   = start + window_size
            if end > total_T:
                # Edge case: last batch may extend past total_T
                valid = total_T - start
                score_sum[start:total_T] += err_cpu[i, :valid]
                count[start:total_T]     += 1
                if has_labels and y is not None:
                    label_buf[start:total_T] = y[i, :valid].cpu().numpy()
            else:
                score_sum[start:end] += err_cpu[i]
                count[start:end]     += 1
                if has_labels and y is not None:
                    # Labels for the same step should be consistent; take max
                    label_buf[start:end] = np.maximum(
                        label_buf[start:end], y[i].cpu().numpy()
                    )
            window_idx += 1

    # Avoid division by zero (should not happen for valid data)
    count = np.where(count == 0, 1, count)
    scores = (score_sum / count).astype(np.float32)

    return scores, (label_buf if has_labels else None)
