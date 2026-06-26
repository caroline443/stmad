"""
SlidingWindowDataset — unified interface for all datasets.

Each sample is a window of shape (T, N) where
    T = window_size (sequence length)
    N = number of sensors

Labels (binary, shape T) are optional and only provided for the test set.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SlidingWindowDataset(Dataset):
    """Sliding-window view over a multivariate time series.

    Args:
        data:        np.ndarray of shape (total_T, N)
        window_size: length of each window
        stride:      step between consecutive windows (1 = dense)
        labels:      optional np.ndarray of shape (total_T,) with binary labels
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        stride: int = 1,
        labels: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        assert data.ndim == 2, f"data must be (T, N), got {data.shape}"
        self.data = torch.from_numpy(data).float()
        self.window_size = window_size
        self.stride = stride
        self.has_labels = labels is not None

        if self.has_labels:
            assert len(labels) == len(data), "labels length must match data length"
            self.labels = torch.from_numpy(labels.astype(np.float32))
        else:
            self.labels = None

        total_T = len(data)
        # Start indices of each window
        self.starts = list(range(0, total_T - window_size + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        e = s + self.window_size
        x = self.data[s:e]  # (T, N)
        if self.has_labels:
            y = self.labels[s:e]  # (T,)
            return x, y
        return x


def build_dataloaders(
    train_data: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    window_size: int,
    batch_size: int,
    num_workers: int = 4,
    val_data: np.ndarray | None = None,
    train_stride: int = 1,
    test_stride: int = 1,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    """Build train / (optional) val / test DataLoaders.

    During training the labels are not needed, so train_dataset has no labels.
    The test DataLoader yields (x, y) pairs.
    """
    train_ds = SlidingWindowDataset(train_data, window_size, stride=train_stride)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_data is not None:
        val_ds = SlidingWindowDataset(val_data, window_size, stride=test_stride)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    test_ds = SlidingWindowDataset(
        test_data, window_size, stride=test_stride, labels=test_labels
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
