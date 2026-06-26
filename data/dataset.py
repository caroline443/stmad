"""
SlidingWindowDataset — unified interface for reconstruction and forecasting.

forecast_horizon = 0  →  重建模式：每个样本返回 (x,)，x 形状 (T, N)
forecast_horizon > 0  →  预测模式：每个样本返回 (x_ctx, x_fut)，
                         x_ctx (T, N) 是上下文，x_fut (F, N) 是需要预测的未来
                         对应 PSTG 的设计：L=250 上下文，F=10 预测目标
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SlidingWindowDataset(Dataset):
    """滑窗数据集，支持重建（F=0）和预测（F>0）两种模式。

    重建模式 (forecast_horizon=0):
        __getitem__ 返回 x (T, N) 或 (x, label) 当 labels 不为 None
    预测模式 (forecast_horizon=F):
        __getitem__ 返回 (x_ctx, x_fut) 或 (x_ctx, x_fut, label)
        x_ctx: (T, N)，x_fut: (F, N)
        label: (F,) 对应未来窗口的标签
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        stride: int = 1,
        labels: np.ndarray | None = None,
        forecast_horizon: int = 0,
    ) -> None:
        super().__init__()
        assert data.ndim == 2, f"data must be (T, N), got {data.shape}"
        self.data             = torch.from_numpy(data).float()
        self.window_size      = window_size
        self.stride           = stride
        self.forecast_horizon = forecast_horizon
        self.has_labels       = labels is not None

        if self.has_labels:
            assert len(labels) == len(data)
            self.labels = torch.from_numpy(labels.astype(np.float32))
        else:
            self.labels = None

        total_T = len(data)
        span    = window_size + forecast_horizon   # 每个样本需要的总步数
        self.starts = list(range(0, total_T - span + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s   = self.starts[idx]
        e   = s + self.window_size
        ef  = e + self.forecast_horizon

        x_ctx = self.data[s:e]           # (T, N)

        if self.forecast_horizon > 0:
            x_fut = self.data[e:ef]       # (F, N)
            if self.has_labels:
                lbl = self.labels[e:ef]   # (F,)  标签对应未来窗口
                return x_ctx, x_fut, lbl
            return x_ctx, x_fut

        # 重建模式
        if self.has_labels:
            lbl = self.labels[s:e]        # (T,)
            return x_ctx, lbl
        return x_ctx


def build_dataloaders(
    train_data: np.ndarray,
    test_data:  np.ndarray,
    test_labels: np.ndarray,
    window_size: int,
    batch_size:  int,
    num_workers: int = 4,
    val_data:    np.ndarray | None = None,
    train_stride: int = 1,
    test_stride:  int = 1,
    forecast_horizon: int = 0,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    """构建 train / val / test DataLoader，同时支持重建和预测模式。"""

    def _make(data, stride, labels=None, shuffle=False):
        ds = SlidingWindowDataset(
            data, window_size, stride=stride,
            labels=labels, forecast_horizon=forecast_horizon,
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )

    train_loader = _make(train_data, train_stride, shuffle=True)
    val_loader   = _make(val_data,   test_stride)  if val_data is not None else None
    test_loader  = _make(test_data,  test_stride,  labels=test_labels)

    return train_loader, val_loader, test_loader
