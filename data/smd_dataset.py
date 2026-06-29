"""
SMD（Server Machine Dataset）数据加载器

数据格式（已预处理好）：
  SMD_train.npy        [T_train, 38]  float32
  SMD_test.npy         [T_test,  38]  float32
  SMD_test_label.npy   [T_test]       int/float  0/1

特点：
  - 38 通道，异常率 ~4.2%（接近 ESA-AD 的 0.4%，比 SMAP/MSL 低得多）
  - 数据已分好 train/test，直接用
  - 归一化：MinMax，只用训练集统计量
"""

from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader


class SlidingWindowDataset(Dataset):
    def __init__(self, data, context_len=250, forecast_len=10, stride=50):
        self.data = data
        self.L, self.F = context_len, forecast_len
        self.indices = list(range(0, len(data) - context_len - forecast_len + 1, stride))

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        ctx = self.data[t: t + self.L]
        fut = self.data[t + self.L: t + self.L + self.F]
        return (torch.tensor(ctx, dtype=torch.float32).T,
                torch.tensor(fut, dtype=torch.float32).T)


class InferenceDataset(Dataset):
    def __init__(self, data, context_len=250, forecast_len=10):
        self.data = data
        self.L, self.F = context_len, forecast_len
        self.indices = list(range(context_len, len(data) - forecast_len + 1))

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        ctx = self.data[t - self.L: t]
        return torch.tensor(ctx, dtype=torch.float32).T, t


def build_datasets_smd(cfg) -> dict:
    data_dir = Path(cfg.DATA_DIR)

    train_raw = np.load(data_dir / "SMD_train.npy").astype(np.float32)
    test_raw  = np.load(data_dir / "SMD_test.npy").astype(np.float32)
    labels    = np.load(data_dir / "SMD_test_label.npy").astype(np.int32)

    # MinMax 归一化（仅用训练集）
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled  = scaler.transform(test_raw)

    # 从训练集尾部切 10% 作验证集
    val_len   = max(1, int(len(train_scaled) * 0.1))
    val_data  = train_scaled[-val_len:]
    train_data = train_scaled[:-val_len]

    print(f"[SMD] train={train_data.shape}  val={val_data.shape}  test={test_scaled.shape}")
    print(f"[SMD] 通道={train_data.shape[1]}  test异常率={labels.mean():.4f}")

    n_workers = 4
    train_ds = SlidingWindowDataset(train_data, cfg.CONTEXT_LEN, cfg.FORECAST_LEN, cfg.TRAIN_STRIDE)
    val_ds   = SlidingWindowDataset(val_data,   cfg.CONTEXT_LEN, cfg.FORECAST_LEN, cfg.CONTEXT_LEN)
    test_ds  = InferenceDataset(test_scaled,    cfg.CONTEXT_LEN, cfg.FORECAST_LEN)

    print(f"[SMD] train样本={len(train_ds)}  val样本={len(val_ds)}  test窗口={len(test_ds)}")

    return {
        "train_loader": DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                                   shuffle=True, num_workers=n_workers, pin_memory=True, drop_last=True),
        "val_loader":   DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                                   shuffle=False, num_workers=n_workers, pin_memory=True),
        "test_loader":  DataLoader(test_ds,  batch_size=cfg.B_S,
                                   shuffle=False, num_workers=n_workers, pin_memory=True),
        "test_data":    test_scaled,
        "test_labels":  labels,
    }
