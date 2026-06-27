"""
SMAP / MSL 数据集加载

数据格式（与主分支 data/smap_msl_loader.py 兼容）：
  <data_path>/
    SMAP_train.npy       # (T_train, N=55)
    SMAP_test.npy        # (T_test,  N=55)
    SMAP_test_label.npy  # (T_test,)
    MSL_train.npy        # (T_train, N=27)
    MSL_test.npy         # (T_test,  N=27)
    MSL_test_label.npy   # (T_test,)

评估说明：
  - with_PA=False：严格评估（与本项目 ESA-AD 评估一致）
  - with_PA=True ：Point Adjustment（业界通行，分数偏高）
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from data.dataset import SlidingWindowDataset, FullSequenceDataset


# ─────────────────────────────────────────────────────────────────────────────
#  原始数据加载（直接从 npy 文件）
# ─────────────────────────────────────────────────────────────────────────────

def load_smap_msl(
    data_path: str | Path,
    dataset:   str   = "smap",    # "smap" 或 "msl"
    val_ratio: float = 0.1,
) -> tuple:
    """
    加载 SMAP 或 MSL 数据。

    Returns:
        (train_data, val_data, test_data, test_labels)
        所有 ndarray，data.shape=(T, N)，labels.shape=(T,)
    """
    data_path = Path(data_path)
    name = dataset.upper()

    train_path = data_path / f"{name}_train.npy"
    test_path  = data_path / f"{name}_test.npy"
    label_path = data_path / f"{name}_test_label.npy"

    # 多种常见命名格式兼容
    if not train_path.exists():
        # 尝试小写
        train_path = data_path / f"{name.lower()}_train.npy"
        test_path  = data_path / f"{name.lower()}_test.npy"
        label_path = data_path / f"{name.lower()}_test_label.npy"
    if not train_path.exists():
        raise FileNotFoundError(
            f"在 {data_path} 未找到 {name} 数据文件。\n"
            f"期望：{name}_train.npy / {name}_test.npy / {name}_test_label.npy\n"
            f"目录内容：{list(data_path.iterdir())[:10]}"
        )

    train_raw = np.load(train_path).astype(np.float32)
    test_raw  = np.load(test_path).astype(np.float32)
    labels    = np.load(label_path).astype(np.int32)

    print(f"[{name}] 训练集: {train_raw.shape}  测试集: {test_raw.shape}")
    print(f"[{name}] 异常率: {labels.mean()*100:.2f}%  通道数: {train_raw.shape[1]}")

    # MinMax 归一化（只用训练集统计量）
    scaler       = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled  = scaler.transform(test_raw)

    # 从训练集末尾切出验证集
    val_len    = max(1, int(len(train_scaled) * val_ratio))
    val_data   = train_scaled[-val_len:]
    train_data = train_scaled[:-val_len]

    print(f"[{name}] train={train_data.shape}  val={val_data.shape}  test={test_scaled.shape}")
    return train_data, val_data, test_scaled, labels


# ─────────────────────────────────────────────────────────────────────────────
#  构建 DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets_smap_msl(cfg) -> dict:
    """
    加载 SMAP/MSL 并构建 DataLoader。
    cfg 需要有：DATA_DIR, DATASET_NAME, CONTEXT_LEN, FORECAST_LEN,
                TRAIN_STRIDE, TAU, BATCH_SIZE, B_S
    """
    print(f"\n=== 加载 {cfg.DATASET_NAME.upper()} 数据集 ===")
    train_data, val_data, test_data, test_labels = load_smap_msl(
        cfg.DATA_DIR, cfg.DATASET_NAME, val_ratio=0.1,
    )

    train_ds = SlidingWindowDataset(
        train_data, cfg.CONTEXT_LEN, cfg.FORECAST_LEN, stride=cfg.TRAIN_STRIDE
    )
    val_ds = SlidingWindowDataset(
        val_data, cfg.CONTEXT_LEN, cfg.FORECAST_LEN, stride=cfg.CONTEXT_LEN
    )
    test_ds = FullSequenceDataset(
        test_data, cfg.CONTEXT_LEN, cfg.FORECAST_LEN, tau=cfg.TAU
    )

    print(f"  train 样本: {len(train_ds):,}  val 样本: {len(val_ds):,}  test 窗口: {len(test_ds):,}")

    num_workers = min(4, os.cpu_count() or 1)
    return {
        "train_loader": DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        "val_loader": DataLoader(
            val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "test_loader": DataLoader(
            test_ds, batch_size=cfg.B_S, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "test_labels": test_labels,
        "test_data":   test_data,
        "train_data":  train_data,
        "val_data":    val_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Point Adjustment (PA) 协议
# ─────────────────────────────────────────────────────────────────────────────

def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Point Adjustment (PA)：业界通行协议（Hundman et al. 2018）。

    若真实异常段 [s,e] 内任意一个时间步被预测为异常，
    则该段内所有时间步均视为正确检测。

    注意：PA 会大幅提高 Recall 和 F1，使结果偏乐观。
          本项目 ESA-AD 评估 **不使用 PA**，但 SMAP/MSL 需用于与文献比较。
    """
    from utils.metrics import extract_events

    y_pred_adj = y_pred.copy()
    for s, e in extract_events(y_true):
        if y_pred[s:e+1].any():
            y_pred_adj[s:e+1] = 1
    return y_pred_adj


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算 Point-wise Precision / Recall / F1（逐点评估）"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}
