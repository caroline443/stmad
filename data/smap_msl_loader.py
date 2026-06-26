"""
SMAP / MSL data loader.

Expected directory layout:
    <data_path>/
        SMAP_train.npy       # (T_train, N)
        SMAP_test.npy        # (T_test,  N)
        SMAP_test_label.npy  # (T_test,)

Replace SMAP with MSL for the MSL dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def load_smap_msl(
    data_path: str | Path,
    dataset: str = "smap",
    val_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load SMAP or MSL dataset.

    Returns
    -------
    train_data : np.ndarray  (T_train, N)  — scaled to [0,1]
    val_data   : np.ndarray  (T_val,   N)  — last val_ratio of train
    test_data  : np.ndarray  (T_test,  N)  — scaled with train scaler
    test_labels: np.ndarray  (T_test,)     — binary {0,1}
    """
    data_path = Path(data_path)
    name = dataset.upper()  # "SMAP" or "MSL"

    train_path = data_path / f"{name}_train.npy"
    test_path  = data_path / f"{name}_test.npy"
    label_path = data_path / f"{name}_test_label.npy"

    for p in (train_path, test_path, label_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")

    train_raw = np.load(train_path).astype(np.float32)   # (T_train, N)
    test_raw  = np.load(test_path).astype(np.float32)    # (T_test,  N)
    labels    = np.load(label_path).astype(np.float32)   # (T_test,)

    logger.info(f"[{name}] raw train {train_raw.shape}, test {test_raw.shape}, "
                f"anomaly ratio {labels.mean():.4f}")

    # MinMax normalisation: fit on train, apply to test
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)   # (T_train, N)
    test_scaled  = scaler.transform(test_raw)        # (T_test,  N)

    # Split off validation from the end of training data
    val_len = max(1, int(len(train_scaled) * val_ratio))
    val_data   = train_scaled[-val_len:]
    train_data = train_scaled[:-val_len]

    logger.info(f"[{name}] after split — train {train_data.shape}, "
                f"val {val_data.shape}, test {test_scaled.shape}")

    return train_data, val_data, test_scaled, labels
