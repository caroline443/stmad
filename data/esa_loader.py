"""
ESA Mission-1 data loader.

Expected directory layout (from the ESA-AD open dataset):
    <data_path>/
        channels/
            channel_1.zip   # each zip contains a CSV with telemetry values
            channel_2.zip
            ...
        telecommands/
            telecommand_1.zip
        channels.csv        # metadata: channel id, name, subsystem, ...
        labels.csv          # anomaly labels with timestamps
        anomaly_type.csv    # anomaly event descriptions
        telecommands.csv    # telecommand metadata

The PSTG paper (entropy-28-00426) uses 6 channels from subsystem 5
(channels 41-46).  We auto-detect them from channels.csv unless
`esa_channel_ids` is explicitly set in the config.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_channel_zip(zip_path: Path) -> pd.Series:
    """Extract a single-channel time series from a zip file.

    Each zip is expected to contain exactly one CSV with at minimum a
    time column and a value column.  We auto-detect both by heuristics:
    - time column: first column whose dtype is object/datetime or whose
      name contains 'time' / 'date' (case-insensitive)
    - value column: first numeric column that is not the time column
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # Pick the first CSV-like file
        csv_name = next((n for n in names if n.endswith(".csv")), names[0])
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    # Detect time column
    time_col = None
    for col in df.columns:
        if any(kw in col.lower() for kw in ("time", "date", "timestamp")):
            time_col = col
            break
    if time_col is None:
        time_col = df.columns[0]  # fall back to first column

    # Detect value column (first numeric col that isn't the time col)
    value_col = None
    for col in df.columns:
        if col == time_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            value_col = col
            break
    if value_col is None:
        raise ValueError(f"No numeric value column found in {zip_path}")

    # Parse timestamps
    try:
        idx = pd.to_datetime(df[time_col])
    except Exception:
        # Fall back to integer index if timestamps are unparsable
        idx = pd.RangeIndex(len(df))

    series = pd.Series(df[value_col].values, index=idx, name=zip_path.stem)
    return series


def _select_channels(channels_meta: pd.DataFrame, subsystem: int | None) -> list[int]:
    """Return channel IDs belonging to the given subsystem.

    Looks for a column whose name contains 'subsystem' (case-insensitive).
    Falls back to returning all channel IDs if not found.
    """
    sub_col = None
    for col in channels_meta.columns:
        if "subsystem" in col.lower():
            sub_col = col
            break

    id_col = channels_meta.columns[0]  # assume first col is the channel id

    if sub_col is not None and subsystem is not None:
        mask = channels_meta[sub_col].astype(str) == str(subsystem)
        ids = channels_meta.loc[mask, id_col].tolist()
        if ids:
            return [int(i) for i in ids]
        logger.warning(f"No channels found for subsystem={subsystem}; using all channels")

    return [int(i) for i in channels_meta[id_col].tolist()]


def _parse_labels(labels_path: Path, time_index: pd.Index) -> np.ndarray:
    """Parse labels.csv and produce a binary label array aligned to time_index.

    Returns
    -------
    labels : np.ndarray of shape (len(time_index),) with dtype float32, values in {0, 1}
    """
    df = pd.read_csv(labels_path)
    labels = np.zeros(len(time_index), dtype=np.float32)

    # Detect start / end columns
    start_col = end_col = None
    for col in df.columns:
        low = col.lower()
        if any(kw in low for kw in ("start", "begin", "from")):
            start_col = col
        if any(kw in low for kw in ("end", "stop", "to")):
            end_col = col

    if start_col is None or end_col is None:
        logger.warning("Cannot detect start/end columns in labels.csv; returning all zeros")
        return labels

    # Detect anomaly indicator column (look for a column that distinguishes anomaly from nominal)
    anomaly_col = None
    for col in df.columns:
        if any(kw in col.lower() for kw in ("anomaly", "label", "type", "class")):
            anomaly_col = col
            break

    if not isinstance(time_index, pd.DatetimeIndex):
        # Integer index — use positional labels if possible
        logger.warning("Time index is not datetime; label alignment may be approximate")
        return labels

    for _, row in df.iterrows():
        # Skip non-anomaly rows if we have an indicator column
        if anomaly_col is not None:
            val = str(row[anomaly_col]).lower()
            if val in ("0", "false", "nominal", "normal", "no"):
                continue

        try:
            t_start = pd.to_datetime(row[start_col])
            t_end   = pd.to_datetime(row[end_col])
        except Exception:
            continue

        mask = (time_index >= t_start) & (time_index <= t_end)
        labels[mask] = 1.0

    return labels


# ── Public API ────────────────────────────────────────────────────────────────

def load_esa(
    data_path: str | Path,
    subsystem: int | None = 5,
    channel_ids: list[int] | None = None,
    val_ratio: float = 0.1,
    train_ratio: float = 0.5,  # fraction of total timeline used for training
    cache_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ESA Mission-1 dataset.

    Parameters
    ----------
    data_path   : root directory of ESA-Mission1
    subsystem   : subsystem ID to filter channels (None → use all)
    channel_ids : explicit list of channel IDs to load (overrides subsystem)
    val_ratio   : fraction of training data reserved for validation
    train_ratio : fraction of the total timeline used as training data
    cache_path  : if given, save/load preprocessed arrays as .npy files here

    Returns
    -------
    train_data  : (T_train, N) float32, scaled [0,1]
    val_data    : (T_val,   N) float32
    test_data   : (T_test,  N) float32
    test_labels : (T_test,)   float32, binary
    """
    data_path = Path(data_path)

    # ── Try cache ──────────────────────────────────────────────────────────
    if cache_path is not None:
        cache_path = Path(cache_path)
        cached_files = ["train.npy", "val.npy", "test.npy", "test_labels.npy"]
        if all((cache_path / f).exists() for f in cached_files):
            logger.info(f"Loading ESA data from cache at {cache_path}")
            train_data  = np.load(cache_path / "train.npy")
            val_data    = np.load(cache_path / "val.npy")
            test_data   = np.load(cache_path / "test.npy")
            test_labels = np.load(cache_path / "test_labels.npy")
            return train_data, val_data, test_data, test_labels

    # ── Resolve channel IDs ────────────────────────────────────────────────
    # If channel_ids is given explicitly, channels.csv is NOT required.
    if channel_ids is None:
        channels_meta_path = data_path / "channels.csv"
        if not channels_meta_path.exists():
            raise FileNotFoundError(
                "channels.csv not found. Either provide it for auto-detection, "
                "or set esa_channel_ids explicitly in configs/esa.yaml."
            )
        channels_meta = pd.read_csv(channels_meta_path)
        channel_ids = _select_channels(channels_meta, subsystem)
    logger.info(f"[ESA] Using {len(channel_ids)} channels: {channel_ids}")

    # ── Load each channel from its zip ────────────────────────────────────
    channels_dir = data_path / "channels"
    series_list: list[pd.Series] = []
    for cid in channel_ids:
        zip_path = channels_dir / f"channel_{cid}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Channel zip not found: {zip_path}")
        s = _read_channel_zip(zip_path)
        series_list.append(s)
        logger.info(f"  channel_{cid}: {len(s)} samples")

    # ── Align on common time index ─────────────────────────────────────────
    # Outer join, then forward-fill gaps
    df_all = pd.concat(series_list, axis=1, join="outer")
    df_all.columns = [f"ch_{cid}" for cid in channel_ids]
    df_all = df_all.ffill().bfill()   # fill gaps
    df_all = df_all.dropna()

    logger.info(f"[ESA] Aligned data shape: {df_all.shape}")

    data_arr = df_all.values.astype(np.float32)   # (total_T, N)
    time_index = df_all.index

    # ── Parse labels ───────────────────────────────────────────────────────
    labels_path = data_path / "labels.csv"
    if labels_path.exists():
        all_labels = _parse_labels(labels_path, time_index)
    else:
        logger.warning("labels.csv not found; using zero labels")
        all_labels = np.zeros(len(data_arr), dtype=np.float32)

    # ── Train / test split (by time, not shuffled) ─────────────────────────
    split = int(len(data_arr) * train_ratio)
    train_raw = data_arr[:split]
    test_raw  = data_arr[split:]
    test_labels = all_labels[split:]

    # ── Normalise ──────────────────────────────────────────────────────────
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled  = scaler.transform(test_raw)

    # ── Validation split (from end of training data) ───────────────────────
    val_len = max(1, int(len(train_scaled) * val_ratio))
    val_data   = train_scaled[-val_len:]
    train_data = train_scaled[:-val_len]

    logger.info(
        f"[ESA] train {train_data.shape}, val {val_data.shape}, "
        f"test {test_scaled.shape}, anomaly ratio {test_labels.mean():.4f}"
    )

    # ── Save cache ─────────────────────────────────────────────────────────
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
        np.save(cache_path / "train.npy",       train_data)
        np.save(cache_path / "val.npy",         val_data)
        np.save(cache_path / "test.npy",        test_scaled)
        np.save(cache_path / "test_labels.npy", test_labels)
        logger.info(f"[ESA] Cached preprocessed arrays to {cache_path}")

    return train_data, val_data, test_scaled, test_labels
