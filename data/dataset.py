"""
ESA-AD 数据集加载与预处理

数据格式（确认）：
    ESA-Mission1/
    ├── channels/
    │   ├── channel_41.zip   ← Pickle / Parquet / CSV 格式的 DataFrame
    │   ├── channel_42.zip
    │   └── ...
    ├── channels.csv         ← 通道元信息（含 subsystem 列）
    ├── labels.csv           ← 异常标注（StartTime, EndTime, Channel, ID）
    ├── anomaly_types.csv    ← 异常类别（ID, Category）
    └── telecommands.csv

数据划分（train_ratio=0.5，val 从训练尾部切出 10%）：
    train ≈ 81 months，val ≈ 8 months，test ≈ 84 months

归一化：MinMax（只用训练集统计量）
"""

from __future__ import annotations

import io
import logging
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  底层数据读取（直接复用 ESA-AD 格式解析逻辑）
# ─────────────────────────────────────────────────────────────────────────────

def _read_channel_zip(zip_path: Path) -> pd.Series:
    """读取单个通道 zip，支持 pickle / Parquet / CSV 格式。"""
    with zipfile.ZipFile(zip_path) as zf:
        inner_name = zf.namelist()[0]
        with zf.open(inner_name) as f:
            raw = f.read()

    # 自动检测格式
    if raw[0] == 0x80:           # pickle magic byte
        df = pickle.loads(raw)
    elif raw[:4] == b"PAR1":     # Parquet magic
        df = pd.read_parquet(io.BytesIO(raw))
    else:                        # CSV
        df = pd.read_csv(io.StringIO(raw.decode("utf-8")))

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"从 {zip_path} 读取到非 DataFrame 类型：{type(df)}")

    # 确保时间索引
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in df.columns:
            if any(k in col.lower() for k in ("time", "date", "datetime")):
                df = df.set_index(col)
                break
        df.index = pd.to_datetime(df.index)

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError(f"{zip_path} 中无数值列")

    series = df[num_cols[0]]
    series.name = zip_path.stem
    return series


def _select_channel_names(
    channels_meta: pd.DataFrame,
    subsystem: int | str | None,
) -> list[str]:
    """从 channels.csv 中按 subsystem 筛选通道名。"""
    sub_col = next(
        (c for c in channels_meta.columns if "subsystem" in c.lower()), None
    )
    chan_col = "Channel"

    if sub_col is None or subsystem is None:
        logger.warning("无 subsystem 列或未指定 subsystem，返回所有通道")
        return channels_meta[chan_col].tolist()

    target = f"subsystem_{subsystem}" if isinstance(subsystem, int) else str(subsystem)
    mask = channels_meta[sub_col].astype(str) == target
    names = channels_meta.loc[mask, chan_col].tolist()

    if not names:
        logger.warning(f"subsystem={target!r} 无匹配通道，返回所有通道")
        return channels_meta[chan_col].tolist()
    return names


def _build_anomaly_labels(
    labels_path: Path,
    anomaly_types_path: Path,
    time_index: pd.DatetimeIndex,
    channel_names: list[str],
) -> np.ndarray:
    """根据 labels.csv + anomaly_types.csv 生成 0/1 标签序列。"""
    labels = np.zeros(len(time_index), dtype=np.float32)

    ldf  = pd.read_csv(labels_path)
    atdf = pd.read_csv(anomaly_types_path)
    merged = ldf.merge(atdf[["ID", "Category"]], on="ID", how="left")

    # 只取 Category == "Anomaly"，过滤 Rare Event / Communication Gap
    anomaly_mask = merged["Category"].str.strip().str.lower() == "anomaly"
    merged = merged[anomaly_mask]
    logger.info(f"[ESA labels] 事件总数 {len(ldf)}，Anomaly {anomaly_mask.sum()}")

    # 过滤到所选通道
    channel_set  = set(channel_names)
    channel_mask = merged["Channel"].isin(channel_set)
    merged = merged[channel_mask]
    logger.info(f"[ESA labels] 过滤后剩余行数: {len(merged)}")

    if merged.empty:
        logger.warning("[ESA labels] 无 Anomaly 事件，标签全为 0")
        return labels

    # 逐行标注时间区间
    for _, row in merged.iterrows():
        try:
            t_start = pd.to_datetime(row["StartTime"], utc=True).tz_localize(None)
            t_end   = pd.to_datetime(row["EndTime"],   utc=True).tz_localize(None)
        except Exception:
            try:
                t_start = pd.to_datetime(row["StartTime"]).tz_localize(None)
                t_end   = pd.to_datetime(row["EndTime"]).tz_localize(None)
            except Exception:
                continue
        mask = (time_index >= t_start) & (time_index <= t_end)
        labels[mask] = 1.0

    logger.info(f"[ESA labels] 异常比例 = {labels.mean():.4f}")
    return labels


def load_esa(
    data_path: str | Path,
    subsystem: int | None = 5,
    channel_ids: list[int] | None = None,
    train_ratio: float = 0.5,
    val_ratio:   float = 0.1,
    cache_path:  str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 ESA-AD Mission 1 数据，返回 (train, val, test, test_labels)。
    所有数组均为 float32，形状 [T, C]（test_labels 为 [T]）。

    Args:
        data_path:   数据根目录（含 channels/ channels.csv labels.csv）
        subsystem:   subsystem 编号（默认 5 → channels 41-46）
        channel_ids: 手动指定通道 ID，如 [41,42,43,44,45,46]
        train_ratio: 前多少比例作为 train+val（默认 0.5）
        val_ratio:   从 train 尾部切出多少作验证集（默认 0.1）
        cache_path:  若指定，则缓存/读取 npy 文件
    """
    data_path = Path(data_path)

    # ── 缓存读取 ─────────────────────────────────────────────────────────────
    if cache_path is not None:
        cache_path = Path(cache_path)
        cached = ["train.npy", "val.npy", "test.npy", "test_labels.npy"]
        if all((cache_path / f).exists() for f in cached):
            logger.info(f"[ESA] 从缓存加载: {cache_path}")
            print(f"  使用缓存: {cache_path}")
            return (
                np.load(cache_path / "train.npy"),
                np.load(cache_path / "val.npy"),
                np.load(cache_path / "test.npy"),
                np.load(cache_path / "test_labels.npy"),
            )

    # ── 通道名解析 ───────────────────────────────────────────────────────────
    if channel_ids is not None:
        channel_names = [f"channel_{i}" for i in channel_ids]
    else:
        meta = pd.read_csv(data_path / "channels.csv")
        channel_names = _select_channel_names(meta, subsystem)

    print(f"  加载通道 ({len(channel_names)} 个): {channel_names}")

    # ── 读取各通道 zip ───────────────────────────────────────────────────────
    channels_dir = data_path / "channels"
    series_list: list[pd.Series] = []
    for name in channel_names:
        zip_path = channels_dir / f"{name}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"通道 zip 不存在: {zip_path}")
        s = _read_channel_zip(zip_path)
        s.name = name
        series_list.append(s)
        print(f"    {name}: {len(s):,} 个时间步")

    # ── 时间对齐（outer join + 前向/后向填充）────────────────────────────────
    df_all = pd.concat(series_list, axis=1, join="outer")
    df_all.columns = channel_names
    df_all = df_all.sort_index().ffill().bfill().dropna()
    print(f"  对齐后: shape={df_all.shape}")

    data_arr   = df_all.values.astype(np.float32)   # [T, C]
    time_index = df_all.index
    if time_index.tz is not None:
        time_index = time_index.tz_localize(None)

    # ── 构建标签 ─────────────────────────────────────────────────────────────
    labels_path        = data_path / "labels.csv"
    anomaly_types_path = data_path / "anomaly_types.csv"
    all_labels = _build_anomaly_labels(
        labels_path, anomaly_types_path, time_index, channel_names
    )

    # ── 时间切分（train_ratio=0.5 对应论文 81 vs 84 months）─────────────────
    split      = int(len(data_arr) * train_ratio)
    train_raw  = data_arr[:split]
    test_raw   = data_arr[split:]
    test_labels = all_labels[split:].astype(np.int32)

    print(f"  切分: train+val={len(train_raw):,}  test={len(test_raw):,}")
    print(f"  test 异常比例: {test_labels.mean():.4f}")

    # ── MinMax 归一化（只用训练集统计量）────────────────────────────────────
    scaler        = MinMaxScaler(feature_range=(0, 1))
    train_scaled  = scaler.fit_transform(train_raw)
    test_scaled   = scaler.transform(test_raw)

    # ── 验证集从训练尾部切出 ─────────────────────────────────────────────────
    val_len    = max(1, int(len(train_scaled) * val_ratio))
    val_data   = train_scaled[-val_len:]
    train_data = train_scaled[:-val_len]

    print(f"  train={train_data.shape}  val={val_data.shape}  test={test_scaled.shape}")

    # ── 缓存写入 ─────────────────────────────────────────────────────────────
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
        np.save(cache_path / "train.npy",       train_data)
        np.save(cache_path / "val.npy",         val_data)
        np.save(cache_path / "test.npy",        test_scaled)
        np.save(cache_path / "test_labels.npy", test_labels)
        print(f"  缓存写入: {cache_path}")

    return train_data, val_data, test_scaled, test_labels


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """
    训练/验证用滑动窗口 Dataset。
    context: [C, L]，future: [C, F]
    """

    def __init__(
        self,
        data: np.ndarray,       # [T, C]，已归一化
        context_len: int = 250,
        forecast_len: int = 10,
        stride: int = 50,
    ):
        self.data = data
        self.L = context_len
        self.F = forecast_len
        T = len(data)
        self.indices = list(range(0, T - self.L - self.F + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        context = self.data[t       : t + self.L]
        future  = self.data[t+self.L: t + self.L + self.F]
        return (
            torch.tensor(context, dtype=torch.float32).T,   # [C, L]
            torch.tensor(future,  dtype=torch.float32).T,   # [C, F]
        )


class FullSequenceDataset(Dataset):
    """
    推理用 Dataset（stride = τ = 1），每次返回 context 窗口。
    """

    def __init__(
        self,
        data: np.ndarray,       # [T, C]，已归一化
        context_len: int = 250,
        forecast_len: int = 10,
        tau: int = 1,
    ):
        self.data = data
        self.L = context_len
        self.F = forecast_len
        T = len(data)
        self.indices = list(range(self.L, T - self.F + 1, tau))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        context = self.data[t - self.L : t]
        return (
            torch.tensor(context, dtype=torch.float32).T,   # [C, L]
            t,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  便捷入口
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(cfg) -> dict:
    """
    加载 ESA-AD 数据，构建训练/验证/测试 Dataset 和 DataLoader。
    """
    print("=== 加载 ESA-AD 数据集 ===")
    cache_dir = os.path.join(cfg.CHECKPOINT_DIR, "data_cache")

    train_data, val_data, test_data, test_labels = load_esa(
        data_path=cfg.DATA_DIR,
        subsystem=5,
        channel_ids=cfg.CHANNELS,
        train_ratio=0.5,
        val_ratio=0.1,
        cache_path=cache_dir,
    )

    # Dataset
    train_ds = SlidingWindowDataset(
        train_data, cfg.CONTEXT_LEN, cfg.FORECAST_LEN, stride=cfg.TRAIN_STRIDE
    )
    val_ds = SlidingWindowDataset(
        val_data, cfg.CONTEXT_LEN, cfg.FORECAST_LEN, stride=cfg.CONTEXT_LEN
    )
    test_ds = FullSequenceDataset(
        test_data, cfg.CONTEXT_LEN, cfg.FORECAST_LEN, tau=cfg.TAU
    )

    print(f"  train 样本数: {len(train_ds):,}")
    print(f"  val 样本数:   {len(val_ds):,}")
    print(f"  test 窗口数:  {len(test_ds):,}")

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.B_S, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return {
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "test_loader":  test_loader,
        "test_labels":  test_labels,
        "test_data":    test_data,
        "train_data":   train_data,
        "val_data":     val_data,
    }
