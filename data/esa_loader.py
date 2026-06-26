"""
ESA Mission-1 data loader.

数据格式（实测确认）：
    channels/channel_41.zip     # zip 内是 pickle 文件，无扩展名
                                # pickle → pd.DataFrame(index=datetime, columns=['channel_41'])
    channels.csv                # Channel / Subsystem / Physical Unit / Group / Target / Categorical
    labels.csv                  # ID / Channel / StartTime / EndTime  (无 Category 列)
    anomaly_types.csv           # ID / Class / Subclass / Category    (Anomaly / Rare Event / ...)

关键设计：
    - 只保留 Category == "Anomaly" 的事件作为正样本（Rare Event / Communication Gap 一律为 0）
    - labels.csv 与 anomaly_types.csv 通过 ID 列 join
    - subsystem_5 的 6 个通道：channel_41 ~ channel_46
    - 数据时间范围 2000-01-01 ~ 2013-12-31（168 个月）
    - train_ratio=0.5 → 测试集从第 84 个月（2007-01）开始，与 PSTG 一致
"""

from __future__ import annotations

import io
import logging
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# ── 内部工具 ───────────────────────────────────────────────────────────────────

def _read_channel_zip(zip_path: Path) -> pd.Series:
    """从 channel zip 中读取单通道时间序列。

    格式：zip 内含无扩展名的 pickle 文件 → pd.DataFrame(index=datetime, col=channel_name)
    返回：pd.Series，index 为 datetime，name 为通道名。
    """
    with zipfile.ZipFile(zip_path) as zf:
        inner_name = zf.namelist()[0]
        with zf.open(inner_name) as f:
            raw = f.read()

    # 检测格式
    if raw[0] == 0x80:                         # pickle 协议标志
        df = pickle.loads(raw)
    elif raw[:4] == b"PAR1":                   # parquet
        df = pd.read_parquet(io.BytesIO(raw))
    else:
        # 兜底：尝试 UTF-8 CSV
        df = pd.read_csv(io.StringIO(raw.decode("utf-8")))

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Unexpected type from {zip_path}: {type(df)}")

    # 确保 datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in df.columns:
            if any(k in col.lower() for k in ("time", "date", "datetime")):
                df = df.set_index(col)
                break
        df.index = pd.to_datetime(df.index)

    # 取唯一数值列
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError(f"No numeric column in {zip_path}")

    series = df[num_cols[0]]
    series.name = zip_path.stem          # e.g. "channel_41"
    return series


def _select_channel_names(
    channels_meta: pd.DataFrame,
    subsystem: int | str | None,
) -> list[str]:
    """从 channels.csv 中返回指定 subsystem 的通道名列表（如 ['channel_41', ...]）。

    channels.csv 的 Subsystem 列值形如 'subsystem_5'，
    因此支持传入 5（int）或 'subsystem_5'（str）。
    """
    sub_col = next(
        (c for c in channels_meta.columns if "subsystem" in c.lower()),
        None,
    )
    chan_col = "Channel"     # 固定列名

    if sub_col is None or subsystem is None:
        logger.warning("无法识别 subsystem 列或未指定 subsystem，返回所有通道")
        return channels_meta[chan_col].tolist()

    # 统一成 "subsystem_X" 格式
    if isinstance(subsystem, int):
        target = f"subsystem_{subsystem}"
    else:
        target = str(subsystem)

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
    """生成与 time_index 对齐的二进制标签数组。

    规则：
      - labels.csv (ID, Channel, StartTime, EndTime)
        与 anomaly_types.csv (ID, Category) 通过 ID join
      - 只保留 Category == "Anomaly" 的事件（忽略 Rare Event / Communication Gap）
      - 只保留与所选通道 (channel_names) 相关的事件（按 Channel 列过滤）
      - 时间段内所有时间步标记为 1
    """
    labels = np.zeros(len(time_index), dtype=np.float32)

    # ── 读取并 join ───────────────────────────────────────────────────────
    ldf = pd.read_csv(labels_path)
    atdf = pd.read_csv(anomaly_types_path)

    # join on ID
    merged = ldf.merge(atdf[["ID", "Category"]], on="ID", how="left")

    # 只保留真正的 Anomaly
    anomaly_mask = merged["Category"].str.strip().str.lower() == "anomaly"
    merged = merged[anomaly_mask]
    logger.info(
        f"[ESA labels] 总事件 {len(ldf)}，Anomaly {anomaly_mask.sum()}，"
        f"Rare Event / Other {(~anomaly_mask).sum()}"
    )

    # 只保留与所选通道相关的行（Channel 列是通道名，如 'channel_41'）
    channel_set = set(channel_names)
    channel_mask = merged["Channel"].isin(channel_set)
    merged = merged[channel_mask]
    logger.info(f"[ESA labels] 过滤到所选通道后剩余事件行数: {len(merged)}")

    if merged.empty:
        logger.warning("[ESA labels] 过滤后无任何 Anomaly 事件，标签全为 0")
        return labels

    # ── 标记时间段 ────────────────────────────────────────────────────────
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

    anomaly_ratio = labels.mean()
    logger.info(f"[ESA labels] 标注完成，异常比例 = {anomaly_ratio:.4f}")
    return labels


# ── 公开接口 ───────────────────────────────────────────────────────────────────

def load_esa(
    data_path: str | Path,
    subsystem: int | str | None = 5,
    channel_ids: list[int] | list[str] | None = None,
    val_ratio: float = 0.1,
    train_ratio: float = 0.5,
    cache_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 ESA Mission-1 数据集。

    Parameters
    ----------
    data_path   : ESA-Mission1 根目录
    subsystem   : subsystem 编号（5 → subsystem_5，即 channel_41~46）
    channel_ids : 显式指定通道名列表，如 ['channel_41',...] 或 [41,...] 时自动加前缀
                  设置后不再读 channels.csv
    val_ratio   : 从训练集末尾切出的验证集比例
    train_ratio : 总时间线中用于训练（含验证）的比例
                  0.5 → 前 84 个月训练+验证，后 84 个月测试，与 PSTG 一致
    cache_path  : 预处理结果缓存目录（.npy），存在则直接读取

    Returns
    -------
    train_data  : (T_train, N) float32, MinMax 归一化
    val_data    : (T_val,   N) float32
    test_data   : (T_test,  N) float32
    test_labels : (T_test,)   float32, binary (0/1)，仅 Anomaly 类事件
    """
    data_path = Path(data_path)

    # ── 缓存命中 ──────────────────────────────────────────────────────────
    if cache_path is not None:
        cache_path = Path(cache_path)
        cached = ["train.npy", "val.npy", "test.npy", "test_labels.npy"]
        if all((cache_path / f).exists() for f in cached):
            logger.info(f"[ESA] 从缓存加载: {cache_path}")
            return (
                np.load(cache_path / "train.npy"),
                np.load(cache_path / "val.npy"),
                np.load(cache_path / "test.npy"),
                np.load(cache_path / "test_labels.npy"),
            )

    # ── 确定通道名列表 ────────────────────────────────────────────────────
    if channel_ids is not None:
        # 支持整数列表（[41,42,...]）或字符串列表（['channel_41',...]）
        if channel_ids and isinstance(channel_ids[0], int):
            channel_names = [f"channel_{i}" for i in channel_ids]
        else:
            channel_names = [str(c) for c in channel_ids]
    else:
        meta = pd.read_csv(data_path / "channels.csv")
        channel_names = _select_channel_names(meta, subsystem)

    logger.info(f"[ESA] 使用通道 ({len(channel_names)} 个): {channel_names}")

    # ── 读取每个通道 ──────────────────────────────────────────────────────
    channels_dir = data_path / "channels"
    series_list: list[pd.Series] = []
    for name in channel_names:
        zip_path = channels_dir / f"{name}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"通道 zip 不存在: {zip_path}")
        s = _read_channel_zip(zip_path)
        s.name = name
        series_list.append(s)
        logger.info(f"  {name}: {len(s)} 个样本，{s.index[0]} → {s.index[-1]}")

    # ── 时间对齐 ─────────────────────────────────────────────────────────
    df_all = pd.concat(series_list, axis=1, join="outer")
    df_all.columns = channel_names
    df_all = df_all.sort_index().ffill().bfill().dropna()
    logger.info(f"[ESA] 对齐后: shape={df_all.shape}，"
                f"{df_all.index[0]} → {df_all.index[-1]}")

    data_arr   = df_all.values.astype(np.float32)    # (total_T, N)
    time_index = df_all.index
    if time_index.tz is not None:
        time_index = time_index.tz_localize(None)

    # ── 生成标签（仅 Anomaly 类事件） ────────────────────────────────────
    labels_path       = data_path / "labels.csv"
    anomaly_types_path = data_path / "anomaly_types.csv"
    all_labels = _build_anomaly_labels(
        labels_path, anomaly_types_path, time_index, channel_names
    )

    # ── 按时间切分 ────────────────────────────────────────────────────────
    split = int(len(data_arr) * train_ratio)
    train_raw   = data_arr[:split]
    test_raw    = data_arr[split:]
    test_labels = all_labels[split:]

    logger.info(
        f"[ESA] 切分: train+val={len(train_raw)} 步，test={len(test_raw)} 步，"
        f"test 异常比例={test_labels.mean():.4f}"
    )

    # ── 归一化 ────────────────────────────────────────────────────────────
    scaler       = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled  = scaler.transform(test_raw)

    # ── 从训练集末尾切出验证集 ────────────────────────────────────────────
    val_len    = max(1, int(len(train_scaled) * val_ratio))
    val_data   = train_scaled[-val_len:]
    train_data = train_scaled[:-val_len]

    logger.info(
        f"[ESA] 最终: train={train_data.shape}，val={val_data.shape}，"
        f"test={test_scaled.shape}"
    )

    # ── 写缓存 ────────────────────────────────────────────────────────────
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
        np.save(cache_path / "train.npy",       train_data)
        np.save(cache_path / "val.npy",         val_data)
        np.save(cache_path / "test.npy",        test_scaled)
        np.save(cache_path / "test_labels.npy", test_labels)
        logger.info(f"[ESA] 缓存写入: {cache_path}")

    return train_data, val_data, test_scaled, test_labels
