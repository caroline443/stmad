"""
SpCA 论文配图生成脚本
====================
生成以下论文配图：

  fig_ablation.pdf     — 消融实验分组条形图（仿 PSTG Fig.6）
  fig_timeline.pdf     — 全时序每通道分数 + 阈值 + 事件标注（仿 MTGFlow Fig.13）
  fig_mshtrans.pdf     — 缩放窗口：原始信号+重建+每通道分数（仿 MSHTrans Fig.3）
  fig_comparison.pdf   — 方法对比水平条形图（可选）

依赖文件（evaluate_spca.py 生成）：
  eval_dir/raw_smoothed.npy
  eval_dir/y_true.npy
  eval_dir/x_true.npy              ← 新版 evaluate_spca.py 才有
  eval_dir/x_pred.npy              ← 新版 evaluate_spca.py 才有
  eval_dir/per_channel_residuals.npy ← 新版 evaluate_spca.py 才有
  eval_dir/evaluation_results.json

用法：
  # 只生成消融图
  python generate_paper_figures.py --ablation_only

  # 生成全部图（需要 eval 目录）
  python generate_paper_figures.py --spca_eval outputs_spca/latest

  # 指定聚焦哪个异常事件（MSHTrans 风格图）
  python generate_paper_figures.py --spca_eval outputs_spca/latest --zoom_event 2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator

# ─── 全局绘图风格（IEEE 会议论文风格）────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   9,
    "xtick.labelsize":  7.5,
    "ytick.labelsize":  7.5,
    "legend.fontsize":  7.5,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       200,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

OUT_DIR = Path("paper_figures")
OUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_events(y: np.ndarray):
    """提取异常事件区间列表 [(start, end), ...]"""
    events, in_e = [], False
    for i, v in enumerate(y):
        if v and not in_e:
            s = i; in_e = True
        elif not v and in_e:
            events.append((s, i - 1)); in_e = False
    if in_e:
        events.append((s, len(y) - 1))
    return events


def _shade_events(ax, t_axis, mask, color="#e74c3c", alpha=0.25, label=None):
    """在 ax 上为 mask==1 的区间着色"""
    in_r, first = False, True
    for i, v in enumerate(mask):
        if v and not in_r:
            s = t_axis[i]; in_r = True
        elif not v and in_r:
            ax.axvspan(s, t_axis[i], alpha=alpha, color=color, zorder=2,
                       label=label if first else None)
            in_r = False; first = False
    if in_r:
        ax.axvspan(s, t_axis[-1], alpha=alpha, color=color, zorder=2,
                   label=label if first else None)


def _maxpool_downsample(arr, factor):
    """沿 axis=0 做 max-pooling 下采样（保留尖峰）"""
    T = (len(arr) // factor) * factor
    if arr.ndim == 1:
        return arr[:T].reshape(-1, factor).max(axis=1)
    else:
        return arr[:T].reshape(-1, factor, arr.shape[1]).max(axis=1)


def _load_threshold(eval_dir: Path, raw_smoothed: np.ndarray, y_true: np.ndarray):
    try:
        res = json.loads((eval_dir / "evaluation_results.json").read_text())
        thr = res.get("threshold")
        if thr:
            return float(thr)
    except Exception:
        pass
    normal = raw_smoothed[y_true == 0]
    return float(np.percentile(normal, 99.5))


def _smooth(arr, w=10):
    """简单移动平均平滑"""
    if w <= 1:
        return arr
    kernel = np.ones(w) / w
    if arr.ndim == 1:
        return np.convolve(arr, kernel, mode="same")
    return np.stack([np.convolve(arr[:, c], kernel, mode="same")
                     for c in range(arr.shape[1])], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  图 1：消融实验条形图（仿 PSTG Fig.6）
# ═══════════════════════════════════════════════════════════════════════════════

def fig_ablation(out_dir: Path = OUT_DIR):
    data = _load_ablation_from_json()
    if data is None:
        data = [
            ("SpCA\n(Full)",         0.934, None),
            ("w/o Spectral\nDecomp", 0.872, None),
            ("w/o Channel\nAttn",    0.931, None),
            ("w/o Both\n(baseline)", 0.872, None),
        ]
        print("  ⚠ 使用内置消融数据（Standard 1）。"
              "运行 run_experiments.py 后可自动读取实测值。")

    labels = [d[0] for d in data]
    ev_f05 = [d[1] for d in data]
    af_f05 = [d[2] if len(d) > 2 and d[2] is not None else float("nan") for d in data]
    has_affil = any(not np.isnan(v) for v in af_f05)

    COLORS = ["#1a4a8a", "#f4a582", "#92c5de", "#d6604d"]

    if has_affil:
        fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.8),
                                  gridspec_kw={"wspace": 0.35})
        groups = [("Event-wise $F_{0.5}$", ev_f05),
                  ("Affiliation-based $F_{0.5}$", af_f05)]
    else:
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        axes = [ax]
        groups = [("Event-wise $F_{0.5}$", ev_f05)]

    for ax, (xlabel, values) in zip(axes, groups):
        x = np.arange(len(labels))
        bars = ax.bar(x, values, width=0.55, color=COLORS[:len(labels)],
                      edgecolor="white", linewidth=0.8, zorder=3)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold" if val == max(v for v in values if not np.isnan(v)) else "normal")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("$F_{0.5}$ Score")
        valid = [v for v in values if not np.isnan(v)]
        ax.set_ylim(max(0, min(valid) - 0.08), max(valid) + 0.06)
        ax.axhline(values[0], color=COLORS[0], lw=1.0, ls="--", alpha=0.45, zorder=2)
        ax.grid(axis="y", color="#e0e0e0", lw=0.6, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle("Ablation Study on ESA-AD", fontsize=10, y=1.02)
    _save(fig, out_dir / "fig_ablation.pdf")


def _load_ablation_from_json():
    p = Path("experiment_results.json")
    if not p.exists():
        return None
    try:
        res = json.loads(p.read_text())
        ablation = res.get("ablation", [])
        name_map = {
            "SpCA Full":            "SpCA\n(Full)",
            "w/o Spectral Decomp":  "w/o Spectral\nDecomp",
            "w/o Channel Attention":"w/o Channel\nAttn",
            "w/o Both (baseline)":  "w/o Both\n(baseline)",
        }
        data = []
        for entry in ablation[:4]:
            if entry is None:
                continue
            name = name_map.get(entry.get("name", ""), entry.get("name", ""))
            ev = entry.get("std1_ev_f05")
            af = entry.get("std1_af_f05")
            if ev is not None:
                data.append((name, float(ev),
                              float(af) if af is not None else float("nan")))
        return data if data else None
    except Exception as e:
        print(f"  读取 experiment_results.json 失败: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  图 2：全时序每通道分数（仿 MTGFlow Fig.13）
# ═══════════════════════════════════════════════════════════════════════════════

def fig_timeline(eval_dir: Path, out_dir: Path = OUT_DIR, ds_target: int = 8000):
    """
    6 个通道堆叠，显示全测试集时序的每通道平滑残差，
    蓝色阈值线，红色着色标注真实异常事件，事件标记 A1, A2…
    """
    y_true = np.load(eval_dir / "y_true.npy").astype(np.int32)
    T = len(y_true)

    # 优先用逐通道残差
    per_ch_path = eval_dir / "per_channel_residuals.npy"
    if per_ch_path.exists():
        per_ch = np.load(per_ch_path).astype(np.float64)   # [T, C]
        C = per_ch.shape[1]
        # 每通道平滑
        per_ch = _smooth(per_ch, w=15)
    else:
        # 退化：所有通道显示相同的 aggregate score
        raw = np.load(eval_dir / "raw_smoothed.npy").astype(np.float64)
        C = 6
        per_ch = np.tile(raw[:, None], (1, C))
        print("  ⚠ 未找到 per_channel_residuals.npy，用 aggregate score 代替所有通道。"
              "请重新运行 evaluate_spca.py 以获取逐通道残差。")

    # 下采样（max-pooling 保留尖峰）
    factor = max(1, T // ds_target)
    per_ch_ds = _maxpool_downsample(per_ch, factor)
    y_ds      = _maxpool_downsample(y_true, factor)
    t_ds      = np.arange(len(per_ch_ds)) * factor

    # 阈值（用全局 aggregate threshold 做参考）
    raw_smooth = np.load(eval_dir / "raw_smoothed.npy").astype(np.float64)
    global_thr = _load_threshold(eval_dir, raw_smooth, y_true)

    # 每通道独立阈值（99.8 分位数的正常段）
    ch_thresholds = []
    for c in range(C):
        normal_vals = per_ch[:, c][y_true == 0]
        ch_thresholds.append(float(np.percentile(normal_vals, 99.8))
                             if len(normal_vals) > 0 else global_thr)

    # 提取事件用于标注
    events = _extract_events(y_true)

    fig, axes = plt.subplots(C, 1, figsize=(7.5, C * 1.05),
                              sharex=True,
                              gridspec_kw={"hspace": 0.06})

    ch_names = [f"Ch {41+c}" for c in range(C)]

    for c, ax in enumerate(axes):
        score = per_ch_ds[:, c]
        thr_c = ch_thresholds[c]

        # 填充曲线
        ax.fill_between(t_ds, score, 0,
                        where=score >= 0, color="#c6dbef", alpha=0.8, lw=0)
        ax.plot(t_ds, score, color="#1a1a2e", lw=0.45, rasterized=True)

        # 阈值线（蓝色实线）
        ax.axhline(thr_c, color="#2166ac", lw=1.1, ls="-", zorder=4,
                   label=f"Thr={thr_c:.3f}")

        # 真实异常区（红色半透明）
        _shade_events(ax, t_ds, y_ds, color="#e74c3c", alpha=0.28,
                      label="Ground truth" if c == 0 else None)

        # 检测到的超阈区（橙色边框）
        detected = (score >= thr_c).astype(np.int32)
        _shade_events(ax, t_ds, detected, color="#e67e22", alpha=0.0)
        # 用竖线标检测到的超阈点
        ax.fill_between(t_ds, score, thr_c,
                        where=score >= thr_c,
                        color="#e74c3c", alpha=0.55, lw=0, zorder=3)

        # 事件标签 A1, A2, ...
        y_max = max(score.max() * 1.15, thr_c * 1.3)
        for ei, (es, ee) in enumerate(events):
            mid = (es + ee) / 2
            if 0 <= mid < T:
                ax.annotate(f"A{ei+1}",
                            xy=(mid, thr_c),
                            xytext=(mid, y_max * 0.88),
                            fontsize=5.5, color="#c0392b", fontweight="bold",
                            ha="center", va="bottom",
                            arrowprops=dict(arrowstyle="-", color="#c0392b",
                                            lw=0.6, alpha=0.7))

        # Y 轴
        ax.set_ylim(0, y_max)
        ax.set_yticks([])
        ax.set_ylabel(ch_names[c], fontsize=8, rotation=0,
                      labelpad=36, va="center")
        ax.spines["left"].set_visible(False)

        # 右侧打印阈值
        ax.text(t_ds[-1] * 1.002, thr_c, f"  {thr_c:.3f}",
                va="center", fontsize=6, color="#2166ac")

    axes[-1].set_xlabel("Time Step")

    # 全图图例
    legend_elems = [
        mpatches.Patch(color="#1a1a2e",  label="Anomaly score"),
        mpatches.Patch(color="#2166ac",  label="Threshold"),
        mpatches.Patch(color="#e74c3c", alpha=0.55, label="Score > threshold"),
        mpatches.Patch(color="#e74c3c", alpha=0.28, label="Ground truth anomaly"),
    ]
    axes[0].legend(handles=legend_elems, loc="upper right",
                   fontsize=6.5, ncol=2, framealpha=0.85, edgecolor="none")

    fig.suptitle("Per-Channel Anomaly Scores on ESA-AD Mission 1  "
                 f"(T = {T:,} steps, {len(events)} events)",
                 fontsize=9, y=1.01)

    _save(fig, out_dir / "fig_timeline.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#  图 3：缩放窗口 — 原始信号 + 重建 + 逐通道分数（仿 MSHTrans Fig.3）
# ═══════════════════════════════════════════════════════════════════════════════

def fig_mshtrans(eval_dir: Path, out_dir: Path = OUT_DIR,
                 zoom_event: int = 0, context: int = 1500,
                 show_channels: list = None):
    """
    MSHTrans Fig.3 风格：
    - 顶部 2 行：All-dims 信号叠加 + aggregate 分数
    - 每选定通道 2 行：原始+重建、该通道分数
    布局类似 MSHTrans 论文的 "Dimension X" 配对行。
    """
    # ── 加载数据 ──────────────────────────────────────────────────────────────
    y_true     = np.load(eval_dir / "y_true.npy").astype(np.int32)
    raw_smooth = np.load(eval_dir / "raw_smoothed.npy").astype(np.float64)
    threshold  = _load_threshold(eval_dir, raw_smooth, y_true)
    T = len(y_true)

    x_true_path = eval_dir / "x_true.npy"
    x_pred_path = eval_dir / "x_pred.npy"
    per_ch_path = eval_dir / "per_channel_residuals.npy"

    has_pred = x_true_path.exists() and x_pred_path.exists()
    has_per  = per_ch_path.exists()

    if not has_pred:
        print("  ⚠ 未找到 x_true.npy / x_pred.npy，只绘制 aggregate score。"
              "请重新运行 evaluate_spca.py。")

    if has_pred:
        x_true = np.load(x_true_path).astype(np.float64)
        x_pred = np.load(x_pred_path).astype(np.float64)
        C = x_true.shape[1]
    else:
        C = 6
        x_true = x_pred = None

    if show_channels is None:
        show_channels = list(range(min(3, C)))   # 默认显示前3个通道

    # ── 确定时间窗口 ──────────────────────────────────────────────────────────
    events = _extract_events(y_true)
    if not events:
        print("  ⚠ 没有异常事件，跳过 MSHTrans 风格图")
        return

    zoom_event = min(zoom_event, len(events) - 1)
    ev_s, ev_e = events[zoom_event]
    win_s = max(0, ev_s - context)
    win_e = min(T, ev_e + context)

    t = np.arange(win_s, win_e)
    gt    = y_true[win_s:win_e]
    score = raw_smooth[win_s:win_e]
    y_pred = (score >= threshold).astype(np.int32)

    # ── 布局 ──────────────────────────────────────────────────────────────────
    # 行结构：[all-signal, all-score] + per-ch × 2
    n_rows = 2 + len(show_channels) * 2
    row_heights = [1.3, 0.8] + [1.1, 0.7] * len(show_channels)

    fig, axes = plt.subplots(n_rows, 1, figsize=(7.0, sum(row_heights) * 0.85),
                              sharex=True,
                              gridspec_kw={"height_ratios": row_heights,
                                           "hspace": 0.06})

    colors_ch = plt.cm.tab10.colors

    # ── 行 0：All dims — 各通道标准化信号叠加 ──────────────────────────────
    ax = axes[0]
    ax.set_ylabel("All dims", fontsize=8, rotation=0, labelpad=36, va="center")
    if x_true is not None:
        seg = x_true[win_s:win_e, :]       # [W, C]
        for c in range(C):
            s = seg[:, c]
            s_n = (s - s.mean()) / (s.std() + 1e-8)
            ax.plot(t, s_n, color=colors_ch[c % 10], lw=0.55, alpha=0.75)
    else:
        ax.plot(t, score, color="#1f3a6e", lw=0.7)
    _shade_events(ax, t, gt,     color="#27ae60", alpha=0.22, label="Ground truth")
    _shade_events(ax, t, y_pred, color="#e67e22", alpha=0.15, label="Predicted")
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.legend(loc="upper right", fontsize=6.5, ncol=2,
              framealpha=0.8, edgecolor="none")

    # ── 行 1：All dims — aggregate 分数 ────────────────────────────────────
    ax = axes[1]
    ax.set_ylabel("All dims", fontsize=8, rotation=0, labelpad=36, va="center")
    ax.fill_between(t, score, 0, where=score >= 0, color="#fdd0cb", alpha=0.8, lw=0)
    ax.plot(t, score, color="#e74c3c", lw=0.7, label="Anomaly scores")
    ax.axhline(threshold, color="#c0392b", ls="--", lw=1.2,
               label=f"Threshold={threshold:.3f}")
    _shade_events(ax, t, y_pred, color="#e74c3c", alpha=0.25)
    _shade_events(ax, t, gt,     color="#27ae60", alpha=0.15)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.legend(loc="upper right", fontsize=6.5, ncol=2,
              framealpha=0.8, edgecolor="none")

    # ── 各通道：信号行 + 分数行 ────────────────────────────────────────────
    if has_per:
        per_ch = np.load(per_ch_path).astype(np.float64)
        per_ch_sm = _smooth(per_ch, w=10)
    else:
        per_ch_sm = None

    for k, c in enumerate(show_channels):
        row_sig   = axes[2 + k * 2]
        row_score = axes[2 + k * 2 + 1]
        ch_name   = f"Ch {41 + c}"
        col       = colors_ch[c % 10]

        # 信号行
        row_sig.set_ylabel(ch_name, fontsize=8, rotation=0,
                           labelpad=36, va="center")
        if x_true is not None:
            sig_seg  = x_true[win_s:win_e, c]
            pred_seg = x_pred[win_s:win_e, c]
            row_sig.plot(t, sig_seg,  color="#2c3e50", lw=0.65,
                         label="Original")
            row_sig.plot(t, pred_seg, color="#e67e22", lw=0.65, alpha=0.85,
                         label="Reconstruction")
            row_sig.legend(loc="upper right", fontsize=6.5,
                           framealpha=0.8, edgecolor="none")
        else:
            row_sig.text(0.5, 0.5, "x_true.npy not available",
                         transform=row_sig.transAxes, ha="center", va="center",
                         fontsize=8, color="gray")
        _shade_events(row_sig, t, gt, color="#27ae60", alpha=0.2)
        row_sig.set_yticks([]); row_sig.spines["left"].set_visible(False)

        # 分数行
        row_score.set_ylabel(ch_name, fontsize=8, rotation=0,
                             labelpad=36, va="center")
        if per_ch_sm is not None:
            ch_score = per_ch_sm[win_s:win_e, c]
            ch_thr_c = float(np.percentile(per_ch_sm[:, c][y_true == 0], 99.8))
        else:
            ch_score = score
            ch_thr_c = threshold
        row_score.fill_between(t, ch_score, 0,
                               where=ch_score >= 0, color="#fdd0cb", alpha=0.75, lw=0)
        row_score.plot(t, ch_score, color="#e74c3c", lw=0.65)
        row_score.axhline(ch_thr_c, color="#c0392b", ls="--", lw=1.0)
        _shade_events(row_score, t, y_pred, color="#e74c3c", alpha=0.22)
        _shade_events(row_score, t, gt,     color="#27ae60", alpha=0.15)
        row_score.set_yticks([]); row_score.spines["left"].set_visible(False)

    axes[-1].set_xlabel("Time Step")

    # 事件标注
    mid = (ev_s + ev_e) / 2
    ev_label = f"Event {zoom_event+1}"
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.annotate(ev_label, xy=(mid, ymax * 0.95),
                    fontsize=6, color="#c0392b", ha="center", va="top",
                    fontweight="bold")

    # 顶部图例行（仿 MSHTrans 图顶部图例）
    legend_elems = [
        mpatches.Patch(color="#2c3e50",  label="Original time series"),
        mpatches.Patch(color="#e74c3c",  label="Anomaly scores"),
        mpatches.Patch(color="#c0392b", alpha=0.7, linestyle="--",
                       label="Threshold", linewidth=1.2),
        mpatches.Patch(color="#e67e22",  label="Reconstruction"),
        mpatches.Patch(color="#27ae60", alpha=0.35, label="Real anomalies"),
        mpatches.Patch(color="#e74c3c", alpha=0.35, label="Predict anomalies"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.04), fontsize=7, framealpha=0.9,
               edgecolor="none", columnspacing=1.0)

    ch_str = ", ".join([f"Ch {41+c}" for c in show_channels])
    fig.suptitle(
        f"SpCA Detection — ESA-AD Mission 1  "
        f"(Event {zoom_event+1}/{len(events)},  "
        f"context ±{context} steps,  channels: {ch_str})",
        fontsize=8.5, y=1.10
    )

    _save(fig, out_dir / "fig_mshtrans.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#  图 4（可选）：方法对比水平条形图
# ═══════════════════════════════════════════════════════════════════════════════

def fig_method_comparison(out_dir: Path = OUT_DIR):
    """SpCA vs PSTG vs baselines 的双指标水平条形图（Standard 2，24 events）"""
    methods = [
        ("DLinear",       0.394, 0.767),
        ("FreTS",         0.764, 0.846),
        ("TSMixer",       0.798, 0.784),
        ("WPMixer",       0.806, 0.849),
        ("iTransformer",  0.834, 0.811),
        ("Crossformer",   0.836, 0.869),
        ("TimeFilter",    0.835, 0.869),
        ("PatchTST",      0.894, 0.885),
        ("PSTG",          0.917, 0.892),
        ("SpCA (Ours)",   0.934, None),
    ]
    names = [m[0] for m in methods]
    ev    = [m[1] for m in methods]
    af    = [m[2] if m[2] is not None else float("nan") for m in methods]
    has_affil = any(not np.isnan(v) for v in af)

    n = len(names)
    y = np.arange(n)

    if has_affil:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3.8),
                                        sharey=True,
                                        gridspec_kw={"wspace": 0.06})
        pairs = [(ax1, ev, "Event-wise $F_{0.5}$"),
                 (ax2, af, "Affiliation-based $F_{0.5}$")]
    else:
        fig, ax1 = plt.subplots(figsize=(3.0, 3.8))
        pairs = [(ax1, ev, "Event-wise $F_{0.5}$")]

    for ax, vals, xlabel in pairs:
        colors = ["#1a4a8a" if "Ours" in nm else
                  "#d6604d" if nm == "PSTG" else "#a8c8e8"
                  for nm in names]
        bars = ax.barh(y, vals, height=0.6, color=colors,
                       edgecolor="white", lw=0.5, zorder=3)
        for bar, val, nm in zip(bars, vals, names):
            if not np.isnan(val):
                ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", ha="left", fontsize=7,
                        fontweight="bold" if "Ours" in nm or nm == "PSTG" else "normal")
        xvals = [v for v in vals if not np.isnan(v)]
        ax.set_xlim(max(0, min(xvals) - 0.06), 1.06)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#e0e0e0", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        if ax is pairs[0][0]:
            ax.set_yticks(y)
            ax.set_yticklabels(names)
        ax.invert_yaxis()

    legend_elems = [
        mpatches.Patch(color="#1a4a8a", label="SpCA (Ours)"),
        mpatches.Patch(color="#d6604d", label="PSTG"),
        mpatches.Patch(color="#a8c8e8", label="Baselines"),
    ]
    pairs[0][0].legend(handles=legend_elems, loc="lower right",
                       fontsize=7, framealpha=0.8, edgecolor="none")
    fig.suptitle("Performance Comparison on ESA-AD (Standard 2)", fontsize=9, y=1.02)
    _save(fig, out_dir / "fig_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#  通用保存
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, path: Path):
    path = Path(path)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="生成 SpCA 论文配图")
    p.add_argument("--spca_eval",    type=str, default=None)
    p.add_argument("--ablation_only",action="store_true")
    p.add_argument("--zoom_event",   type=int, default=0,
                   help="MSHTrans 风格图聚焦第几个事件（0-indexed）")
    p.add_argument("--context",      type=int, default=1500,
                   help="事件前后显示步数")
    p.add_argument("--channels",     type=int, nargs="+", default=None,
                   help="MSHTrans 图显示哪些通道（0-indexed，默认 0 1 2）")
    p.add_argument("--out_dir",      type=str, default="paper_figures")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    sys.path.insert(0, ".")

    print(f"\n=== 生成 SpCA 论文配图 ===")
    print(f"输出目录：{out_dir.absolute()}\n")

    # 图 1：消融条形图（始终生成）
    print("▶ 消融实验条形图（仿 PSTG Fig.6）...")
    fig_ablation(out_dir)

    if args.ablation_only:
        print("\n完成（--ablation_only）")
        return

    # 解析 eval 目录
    eval_dir = None
    if args.spca_eval:
        eval_dir = Path(args.spca_eval)
        if not eval_dir.exists():
            eval_dir = Path("outputs_spca") / "latest"
    else:
        # 自动找最新的 eval 目录
        for candidate in [Path("outputs_spca/latest"),
                          Path("outputs_spca")]:
            if (candidate / "raw_smoothed.npy").exists():
                eval_dir = candidate
                break
            latest = sorted(candidate.glob("eval_*/raw_smoothed.npy"))
            if latest:
                eval_dir = latest[-1].parent
                break

    if eval_dir is None or not (eval_dir / "raw_smoothed.npy").exists():
        print("  ⚠ 找不到 eval 目录，跳过需要评估数据的图。"
              "请指定 --spca_eval <路径>")
    else:
        print(f"  使用 eval 目录：{eval_dir}\n")

        # 图 2：全时序每通道（仿 MTGFlow Fig.13）
        print("▶ 全时序每通道分数图（仿 MTGFlow Fig.13）...")
        fig_timeline(eval_dir, out_dir)

        # 图 3：缩放窗口（仿 MSHTrans Fig.3）
        print(f"▶ 缩放窗口检测图（仿 MSHTrans Fig.3，event={args.zoom_event}）...")
        fig_mshtrans(eval_dir, out_dir,
                     zoom_event=args.zoom_event,
                     context=args.context,
                     show_channels=args.channels)

    # 图 4：方法对比
    print("▶ 方法对比水平条形图...")
    fig_method_comparison(out_dir)

    print(f"\n完成！图片位于：{out_dir.absolute()}")
    print("─" * 48)
    print("  fig_ablation.pdf   — 消融分组条形图        → Section 消融实验")
    print("  fig_timeline.pdf   — 全时序逐通道分数图    → Section 检测结果")
    print("  fig_mshtrans.pdf   — 信号+重建+分数缩放图  → Section 检测结果")
    print("  fig_comparison.pdf — 方法对比水平条形图    → Section 对比实验")


if __name__ == "__main__":
    main()
