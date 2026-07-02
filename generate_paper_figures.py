"""
SpCA 论文配图生成脚本
====================
生成两张论文配图：

  fig_detection.pdf — 局部多事件检测图（仿 MTGFlow Fig.13 风格）
                      自动找事件最密集的 N 步窗口，6 通道叠放，
                      每通道：分数折线 + 阈值线 + 真实异常着色

  fig_casestudy.pdf — 单事件缩放图（仿 MSHTrans Fig.3 风格）
                      自动选最短且检测效果最好的事件，
                      每通道：原始信号+重建 / 逐通道分数+阈值

依赖文件（evaluate_spca.py 生成）：
  eval_dir/raw_smoothed.npy
  eval_dir/y_true.npy
  eval_dir/x_true.npy              ← 新版 evaluate_spca.py
  eval_dir/x_pred.npy
  eval_dir/per_channel_residuals.npy

用法：
  python generate_paper_figures.py --spca_eval outputs_spca/latest
  python generate_paper_figures.py --spca_eval outputs_spca/latest \\
      --window 400000 --zoom_event 5 --context 2000 --channels 0 2 4
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         9,
    "axes.labelsize":    9,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.fontsize":   7.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        200,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

OUT_DIR = Path("paper_figures")
OUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────
#  工具函数
# ──────────────────────────────────────────────────────────────────

def _extract_events(y):
    evs, in_e = [], False
    for i, v in enumerate(y):
        if v and not in_e:  s = i; in_e = True
        elif not v and in_e: evs.append((s, i-1)); in_e = False
    if in_e: evs.append((s, len(y)-1))
    return evs

def _shade(ax, t, mask, color, alpha, label=None):
    in_r = False; first = True
    for i, v in enumerate(mask):
        if v and not in_r:  s0 = t[i]; in_r = True
        elif not v and in_r:
            ax.axvspan(s0, t[i], color=color, alpha=alpha, lw=0, zorder=2,
                       label=label if first else None)
            in_r = False; first = False
    if in_r:
        ax.axvspan(s0, t[-1], color=color, alpha=alpha, lw=0, zorder=2,
                   label=label if first else None)

def _load_threshold(eval_dir, raw, y):
    try:
        v = json.loads((eval_dir/"evaluation_results.json").read_text()).get("threshold")
        if v: return float(v)
    except: pass
    return float(np.percentile(raw[y == 0], 99.5))

def _smooth(x, w=15):
    k = np.ones(w)/w
    if x.ndim == 1: return np.convolve(x, k, mode="same")
    return np.stack([np.convolve(x[:,c], k, mode="same") for c in range(x.shape[1])], 1)

def _maxpool(arr, factor):
    T = (len(arr)//factor)*factor
    if arr.ndim == 1: return arr[:T].reshape(-1, factor).max(1)
    return arr[:T].reshape(-1, factor, arr.shape[1]).max(1)

def _save(fig, path):
    path = Path(path)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf",".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ──────────────────────────────────────────────────────────────────
#  图 1：局部多事件检测图（仿 MTGFlow Fig.13）
# ──────────────────────────────────────────────────────────────────

def _densest_window(events, T, win_size):
    """找包含最多完整异常事件的 win_size 步窗口，返回 (start, end)"""
    if not events: return 0, min(win_size, T)
    best_start, best_count = 0, 0
    # 滑动窗口：以每个事件起点为候选窗口起点
    candidates = sorted(set(
        [max(0, s - win_size//4) for s, e in events] +
        [max(0, T - win_size)]
    ))
    for cs in candidates:
        ce = min(cs + win_size, T)
        count = sum(1 for s, e in events if cs <= s and e <= ce)
        if count > best_count:
            best_count, best_start = count, cs
    return best_start, best_start + win_size


def fig_detection(eval_dir: Path, out_dir: Path = OUT_DIR,
                  win_size: int = 500_000, ds_pts: int = 3000):
    """
    MTGFlow Fig.13 风格：截取事件最密集的 win_size 步，
    6 通道叠放，每通道 = 逐通道平滑残差 + 阈值线 + 真实异常着色。
    """
    y_true     = np.load(eval_dir/"y_true.npy").astype(np.int32)
    raw_smooth = np.load(eval_dir/"raw_smoothed.npy").astype(np.float64)
    threshold  = _load_threshold(eval_dir, raw_smooth, y_true)
    T          = len(y_true)
    events     = _extract_events(y_true)

    # 加载逐通道数据
    per_ch_path = eval_dir/"per_channel_residuals.npy"
    if per_ch_path.exists():
        per_ch = _smooth(np.load(per_ch_path).astype(np.float64), w=20)
        C = per_ch.shape[1]
    else:
        print("  ⚠ 无 per_channel_residuals.npy，用 aggregate score 代替所有通道")
        C = 6
        per_ch = np.tile(raw_smooth[:,None], (1, C))

    # 找最密集窗口
    win_start, win_end = _densest_window(events, T, win_size)
    win_events = [(s, e) for s, e in events if win_start <= s and e <= win_end]
    print(f"  窗口 [{win_start:,}–{win_end:,}]，含 {len(win_events)}/{len(events)} 个事件")

    # 切片
    sl        = slice(win_start, win_end)
    per_sl    = per_ch[sl]      # [W, C]
    y_sl      = y_true[sl]

    # 下采样（max-pooling 保留 spike）
    factor = max(1, (win_end - win_start) // ds_pts)
    per_ds = _maxpool(per_sl, factor)
    y_ds   = _maxpool(y_sl.astype(float), factor)
    t_ds   = np.arange(len(per_ds)) * factor + win_start

    # 每通道独立阈值（正常段 99.5 分位）
    ch_thr = []
    for c in range(C):
        nm = per_ch[:, c][y_true == 0]
        ch_thr.append(float(np.percentile(nm, 99.5)) if len(nm) else threshold)

    # ── 绘图：6 行叠放 ────────────────────────────────────────────
    fig, axes = plt.subplots(C, 1, figsize=(7.5, C * 1.15),
                              sharex=True,
                              gridspec_kw={"hspace": 0.06})
    CH_COLORS = ["#1b7837","#762a83","#e08214","#2166ac","#d6604d","#4d9221"]

    for c, ax in enumerate(axes):
        score = per_ds[:, c]
        thr_c = ch_thr[c]
        col   = CH_COLORS[c % len(CH_COLORS)]

        # 背景填充
        ax.fill_between(t_ds, score, 0, color="#deebf7", alpha=0.7, lw=0)
        # 超阈区填充（深色）
        ax.fill_between(t_ds, score, thr_c,
                        where=score >= thr_c,
                        color=col, alpha=0.55, lw=0, zorder=3)
        # 分数折线
        ax.plot(t_ds, score, color=col, lw=0.75, zorder=4)
        # 阈值线（蓝色实线，仿 MTGFlow）
        ax.axhline(thr_c, color="#2166ac", lw=1.2, ls="-", zorder=5)
        # 真实异常（粉色半透明）
        _shade(ax, t_ds, y_ds, "#f4a582", 0.45,
               label="Ground truth" if c == 0 else None)

        # 事件编号标注（只标在窗口内的）
        for ei, (es, ee) in enumerate(events):
            if win_start <= (es+ee)//2 <= win_end:
                mid = (es + ee) / 2
                ymax = max(score.max(), thr_c) * 1.15
                ax.text(mid, ymax, f"A{ei+1}",
                        ha="center", va="bottom", fontsize=6.5,
                        color="#c0392b", fontweight="bold")

        ax.set_ylabel(f"Ch {41+c}", fontsize=8.5, rotation=0,
                      labelpad=36, va="center")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        # 右侧阈值标注
        ax.text(t_ds[-1]*1.001, thr_c, f" {thr_c:.3f}",
                va="center", fontsize=6, color="#2166ac",
                clip_on=False)

    axes[-1].set_xlabel("Time Step")
    axes[-1].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M" if x >= 1e6 else f"{int(x):,}"))

    # 图例
    legend_elems = [
        mpatches.Patch(color="#2166ac", label="Threshold"),
        mpatches.Patch(color="#f4a582", alpha=0.6, label="Ground truth anomaly"),
        mpatches.Patch(color="gray",    alpha=0.55, label="Score > threshold"),
    ]
    axes[0].legend(handles=legend_elems, loc="upper right",
                   fontsize=7, ncol=3, framealpha=0.9, edgecolor="none")

    fig.suptitle(
        f"SpCA Per-Channel Anomaly Scores — ESA-AD Mission 1  "
        f"({win_start:,}–{win_end:,}, {len(win_events)} events shown)",
        fontsize=9, y=1.01
    )
    _save(fig, out_dir/"fig_detection.pdf")


# ──────────────────────────────────────────────────────────────────
#  图 2：单事件缩放图（仿 MSHTrans Fig.3）
# ──────────────────────────────────────────────────────────────────

def _pick_best_event(events, raw_smooth, y_true, threshold):
    """
    选最适合展示的事件：
    1. 检测到了（peak > threshold）
    2. 持续时间短（优先 < 3000 步，实在没有则放宽）
    3. IoU(detection, ground_truth) 最高
    """
    y_pred = (raw_smooth >= threshold).astype(np.int32)

    def score_event(idx, max_dur):
        s, e = events[idx]
        if e - s + 1 > max_dur: return -1
        peak = raw_smooth[s:e+1].max()
        if peak < threshold: return -1
        inter = (y_true[s:e+1] & y_pred[s:e+1]).sum()
        union = (y_true[s:e+1] | y_pred[s:e+1]).sum()
        iou   = inter / (union + 1e-8)
        return iou * (peak / threshold)

    # 逐步放宽持续时间限制
    for max_dur in [2000, 5000, 15000, 10**9]:
        scores = [(score_event(i, max_dur), i) for i in range(len(events))]
        valid  = [(sc, i) for sc, i in scores if sc > 0]
        if valid:
            best_i = max(valid)[1]
            print(f"  自动选择 Event {best_i+1}/{len(events)}"
                  f"  duration={events[best_i][1]-events[best_i][0]+1}"
                  f"  peak={raw_smooth[events[best_i][0]:events[best_i][1]+1].max():.4f}"
                  f"  (max_dur={max_dur})")
            return best_i

    print("  ⚠ 没有任何事件被检测到，使用 Event 0")
    return 0


def fig_casestudy(eval_dir: Path, out_dir: Path = OUT_DIR,
                  zoom_event: int = -1, context: int = -1,
                  show_channels: list = None):
    """
    MSHTrans Fig.3 风格：
      Row 0      : 聚合分数 + 阈值（整体检测视图）
      Row 1,3,5  : Ch X  原始信号(深蓝) + 预测重建(橙)
      Row 2,4,6  : Ch X  逐通道分数 + 阈值
    """
    sys.path.insert(0, ".")
    y_true     = np.load(eval_dir/"y_true.npy").astype(np.int32)
    raw_smooth = np.load(eval_dir/"raw_smoothed.npy").astype(np.float64)
    threshold  = _load_threshold(eval_dir, raw_smooth, y_true)
    T          = len(y_true)

    has_pred = (eval_dir/"x_true.npy").exists()
    has_per  = (eval_dir/"per_channel_residuals.npy").exists()
    x_true_all = np.load(eval_dir/"x_true.npy").astype(np.float64) if has_pred else None
    x_pred_all = np.load(eval_dir/"x_pred.npy").astype(np.float64) if has_pred else None
    per_ch_all = _smooth(np.load(eval_dir/"per_channel_residuals.npy").astype(np.float64), w=10) \
                 if has_per else None
    C = x_true_all.shape[1] if has_pred else 6

    if show_channels is None:
        show_channels = list(range(min(3, C)))

    events = _extract_events(y_true)
    if not events:
        print("  ⚠ 无异常事件，跳过"); return

    # 自动或手动选事件
    if zoom_event < 0:
        zoom_event = _pick_best_event(events, raw_smooth, y_true, threshold)
    else:
        zoom_event = min(zoom_event, len(events)-1)

    ev_s, ev_e = events[zoom_event]
    ev_dur = ev_e - ev_s + 1

    # context：自动设为事件长度的 1.5 倍（至少 500 步）
    if context < 0:
        context = max(500, ev_dur * 3 // 2)
        print(f"  自动 context={context} 步（事件长度={ev_dur}）")

    win_s = max(0, ev_s - context)
    win_e = min(T, ev_e + context)
    t      = np.arange(win_s, win_e)
    gt     = y_true[win_s:win_e]
    agg_sc = raw_smooth[win_s:win_e]
    y_pred = (agg_sc >= threshold).astype(np.int32)

    # ── 布局 ──────────────────────────────────────────────────────
    N_ch  = len(show_channels)
    row_h = [0.9] + [1.2, 0.7] * N_ch
    fig, axes = plt.subplots(len(row_h), 1,
                              figsize=(7.0, sum(row_h) * 0.92),
                              sharex=True,
                              gridspec_kw={"height_ratios": row_h, "hspace": 0.04})

    # ── Row 0：聚合分数 ────────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(t, agg_sc, 0, color="#fdd0cb", alpha=0.6, lw=0)
    ax.fill_between(t, agg_sc, threshold,
                    where=agg_sc >= threshold,
                    color="#d6604d", alpha=0.65, lw=0, zorder=3, label="Detected")
    ax.plot(t, agg_sc, color="#d6604d", lw=0.85, label="Anomaly score")
    ax.axhline(threshold, color="#67001f", ls="--", lw=1.3,
               label=f"Threshold ε*={threshold:.3f}")
    _shade(ax, t, gt, "#41ab5d", 0.25, "Ground truth")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("Score\n(all ch.)", fontsize=7.5, rotation=0, labelpad=52, va="center")
    ax.legend(loc="upper right", fontsize=7, ncol=4,
              framealpha=0.9, edgecolor="none", columnspacing=0.7)

    # ── 各通道：信号行 + 分数行 ────────────────────────────────────
    CH_COLORS = plt.cm.tab10.colors
    for k, c in enumerate(show_channels):
        ax_s = axes[1 + k*2]
        ax_r = axes[2 + k*2]
        ch   = f"Ch {41+c}"

        # 信号行
        ax_s.set_ylabel(ch, fontsize=8.5, rotation=0, labelpad=52, va="center")
        if x_true_all is not None:
            sig  = x_true_all[win_s:win_e, c]
            pred = x_pred_all[win_s:win_e, c]
            lo, hi = np.percentile(sig, 0.5), np.percentile(sig, 99.5)
            pad = (hi - lo) * 0.2
            ax_s.set_ylim(lo - pad, hi + pad * 2.5)
            ax_s.plot(t, sig,  color="#2c3e50", lw=0.6, label="Original", zorder=3)
            ax_s.plot(t, pred, color="#e67e22", lw=0.6, label="Reconstruction",
                      alpha=0.9, zorder=4)
            if k == 0:
                ax_s.legend(loc="upper right", fontsize=7,
                             framealpha=0.9, edgecolor="none")
        _shade(ax_s, t, gt, "#41ab5d", 0.22)
        ax_s.set_yticks([]); ax_s.spines["left"].set_visible(False)

        # 分数行
        ax_r.set_ylabel(ch, fontsize=8.5, rotation=0, labelpad=52, va="center")
        if per_ch_all is not None:
            ch_sc  = per_ch_all[win_s:win_e, c]
            ch_thr = float(np.percentile(per_ch_all[:, c][y_true == 0], 99.5))
        else:
            ch_sc = agg_sc; ch_thr = threshold
        ax_r.fill_between(t, ch_sc, 0, color="#fdd0cb", alpha=0.6, lw=0)
        ax_r.fill_between(t, ch_sc, ch_thr,
                          where=ch_sc >= ch_thr,
                          color="#d6604d", alpha=0.6, lw=0, zorder=3)
        ax_r.plot(t, ch_sc, color="#d6604d", lw=0.65)
        ax_r.axhline(ch_thr, color="#67001f", ls="--", lw=1.0)
        _shade(ax_r, t, gt, "#41ab5d", 0.20)
        ax_r.set_yticks([]); ax_r.spines["left"].set_visible(False)

    axes[-1].set_xlabel("Time Step")

    # 事件标注（只在顶部标一次）
    ev_mid  = (ev_s + ev_e) / 2
    y0_top  = axes[0].get_ylim()[1]
    axes[0].annotate(f"Event {zoom_event+1}",
                     xy=(ev_mid, y0_top), xytext=(ev_mid, y0_top * 1.08),
                     fontsize=8.5, color="#c0392b", fontweight="bold",
                     ha="center", va="bottom", annotation_clip=False,
                     arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.8))

    ch_str = "+".join([f"Ch{41+c}" for c in show_channels])
    fig.suptitle(
        f"SpCA Detection on ESA-AD Mission 1  "
        f"(Event {zoom_event+1}/{len(events)},  ±{context} steps,  {ch_str})",
        fontsize=8.5, y=1.03
    )
    _save(fig, out_dir/"fig_casestudy.pdf")


# ──────────────────────────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spca_eval",  type=str, default=None)
    p.add_argument("--window",     type=int, default=500_000,
                   help="fig_detection 截取的窗口步数（默认 500000）")
    p.add_argument("--zoom_event", type=int, default=-1,
                   help="fig_casestudy 聚焦事件序号（0-indexed，-1=自动）")
    p.add_argument("--context",    type=int, default=-1,
                   help="fig_casestudy 事件前后步数（-1=自动=事件长度×1.5）")
    p.add_argument("--channels",   type=int, nargs="+", default=None,
                   help="fig_casestudy 显示哪些通道（0-indexed，默认 0 1 2）")
    p.add_argument("--out_dir",    type=str, default="paper_figures")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    sys.path.insert(0, ".")

    # 找 eval 目录
    eval_dir = Path(args.spca_eval) if args.spca_eval else None
    if eval_dir is None or not eval_dir.exists():
        for cand in [Path("outputs_spca/latest"),
                     *[p.parent for p in sorted(
                         Path("outputs_spca").glob("eval_*/raw_smoothed.npy"),
                         reverse=True)]]:
            if (cand/"raw_smoothed.npy").exists():
                eval_dir = cand; break

    if eval_dir is None:
        print("❌ 找不到 eval 目录，请指定 --spca_eval <路径>"); return

    print(f"\neval 目录：{eval_dir}")
    print(f"输出目录：{out_dir.absolute()}\n")

    print("▶ 局部多事件检测图 (fig_detection.pdf) ...")
    fig_detection(eval_dir, out_dir, win_size=args.window)

    print("\n▶ 单事件缩放图 (fig_casestudy.pdf) ...")
    fig_casestudy(eval_dir, out_dir,
                  zoom_event=args.zoom_event,
                  context=args.context,
                  show_channels=args.channels)

    print("\n完成。")
    print("─" * 45)
    print("  fig_detection.pdf  — 多事件局部检测（仿 MTGFlow Fig.13）")
    print("  fig_casestudy.pdf  — 单事件缩放（仿 MSHTrans Fig.3）")


if __name__ == "__main__":
    main()
