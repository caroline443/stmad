"""
生成论文专用可视化图（publication-quality）

从已有的 eval 目录加载数据，自动找到最显著的异常事件并放大展示。

输出：
  paper_figure.png  —— 双面板图：
    上：2个代表性通道的原始信号（真实异常区高亮）
    下：MTA异常分数 + 阈值线 + 检测区 + 真实区
  paper_figure.pdf  —— 矢量版（LaTeX 导入用）

用法：
  python plot_mta_paper.py --eval_dir outputs_mta/eval_20260628_151233
  python plot_mta_paper.py   # 默认读 outputs_mta/latest
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ─────────────────────────────────────────────────────────────────────────────
#  工具：提取连续事件
# ─────────────────────────────────────────────────────────────────────────────

def extract_events(binary):
    """返回 [(start, end), ...]"""
    events, in_ev, s = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_ev:   s = i; in_ev = True
        elif not v and in_ev: events.append((s, i - 1)); in_ev = False
    if in_ev: events.append((s, len(binary) - 1))
    return events


def find_best_window(y_true, raw_smoothed, anomaly_scores,
                     half_win=1500, min_dur=5):
    """
    找最适合展示的窗口：
    - 包含至少一个真实异常（duration >= min_dur）
    - 且该异常期间重建误差峰值最高（视觉效果最好）
    """
    T = len(y_true)
    events = [(s, e) for s, e in extract_events(y_true) if e - s + 1 >= min_dur]
    if not events:
        # 没有长事件，取误差最高点附近
        peak = int(np.argmax(raw_smoothed))
        return max(0, peak - half_win), min(T, peak + half_win)

    # 按峰值误差排序，取最高的事件
    scores = [raw_smoothed[s:e+1].max() for s, e in events]
    best_idx = int(np.argmax(scores))
    s, e = events[best_idx]
    center = (s + e) // 2
    t0 = max(0, center - half_win)
    t1 = min(T, center + half_win)
    return t0, t1


# ─────────────────────────────────────────────────────────────────────────────
#  主图
# ─────────────────────────────────────────────────────────────────────────────

def plot_paper_figure(
    x_true, raw_smoothed, anomaly_scores, y_true,
    t0, t1, threshold,
    n_channels=6, channels_to_show=(0, 2),   # 显示哪两个通道（索引）
    output_dir=None,
    dpi=200,
):
    """
    双面板 publication-quality 图：
      上：原始信号（选2个通道），真实异常区高亮
      下：MTA异常分数，含阈值、检测区、真实区
    """
    t   = np.arange(t0, t1)
    seg = slice(t0, t1)
    gt  = y_true[seg].astype(bool)
    det = (anomaly_scores[seg] > 0).astype(bool)
    scr = raw_smoothed[seg]
    sig = x_true[seg]             # [T_seg, C]

    colors_ch = plt.cm.tab10.colors

    # ── 布局 ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 5), dpi=dpi)
    gs  = GridSpec(
        3, 1, figure=fig,
        height_ratios=[1, 1, 1.2],
        hspace=0.08,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    # ── 辅助：标注区域 ────────────────────────────────────────────────────────
    def shade_gt(ax):
        """绿色半透明：真实异常区"""
        for s, e in extract_events(gt):
            ax.axvspan(t[s], t[e], color="#2ecc71", alpha=0.20, lw=0)

    def shade_det(ax):
        """橙色半透明：MTA 检出区（非 GT 区域为 FP）"""
        for s, e in extract_events(det):
            ax.axvspan(t[s], t[e], color="#e67e22", alpha=0.18, lw=0)

    # ── 上面板：通道信号 ───────────────────────────────────────────────────────
    ch_labels = [f"Ch {41 + c}" for c in range(n_channels)]

    for row, ch in enumerate(channels_to_show[:2]):   # 最多画2个通道
        ax = axes[row]
        ax.plot(t, sig[:, ch],
                color=colors_ch[ch % 10], lw=0.8, zorder=3)
        shade_gt(ax)
        shade_det(ax)
        ax.set_ylabel(ch_labels[ch], fontsize=9, labelpad=2)
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(labelbottom=False, labelsize=8)
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=3, prune="both"))
        # 去掉多余的 spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── 下面板：异常分数 ───────────────────────────────────────────────────────
    ax_s = axes[2]
    shade_gt(ax_s)
    shade_det(ax_s)
    ax_s.plot(t, scr, color="#2980b9", lw=0.9, zorder=3, label="Anomaly Score")
    ax_s.axhline(threshold, color="#e74c3c", ls="--", lw=1.2,
                 label=f"Threshold ({threshold:.3f})", zorder=4)
    ax_s.set_xlim(t[0], t[-1])
    ax_s.set_xlabel("Time Step", fontsize=9)
    ax_s.set_ylabel("Score", fontsize=9, labelpad=2)
    ax_s.tick_params(labelsize=8)
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    ax_s.yaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(nbins=3, prune="both"))

    # ── 图例 ──────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color="#2ecc71", alpha=0.5, label="Ground-Truth Anomaly"),
        mpatches.Patch(color="#e67e22", alpha=0.5, label="MTA Detection"),
        mpatches.Patch(color="#2980b9",             label="Anomaly Score"),
        mpatches.Patch(color="#e74c3c",             label="POT Threshold"),
    ]
    ax_s.legend(handles=legend_patches, fontsize=7.5,
                loc="upper right", framealpha=0.85,
                ncol=2, handlelength=1.2, columnspacing=0.8)

    # ── 标题 ──────────────────────────────────────────────────────────────────
    fig.suptitle(
        "MTA Anomaly Detection on ESA-AD Spacecraft Telemetry",
        fontsize=10, y=0.98,
    )

    # ── 输出 ──────────────────────────────────────────────────────────────────
    out_dir = Path(output_dir) if output_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "paper_figure.png"
    pdf_path = out_dir / "paper_figure.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path,           bbox_inches="tight")
    plt.close(fig)
    print(f"  → {png_path}")
    print(f"  → {pdf_path}  (LaTeX 用矢量版)")


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir",    type=str, default=None,
                   help="eval 目录（默认 outputs_mta/latest）")
    p.add_argument("--channels",    type=int, nargs="+", default=[0, 2],
                   help="展示哪两个通道（0-index），默认 0 2（Ch41, Ch43）")
    p.add_argument("--half_win",    type=int, default=1500,
                   help="窗口半径（时间步数），默认 1500")
    p.add_argument("--t0",          type=int, default=None,
                   help="手动指定窗口起点（覆盖自动查找）")
    p.add_argument("--t1",          type=int, default=None,
                   help="手动指定窗口终点（覆盖自动查找）")
    p.add_argument("--out",         type=str, default=None,
                   help="输出目录（默认与 eval_dir 相同）")
    p.add_argument("--n_channels",  type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()

    # ── 定位 eval 目录 ────────────────────────────────────────────────────────
    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
    else:
        eval_dir = Path("outputs_mta/latest")
        if eval_dir.is_symlink():
            eval_dir = eval_dir.resolve()

    if not eval_dir.exists():
        print(f"目录不存在：{eval_dir}"); return

    print(f"读取：{eval_dir}")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    raw_smoothed   = np.load(eval_dir / "raw_smoothed.npy")
    anomaly_scores = np.load(eval_dir / "anomaly_scores.npy")
    y_true         = np.load(eval_dir / "y_true.npy")

    results   = json.loads((eval_dir / "evaluation_results.json").read_text())
    threshold = float(results.get("threshold", raw_smoothed[anomaly_scores > 0].min()
                                  if (anomaly_scores > 0).any() else 0.0))

    x_true_path = eval_dir / "x_true.npy"
    if not x_true_path.exists():
        print("未找到 x_true.npy，请重新运行 evaluate_mta.py"); return
    x_true = np.load(x_true_path)

    # ── 自动/手动定位窗口 ─────────────────────────────────────────────────────
    if args.t0 is not None and args.t1 is not None:
        t0, t1 = args.t0, args.t1
        print(f"手动窗口：[{t0}, {t1}]")
    else:
        t0, t1 = find_best_window(
            y_true, raw_smoothed, anomaly_scores, half_win=args.half_win
        )
        print(f"自动选取窗口：[{t0}, {t1}]（含最显著异常事件）")

    n_gt  = int(y_true[t0:t1].sum())
    n_det = int((anomaly_scores[t0:t1] > 0).sum())
    print(f"  窗口内真实异常步数：{n_gt}  检出步数：{n_det}")

    # ── 出图 ──────────────────────────────────────────────────────────────────
    out_dir = args.out or str(eval_dir)
    print("\n生成 paper_figure.png / .pdf ...")
    plot_paper_figure(
        x_true         = x_true,
        raw_smoothed   = raw_smoothed,
        anomaly_scores = anomaly_scores,
        y_true         = y_true,
        t0=t0, t1=t1,
        threshold      = threshold,
        n_channels     = args.n_channels,
        channels_to_show = args.channels[:2],
        output_dir     = out_dir,
    )
    print("\n完成！")
    print(f"  若窗口效果不理想，用 --t0 --t1 手动指定范围")
    print(f"  换通道用 --channels 0 5（例如显示 Ch41, Ch46）")


if __name__ == "__main__":
    main()
