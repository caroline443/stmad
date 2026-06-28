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
    找最适合论文展示的窗口，条件优先级：
      1. 真正被 MTA 检出的事件（True Positive）
      2. 事件两侧有足够的正常数据（对比明显）
      3. 重建误差峰值高（视觉效果好）
      4. 事件本身不能太长（否则整个窗口都是异常，失去对比）
    """
    T   = len(y_true)
    det = (anomaly_scores > 0)

    # ── 提取所有 GT 事件 ──────────────────────────────────────────────────────
    gt_events = [(s, e) for s, e in extract_events(y_true)
                 if e - s + 1 >= min_dur]

    # ── 优先找 True Positive 事件 ────────────────────────────────────────────
    candidates = []
    for s, e in gt_events:
        dur = e - s + 1
        is_tp = det[s:e+1].any()          # MTA 在这段里检出过

        # 窗口左右各留 half_win 步，计算正常数据比例
        t0 = max(0, s - half_win)
        t1 = min(T, e + half_win)
        pre_anom_rate  = y_true[t0:s].mean()     if s > t0 else 1.0
        post_anom_rate = y_true[e+1:t1].mean()   if t1 > e+1 else 1.0
        has_context    = (pre_anom_rate < 0.3 and post_anom_rate < 0.3)

        # 事件不要太长（超过窗口一半就没对比了）
        not_too_long = dur < half_win

        peak_score = float(raw_smoothed[s:e+1].max())

        priority = (
            int(is_tp) * 1000         # TP 优先
            + int(has_context) * 100  # 有正常上下文次优
            + int(not_too_long) * 10  # 事件不太长
            + peak_score              # 峰值高
        )
        candidates.append((priority, s, e, t0, t1))

    if not candidates:
        peak = int(np.argmax(raw_smoothed))
        return max(0, peak - half_win), min(T, peak + half_win)

    # 取优先级最高的候选
    candidates.sort(reverse=True)
    _, s, e, t0, t1 = candidates[0]

    # 以事件中心对称
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
    n_channels=6, channels_to_show=(0, 2),
    output_dir=None,
    dpi=200,
    show_detection=False,   # 关闭橙色检出区（避免大范围 FP 干扰叙事）
):
    """
    Publication-quality 双面板图：
      上：2个通道原始信号，真实异常区绿色高亮
      下：MTA重建误差 + POT阈值线 + 真实异常区绿色高亮
    """
    t   = np.arange(t0, t1)
    seg = slice(t0, t1)
    gt  = y_true[seg].astype(bool)
    det = (anomaly_scores[seg] > 0).astype(bool)
    scr = raw_smoothed[seg]
    sig = x_true[seg]

    colors_ch = plt.cm.tab10.colors

    # ── 布局 ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 4.5), dpi=dpi)
    gs  = GridSpec(3, 1, figure=fig,
                   height_ratios=[1, 1, 1.2], hspace=0.06)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    # ── 辅助：标注真实异常区 ──────────────────────────────────────────────────
    def shade_gt(ax):
        for s, e in extract_events(gt):
            ax.axvspan(t[s], t[min(e, len(t)-1)],
                       color="#e74c3c", alpha=0.18, lw=0, zorder=2)

    def shade_det(ax):
        for s, e in extract_events(det):
            ax.axvspan(t[s], t[min(e, len(t)-1)],
                       color="#e67e22", alpha=0.15, lw=0, zorder=1)

    ch_labels = [f"Ch {41 + c}" for c in range(n_channels)]

    # ── 上面板：通道信号 ───────────────────────────────────────────────────────
    for row, ch in enumerate(channels_to_show[:2]):
        ax = axes[row]
        if show_detection:
            shade_det(ax)
        shade_gt(ax)
        ax.plot(t, sig[:, ch], color=colors_ch[ch % 10], lw=0.75, zorder=3)
        ax.set_ylabel(ch_labels[ch], fontsize=9, labelpad=2)
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(labelbottom=False, labelsize=8)
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=3, prune="both"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── 下面板：重建误差 + 阈值 ────────────────────────────────────────────────
    ax_s = axes[2]
    if show_detection:
        shade_det(ax_s)
    shade_gt(ax_s)

    ax_s.plot(t, scr, color="#2980b9", lw=0.9, zorder=3)
    ax_s.axhline(threshold, color="#c0392b", ls="--", lw=1.3, zorder=4)

    # y 轴：下界留 0，上界取窗口内最大值的 1.15 倍
    y_max = float(scr.max()) * 1.15
    y_min = 0.0
    ax_s.set_ylim(y_min, y_max)

    ax_s.set_xlim(t[0], t[-1])
    ax_s.set_xlabel("Time Step", fontsize=9)
    ax_s.set_ylabel("Recon. Error", fontsize=9, labelpad=2)
    ax_s.tick_params(labelsize=8)
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    ax_s.yaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(nbins=4, prune="both"))

    # 阈值文字标注
    ax_s.text(t[-1], threshold, f" ε*={threshold:.3f}",
              va="center", ha="left", fontsize=7.5,
              color="#c0392b", clip_on=False)

    # ── 图例 ──────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color="#e74c3c", alpha=0.4, label="Ground-Truth Anomaly"),
        plt.Line2D([0], [0], color="#2980b9", lw=1.5,  label="Reconstruction Error"),
        plt.Line2D([0], [0], color="#c0392b", lw=1.5,
                   ls="--", label=f"POT Threshold"),
    ]
    if show_detection:
        legend_items.insert(1, mpatches.Patch(
            color="#e67e22", alpha=0.4, label="MTA Detection"))

    ax_s.legend(handles=legend_items, fontsize=7.5,
                loc="upper right", framealpha=0.85,
                ncol=2, handlelength=1.5, columnspacing=0.8)

    # ── 标题 ──────────────────────────────────────────────────────────────────
    fig.suptitle(
        "MTA Anomaly Detection on ESA-AD Spacecraft Telemetry",
        fontsize=10, y=0.99,
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
    p.add_argument("--half_win",    type=int, default=500,
                   help="窗口半径（时间步数），默认 500")
    p.add_argument("--show_detection", action="store_true",
                   help="显示橙色MTA检出区（默认关闭，避免大范围FP干扰）")
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
        # 诊断输出
        seg = slice(t0, t1)
        gt_in_win  = y_true[seg].sum()
        det_in_win = (anomaly_scores[seg] > 0).sum()
        tp_in_win  = (y_true[seg].astype(bool) & (anomaly_scores[seg] > 0)).sum()
        print(f"自动选取窗口：[{t0}, {t1}]  共 {t1-t0} 步")
        print(f"  真实异常步数：{gt_in_win}  检出步数：{det_in_win}  "
              f"True Positive：{tp_in_win}")
        if gt_in_win == 0:
            print("  ⚠ 该窗口无真实异常，建议用 --t0 --t1 手动指定")

    n_gt  = int(y_true[t0:t1].sum())
    n_det = int((anomaly_scores[t0:t1] > 0).sum())
    print(f"  窗口内真实异常步数：{n_gt}  检出步数：{n_det}")

    # ── 出图 ──────────────────────────────────────────────────────────────────
    out_dir = args.out or str(eval_dir)
    print("\n生成 paper_figure.png / .pdf ...")
    plot_paper_figure(
        x_true           = x_true,
        raw_smoothed     = raw_smoothed,
        anomaly_scores   = anomaly_scores,
        y_true           = y_true,
        t0=t0, t1=t1,
        threshold        = threshold,
        n_channels       = args.n_channels,
        channels_to_show = args.channels[:2],
        output_dir       = out_dir,
        show_detection   = args.show_detection,
    )
    print("\n完成！")
    print(f"  若窗口效果不理想，用 --t0 --t1 手动指定范围")
    print(f"  换通道用 --channels 0 5（例如显示 Ch41, Ch46）")


if __name__ == "__main__":
    main()
