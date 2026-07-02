"""
SpCA 论文配图生成脚本
====================
生成两张论文配图：

  fig_mshtrans.pdf  — 缩放窗口：每通道原始信号+重建+分数（仿 MSHTrans Fig.3）
  fig_timeline.pdf  — 全时序：聚合分数 + 通道热图

用法：
  python generate_paper_figures.py --spca_eval outputs_spca/latest
  python generate_paper_figures.py --spca_eval outputs_spca/latest --zoom_event 5 --context 3000
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
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
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


# ─────────────────────────────────────────────────────────────────
#  辅助
# ─────────────────────────────────────────────────────────────────

def _extract_events(y):
    events, in_e = [], False
    for i, v in enumerate(y):
        if v and not in_e:
            s = i; in_e = True
        elif not v and in_e:
            events.append((s, i - 1)); in_e = False
    if in_e:
        events.append((s, len(y) - 1))
    return events


def _shade(ax, t, mask, color, alpha, label=None):
    in_r = False; first = True
    for i, v in enumerate(mask):
        if v and not in_r:
            s0 = t[i]; in_r = True
        elif not v and in_r:
            ax.axvspan(s0, t[i], color=color, alpha=alpha, lw=0, zorder=2,
                       label=label if first else None)
            in_r = False; first = False
    if in_r:
        ax.axvspan(s0, t[-1], color=color, alpha=alpha, lw=0, zorder=2,
                   label=label if first else None)


def _load_threshold(eval_dir, raw_smooth, y_true):
    try:
        res = json.loads((eval_dir / "evaluation_results.json").read_text())
        v = res.get("threshold")
        if v:
            return float(v)
    except Exception:
        pass
    return float(np.percentile(raw_smooth[y_true == 0], 99.5))


def _maxpool(arr, factor):
    T = (len(arr) // factor) * factor
    if arr.ndim == 1:
        return arr[:T].reshape(-1, factor).max(axis=1)
    return arr[:T].reshape(-1, factor, arr.shape[1]).max(axis=1)


def _smooth(x, w=10):
    k = np.ones(w) / w
    if x.ndim == 1:
        return np.convolve(x, k, mode="same")
    return np.stack([np.convolve(x[:, c], k, mode="same")
                     for c in range(x.shape[1])], axis=1)


def _save(fig, path):
    path = Path(path)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────
#  图 1：缩放窗口（仿 MSHTrans Fig.3，去掉 All-dims 废行）
# ─────────────────────────────────────────────────────────────────

def fig_mshtrans(eval_dir: Path, out_dir: Path = OUT_DIR,
                 zoom_event: int = 0, context: int = 1500,
                 show_channels: list = None):
    """
    布局（每行对应一对面板）：
      Row 0   : 聚合分数 + 阈值（整体检测效果）
      Row 1,3,5: Ch X 原始信号（深蓝）+ 预测重建（橙）
      Row 2,4,6: Ch X 逐通道分数 + 阈值
    """
    sys.path.insert(0, ".")

    y_true     = np.load(eval_dir / "y_true.npy").astype(np.int32)
    raw_smooth = np.load(eval_dir / "raw_smoothed.npy").astype(np.float64)
    threshold  = _load_threshold(eval_dir, raw_smooth, y_true)
    T = len(y_true)

    has_pred = (eval_dir / "x_true.npy").exists()
    has_per  = (eval_dir / "per_channel_residuals.npy").exists()

    if has_pred:
        x_true_all = np.load(eval_dir / "x_true.npy").astype(np.float64)
        x_pred_all = np.load(eval_dir / "x_pred.npy").astype(np.float64)
        C = x_true_all.shape[1]
    else:
        C = 6; x_true_all = x_pred_all = None
        print("  ⚠ 未找到 x_true.npy，信号行将留空。重新运行 evaluate_spca.py 可修复。")

    if has_per:
        per_ch_all = _smooth(
            np.load(eval_dir / "per_channel_residuals.npy").astype(np.float64), w=10)
    else:
        per_ch_all = None

    if show_channels is None:
        show_channels = list(range(min(3, C)))

    # 时间窗口
    events = _extract_events(y_true)
    if not events:
        print("  ⚠ 无异常事件，跳过"); return
    zoom_event = min(zoom_event, len(events) - 1)
    ev_s, ev_e = events[zoom_event]
    win_s = max(0, ev_s - context)
    win_e = min(T, ev_e + context)
    t      = np.arange(win_s, win_e)
    gt     = y_true[win_s:win_e]
    agg_sc = raw_smooth[win_s:win_e]
    y_pred = (agg_sc >= threshold).astype(np.int32)

    N_ch   = len(show_channels)
    # 行高：[聚合分数] + [信号, 分数] × N_ch
    row_h  = [1.0] + [1.2, 0.75] * N_ch
    fig, axes = plt.subplots(len(row_h), 1,
                              figsize=(7.0, sum(row_h) * 0.9),
                              sharex=True,
                              gridspec_kw={"height_ratios": row_h, "hspace": 0.04})

    # ── 聚合分数行 ───────────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(t, agg_sc, 0, where=agg_sc >= 0,
                    color="#fcc5c0", alpha=0.7, lw=0)
    ax.plot(t, agg_sc, color="#d6604d", lw=0.8, label="Anomaly score")
    ax.axhline(threshold, color="#67001f", ls="--", lw=1.3,
               label=f"Threshold  ε*={threshold:.3f}")
    # 检测到的区域（深红填充）
    ax.fill_between(t, agg_sc, threshold,
                    where=agg_sc >= threshold,
                    color="#d6604d", alpha=0.55, lw=0, zorder=3,
                    label="Detected")
    # 真实异常（绿色半透明）
    _shade(ax, t, gt, "#27ae60", 0.22, "Ground truth")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("Score\n(all ch.)", fontsize=8, rotation=0,
                  labelpad=50, va="center")
    ax.legend(loc="upper left" if t[0] > 0 else "upper right",
              fontsize=7, ncol=4, framealpha=0.85, edgecolor="none",
              columnspacing=0.8)

    # ── 各通道：信号行 + 分数行 ─────────────────────────────────
    CH_COLORS = plt.cm.tab10.colors
    for k, c in enumerate(show_channels):
        ax_sig   = axes[1 + k * 2]
        ax_score = axes[2 + k * 2]
        col = CH_COLORS[c % 10]
        ch_name = f"Ch {41 + c}"

        # 信号行
        ax_sig.set_ylabel(ch_name, fontsize=8.5, rotation=0,
                          labelpad=50, va="center")
        if x_true_all is not None:
            sig  = x_true_all[win_s:win_e, c]
            pred = x_pred_all[win_s:win_e, c]
            # 自动 y 范围：clip 到 99.5 分位以避免单点尖峰撑爆轴
            ylo = np.percentile(sig, 0.5); yhi = np.percentile(sig, 99.5)
            pad = (yhi - ylo) * 0.15
            ax_sig.set_ylim(ylo - pad, yhi + pad * 2)
            ax_sig.plot(t, sig,  color="#2c3e50", lw=0.6, label="Original",      zorder=3)
            ax_sig.plot(t, pred, color="#e67e22", lw=0.6, label="Reconstruction", zorder=4,
                        alpha=0.85)
            if k == 0:
                ax_sig.legend(loc="upper right", fontsize=7,
                              framealpha=0.85, edgecolor="none")
        _shade(ax_sig, t, gt, "#27ae60", 0.22)
        ax_sig.set_yticks([])
        ax_sig.spines["left"].set_visible(False)

        # 分数行
        ax_score.set_ylabel(ch_name, fontsize=8.5, rotation=0,
                            labelpad=50, va="center")
        if per_ch_all is not None:
            ch_sc = per_ch_all[win_s:win_e, c]
            # 每通道阈值：训练集正常样本的 99.8 分位
            ch_thr = float(np.percentile(per_ch_all[:, c][y_true == 0], 99.5))
        else:
            ch_sc  = agg_sc
            ch_thr = threshold
        ax_score.fill_between(t, ch_sc, 0,
                              where=ch_sc >= 0, color="#fcc5c0", alpha=0.65, lw=0)
        ax_score.plot(t, ch_sc, color="#d6604d", lw=0.65)
        ax_score.axhline(ch_thr, color="#67001f", ls="--", lw=1.0)
        ax_score.fill_between(t, ch_sc, ch_thr,
                              where=ch_sc >= ch_thr,
                              color="#d6604d", alpha=0.55, lw=0, zorder=3)
        _shade(ax_score, t, gt, "#27ae60", 0.20)
        ax_score.set_yticks([])
        ax_score.spines["left"].set_visible(False)

    axes[-1].set_xlabel("Time Step")

    # 只在顶部标一次事件名
    ev_mid = (ev_s + ev_e) / 2
    axes[0].annotate(
        f"Event {zoom_event + 1}",
        xy=(ev_mid, axes[0].get_ylim()[1]),
        xytext=(ev_mid, axes[0].get_ylim()[1] * 1.02),
        fontsize=8, color="#c0392b", fontweight="bold",
        ha="center", va="bottom",
        annotation_clip=False,
    )

    ch_str = "+".join([f"Ch{41+c}" for c in show_channels])
    fig.suptitle(
        f"SpCA Detection on ESA-AD Mission 1  "
        f"(Event {zoom_event+1}/{len(events)},  ±{context} steps,  {ch_str})",
        fontsize=8.5, y=1.03
    )

    _save(fig, out_dir / "fig_mshtrans.pdf")


# ─────────────────────────────────────────────────────────────────
#  图 2：全时序（聚合分数 + 通道热图）
# ─────────────────────────────────────────────────────────────────

def fig_timeline(eval_dir: Path, out_dir: Path = OUT_DIR,
                 ds_target: int = 6000):
    """
    两层布局：
      上层：聚合分数 + 阈值 + 所有异常事件标注
      下层：热图（通道 × 时间，颜色 = 每通道标准化残差）

    优点：解决了「33个标签挤在一起」和「6个面板一模一样」两个问题。
    """
    y_true     = np.load(eval_dir / "y_true.npy").astype(np.int32)
    raw_smooth = np.load(eval_dir / "raw_smoothed.npy").astype(np.float64)
    threshold  = _load_threshold(eval_dir, raw_smooth, y_true)
    T = len(y_true)

    has_per = (eval_dir / "per_channel_residuals.npy").exists()
    if has_per:
        per_ch = np.load(eval_dir / "per_channel_residuals.npy").astype(np.float64)
        per_ch = _smooth(per_ch, w=20)
        C = per_ch.shape[1]
    else:
        C = 6
        per_ch = None
        print("  ⚠ 未找到 per_channel_residuals.npy，热图将使用 aggregate score 填充。")

    events = _extract_events(y_true)

    # ── 下采样 ────────────────────────────────────────────────────
    factor = max(1, T // ds_target)
    agg_ds = _maxpool(raw_smooth, factor)
    y_ds   = _maxpool(y_true.astype(np.float64), factor)
    t_ds   = np.arange(len(agg_ds)) * factor

    if has_per:
        per_ds = _maxpool(per_ch, factor)   # [T_ds, C]
        # 每通道归一化到 [0, 1]（相对于该通道正常段的最大值）
        per_norm = np.zeros_like(per_ds)
        for c in range(C):
            normal_max = np.percentile(per_ds[:, c][y_ds == 0], 99.5) + 1e-8
            per_norm[:, c] = np.clip(per_ds[:, c] / normal_max, 0, 1)
    else:
        per_norm = np.tile((agg_ds / (agg_ds.max() + 1e-8))[:, None], (1, C))

    # ── 画布 ─────────────────────────────────────────────────────
    fig, (ax_top, ax_hm) = plt.subplots(
        2, 1, figsize=(8.0, 4.2),
        gridspec_kw={"height_ratios": [1.4, 1.0], "hspace": 0.08},
        sharex=True
    )

    # ── 上层：聚合分数 ────────────────────────────────────────────
    ax_top.fill_between(t_ds, agg_ds, 0,
                        where=agg_ds >= 0, color="#c6dbef", alpha=0.75, lw=0)
    ax_top.plot(t_ds, agg_ds, color="#2166ac", lw=0.7, label="Anomaly score", zorder=3)
    ax_top.axhline(threshold, color="#d6604d", ls="--", lw=1.4, zorder=4,
                   label=f"Threshold ε*={threshold:.3f}")
    ax_top.fill_between(t_ds, agg_ds, threshold,
                        where=agg_ds >= threshold,
                        color="#d6604d", alpha=0.6, lw=0, zorder=3,
                        label="Detected")
    _shade(ax_top, t_ds, y_ds, "#27ae60", 0.20, "Ground truth")
    ax_top.set_ylabel("Score", fontsize=8.5)
    ax_top.legend(loc="upper right", fontsize=7.5, ncol=4,
                  framealpha=0.9, edgecolor="none", columnspacing=0.8)

    # 事件标注：只标在顶部，用竖虚线 + 文字，相邻太近的合并标注
    ymax = ax_top.get_ylim()[1] if ax_top.get_ylim()[1] > 0 else agg_ds.max() * 1.2
    ev_positions = [(ev_s + ev_e) / 2 for ev_s, ev_e in events]
    _place_event_labels(ax_top, ev_positions, t_ds, ymax, factor)

    # ── 下层：通道热图 ────────────────────────────────────────────
    cmap = LinearSegmentedColormap.from_list(
        "anomaly", ["#f7fbff", "#6baed6", "#08519c", "#67000d"])
    im = ax_hm.imshow(
        per_norm.T,                        # shape: [C, T_ds]
        aspect="auto",
        cmap=cmap,
        vmin=0, vmax=1,
        extent=[t_ds[0], t_ds[-1], C - 0.5, -0.5],
        interpolation="nearest",
        rasterized=True
    )
    # 真实异常事件：红色竖线
    for ev_s, ev_e in events:
        ax_hm.axvline(ev_s, color="#d6604d", lw=0.6, alpha=0.7)
        ax_hm.axvline(ev_e, color="#d6604d", lw=0.6, alpha=0.7)
        ax_hm.axvspan(ev_s, max(ev_e, ev_s + factor),
                      color="#d6604d", alpha=0.22, lw=0)
    ax_hm.set_yticks(range(C))
    ax_hm.set_yticklabels([f"Ch {41+c}" for c in range(C)], fontsize=8)
    ax_hm.set_xlabel("Time Step")
    ax_hm.set_ylabel("Channel", fontsize=8.5)

    # 色标
    cbar = fig.colorbar(im, ax=ax_hm, pad=0.01, fraction=0.015,
                        label="Normalized residual")
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"SpCA Anomaly Detection — ESA-AD Mission 1  "
        f"(T={T:,}  |  {len(events)} events)",
        fontsize=9, y=1.01
    )

    _save(fig, out_dir / "fig_timeline.pdf")


def _place_event_labels(ax, positions, t_ds, ymax, factor):
    """
    在 ax 上标注事件：竖虚线 + 文字。
    相邻距离 < min_gap（像素对应的数据单位）的合并为一个 tick，不打文字。
    """
    if not positions:
        return
    T_range = t_ds[-1] - t_ds[0]
    min_gap = T_range * 0.025   # 2.5% 的时间轴宽度

    # 将太近的事件合并
    groups = []
    cur_group = [positions[0]]
    for p in positions[1:]:
        if p - cur_group[-1] < min_gap:
            cur_group.append(p)
        else:
            groups.append(cur_group)
            cur_group = [p]
    groups.append(cur_group)

    for gi, grp in enumerate(groups):
        mid = np.mean(grp)
        # 只标所有事件中第一个事件的序号
        first_idx = positions.index(grp[0]) + 1
        if len(grp) == 1:
            label = f"A{first_idx}"
        else:
            last_idx = positions.index(grp[-1]) + 1
            label = f"A{first_idx}–{last_idx}"

        ax.axvline(mid, color="#d6604d", lw=0.7, ls=":", alpha=0.7, zorder=2)
        ax.text(mid, ymax * 0.98, label,
                ha="center", va="top", fontsize=6, color="#c0392b",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))


# ─────────────────────────────────────────────────────────────────
#  图 3：分数分布直方图（仿 MTGFlow Fig.5/7）
# ─────────────────────────────────────────────────────────────────

def fig_score_dist(eval_dirs: dict, out_dir: Path = OUT_DIR):
    """
    正常段 vs 异常段的分数分布对比直方图（仿 MTGFlow Fig.5/7）。

    eval_dirs: {"SpCA": Path(...), "PSTG": Path(...)}  —— 只传一个也可以
    布局：每个方法一列，列数 = len(eval_dirs)

    好的检测器：两个分布分离明显（正常低分、异常高分）。
    """
    from scipy.stats import gaussian_kde

    n_methods = len(eval_dirs)
    fig, axes = plt.subplots(1, n_methods,
                              figsize=(2.8 * n_methods, 2.8),
                              sharey=False,
                              gridspec_kw={"wspace": 0.35})
    if n_methods == 1:
        axes = [axes]

    for ax, (method_name, eval_dir) in zip(axes, eval_dirs.items()):
        eval_dir = Path(eval_dir)
        raw   = np.load(eval_dir / "raw_smoothed.npy").astype(np.float64)
        y     = np.load(eval_dir / "y_true.npy").astype(np.int32)
        thr   = _load_threshold(eval_dir, raw, y)

        normal  = raw[y == 0]
        anomaly = raw[y == 1]

        # 裁剪右侧长尾（0.1% 之后不影响主体可视）
        x_max = min(np.percentile(raw, 99.9), thr * 4)
        bins  = np.linspace(0, x_max, 60)

        ax.hist(normal,  bins=bins, density=True, alpha=0.55,
                color="#4393c3", label="Normal", zorder=2)
        ax.hist(anomaly, bins=bins, density=True, alpha=0.70,
                color="#d6604d", label="Anomaly", zorder=3)

        # KDE 曲线
        xs = np.linspace(0, x_max, 400)
        for vals, col, lw in [(normal, "#2166ac", 1.5), (anomaly, "#b2182b", 1.5)]:
            clipped = np.clip(vals, 0, x_max)
            if len(clipped) > 50:
                try:
                    kde = gaussian_kde(clipped, bw_method=0.12)
                    ax.plot(xs, kde(xs), color=col, lw=lw, zorder=4)
                except Exception:
                    pass

        # 阈值线
        ax.axvline(thr, color="#1a1a2e", ls="--", lw=1.3, zorder=5,
                   label=f"Threshold={thr:.3f}")

        ax.set_xlabel("Anomaly Score", fontsize=8.5)
        ax.set_ylabel("Density", fontsize=8.5)
        ax.set_title(method_name, fontsize=9)
        ax.set_xlim(0, x_max)
        ax.legend(fontsize=7.5, framealpha=0.85, edgecolor="none")

    fig.suptitle("Normal vs. Anomaly Score Distribution — ESA-AD Mission 1",
                 fontsize=9, y=1.03)
    _save(fig, out_dir / "fig_score_dist.pdf")


# ─────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spca_eval",  type=str, default=None)
    p.add_argument("--pstg_eval",  type=str, default=None,
                   help="PSTG eval 目录（可选，用于分数分布对比图）")
    p.add_argument("--zoom_event", type=int, default=0)
    p.add_argument("--context",    type=int, default=1500)
    p.add_argument("--channels",   type=int, nargs="+", default=None)
    p.add_argument("--out_dir",    type=str, default="paper_figures")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    sys.path.insert(0, ".")

    # 找 eval 目录
    eval_dir = None
    if args.spca_eval:
        eval_dir = Path(args.spca_eval)
    if eval_dir is None or not eval_dir.exists():
        for cand in [Path("outputs_spca/latest"),
                     *sorted(Path("outputs_spca").glob("eval_*/raw_smoothed.npy"),
                             reverse=True)]:
            p = cand if cand.name == "latest" else cand.parent
            if (p / "raw_smoothed.npy").exists():
                eval_dir = p; break

    if eval_dir is None:
        print("❌ 找不到 eval 目录，请指定 --spca_eval <路径>"); return

    print(f"\n使用 eval 目录：{eval_dir}")
    print(f"输出目录：{out_dir.absolute()}\n")

    print("▶ MSHTrans 风格缩放图 (fig_mshtrans.pdf) ...")
    fig_mshtrans(eval_dir, out_dir,
                 zoom_event=args.zoom_event,
                 context=args.context,
                 show_channels=args.channels)

    print("▶ 全时序聚合+热图 (fig_timeline.pdf) ...")
    fig_timeline(eval_dir, out_dir)

    print("▶ 分数分布直方图 (fig_score_dist.pdf) ...")
    dist_dirs = {"SpCA": eval_dir}
    if args.pstg_eval:
        pstg_path = Path(args.pstg_eval)
        if (pstg_path / "raw_smoothed.npy").exists():
            dist_dirs["PSTG"] = pstg_path
        else:
            print(f"  ⚠ PSTG eval 目录无效，跳过 PSTG 对比列")
    elif (Path("outputs") / "latest" / "raw_smoothed.npy").exists():
        dist_dirs["PSTG"] = Path("outputs") / "latest"
        print("  自动找到 outputs/latest 作为 PSTG eval")
    fig_score_dist(dist_dirs, out_dir)

    print("\n完成。")
    print("─" * 45)
    print("  fig_mshtrans.pdf  — 信号+重建+分数缩放图")
    print("  fig_timeline.pdf  — 全时序聚合+通道热图")
    print("  fig_score_dist.pdf— 正常vs异常分数分布")


if __name__ == "__main__":
    main()
