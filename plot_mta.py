"""
从已有的评估结果目录直接生成（或重新生成）图片，无需重跑推理。

用法：
  # 指定 eval 目录（可以是任意历史 run）
  python plot_mta.py --eval_dir outputs_mta/eval_20260628_131518

  # 默认读 latest/
  python plot_mta.py

输出（写入同一 eval_dir）：
  anomaly_scores.png        — 平滑重建误差 + 阈值 + 真实异常区域
  channel_reconstruction.png — 各通道原始 vs 重建（需要 x_recon.npy）
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  图 1：异常分数（anomaly_scores.png）
# ─────────────────────────────────────────────────────────────────────────────

def plot_scores(raw_smoothed, anomaly_scores, y_true, threshold, eval_dir,
                max_plot_len=5000):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    T  = min(len(raw_smoothed), max_plot_len)
    t  = np.arange(T)
    gt = y_true[:T].astype(bool)

    def shade(ax, mask):
        in_r = False
        for i in range(len(mask)):
            if mask[i] and not in_r:  s = i; in_r = True
            elif not mask[i] and in_r:
                ax.axvspan(s, i, alpha=0.2, color="green"); in_r = False
        if in_r: ax.axvspan(s, len(mask), alpha=0.2, color="green")

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(t, raw_smoothed[:T], color="steelblue", lw=0.8)
    ax.axhline(threshold, color="red", ls="--", lw=1.2)
    shade(ax, gt)
    ax.set_title("MTA — Smoothed Reconstruction Error & Detection Threshold")
    ax.legend(handles=[
        mpatches.Patch(color="steelblue",        label="Smoothed Recon Error"),
        mpatches.Patch(color="red",   alpha=0.8, label=f"Threshold={threshold:.4f}"),
        mpatches.Patch(color="green", alpha=0.3, label="Ground Truth Anomaly"),
    ])
    plt.tight_layout()
    out = eval_dir / "anomaly_scores.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  图 2：通道重建（channel_reconstruction.png）
# ─────────────────────────────────────────────────────────────────────────────

def plot_channels(x_true, x_recon, y_true, n_channels, eval_dir,
                  max_plot_len=5000):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    T  = min(len(x_true), max_plot_len)
    t  = np.arange(T)
    gt = y_true[:T].astype(bool)

    def shade(ax, mask):
        in_r = False
        for i in range(len(mask)):
            if mask[i] and not in_r:  s = i; in_r = True
            elif not mask[i] and in_r:
                ax.axvspan(s, i, alpha=0.2, color="green"); in_r = False
        if in_r: ax.axvspan(s, len(mask), alpha=0.2, color="green")

    fig = plt.figure(figsize=(16, 2.5 * n_channels))
    gs  = GridSpec(n_channels, 1, figure=fig, hspace=0.4)
    colors = plt.cm.tab10.colors

    for c in range(n_channels):
        ax = fig.add_subplot(gs[c])
        ax.plot(t, x_true[:T, c],  color=colors[c % 10], lw=0.7, label="Original")
        ax.plot(t, x_recon[:T, c], color="gray", lw=0.7, ls="--",
                alpha=0.8, label="MTA Reconstruction")
        shade(ax, gt)
        ax.set_ylabel(f"Ch {c+41}"); ax.set_xlim(0, T)
        if c == 0:
            ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Time Step")
    fig.suptitle("MTA: Reconstruction vs. Original (all channels)", fontsize=12)
    plt.tight_layout()
    out = eval_dir / "channel_reconstruction.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="从已有评估结果生成图片")
    p.add_argument("--eval_dir", type=str, default=None,
                   help="评估目录（默认 outputs_mta/latest）")
    p.add_argument("--n_channels", type=int, default=6)
    p.add_argument("--max_len",    type=int, default=5000,
                   help="绘图最大时间步数（默认 5000）")
    return p.parse_args()


def main():
    args = parse_args()

    # 定位 eval 目录
    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
    else:
        eval_dir = Path("outputs_mta/latest")
        if eval_dir.is_symlink():
            eval_dir = eval_dir.resolve()

    if not eval_dir.exists():
        print(f"目录不存在：{eval_dir}")
        return

    print(f"读取评估目录：{eval_dir}")

    # ── 加载必要文件 ─────────────────────────────────────────────────────────
    raw_smoothed   = np.load(eval_dir / "raw_smoothed.npy")
    anomaly_scores = np.load(eval_dir / "anomaly_scores.npy")

    # 从 evaluation_results.json 读阈值和通道数
    results = json.loads((eval_dir / "evaluation_results.json").read_text())
    threshold = results.get("threshold", 0.0)

    # 从 eval_info.json 读配置
    info = json.loads((eval_dir / "eval_info.json").read_text())
    n_channels = args.n_channels

    # y_true：优先从 data_cache 读，没有就用全 0 占位
    # 这里简单地用 anomaly_scores > 0 推算，或者让用户手动传
    # 实际用法：如果有 test_labels，直接 np.load；否则用 anomaly_scores 近似
    # 为通用性，这里提示用户若想要精确真值区域，需要在 evaluate_mta.py 保存 y_true
    T = len(raw_smoothed)
    y_true_path = eval_dir / "y_true.npy"
    if y_true_path.exists():
        y_true = np.load(y_true_path)
    else:
        print("  ⚠ 未找到 y_true.npy，绿色标注区域将为空（不影响分数图）")
        y_true = np.zeros(T, dtype=np.int32)

    print(f"  T={T:,}  threshold={threshold:.4f}")

    # ── 图 1：分数图 ──────────────────────────────────────────────────────────
    print("\n生成 anomaly_scores.png ...")
    plot_scores(raw_smoothed, anomaly_scores, y_true, threshold, eval_dir,
                max_plot_len=args.max_len)

    # ── 图 2：通道重建图 ──────────────────────────────────────────────────────
    x_recon_path = eval_dir / "x_recon.npy"
    x_true_path  = eval_dir / "x_true.npy"

    if x_recon_path.exists() and x_true_path.exists():
        print("生成 channel_reconstruction.png ...")
        x_recon = np.load(x_recon_path)
        x_true  = np.load(x_true_path)
        plot_channels(x_true, x_recon, y_true, n_channels, eval_dir,
                      max_plot_len=args.max_len)
    else:
        print(
            "\n⚠ 未找到 x_recon.npy / x_true.npy（旧版 evaluate_mta.py 未保存）。\n"
            "  channel_reconstruction.png 无法生成。\n"
            "  解决方案：重新运行 evaluate_mta.py 一次（约 13 分钟），"
            "之后所有图片都可用此脚本即时生成。"
        )

    print("\n完成！")


if __name__ == "__main__":
    main()
