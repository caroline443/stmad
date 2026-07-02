"""
论文补充结果图生成脚本
========================
从已有的 checkpoint 和 .npy 评估结果生成额外的结果图，无需重新训练。

生成内容：
  fig1_score_distribution.pdf  — 正常/异常分数分布对比（直方图+KDE）
  fig2_pr_curve.pdf            — Precision-Recall 曲线（SpCA vs PSTG）
  fig3_efficiency.pdf          — 推理时间 × 参数量 × 性能气泡图
  fig4_band_weights.pdf        — 学习到的频段融合权重

用法：
  python generate_result_figures.py \
    --spca_eval  outputs_spca/eval_20260629_190409 \
    --spca_ckpt  checkpoints_ab_full/best.pt \
    --data_dir   /root/autodl-tmp/data/ESA-Mission1

  如果有 PSTG 的评估结果（用于 PR 曲线对比）：
    --pstg_eval  outputs/latest
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
from scipy.stats import gaussian_kde

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

OUT_DIR = Path("paper_figures")
OUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  图 1：异常分数分布
# ─────────────────────────────────────────────────────────────────────────────

def fig_score_distribution(eval_dir: Path):
    """正常区 vs 异常区的分数分布——越分离越好"""
    raw  = np.load(eval_dir / "raw_smoothed.npy").astype(np.float64)
    y    = np.load(eval_dir / "y_true.npy").astype(np.int32)

    normal  = raw[y == 0]
    anomaly = raw[y == 1]

    # 加载阈值
    try:
        res = json.loads((eval_dir / "evaluation_results.json").read_text())
        thr = res.get("threshold", None)
    except Exception:
        thr = None

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    # 直方图（对数纵轴，更好看）
    bins = np.linspace(raw.min(), min(raw.max(), np.percentile(raw, 99.9)), 80)
    ax.hist(normal,  bins=bins, density=True, alpha=0.55,
            color="#2980b9", label="Normal periods")
    ax.hist(anomaly, bins=bins, density=True, alpha=0.70,
            color="#e74c3c", label="Anomalous periods")

    # KDE 曲线
    if len(normal) > 100:
        kde_n  = gaussian_kde(np.clip(normal,  raw.min(), bins[-1]), bw_method=0.08)
        kde_a  = gaussian_kde(np.clip(anomaly, raw.min(), bins[-1]), bw_method=0.08)
        xs     = np.linspace(bins[0], bins[-1], 300)
        ax.plot(xs, kde_n(xs), color="#1a4f8a", lw=2)
        ax.plot(xs, kde_a(xs), color="#c0392b", lw=2)

    # 阈值线
    if thr:
        ax.axvline(thr, color="#27ae60", ls="--", lw=1.5, label=f"POT threshold ε*={thr:.3f}")

    ax.set_xlabel("Smoothed Reconstruction Error")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution")
    ax.legend(fontsize=8.5)
    ax.set_xlim(bins[0], bins[-1])

    out = OUT_DIR / "fig1_score_distribution.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  图 2：Precision-Recall 曲线
# ─────────────────────────────────────────────────────────────────────────────

def compute_pr(raw_smoothed, y_true, alphas):
    """扫描 POT alpha，计算 Precision-Recall 各点（Standard 1，含单点事件）"""
    from anomaly.detector import _pot_threshold, smooth_residuals
    from utils.metrics import event_wise_metrics, extract_events

    ps, rs = [], []
    for alpha in alphas:
        try:
            eps = _pot_threshold(raw_smoothed.astype(np.float64), q0_pct=0.98, alpha=alpha)
            y_pred = (raw_smoothed >= eps).astype(np.int32)
            m = event_wise_metrics(y_true, y_pred)
            ps.append(m["precision"])
            rs.append(m["recall"])
        except Exception:
            pass
    return np.array(rs), np.array(ps)


def fig_pr_curve(spca_eval: Path, pstg_eval: Path = None):
    """Event-wise Precision-Recall 曲线"""
    import sys
    sys.path.insert(0, ".")

    raw_s = np.load(spca_eval / "raw_smoothed.npy").astype(np.float32)
    y_s   = np.load(spca_eval / "y_true.npy").astype(np.int32)

    alphas = np.logspace(-4, -0.5, 50)

    fig, ax = plt.subplots(figsize=(4.8, 3.5))

    # SpCA
    rs, ps = compute_pr(raw_s, y_s, alphas)
    ax.plot(rs, ps, "o-", color="#e74c3c", ms=3, lw=1.8, label="SpCA (ours)")
    # 标注默认工作点
    ax.scatter([0.818], [0.968], s=80, color="#e74c3c", zorder=5, edgecolors="white", lw=1.2)
    ax.annotate("Default\n(0.968, 0.818)", xy=(0.818,0.968), xytext=(0.68,0.88),
                fontsize=7.5, color="#c0392b",
                arrowprops=dict(arrowstyle="->",color="#c0392b",lw=1))

    # PSTG（如果有）
    if pstg_eval and (pstg_eval / "raw_smoothed.npy").exists():
        raw_p = np.load(pstg_eval / "raw_smoothed.npy").astype(np.float32)
        y_p   = np.load(pstg_eval / "y_true.npy").astype(np.int32) \
                if (pstg_eval / "y_true.npy").exists() else y_s
        rp, pp = compute_pr(raw_p, y_p, alphas)
        ax.plot(rp, pp, "s--", color="#2980b9", ms=3, lw=1.5, label="PSTG")
        ax.scatter([0.862], [0.932], s=70, color="#2980b9", zorder=5, edgecolors="white", lw=1.2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Event-wise Precision-Recall Curve")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5)
    # F0.5 等值线
    for f in [0.6, 0.7, 0.8, 0.9]:
        ps_line = np.linspace(0.01, 1, 200)
        rs_line = f * ps_line / (1.25 * ps_line - 0.25 * f)
        mask = (rs_line > 0) & (rs_line < 1)
        ax.plot(rs_line[mask], ps_line[mask], ":", color="gray", lw=0.8, alpha=0.6)
        ax.text(rs_line[mask][-1]+0.01, ps_line[mask][-1],
                f"F₀.₅={f}", fontsize=6.5, color="gray", va="center")

    out = OUT_DIR / "fig2_pr_curve.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  图 3：推理效率气泡图
# ─────────────────────────────────────────────────────────────────────────────

def fig_efficiency():
    """参数量 × 推理时间 × Event F0.5（Standard 1）气泡图"""
    # 数据：(方法名, 参数量M, 推理时间ms, Event F0.5, 是否是我们的方法)
    methods = [
        ("DLinear",      0.02,  2.1, 0.394, False),
        ("TSMixer",      1.40,  4.5, 0.798, False),
        ("FreTS",        1.10,  3.8, 0.764, False),
        ("iTransformer", 8.50,  8.2, 0.834, False),
        ("PatchTST",     5.20,  6.4, 0.894, False),
        ("WPMixer",      2.80,  5.1, 0.806, False),
        ("PSTG",         2.33, 12.5, 0.917, False),  # placeholder，待补充实测
        ("SpCA (ours)",  2.37,  None, 0.934, True),  # 待补充实测时间
    ]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for name, params, time_ms, f05, ours in methods:
        if time_ms is None:
            continue
        size = (f05 - 0.3) * 800
        color = "#e74c3c" if ours else "#2980b9"
        alpha = 0.9 if ours else 0.55
        ax.scatter(time_ms, params, s=size, c=color, alpha=alpha,
                   edgecolors="white" if ours else "none", linewidths=1.5, zorder=3)
        offset = (2, 0.08) if not ours else (-1.5, -0.25)
        ax.annotate(name, (time_ms, params),
                    xytext=(time_ms+offset[0], params+offset[1]),
                    fontsize=7.5 if not ours else 8.5,
                    fontweight="bold" if ours else "normal",
                    color=color)

    ax.set_xlabel("Inference Time (ms/batch, batch=70)")
    ax.set_ylabel("Parameters (M)")
    ax.set_title("Efficiency vs. Performance\n(bubble size = Event F₀.₅)")
    # 图例气泡
    for f, label in [(0.4,"F₀.₅=0.4"), (0.7,"0.7"), (0.9,"0.9")]:
        ax.scatter([], [], s=(f-0.3)*800, c="gray", alpha=0.5, label=label)
    ax.legend(title="Event F₀.₅", fontsize=8, title_fontsize=8.5,
              loc="upper left", scatterpoints=1)

    out = OUT_DIR / "fig3_efficiency.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  图 4：学习到的频段融合权重
# ─────────────────────────────────────────────────────────────────────────────

def fig_band_weights(ckpt_path: str):
    """从 checkpoint 提取 SpectralFusion 的 softmax 权重"""
    import torch
    import sys; sys.path.insert(0, ".")
    from models.spca import SpCA
    from config_spca import ConfigSpCA

    cfg  = ConfigSpCA()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_c = ckpt.get("config", {})
    model = SpCA(
        n_channels=cfg_c.get("n_channels", cfg.NUM_CHANNELS),
        d_model   =cfg_c.get("d_model",    cfg.D_MODEL),
        n_bands   =cfg_c.get("n_bands",    cfg.N_BANDS),
        band_splits=cfg_c.get("band_splits",cfg.BAND_SPLITS),
        n_patches =cfg_c.get("n_patches",  0),
        use_spectral    =cfg_c.get("use_spectral",     True),
        use_channel_attn=cfg_c.get("use_channel_attn", True),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    import torch.nn.functional as F
    w = F.softmax(model.fusion.weights.detach(), dim=0).numpy()
    print(f"  频段权重: 低频={w[0]:.3f}  中频={w[1]:.3f}  高频={w[2]:.3f}")

    labels  = ["Low-freq\n(0–10% Nyq)", "Mid-freq\n(10–40% Nyq)", "High-freq\n(40–100% Nyq)"]
    colors  = ["#e74c3c", "#2980b9", "#27ae60"]
    descs   = ["Long-period\ntrends", "Operational\ncycles", "Transient\nresponses"]
    phys    = ["Orbital &\nthermal drift", "Charge/discharge\nrhythm", "Measurement\nnoise & spikes"]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0),
                              gridspec_kw={"width_ratios": [1, 1.4]})

    # 左：条形图
    ax = axes[0]
    bars = ax.bar(range(3), w, color=colors, width=0.5, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Softmax Weight α")
    ax.set_ylim(0, max(w)*1.35)
    ax.set_title("Learned Band Weights")
    for i, (bar, val) in enumerate(zip(bars, w)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 右：环形图
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        w, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for t in texts: t.set_fontsize(7.5)
    for at in autotexts: at.set_fontsize(8.5); at.set_fontweight("bold")
    ax2.set_title("Weight Distribution", fontsize=9)

    out = OUT_DIR / "fig4_band_weights.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spca_eval",  type=str, required=True,
                   help="SpCA 评估结果目录（含 raw_smoothed.npy, y_true.npy）")
    p.add_argument("--spca_ckpt",  type=str, required=True,
                   help="SpCA checkpoint 路径（用于提取频段权重）")
    p.add_argument("--pstg_eval",  type=str, default=None,
                   help="PSTG 评估结果目录（可选，用于 PR 曲线对比）")
    p.add_argument("--skip_pr",    action="store_true", help="跳过 PR 曲线（较耗时）")
    return p.parse_args()


def main():
    args = parse_args()
    spca_eval = Path(args.spca_eval)
    pstg_eval = Path(args.pstg_eval) if args.pstg_eval else None

    print("\n=== 生成论文附加结果图 ===")
    print(f"输出目录：{OUT_DIR.absolute()}\n")

    print("图1：异常分数分布...")
    fig_score_distribution(spca_eval)

    if not args.skip_pr:
        print("图2：Precision-Recall 曲线（可能需要几分钟）...")
        fig_pr_curve(spca_eval, pstg_eval)
    else:
        print("图2：已跳过（--skip_pr）")

    print("图3：推理效率气泡图...")
    fig_efficiency()

    print("图4：学习到的频段融合权重...")
    fig_band_weights(args.spca_ckpt)

    print(f"\n全部完成！所有图片在：{OUT_DIR.absolute()}")
    print("注意：图3 中 SpCA 和 PSTG 的推理时间需要补充实测值，")
    print("     在图3 的 methods 列表中修改 time_ms 字段即可。")


if __name__ == "__main__":
    main()
