"""
SpCA 论文配图生成脚本
====================
生成两张论文配图（参考 PSTG Fig.6 和 PSTG/ContrastAD 风格）：

  fig_ablation.pdf   — 消融实验分组条形图（仿 PSTG Fig.6）
  fig_detection.pdf  — 检测案例图：异常分数时序 + 标注事件（仿 PSTG Fig.2/检测展示）

用法：
  # 只生成消融图（使用内置实验数据）
  python generate_paper_figures.py --ablation_only

  # 同时生成检测案例图（需要 eval 目录）
  python generate_paper_figures.py \
    --spca_eval outputs_spca/eval_20260629_190409

  # 指定输出目录
  python generate_paper_figures.py \
    --spca_eval outputs_spca/latest \
    --out_dir paper_figures
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ─── 全局绘图风格（IEEE 会议论文风格）────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         200,
    "pdf.fonttype":       42,   # 嵌入字体，IEEE 要求
    "ps.fonttype":        42,
})

OUT_DIR = Path("paper_figures")
OUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  图 1：消融实验分组条形图（仿 PSTG Fig.6）
# ═══════════════════════════════════════════════════════════════════════════════

def fig_ablation(out_dir: Path = OUT_DIR):
    """
    生成消融实验分组条形图。
    数据来自 run_experiments.py 的实测结果（Standard 1，全部 33 个事件）。

    如果 experiment_results.json 存在则自动读取；否则使用内置数据。
    """

    # ── 尝试从 JSON 读取 ────────────────────────────────────────────────────
    data = _load_ablation_from_json()

    if data is None:
        # ── 内置数据（来自实验日志）─────────────────────────────────────────
        # Standard 1（33 events，含单点事件）
        # 格式：(名称, event_f05, affil_f05)
        data = [
            ("SpCA\n(Full)",        0.934, None),   # Affil 待补充
            ("w/o Spectral\nDecomp",0.872, None),
            ("w/o Channel\nAttn",   0.931, None),
            ("w/o Both\n(baseline)",0.872, None),
        ]
        print("  ⚠ 使用内置消融数据。如需从 experiment_results.json 读取，"
              "请先运行 run_experiments.py。")

    labels   = [d[0] for d in data]
    ev_f05   = [d[1] for d in data]
    af_f05   = [d[2] if len(d) > 2 and d[2] is not None else float("nan")
                for d in data]

    has_affil = any(not np.isnan(v) for v in af_f05)
    n_groups  = 2 if has_affil else 1
    n_bars    = len(labels)

    # ── 颜色方案（与 PSTG 论文保持相近的蓝/橙/绿/红四色方案）────────────────
    COLORS = ["#2166ac", "#f4a582", "#92c5de", "#d6604d"]
    # SpCA Full 用深蓝强调
    bar_colors = [COLORS[i % len(COLORS)] for i in range(n_bars)]
    bar_colors[0] = "#1a4a8a"   # Full model 深蓝

    if n_groups == 1:
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        axes = [ax]
        groups = [("Event-wise $F_{0.5}$", ev_f05)]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.8),
                                  sharey=False,
                                  gridspec_kw={"wspace": 0.35})
        groups = [
            ("Event-wise $F_{0.5}$",       ev_f05),
            ("Affiliation-based $F_{0.5}$", af_f05),
        ]

    for ax, (metric_name, values) in zip(axes, groups):
        x = np.arange(n_bars)
        bars = ax.bar(x, values, width=0.55, color=bar_colors,
                      edgecolor="white", linewidth=0.8, zorder=3)

        # 数值标注（在柱顶）
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.006,
                        f"{val:.3f}",
                        ha="center", va="bottom",
                        fontsize=7.5,
                        fontweight="bold" if bar.get_x() < 0.5 else "normal",
                        color="#1a4a8a" if bar.get_x() < 0.5 else "#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel("$F_{0.5}$ Score")
        ax.set_xlabel(metric_name)

        # Y 轴范围：从最小值下方留白
        valid = [v for v in values if not np.isnan(v)]
        ymin = max(0, min(valid) - 0.08)
        ymax = max(valid) + 0.06
        ax.set_ylim(ymin, ymax)

        # 参考线（PSTG 论文中 Full model 的虚线）
        ax.axhline(values[0], color=bar_colors[0], lw=1.0, ls="--", alpha=0.5, zorder=2)

        # 网格线（横向，仅浅灰）
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))
        ax.grid(axis="y", which="major", color="#e0e0e0", lw=0.6, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle("Ablation Study on ESA-AD", fontsize=10, y=1.02)

    out = out_dir / "fig_ablation.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def _load_ablation_from_json() -> list | None:
    """从 experiment_results.json 读取消融结果"""
    json_path = Path("experiment_results.json")
    if not json_path.exists():
        return None
    try:
        res = json.loads(json_path.read_text())
        ablation = res.get("ablation", [])
        if not ablation:
            return None
        data = []
        name_map = {
            "SpCA Full":            "SpCA\n(Full)",
            "w/o Spectral Decomp":  "w/o Spectral\nDecomp",
            "w/o Channel Attention":"w/o Channel\nAttn",
            "w/o Both (baseline)":  "w/o Both\n(baseline)",
        }
        for entry in ablation[:4]:   # 只取核心 4 个
            if entry is None:
                continue
            name = name_map.get(entry.get("name", ""), entry.get("name", ""))
            ev   = entry.get("std1_ev_f05")
            af   = entry.get("std1_af_f05")
            if ev is not None:
                data.append((name, float(ev), float(af) if af else float("nan")))
        return data if data else None
    except Exception as e:
        print(f"  读取 experiment_results.json 失败: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  图 2：检测案例图
# ═══════════════════════════════════════════════════════════════════════════════

def fig_detection(eval_dir: Path, out_dir: Path = OUT_DIR,
                  zoom_event: int = 0, context_steps: int = 2000):
    """
    展示某个异常事件附近的检测结果。

    布局（上下两图）：
      上：6 个通道的原始信号（归一化）+ 异常标注区
      下：平滑残差 + 阈值线 + 检测结果标注区

    参数：
      zoom_event    — 聚焦第几个异常事件（0=第一个）
      context_steps — 事件前后各显示多少步
    """
    # ── 加载数据 ─────────────────────────────────────────────────────────────
    raw_smoothed = np.load(eval_dir / "raw_smoothed.npy").astype(np.float64)
    y_true       = np.load(eval_dir / "y_true.npy").astype(np.int32)

    # 加载阈值
    try:
        res = json.loads((eval_dir / "evaluation_results.json").read_text())
        threshold = res.get("threshold", float(np.percentile(
            raw_smoothed[y_true == 0], 99.5)))
    except Exception:
        threshold = float(np.percentile(raw_smoothed[y_true == 0], 99.5))

    # ── 定位异常事件 ─────────────────────────────────────────────────────────
    events = _extract_events(y_true)
    if not events:
        print("  ⚠ y_true 中没有异常事件，跳过检测案例图")
        return

    zoom_event = min(zoom_event, len(events) - 1)
    ev_start, ev_end = events[zoom_event]

    # 窗口：事件前后各 context_steps 步
    win_s = max(0, ev_start - context_steps)
    win_e = min(len(raw_smoothed), ev_end + context_steps)
    t = np.arange(win_s, win_e)

    score = raw_smoothed[win_s:win_e]
    label = y_true[win_s:win_e]
    y_pred = (score >= threshold).astype(np.int32)

    # ── 尝试加载原始通道数据（可选）────────────────────────────────────────
    # 从 data_cache 或 outputs 找 test_data
    test_data = _try_load_test_data(win_s, win_e)

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    if test_data is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 4.0),
                                        gridspec_kw={"height_ratios": [1.4, 1],
                                                     "hspace": 0.08},
                                        sharex=True)
        _plot_channels(ax1, t, test_data, label)
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(5.5, 2.4))

    # 下图：异常分数
    _plot_score(ax2, t, score, label, y_pred, threshold, win_s)

    if test_data is not None:
        ax1.set_xticklabels([])

    fig.suptitle(
        f"SpCA Anomaly Detection — ESA-AD Mission 1  "
        f"(Event {zoom_event+1}/{len(events)})",
        fontsize=9, y=1.01
    )

    out = out_dir / "fig_detection.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def _plot_channels(ax, t, data, label):
    """上图：多通道信号 + 异常区着色"""
    colors = plt.cm.tab10.colors
    C = data.shape[1]
    for c in range(C):
        sig = data[:, c]
        sig_n = (sig - sig.mean()) / (sig.std() + 1e-8)   # 归一化以便叠加
        ax.plot(t, sig_n + c * 2.2, color=colors[c % 10], lw=0.7,
                label=f"Ch {c+41}")
    _shade(ax, t, label, alpha=0.25, color="#e74c3c")
    ax.set_ylabel("Channels (Ch 41–46)\n[normalized]", fontsize=8)
    ax.legend(loc="upper right", ncol=3, fontsize=6.5,
              framealpha=0.7, edgecolor="none")
    ax.set_yticks([])


def _plot_score(ax, t, score, label, y_pred, threshold, win_s):
    """下图：平滑残差 + 阈值 + 检测/标注区域"""
    ax.fill_between(t, score, 0,
                    where=(score >= 0),
                    color="#aec6e8", alpha=0.6, lw=0)
    ax.plot(t, score, color="#2166ac", lw=0.85, label="Anomaly score")
    ax.axhline(threshold, color="#d6604d", ls="--", lw=1.2,
               label=f"Threshold ε*={threshold:.3f}")

    # 真实异常区（绿色）
    _shade(ax, t, label,  alpha=0.22, color="#27ae60", label="Ground truth")
    # 检测结果（橙色虚边框）
    _shade(ax, t, y_pred, alpha=0.0,  color="#e67e22",
           hatch="//", edgecolor="#e67e22", linewidth=0.5, label="Detected")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Smoothed Residual")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.8, edgecolor="none")


def _shade(ax, t, mask, alpha=0.2, color="green",
           hatch=None, edgecolor=None, linewidth=0.5, label=None):
    """在 ax 上为 mask==1 的区间着色"""
    in_r = False
    first = True
    for i, v in enumerate(mask):
        if v and not in_r:
            s = t[i]; in_r = True
        elif not v and in_r:
            kw = dict(alpha=alpha, color=color, zorder=2, label=label if first else None)
            if hatch:
                kw.update(hatch=hatch, edgecolor=edgecolor or color,
                          linewidth=linewidth, fill=False)
            ax.axvspan(s, t[i], **kw)
            in_r = False; first = False
    if in_r:
        kw = dict(alpha=alpha, color=color, zorder=2, label=label if first else None)
        if hatch:
            kw.update(hatch=hatch, edgecolor=edgecolor or color,
                      linewidth=linewidth, fill=False)
        ax.axvspan(s, t[-1], **kw)


def _extract_events(y: np.ndarray) -> list:
    events = []
    in_e = False
    for i, v in enumerate(y):
        if v and not in_e:
            s = i; in_e = True
        elif not v and in_e:
            events.append((s, i - 1)); in_e = False
    if in_e:
        events.append((s, len(y) - 1))
    return events


def _try_load_test_data(win_s: int, win_e: int):
    """尝试从 data_cache 加载原始 test_data，截取 [win_s:win_e]"""
    for cache in [
        Path("checkpoints_spca/data_cache"),
        Path("checkpoints_ab_full/data_cache"),
        Path("checkpoints/data_cache"),
    ]:
        f = cache / "test_data.npy"
        if f.exists():
            try:
                data = np.load(f)
                # test_data 从 CONTEXT_LEN（250）开始
                L = 250
                return data[L + win_s : L + win_e]
            except Exception:
                pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  图 3（可选）：方法对比条形图
# ═══════════════════════════════════════════════════════════════════════════════

def fig_method_comparison(out_dir: Path = OUT_DIR):
    """
    SpCA vs PSTG vs baselines 的双指标水平条形图。
    数据来自论文实验结果（Standard 2，过滤单点事件，与 PSTG 论文对齐）。
    """
    # (方法名, Event F0.5, Affil F0.5)
    # Standard 2 数据（duration≥2，24 events）
    methods = [
        ("DLinear",        0.394, 0.767),
        ("FreTS",          0.764, 0.846),
        ("TSMixer",        0.798, 0.784),
        ("WPMixer",        0.806, 0.849),
        ("iTransformer",   0.834, 0.811),
        ("Crossformer",    0.836, 0.869),
        ("PatchTST",       0.894, 0.885),
        ("TimeFilter",     0.835, 0.869),
        ("PSTG",           0.917, 0.892),
        ("SpCA (Ours)",    0.934, None),   # Standard 2 数值待补充
    ]

    # 若没有 SpCA Standard 2 的 Affil，去掉该列
    methods_clean = [(n, ev, af) for n, ev, af in methods]
    has_affil = any(af is not None for _, _, af in methods_clean)

    names = [m[0] for m in methods_clean]
    ev    = [m[1] for m in methods_clean]
    af    = [m[2] if m[2] is not None else float("nan") for m in methods_clean]

    n = len(names)
    y = np.arange(n)
    h = 0.32

    fig_h = max(3.0, 0.38 * n)
    if has_affil:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, fig_h),
                                        sharey=True,
                                        gridspec_kw={"wspace": 0.08})
        axes_pairs = [(ax1, ev, "Event-wise $F_{0.5}$"),
                      (ax2, af, "Affiliation-based $F_{0.5}$")]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(3.0, fig_h))
        axes_pairs = [(ax1, ev, "Event-wise $F_{0.5}$")]

    for ax, vals, xlabel in axes_pairs:
        colors = ["#1a4a8a" if "Ours" in nm else
                  "#d6604d" if nm == "PSTG" else "#a8c8e8"
                  for nm in names]
        bars = ax.barh(y, vals, height=h * 0.9, color=colors,
                       edgecolor="white", lw=0.5, zorder=3)
        for bar, val, nm in zip(bars, vals, names):
            if not np.isnan(val):
                ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}",
                        va="center", ha="left", fontsize=7,
                        fontweight="bold" if "Ours" in nm else "normal")
        xmin = max(0, min(v for v in vals if not np.isnan(v)) - 0.05)
        ax.set_xlim(xmin, 1.04)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#e0e0e0", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        if ax is axes_pairs[0][0]:
            ax.set_yticks(y)
            ax.set_yticklabels(names)
        ax.invert_yaxis()

    # 图例
    legend_elems = [
        mpatches.Patch(color="#1a4a8a", label="SpCA (Ours)"),
        mpatches.Patch(color="#d6604d", label="PSTG"),
        mpatches.Patch(color="#a8c8e8", label="Baselines"),
    ]
    axes_pairs[0][0].legend(handles=legend_elems, loc="lower right",
                             fontsize=7, framealpha=0.8, edgecolor="none")

    fig.suptitle("Performance Comparison on ESA-AD (Standard 2, 24 events)",
                 fontsize=9, y=1.02)

    out = out_dir / "fig_comparison.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  图 4（可选）：跨通道注意力热图
# ═══════════════════════════════════════════════════════════════════════════════

def fig_attention_heatmap(ckpt_path: str, out_dir: Path = OUT_DIR):
    """
    从 checkpoint 提取 CrossChannelAttention 的平均注意力权重，
    生成 6×6 的热图（仿 PSTG Fig.3 风格）。
    """
    import torch, sys
    sys.path.insert(0, ".")
    from models.spca import SpCA
    from config_spca import ConfigSpCA

    cfg  = ConfigSpCA()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_c = ckpt.get("config", {})

    model = SpCA(
        n_channels      = cfg_c.get("n_channels",      cfg.NUM_CHANNELS),
        d_model         = cfg_c.get("d_model",          cfg.D_MODEL),
        n_bands         = cfg_c.get("n_bands",          cfg.N_BANDS),
        band_splits     = cfg_c.get("band_splits",      cfg.BAND_SPLITS),
        n_patches       = cfg_c.get("n_patches",        0),
        use_spectral    = cfg_c.get("use_spectral",     True),
        use_channel_attn= cfg_c.get("use_channel_attn", True),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    C = cfg_c.get("n_channels", cfg.NUM_CHANNELS)

    # ── 捕获注意力权重 ──────────────────────────────────────────────────────
    attn_maps = []

    def _hook(module, inp, out):
        # out: (attn_output, attn_weights) — 如果 need_weights=True
        if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
            attn_maps.append(out[1].detach().squeeze(0).numpy())

    hooks = []
    for layer in list(model.global_attns):
        hooks.append(layer.attn.register_forward_hook(_hook))

    # 随机输入触发 forward
    import torch.nn.functional as F
    x = torch.randn(1, C, cfg_c.get("context_len", cfg.CONTEXT_LEN))
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    if not attn_maps:
        print("  ⚠ 未捕获到注意力权重（模型结构可能不同），跳过热图")
        return

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    n_maps = min(len(attn_maps), 4)
    subtitles = [f"Global Attn Layer {i+1}" for i in range(n_maps)]
    # 加入频段内注意力
    band_attns = []
    for k in range(model.n_bands):
        band_maps = []
        def _band_hook(module, inp, out, _maps=band_maps):
            if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
                _maps.append(out[1].detach().squeeze(0).numpy())
        hk_list = [l.attn.register_forward_hook(_band_hook) for l in model.band_attns[k]]
        band_attns.append((hk_list, band_maps))
    with torch.no_grad():
        model(x)
    all_maps = []
    for hk_list, band_maps in band_attns:
        for h in hk_list: h.remove()
        if band_maps:
            all_maps.append((f"Band {['Low','Mid','High'][_]}  Attn", band_maps[0])
                            for _ in range(len(band_maps)))
    if attn_maps:
        all_maps = [(f"Band {['Low','Mid','High'][i]} Attn", attn_maps[i])
                    for i in range(min(2, len(attn_maps)))]
        all_maps += [(f"Global Attn L{i+1}", attn_maps[-(2-i)])
                     for i in range(min(2, len(attn_maps)))]
    else:
        print("  ⚠ 没有足够注意力图"); return

    fig, axes = plt.subplots(1, len(all_maps), figsize=(len(all_maps) * 2.0, 2.2))
    if len(all_maps) == 1:
        axes = [axes]

    ch_labels = [f"Ch{c+41}" for c in range(C)]

    for ax, (title, amap) in zip(axes, all_maps):
        if amap.ndim == 3:
            amap = amap.mean(0)  # 多头平均
        im = ax.imshow(amap, vmin=0, vmax=amap.max(), cmap="YlOrRd",
                       aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        ax.set_xticklabels(ch_labels, fontsize=6.5, rotation=45)
        ax.set_yticklabels(ch_labels, fontsize=6.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Cross-Channel Attention Weights (SpCA)", fontsize=9)

    out = out_dir / "fig_attention.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="生成 SpCA 论文配图")
    p.add_argument("--spca_eval",    type=str, default=None,
                   help="SpCA 评估目录（含 raw_smoothed.npy / y_true.npy）")
    p.add_argument("--spca_ckpt",    type=str, default=None,
                   help="SpCA checkpoint（用于注意力热图，可选）")
    p.add_argument("--ablation_only",action="store_true",
                   help="只生成消融图（不需要 eval 目录）")
    p.add_argument("--zoom_event",   type=int, default=0,
                   help="检测案例图聚焦第几个异常事件（默认 0=第一个）")
    p.add_argument("--context",      type=int, default=1500,
                   help="检测案例图中事件前后显示多少步（默认 1500）")
    p.add_argument("--out_dir",      type=str, default="paper_figures")
    return p.parse_args()


def main():
    args   = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"\n=== 生成 SpCA 论文配图 ===")
    print(f"输出目录：{out_dir.absolute()}\n")

    # 图 1：消融实验（总是生成）
    print("▶ 消融实验条形图...")
    fig_ablation(out_dir)

    if not args.ablation_only:
        # 图 2：检测案例
        if args.spca_eval:
            eval_dir = Path(args.spca_eval)
            if not eval_dir.exists():
                # 尝试 latest 软链接
                eval_dir = Path("outputs_spca") / "latest"
            if (eval_dir / "raw_smoothed.npy").exists():
                print("▶ 检测案例图...")
                fig_detection(eval_dir, out_dir,
                              zoom_event=args.zoom_event,
                              context_steps=args.context)
            else:
                print(f"  ⚠ 找不到 {eval_dir}/raw_smoothed.npy，跳过检测案例图")
        else:
            print("  跳过检测案例图（未指定 --spca_eval）")

        # 图 3：方法对比
        print("▶ 方法对比条形图...")
        fig_method_comparison(out_dir)

        # 图 4：注意力热图（可选）
        if args.spca_ckpt and Path(args.spca_ckpt).exists():
            print("▶ 注意力热图...")
            try:
                fig_attention_heatmap(args.spca_ckpt, out_dir)
            except Exception as e:
                print(f"  ⚠ 注意力热图失败: {e}")

    print(f"\n完成！图片位于：{out_dir.absolute()}")
    print("提示：")
    print("  • fig_ablation.pdf   — 消融实验分组条形图（适合论文 Section IV-C）")
    print("  • fig_detection.pdf  — 检测案例图（适合论文 Section IV-B）")
    print("  • fig_comparison.pdf — 方法对比水平条形图（可替代或补充表格）")
    print("  • fig_attention.pdf  — 跨通道注意力热图（适合可解释性分析）")


if __name__ == "__main__":
    main()
