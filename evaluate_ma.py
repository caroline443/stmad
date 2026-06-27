"""
PSTG-MA 推理与评估脚本

双信号异常检测：
  Signal 1：预测残差 R_pred = |x_true - x̂|（与 PSTG 相同）
  Signal 2：记忆重构误差 R_mem = MemoryBank 的 mem_error
  最终分数：α·R_pred + (1-α)·R_mem

用法：
  python evaluate_ma.py \\
    --data_dir /root/autodl-tmp/data/ESA-Mission1 \\
    --ckpt checkpoints_ma/best.pt

对比基线（PSTG）：
  python evaluate.py \\
    --data_dir /root/autodl-tmp/data/ESA-Mission1 \\
    --ckpt checkpoints/best.pt
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config_ma import ConfigMA
from data.dataset import build_datasets
from models.pstg_ma import PSTG_MA
from anomaly.detector import smooth_residuals, detect_anomalies
from utils.metrics import find_best_threshold  # noqa: F401 (备用)
from evaluate import EvalManager, plot_results   # 复用 PSTG 的可视化


# ── 推理（双信号）────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference_ma(
    model:  PSTG_MA,
    loader,
    device: str,
    tau:    int,
) -> tuple:
    """
    推理，同时收集预测值 x_pred 和记忆重构误差 mem_error。

    Returns:
        x_pred    : [T, C]  主预测序列
        x_pred_mem: [T, C]  记忆引导预测序列（v2：数据空间，与 x_pred 同量级）
    """
    model.eval()
    all_preds      = []
    all_mem_errors = []   # 存 x_hat_mem（名称保持兼容）

    for context, _ in tqdm(loader, desc="  推理（双信号）"):
        context = context.to(device, non_blocking=True)           # [B, C, L]
        x_hat, x_hat_mem, mem_outputs = model(context)            # v2：3个返回值

        # Signal 1：主预测（只保留前 τ 步）
        pred_tau = x_hat[:, :, :tau].permute(0, 2, 1).reshape(-1, x_hat.shape[1])
        all_preds.append(pred_tau.cpu().numpy())

        # Signal 2：记忆引导预测（v2：在数据空间，与主预测同量级）
        # 存储 x_hat_mem 的前 τ 步，评估时与 x_true 作差得到误差
        pred_mem_tau = x_hat_mem[:, :, :tau].permute(0, 2, 1).reshape(-1, x_hat_mem.shape[1])
        all_mem_errors.append(pred_mem_tau.cpu().numpy())

    x_pred     = np.concatenate(all_preds,      axis=0).astype(np.float32)
    mem_errors = np.concatenate(all_mem_errors, axis=0).astype(np.float32)
    return x_pred, mem_errors


def dual_signal_scores(
    x_true:      np.ndarray,   # [T, C]  真实值
    x_pred:      np.ndarray,   # [T, C]  主预测
    x_pred_mem:  np.ndarray,   # [T, C]  记忆引导预测（v2：数据空间）
    alpha:       float,         # 主预测残差权重
    smooth_window: int,
) -> tuple:
    """
    v2：两路信号均在数据空间计算，量级相同，信号更强。

    Returns:
        combined:      [T] 融合分数（用于阈值搜索）
        r_pred_smooth: [T] 主预测残差
        r_mem_smooth:  [T] 记忆引导残差
    """
    # Signal 1：主预测残差（跨通道最大值）
    r_pred = np.abs(x_true - x_pred).max(axis=1)
    r_pred_smooth = smooth_residuals(r_pred, smooth_window).astype(np.float32)

    # Signal 2：记忆引导预测残差（v2 核心改动：数据空间，同量级）
    r_mem = np.abs(x_true - x_pred_mem).max(axis=1)
    r_mem_smooth = smooth_residuals(r_mem, smooth_window).astype(np.float32)

    # MinMax 归一化后加权融合
    def minmax(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-9)

    combined = alpha * minmax(r_pred_smooth) + (1 - alpha) * minmax(r_mem_smooth)
    return combined, r_pred_smooth, r_mem_smooth


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PSTG-MA 评估")
    p.add_argument("--ckpt",      type=str, default=None)
    p.add_argument("--data_dir",  type=str, default=None)
    p.add_argument("--device",    type=str, default=None)
    p.add_argument("--output",    type=str, default=None)
    p.add_argument("--alpha",     type=float, default=None,
                   help="预测残差权重（覆盖 checkpoint 中保存的值）")
    p.add_argument("--no_plot",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = ConfigMA()

    if args.data_dir: cfg.DATA_DIR   = args.data_dir
    if args.device:   cfg.DEVICE     = args.device
    output_dir = args.output or (cfg.OUTPUT_DIR + "_ma")

    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")

    eval_mgr = EvalManager(output_dir)

    # ── 加载 checkpoint ───────────────────────────────────────────────────
    ckpt_path = args.ckpt or str(Path(cfg.CHECKPOINT_DIR + "_ma") / "best.pt")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint 不存在：{ckpt_path}")

    print(f"\n加载 checkpoint：{ckpt_path}")
    ckpt     = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("config", {})

    model = PSTG_MA(
        patch_sizes=  ckpt_cfg.get("patch_sizes",  cfg.PATCH_SIZES),
        d_model=      ckpt_cfg.get("d_model",       cfg.D_MODEL),
        num_heads=    ckpt_cfg.get("num_heads",     cfg.NUM_HEADS),
        num_layers=   ckpt_cfg.get("num_layers",    cfg.NUM_LAYERS),
        n_channels=   ckpt_cfg.get("n_channels",    cfg.NUM_CHANNELS),
        context_len=  ckpt_cfg.get("context_len",   cfg.CONTEXT_LEN),
        forecast_len= ckpt_cfg.get("forecast_len",  cfg.FORECAST_LEN),
        top_k=cfg.top_k,
        dropout=0.0,
        num_memory_slots=  cfg.NUM_MEMORY_SLOTS,
        memory_temperature=cfg.MEMORY_TEMPERATURE,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    alpha = args.alpha if args.alpha is not None else ckpt.get("alpha_pred", cfg.ALPHA_PRED)
    print(f"  epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss','?')}")
    print(f"  双信号权重：α={alpha}（pred）+ {1-alpha:.2f}（mem）")

    # ── 数据 & 推理 ───────────────────────────────────────────────────────
    print("\n=== 加载测试数据 ===")
    data        = build_datasets(cfg)
    test_loader = data["test_loader"]
    test_labels = data["test_labels"]
    test_data   = data["test_data"]

    print("\n=== 双信号推理 ===")
    x_pred, x_pred_mem = run_inference_ma(model, test_loader, device, cfg.TAU)
    T_pred = len(x_pred)
    print(f"预测序列长度：{T_pred:,}")

    x_true = test_data[cfg.CONTEXT_LEN : cfg.CONTEXT_LEN + T_pred]
    y_true = test_labels[cfg.CONTEXT_LEN : cfg.CONTEXT_LEN + T_pred]

    # ── 双信号融合 ────────────────────────────────────────────────────────
    print("\n=== 双信号融合异常检测（v2：数据空间）===")
    combined, r_pred_s, r_mem_s = dual_signal_scores(
        x_true, x_pred, x_pred_mem[:T_pred], alpha, cfg.smooth_window
    )
    print(f"  主预测残差范围：  [{r_pred_s.min():.4f}, {r_pred_s.max():.4f}]")
    print(f"  记忆引导残差范围：[{r_mem_s.min():.4f}, {r_mem_s.max():.4f}]  ← v2 与主残差同量级")
    print(f"  融合分数范围：    [{combined.min():.4f}, {combined.max():.4f}]")

    # ── 评估（与 PSTG 一致：用改进版 Telemanom 检测双信号融合分数）────────
    print("\n=== 评估 ===")
    from utils.metrics import extract_events
    from anomaly.detector import _find_optimal_threshold, _compute_anomaly_scores

    # 对 combined 信号（已在 [0,1] 区间）直接应用改进版 Telemanom 阈值
    r_combined = combined.astype(np.float64)
    mu_c    = float(np.mean(r_combined))
    sigma_c = float(np.std(r_combined))
    eps_c   = _find_optimal_threshold(r_combined)
    scores_c = _compute_anomaly_scores(r_combined, eps_c, mu_c, sigma_c, min_peak_z=1.5)
    y_pred = (scores_c > 0).astype(np.int32)
    pred_rate = float(y_pred.mean())
    print(f"  combined 阈值: {eps_c:.4f}  预测异常率：{pred_rate*100:.3f}%  真实异常率：{y_true.mean()*100:.3f}%")

    from utils.metrics import event_wise_metrics, affiliation_metrics
    ew = event_wise_metrics(y_true, y_pred)
    af = affiliation_metrics(y_true, y_pred)
    n_events_all = len(extract_events(y_true))

    # ── 标准2：过滤单点事件（duration≥2，与论文协议一致）────────────────
    y_true_filt = np.zeros_like(y_true)
    for s, e in extract_events(y_true):
        if e - s + 1 >= 2:
            y_true_filt[s:e+1] = 1
    ew2 = event_wise_metrics(y_true_filt, y_pred)
    af2 = affiliation_metrics(y_true_filt, y_pred)
    n_events_filt = len(extract_events(y_true_filt))

    print(f"\n─── 标准1：全部 {n_events_all} 个事件（含单点标注）───")
    print(f"  Event-wise  P={ew['precision']:.4f}  R={ew['recall']:.4f}  F0.5={ew['f0.5']:.4f}")
    print(f"  Affiliation P={af['precision']:.4f}  R={af['recall']:.4f}  F0.5={af['f0.5']:.4f}")

    print(f"\n─── 标准2：{n_events_filt} 个事件（duration≥2，与论文协议一致）───")
    print(f"  Event-wise  P={ew2['precision']:.4f}  R={ew2['recall']:.4f}  "
          f"F0.5={ew2['f0.5']:.4f}  (PSTG复现: 0.921)")
    print(f"  Affiliation P={af2['precision']:.4f}  R={af2['recall']:.4f}  "
          f"F0.5={af2['f0.5']:.4f}  (PSTG复现: 0.741)")

    # ── 保存 ──────────────────────────────────────────────────────────────
    metrics = {
        "event_wise":  {"precision": float(ew["precision"]),
                        "recall":    float(ew["recall"]),
                        "f0.5":      float(ew["f0.5"]),
                        "n_events":  n_events_all},
        "affiliation": {"precision": float(af["precision"]),
                        "recall":    float(af["recall"]),
                        "f0.5":      float(af["f0.5"])},
        "event_wise_filt":  {"precision": float(ew2["precision"]),
                             "recall":    float(ew2["recall"]),
                             "f0.5":      float(ew2["f0.5"]),
                             "n_events":  n_events_filt},
        "affiliation_filt": {"precision": float(af2["precision"]),
                             "recall":    float(af2["recall"]),
                             "f0.5":      float(af2["f0.5"])},
        "threshold":         float(eps_c),
        "pred_anomaly_rate": float(pred_rate),
        "alpha_pred":        float(alpha),
        "pstg_baseline":     {"event_f05": 0.921, "affil_f05": 0.741},
    }
    info = {
        "ckpt_path":   ckpt_path,
        "ckpt_epoch":  ckpt.get("epoch", "?"),
        "model_type":  "PSTG_MA",
        "alpha_pred":  float(alpha),
        "eval_time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    eval_mgr.save_results(metrics, info)
    np.save(eval_mgr.eval_dir / "combined_scores.npy",   combined)
    np.save(eval_mgr.eval_dir / "pred_residuals.npy",    r_pred_s)
    np.save(eval_mgr.eval_dir / "memory_residuals.npy",  r_mem_s)   # v2：数据空间残差
    print(f"\n  → 结果保存至：{eval_mgr.eval_dir}")

    # ── 绘图 ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        print("\n=== 绘图 ===")
        # 标准可视化（用融合分数）
        plot_results(
            y_true=y_true, raw_smoothed=combined,
            anomaly_scores=combined, x_true=x_true, x_pred=x_pred,
            threshold=eps_c, eval_dir=eval_mgr.eval_dir,
            n_channels=cfg.NUM_CHANNELS,
        )
        # 双信号对比图
        _plot_dual_signals(y_true, r_pred_s, r_mem_s, combined,
                           eps_c, alpha, eval_mgr.eval_dir)

    eval_mgr.finalize(metrics, info)

    print(f"\n{'='*50}")
    print(f"PSTG-MA 评估完成！")
    print(f"  [标准1 全部事件] Event F0.5 = {ew['f0.5']:.4f}  Affil F0.5 = {af['f0.5']:.4f}")
    print(f"  [标准2 过滤单点] Event F0.5 = {ew2['f0.5']:.4f}  Affil F0.5 = {af2['f0.5']:.4f}  ← 与论文对比")
    print(f"  PSTG 复现基线:   Event F0.5 = 0.9211  Affil F0.5 = 0.7410")
    gain_e = ew2['f0.5'] - 0.9211
    gain_a = af2['f0.5'] - 0.7410
    print(f"  相比 PSTG 复现:  Event {gain_e:+.4f}，Affil {gain_a:+.4f}")
    print(f"{'='*50}")
    print(f"PSTG-MA 评估完成！")
    print(f"  Event-wise  F0.5 = {ew['f0.5']:.4f}  (PSTG 论文: 0.917)")
    print(f"  Affiliation F0.5 = {af['f0.5']:.4f}  (PSTG 论文: 0.892)")
    gain_e = ew['f0.5'] - 0.917
    gain_a = af['f0.5'] - 0.892
    print(f"  相比 PSTG：Event {gain_e:+.4f}，Affil {gain_a:+.4f}")
    print(f"{'='*50}")


def _plot_dual_signals(y_true, r_pred, r_mem, combined, threshold, alpha, eval_dir):
    """绘制三条曲线对比：预测残差 / 记忆误差 / 融合分数"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    T  = min(len(combined), 5000)
    t  = np.arange(T)
    gt = y_true[:T].astype(bool)

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    labels_data = [
        (r_pred[:T], "Signal 1: Pred Residual",  "steelblue"),
        (r_mem[:T],  "Signal 2: Memory Error",   "darkorange"),
        (combined[:T], f"Combined (α={alpha})",  "crimson"),
    ]
    for ax, (data, label, color) in zip(axes, labels_data):
        ax.plot(t, data, color=color, lw=0.8, label=label)
        # 真实异常区间
        in_r = False
        for i in range(T):
            if gt[i] and not in_r:
                s = i; in_r = True
            elif not gt[i] and in_r:
                ax.axvspan(s, i, alpha=0.2, color="green")
                in_r = False
        if in_r:
            ax.axvspan(s, T, alpha=0.2, color="green")
        ax.set_ylabel(label, fontsize=8)
        ax.legend(loc="upper right", fontsize=8)
    axes[2].axhline(threshold, color="red", ls="--", lw=1.0, label=f"thresh={threshold:.4f}")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time Step")
    fig.suptitle("PSTG-MA Dual-Signal Anomaly Detection", fontsize=11)
    plt.tight_layout()
    fig.savefig(eval_dir / "dual_signals.png", dpi=150)
    plt.close(fig)
    print(f"  → {eval_dir}/dual_signals.png")


if __name__ == "__main__":
    main()
