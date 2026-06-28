"""
MTA 推理与评估脚本

与 evaluate.py 的核心区别：
  1. 异常分数来源：重建误差（不是预测残差）
     - PSTG：score = max_c |x_true_c - x_pred_c|（预测未来的误差）
     - MTA ：score = max_c(mean_patch |x_recon - x_orig|)（重建当前的误差）
  2. 不需要 x_pred（无未来预测），直接输出 per-window 异常分数
  3. smooth + POT 流程完全相同（保证评估协议一致）

用法：
  python evaluate_mta.py
  python evaluate_mta.py --ckpt checkpoints_mta/run_xxx/best.pt
  python evaluate_mta.py --pot_alpha 4e-3 --min_peak_z 1.5
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config_mta import ConfigMTA
from data.dataset import build_datasets
from models.mta import MTA
from anomaly.detector import smooth_residuals, threshold_signal
from utils.metrics import event_wise_metrics, affiliation_metrics, extract_events


# ─────────────────────────────────────────────────────────────────────────────
#  评估结果管理器（与 evaluate.py 相同结构）
# ─────────────────────────────────────────────────────────────────────────────

class EvalManager:
    def __init__(self, base_output_dir: str):
        self.base_dir = Path(base_output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_name = f"eval_{ts}"
        self.eval_dir  = self.base_dir / self.eval_name
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        print(f"本次评估目录：{self.eval_dir}")

    def save_results(self, metrics, info):
        (self.eval_dir / "evaluation_results.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False)
        )
        (self.eval_dir / "eval_info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False)
        )

    def save_scores(self, scores: np.ndarray):
        np.save(self.eval_dir / "anomaly_scores.npy", scores)

    def finalize(self, metrics, info):
        summary_path = self.base_dir / "eval_summary.json"
        history = json.loads(summary_path.read_text()) if summary_path.exists() else []
        history.append({
            "eval_name":  self.eval_name,
            "eval_dir":   str(self.eval_dir),
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ckpt_path":  info.get("ckpt_path", ""),
            "event_f05":  metrics["event_wise"]["f0.5"],
            "event_prec": metrics["event_wise"]["precision"],
            "event_rec":  metrics["event_wise"]["recall"],
            "affil_f05":  metrics["affiliation"]["f0.5"],
            "affil_prec": metrics["affiliation"]["precision"],
            "affil_rec":  metrics["affiliation"]["recall"],
        })
        summary_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))

        latest_link = self.base_dir / "latest"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(self.eval_dir.name)
        except Exception:
            pass

        print(f"\n汇总已追加至：{summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  推理：计算 per-timestep 重建误差
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference_mta(model, test_loader, device: str, last_k: int = 0) -> np.ndarray:
    """
    逐窗口推理，计算每个窗口的重建异常分数。

    返回形状 [T_windows]，每个元素是对应时间步的重建误差：
      score = max_c( mean_patch |recon_patch - orig_patch| )

    与 PSTG 的 run_inference 对齐：
      - 窗口数 T_windows = T_test - L - F + 1（tau=1）
      - 索引 i 对应测试集时间步 L + i
    """
    model.eval()
    all_scores  = []
    all_recon   = []   # 最后一个 patch 的重建值，用于通道对比图 [T, C]

    for context, _ in tqdm(test_loader, desc="  MTA 推理（重建模式）"):
        context = context.to(device, non_blocking=True)   # [B, C, L]

        # 推理模式：mask=None（不掩码，重建全部 patch）
        recon, _, target = model(context, mask=None)      # [B, C, N, p_main]

        # 保存最后一个 patch 的重建均值，作为每时间步的"重建信号"[B, C]
        all_recon.append(recon[:, :, -1, :].mean(dim=-1).cpu().numpy())

        # patch 级绝对误差：[B, C, N, p_main] → [B, C, N]
        patch_err = (recon - target).abs().mean(dim=-1)

        # last_k > 0：只看最近 k 个 patch（提升时间局部性，改善 Affil）
        # last_k = 0：看全部 patch（默认，Event F0.5 最优）
        if last_k > 0:
            patch_err = patch_err[:, :, -last_k:]   # [B, C, last_k]

        # 跨通道取 max，跨时间 patch 取 max → 单值异常分数 [B]
        score = patch_err.max(dim=1).values.max(dim=-1).values

        all_scores.append(score.cpu().numpy())

    x_recon = np.concatenate(all_recon, axis=0).astype(np.float32)  # [T, C]
    return np.concatenate(all_scores, axis=0).astype(np.float32), x_recon


# ─────────────────────────────────────────────────────────────────────────────
#  可视化（可选）
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(y_true, raw_smoothed, anomaly_scores, threshold, eval_dir, max_plot_len=5000):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib 未安装，跳过绘图")
        return

    T  = min(len(raw_smoothed), max_plot_len)
    t  = np.arange(T)
    gt = y_true[:T].astype(bool)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(t, raw_smoothed[:T], color="steelblue", lw=0.8, label="Smoothed Recon Error")
    ax.axhline(threshold, color="red", ls="--", lw=1.2)

    in_r = False
    for i in range(len(gt)):
        if gt[i] and not in_r:  s = i; in_r = True
        elif not gt[i] and in_r:
            ax.axvspan(s, i, alpha=0.2, color="green"); in_r = False
    if in_r:
        ax.axvspan(s, len(gt), alpha=0.2, color="green")

    ax.set_title("MTA — Smoothed Reconstruction Error & Detection Threshold")
    ax.legend(handles=[
        mpatches.Patch(color="steelblue",        label="Smoothed Recon Error"),
        mpatches.Patch(color="red",   alpha=0.8, label=f"Threshold={threshold:.4f}"),
        mpatches.Patch(color="green", alpha=0.3, label="Ground Truth Anomaly"),
    ])
    plt.tight_layout()
    fig.savefig(eval_dir / "anomaly_scores.png", dpi=150)
    plt.close(fig)
    print(f"  → {eval_dir}/anomaly_scores.png")


def plot_channel_reconstruction(
    y_true, x_true, x_recon, eval_dir, n_channels: int,
    max_plot_len: int = 5000,
):
    """各通道：原始信号 vs MTA重建信号（对应 evaluate.py 的 channel_predictions.png）"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        return

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
        ax.plot(t, x_recon[:T, c], color="gray", lw=0.7, ls="--", alpha=0.8,
                label="MTA Reconstruction")
        shade(ax, gt)
        ax.set_ylabel(f"Ch {c+41}"); ax.set_xlim(0, T)
        if c == 0:
            ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Time Step")
    fig.suptitle("MTA: Reconstruction vs. Original (all channels)", fontsize=12)
    plt.tight_layout()
    fig.savefig(eval_dir / "channel_reconstruction.png", dpi=150)
    plt.close(fig)
    print(f"  → {eval_dir}/channel_reconstruction.png")


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MTA 推理与评估")
    p.add_argument("--ckpt",       type=str,   default=None)
    p.add_argument("--data_dir",   type=str,   default=None)
    p.add_argument("--device",     type=str,   default=None)
    p.add_argument("--output",     type=str,   default=None)
    p.add_argument("--no_plot",      action="store_true")
    p.add_argument("--smooth_window", type=int, default=None,
                   help="平滑窗口大小（默认 cfg.smooth_window=105）")
    p.add_argument("--last_k", type=int, default=0,
                   help="只用最近 k 个 patch 计分（0=全部10个；1/2/3 可改善 Affil，更时间局部）")
    p.add_argument("--method",     type=str,   default="pot",
                   choices=["pot", "robust"])
    p.add_argument("--pot_alpha",  type=float, default=4e-3,
                   help="POT 目标超阈率（默认 4e-3，与 PSTG 最优参数一致）")
    p.add_argument("--pot_q0",     type=float, default=0.98)
    p.add_argument("--min_peak_z", type=float, default=1.5,
                   help="假阳性剪枝 z 分数（默认 1.5，与 PSTG 最优参数一致）")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = ConfigMTA()

    if args.data_dir:     cfg.DATA_DIR   = args.data_dir
    if args.device:       cfg.DEVICE     = args.device
    if args.output:       cfg.OUTPUT_DIR = args.output
    smooth_window = args.smooth_window if args.smooth_window else cfg.smooth_window

    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")

    eval_mgr = EvalManager(cfg.OUTPUT_DIR)

    # ── 加载 checkpoint ─────────────────────────────────────────────────────
    ckpt_path = args.ckpt or str(Path(cfg.CHECKPOINT_DIR) / "best.pt")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint 不存在：{ckpt_path}\n请先运行 python train_mta.py"
        )

    print(f"\n加载 checkpoint：{ckpt_path}")
    ckpt     = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("config", {})

    model = MTA(
        patch_sizes=ckpt_cfg.get("patch_sizes",  cfg.PATCH_SIZES),
        d_model=    ckpt_cfg.get("d_model",       cfg.D_MODEL),
        num_heads=  ckpt_cfg.get("num_heads",     cfg.NUM_HEADS),
        num_layers= ckpt_cfg.get("num_layers",    cfg.NUM_LAYERS),
        n_channels= ckpt_cfg.get("n_channels",    cfg.NUM_CHANNELS),
        context_len=ckpt_cfg.get("context_len",   cfg.CONTEXT_LEN),
        mask_ratio= ckpt_cfg.get("mask_ratio",    cfg.MASK_RATIO),
        top_k=cfg.top_k,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_run   = ckpt.get("run_name", "unknown")
    ckpt_val   = ckpt.get("val_loss", "?")
    print(f"  来自 run: {ckpt_run}  epoch={ckpt_epoch}  val_loss={ckpt_val}")

    # ── 加载数据 ────────────────────────────────────────────────────────────
    print("\n=== 加载测试数据 ===")
    data        = build_datasets(cfg)
    test_loader = data["test_loader"]
    test_labels = data["test_labels"]
    test_data   = data["test_data"]

    # ── 推理（重建误差）────────────────────────────────────────────────────
    print("\n=== MTA 推理（重建模式，无掩码）===")
    raw_scores, x_recon = run_inference_mta(model, test_loader, device, last_k=args.last_k)
    T_windows  = len(raw_scores)
    print(f"窗口数（= 测试时间步数）：{T_windows:,}")
    print(f"原始重建误差范围：[{raw_scores.min():.4f}, {raw_scores.max():.4f}]")

    # 与 PSTG 评估对齐：取测试集中 [L, L+T_windows) 范围的数据和标签
    x_true = test_data  [cfg.CONTEXT_LEN : cfg.CONTEXT_LEN + T_windows]   # [T, C]
    y_true = test_labels[cfg.CONTEXT_LEN : cfg.CONTEXT_LEN + T_windows].astype(np.int32)
    print(f"真实异常率：{y_true.mean()*100:.3f}%")

    # ── smooth + POT（与 PSTG 完全相同的后处理流程）───────────────────────
    print("\n=== 异常检测 ===")
    raw_smoothed = smooth_residuals(raw_scores, smooth_window).astype(np.float32)

    print(f"  平滑窗口：{smooth_window}  方法：{args.method}")
    print(f"  平滑后范围：[{raw_smoothed.min():.4f}, {raw_smoothed.max():.4f}]")

    anomaly_scores = threshold_signal(
        raw_smoothed,
        method=args.method,
        pot_q0=args.pot_q0,
        pot_alpha=args.pot_alpha,
        min_peak_z=args.min_peak_z,
    )

    # ── 评估 ────────────────────────────────────────────────────────────────
    print("\n=== 评估 ===")
    y_pred    = (anomaly_scores > 0).astype(np.int32)
    pred_rate = float(y_pred.mean())
    print(f"  预测异常率：{pred_rate*100:.3f}%  真实异常率：{y_true.mean()*100:.3f}%")

    # 标准1：全部事件
    ew = event_wise_metrics(y_true, y_pred)
    af = affiliation_metrics(y_true, y_pred)
    n_events_all = len(extract_events(y_true))
    print(f"\n─── 标准1：全部 {n_events_all} 个事件（含单点标注）───")
    print(f"  Event-wise  P={ew['precision']:.4f}  R={ew['recall']:.4f}  F0.5={ew['f0.5']:.4f}")
    print(f"  Affiliation P={af['precision']:.4f}  R={af['recall']:.4f}  F0.5={af['f0.5']:.4f}")

    # 标准2：过滤单点事件（与论文协议一致）
    y_true_filt = np.zeros_like(y_true)
    for s, e in extract_events(y_true):
        if e - s + 1 >= 2:
            y_true_filt[s:e+1] = 1
    ew2 = event_wise_metrics(y_true_filt, y_pred)
    af2 = affiliation_metrics(y_true_filt, y_pred)
    n_events_filt = len(extract_events(y_true_filt))

    print(f"\n─── 标准2：{n_events_filt} 个事件（duration≥2，与论文协议一致）───")
    print(f"  Event-wise  P={ew2['precision']:.4f}  R={ew2['recall']:.4f}  "
          f"F0.5={ew2['f0.5']:.4f}  (PSTG论文: 0.917, 我们复现: 0.924)")
    print(f"  Affiliation P={af2['precision']:.4f}  R={af2['recall']:.4f}  "
          f"F0.5={af2['f0.5']:.4f}  (PSTG论文: 0.892, 我们复现: 0.876)")

    # ── 保存 ────────────────────────────────────────────────────────────────
    threshold_val = float(raw_smoothed[y_pred == 1].min()) if y_pred.any() else 0.0

    metrics = {
        "event_wise":       {"precision": float(ew["precision"]),  "recall": float(ew["recall"]),  "f0.5": float(ew["f0.5"]),  "n_events": n_events_all},
        "affiliation":      {"precision": float(af["precision"]),  "recall": float(af["recall"]),  "f0.5": float(af["f0.5"])},
        "event_wise_filt":  {"precision": float(ew2["precision"]), "recall": float(ew2["recall"]), "f0.5": float(ew2["f0.5"]), "n_events": n_events_filt},
        "affiliation_filt": {"precision": float(af2["precision"]), "recall": float(af2["recall"]), "f0.5": float(af2["f0.5"])},
        "threshold":        threshold_val,
        "pred_anomaly_rate": pred_rate,
        "pstg_paper_event_f05": 0.917,
        "pstg_paper_affil_f05": 0.892,
        "pstg_ours_event_f05":  0.924,
        "pstg_ours_affil_f05":  0.876,
    }
    info = {
        "model":        "MTA",
        "ckpt_path":    ckpt_path,
        "ckpt_epoch":   ckpt_epoch,
        "ckpt_run":     ckpt_run,
        "ckpt_val_loss": str(ckpt_val),
        "eval_time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_windows":    T_windows,
        "anomaly_rate": float(y_true.mean()),
        "threshold_method": args.method,
        "pot_alpha":    args.pot_alpha,
        "pot_q0":       args.pot_q0,
        "min_peak_z":   args.min_peak_z,
        "smooth_window": smooth_window,
    }
    eval_mgr.save_results(metrics, info)
    eval_mgr.save_scores(anomaly_scores)
    np.save(eval_mgr.eval_dir / "raw_smoothed.npy", raw_smoothed)
    np.save(eval_mgr.eval_dir / "x_recon.npy",      x_recon)
    np.save(eval_mgr.eval_dir / "x_true.npy",       x_true)
    np.save(eval_mgr.eval_dir / "y_true.npy",       y_true)

    # ── 绘图 ────────────────────────────────────────────────────────────────
    if not args.no_plot:
        print("\n=== 绘图 ===")
        plot_results(y_true, raw_smoothed, anomaly_scores, threshold_val, eval_mgr.eval_dir)
        plot_channel_reconstruction(
            y_true=y_true, x_true=x_true, x_recon=x_recon,
            eval_dir=eval_mgr.eval_dir, n_channels=cfg.NUM_CHANNELS,
        )

    eval_mgr.finalize(metrics, info)

    print(f"\n{'='*55}")
    print(f"MTA 评估完成！结果目录：{eval_mgr.eval_dir}")
    print(f"\n  标准2（论文协议）结果：")
    print(f"  Event-wise  F0.5 = {ew2['f0.5']:.4f}  "
          f"(P={ew2['precision']:.4f}, R={ew2['recall']:.4f})")
    print(f"  Affiliation F0.5 = {af2['f0.5']:.4f}  "
          f"(P={af2['precision']:.4f}, R={af2['recall']:.4f})")
    print(f"\n  对比：PSTG论文={0.917}/{0.892}，PSTG复现={0.924}/{0.876}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
