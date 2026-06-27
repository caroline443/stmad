"""
PSTG 推理与评估脚本（对应 Algorithm 1 Part 2 & 3）

保存机制：
  每次评估创建带时间戳的独立目录，旧结果永不被覆盖。

  outputs/
  ├── eval_summary.json              ← 所有评估的汇总表（持续追加）
  ├── latest -> eval_YYYYMMDD_HHmmss ← 符号链接，始终指向最新一次
  └── eval_YYYYMMDD_HHmmss/          ← 本次评估的独立目录
      ├── evaluation_results.json    ← Event-wise & Affiliation 指标
      ├── eval_info.json             ← 配置信息（用了哪个 ckpt、epoch 等）
      ├── anomaly_scores.npy         ← 连续异常分数
      ├── anomaly_scores.png         ← 分数时序图
      └── channel_predictions.png   ← 各通道预测对比图

用法：
  python evaluate.py
  python evaluate.py --ckpt checkpoints/run_20260626_180000/best.pt
  python evaluate.py --no_plot   # 跳过绘图（更快）
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import Config
from data.dataset import build_datasets
from models.pstg import PSTG
from anomaly.detector import detect_anomalies, smooth_residuals
from utils.metrics import find_best_threshold


# ─────────────────────────────────────────────────────────────────────────────
#  评估结果管理器
# ─────────────────────────────────────────────────────────────────────────────

class EvalManager:
    """每次评估创建独立目录，汇总写入 eval_summary.json。"""

    def __init__(self, base_output_dir: str):
        self.base_dir = Path(base_output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_name = f"eval_{ts}"
        self.eval_dir  = self.base_dir / self.eval_name
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        print(f"本次评估目录：{self.eval_dir}")

    def save_results(self, metrics: dict, info: dict):
        """保存指标 JSON 和配置 JSON。"""
        # 指标
        result_path = self.eval_dir / "evaluation_results.json"
        with open(result_path, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # 配置信息
        info_path = self.eval_dir / "eval_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        return result_path

    def save_scores(self, anomaly_scores: np.ndarray):
        """保存异常分数 npy。"""
        score_path = self.eval_dir / "anomaly_scores.npy"
        np.save(score_path, anomaly_scores)
        return score_path

    def finalize(self, metrics: dict, info: dict):
        """把本次评估摘要追加到全局 eval_summary.json。"""
        summary_path = self.base_dir / "eval_summary.json"
        history = []
        if summary_path.exists():
            with open(summary_path) as f:
                history = json.load(f)

        entry = {
            "eval_name":      self.eval_name,
            "eval_dir":       str(self.eval_dir),
            "finished_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ckpt_path":      info.get("ckpt_path", ""),
            "ckpt_epoch":     info.get("ckpt_epoch", ""),
            "ckpt_run":       info.get("ckpt_run", ""),
            "event_f05":      metrics["event_wise"]["f0.5"],
            "event_prec":     metrics["event_wise"]["precision"],
            "event_rec":      metrics["event_wise"]["recall"],
            "affil_f05":      metrics["affiliation"]["f0.5"],
            "affil_prec":     metrics["affiliation"]["precision"],
            "affil_rec":      metrics["affiliation"]["recall"],
        }
        history.append(entry)
        with open(summary_path, "w") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # latest 软链接（Linux/Mac）
        latest_link = self.base_dir / "latest"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(self.eval_dir.name)
        except Exception:
            pass  # Windows 不支持软链接也无妨

        print(f"\n汇总已追加至：{summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  推理
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, test_loader, device: str, tau: int) -> np.ndarray:
    """滑动窗口推理，每次只保留前 τ=1 步，拼接为完整预测序列。"""
    model.eval()
    all_preds = []
    for context, _ in tqdm(test_loader, desc="  推理"):
        context  = context.to(device, non_blocking=True)
        pred     = model(context)                        # [B, C, F]
        pred_tau = pred[:, :, :tau]                      # [B, C, τ]
        pred_tau = pred_tau.permute(0, 2, 1).reshape(-1, pred.shape[1])
        all_preds.append(pred_tau.cpu().numpy())
    return np.concatenate(all_preds, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  可视化
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    y_true, raw_smoothed, anomaly_scores, x_true, x_pred,
    threshold, eval_dir: Path, n_channels: int,
    max_plot_len: int = 5000,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  matplotlib 未安装，跳过绘图")
        return

    T  = min(len(raw_smoothed), max_plot_len)
    t  = np.arange(T)
    gt = y_true[:T].astype(bool)

    def shade_anomalies(ax, mask):
        in_r = False
        for i in range(len(mask)):
            if mask[i] and not in_r:
                s = i; in_r = True
            elif not mask[i] and in_r:
                ax.axvspan(s, i, alpha=0.2, color="green")
                in_r = False
        if in_r:
            ax.axvspan(s, len(mask), alpha=0.2, color="green")

    # ── 图 1：原始平滑残差 + 阈值 ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(t, raw_smoothed[:T], color="steelblue", lw=0.8, label="Smoothed Residual")
    ax.axhline(threshold, color="red", ls="--", lw=1.2, label=f"Threshold={threshold:.4f}")
    shade_anomalies(ax, gt)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Residual")
    ax.set_title("PSTG — Smoothed Residuals & Detection Threshold")
    ax.legend(handles=[
        mpatches.Patch(color="steelblue",           label="Smoothed Residual"),
        mpatches.Patch(color="red",   alpha=0.8,    label=f"Threshold={threshold:.4f}"),
        mpatches.Patch(color="green", alpha=0.3,    label="Ground Truth Anomaly"),
    ])
    plt.tight_layout()
    fig.savefig(eval_dir / "anomaly_scores.png", dpi=150)
    plt.close(fig)

    # ── 图 2：各通道预测 vs 真实 ─────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 2.5 * n_channels))
    gs  = GridSpec(n_channels, 1, figure=fig, hspace=0.4)
    colors = plt.cm.tab10.colors
    for c in range(n_channels):
        ax = fig.add_subplot(gs[c])
        ax.plot(t, x_true[:T, c], color=colors[c % 10], lw=0.7, label="Ground Truth")
        ax.plot(t, x_pred[:T, c], color="gray",          lw=0.7, ls="--", alpha=0.8, label="Prediction")
        shade_anomalies(ax, gt)
        ax.set_ylabel(f"Ch {c+41}"); ax.set_xlim(0, T)
        if c == 0:
            ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Time Step")
    fig.suptitle("PSTG: Prediction vs Ground Truth (all channels)", fontsize=12)
    plt.tight_layout()
    fig.savefig(eval_dir / "channel_predictions.png", dpi=150)
    plt.close(fig)

    print(f"  → {eval_dir}/anomaly_scores.png")
    print(f"  → {eval_dir}/channel_predictions.png")


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PSTG 推理与评估")
    p.add_argument("--ckpt",      type=str,   default=None,
                   help="Checkpoint 路径（默认 checkpoints/best.pt）")
    p.add_argument("--data_dir",  type=str,   default=None)
    p.add_argument("--device",    type=str,   default=None)
    p.add_argument("--output",    type=str,   default=None,
                   help="评估结果根目录（默认 outputs/）")
    p.add_argument("--no_plot",   action="store_true", help="跳过绘图")
    # 阈值算法参数
    p.add_argument("--method",    type=str,   default="pot",
                   choices=["pot", "robust"],
                   help="阈值算法：pot=极值理论(默认)，robust=鲁棒正态拟合")
    p.add_argument("--pot_alpha", type=float, default=4e-3,
                   help="POT 目标超阈率，越小阈值越高（默认 4e-3，约等于ESA-AD真实异常率）")
    p.add_argument("--pot_q0",     type=float, default=0.98,
                   help="POT 初始截断分位数（默认 0.98）")
    p.add_argument("--min_peak_z", type=float, default=1.5,
                   help="假阳性剪枝：序列峰值需超过 μ+z*σ 才保留（默认 1.5）")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()

    if args.data_dir: cfg.DATA_DIR   = args.data_dir
    if args.device:   cfg.DEVICE     = args.device
    if args.output:   cfg.OUTPUT_DIR = args.output

    method      = args.method
    pot_alpha   = args.pot_alpha
    pot_q0      = args.pot_q0
    min_peak_z  = args.min_peak_z

    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")

    # ── 初始化评估管理器 ──────────────────────────────────────────────────
    eval_mgr = EvalManager(cfg.OUTPUT_DIR)

    # ── 加载 checkpoint ───────────────────────────────────────────────────
    ckpt_path = args.ckpt or str(Path(cfg.CHECKPOINT_DIR) / "best.pt")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint 不存在：{ckpt_path}\n请先运行 train.py")

    print(f"\n加载 checkpoint：{ckpt_path}")
    ckpt     = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("config", {})

    model = PSTG(
        patch_sizes=  ckpt_cfg.get("patch_sizes",  cfg.PATCH_SIZES),
        d_model=      ckpt_cfg.get("d_model",       cfg.D_MODEL),
        num_heads=    ckpt_cfg.get("num_heads",     cfg.NUM_HEADS),
        num_layers=   ckpt_cfg.get("num_layers",    cfg.NUM_LAYERS),
        n_channels=   ckpt_cfg.get("n_channels",    cfg.NUM_CHANNELS),
        context_len=  ckpt_cfg.get("context_len",   cfg.CONTEXT_LEN),
        forecast_len= ckpt_cfg.get("forecast_len",  cfg.FORECAST_LEN),
        top_k=cfg.top_k,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    ckpt_epoch   = ckpt.get("epoch", "?")
    ckpt_run     = ckpt.get("run_name", "unknown")
    ckpt_val     = ckpt.get("val_loss", "?")
    print(f"  来自 run: {ckpt_run}  epoch={ckpt_epoch}  val_loss={ckpt_val}")

    # ── 加载数据 ──────────────────────────────────────────────────────────
    print("\n=== 加载测试数据 ===")
    data        = build_datasets(cfg)
    test_loader = data["test_loader"]
    test_labels = data["test_labels"]
    test_data   = data["test_data"]

    # ── 推理 ──────────────────────────────────────────────────────────────
    print("\n=== 推理 ===")
    x_pred = run_inference(model, test_loader, device, cfg.TAU)
    T_pred = len(x_pred)
    print(f"预测序列长度：{T_pred:,}")

    x_true = test_data[cfg.CONTEXT_LEN : cfg.CONTEXT_LEN + T_pred]
    y_true = test_labels[cfg.CONTEXT_LEN : cfg.CONTEXT_LEN + T_pred]

    # ── 残差计算 & 动态阈值检测 ────────────────────────────────────────────
    print("\n=== 异常检测 ===")
    raw_residuals = np.abs(x_true - x_pred).max(axis=1).astype(np.float32)
    raw_smoothed  = smooth_residuals(raw_residuals, cfg.smooth_window).astype(np.float32)

    anomaly_scores = detect_anomalies(
        x_true=x_true, x_pred=x_pred,
        smooth_window=cfg.smooth_window,
        method=method, pot_alpha=pot_alpha, pot_q0=pot_q0,
        min_peak_z=min_peak_z,
    )
    print(f"  平滑残差范围：[{raw_smoothed.min():.4f}, {raw_smoothed.max():.4f}]")
    print(f"  异常分数范围：[{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")

    # ── 评估（直接用动态阈值二值输出，与论文评估协议一致）──────────────
    print("\n=== 评估 ===")
    from utils.metrics import event_wise_metrics, affiliation_metrics, extract_events

    # detect_anomalies 已做阈值决策：>0 即为检测到的异常
    y_pred = (anomaly_scores > 0).astype(np.int32)
    pred_rate = float(y_pred.mean())
    print(f"  预测异常率：{pred_rate*100:.3f}%  真实异常率：{y_true.mean()*100:.3f}%")

    # ── 标准1：全部事件（含单点事件）────────────────────────────────────
    ew = event_wise_metrics(y_true, y_pred)
    af = affiliation_metrics(y_true, y_pred)
    best_thresh = float(raw_smoothed[y_pred == 1].min()) if y_pred.any() else 0.0

    n_events_all = len(extract_events(y_true))
    print(f"\n─── 标准1：全部 {n_events_all} 个事件（含单点标注）───")
    print(f"  Event-wise  P={ew['precision']:.4f}  R={ew['recall']:.4f}  F0.5={ew['f0.5']:.4f}")
    print(f"  Affiliation P={af['precision']:.4f}  R={af['recall']:.4f}  F0.5={af['f0.5']:.4f}")

    # ── 标准2：过滤单点事件（duration≥2，与 TimeEval/论文协议一致）──────
    y_true_filt = np.zeros_like(y_true)
    for s, e in extract_events(y_true):
        if e - s + 1 >= 2:
            y_true_filt[s:e+1] = 1
    ew2 = event_wise_metrics(y_true_filt, y_pred)
    af2 = affiliation_metrics(y_true_filt, y_pred)

    n_events_filt = len(extract_events(y_true_filt))
    print(f"\n─── 标准2：{n_events_filt} 个事件（duration≥2，与论文协议一致）───")
    print(f"  Event-wise  P={ew2['precision']:.4f}  R={ew2['recall']:.4f}  "
          f"F0.5={ew2['f0.5']:.4f}  (论文目标: 0.917)")
    print(f"  Affiliation P={af2['precision']:.4f}  R={af2['recall']:.4f}  "
          f"F0.5={af2['f0.5']:.4f}  (论文目标: 0.892)")

    # ── 保存 ──────────────────────────────────────────────────────────────
    metrics = {
        # 标准1：全部事件
        "event_wise":  {"precision": float(ew["precision"]),
                        "recall":    float(ew["recall"]),
                        "f0.5":      float(ew["f0.5"]),
                        "n_events":  n_events_all},
        "affiliation": {"precision": float(af["precision"]),
                        "recall":    float(af["recall"]),
                        "f0.5":      float(af["f0.5"])},
        # 标准2：过滤单点（与论文对齐）
        "event_wise_filt":  {"precision": float(ew2["precision"]),
                             "recall":    float(ew2["recall"]),
                             "f0.5":      float(ew2["f0.5"]),
                             "n_events":  n_events_filt},
        "affiliation_filt": {"precision": float(af2["precision"]),
                             "recall":    float(af2["recall"]),
                             "f0.5":      float(af2["f0.5"])},
        "threshold":         float(best_thresh),
        "pred_anomaly_rate": float(pred_rate),
        "target_event_f05":  0.917,
        "target_affil_f05":  0.892,
    }
    info = {
        "ckpt_path":  ckpt_path,
        "ckpt_epoch": ckpt_epoch,
        "ckpt_run":   ckpt_run,
        "ckpt_val_loss": str(ckpt_val),
        "eval_time":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_test_steps": T_pred,
        "anomaly_rate": float(y_true.mean()),
        "threshold_method": method,
        "pot_alpha":  pot_alpha,
        "pot_q0":     pot_q0,
    }

    result_path = eval_mgr.save_results(metrics, info)
    score_path  = eval_mgr.save_scores(anomaly_scores)
    # raw_smoothed 也保存，方便后续诊断和阈值调试
    np.save(eval_mgr.eval_dir / "raw_smoothed.npy", raw_smoothed)
    print(f"\n  → 指标：{result_path}")
    print(f"  → 分数：{score_path}")
    print(f"  → 原始残差：{eval_mgr.eval_dir}/raw_smoothed.npy")

    # ── 绘图 ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        print("\n=== 绘图 ===")
        plot_results(
            y_true=y_true,
            raw_smoothed=raw_smoothed,
            anomaly_scores=anomaly_scores,
            x_true=x_true,
            x_pred=x_pred,
            threshold=best_thresh,
            eval_dir=eval_mgr.eval_dir,
            n_channels=cfg.NUM_CHANNELS,
        )

    # ── 收尾 ──────────────────────────────────────────────────────────────
    eval_mgr.finalize(metrics, info)

    print(f"\n{'='*45}")
    print(f"评估完成！结果目录：{eval_mgr.eval_dir}")
    print(f"  Event-wise  F0.5 = {ew['f0.5']:.4f}  (目标 0.917)")
    print(f"  Affiliation F0.5 = {af['f0.5']:.4f}  (目标 0.892)")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
