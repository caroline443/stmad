"""
PSTG-FAM 评估脚本

与 evaluate.py 完全一致的评估协议（detect_anomalies + 双标准）。
唯一区别：加载 PSTG_FAM 模型而不是 PSTG。

用法：
  python evaluate_fam.py \\
    --data_dir /root/autodl-tmp/data/ESA-Mission1 \\
    --ckpt checkpoints_fam/best.pt
"""

import os
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config_fam import ConfigFAM
from data.dataset import build_datasets
from models.pstg_fam import PSTG_FAM
from anomaly.detector import detect_anomalies, smooth_residuals
from utils.metrics import (event_wise_metrics, affiliation_metrics,
                            extract_events, find_best_threshold)
from evaluate import EvalManager, plot_results   # 复用 PSTG 的公共工具


@torch.no_grad()
def run_inference(model, test_loader, device: str, tau: int) -> np.ndarray:
    model.eval()
    all_preds = []
    for context, _ in tqdm(test_loader, desc="  推理"):
        context  = context.to(device, non_blocking=True)
        pred     = model(context)
        pred_tau = pred[:, :, :tau].permute(0, 2, 1).reshape(-1, pred.shape[1])
        all_preds.append(pred_tau.cpu().numpy())
    return np.concatenate(all_preds, axis=0).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="PSTG-FAM 评估")
    p.add_argument("--ckpt",     type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--device",   type=str, default=None)
    p.add_argument("--output",   type=str, default=None)
    p.add_argument("--no_plot",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = ConfigFAM()

    if args.data_dir: cfg.DATA_DIR   = args.data_dir
    if args.device:   cfg.DEVICE     = args.device
    output_dir = args.output or cfg.OUTPUT_DIR

    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")

    eval_mgr = EvalManager(output_dir)

    # ── 加载 checkpoint ───────────────────────────────────────────────────
    ckpt_path = args.ckpt or str(Path(cfg.CHECKPOINT_DIR) / "best.pt")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint 不存在：{ckpt_path}")

    print(f"\n加载 checkpoint：{ckpt_path}")
    ckpt     = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("config", {})

    model = PSTG_FAM(
        patch_sizes=  ckpt_cfg.get("patch_sizes",  cfg.PATCH_SIZES),
        d_model=      ckpt_cfg.get("d_model",       cfg.D_MODEL),
        num_heads=    ckpt_cfg.get("num_heads",     cfg.NUM_HEADS),
        num_layers=   ckpt_cfg.get("num_layers",    cfg.NUM_LAYERS),
        n_channels=   ckpt_cfg.get("n_channels",    cfg.NUM_CHANNELS),
        context_len=  ckpt_cfg.get("context_len",   cfg.CONTEXT_LEN),
        forecast_len= ckpt_cfg.get("forecast_len",  cfg.FORECAST_LEN),
        top_k=cfg.top_k, dropout=0.0,
        top_k_rate=ckpt.get("top_k_rate", cfg.FAM_TOP_K_RATE),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_val   = ckpt.get("val_loss", "?")
    print(f"  epoch={ckpt_epoch}  val_loss={ckpt_val}  "
          f"FAM top_k_rate={ckpt.get('top_k_rate', cfg.FAM_TOP_K_RATE)}")

    # ── 数据 & 推理 ───────────────────────────────────────────────────────
    print("\n=== 加载测试数据 ===")
    data        = build_datasets(cfg)
    test_loader = data["test_loader"]
    test_labels = data["test_labels"]
    test_data   = data["test_data"]

    print("\n=== 推理 ===")
    x_pred = run_inference(model, test_loader, device, cfg.TAU)
    T_pred = len(x_pred)
    print(f"预测序列长度：{T_pred:,}")

    x_true = test_data[cfg.CONTEXT_LEN : cfg.CONTEXT_LEN + T_pred]
    y_true = test_labels[cfg.CONTEXT_LEN : cfg.CONTEXT_LEN + T_pred]

    # ── 异常检测（与 PSTG 完全一致的协议）───────────────────────────────
    print("\n=== 异常检测 ===")
    raw_residuals = np.abs(x_true - x_pred).max(axis=1).astype(np.float32)
    raw_smoothed  = smooth_residuals(raw_residuals, cfg.smooth_window).astype(np.float32)

    anomaly_scores = detect_anomalies(
        x_true=x_true, x_pred=x_pred,
        smooth_window=cfg.smooth_window,
        p_tfi=cfg.P_TFI, n_candidates=300,
    )
    print(f"  平滑残差范围：[{raw_smoothed.min():.4f}, {raw_smoothed.max():.4f}]")
    print(f"  异常分数范围：[{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")

    # ── 评估（双标准）────────────────────────────────────────────────────
    print("\n=== 评估 ===")
    y_pred = (anomaly_scores > 0).astype(np.int32)
    pred_rate = float(y_pred.mean())
    print(f"  预测异常率：{pred_rate*100:.3f}%  真实异常率：{y_true.mean()*100:.3f}%")

    ew = event_wise_metrics(y_true, y_pred)
    af = affiliation_metrics(y_true, y_pred)
    n_events_all = len(extract_events(y_true))

    # 标准2：过滤单点
    y_true_filt = np.zeros_like(y_true)
    for s, e in extract_events(y_true):
        if e - s + 1 >= 2:
            y_true_filt[s:e+1] = 1
    ew2 = event_wise_metrics(y_true_filt, y_pred)
    af2 = affiliation_metrics(y_true_filt, y_pred)
    n_events_filt = len(extract_events(y_true_filt))

    print(f"\n─── 标准1：全部 {n_events_all} 个事件 ───")
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
        "threshold":         float(raw_smoothed[y_pred==1].min()) if y_pred.any() else 0.0,
        "pred_anomaly_rate": float(pred_rate),
        "pstg_baseline":     {"event_f05": 0.921, "affil_f05": 0.741},
    }
    info = {
        "ckpt_path":   ckpt_path,
        "ckpt_epoch":  ckpt_epoch,
        "model_type":  "PSTG_FAM",
        "top_k_rate":  float(ckpt.get("top_k_rate", cfg.FAM_TOP_K_RATE)),
        "eval_time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    result_path = eval_mgr.save_results(metrics, info)
    np.save(eval_mgr.eval_dir / "anomaly_scores.npy", anomaly_scores)
    np.save(eval_mgr.eval_dir / "raw_smoothed.npy",   raw_smoothed)
    print(f"\n  → 结果：{result_path}")

    # ── 绘图 ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        thresh = float(raw_smoothed[y_pred==1].min()) if y_pred.any() else 0.0
        plot_results(
            y_true=y_true, raw_smoothed=raw_smoothed,
            anomaly_scores=anomaly_scores, x_true=x_true, x_pred=x_pred,
            threshold=thresh, eval_dir=eval_mgr.eval_dir, n_channels=cfg.NUM_CHANNELS,
        )

    eval_mgr.finalize(metrics, info)

    print(f"\n{'='*50}")
    print(f"PSTG-FAM 评估完成！")
    print(f"  [标准1 全部事件] Event F0.5={ew['f0.5']:.4f}  Affil F0.5={af['f0.5']:.4f}")
    print(f"  [标准2 过滤单点] Event F0.5={ew2['f0.5']:.4f}  Affil F0.5={af2['f0.5']:.4f}")
    print(f"  PSTG 复现基线:   Event F0.5=0.9211  Affil F0.5=0.7410")
    gain_e = ew2['f0.5'] - 0.9211
    gain_a = af2['f0.5'] - 0.7410
    print(f"  相比 PSTG 复现:  Event {gain_e:+.4f}，Affil {gain_a:+.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
