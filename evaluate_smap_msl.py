"""
SMAP / MSL 评估脚本

同时报告两种评估协议：
  - 无 PA（与本项目 ESA-AD 评估一致，严格）
  - 有 PA（Point Adjustment，业界通行，便于与文献比较）

用法：
  python evaluate_smap_msl.py --dataset msl \\
    --data_dir /root/autodl-tmp/data/AT/MSL \\
    --ckpt checkpoints_msl/best.pt
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config_smap_msl import ConfigSMAP, ConfigMSL
from data.smap_msl_dataset import build_datasets_smap_msl, point_adjust, compute_f1
from models.pstg import PSTG
from models.spca import SpCA
from anomaly.detector import detect_anomalies, smooth_residuals
from utils.metrics import find_best_threshold
from evaluate import EvalManager


@torch.no_grad()
def run_inference(model, test_loader, device, tau):
    model.eval()
    all_preds = []
    for context, _ in tqdm(test_loader, desc="  推理"):
        context  = context.to(device, non_blocking=True)
        pred     = model(context)
        pred_tau = pred[:, :, :tau].permute(0, 2, 1).reshape(-1, pred.shape[1])
        all_preds.append(pred_tau.cpu().numpy())
    return np.concatenate(all_preds).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  type=str, default="msl", choices=["smap","msl"])
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--ckpt",     type=str, default=None)
    p.add_argument("--device",   type=str, default=None)
    p.add_argument("--output",   type=str, default=None)
    p.add_argument("--no_plot",  action="store_true")
    p.add_argument("--model",    type=str, default="pstg",
                   choices=["pstg", "spca"], help="使用哪个模型（默认 pstg）")
    p.add_argument("--pot_alpha",  type=float, default=4e-3)
    p.add_argument("--min_peak_z", type=float, default=1.5)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = ConfigSMAP() if args.dataset == "smap" else ConfigMSL()

    if args.data_dir: cfg.DATA_DIR = args.data_dir
    if args.device:   cfg.DEVICE   = args.device
    output_dir = args.output or cfg.OUTPUT_DIR

    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"数据集: {cfg.DATASET_NAME.upper()}  设备: {device}")

    eval_mgr = EvalManager(output_dir)

    # ── 加载 checkpoint ───────────────────────────────────────────────────
    ckpt_path = args.ckpt or str(Path(cfg.CHECKPOINT_DIR) / "best.pt")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint 不存在：{ckpt_path}")

    print(f"\n加载 checkpoint：{ckpt_path}")
    ckpt     = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("config", {})

    if args.model == "spca":
        from config_spca import ConfigSpCA
        spca_cfg = ConfigSpCA()
        model = SpCA(
            n_channels      = ckpt_cfg.get("n_channels",      cfg.NUM_CHANNELS),
            context_len     = ckpt_cfg.get("context_len",     cfg.CONTEXT_LEN),
            forecast_len    = ckpt_cfg.get("forecast_len",    cfg.FORECAST_LEN),
            d_model         = ckpt_cfg.get("d_model",         spca_cfg.D_MODEL),
            n_heads         = ckpt_cfg.get("n_heads",         spca_cfg.NUM_HEADS),
            n_bands         = ckpt_cfg.get("n_bands",         spca_cfg.N_BANDS),
            band_splits     = ckpt_cfg.get("band_splits",     spca_cfg.BAND_SPLITS),
            # n_patches=0 → v1 BandProjection；checkpoint 未保存时默认 0（向后兼容）
            n_patches       = ckpt_cfg.get("n_patches",       0),
            n_layers_band   = ckpt_cfg.get("n_layers_band",   spca_cfg.N_LAYERS_BAND),
            n_layers_global = ckpt_cfg.get("n_layers_global", spca_cfg.N_LAYERS_GLOBAL),
            dropout=0.0,
        ).to(device)
    else:
        model = PSTG(
            patch_sizes=  ckpt_cfg.get("patch_sizes",  cfg.PATCH_SIZES),
            d_model=      ckpt_cfg.get("d_model",       cfg.D_MODEL),
            num_heads=    ckpt_cfg.get("num_heads",     cfg.NUM_HEADS),
            num_layers=   ckpt_cfg.get("num_layers",    cfg.NUM_LAYERS),
            n_channels=   ckpt_cfg.get("n_channels",    cfg.NUM_CHANNELS),
            context_len=  ckpt_cfg.get("context_len",   cfg.CONTEXT_LEN),
            forecast_len= ckpt_cfg.get("forecast_len",  cfg.FORECAST_LEN),
            top_k=cfg.top_k, dropout=0.0,
        ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss','?')}")

    # ── 数据 & 推理 ───────────────────────────────────────────────────────
    data        = build_datasets_smap_msl(cfg)
    test_loader = data["test_loader"]
    test_labels = data["test_labels"]
    test_data   = data["test_data"]

    print("\n=== 推理 ===")
    x_pred = run_inference(model, test_loader, device, cfg.TAU)
    T_pred = len(x_pred)
    x_true = test_data[cfg.CONTEXT_LEN : cfg.CONTEXT_LEN + T_pred]
    y_true = test_labels[cfg.CONTEXT_LEN : cfg.CONTEXT_LEN + T_pred]

    # ── 异常检测 ──────────────────────────────────────────────────────────
    print("\n=== 异常检测 ===")
    raw_residuals = np.abs(x_true - x_pred).max(axis=1).astype(np.float32)
    raw_smoothed  = smooth_residuals(raw_residuals, cfg.smooth_window).astype(np.float32)

    anomaly_scores = detect_anomalies(
        x_true=x_true, x_pred=x_pred,
        smooth_window=cfg.smooth_window,
        method="pot",
        pot_alpha=args.pot_alpha,
        min_peak_z=args.min_peak_z,
    )
    y_pred_bin = (anomaly_scores > 0).astype(np.int32)
    print(f"  残差范围：[{raw_smoothed.min():.4f}, {raw_smoothed.max():.4f}]")
    print(f"  预测异常率：{y_pred_bin.mean()*100:.3f}%  真实异常率：{y_true.mean()*100:.3f}%")

    # ── 评估：不用 PA（逐点 Precision/Recall/F1）────────────────────────
    print("\n─── 无 PA（严格逐点，与 ESA-AD 一致）───")
    m_nopa = compute_f1(y_true, y_pred_bin)
    print(f"  P={m_nopa['precision']:.4f}  R={m_nopa['recall']:.4f}  F1={m_nopa['f1']:.4f}")

    # ── 评估：用 PA（业界通行，便于文献比较）────────────────────────────
    print("\n─── 有 PA（Point Adjustment，文献通行）───")
    y_pred_pa  = point_adjust(y_true, y_pred_bin)
    m_pa       = compute_f1(y_true, y_pred_pa)
    print(f"  P={m_pa['precision']:.4f}  R={m_pa['recall']:.4f}  F1={m_pa['f1']:.4f}")

    # ── 用 find_best_threshold 搜索最优阈值（参考）──────────────────────
    print("\n─── 最优阈值（非 PA，搜索 raw_smoothed）───")
    best_thresh, best_result = find_best_threshold(
        y_true, raw_smoothed, metric="event_f05", n_thresholds=200
    )
    y_pred_opt = (raw_smoothed > best_thresh).astype(np.int32)
    m_opt      = compute_f1(y_true, y_pred_opt)
    m_opt_pa   = compute_f1(y_true, point_adjust(y_true, y_pred_opt))
    print(f"  阈值={best_thresh:.4f}  无PA: F1={m_opt['f1']:.4f}  有PA: F1={m_opt_pa['f1']:.4f}")

    # ── 保存 ──────────────────────────────────────────────────────────────
    metrics = {
        "dataset":  cfg.DATASET_NAME.upper(),
        "no_pa":    m_nopa,
        "with_pa":  m_pa,
        "opt_no_pa": m_opt,
        "opt_pa":    m_opt_pa,
        "best_thresh": float(best_thresh),
        "pred_anomaly_rate": float(y_pred_bin.mean()),
        "true_anomaly_rate": float(y_true.mean()),
    }
    info = {
        "ckpt_path":  ckpt_path,
        "ckpt_epoch": ckpt.get("epoch", "?"),
        "model_type": args.model.upper(),
        "dataset":    cfg.DATASET_NAME,
        "eval_time":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    eval_mgr.save_results(metrics, info)
    np.save(eval_mgr.eval_dir / "anomaly_scores.npy", anomaly_scores)
    np.save(eval_mgr.eval_dir / "raw_smoothed.npy",   raw_smoothed)

    # SMAP/MSL 使用独立的 summary（不调用 ESA-AD 的 finalize，格式不同）
    summary_path = eval_mgr.base_dir / "eval_summary.json"
    history = json.loads(summary_path.read_text()) if summary_path.exists() else []
    history.append({
        "eval_name":    eval_mgr.eval_name,
        "finished_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":        args.model,
        "dataset":      cfg.DATASET_NAME,
        "f1_no_pa":     m_nopa["f1"],
        "f1_with_pa":   m_pa["f1"],
        "f1_opt_pa":    m_opt_pa["f1"],
        "precision_pa": m_pa["precision"],
        "recall_pa":    m_pa["recall"],
    })
    summary_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))

    model_name = args.model.upper()
    ds_name    = cfg.DATASET_NAME.upper()
    print(f"\n{'='*55}")
    print(f"{model_name} on {ds_name} 评估完成！结果目录：{eval_mgr.eval_dir}")
    print(f"\n  === ContrastAD 标准（F1 with PA）===")
    print(f"  有  PA → F1={m_pa['f1']:.4f}  P={m_pa['precision']:.4f}  R={m_pa['recall']:.4f}")
    print(f"  最优PA → F1={m_opt_pa['f1']:.4f}  (阈值={best_thresh:.4f})")
    print(f"\n  === 严格逐点（无 PA）===")
    print(f"  无  PA → F1={m_nopa['f1']:.4f}  P={m_nopa['precision']:.4f}  R={m_nopa['recall']:.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
