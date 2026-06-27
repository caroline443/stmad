"""
PSTG-Mamba 评估脚本（与 PSTG evaluate.py 协议完全一致）

用法：
  python evaluate_mamba.py \\
    --data_dir /root/autodl-tmp/data/ESA-Mission1 \\
    --ckpt checkpoints_mamba/best.pt
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config_mamba import ConfigMamba
from data.dataset import build_datasets
from models.pstg_mamba import PSTG_Mamba
from anomaly.detector import detect_anomalies, smooth_residuals
from utils.metrics import event_wise_metrics, affiliation_metrics, extract_events
from evaluate import EvalManager, plot_results


@torch.no_grad()
def run_inference(model, loader, device, tau):
    model.eval()
    all_preds = []
    for context, _ in tqdm(loader, desc="  推理"):
        pred     = model(context.to(device, non_blocking=True))
        pred_tau = pred[:, :, :tau].permute(0, 2, 1).reshape(-1, pred.shape[1])
        all_preds.append(pred_tau.cpu().numpy())
    return np.concatenate(all_preds).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      type=str,   default=None)
    p.add_argument("--data_dir",  type=str,   default=None)
    p.add_argument("--device",    type=str,   default=None)
    p.add_argument("--output",    type=str,   default=None)
    p.add_argument("--no_plot",   action="store_true")
    p.add_argument("--method",    type=str,   default="pot",
                   choices=["pot", "robust"])
    p.add_argument("--pot_alpha", type=float, default=4e-3)
    p.add_argument("--pot_q0",    type=float, default=0.98)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = ConfigMamba()
    if args.data_dir: cfg.DATA_DIR   = args.data_dir
    if args.device:   cfg.DEVICE     = args.device
    output_dir = args.output or cfg.OUTPUT_DIR

    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    eval_mgr = EvalManager(output_dir)

    ckpt_path = args.ckpt or str(Path(cfg.CHECKPOINT_DIR) / "best.pt")
    print(f"\n加载：{ckpt_path}")
    ckpt     = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("config", {})

    model = PSTG_Mamba(
        patch_sizes=  ckpt_cfg.get("patch_sizes",  cfg.PATCH_SIZES),
        d_model=      ckpt_cfg.get("d_model",       cfg.D_MODEL),
        num_heads=    ckpt_cfg.get("num_heads",     cfg.NUM_HEADS),
        num_layers=   ckpt_cfg.get("num_layers",    cfg.NUM_LAYERS),
        n_channels=   ckpt_cfg.get("n_channels",    cfg.NUM_CHANNELS),
        context_len=  ckpt_cfg.get("context_len",   cfg.CONTEXT_LEN),
        forecast_len= ckpt_cfg.get("forecast_len",  cfg.FORECAST_LEN),
        top_k=cfg.top_k, dropout=0.0,
        d_state=ckpt.get("d_state", cfg.MAMBA_D_STATE),
        d_conv=ckpt.get("d_conv",   cfg.MAMBA_D_CONV),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss','?')}")

    data        = build_datasets(cfg)
    test_loader = data["test_loader"]
    test_labels = data["test_labels"]
    test_data   = data["test_data"]

    print("\n=== 推理 ===")
    x_pred = run_inference(model, test_loader, device, cfg.TAU)
    T_pred = len(x_pred)
    x_true = test_data[cfg.CONTEXT_LEN:cfg.CONTEXT_LEN+T_pred]
    y_true = test_labels[cfg.CONTEXT_LEN:cfg.CONTEXT_LEN+T_pred]

    print("\n=== 异常检测 ===")
    raw_residuals = np.abs(x_true - x_pred).max(axis=1).astype(np.float32)
    raw_smoothed  = smooth_residuals(raw_residuals, cfg.smooth_window).astype(np.float32)
    anomaly_scores = detect_anomalies(
        x_true=x_true, x_pred=x_pred,
        smooth_window=cfg.smooth_window,
        method=args.method, pot_alpha=args.pot_alpha, pot_q0=args.pot_q0,
    )

    y_pred = (anomaly_scores > 0).astype(np.int32)
    pred_rate = float(y_pred.mean())
    print(f"  预测异常率：{pred_rate*100:.3f}%  真实异常率：{y_true.mean()*100:.3f}%")

    print("\n=== 评估 ===")
    ew = event_wise_metrics(y_true, y_pred)
    af = affiliation_metrics(y_true, y_pred)

    y_true_filt = np.zeros_like(y_true)
    for s, e in extract_events(y_true):
        if e - s + 1 >= 2:
            y_true_filt[s:e+1] = 1
    ew2 = event_wise_metrics(y_true_filt, y_pred)
    af2 = affiliation_metrics(y_true_filt, y_pred)
    n_all  = len(extract_events(y_true))
    n_filt = len(extract_events(y_true_filt))

    print(f"\n─── 标准1：全部 {n_all} 个事件 ───")
    print(f"  Event P={ew['precision']:.4f}  R={ew['recall']:.4f}  F0.5={ew['f0.5']:.4f}")
    print(f"  Affil P={af['precision']:.4f}  R={af['recall']:.4f}  F0.5={af['f0.5']:.4f}")

    print(f"\n─── 标准2：{n_filt} 个事件（duration≥2，论文协议）───")
    print(f"  Event P={ew2['precision']:.4f}  R={ew2['recall']:.4f}  "
          f"F0.5={ew2['f0.5']:.4f}  (PSTG复现: 0.921)")
    print(f"  Affil P={af2['precision']:.4f}  R={af2['recall']:.4f}  "
          f"F0.5={af2['f0.5']:.4f}  (PSTG复现: 0.741)")

    metrics = {
        "event_wise":      {"precision": float(ew['precision']),  "recall": float(ew['recall']),  "f0.5": float(ew['f0.5'])},
        "affiliation":     {"precision": float(af['precision']),  "recall": float(af['recall']),  "f0.5": float(af['f0.5'])},
        "event_wise_filt": {"precision": float(ew2['precision']), "recall": float(ew2['recall']), "f0.5": float(ew2['f0.5'])},
        "affiliation_filt":{"precision": float(af2['precision']), "recall": float(af2['recall']), "f0.5": float(af2['f0.5'])},
        "pred_anomaly_rate": float(pred_rate),
        "pstg_baseline":     {"event_f05": 0.921, "affil_f05": 0.741},
    }
    info = {"ckpt_path": ckpt_path, "model_type": "PSTG_Mamba",
            "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    eval_mgr.save_results(metrics, info)
    np.save(eval_mgr.eval_dir / "anomaly_scores.npy", anomaly_scores)
    np.save(eval_mgr.eval_dir / "raw_smoothed.npy",   raw_smoothed)
    eval_mgr.finalize(metrics, info)

    if not args.no_plot:
        thresh = float(raw_smoothed[y_pred==1].min()) if y_pred.any() else 0.0
        plot_results(y_true, raw_smoothed, anomaly_scores, x_true, x_pred,
                     thresh, eval_mgr.eval_dir, cfg.NUM_CHANNELS)

    print(f"\n{'='*50}")
    print(f"PSTG-Mamba 评估完成！")
    print(f"  [标准2] Event F0.5={ew2['f0.5']:.4f}  Affil F0.5={af2['f0.5']:.4f}")
    g_e = ew2['f0.5'] - 0.921; g_a = af2['f0.5'] - 0.741
    print(f"  相比 PSTG：Event {g_e:+.4f}，Affil {g_a:+.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
