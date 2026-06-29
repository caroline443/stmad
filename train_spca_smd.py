"""
SpCA 在 SMD 数据集上的训练与评估

用法：
  python train_spca_smd.py \
    --data_dir /root/autodl-tmp/data/AT/SMD

  # 评估
  python train_spca_smd.py \
    --data_dir /root/autodl-tmp/data/AT/SMD \
    --eval_only --ckpt checkpoints_spca_smd/best.pt
"""

import argparse, time, json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config_spca import ConfigSpCA
from data.smd_dataset import build_datasets_smd
from models.spca import SpCA
from utils.loss import PSTGLoss
from anomaly.detector import detect_anomalies, smooth_residuals
from train import CheckpointManager, set_seed


# ── SMD 专用配置 ──────────────────────────────────────────────────────────────

class ConfigSMD(ConfigSpCA):
    DATA_DIR       = "/root/autodl-tmp/data/AT/SMD"
    NUM_CHANNELS   = 38
    CHANNELS       = list(range(38))
    BATCH_SIZE     = 64
    B_S            = 70
    TRAIN_STRIDE   = 50
    CHECKPOINT_DIR = "checkpoints_spca_smd"
    OUTPUT_DIR     = "outputs_spca_smd"
    DATASET_NAME   = "smd"


# ── 评估（F1 with PA，文献标准）──────────────────────────────────────────────

def point_adjust(y_true, y_pred):
    """Point Adjustment：若异常段内有任意预测为 1，则整段都标为 1"""
    y_adj = y_pred.copy()
    in_seg = False
    seg_has_det = False
    seg_start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_seg:
            in_seg, seg_has_det, seg_start = True, False, i
        if in_seg and y_pred[i] == 1:
            seg_has_det = True
        if (y_true[i] == 0 or i == len(y_true) - 1) and in_seg:
            if seg_has_det:
                y_adj[seg_start: i + 1] = 1
            in_seg = False
    return y_adj


def compute_f1(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return {"precision": p, "recall": r, "f1": f1}


def find_best_f1_threshold(y_true, scores, n=200):
    """在 raw_smoothed 上搜索最大化 F1(PA) 的阈值"""
    best_f1, best_t = 0, 0
    for t in np.linspace(scores.min(), scores.max(), n):
        y_pred = (scores >= t).astype(np.int32)
        y_adj  = point_adjust(y_true, y_pred)
        m      = compute_f1(y_true, y_adj)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], t
    return best_t, best_f1


# ── 训练 ──────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total, n = 0.0, 0
    for ctx, fut in tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False):
        ctx, fut = ctx.to(device), fut.to(device)
        loss, _ = criterion(model(ctx), fut)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); n += 1
    return total / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device, epoch):
    model.eval()
    total, n = 0.0, 0
    for ctx, fut in tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False):
        ctx, fut = ctx.to(device), fut.to(device)
        loss, _ = criterion(model(ctx), fut)
        total += loss.item(); n += 1
    return total / n


# ── 推理与评估 ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(model, cfg, data, pot_alpha=4e-3, min_peak_z=1.5):
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    for ctx, _ in tqdm(data["test_loader"], desc="  推理", leave=False):
        ctx  = ctx.to(device)
        pred = model(ctx)[:, :, :1].permute(0, 2, 1).reshape(-1, cfg.NUM_CHANNELS)
        all_preds.append(pred.cpu().numpy())
    x_pred = np.concatenate(all_preds, 0).astype(np.float32)

    x_true = data["test_data"][cfg.CONTEXT_LEN: cfg.CONTEXT_LEN + len(x_pred)]
    y_true = data["test_labels"][cfg.CONTEXT_LEN: cfg.CONTEXT_LEN + len(x_pred)].astype(np.int32)

    # 残差 → 平滑 → POT
    raw_smoothed = smooth_residuals(
        np.abs(x_true - x_pred).max(axis=1), cfg.smooth_window
    ).astype(np.float32)

    anomaly_scores = detect_anomalies(
        x_true=x_true, x_pred=x_pred,
        smooth_window=cfg.smooth_window,
        method="pot", pot_alpha=pot_alpha, min_peak_z=min_peak_z,
    )
    y_pred = (anomaly_scores > 0).astype(np.int32)

    # 无 PA
    m_nopa = compute_f1(y_true, y_pred)
    # 有 PA
    y_adj = point_adjust(y_true, y_pred)
    m_pa  = compute_f1(y_true, y_adj)
    # 最优阈值
    best_t, best_f1_pa = find_best_f1_threshold(y_true, raw_smoothed)

    pred_rate = float(y_pred.mean())
    true_rate = float(y_true.mean())

    print(f"\n  预测异常率={pred_rate*100:.3f}%  真实={true_rate*100:.3f}%")
    print(f"\n  === ContrastAD 标准（F1 with PA）===")
    print(f"  F1={m_pa['f1']:.4f}  P={m_pa['precision']:.4f}  R={m_pa['recall']:.4f}")
    print(f"  最优PA → F1={best_f1_pa:.4f}  (阈值={best_t:.4f})")
    print(f"\n  === 严格逐点（无 PA）===")
    print(f"  F1={m_nopa['f1']:.4f}  P={m_nopa['precision']:.4f}  R={m_nopa['recall']:.4f}")

    return {"with_pa": m_pa, "no_pa": m_nopa,
            "best_f1_pa": best_f1_pa, "best_thresh": float(best_t),
            "pred_rate": pred_rate, "true_rate": true_rate}


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--device",     type=str, default=None)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--temporal",   action="store_true")
    p.add_argument("--eval_only",  action="store_true", help="跳过训练，直接评估")
    p.add_argument("--ckpt",       type=str, default=None)
    p.add_argument("--pot_alpha",  type=float, default=4e-3)
    p.add_argument("--min_peak_z", type=float, default=1.5)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = ConfigSMD()
    if args.data_dir: cfg.DATA_DIR   = args.data_dir
    if args.epochs:   cfg.NUM_EPOCHS = args.epochs
    if args.device:   cfg.DEVICE     = args.device
    if args.temporal: cfg.N_PATCHES  = 10

    set_seed(cfg.SEED)
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"SMD  通道={cfg.NUM_CHANNELS}  设备={device}")

    print("\n=== 加载 SMD 数据 ===")
    data = build_datasets_smd(cfg)

    model     = SpCA.from_config(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_MAX, eta_min=cfg.ETA_MIN)
    criterion = PSTGLoss(lambda1=cfg.LAMBDA1, lambda2=cfg.LAMBDA2)
    print(f"SpCA 参数量：{model.count_parameters():,}")

    if args.eval_only:
        ckpt_path = args.ckpt or f"{cfg.CHECKPOINT_DIR}/best.pt"
        ckpt      = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"\n加载 checkpoint: {ckpt_path}  epoch={ckpt.get('epoch','?')}")
        print("\n=== 评估 ===")
        run_eval(model, cfg, data, args.pot_alpha, args.min_peak_z)
        return

    ckpt_mgr = CheckpointManager(cfg.CHECKPOINT_DIR, save_every=args.save_every)

    print(f"\n=== 开始训练 ({cfg.NUM_EPOCHS} 轮) ===")
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()
        tr = train_epoch(model, data["train_loader"], optimizer, criterion, device, epoch)
        vl = val_epoch(model, data["val_loader"],   criterion, device, epoch)
        scheduler.step()
        lr  = scheduler.get_last_lr()[0]
        tag = " ★" if vl < ckpt_mgr.global_best_val_loss else ""
        print(f"Epoch {epoch:03d}/{cfg.NUM_EPOCHS}  train={tr:.4f}  val={vl:.4f}  "
              f"lr={lr:.2e}  t={time.time()-t0:.1f}s{tag}")
        ckpt_mgr.save(model, optimizer, scheduler, epoch, vl, cfg)
        ckpt_mgr.log(epoch, tr, vl, lr, time.time() - t0)

    ckpt_mgr.finalize()
    print("\n=== 评估（最优 checkpoint）===")
    best = torch.load(f"{cfg.CHECKPOINT_DIR}/best.pt", map_location=device)
    model.load_state_dict(best["model"])
    run_eval(model, cfg, data, args.pot_alpha, args.min_peak_z)
    print(f"\n下次单独评估：python train_spca_smd.py --eval_only "
          f"--ckpt {cfg.CHECKPOINT_DIR}/best.pt --pot_alpha {args.pot_alpha}")


if __name__ == "__main__":
    main()
