"""
PSTG-FAM 训练脚本

用法：
  # 热启动（推荐）：直接从 PSTG 70 轮 checkpoint 开始，几轮即可收敛
  python train_fam.py \\
    --data_dir /root/autodl-tmp/data/ESA-Mission1 \\
    --pstg_ckpt checkpoints/best.pt \\
    --epochs 30 \\
    --train_stride 50

  # 续训
  python train_fam.py \\
    --data_dir /root/autodl-tmp/data/ESA-Mission1 \\
    --resume checkpoints_fam/last.pt \\
    --epochs 30

评估：
  python evaluate_fam.py --data_dir ... --ckpt checkpoints_fam/best.pt
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm

from config_fam import ConfigFAM
from data.dataset import build_datasets
from models.pstg_fam import PSTG_FAM
from utils.loss import PSTGLoss
from train import CheckpointManager, set_seed


# ── 训练/验证（与 train.py 完全一样的逻辑）─────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, log_interval, epoch):
    model.train()
    totals = {"loss": 0, "mse": 0, "freq": 0, "shape": 0}
    n = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for batch_idx, (context, future) in enumerate(pbar):
        context = context.to(device, non_blocking=True)
        future  = future.to(device,  non_blocking=True)
        pred = model(context)
        loss, (mse, freq, shape) = criterion(pred, future)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        totals["loss"] += loss.item()
        totals["mse"]  += mse
        n += 1
        if batch_idx % log_interval == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "mse": f"{mse:.4f}"})
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = total_mse = 0.0
    n = 0
    for context, future in tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False):
        context = context.to(device, non_blocking=True)
        future  = future.to(device,  non_blocking=True)
        pred = model(context)
        loss, (mse, _, _) = criterion(pred, future)
        total_loss += loss.item()
        total_mse  += mse
        n += 1
    return {"loss": total_loss / n, "mse": total_mse / n}


# ── 参数解析 ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PSTG-FAM 训练")
    p.add_argument("--data_dir",     type=str,   default=None)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--train_stride", type=int,   default=None)
    p.add_argument("--device",       type=str,   default=None)
    p.add_argument("--save_every",   type=int,   default=5)
    p.add_argument("--top_k_rate",   type=float, default=None,
                   help="FAM 保留频率分量比例（默认 0.5）")
    p.add_argument("--pstg_ckpt",    type=str,   default=None,
                   help="PSTG checkpoint 路径（热启动，推荐）")
    p.add_argument("--resume",       type=str,   default=None,
                   help="从 PSTG-FAM checkpoint 续训")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = ConfigFAM()

    if args.data_dir:    cfg.DATA_DIR      = args.data_dir
    if args.epochs:      cfg.NUM_EPOCHS    = args.epochs
    if args.train_stride: cfg.TRAIN_STRIDE = args.train_stride
    if args.device:      cfg.DEVICE        = args.device
    if args.top_k_rate is not None: cfg.FAM_TOP_K_RATE = args.top_k_rate

    set_seed(cfg.SEED)
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")
    print(f"FAM top_k_rate = {cfg.FAM_TOP_K_RATE}（保留前 {cfg.FAM_TOP_K_RATE*100:.0f}% 能量最大频率）")

    # ── 数据 ──────────────────────────────────────────────────────────────
    print("\n=== 构建数据集 ===")
    data         = build_datasets(cfg)
    train_loader = data["train_loader"]
    val_loader   = data["val_loader"]

    # ── 模型 ──────────────────────────────────────────────────────────────
    print("\n=== 构建 PSTG-FAM 模型 ===")
    if args.pstg_ckpt and Path(args.pstg_ckpt).exists():
        model = PSTG_FAM.from_pstg_checkpoint(args.pstg_ckpt, cfg, device)
        print("FAM 新增参数：0（纯 FFT，无可学习参数）")
    else:
        model = PSTG_FAM.from_config(cfg).to(device)
        print("从头初始化")

    print(f"总参数量：{model.count_parameters():,}（与 PSTG 相同）")

    # ── 优化器 ────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_MAX, eta_min=cfg.ETA_MIN)
    criterion = PSTGLoss(lambda1=cfg.LAMBDA1, lambda2=cfg.LAMBDA2)

    # ── 续训 ──────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            resume_path = Path(cfg.CHECKPOINT_DIR) / "last.pt"
        if resume_path.exists():
            print(f"\n续训自：{resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            print(f"从 epoch {start_epoch} 继续")

    # ── Checkpoint 管理器 ─────────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(cfg.CHECKPOINT_DIR, save_every=args.save_every)

    # ── 训练 ──────────────────────────────────────────────────────────────
    print(f"\n=== 开始训练 PSTG-FAM（epoch {start_epoch} → {cfg.NUM_EPOCHS}）===\n")

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, criterion,
                                  device, cfg.LOG_INTERVAL, epoch)
        val_m   = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        lr_cur  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        is_run_best    = val_m["loss"] < ckpt_mgr.run_best_val_loss
        is_global_best = val_m["loss"] < ckpt_mgr.global_best_val_loss
        flag = " ★" if is_global_best else (" ✓" if is_run_best else "")

        print(f"Epoch {epoch:03d}/{cfg.NUM_EPOCHS}  "
              f"train={train_m['loss']:.4f}  val={val_m['loss']:.4f}  "
              f"lr={lr_cur:.2e}  t={elapsed:.1f}s{flag}")

        extra = {"model_type": "PSTG_FAM", "top_k_rate": cfg.FAM_TOP_K_RATE}
        saved = ckpt_mgr.save(model, optimizer, scheduler, epoch, val_m["loss"], cfg, extra)
        for p in saved:
            print(f"  → {p}")

        ckpt_mgr.log(epoch, train_m["loss"], val_m["loss"], lr_cur, elapsed)

    ckpt_mgr.finalize()

    print(f"\n{'='*50}")
    print(f"PSTG-FAM 训练完成！")
    print(f"全局最优 val_loss = {ckpt_mgr.global_best_val_loss:.6f}")
    print(f"checkpoint = {cfg.CHECKPOINT_DIR}/best.pt")
    print(f"\n评估：")
    print(f"  python evaluate_fam.py --data_dir {cfg.DATA_DIR} --ckpt {cfg.CHECKPOINT_DIR}/best.pt")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
