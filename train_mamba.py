"""
PSTG-Mamba 训练脚本

用法：
  python train_mamba.py \\
    --data_dir /root/autodl-tmp/data/ESA-Mission1 \\
    --epochs 70 --train_stride 50 --save_every 10
"""

import os, argparse, time
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config_mamba import ConfigMamba
from data.dataset import build_datasets
from models.pstg_mamba import PSTG_Mamba
from utils.loss import PSTGLoss
from train import CheckpointManager, set_seed


def train_one_epoch(model, loader, optimizer, criterion, device, log_interval, epoch):
    model.train()
    total_loss = total_mse = 0.0; n = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for bi, (context, future) in enumerate(pbar):
        context = context.to(device, non_blocking=True)
        future  = future.to(device,  non_blocking=True)
        pred = model(context)
        loss, (mse, _, _) = criterion(pred, future)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item(); total_mse += mse; n += 1
        if bi % log_interval == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "mse": f"{mse:.4f}"})
    return {"loss": total_loss / n, "mse": total_mse / n}


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = total_mse = 0.0; n = 0
    for context, future in tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False):
        context = context.to(device, non_blocking=True)
        future  = future.to(device,  non_blocking=True)
        pred = model(context)
        loss, (mse, _, _) = criterion(pred, future)
        total_loss += loss.item(); total_mse += mse; n += 1
    return {"loss": total_loss / n, "mse": total_mse / n}


def parse_args():
    p = argparse.ArgumentParser(description="PSTG-Mamba 训练")
    p.add_argument("--data_dir",     type=str,   default=None)
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--train_stride", type=int,   default=None)
    p.add_argument("--device",       type=str,   default=None)
    p.add_argument("--save_every",   type=int,   default=10)
    p.add_argument("--d_state",      type=int,   default=None)
    p.add_argument("--resume",       type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = ConfigMamba()
    if args.data_dir:     cfg.DATA_DIR      = args.data_dir
    if args.epochs:       cfg.NUM_EPOCHS    = args.epochs
    if args.train_stride: cfg.TRAIN_STRIDE  = args.train_stride
    if args.device:       cfg.DEVICE        = args.device
    if args.d_state:      cfg.MAMBA_D_STATE = args.d_state

    set_seed(cfg.SEED)
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"PSTG-Mamba  设备: {device}  d_state={cfg.MAMBA_D_STATE}  d_conv={cfg.MAMBA_D_CONV}")

    data         = build_datasets(cfg)
    train_loader = data["train_loader"]
    val_loader   = data["val_loader"]

    model     = PSTG_Mamba.from_config(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_MAX, eta_min=cfg.ETA_MIN)
    criterion = PSTGLoss(lambda1=cfg.LAMBDA1, lambda2=cfg.LAMBDA2)

    print(f"总参数: {model.count_parameters():,}  "
          f"(Mamba嵌入: {model.mamba_param_count():,}  图层: {model.graph_param_count():,})")

    start_epoch = 1
    if args.resume:
        from pathlib import Path
        rp = Path(args.resume) if Path(args.resume).exists() else Path(cfg.CHECKPOINT_DIR)/"last.pt"
        if rp.exists():
            ckpt = torch.load(rp, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            print(f"续训自 epoch {start_epoch}")

    ckpt_mgr = CheckpointManager(cfg.CHECKPOINT_DIR, save_every=args.save_every)
    print(f"\n=== 开始训练 PSTG-Mamba（epoch {start_epoch} → {cfg.NUM_EPOCHS}）===\n")

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, criterion,
                                  device, cfg.LOG_INTERVAL, epoch)
        val_m   = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        lr_cur = scheduler.get_last_lr()[0]
        flag   = " ★" if val_m["loss"] < ckpt_mgr.global_best_val_loss else \
                 (" ✓" if val_m["loss"] < ckpt_mgr.run_best_val_loss else "")
        print(f"Epoch {epoch:03d}/{cfg.NUM_EPOCHS}  train={train_m['loss']:.4f}  "
              f"val={val_m['loss']:.4f}  lr={lr_cur:.2e}  t={time.time()-t0:.1f}s{flag}")

        extra = {"model_type": "PSTG_Mamba",
                 "d_state": cfg.MAMBA_D_STATE, "d_conv": cfg.MAMBA_D_CONV}
        for p in ckpt_mgr.save(model, optimizer, scheduler, epoch, val_m["loss"], cfg, extra):
            print(f"  → {p}")
        ckpt_mgr.log(epoch, train_m["loss"], val_m["loss"], lr_cur, time.time()-t0)

    ckpt_mgr.finalize()
    print(f"\n训练完成！  评估：python evaluate_mamba.py --ckpt {cfg.CHECKPOINT_DIR}/best.pt")


if __name__ == "__main__":
    main()
