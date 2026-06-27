"""
SMAP / MSL 训练脚本

用法：
  # SMAP
  python train_smap_msl.py --dataset smap --data_dir /root/autodl-tmp/data/AT/SMAP

  # MSL
  python train_smap_msl.py --dataset msl --data_dir /root/autodl-tmp/data/AT/MSL
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config_smap_msl import ConfigSMAP, ConfigMSL
from data.smap_msl_dataset import build_datasets_smap_msl
from models.pstg import PSTG
from utils.loss import PSTGLoss
from train import CheckpointManager, set_seed


def train_one_epoch(model, loader, optimizer, criterion, device, log_interval, epoch):
    model.train()
    total_loss = total_mse = 0.0
    n = 0
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
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return {"loss": total_loss / n, "mse": total_mse / n}


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
        total_loss += loss.item(); total_mse += mse; n += 1
    return {"loss": total_loss / n, "mse": total_mse / n}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      type=str, default="msl", choices=["smap","msl"])
    p.add_argument("--data_dir",     type=str, default=None)
    p.add_argument("--epochs",       type=int, default=70)
    p.add_argument("--train_stride", type=int, default=None)
    p.add_argument("--device",       type=str, default=None)
    p.add_argument("--save_every",   type=int, default=10)
    p.add_argument("--resume",       type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = ConfigSMAP() if args.dataset == "smap" else ConfigMSL()

    if args.data_dir:     cfg.DATA_DIR     = args.data_dir
    if args.epochs:       cfg.NUM_EPOCHS   = args.epochs
    if args.train_stride: cfg.TRAIN_STRIDE = args.train_stride
    if args.device:       cfg.DEVICE       = args.device

    set_seed(cfg.SEED)
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"数据集: {cfg.DATASET_NAME.upper()}  通道数: {cfg.NUM_CHANNELS}  设备: {device}")

    data         = build_datasets_smap_msl(cfg)
    train_loader = data["train_loader"]
    val_loader   = data["val_loader"]

    model = PSTG(
        patch_sizes=cfg.PATCH_SIZES, patch_main=cfg.PATCH_MAIN,
        d_model=cfg.D_MODEL, num_heads=cfg.NUM_HEADS,
        num_layers=cfg.NUM_LAYERS, top_k=cfg.top_k,
        n_channels=cfg.NUM_CHANNELS,
        context_len=cfg.CONTEXT_LEN, forecast_len=cfg.FORECAST_LEN,
        dropout=cfg.P_DROPOUT,
    ).to(device)

    print(f"模型参数量: {model.count_parameters():,}  节点数 n={cfg.NUM_CHANNELS}×{cfg.NUM_PATCHES}={cfg.num_nodes}")

    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_MAX, eta_min=cfg.ETA_MIN)
    criterion = PSTGLoss(lambda1=cfg.LAMBDA1, lambda2=cfg.LAMBDA2)

    start_epoch = 1
    if args.resume:
        from pathlib import Path
        p = Path(args.resume) if Path(args.resume).exists() else Path(cfg.CHECKPOINT_DIR) / "last.pt"
        if p.exists():
            ckpt = torch.load(p, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            print(f"续训自 epoch {start_epoch}")

    ckpt_mgr = CheckpointManager(cfg.CHECKPOINT_DIR, save_every=args.save_every)
    print(f"\n=== 开始训练（epoch {start_epoch} → {cfg.NUM_EPOCHS}）===\n")

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

        extra = {"model_type": "PSTG", "dataset": cfg.DATASET_NAME,
                 "n_channels": cfg.NUM_CHANNELS}
        saved = ckpt_mgr.save(model, optimizer, scheduler, epoch, val_m["loss"], cfg, extra)
        for s in saved: print(f"  → {s}")
        ckpt_mgr.log(epoch, train_m["loss"], val_m["loss"], lr_cur, time.time()-t0)

    ckpt_mgr.finalize()
    print(f"\n训练完成！checkpoint = {cfg.CHECKPOINT_DIR}/best.pt")
    print(f"评估：python evaluate_smap_msl.py --dataset {cfg.DATASET_NAME} "
          f"--data_dir {cfg.DATA_DIR} --ckpt {cfg.CHECKPOINT_DIR}/best.pt")


if __name__ == "__main__":
    main()
