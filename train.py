"""
PSTG 训练脚本（对应 Algorithm 1 Part 1）

用法：
    python train.py [--epochs 70] [--batch_size 64] [--data_dir /path/to/data]

特性：
    - AdamW 优化器 + CosineAnnealing 学习率调度
    - 在验证集上选最优 checkpoint（按 val loss 最小）
    - 断点续训（--resume）
    - TensorBoard 日志（可选）
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import Config
from data.dataset import build_datasets
from models.pstg import PSTG
from utils.loss import PSTGLoss


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="PSTG 训练")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--data_dir",   type=str,   default=None)
    parser.add_argument("--lambda1",    type=float, default=None)
    parser.add_argument("--lambda2",    type=float, default=None)
    parser.add_argument("--device",     type=str,   default=None)
    parser.add_argument("--resume",     type=str,   default=None,
                        help="从指定 checkpoint 续训")
    parser.add_argument("--train_stride", type=int, default=None,
                        help="训练集滑窗步长（默认 50）")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion: PSTGLoss,
    device: str,
    log_interval: int,
    epoch: int,
) -> dict:
    model.train()
    total_loss = total_mse = total_freq = total_shape = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for batch_idx, (context, future) in enumerate(pbar):
        context = context.to(device, non_blocking=True)   # [B, C, L]
        future  = future.to(device,  non_blocking=True)   # [B, C, F]

        pred = model(context)                              # [B, C, F]
        loss, (mse, freq, shape) = criterion(pred, future)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 梯度裁剪防止梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item()
        total_mse   += mse
        total_freq  += freq
        total_shape += shape
        n_batches   += 1

        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mse":  f"{mse:.4f}",
            })

    return {
        "loss":  total_loss  / n_batches,
        "mse":   total_mse   / n_batches,
        "freq":  total_freq  / n_batches,
        "shape": total_shape / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: PSTGLoss,
    device: str,
    epoch: int,
) -> dict:
    model.eval()
    total_loss = total_mse = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]", leave=False)
    for context, future in pbar:
        context = context.to(device, non_blocking=True)
        future  = future.to(device,  non_blocking=True)
        pred = model(context)
        loss, (mse, _, _) = criterion(pred, future)
        total_loss += loss.item()
        total_mse  += mse
        n_batches  += 1

    return {
        "loss": total_loss / n_batches,
        "mse":  total_mse  / n_batches,
    }


def main():
    args = parse_args()
    cfg = Config()

    # 命令行参数覆盖配置
    if args.epochs:     cfg.NUM_EPOCHS = args.epochs
    if args.batch_size: cfg.BATCH_SIZE = args.batch_size
    if args.data_dir:   cfg.DATA_DIR   = args.data_dir
    if args.lambda1:    cfg.LAMBDA1    = args.lambda1
    if args.lambda2:    cfg.LAMBDA2    = args.lambda2
    if args.device:     cfg.DEVICE     = args.device
    if args.train_stride: cfg.TRAIN_STRIDE = args.train_stride

    set_seed(cfg.SEED)
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")

    # ── 数据集 ────────────────────────────────────────────────────────────
    print("\n=== 构建数据集 ===")
    data = build_datasets(cfg)
    train_loader = data["train_loader"]
    val_loader   = data["val_loader"]

    # ── 模型 ──────────────────────────────────────────────────────────────
    print("\n=== 构建模型 ===")
    model = PSTG.from_config(cfg).to(device)
    print(f"模型参数量：{model.count_parameters():,}")

    # ── 优化器 & 调度器 ────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.T_MAX,
        eta_min=cfg.ETA_MIN,
    )
    criterion = PSTGLoss(lambda1=cfg.LAMBDA1, lambda2=cfg.LAMBDA2)

    # ── 断点续训 ──────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"从 epoch {start_epoch} 续训，最优 val loss={best_val_loss:.6f}")

    # ── 训练循环 ──────────────────────────────────────────────────────────
    print(f"\n=== 开始训练（共 {cfg.NUM_EPOCHS} 轮）===")
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg.LOG_INTERVAL, epoch
        )
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        lr_cur = scheduler.get_last_lr()[0]
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["lr"].append(lr_cur)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{cfg.NUM_EPOCHS}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"lr={lr_cur:.2e}  "
            f"time={elapsed:.1f}s"
        )

        # 保存最新 checkpoint
        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "last.pt")
        torch.save({
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": {
                "patch_sizes":  cfg.PATCH_SIZES,
                "d_model":      cfg.D_MODEL,
                "num_heads":    cfg.NUM_HEADS,
                "num_layers":   cfg.NUM_LAYERS,
                "n_channels":   cfg.NUM_CHANNELS,
                "context_len":  cfg.CONTEXT_LEN,
                "forecast_len": cfg.FORECAST_LEN,
            },
        }, ckpt_path)

        # 保存最优 checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = os.path.join(cfg.CHECKPOINT_DIR, "best.pt")
            torch.save(torch.load(ckpt_path), best_path)
            print(f"  ✓ 新最优 val_loss={best_val_loss:.6f}，已保存至 {best_path}")

    # ── 保存训练历史 ──────────────────────────────────────────────────────
    import json
    history_path = os.path.join(cfg.OUTPUT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n训练完成！历史记录保存至 {history_path}")
    print(f"最优 val_loss = {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
