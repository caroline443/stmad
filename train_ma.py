"""
PSTG-MA 训练脚本

用法：
  # 从头训练
  python train_ma.py --data_dir /root/autodl-tmp/data/ESA-Mission1

  # 从 PSTG checkpoint 热启动（推荐，跳过预热，训练更快）
  python train_ma.py \\
    --data_dir /root/autodl-tmp/data/ESA-Mission1 \\
    --pstg_ckpt checkpoints/best.pt \\
    --warmup_epochs 0

  # 完整训练
  python train_ma.py \\
    --data_dir /root/autodl-tmp/data/ESA-Mission1 \\
    --pstg_ckpt checkpoints/best.pt \\
    --epochs 70 \\
    --warmup_epochs 0 \\
    --save_every 10
"""

import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config_ma import ConfigMA
from data.dataset import build_datasets
from models.pstg_ma import PSTG_MA
from utils.loss_ma import PSTGMALoss
from train import CheckpointManager, set_seed, validate as validate_pred


# ── 训练一轮 ─────────────────────────────────────────────────────────────────

def train_one_epoch_ma(
    model:     PSTG_MA,
    loader,
    optimizer,
    criterion: PSTGMALoss,
    device:    str,
    log_interval: int,
    epoch:     int,
) -> dict:
    model.train()
    totals = {"loss": 0, "pred": 0, "mse": 0, "mem": 0, "ent": 0}
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for batch_idx, (context, future) in enumerate(pbar):
        context = context.to(device, non_blocking=True)   # [B, C, L]
        future  = future.to(device,  non_blocking=True)   # [B, C, F]

        # 前向（v2：同时返回主预测和记忆引导预测）
        x_hat, x_hat_mem, mem_outputs = model(context)

        loss, detail = criterion(x_hat, x_hat_mem, future, mem_outputs, epoch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals["loss"] += loss.item()
        for k in ["pred", "mse", "mem", "ent"]:
            totals[k] += detail.get(k, 0)
        n_batches += 1

        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mem":  f"{detail.get('mem', 0):.4f}",
                "warmup": detail.get("warmup", False),
            })

    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def validate_ma(model, loader, criterion, device, epoch):
    """验证集只看 val loss（预测误差），不跑 memory（加速）"""
    model.eval()
    total_loss = total_mse = 0.0
    n = 0
    for context, future in tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False):
        context = context.to(device, non_blocking=True)
        future  = future.to(device,  non_blocking=True)
        x_hat, x_hat_mem, _ = model(context)
        # 验证集只看主预测 loss（记忆 loss 不稳定，不用于 checkpoint 判断）
        from utils.loss import PSTGLoss
        pred_loss = PSTGLoss(lambda1=criterion.pred_loss.lambda1,
                             lambda2=criterion.pred_loss.lambda2)
        loss, (mse, _, _) = pred_loss(x_hat, future)
        total_loss += loss.item()
        total_mse  += mse
        n += 1
    return {"loss": total_loss / n, "mse": total_mse / n}


# ── 参数解析 ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PSTG-MA 训练")
    p.add_argument("--data_dir",     type=str,   default=None)
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--train_stride", type=int,   default=None)
    p.add_argument("--device",       type=str,   default=None)
    p.add_argument("--save_every",   type=int,   default=10)
    p.add_argument("--warmup_epochs",type=int,   default=None,
                   help="前 N 轮只用预测损失（默认读 config）")
    p.add_argument("--lambda_mem",   type=float, default=None)
    p.add_argument("--lambda_ent",   type=float, default=None)
    p.add_argument("--alpha_pred",   type=float, default=None,
                   help="推理时预测残差的权重（0~1）")
    # 热启动：从已有的 PSTG checkpoint 加载共享参数
    p.add_argument("--pstg_ckpt",    type=str,   default=None,
                   help="PSTG checkpoint 路径，用于热启动（推荐）")
    # 续训
    p.add_argument("--resume",       type=str,   default=None,
                   help="从 PSTG-MA checkpoint 续训")
    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = ConfigMA()

    if args.data_dir:     cfg.DATA_DIR      = args.data_dir
    if args.epochs:       cfg.NUM_EPOCHS    = args.epochs
    if args.batch_size:   cfg.BATCH_SIZE    = args.batch_size
    if args.train_stride: cfg.TRAIN_STRIDE  = args.train_stride
    if args.device:       cfg.DEVICE        = args.device
    if args.warmup_epochs is not None: cfg.WARMUP_EPOCHS = args.warmup_epochs
    if args.lambda_mem:   cfg.LAMBDA_MEM    = args.lambda_mem
    if args.lambda_ent:   cfg.LAMBDA_ENT    = args.lambda_ent
    if args.alpha_pred is not None:    cfg.ALPHA_PRED    = args.alpha_pred

    set_seed(cfg.SEED)
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")

    # ── 数据集 ────────────────────────────────────────────────────────────
    print("\n=== 构建数据集 ===")
    data         = build_datasets(cfg)
    train_loader = data["train_loader"]
    val_loader   = data["val_loader"]

    # ── 模型 ──────────────────────────────────────────────────────────────
    print("\n=== 构建 PSTG-MA 模型 ===")
    if args.pstg_ckpt and Path(args.pstg_ckpt).exists():
        print(f"热启动自 PSTG checkpoint：{args.pstg_ckpt}")
        model = PSTG_MA.from_pstg_checkpoint(args.pstg_ckpt, cfg, device)
    else:
        model = PSTG_MA.from_config(cfg).to(device)
        print("从头初始化")

    total_params   = model.count_parameters()
    new_params     = model.count_new_parameters()
    pstg_params    = total_params - new_params
    print(f"总参数量：{total_params:,}")
    print(f"  PSTG 共享参数：{pstg_params:,}")
    print(f"  记忆库新增参数：{new_params:,}（{new_params/total_params*100:.1f}%）")

    # ── 优化器 ────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_MAX, eta_min=cfg.ETA_MIN)
    criterion = PSTGMALoss(
        lambda1=cfg.LAMBDA1, lambda2=cfg.LAMBDA2,
        lambda_mem=cfg.LAMBDA_MEM, lambda_ent=cfg.LAMBDA_ENT,
        warmup_epochs=cfg.WARMUP_EPOCHS,
    )

    # ── 续训 ──────────────────────────────────────────────────────────────
    start_epoch = 1
    ckpt_dir_ma = cfg.CHECKPOINT_DIR + "_ma"
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\n续训自：{resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            print(f"从 epoch {start_epoch} 继续")

    # ── Checkpoint 管理器 ─────────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(ckpt_dir_ma, save_every=args.save_every)

    # ── 训练循环 ──────────────────────────────────────────────────────────
    print(f"\n=== 开始训练 PSTG-MA（epoch {start_epoch} → {cfg.NUM_EPOCHS}）===")
    print(f"Warmup 前 {cfg.WARMUP_EPOCHS} 轮只用预测损失\n")

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_m = train_one_epoch_ma(
            model, train_loader, optimizer, criterion,
            device, cfg.LOG_INTERVAL, epoch
        )
        val_m = validate_ma(model, val_loader, criterion, device, epoch)
        scheduler.step()

        lr_cur  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        is_best = val_m["loss"] < ckpt_mgr.run_best_val_loss
        is_global = val_m["loss"] < ckpt_mgr.global_best_val_loss
        flag = " ★" if is_global else (" ✓" if is_best else "")

        warmup_flag = " [warmup]" if epoch <= cfg.WARMUP_EPOCHS else ""
        print(
            f"Epoch {epoch:03d}/{cfg.NUM_EPOCHS}  "
            f"train={train_m['loss']:.4f}(mem={train_m['mem']:.4f})  "
            f"val={val_m['loss']:.4f}  lr={lr_cur:.2e}  t={elapsed:.1f}s{flag}{warmup_flag}"
        )

        # 保存
        extra = {"alpha_pred": cfg.ALPHA_PRED, "model_type": "PSTG_MA"}
        saved = ckpt_mgr.save(model, optimizer, scheduler, epoch, val_m["loss"], cfg, extra)
        for p in saved:
            print(f"  → {p}")

        ckpt_mgr.log(epoch, train_m["loss"], val_m["loss"], lr_cur, elapsed)

    ckpt_mgr.finalize()

    print(f"\n{'='*50}")
    print(f"PSTG-MA 训练完成！")
    print(f"全局最优 val_loss = {ckpt_mgr.global_best_val_loss:.6f}")
    print(f"checkpoint = {ckpt_dir_ma}/best.pt")
    print(f"{'='*50}")
    print(f"\n运行评估：")
    print(f"  python evaluate_ma.py --data_dir {cfg.DATA_DIR} --ckpt {ckpt_dir_ma}/best.pt")


if __name__ == "__main__":
    main()
