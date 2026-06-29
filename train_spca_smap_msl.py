"""
SpCA 在 SMAP / MSL 数据集上的训练脚本

用法：
  python train_spca_smap_msl.py --dataset smap --data_dir /root/autodl-tmp/data/AT/SMAP
  python train_spca_smap_msl.py --dataset msl  --data_dir /root/autodl-tmp/data/AT/MSL
"""

import argparse
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config_smap_msl import ConfigSMAP, ConfigMSL
from config_spca import ConfigSpCA
from data.smap_msl_dataset import build_datasets_smap_msl
from models.spca import SpCA
from utils.loss import PSTGLoss
from train import CheckpointManager, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  choices=["smap", "msl"], required=True)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--epochs",   type=int, default=None)
    p.add_argument("--device",   type=str, default=None)
    p.add_argument("--save_every", type=int, default=10)
    return p.parse_args()


def build_spca_config(dataset_name, data_dir=None):
    """
    把 SMAP/MSL 的数据配置合并到 SpCA 结构配置。
    SpCA 架构参数（D_MODEL, N_BANDS 等）不变；
    NUM_CHANNELS / DATA_DIR 等数据参数从 SMAP/MSL 配置里取。
    """
    base = ConfigSMAP() if dataset_name == "smap" else ConfigMSL()
    cfg  = ConfigSpCA()

    # 覆盖数据相关配置
    cfg.NUM_CHANNELS  = base.NUM_CHANNELS
    cfg.CHANNELS      = base.CHANNELS
    cfg.BATCH_SIZE    = base.BATCH_SIZE
    cfg.B_S           = base.BATCH_SIZE   # 测试 batch size 同步
    cfg.TRAIN_STRIDE  = base.TRAIN_STRIDE
    cfg.DATA_DIR      = data_dir or base.DATA_DIR
    cfg.DATASET_NAME  = dataset_name

    # 输出目录
    cfg.CHECKPOINT_DIR = f"checkpoints_spca_{dataset_name}"
    cfg.OUTPUT_DIR     = f"outputs_spca_{dataset_name}"

    return cfg


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total, n = 0.0, 0
    for context, future in tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False):
        context, future = context.to(device), future.to(device)
        pred = model(context)
        loss, _ = criterion(pred, future)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); n += 1
    return total / n


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    total, n = 0.0, 0
    for context, future in tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False):
        context, future = context.to(device), future.to(device)
        loss, _ = criterion(model(context), future)
        total += loss.item(); n += 1
    return total / n


def main():
    args = parse_args()
    cfg  = build_spca_config(args.dataset, args.data_dir)
    if args.epochs: cfg.NUM_EPOCHS = args.epochs
    if args.device: cfg.DEVICE     = args.device

    set_seed(cfg.SEED)
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"数据集：{args.dataset}  通道数：{cfg.NUM_CHANNELS}  设备：{device}")

    # 数据
    data = build_datasets_smap_msl(cfg)
    train_loader = data["train_loader"]
    val_loader   = data["val_loader"]

    # 模型
    model     = SpCA.from_config(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_MAX, eta_min=cfg.ETA_MIN)
    criterion = PSTGLoss(lambda1=cfg.LAMBDA1, lambda2=cfg.LAMBDA2)
    print(f"SpCA 参数量：{model.count_parameters():,}")

    ckpt_mgr = CheckpointManager(cfg.CHECKPOINT_DIR, save_every=args.save_every)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss   = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        lr  = scheduler.get_last_lr()[0]
        tag = " ★" if val_loss < ckpt_mgr.global_best_val_loss else ""
        print(f"Epoch {epoch:03d}/{cfg.NUM_EPOCHS}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={lr:.2e}  t={time.time()-t0:.1f}s{tag}")

        ckpt_mgr.save(model, optimizer, scheduler, epoch, val_loss, cfg)
        ckpt_mgr.log(epoch, train_loss, val_loss, lr, time.time() - t0)

    ckpt_mgr.finalize()
    print(f"\n训练完成！checkpoint → {cfg.CHECKPOINT_DIR}/best.pt")
    print(f"评估：python evaluate_smap_msl.py --dataset {args.dataset} "
          f"--ckpt {cfg.CHECKPOINT_DIR}/best.pt")


if __name__ == "__main__":
    main()
