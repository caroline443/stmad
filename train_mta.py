"""
MTA 训练脚本

与 train.py 的核心区别：
  1. 模型换为 MTA（掩码重建范式，不预测未来）
  2. 损失换为 MTALoss（仅对掩码 patch 计算重建损失）
  3. DataLoader 只需要 context（future 被丢弃）
  4. Checkpoint 目录默认为 checkpoints_mta/

用法：
  python train_mta.py
  python train_mta.py --epochs 70 --mask_ratio 0.4
  python train_mta.py --resume ./checkpoints_mta/run_xxx/best.pt --epochs 70
  python train_mta.py --data_dir /path/to/ESA-Mission1
"""

import os
import json
import argparse
import time
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config_mta import ConfigMTA
from data.dataset import build_datasets
from models.mta import MTA, MTALoss


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint 管理器（与 train.py 相同，复用）
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, ckpt_dir: str, save_every: int = 10):
        self.ckpt_dir   = Path(ckpt_dir)
        self.save_every = save_every

        run_name       = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir   = self.ckpt_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.global_best_val_loss = self._load_global_best()
        self.run_best_val_loss    = float("inf")
        self.run_name  = run_name
        self.history   = []

        print(f"本次 Run 目录：{self.run_dir}")
        print(f"历史全局最优 val_loss：{self.global_best_val_loss:.6f}")

    def _load_global_best(self) -> float:
        runs_json = self.ckpt_dir / "runs.json"
        if runs_json.exists():
            with open(runs_json) as f:
                runs = json.load(f)
            if runs:
                return min(r.get("best_val_loss", float("inf")) for r in runs)
        return float("inf")

    def save(self, model, optimizer, scheduler, epoch, val_loss, cfg):
        state = {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "val_loss":   val_loss,
            "run_name":   self.run_name,
            "config": {
                "patch_sizes":  cfg.PATCH_SIZES,
                "d_model":      cfg.D_MODEL,
                "num_heads":    cfg.NUM_HEADS,
                "num_layers":   cfg.NUM_LAYERS,
                "n_channels":   cfg.NUM_CHANNELS,
                "context_len":  cfg.CONTEXT_LEN,
                "mask_ratio":   cfg.MASK_RATIO,
            },
        }
        saved = []
        torch.save(state, self.ckpt_dir / "last.pt")

        if epoch % self.save_every == 0:
            ep_path = self.run_dir / f"epoch_{epoch:03d}.pt"
            torch.save(state, ep_path)
            saved.append(str(ep_path))

        if val_loss < self.run_best_val_loss:
            self.run_best_val_loss = val_loss
            torch.save(state, self.run_dir / "best.pt")
            saved.append(str(self.run_dir / "best.pt"))

            if val_loss < self.global_best_val_loss:
                self.global_best_val_loss = val_loss
                torch.save(state, self.ckpt_dir / "best.pt")
                saved.append(str(self.ckpt_dir / "best.pt") + "  [全局最优 ★]")

        return saved

    def log(self, epoch, train_loss, val_loss, lr, elapsed):
        self.history.append({
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6), "lr": round(lr, 8),
            "time_s": round(elapsed, 1),
        })
        with open(self.run_dir / "train_log.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def finalize(self):
        runs_json = self.ckpt_dir / "runs.json"
        runs = []
        if runs_json.exists():
            with open(runs_json) as f:
                runs = json.load(f)
        runs.append({
            "run_name":      self.run_name,
            "run_dir":       str(self.run_dir),
            "best_val_loss": self.run_best_val_loss,
            "total_epochs":  self.history[-1]["epoch"] if self.history else 0,
            "finished_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        with open(runs_json, "w") as f:
            json.dump(runs, f, indent=2, ensure_ascii=False)
        print(f"\n本次 Run 已写入：{runs_json}")
        print(f"  best_val_loss：{self.run_best_val_loss:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
#  训练辅助
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="MTA 掩码重建训练")
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--data_dir",     type=str,   default=None)
    p.add_argument("--mask_ratio",   type=float, default=None,
                   help="掩码比例，默认 0.4（覆盖 ConfigMTA.MASK_RATIO）")
    p.add_argument("--lambda1",      type=float, default=None)
    p.add_argument("--lambda2",      type=float, default=None)
    p.add_argument("--device",       type=str,   default=None)
    p.add_argument("--train_stride", type=int,   default=None)
    p.add_argument("--save_every",   type=int,   default=10)
    p.add_argument("--resume",       type=str,   default=None,
                   help="从指定 checkpoint 续训")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  单轮训练（掩码重建）
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, log_interval, epoch):
    model.train()
    total_loss = total_mse = total_freq = total_shape = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for batch_idx, (context, _) in enumerate(pbar):
        # 注意：MTA 不需要 future，丢弃 _
        context = context.to(device, non_blocking=True)   # [B, C, L]

        # 前向（model.training=True → 自动生成随机掩码）
        recon, mask, target = model(context)              # mask=[B,N], recon/target=[B,C,N,p]

        # 重建损失（只对掩码 patch 计算）
        loss, (mse, freq, shape) = criterion(recon, target, mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
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
def validate(model, loader, criterion, device, epoch):
    """
    验证：eval 模式（关闭 Dropout），显式传入随机掩码（不依赖 training flag）。
    """
    model.eval()
    total_loss = total_mse = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False)
    for context, _ in pbar:
        context = context.to(device, non_blocking=True)
        # 显式生成掩码（eval 模式下 model.forward 不会自动生成）
        mask = model._generate_mask(context.shape[0], context.device)
        recon, mask, target = model(context, mask=mask)
        loss, (mse, _, _) = criterion(recon, target, mask)
        total_loss += loss.item()
        total_mse  += mse
        n_batches  += 1

    return {"loss": total_loss / n_batches, "mse": total_mse / n_batches}


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = ConfigMTA()

    if args.epochs:       cfg.NUM_EPOCHS    = args.epochs
    if args.batch_size:   cfg.BATCH_SIZE    = args.batch_size
    if args.data_dir:     cfg.DATA_DIR      = args.data_dir
    if args.mask_ratio:   cfg.MASK_RATIO    = args.mask_ratio
    if args.lambda1:      cfg.LAMBDA1       = args.lambda1
    if args.lambda2:      cfg.LAMBDA2       = args.lambda2
    if args.device:       cfg.DEVICE        = args.device
    if args.train_stride: cfg.TRAIN_STRIDE  = args.train_stride

    set_seed(cfg.SEED)
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")
    print(f"掩码比例：{cfg.MASK_RATIO}  (N={cfg.CONTEXT_LEN // cfg.PATCH_MAIN} patches, "
          f"掩码 {round(cfg.CONTEXT_LEN // cfg.PATCH_MAIN * cfg.MASK_RATIO)} 个)")

    # ── 数据集 ──────────────────────────────────────────────────────────────
    print("\n=== 构建数据集 ===")
    data         = build_datasets(cfg)
    train_loader = data["train_loader"]
    val_loader   = data["val_loader"]

    # ── 模型 ────────────────────────────────────────────────────────────────
    print("\n=== 构建 MTA 模型 ===")
    model     = MTA.from_config(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_MAX, eta_min=cfg.ETA_MIN)
    criterion = MTALoss(lambda1=cfg.LAMBDA1, lambda2=cfg.LAMBDA2)

    n_params = model.count_parameters()
    print(f"模型参数量：{n_params:,}")
    print(f"  编码器（Patch + Graph）：复用 PSTG 结构")
    print(f"  解码器（PatchDecoder）  ：新增 MLP")

    # ── 断点续训 ────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        rp = Path(args.resume)
        if not rp.exists():
            rp = Path(cfg.CHECKPOINT_DIR) / "last.pt"
        if rp.exists():
            print(f"\n续训自：{rp}")
            ckpt = torch.load(rp, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            print(f"  从 epoch {start_epoch} 继续，val_loss={ckpt.get('val_loss', '?')}")
        else:
            print(f"警告：续训文件不存在 {rp}，从头开始")

    # ── Checkpoint 管理器 ────────────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(cfg.CHECKPOINT_DIR, save_every=args.save_every)

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    print(f"\n=== 开始训练（epoch {start_epoch} → {cfg.NUM_EPOCHS}）===")

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg.LOG_INTERVAL, epoch
        )
        val_m = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        lr_cur  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        is_run_best    = val_m["loss"] < ckpt_mgr.run_best_val_loss
        is_global_best = val_m["loss"] < ckpt_mgr.global_best_val_loss
        flag = " ★ 全局最优" if is_global_best else (" ✓ Run最优" if is_run_best else "")

        print(
            f"Epoch {epoch:03d}/{cfg.NUM_EPOCHS}  "
            f"train={train_m['loss']:.4f} (mse={train_m['mse']:.4f})  "
            f"val={val_m['loss']:.4f}  "
            f"lr={lr_cur:.2e}  t={elapsed:.1f}s{flag}"
        )

        saved = ckpt_mgr.save(model, optimizer, scheduler, epoch, val_m["loss"], cfg)
        for p in saved:
            print(f"  → 已保存：{p}")

        ckpt_mgr.log(epoch, train_m["loss"], val_m["loss"], lr_cur, elapsed)

    ckpt_mgr.finalize()

    print(f"\n{'='*45}")
    print(f"MTA 训练完成！")
    print(f"全局最优 val_loss = {ckpt_mgr.global_best_val_loss:.6f}")
    print(f"checkpoint = {cfg.CHECKPOINT_DIR}/best.pt")
    print(f"\n下一步：python evaluate_mta.py --ckpt {cfg.CHECKPOINT_DIR}/best.pt")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
