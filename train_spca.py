"""
PSTG 训练脚本（对应 Algorithm 1 Part 1）

保存机制：
  每次运行自动创建带时间戳的独立目录，旧 checkpoint 永不被覆盖。

  checkpoints/
  ├── best.pt                     ← 全局最优（跨所有 run）
  ├── last.pt                     ← 最近一次 run 的最新 epoch
  ├── runs.json                   ← 所有 run 的汇总记录
  └── run_YYYYMMDD_HHMMSS/        ← 本次 run 的独立目录
      ├── best.pt                 ← 本次 run 最优 checkpoint
      ├── epoch_010.pt            ← 每 --save_every 轮保存一次
      ├── epoch_020.pt
      └── train_log.json          ← 本次 run 的完整 loss 曲线

用法：
  python train.py --data_dir /root/autodl-tmp/data/ESA-Mission1
  python train.py --epochs 70 --train_stride 50
  python train.py --resume ./checkpoints/run_20260626_170530/epoch_030.pt --epochs 70
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

from config_spca import ConfigSpCA as Config
from data.dataset import build_datasets
from models.spca import SpCA
from utils.loss import PSTGLoss


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint 管理器
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    每次训练创建独立的 run 目录，防止旧 checkpoint 被覆盖。

    目录结构：
      checkpoints/
      ├── best.pt          ← 全局最优（所有 run 中 val_loss 最低的）
      ├── last.pt          ← 最近一次 run 的最新 epoch
      ├── runs.json        ← 所有 run 的汇总
      └── run_YYYYMMDD_HHMMSS/
          ├── best.pt      ← 本次 run 最优
          ├── epoch_010.pt
          └── train_log.json
    """

    def __init__(self, ckpt_dir: str, save_every: int = 10):
        self.ckpt_dir  = Path(ckpt_dir)
        self.save_every = save_every

        # 为本次 run 创建独立目录
        run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.ckpt_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # 全局最优 val_loss（跨所有 run）
        self.global_best_val_loss = self._load_global_best()
        # 本次 run 最优
        self.run_best_val_loss = float("inf")

        self.run_name = run_name
        self.history  = []   # 本次 run 的 loss 记录

        print(f"本次 Run 目录：{self.run_dir}")
        print(f"历史全局最优 val_loss：{self.global_best_val_loss:.6f}")

    def _load_global_best(self) -> float:
        """从 runs.json 读取历史全局最优 val_loss。"""
        runs_json = self.ckpt_dir / "runs.json"
        if runs_json.exists():
            with open(runs_json) as f:
                runs = json.load(f)
            if runs:
                return min(r.get("best_val_loss", float("inf")) for r in runs)
        return float("inf")

    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        val_loss: float,
        cfg,
        extra: dict = None,
    ) -> dict:
        """
        保存逻辑：
        1. 每个 epoch 覆盖 last.pt（只留最新）
        2. 每 save_every 轮保存一份 epoch_XXX.pt（永久保留）
        3. 本次 run 最优 → run_dir/best.pt（永久保留）
        4. 全局最优     → checkpoints/best.pt（永久保留）
        """
        state = {
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "val_loss":      val_loss,
            "run_name":      self.run_name,
            "config": {
                "n_channels":      cfg.NUM_CHANNELS,
                "context_len":     cfg.CONTEXT_LEN,
                "forecast_len":    cfg.FORECAST_LEN,
                "d_model":         cfg.D_MODEL,
                "n_heads":         cfg.NUM_HEADS,
                "n_bands":         cfg.N_BANDS,
                "band_splits":     cfg.BAND_SPLITS,
                "n_patches":         cfg.N_PATCHES,
                "n_layers_band":     cfg.N_LAYERS_BAND,
                "n_layers_global":   cfg.N_LAYERS_GLOBAL,
                "use_spectral":      cfg.USE_SPECTRAL,
                "use_channel_attn":  cfg.USE_CHANNEL_ATTN,
            },
        }
        if extra:
            state.update(extra)

        saved_paths = []

        # 1. last.pt（全局，只留最新 epoch 方便快速恢复）
        last_path = self.ckpt_dir / "last.pt"
        torch.save(state, last_path)

        # 2. 每 save_every 轮永久保存一份（永不被覆盖）
        if epoch % self.save_every == 0:
            ep_path = self.run_dir / f"epoch_{epoch:03d}.pt"
            torch.save(state, ep_path)
            saved_paths.append(str(ep_path))

        # 3. 本次 run 最优
        if val_loss < self.run_best_val_loss:
            self.run_best_val_loss = val_loss
            run_best_path = self.run_dir / "best.pt"
            torch.save(state, run_best_path)
            saved_paths.append(str(run_best_path))

            # 4. 全局最优（跨所有 run）
            if val_loss < self.global_best_val_loss:
                self.global_best_val_loss = val_loss
                global_best_path = self.ckpt_dir / "best.pt"
                torch.save(state, global_best_path)
                saved_paths.append(str(global_best_path) + "  [全局最优 ★]")

        return saved_paths

    def log(self, epoch: int, train_loss: float, val_loss: float, lr: float, elapsed: float):
        """记录本次 epoch 的指标。"""
        self.history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "lr":         round(lr, 8),
            "time_s":     round(elapsed, 1),
        })
        # 实时写入，训练中断也不丢数据
        log_path = self.run_dir / "train_log.json"
        with open(log_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def finalize(self):
        """训练结束时，把本次 run 的摘要写入 runs.json。"""
        runs_json = self.ckpt_dir / "runs.json"
        runs = []
        if runs_json.exists():
            with open(runs_json) as f:
                runs = json.load(f)

        summary = {
            "run_name":        self.run_name,
            "run_dir":         str(self.run_dir),
            "best_val_loss":   self.run_best_val_loss,
            "total_epochs":    self.history[-1]["epoch"] if self.history else 0,
            "finished_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_log":       str(self.run_dir / "train_log.json"),
        }
        runs.append(summary)
        with open(runs_json, "w") as f:
            json.dump(runs, f, indent=2, ensure_ascii=False)

        print(f"\n本次 Run 摘要已写入：{runs_json}")
        print(f"  run_name     : {self.run_name}")
        print(f"  best_val_loss: {self.run_best_val_loss:.6f}")
        print(f"  checkpoint 目录: {self.run_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  训练辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="SpCA 训练")
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--data_dir",    type=str,   default=None)
    parser.add_argument("--lambda1",     type=float, default=None)
    parser.add_argument("--lambda2",     type=float, default=None)
    parser.add_argument("--device",      type=str,   default=None)
    parser.add_argument("--train_stride",type=int,   default=None)
    parser.add_argument("--save_every",  type=int,   default=10,
                        help="每隔多少轮永久保存一个 checkpoint（默认 10）")
    parser.add_argument("--resume",      type=str,   default=None,
                        help="从指定 checkpoint 续训，例如 ./checkpoints/run_xxx/epoch_030.pt")
    parser.add_argument("--temporal",    action="store_true",
                        help="启用时序注意力编码（v2，N_PATCHES=10）；默认用线性投影（v1）")
    # 消融实验参数
    parser.add_argument("--n_bands",     type=int,   default=None,
                        help="频段数（默认 3）")
    parser.add_argument("--band_splits", type=float, nargs="+", default=None,
                        help="频段分割点，如 0.1 0.4（默认 0.1 0.4）")
    parser.add_argument("--seed",        type=int,   default=None,
                        help="随机种子（覆盖 Config.SEED）")
    parser.add_argument("--ckpt_dir",    type=str,   default=None,
                        help="checkpoint 输出目录（覆盖 cfg.CHECKPOINT_DIR）")
    # 组件消融
    parser.add_argument("--no_spectral",     action="store_true",
                        help="消融：去掉频域分解（验证FFT的贡献）")
    parser.add_argument("--no_channel_attn", action="store_true",
                        help="消融：去掉跨通道注意力（验证通道建模的贡献）")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device, log_interval, epoch):
    model.train()
    total_loss = total_mse = total_freq = total_shape = 0.0
    n_batches = 0

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

        total_loss  += loss.item()
        total_mse   += mse
        total_freq  += freq
        total_shape += shape
        n_batches   += 1

        if batch_idx % log_interval == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "mse": f"{mse:.4f}"})

    return {
        "loss":  total_loss  / n_batches,
        "mse":   total_mse   / n_batches,
        "freq":  total_freq  / n_batches,
        "shape": total_shape / n_batches,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = total_mse = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False)
    for context, future in pbar:
        context = context.to(device, non_blocking=True)
        future  = future.to(device,  non_blocking=True)
        pred = model(context)
        loss, (mse, _, _) = criterion(pred, future)
        total_loss += loss.item()
        total_mse  += mse
        n_batches  += 1

    return {"loss": total_loss / n_batches, "mse": total_mse / n_batches}


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = Config()

    if args.epochs:      cfg.NUM_EPOCHS   = args.epochs
    if args.batch_size:  cfg.BATCH_SIZE   = args.batch_size
    if args.data_dir:    cfg.DATA_DIR     = args.data_dir
    if args.lambda1:     cfg.LAMBDA1      = args.lambda1
    if args.lambda2:     cfg.LAMBDA2      = args.lambda2
    if args.device:      cfg.DEVICE       = args.device
    if args.train_stride: cfg.TRAIN_STRIDE = args.train_stride
    if args.temporal:     cfg.N_PATCHES    = 10
    if args.n_bands:          cfg.N_BANDS          = args.n_bands
    if args.band_splits:      cfg.BAND_SPLITS      = tuple(args.band_splits)
    if args.seed:             cfg.SEED             = args.seed
    if args.ckpt_dir:         cfg.CHECKPOINT_DIR   = args.ckpt_dir
    if args.no_spectral:      cfg.USE_SPECTRAL     = False
    if args.no_channel_attn:  cfg.USE_CHANNEL_ATTN = False

    set_seed(cfg.SEED)
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")

    # ── 数据集 ────────────────────────────────────────────────────────────
    print("\n=== 构建数据集 ===")
    data         = build_datasets(cfg)
    train_loader = data["train_loader"]
    val_loader   = data["val_loader"]

    # ── 模型 + 优化器 ─────────────────────────────────────────────────────
    print("\n=== 构建模型 ===")
    model     = SpCA.from_config(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_MAX, eta_min=cfg.ETA_MIN)
    criterion = PSTGLoss(lambda1=cfg.LAMBDA1, lambda2=cfg.LAMBDA2)
    print(f"模型参数量：{model.count_parameters():,}")

    # ── 断点续训 ──────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            # 自动尝试 last.pt
            resume_path = Path(cfg.CHECKPOINT_DIR) / "last.pt"
        if resume_path.exists():
            print(f"\n续训自：{resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            print(f"  从 epoch {start_epoch} 继续，上次 val_loss={ckpt.get('val_loss', '?')}")
        else:
            print(f"警告：续训文件不存在 {resume_path}，从头开始训练")

    # ── Checkpoint 管理器 ─────────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(cfg.CHECKPOINT_DIR, save_every=args.save_every)

    # ── 训练循环 ──────────────────────────────────────────────────────────
    print(f"\n=== 开始训练（epoch {start_epoch} → {cfg.NUM_EPOCHS}）===")
    print(f"每 {args.save_every} 轮永久保存 | save_every={args.save_every}\n")

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg.LOG_INTERVAL, epoch
        )
        val_m = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        lr_cur  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # 打印
        is_run_best    = val_m["loss"] < ckpt_mgr.run_best_val_loss
        is_global_best = val_m["loss"] < ckpt_mgr.global_best_val_loss
        flag = " ★ 全局最优" if is_global_best else (" ✓ Run最优" if is_run_best else "")
        print(
            f"Epoch {epoch:03d}/{cfg.NUM_EPOCHS}  "
            f"train={train_m['loss']:.4f}  val={val_m['loss']:.4f}  "
            f"lr={lr_cur:.2e}  t={elapsed:.1f}s{flag}"
        )

        # 保存
        saved = ckpt_mgr.save(model, optimizer, scheduler, epoch, val_m["loss"], cfg)
        for p in saved:
            print(f"  → 已保存：{p}")

        # 记录 log
        ckpt_mgr.log(epoch, train_m["loss"], val_m["loss"], lr_cur, elapsed)

    # ── 收尾 ──────────────────────────────────────────────────────────────
    ckpt_mgr.finalize()

    print(f"\n======================================")
    print(f"训练完成！")
    print(f"全局最优  val_loss = {ckpt_mgr.global_best_val_loss:.6f}")
    print(f"          checkpoint = {cfg.CHECKPOINT_DIR}/best.pt")
    print(f"本次 Run  val_loss = {ckpt_mgr.run_best_val_loss:.6f}")
    print(f"          checkpoint = {ckpt_mgr.run_dir}/best.pt")
    print(f"======================================")


if __name__ == "__main__":
    main()
