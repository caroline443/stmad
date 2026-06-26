"""
Trainer — training and validation loop for STMAD.

所有输出（checkpoint / 训练曲线）保存到 run_dir，永不覆盖不同实验。
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator for STMAD.

    Args:
        model:        STMAD model
        train_loader: training DataLoader
        val_loader:   validation DataLoader (may be None)
        config:       merged config dict
        device:       torch device
        run_dir:      run-specific output directory (contains best.pt, last.pt)
        log_csv:      path to per-epoch CSV log (optional; defaults to run_dir/train_log.csv)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: dict,
        device: torch.device,
        run_dir: str | Path,
        log_csv: str | Path | None = None,
        loss_fn=None,   # 自定义 loss 函数，None 时用 MSE
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = device
        self.run_dir      = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # CSV log
        self._csv_path = Path(log_csv) if log_csv else self.run_dir / "train_log.csv"
        self._init_csv()

        # Optimiser
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 5e-4),
            weight_decay=config.get("weight_decay", 4e-4),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("epochs", 70),
            eta_min=1e-6,
        )

        self.grad_clip     = config.get("grad_clip", 1.0)
        self.best_val_loss = float("inf")
        self._loss_fn      = loss_fn   # None → 用 F.mse_loss

        # Early stopping
        self.patience      = config.get("patience", 0)   # 0 = 关闭
        self._no_improve   = 0                            # 连续未改善 epoch 数

        # Optional W&B
        self._wandb = None
        if config.get("use_wandb", False):
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not installed; skipping W&B logging")

    # ── CSV init ──────────────────────────────────────────────────────────────

    def _init_csv(self) -> None:
        with open(self._csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for batch in pbar:
            x, target = self._unpack_forecast(batch, self.device)
            x_hat = self.model(x)
            loss  = self._loss_fn(x_hat, target) if self._loss_fn else F.mse_loss(x_hat, target)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        self.scheduler.step()
        return total_loss / max(n_batches, 1)

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> float:
        if self.val_loader is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0
        for batch in self.val_loader:
            x, target = self._unpack_forecast(batch, self.device)
            x_hat     = self.model(x)
            loss_val  = self._loss_fn(x_hat, target) if self._loss_fn else F.mse_loss(x_hat, target)
            total_loss += loss_val.item()
            n_batches  += 1
        return total_loss / max(n_batches, 1)

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Append one row to train_log.csv and optionally to W&B."""
        lr = self._lr()
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{lr:.2e}"])

        if self._wandb is not None:
            self._wandb.log(
                {"train/loss": train_loss, "val/loss": val_loss, "train/lr": lr},
                step=epoch,
            )

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_if_best(self, val_loss: float, epoch: int) -> bool:
        """Overwrite best.pt only when val_loss improves.
        同时更新早停计数器。
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._no_improve   = 0
            self._save("best.pt", epoch, val_loss)
            logger.info(
                f"  ✓ new best  val_loss={val_loss:.6f}  → {self.run_dir}/best.pt"
            )
            return True
        else:
            self._no_improve += 1
            return False

    def should_stop(self) -> bool:
        """早停判断：patience=0 表示不启用早停。"""
        if self.patience <= 0:
            return False
        return self._no_improve >= self.patience

    def save_last(self, epoch: int) -> Path:
        """Save last.pt at the end of training."""
        return self._save("last.pt", epoch, self.best_val_loss)

    def _save(self, filename: str, epoch: int, val_loss: float) -> Path:
        path = self.run_dir / filename
        tmp  = path.with_suffix(".tmp")   # 先写临时文件
        torch.save(
            {
                "epoch":      epoch,
                "state_dict": self.model.state_dict(),
                "val_loss":   val_loss,
                "config":     self.config,
                "optimizer":  self.optimizer.state_dict(),
                "scheduler":  self.scheduler.state_dict(),
            },
            tmp,
        )
        tmp.rename(path)   # 原子替换，写完再覆盖，避免中断导致损坏
        return path

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _unpack(batch) -> torch.Tensor:
        """重建模式：返回 x。"""
        return batch[0] if isinstance(batch, (list, tuple)) else batch

    @staticmethod
    def _unpack_forecast(batch, device) -> tuple[torch.Tensor, torch.Tensor]:
        """统一处理重建和预测两种模式。

        重建模式 batch: x 或 (x, label)           → target = x
        预测模式 batch: (x_ctx, x_fut) 或 (x_ctx, x_fut, label) → target = x_fut
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                x      = batch[0].to(device)
                target = batch[1].to(device)
                # 预测模式：batch[1] 是 x_fut (F, N)
                # 重建模式：batch[1] 是 label (T,) - 1D，不能当 target
                if target.ndim == 1 or target.shape[-1] == 1:
                    # 这是 label，不是 target；重建模式
                    target = x
                return x, target
        x = batch.to(device)
        return x, x   # 重建模式：target = input

    def _lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
