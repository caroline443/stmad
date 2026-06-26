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
            x     = self._unpack(batch).to(self.device)
            x_hat = self.model(x)
            loss  = F.mse_loss(x_hat, x)

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
            x     = self._unpack(batch).to(self.device)
            x_hat = self.model(x)
            total_loss += F.mse_loss(x_hat, x).item()
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
        """Overwrite best.pt only when val_loss improves."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save("best.pt", epoch, val_loss)
            logger.info(
                f"  ✓ new best  val_loss={val_loss:.6f}  → {self.run_dir}/best.pt"
            )
            return True
        return False

    def save_last(self, epoch: int) -> Path:
        """Save last.pt at the end of training."""
        return self._save("last.pt", epoch, self.best_val_loss)

    def _save(self, filename: str, epoch: int, val_loss: float) -> Path:
        path = self.run_dir / filename
        torch.save(
            {
                "epoch":      epoch,
                "state_dict": self.model.state_dict(),
                "val_loss":   val_loss,
                "config":     self.config,
                "optimizer":  self.optimizer.state_dict(),
                "scheduler":  self.scheduler.state_dict(),
            },
            path,
        )
        return path

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _unpack(batch) -> torch.Tensor:
        return batch[0] if isinstance(batch, (list, tuple)) else batch

    def _lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
