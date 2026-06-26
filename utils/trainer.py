"""
Trainer — training and validation loop for STMAD.

Handles:
    • One epoch of training (MSE loss, gradient clipping)
    • Validation loss computation
    • Model checkpointing (best val loss)
    • Optional W&B logging
"""

from __future__ import annotations

import logging
import time
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
        model:          STMAD model
        train_loader:   training DataLoader (yields x or (x, y))
        val_loader:     validation DataLoader (may be None)
        config:         merged config dict
        device:         torch device
        checkpoint_dir: directory to save best model weights

    Usage::

        trainer = Trainer(model, train_loader, val_loader, config, device)
        for epoch in range(config["epochs"]):
            train_loss = trainer.train_epoch(epoch)
            val_loss   = trainer.validate()
            trainer.save_if_best(val_loss, epoch)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: dict,
        device: torch.device,
        checkpoint_dir: str | Path = "checkpoints",
    ) -> None:
        self.model       = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimiser
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 5e-4),
            weight_decay=config.get("weight_decay", 4e-4),
        )

        # Scheduler: cosine annealing over all epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("epochs", 70),
            eta_min=1e-6,
        )

        self.grad_clip   = config.get("grad_clip", 1.0)
        self.best_val_loss = float("inf")

        # Optional W&B
        self._wandb = None
        if config.get("use_wandb", False):
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not installed; skipping W&B logging")

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch.

        Returns
        -------
        mean train loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for batch in pbar:
            x = self._unpack(batch).to(self.device)  # (B, T, N)

            x_hat = self.model(x)                    # (B, T, N)
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
        mean_loss = total_loss / max(n_batches, 1)

        if self._wandb is not None:
            self._wandb.log({"train/loss": mean_loss, "train/lr": self._lr()}, step=epoch)

        return mean_loss

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> float:
        """Compute validation MSE loss.

        Returns 0.0 if no val_loader is set.
        """
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.val_loader:
            x    = self._unpack(batch).to(self.device)
            x_hat = self.model(x)
            total_loss += F.mse_loss(x_hat, x).item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_if_best(self, val_loss: float, epoch: int) -> bool:
        """Save model weights if val_loss is the best seen so far.

        Returns True if the model was saved.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch":      epoch,
                    "state_dict": self.model.state_dict(),
                    "val_loss":   val_loss,
                    "config":     self.config,
                },
                path,
            )
            logger.info(f"Epoch {epoch}: new best val_loss={val_loss:.6f}  → saved to {path}")
            return True
        return False

    def save_checkpoint(self, epoch: int, tag: str = "last") -> Path:
        """Save a named checkpoint (e.g. last epoch)."""
        path = self.checkpoint_dir / f"{tag}_model.pt"
        torch.save(
            {
                "epoch":      epoch,
                "state_dict": self.model.state_dict(),
                "val_loss":   self.best_val_loss,
                "config":     self.config,
            },
            path,
        )
        return path

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _unpack(batch) -> torch.Tensor:
        """Extract the x tensor from a batch that may be (x,) or (x, y)."""
        if isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    def _lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
