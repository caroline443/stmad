"""
STMAD — main entry point.

Usage
-----
Train:
    python main.py --config configs/esa.yaml

Train + evaluate:
    python main.py --config configs/esa.yaml --mode both

Evaluate only (load checkpoint):
    python main.py --config configs/esa.yaml --mode eval --checkpoint checkpoints/best_model.pt

Override any config key on the command line:
    python main.py --config configs/smap.yaml --epochs 100 --batch_size 128
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Project imports ────────────────────────────────────────────────────────────
from data             import build_dataloaders, load_smap_msl, load_esa
from model            import build_model
from anomaly          import compute_anomaly_scores, DynamicThreshold
from utils            import evaluate, Trainer, get_logger

logger = get_logger(__name__)


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: dict) -> dict:
    """Merge base.yaml + dataset yaml + CLI overrides."""
    base_path = Path("configs/base.yaml")
    with open(base_path) as f:
        config = yaml.safe_load(f)

    with open(config_path) as f:
        dataset_cfg = yaml.safe_load(f)

    config.update(dataset_cfg)
    config.update({k: v for k, v in overrides.items() if v is not None})
    return config


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(config: dict):
    """Dispatch to the correct loader based on config['dataset']."""
    dataset = config["dataset"].lower()

    if dataset in ("smap", "msl"):
        train_data, val_data, test_data, test_labels = load_smap_msl(
            data_path  = config["data_path"],
            dataset    = dataset,
            val_ratio  = config.get("val_ratio", 0.1),
        )
    elif dataset == "esa":
        train_data, val_data, test_data, test_labels = load_esa(
            data_path   = config["data_path"],
            subsystem   = config.get("esa_subsystem", 5),
            channel_ids = config.get("esa_channel_ids", None),
            val_ratio   = config.get("val_ratio", 0.1),
            cache_path  = config.get("esa_cache_path", None),
        )
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")

    logger.info(
        f"Dataset={config['dataset'].upper()} | "
        f"train={train_data.shape} val={val_data.shape} test={test_data.shape} | "
        f"anomaly_ratio={test_labels.mean():.4f}"
    )
    return train_data, val_data, test_data, test_labels


# ── Training ───────────────────────────────────────────────────────────────────

def run_training(model, train_loader, val_loader, config, device) -> None:
    trainer = Trainer(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        config          = config,
        device          = device,
        checkpoint_dir  = config.get("checkpoint_dir", "checkpoints"),
    )

    epochs = config.get("epochs", 70)
    logger.info(f"Training for {epochs} epochs ...")

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(epoch)
        val_loss   = trainer.validate()
        trainer.save_if_best(val_loss if val_loss > 0 else train_loss, epoch)

        if epoch % 10 == 0 or epoch == epochs:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )

    trainer.save_checkpoint(epochs, tag="last")
    logger.info("Training complete.")


# ── Evaluation ─────────────────────────────────────────────────────────────────

def run_evaluation(
    model,
    train_loader,
    val_loader,
    test_loader,
    test_data,
    test_labels,
    config,
    device,
) -> dict:
    window_size = config["window_size"]
    stride      = config.get("stride", 1)
    beta        = config.get("beta", 0.5)

    # ── Compute scores on train/val (for threshold fitting) ──────────────
    logger.info("Computing train scores for threshold fitting ...")
    train_scores, _ = compute_anomaly_scores(
        model, train_loader, device,
        window_size=window_size, stride=stride,
        total_T=len(train_loader.dataset.data),
    )

    # ── Fit dynamic threshold ─────────────────────────────────────────────
    thr = DynamicThreshold(
        p_fit = config.get("p_fit", 0.21),
        p_s   = config.get("p_s",   0.05),
        n_s   = config.get("n_s",   30),
    )

    if val_loader is not None:
        logger.info("Fitting optimal threshold on val set ...")
        val_scores, val_labels = compute_anomaly_scores(
            model, val_loader, device,
            window_size=window_size, stride=stride,
            total_T=len(val_loader.dataset.data),
        )
        if val_labels is not None and len(np.unique(val_labels)) > 1:
            thr.fit_optimal(val_scores, val_labels, beta=beta)
        else:
            thr.fit(train_scores)
    else:
        thr.fit(train_scores)

    logger.info(f"Threshold = {thr.threshold:.6f}")

    # ── Score the test set ────────────────────────────────────────────────
    logger.info("Scoring test set ...")
    test_scores, _ = compute_anomaly_scores(
        model, test_loader, device,
        window_size=window_size, stride=stride,
        total_T=len(test_data),
    )

    y_pred = thr.predict(test_scores)

    # Trim to min length in case of off-by-one from windowing
    n = min(len(test_labels), len(y_pred), len(test_scores))
    results = evaluate(
        y_true  = test_labels[:n].astype(int),
        y_pred  = y_pred[:n],
        scores  = test_scores[:n],
        beta    = beta,
    )

    # ── Log results ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    pw = results["point"]
    logger.info(
        f"[Point-wise]   P={pw['precision']:.4f}  R={pw['recall']:.4f}  "
        f"F1={pw['f1']:.4f}  F0.5={pw['f05']:.4f}  "
        f"AUC={pw.get('auc', float('nan')):.4f}"
    )

    ev = results["event"]
    logger.info(
        f"[Event-wise]   P={ev['precision']:.4f}  R={ev['recall']:.4f}  "
        f"F{beta}={ev['f_score']:.4f}"
    )

    af = results["affiliation"]
    logger.info(
        f"[Affiliation]  P={af['precision']:.4f}  R={af['recall']:.4f}  "
        f"F{beta}={af['f_score']:.4f}"
    )

    logger.info("=" * 60)
    logger.info(f"PSTG baseline → Event F0.5=0.917, Affiliation F0.5=0.892")
    logger.info("=" * 60)

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="STMAD: Spacecraft Anomaly Detection")
    p.add_argument("--config",     type=str, required=True,
                   help="Path to dataset config yaml (e.g. configs/esa.yaml)")
    p.add_argument("--mode",       type=str, default="both",
                   choices=["train", "eval", "both"],
                   help="Run mode: train / eval / both (default: both)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint .pt file (for eval mode)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     type=str, default=None,
                   help="Force device: cuda / cpu (default: auto-detect)")
    # Allow overriding any integer/float config key
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None, dest="learning_rate")
    p.add_argument("--d_model",    type=int,   default=None)
    p.add_argument("--n_layers",   type=int,   default=None)
    p.add_argument("--top_k",      type=int,   default=None)
    p.add_argument("--wandb",      action="store_true", dest="use_wandb")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Seed ──────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Config ────────────────────────────────────────────────────────────
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config", "mode", "checkpoint", "seed", "device")}
    config = load_config(args.config, overrides)

    # ── Logger ────────────────────────────────────────────────────────────
    get_logger("stmad", log_dir=config.get("log_dir", "logs"))
    logger.info(f"Config: {json.dumps(config, indent=2, default=str)}")

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── W&B ───────────────────────────────────────────────────────────────
    if config.get("use_wandb", False):
        import wandb
        wandb.init(
            project = config.get("wandb_project", "stmad"),
            name    = f"{config['dataset']}_{config.get('d_model', 64)}d",
            config  = config,
        )

    # ── Data ──────────────────────────────────────────────────────────────
    train_data, val_data, test_data, test_labels = load_data(config)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_data  = train_data,
        test_data   = test_data,
        test_labels = test_labels,
        window_size = config["window_size"],
        batch_size  = config["batch_size"],
        num_workers = config.get("num_workers", 4),
        val_data    = val_data,
        train_stride = 1,
        test_stride  = config.get("stride", 1),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(config).to(device)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # ── Train ─────────────────────────────────────────────────────────────
    if args.mode in ("train", "both"):
        run_training(model, train_loader, val_loader, config, device)

    # ── Load checkpoint for eval ──────────────────────────────────────────
    if args.mode == "eval" or (args.mode == "both" and args.checkpoint):
        ckpt_path = args.checkpoint or (
            Path(config.get("checkpoint_dir", "checkpoints")) / "best_model.pt"
        )
        if Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            logger.info(f"Loaded checkpoint from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
        else:
            logger.warning(f"Checkpoint not found at {ckpt_path}; using current weights")

    # ── Evaluate ──────────────────────────────────────────────────────────
    if args.mode in ("eval", "both"):
        results = run_evaluation(
            model, train_loader, val_loader, test_loader,
            test_data, test_labels, config, device,
        )

        # Save results JSON
        results_path = Path(config.get("log_dir", "logs")) / "results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
