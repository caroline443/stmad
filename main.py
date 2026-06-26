"""
STMAD — main entry point.

每次运行自动在 runs/ 下建立独立目录，永不覆盖旧实验：
    runs/20260626_1430_esa_d64_l2_k3/
        config.yaml   train_log.csv   stmad.log   best.pt   last.pt   metrics.json

Usage
-----
训练 + 评测（最常用）:
    python main.py --config configs/esa.yaml

纯训练:
    python main.py --config configs/esa.yaml --mode train

复现某次实验（从 run 目录加载配置 + 权重):
    python main.py --mode eval --run_dir runs/20260626_1430_esa_d64_l2_k3

加自定义标签区分消融:
    python main.py --config configs/esa.yaml --tag no_gat

覆盖任意超参:
    python main.py --config configs/esa.yaml --epochs 100 --d_model 128
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from data   import build_dataloaders, load_smap_msl, load_esa
from model  import build_model
from anomaly import compute_anomaly_scores, DynamicThreshold
from utils  import evaluate, Trainer, get_logger

logger = logging.getLogger(__name__)


# ── Run directory ─────────────────────────────────────────────────────────────

def make_run_dir(config: dict, tag: str | None = None) -> Path:
    """Create a unique run directory under runs/.

    Name pattern:  runs/<YYYYMMDD_HHMM>_<dataset>_d<d>_l<layers>_k<topk>[_<tag>]/
    Example:       runs/20260626_1430_esa_d64_l2_k3_ablation/
    """
    ts      = datetime.now().strftime("%Y%m%d_%H%M")
    dataset = config.get("dataset", "unknown")
    d       = config.get("d_model",  64)
    l       = config.get("n_layers", 2)
    k       = config.get("top_k",    5)
    name    = f"{ts}_{dataset}_d{d}_l{l}_k{k}"
    if tag:
        name += f"_{tag}"

    run_dir = Path("runs") / name
    # If a dir with the same name already exists (same minute, rare), add suffix
    if run_dir.exists():
        run_dir = Path(str(run_dir) + "_1")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_dir_from_existing(path: str) -> Path:
    """Validate and return an existing run directory."""
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"run_dir not found: {p}")
    return p


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: dict) -> dict:
    """Merge base.yaml → dataset yaml → CLI overrides."""
    base_path = Path(__file__).parent / "configs" / "base.yaml"
    with open(base_path) as f:
        config = yaml.safe_load(f)
    with open(config_path) as f:
        config.update(yaml.safe_load(f))
    config.update({k: v for k, v in overrides.items() if v is not None})
    return config


def load_config_from_run(run_dir: Path, overrides: dict) -> dict:
    """Load the config snapshot saved inside a previous run directory."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    config.update({k: v for k, v in overrides.items() if v is not None})
    return config


def save_config(config: dict, run_dir: Path) -> None:
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(config: dict):
    dataset = config["dataset"].lower()
    if dataset in ("smap", "msl"):
        return load_smap_msl(
            data_path = config["data_path"],
            dataset   = dataset,
            val_ratio = config.get("val_ratio", 0.1),
        )
    elif dataset == "esa":
        return load_esa(
            data_path   = config["data_path"],
            subsystem   = config.get("esa_subsystem", 5),
            channel_ids = config.get("esa_channel_ids", None),
            val_ratio   = config.get("val_ratio", 0.1),
            train_ratio = config.get("train_ratio", 0.5),
            cache_path  = config.get("esa_cache_path", None),
        )
    raise ValueError(f"Unknown dataset: {config['dataset']}")


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(model, train_loader, val_loader, config, device, run_dir: Path) -> None:
    log_csv = run_dir / "train_log.csv"
    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        config       = config,
        device       = device,
        run_dir      = run_dir,
        log_csv      = log_csv,
    )

    epochs = config.get("epochs", 70)
    logger.info(f"Training for {epochs} epochs  →  {run_dir}")

    has_val = val_loader is not None
    if not has_val:
        logger.warning("No validation loader — model selection uses train loss. "
                       "Set val_ratio > 0 in config to enable a validation split.")

    patience = config.get("patience", 0)
    if patience > 0:
        logger.info(f"Early stopping: patience={patience} epochs")

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(epoch)
        val_loss   = trainer.validate()

        log_val        = val_loss if has_val else float("nan")
        selection_loss = val_loss if has_val else train_loss
        trainer.log_epoch(epoch, train_loss, log_val)
        trainer.save_if_best(selection_loss, epoch)

        if epoch % 10 == 0 or epoch == epochs:
            val_str = f"{val_loss:.6f}" if has_val else "  n/a  "
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train={train_loss:.6f} | val={val_str} | "
                f"best={trainer.best_val_loss:.6f}"
                + (f" | no_improve={trainer._no_improve}/{patience}"
                   if patience > 0 else "")
            )

        trainer.save_last(epoch)   # 每 epoch 覆盖 last.pt，中断后可恢复

        if trainer.should_stop():
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break
    logger.info(f"Training complete.  Run dir: {run_dir}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_evaluation(
    model, train_loader, val_loader, test_loader,
    test_data, test_labels, config, device, run_dir: Path,
) -> dict:
    window_size  = config["window_size"]
    stride       = config.get("stride", 1)            # 推理/测试步长
    train_stride = config.get("train_stride", stride)  # 训练集步长（可能更大）
    beta         = config.get("beta", 0.5)

    logger.info("Computing train scores for threshold fitting ...")
    train_scores, _ = compute_anomaly_scores(
        model, train_loader, device,
        window_size=window_size, stride=train_stride,  # ← 用 train_stride
        total_T=len(train_loader.dataset.data),
    )

    thr = DynamicThreshold(
        p_fit=config.get("p_fit", 0.21),
        p_s  =config.get("p_s",   0.05),
        n_s  =config.get("n_s",   30),
    )

    if val_loader is not None:
        logger.info("Fitting optimal threshold on val set ...")
        val_scores, val_labels = compute_anomaly_scores(
            model, val_loader, device,
            window_size=window_size, stride=stride,    # val 用测试步长
            total_T=len(val_loader.dataset.data),
        )
        if val_labels is not None and len(np.unique(val_labels)) > 1:
            thr.fit_optimal(val_scores, val_labels, beta=beta)
        else:
            thr.fit(train_scores)
    else:
        thr.fit(train_scores)

    logger.info(f"Threshold = {thr.threshold:.6f}")

    logger.info("Scoring test set ...")
    test_scores, _ = compute_anomaly_scores(
        model, test_loader, device,
        window_size=window_size, stride=stride,
        total_T=len(test_data),
    )
    y_pred = thr.predict(test_scores)

    n = min(len(test_labels), len(y_pred), len(test_scores))
    results = evaluate(
        y_true=test_labels[:n].astype(int),
        y_pred=y_pred[:n],
        scores=test_scores[:n],
        beta=beta,
    )

    # ── Print ─────────────────────────────────────────────────────────────
    pw = results["point"]
    ev = results["event"]
    af = results["affiliation"]
    logger.info("=" * 65)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 65)
    logger.info(f"[Point-wise]   P={pw['precision']:.4f}  R={pw['recall']:.4f}  "
                f"F1={pw['f1']:.4f}  F0.5={pw['f05']:.4f}  "
                f"AUC={pw.get('auc', float('nan')):.4f}")
    logger.info(f"[Event-wise]   P={ev['precision']:.4f}  R={ev['recall']:.4f}  "
                f"F{beta}={ev['f_score']:.4f}    ← PSTG: 0.917")
    logger.info(f"[Affiliation]  P={af['precision']:.4f}  R={af['recall']:.4f}  "
                f"F{beta}={af['f_score']:.4f}    ← PSTG: 0.892")
    logger.info("=" * 65)

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        "run_dir":   str(run_dir),
        "threshold": thr.threshold,
        "results":   results,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Metrics saved → {run_dir}/metrics.json")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="STMAD: Spacecraft Anomaly Detection")

    # Mode / config
    p.add_argument("--config",   type=str, default=None,
                   help="Dataset config yaml (required unless --run_dir is given)")
    p.add_argument("--mode",     type=str, default="both",
                   choices=["train", "eval", "both"])
    p.add_argument("--run_dir",  type=str, default=None,
                   help="Existing run directory to evaluate (loads its config + best.pt)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Explicit checkpoint path (overrides run_dir's best.pt)")
    p.add_argument("--tag",      type=str, default=None,
                   help="Optional label appended to the run directory name")

    # Reproducibility
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   type=str, default=None)

    # Hyperparameter overrides
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None, dest="learning_rate")
    p.add_argument("--d_model",    type=int,   default=None)
    p.add_argument("--n_layers",   type=int,   default=None)
    p.add_argument("--top_k",      type=int,   default=None)
    p.add_argument("--wandb",      action="store_true", dest="use_wandb")

    args = p.parse_args()

    if args.config is None and args.run_dir is None:
        p.error("Provide either --config or --run_dir")
    if args.mode == "eval" and args.run_dir is None and args.checkpoint is None:
        p.error("--mode eval requires --run_dir or --checkpoint")

    return args


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    overrides = {
        k: v for k, v in vars(args).items()
        if k not in ("config", "mode", "run_dir", "checkpoint", "tag", "seed", "device")
        and v is not None
    }

    # ── Resolve config + run_dir ──────────────────────────────────────────
    if args.run_dir is not None:
        # Re-run / eval from an existing experiment
        run_dir = run_dir_from_existing(args.run_dir)
        config  = load_config_from_run(run_dir, overrides)
        is_new_run = False
    else:
        config     = load_config(args.config, overrides)
        run_dir    = make_run_dir(config, tag=args.tag)
        save_config(config, run_dir)
        is_new_run = True

    # ── Logger (writes to run_dir/stmad.log) ─────────────────────────────
    get_logger("stmad", log_dir=run_dir)
    logger.info(f"Run dir : {run_dir}")
    logger.info(f"Config  : {json.dumps(config, indent=2, default=str)}")

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Device  : {device}")

    # ── W&B ───────────────────────────────────────────────────────────────
    if config.get("use_wandb", False):
        try:
            import wandb
            wandb.init(
                project=config.get("wandb_project", "stmad"),
                name   =run_dir.name,
                config =config,
            )
        except ImportError:
            logger.warning("wandb not installed; skipping")

    # ── Data ──────────────────────────────────────────────────────────────
    train_data, val_data, test_data, test_labels = load_data(config)
    logger.info(
        f"Data    : train={train_data.shape} val={val_data.shape} "
        f"test={test_data.shape} anomaly={test_labels.mean():.4f}"
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        train_data  =train_data,
        test_data   =test_data,
        test_labels =test_labels,
        window_size =config["window_size"],
        batch_size  =config["batch_size"],
        num_workers =config.get("num_workers", 4),
        val_data    =val_data,
        train_stride=config.get("train_stride", 1),
        test_stride =config.get("stride", 1),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(config).to(device)
    logger.info(f"Model   : {model.count_parameters():,} parameters")

    # ── Train ─────────────────────────────────────────────────────────────
    if args.mode in ("train", "both"):
        run_training(model, train_loader, val_loader, config, device, run_dir)

    # ── Load checkpoint ───────────────────────────────────────────────────
    if args.mode in ("eval", "both"):
        # Priority: explicit --checkpoint > run_dir/best.pt
        ckpt_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            logger.info(f"Loaded  : {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
        else:
            logger.warning(f"Checkpoint not found at {ckpt_path}; using current weights")

    # ── Evaluate ──────────────────────────────────────────────────────────
    if args.mode in ("eval", "both"):
        run_evaluation(
            model, train_loader, val_loader, test_loader,
            test_data, test_labels, config, device, run_dir,
        )


if __name__ == "__main__":
    main()
