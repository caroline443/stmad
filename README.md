# STMAD: Spatiotemporal Mamba with Dynamic GAT for Spacecraft Anomaly Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" />
  <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg" />
  <img src="https://img.shields.io/badge/mamba--ssm-1.2+-green.svg" />
</p>

## Overview

STMAD introduces two core innovations for multivariate time-series anomaly detection in spacecraft telemetry:

1. **Mamba Temporal Encoder** — replaces the quadratic-complexity Transformer used in PSTG with a linear-complexity Selective State Space Model (SSM). Mamba's selective-scan mechanism naturally amplifies state changes at anomalous steps.

2. **Dynamic GAT Spatial Encoder** — instead of a static sensor-relationship graph (GDN, FuSAGNet) or coarse temporal snapshots (ContrastAD), we compute per-timestep, per-layer attention coefficients α_ij(t) directly from node features. This yields a *continuously time-varying* adjacency matrix that adapts to the evolving spacecraft state and doubles as an XAI visualisation artefact.

### Architecture

```
Input X ∈ R^{B×T×N}
       │
       ▼
 Multi-scale Patch Embedding   (patch_sizes = [25, 50, 125])
       │
       ▼  ×n_layers
 ┌─────────────────────────────┐
 │  STMAD Block                │
 │  ├─ Mamba Temporal Encoder  │  → H_time  (per-sensor SSM)
 │  ├─ Dynamic GAT              │  → H_space (per-step attention)
 │  └─ Gated Fusion            │  → H = g·H_time + (1-g)·H_space
 └─────────────────────────────┘
       │
       ▼
 Reconstruction Decoder        → X̂ ∈ R^{B×T×N}

Anomaly score: s(t) = mean_N ‖x(t) - x̂(t)‖²
```

### Results (target)

| Method    | Event F0.5 | Affiliation F0.5 | Complexity |
|-----------|-----------|-----------------|-----------|
| GDN       | —         | —               | O(L·N²)   |
| FuSAGNet  | —         | —               | O(L·N²)   |
| PSTG      | 0.917     | 0.892           | O(L²)     |
| **STMAD** | **>0.917** | **>0.892**    | **O(L·N)** |

*All metrics evaluated without point adjustment, following PSTG protocol.*

---

## Installation

```bash
# 1. Clone
git clone https://github.com/caroline443/stmad.git
cd stmad

# 2. Create conda environment (Python 3.9)
conda create -n stmad python=3.9 -y
conda activate stmad

# 3. Install PyTorch (adjust CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install Mamba SSM (requires CUDA)
pip install mamba-ssm causal-conv1d

# 5. Install remaining dependencies
pip install -r requirements.txt
```

---

## Datasets

### SMAP / MSL (NASA)

Download from the [OmniAnomaly repo](https://github.com/NetManAIOps/OmniAnomaly) or the [CATCH paper](https://github.com/decisionintelligence/CATCH).

Expected layout:
```
SMAP/
├── SMAP_train.npy        # (T_train, 25)
├── SMAP_test.npy         # (T_test, 25)
└── SMAP_test_label.npy   # (T_test,)
MSL/
├── MSL_train.npy         # (T_train, 55)
├── MSL_test.npy          # (T_test, 55)
└── MSL_test_label.npy    # (T_test,)
```

### ESA Mission-1

Download from the [ESA-ADB benchmark](https://github.com/ESA-PhiLab/ESA-ADB).

Expected layout:
```
ESA-Mission1/
├── channels/
│   ├── channel_41.zip    # 6 channels from subsystem 5
│   ├── channel_42.zip
│   └── ...
├── telecommands/
├── channels.csv
├── labels.csv
└── anomaly_type.csv
```

Update `data_path` in the corresponding `configs/*.yaml` file before running.

---

## Quick Start

### Edit config

```yaml
# configs/esa.yaml
data_path: /path/to/ESA-Mission1   # ← set this
```

### Run

```bash
# ESA Mission-1 (train + evaluate)
python main.py --config configs/esa.yaml --mode both

# SMAP
python main.py --config configs/smap.yaml --mode both

# MSL
python main.py --config configs/msl.yaml --mode both

# Train only
python main.py --config configs/esa.yaml --mode train

# Evaluate with an existing checkpoint
python main.py --config configs/esa.yaml --mode eval \
    --checkpoint checkpoints/best_model.pt
```

Or use the convenience scripts:
```bash
bash scripts/run_esa.sh
bash scripts/run_smap.sh
bash scripts/run_msl.sh
```

---

## Configuration

All hyperparameters live in YAML files under `configs/`.  Dataset configs inherit from `configs/base.yaml` and override specific keys.

| Key | Default | Description |
|-----|---------|-------------|
| `d_model` | 64 | Embedding dimension |
| `d_state` | 16 | Mamba SSM state size |
| `n_heads` | 4 | GAT attention heads |
| `n_layers` | 2 | Number of STMAD blocks |
| `patch_sizes` | [25,50,125] | Multi-scale patch sizes |
| `top_k` | 5 | GAT top-k neighbours per node |
| `window_size` | 250 (ESA) / 100 (SMAP/MSL) | Sliding window length |
| `batch_size` | 70 (ESA) / 64 (SMAP/MSL) | Batch size |
| `epochs` | 70 | Training epochs |
| `learning_rate` | 5e-4 | AdamW learning rate |
| `p_fit` | 0.21 | Threshold percentile (PSTG default) |
| `n_s` | 30 | Score smoothing window |

---

## Output

After training and evaluation:
```
logs/
├── stmad.log       # full training log
└── results.json    # evaluation metrics (point / event / affiliation)

checkpoints/
├── best_model.pt   # best validation loss checkpoint
└── last_model.pt   # final epoch checkpoint
```

`results.json` example:
```json
{
  "point":       {"precision": 0.94, "recall": 0.91, "f1": 0.92, "f05": 0.93, "auc": 0.97},
  "event":       {"precision": 0.95, "recall": 0.90, "f_score": 0.94},
  "affiliation": {"precision": 0.92, "recall": 0.89, "f_score": 0.91}
}
```

---

## Visualising Dynamic Attention

The dynamic GAT attention matrix A(t) ∈ R^{L×N×N} is stored after each forward pass:

```python
import torch, matplotlib.pyplot as plt
from model import build_model

model = build_model(config)
model.load_state_dict(torch.load("checkpoints/best_model.pt")["state_dict"])
model.eval()

with torch.no_grad():
    x_hat = model(x_test_batch)

attn = model.get_attn_weights()  # (B, L, N, N)
# Plot sensor-relationship heatmap at patch token t=5
plt.imshow(attn[0, 5].cpu().numpy(), cmap="hot")
plt.title("Dynamic GAT attention at patch t=5")
plt.savefig("attn_heatmap.png")
```

---

## Project Structure

```
stmad/
├── main.py               # unified entry point
├── requirements.txt
├── configs/
│   ├── base.yaml         # shared defaults
│   ├── smap.yaml
│   ├── msl.yaml
│   └── esa.yaml
├── data/
│   ├── dataset.py        # SlidingWindowDataset
│   ├── smap_msl_loader.py
│   └── esa_loader.py     # ESA zip → aligned arrays
├── model/
│   ├── patch_embed.py    # multi-scale patch embedding
│   ├── mamba_encoder.py  # Mamba SSM (per-sensor)
│   ├── dynamic_gat.py    # dynamic GAT (per-timestep)
│   ├── stmad_block.py    # Mamba→GAT→Fusion block
│   ├── decoder.py        # reconstruction MLP
│   └── stmad.py          # top-level model + build_model()
├── anomaly/
│   ├── scorer.py         # sliding-window reconstruction error
│   └── threshold.py      # PSTG-style dynamic thresholding
├── utils/
│   ├── metrics.py        # Event-wise F0.5, Affiliation F0.5, AUC
│   ├── trainer.py        # training / validation loop
│   └── logger.py         # logging setup
└── scripts/
    ├── run_esa.sh
    ├── run_smap.sh
    └── run_msl.sh
```

---

## Citation

If you use STMAD in your research, please cite:

```bibtex
@article{stmad2026,
  title   = {STMAD: Spatiotemporal Mamba with Dynamic GAT for Spacecraft Anomaly Detection},
  author  = {},
  journal = {},
  year    = {2026}
}
```

Baselines used for comparison:
- **PSTG**: Chen et al., *Entropy* 2026
- **ContrastAD**: Pei et al., arXiv 2605.23744, 2026
- **GDN**: Deng & Hooi, AAAI 2021
- **FuSAGNet**: Han & Woo, KDD 2022

---

## License

MIT
