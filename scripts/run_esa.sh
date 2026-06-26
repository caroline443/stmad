#!/usr/bin/env bash
# Run STMAD on the ESA Mission-1 dataset.
# The first run extracts and aligns channel zip files; subsequent runs
# can use the --esa_cache_path to skip preprocessing.
#
# Usage:
#   bash scripts/run_esa.sh
#   bash scripts/run_esa.sh --epochs 50

set -euo pipefail

DATA_PATH="/path/to/ESA-Mission1"     # ← change this
CACHE_PATH="/path/to/ESA-Mission1/cache"  # preprocessed .npy cache

python main.py \
    --config configs/esa.yaml \
    --mode   both \
    --device cuda \
    --seed   42 \
    "$@"

# To watch GPU utilisation during training:
#   watch -n 2 nvidia-smi
