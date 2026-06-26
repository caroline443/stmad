#!/usr/bin/env bash
# Run STMAD on the SMAP dataset.
# Edit DATA_PATH to point to your SMAP directory before running.
#
# Usage:
#   bash scripts/run_smap.sh
#   bash scripts/run_smap.sh --epochs 100

set -euo pipefail

DATA_PATH="/path/to/SMAP"   # ← change this

python main.py \
    --config configs/smap.yaml \
    --mode   both \
    --device cuda \
    --seed   42 \
    "$@"

# Override data_path inline if needed:
# python main.py --config configs/smap.yaml data_path=$DATA_PATH
