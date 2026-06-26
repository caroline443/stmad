#!/usr/bin/env bash
# Run STMAD on the MSL dataset.
#
# Usage:
#   bash scripts/run_msl.sh
#   bash scripts/run_msl.sh --epochs 100

set -euo pipefail

DATA_PATH="/path/to/MSL"   # ← change this

python main.py \
    --config configs/msl.yaml \
    --mode   both \
    --device cuda \
    --seed   42 \
    "$@"
