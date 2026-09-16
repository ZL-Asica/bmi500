#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${1:-data}
OUT_DIR=${2:-profile-results}
THREADS=${3:-1}
PYTHON=${PYTHON:-python3}
mkdir -p "$OUT_DIR"

for dataset in pbmc3k pbmc6k pbmc10k; do
  mkdir -p "$OUT_DIR/$dataset"
  "$PYTHON" coarse_profile.py --data-dir "$DATA_DIR" --data-set "$dataset" \
    --out-dir "$OUT_DIR/$dataset" --num-threads "$THREADS" \
    > "$OUT_DIR/$dataset/coarse-console.txt" 2>&1
done
