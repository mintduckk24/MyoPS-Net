#!/usr/bin/env bash
set -e

CHECKPOINT="${1:-runs/myopsnet_seed1234/checkpoints/myopsnet_seed1234.pt}"
RUN_DIR="${2:-runs/myopsnet_seed1234}"
DATA_ROOT="${3:-D:/NCKH/Dataset/MyoPS380/Processed_data}"
SPLIT_DIR="${4:-D:/NCKH/SCAR/preprocessing/splits/myops380_no_val}"
BATCH_SIZE="${5:-8}"
AMP="${6:-auto}"

echo "Running Deterministic Inference on 76 Test Volumes..."

python predict.py \
    --checkpoint "$CHECKPOINT" \
    --run_dir "$RUN_DIR" \
    --data_root "$DATA_ROOT" \
    --split_dir "$SPLIT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --amp "$AMP"
