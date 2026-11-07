#!/usr/bin/env bash
set -e

PRED_DIR="${1:-runs/myopsnet_seed1234/test_predictions}"
OUTPUT_DIR="${2:-runs/myopsnet_seed1234/evaluation}"
DATA_ROOT="${3:-D:/NCKH/Dataset/MyoPS380/Processed_data}"
SPLIT_DIR="${4:-D:/NCKH/SCAR/preprocessing/splits/myops380_no_val/test_vol.txt}"

echo "Running Standalone Evaluation (myops380_voxel_v3)..."

python evaluate_predictions.py \
    --pred_dir "$PRED_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --data_root "$DATA_ROOT" \
    --split_dir "$SPLIT_DIR"
