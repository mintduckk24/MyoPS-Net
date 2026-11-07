#!/usr/bin/env bash
set -e

DATA_ROOT="${1:-D:/NCKH/Dataset/MyoPS380/Processed_data}"
SPLIT_DIR="${2:-D:/NCKH/SCAR/preprocessing/splits/myops380_no_val}"
RUN_DIR="${3:-runs/myopsnet_seed1234}"
SEED="${4:-1234}"
EPOCHS="${5:-300}"
BATCH_SIZE="${6:-16}"
AMP="${7:-auto}"
EPOCHS_PER_RUN="${8:-0}"

echo "Starting MyoPS-Net 300-Epoch Training on MyoPS380..."
echo "Data root: $DATA_ROOT"
echo "Split dir: $SPLIT_DIR"
echo "Run dir:   $RUN_DIR"
echo "Seed:      $SEED"

python train.py \
    --data_root "$DATA_ROOT" \
    --split_dir "$SPLIT_DIR" \
    --run_dir "$RUN_DIR" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --base_lr 0.0003333 \
    --weight_decay 0.0001 \
    --amp "$AMP" \
    --epochs_per_run "$EPOCHS_PER_RUN"
