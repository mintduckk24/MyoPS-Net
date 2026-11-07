#!/usr/bin/env bash
set -e

CHECKPOINT="${1:-runs/myopsnet_seed1234/checkpoints/myopsnet_seed1234.pt}"
DATA_ROOT="${2:-D:/NCKH/Dataset/MyoPS380/Processed_data}"
SPLIT_DIR="${3:-D:/NCKH/SCAR/preprocessing/splits/myops380_no_val}"
RUN_DIR="${4:-runs/myopsnet_seed1234}"
SEED="${5:-1234}"
EPOCHS="${6:-300}"
BATCH_SIZE="${7:-16}"
AMP="${8:-auto}"
EPOCHS_PER_RUN="${9:-0}"

echo "Resuming MyoPS-Net Training from $CHECKPOINT..."

python train.py \
    --resume "$CHECKPOINT" \
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
