#!/usr/bin/env bash
set -e

DATA_ROOT="${1:-D:/NCKH/Dataset/MyoPS380/Processed_data}"
SPLIT_DIR="${2:-D:/NCKH/SCAR/preprocessing/splits/myops380_no_val}"
RUN_DIR="${3:-runs/myopsnet_seed1234}"
SEED="${4:-1234}"
EPOCHS="${5:-300}"
BATCH_SIZE="${6:-16}"
AMP="${7:-auto}"

echo "================ Step 1: Training ($EPOCHS Epochs) ================"
python train.py \
    --data_root "$DATA_ROOT" \
    --split_dir "$SPLIT_DIR" \
    --run_dir "$RUN_DIR" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --base_lr 0.0003333 \
    --weight_decay 0.0001 \
    --amp "$AMP"

echo "================ Step 2: Test Volume Inference (76 Cases) ================"
python predict.py \
    --checkpoint "$RUN_DIR/checkpoints/myopsnet_seed${SEED}.pt" \
    --run_dir "$RUN_DIR" \
    --data_root "$DATA_ROOT" \
    --split_dir "$SPLIT_DIR" \
    --batch_size 8 \
    --amp "$AMP"

echo "================ Step 3: Standalone Evaluation (myops380_voxel_v3) ================"
python evaluate_predictions.py \
    --pred_dir "$RUN_DIR/test_predictions" \
    --output_dir "$RUN_DIR/evaluation" \
    --data_root "$DATA_ROOT" \
    --split_dir "$SPLIT_DIR/test_vol.txt"

echo "================ Benchmark Completed Successfully! ================"
echo "Results saved to: $RUN_DIR/evaluation/"
