param(
    [string]$PredDir = "runs/myopsnet_seed1234/test_predictions",
    [string]$OutputDir = "runs/myopsnet_seed1234/evaluation",
    [string]$DataRoot = "D:/NCKH/Dataset/MyoPS380/Processed_data",
    [string]$SplitDir = "D:/NCKH/SCAR/preprocessing/splits/myops380_no_val/test_vol.txt"
)

$ErrorActionPreference = "Stop"

Write-Host "Running Standalone Evaluation (myops380_voxel_v3)..." -ForegroundColor Cyan

python evaluate_predictions.py `
    --pred_dir "$PredDir" `
    --output_dir "$OutputDir" `
    --data_root "$DataRoot" `
    --split_dir "$SplitDir"
