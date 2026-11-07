param(
    [string]$Checkpoint = "runs/myopsnet_seed1234/checkpoints/myopsnet_seed1234.pt",
    [string]$RunDir = "runs/myopsnet_seed1234",
    [string]$DataRoot = "D:/NCKH/Dataset/MyoPS380/Processed_data",
    [string]$SplitDir = "D:/NCKH/SCAR/preprocessing/splits/myops380_no_val",
    [int]$BatchSize = 8,
    [string]$Amp = "auto"
)

$ErrorActionPreference = "Stop"

Write-Host "Running Deterministic Inference on 76 Test Volumes..." -ForegroundColor Cyan

python predict.py `
    --checkpoint "$Checkpoint" `
    --run_dir "$RunDir" `
    --data_root "$DataRoot" `
    --split_dir "$SplitDir" `
    --batch_size $BatchSize `
    --amp "$Amp"
