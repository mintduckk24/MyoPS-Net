param(
    [string]$Checkpoint = "runs/myopsnet_seed1234/checkpoints/myopsnet_seed1234.pt",
    [string]$DataRoot = "D:/NCKH/Dataset/MyoPS380/Processed_data",
    [string]$SplitDir = "D:/NCKH/SCAR/preprocessing/splits/myops380_no_val",
    [string]$RunDir = "runs/myopsnet_seed1234",
    [int]$Seed = 1234,
    [int]$Epochs = 300,
    [int]$BatchSize = 16,
    [string]$Amp = "auto",
    [int]$EpochsPerRun = 0
)

$ErrorActionPreference = "Stop"

Write-Host "Resuming MyoPS-Net Training from $Checkpoint..." -ForegroundColor Cyan

python train.py `
    --resume "$Checkpoint" `
    --data_root "$DataRoot" `
    --split_dir "$SplitDir" `
    --run_dir "$RunDir" `
    --seed $Seed `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --base_lr 0.0003333 `
    --weight_decay 0.0001 `
    --amp "$Amp" `
    --epochs_per_run $EpochsPerRun
