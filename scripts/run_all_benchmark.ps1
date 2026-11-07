param(
    [string]$DataRoot = "D:/NCKH/Dataset/MyoPS380/Processed_data",
    [string]$SplitDir = "D:/NCKH/SCAR/preprocessing/splits/myops380_no_val",
    [string]$RunDir = "runs/myopsnet_seed1234",
    [int]$Seed = 1234,
    [int]$Epochs = 300,
    [int]$BatchSize = 16,
    [string]$Amp = "auto"
)

$ErrorActionPreference = "Stop"

Write-Host "================ Step 1: Training ($Epochs Epochs) ================" -ForegroundColor Cyan
python train.py `
    --data_root "$DataRoot" `
    --split_dir "$SplitDir" `
    --run_dir "$RunDir" `
    --seed $Seed `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --base_lr 0.0003333 `
    --weight_decay 0.0001 `
    --amp "$Amp"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Training failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "================ Step 2: Test Volume Inference (76 Cases) ================" -ForegroundColor Cyan
python predict.py `
    --checkpoint "$RunDir/checkpoints/myopsnet_seed$Seed.pt" `
    --run_dir "$RunDir" `
    --data_root "$DataRoot" `
    --split_dir "$SplitDir" `
    --batch_size 8 `
    --amp "$Amp"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Inference failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "================ Step 3: Standalone Evaluation (myops380_voxel_v3) ================" -ForegroundColor Cyan
python evaluate_predictions.py `
    --pred_dir "$RunDir/test_predictions" `
    --output_dir "$RunDir/evaluation" `
    --data_root "$DataRoot" `
    --split_dir "$SplitDir/test_vol.txt"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Evaluation failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "================ Benchmark Completed Successfully! ================" -ForegroundColor Green
Write-Host "Results saved to: $RunDir/evaluation/"
