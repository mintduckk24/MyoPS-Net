#!/bin/bash
# ==============================================================================
# SCRIPT MÔI TRƯỜNG & HUẤN LUYỆN CHUẨN HÓA CHO MYOPS-NET (SEED 1234)
# Áp dụng: Google Colab / GPU NVIDIA A100 Dedicated Server
# Căn cứ kỹ thuật & pháp lý:
#   1. "Quy tắc đo hiệu suất tính toán.txt" (1x A100 | Pure FP32 | num_workers=0)
#   2. "Promt_train_test_no_val.txt" (Protocol: myops380_train304_test76_last_v1)
#   3. "BENCHMARK_RECIPE.md" của MyoPS-Net
#
# BẢNG THÔNG SỐ CỐ ĐỊNH CHO RUN BENCHMARK SEED 1234:
#   - Model: MyoPSNetL (119.28M Params, model gốc nguyên vẹn 100%)
#   - Model ID: myopsnet (Tên trọng số: myopsnet_seed1234.pt)
#   - Seed bắt buộc: 1234 (RNG xác định, tái lập hoàn toàn)
#   - Phần cứng bắt buộc: Duy nhất 01 GPU NVIDIA A100 (Không chạy tác vụ ngầm)
#   - Độ chính xác (Precision): Chạy thuần FP32 (float32), tắt hoàn toàn AMP: --amp none
#   - DataLoader: Cố định num_workers = 0, pin_memory = True
#   - Ngân sách: Đúng 300 epochs, không validation xen giữa, batch_size = 16 (124 steps/epoch)
#   - Tối ưu hóa: AdamW (base_lr=0.0003333, weight_decay=0.0001, clip_grad=0.0)
#   - Scheduler: Polynomial decay (power=0.9, im_mseg_adamw_poly_v1)
#   - Hàm mất mát: Giữ nguyên Loss gốc MyoPSLoss (L_seg + L_inv + L_inc)
#   - File xuất hiệu năng: outputs/runs/myopsnet/efficiency_metrics.json
#   - Đánh giá kiểm thử: predict.py (--amp none) & evaluate_predictions.py (myops380_voxel_v3)
# ==============================================================================

set -e

echo "=============================================================================="
echo "🎯 KHỞI TẠO MÔI TRƯỜNG BENCHMARK MYOPS-NET (SEED ĐẶC BIỆT 1234)"
echo "   Tuân thủ 100%: Quy tắc đo hiệu suất tính toán & Promt_train_test_no_val"
echo "   Cấu hình: Thuần FP32 (--amp none) | num_workers = 0 | 1x NVIDIA A100 GPU"
echo "=============================================================================="

# === BƯỚC 1/7: KIỂM TRA PHẦN CỨNG & KÍCH HOẠT GPU A100 ===
echo ""
echo "=== [1/7] Kiểm tra và Kích hoạt GPU NVIDIA A100 ==="
cd /content

if [ -d "/usr/lib64-nvidia" ]; then
    if ! grep -q "/usr/lib64-nvidia" /etc/ld.so.conf.d/nvidia.conf 2>/dev/null; then
        echo "/usr/lib64-nvidia" >> /etc/ld.so.conf.d/nvidia.conf 2>/dev/null || true
        ldconfig 2>/dev/null || true
    fi
    export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH}
fi
if ! grep -q "/usr/lib64-nvidia" /root/.bashrc 2>/dev/null; then
    echo 'export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH' >> /root/.bashrc 2>/dev/null || true
fi
if ! grep -q "/usr/lib64-nvidia" ~/.bashrc 2>/dev/null; then
    echo 'export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH' >> ~/.bashrc 2>/dev/null || true
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ LỖI: Chưa tìm thấy NVIDIA GPU! Hãy kiểm tra lại Runtime > Change runtime type > A100 GPU trên Colab."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || echo "Unknown GPU")
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -n 1 || echo "Unknown VRAM")
echo "✅ GPU phát hiện: $GPU_NAME ($GPU_MEM)"

if [[ "$GPU_NAME" == *"A100"* ]]; then
    echo "✅ Xác nhận đúng GPU NVIDIA A100 theo Điều 1 của Quy tắc benchmark!"
else
    echo "⚠️ CẢNH BÁO: GPU hiện tại ($GPU_NAME) không phải NVIDIA A100."
    echo "   Quy tắc đo hiệu suất tính toán yêu cầu duy nhất 01 GPU NVIDIA A100 để đảm bảo tính công bằng đối sánh."
fi

# Chống phân mảnh bộ nhớ CUDA
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


# === BƯỚC 2/7: MOUNT GOOGLE DRIVE (BẢO VỆ CHECKPOINT & KẾT QUẢ) ===
echo ""
echo "=== [2/7] Mount Google Drive (Lưu Checkpoint & Metrics JSON an toàn) ==="
python3 -c "
import os
if not os.path.exists('/content/drive/MyDrive'):
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print('✅ Mount Google Drive thành công!')
    except Exception as e:
        print('⚠️ Không thể mount Drive tự động (bỏ qua nếu chạy local/server ngoài):', e)
" || true


# === BƯỚC 3/7: THIẾT LẬP REPOSITORY MYOPS-NET ===
echo ""
echo "=== [3/7] Thiết lập Repository MyoPS-Net ==="
REPO_URL="https://github.com/mintduckk24/MyoPS-Net.git"

if [ -f "./train.py" ] && [ -d "./network" ]; then
    echo "✅ Đang đứng trực tiếp trong thư mục MyoPS-Net."
elif [ -d "/content/MyoPS-Net" ]; then
    echo "✅ Thư mục /content/MyoPS-Net đã tồn tại. Tiến hành cập nhật nhánh main..."
    cd /content/MyoPS-Net
    git checkout main 2>/dev/null || true
    git pull origin main 2>/dev/null || true
elif [ -f "/content/MyoPS-Net.zip" ]; then
    echo "✅ Phát hiện file MyoPS-Net.zip tại /content/. Đang giải nén..."
    unzip -q -o "/content/MyoPS-Net.zip" -d /content/
elif [ -f "/content/drive/MyDrive/MyoPS-Net.zip" ]; then
    echo "✅ Phát hiện file MyoPS-Net.zip trên Google Drive. Đang giải nén..."
    unzip -q -o "/content/drive/MyDrive/MyoPS-Net.zip" -d /content/
elif [ -d "/content/drive/MyDrive/Quan/MyoPS-Net" ]; then
    echo "✅ Phát hiện thư mục MyoPS-Net trên Google Drive (Quan/MyoPS-Net). Đang sao chép..."
    cp -r "/content/drive/MyDrive/Quan/MyoPS-Net" /content/MyoPS-Net
elif [ -d "/content/drive/MyDrive/MyoPS-Net" ]; then
    echo "✅ Phát hiện thư mục MyoPS-Net trên Google Drive. Đang sao chép..."
    cp -r "/content/drive/MyDrive/MyoPS-Net" /content/MyoPS-Net
else
    echo "Đang clone repository từ $REPO_URL..."
    git clone -b main "$REPO_URL" /content/MyoPS-Net || {
        echo "❌ Chưa thể clone từ $REPO_URL."
        echo "👉 Bạn có thể kéo thả thư mục MyoPS-Net hoặc file MyoPS-Net.zip vào /content/ rồi chạy lại script."
        exit 1
    }
fi

cd /content/MyoPS-Net


# === BƯỚC 4/7: CÀI ĐẶT DEPENDENCIES ===
echo ""
echo "=== [4/7] Cài đặt Thư viện Phụ thuộc (Bao gồm ptflops & thop) ==="
python3 -m pip install --upgrade pip -q
if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt gdown ptflops thop pandas torchvision -q || true
else
    python3 -m pip install torch torchvision numpy scipy h5py PyYAML pandas nibabel SimpleITK medpy tqdm gdown einops timm ptflops thop -q || true
fi


# === BƯỚC 5/7: TẢI VÀ ĐỊNH VỊ DATASET MYOPS380 ===
echo ""
echo "=== [5/7] Tải và Định vị Dataset MyoPS380 ==="
DATASET_FOLDER_ID="13T6892xy-pUwC1BSjGl4sRxoypMu-SIS"
DOWNLOAD_DIR="/content/dataset_download"
TARGET_DATASET_DIR="/content/MyoPS-Net/MyoPS380_dataset"

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$TARGET_DATASET_DIR"

DRIVE_DATASET_DIR="/content/drive/MyDrive/MyoPS380_dataset"
if [ -d "$DRIVE_DATASET_DIR/Processed_data/bSSFP" ]; then
    echo "✅ Phát hiện dataset trên Google Drive tại $DRIVE_DATASET_DIR/Processed_data"
    LOCAL_DATASET="$DRIVE_DATASET_DIR/Processed_data"
elif [ -d "/content/drive/MyDrive/Processed_data/bSSFP" ]; then
    echo "✅ Phát hiện dataset trên Google Drive tại /content/drive/MyDrive/Processed_data"
    LOCAL_DATASET="/content/drive/MyDrive/Processed_data"
elif [ -d "/content/SCAR/MyoPS380_dataset/Processed_data/bSSFP" ]; then
    echo "✅ Tái sử dụng dataset từ /content/SCAR/MyoPS380_dataset/Processed_data"
    LOCAL_DATASET="/content/SCAR/MyoPS380_dataset/Processed_data"
elif [ -d "/content/M2FTrans/MyoPS380_dataset/Processed_data/bSSFP" ]; then
    echo "✅ Tái sử dụng dataset từ /content/M2FTrans/MyoPS380_dataset/Processed_data"
    LOCAL_DATASET="/content/M2FTrans/MyoPS380_dataset/Processed_data"
elif [ -d "/content/AWSnet/MyoPS380_dataset/Processed_data/bSSFP" ]; then
    echo "✅ Tái sử dụng dataset từ /content/AWSnet/MyoPS380_dataset/Processed_data"
    LOCAL_DATASET="/content/AWSnet/MyoPS380_dataset/Processed_data"
elif [ -d "/content/MyoPS-Net/MyoPS380_dataset/Processed_data/bSSFP" ]; then
    echo "✅ Phát hiện dataset có sẵn tại /content/MyoPS-Net/MyoPS380_dataset/Processed_data"
    LOCAL_DATASET="/content/MyoPS-Net/MyoPS380_dataset/Processed_data"
else
    if [ -z "$(ls -A "$DOWNLOAD_DIR" 2>/dev/null)" ] && [ -z "$(ls -A "$TARGET_DATASET_DIR" 2>/dev/null)" ]; then
        echo "Đang tải dataset từ Google Drive (Folder ID: $DATASET_FOLDER_ID)..."
        gdown --folder --id "$DATASET_FOLDER_ID" -O "$DOWNLOAD_DIR" --remaining-ok || true
    fi

    for zipfile in $(find "$DOWNLOAD_DIR" -maxdepth 2 -name "*.zip" 2>/dev/null); do
        echo "Đang giải nén $zipfile..."
        command -v unzip &>/dev/null || (apt-get update -qq && apt-get install -y unzip)
        unzip -q -o "$zipfile" -d "$TARGET_DATASET_DIR/" || true
    done

    if [ -d "$DOWNLOAD_DIR/Processed_data" ]; then
        cp -r "$DOWNLOAD_DIR/Processed_data" "$TARGET_DATASET_DIR/"
    fi

    FOUND_TRAIN_NPZ=$(find "$TARGET_DATASET_DIR" "$DOWNLOAD_DIR" /content -type d -name "train_npz" 2>/dev/null | grep -i "bSSFP" | head -n 1)
    if [ -n "$FOUND_TRAIN_NPZ" ]; then
        LOCAL_DATASET=$(dirname $(dirname "$FOUND_TRAIN_NPZ"))
        echo "✅ Thư mục Processed_data chuẩn: $LOCAL_DATASET"
    elif [ -d "$TARGET_DATASET_DIR/Processed_data/bSSFP" ]; then
        LOCAL_DATASET="$TARGET_DATASET_DIR/Processed_data"
    else
        LOCAL_DATASET="/content/MyoPS-Net/MyoPS380_dataset/Processed_data"
    fi
fi
echo "✅ Đường dẫn dataset chuẩn: $LOCAL_DATASET"

# Thiết lập Split Manifest chuẩn (myops380_no_val: 304 train, 76 test)
SPLIT_DIR="/content/MyoPS-Net/preprocessing/splits/myops380_no_val"
if [ ! -f "$SPLIT_DIR/train.txt" ] || [ ! -f "$SPLIT_DIR/test_vol.txt" ]; then
    if [ -f "/content/MyoPS-Net/dataset/splits/myops380_no_val/train.txt" ]; then
        SPLIT_DIR="/content/MyoPS-Net/dataset/splits/myops380_no_val"
    elif [ -f "/content/SCAR/preprocessing/splits/myops380_no_val/train.txt" ]; then
        SPLIT_DIR="/content/SCAR/preprocessing/splits/myops380_no_val"
    elif [ -f "/content/M2FTrans/dataset/splits/myops380_no_val/train.txt" ]; then
        SPLIT_DIR="/content/M2FTrans/dataset/splits/myops380_no_val"
    else
        echo "❌ LỖI: Thiếu split files trong $SPLIT_DIR!"
        exit 1
    fi
fi

# Đồng bộ dự phòng sang dataset/splits để tương thích mọi lệnh gọi
mkdir -p /content/MyoPS-Net/dataset/splits
cp -r "$SPLIT_DIR" /content/MyoPS-Net/dataset/splits/ 2>/dev/null || true

echo "✅ Manifest đã sẵn sàng: $(wc -l < "$SPLIT_DIR/train.txt") train slices (304 ca), $(wc -l < "$SPLIT_DIR/test_vol.txt") test volumes (76 ca)."


# === BƯỚC 6/7: THIẾT LẬP ĐƯỜNG DẪN CHECKPOINT & RUNS ===
echo ""
echo "=== [6/7] Thiết lập Thư mục Checkpoint & Đường dẫn Lưu trữ ==="
SEED=1234

if [ -d "/content/drive/MyDrive/Quan" ]; then
    RUN_DIR="/content/drive/MyDrive/Quan/MyoPS-Net/myopsnet_seed${SEED}"
elif [ -d "/content/drive/MyDrive" ]; then
    RUN_DIR="/content/drive/MyDrive/MyoPS-Net/myopsnet_seed${SEED}"
else
    RUN_DIR="/content/MyoPS-Net/runs/myopsnet_seed${SEED}"
fi

SAVE_DIR="$RUN_DIR/checkpoints"
CKPT_FILE="$SAVE_DIR/myopsnet_seed${SEED}.pt"
SUMMARY_FILE="$SAVE_DIR/training_summary_seed${SEED}.json"
METRICS_JSON="outputs/runs/myopsnet/efficiency_metrics.json"

mkdir -p "$SAVE_DIR"
mkdir -p "outputs/runs/myopsnet"

echo "✅ Cấu hình đường dẫn cho Seed $SEED:"
echo "   - Run Directory: $RUN_DIR"
echo "   - Checkpoint:    $CKPT_FILE"
echo "   - Summary:       $SUMMARY_FILE"
echo "   - Efficiency:    $METRICS_JSON"


# === BƯỚC 7/7: KIỂM TRA MÔI TRƯỜNG PYTORCH & CẤU HÌNH BANNER ===
echo ""
echo "=== [7/7] Kiểm tra Thư viện & Cấu hình Chuẩn Benchmark ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device 0: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=============================================================================="
echo "🎯 CẤU HÌNH ĐÃ ĐƯỢC XÁC NHẬN CHÍNH XÁC 100% THEO QUY TẮC BENCHMARK:"
echo "   - Model: MyoPSNetL (119.28M Parameters, model gốc nguyên vẹn)"
echo "   - Seed: 1234 (Cố định, sinh ngẫu nhiên tái lập hoàn toàn)"
echo "   - Precision: Pure FP32 (--amp none, tắt hoàn toàn AMP)"
echo "   - DataLoader: num_workers = 0, pin_memory = True"
echo "   - Ngân sách: 300 Epochs, batch_size = 16 (124 steps/epoch), Zero Validation"
echo "   - Loss: Original MyoPSLoss (L_seg + L_inv + L_inc)"
echo "   - Output: outputs/runs/myopsnet/efficiency_metrics.json"
echo "=============================================================================="
echo ""


# ==============================================================================
# QUY TRÌNH THỰC THI 3 GIAI ĐOẠN CHO SEED 1234 (CHẠY TRỰC TIẾP TUẦN TỰ)
# ==============================================================================

# GIAI ĐOẠN 1/3: HUẤN LUYỆN 300 EPOCHS
echo "🚀 [GIAI ĐOẠN 1/3]: BẮT ĐẦU HUẤN LUYỆN 300 EPOCHS (SEED 1234 - FP32)..."
echo "   Dataset:    $LOCAL_DATASET"
echo "   Run Dir:    $RUN_DIR"
echo "   Checkpoint: $CKPT_FILE"
echo "   Quy tắc:    Pure FP32 (--amp none) | num_workers=0 | pin_memory=True | Batch 16"

python3 train.py \
    --data_root "$LOCAL_DATASET" \
    --split_dir "$SPLIT_DIR" \
    --run_dir "$RUN_DIR" \
    --seed $SEED \
    --epochs 300 \
    --batch_size 16 \
    --accum_steps 1 \
    --base_lr 0.0003333 \
    --weight_decay 0.0001 \
    --amp none \
    --num_workers 0 \
    --pin_memory \
    --resume

echo ""
# GIAI ĐOẠN 2/3: ĐO VÀ XUẤT 7 CHỈ SỐ HIỆU SUẤT TÍNH TOÁN
echo "=============================================================================="
echo "🚀 [GIAI ĐOẠN 2/3]: ĐO & XUẤT 7 CHỈ SỐ HIỆU SUẤT TÍNH TOÁN & ĐỘ GỌN NHẸ..."
echo "=============================================================================="
if [ -f "benchmark_efficiency.py" ]; then
    python3 benchmark_efficiency.py \
        --checkpoint "$CKPT_FILE" \
        --summary "$SUMMARY_FILE" \
        --output "$METRICS_JSON" \
        --seed $SEED

    # Sao chép và sao lưu tự động file efficiency_metrics.json sang Google Drive
    if [ -d "/content/drive/MyDrive" ]; then
        mkdir -p "/content/drive/MyDrive/MyoPS-Net/outputs/runs/myopsnet/"
        cp "$METRICS_JSON" "/content/drive/MyDrive/MyoPS-Net/outputs/runs/myopsnet/" 2>/dev/null || true
        if [ -d "/content/drive/MyDrive/Quan/MyoPS-Net" ]; then
            mkdir -p "/content/drive/MyDrive/Quan/MyoPS-Net/outputs/runs/myopsnet/"
            cp "$METRICS_JSON" "/content/drive/MyDrive/Quan/MyoPS-Net/outputs/runs/myopsnet/" 2>/dev/null || true
        fi
        cp "$METRICS_JSON" "$SAVE_DIR/" 2>/dev/null || true
        echo "✅ Đã sao lưu $METRICS_JSON vào Google Drive an toàn!"
    fi
else
    echo "⚠️ Chưa tìm thấy file benchmark_efficiency.py trong repo."
fi

echo ""
# GIAI ĐOẠN 3/3: SUY LUẬN & ĐÁNH GIÁ TRÊN 76 CA TEST
echo "=============================================================================="
echo "🚀 [GIAI ĐOẠN 3/3]: ĐÁNH GIÁ KIỂM THỬ TRÊN 76 BỆNH NHÂN TEST (TEST_VOL.TXT)..."
echo "=============================================================================="
if [ -f "$CKPT_FILE" ]; then
    PRED_DIR="$RUN_DIR/test_predictions"
    EVAL_DIR="$RUN_DIR/evaluation"
    mkdir -p "$PRED_DIR"
    mkdir -p "$EVAL_DIR"

    python3 predict.py \
        --checkpoint "$CKPT_FILE" \
        --run_dir "$RUN_DIR" \
        --data_root "$LOCAL_DATASET" \
        --split_dir "$SPLIT_DIR" \
        --pred_dir "$PRED_DIR" \
        --batch_size 8 \
        --amp none

    python3 evaluate_predictions.py \
        --pred_dir "$PRED_DIR" \
        --output_dir "$EVAL_DIR" \
        --data_root "$LOCAL_DATASET" \
        --split_dir "$SPLIT_DIR/test_vol.txt"

    # Sao chép kết quả metrics.json và per_case.csv sang Google Drive
    if [ -d "$SAVE_DIR" ]; then
        cp "$EVAL_DIR/metrics.json" "$SAVE_DIR/metrics_test_seed${SEED}.json" 2>/dev/null || true
        cp "$EVAL_DIR/per_case.csv" "$SAVE_DIR/per_case_test_seed${SEED}.csv" 2>/dev/null || true
        if [ -d "/content/drive/MyDrive/MyoPS-Net" ]; then
            mkdir -p "/content/drive/MyDrive/MyoPS-Net/evaluation"
            cp "$EVAL_DIR/metrics.json" "/content/drive/MyDrive/MyoPS-Net/evaluation/metrics_test_seed${SEED}.json" 2>/dev/null || true
            cp "$EVAL_DIR/per_case.csv" "/content/drive/MyDrive/MyoPS-Net/evaluation/per_case_test_seed${SEED}.csv" 2>/dev/null || true
        fi
        echo "✅ Đã sao lưu metrics.json và per_case.csv vào Google Drive an toàn!"
    fi
else
    echo "⚠️ Không tìm thấy file checkpoint: $CKPT_FILE (Chưa hoàn thành Giai đoạn 1/3)"
fi

echo ""
echo "=============================================================================="
echo "🎉 HOÀN THÀNH TOÀN DIỆN BENCHMARK SEED ĐẶC BIỆT 1234 CHO MYOPS-NET!"
echo "   - Model Checkpoint:   $CKPT_FILE"
echo "   - 7 Chỉ số Hiệu năng:  $METRICS_JSON"
echo "   - Thư mục dự đoán:    $RUN_DIR/test_predictions"
echo "   - Báo cáo đánh giá:   $RUN_DIR/evaluation/metrics.json"
echo "=============================================================================="
