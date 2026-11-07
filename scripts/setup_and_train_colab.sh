#!/bin/bash
# ==============================================================================
# Script cài đặt môi trường và huấn luyện cho MyoPS-Net trên Google Colab / GPU G4
# Repository: https://github.com/thanhquan123hi1/MyoPS-Net.git (Nhánh: main)
# Protocol: myops380_train304_test76_last_v1 (304 train / 76 test / 300 epochs / no-val)
# ==============================================================================

set -e

echo "=== [1/7] Kiểm tra và Kích hoạt GPU NVIDIA ==="
cd /content

# Tự động liên kết thư viện driver NVIDIA nếu có trên Colab (/usr/lib64-nvidia)
if [ -d "/usr/lib64-nvidia" ]; then
    if ! grep -q "/usr/lib64-nvidia" /etc/ld.so.conf.d/nvidia.conf 2>/dev/null; then
        echo "/usr/lib64-nvidia" >> /etc/ld.so.conf.d/nvidia.conf
        ldconfig 2>/dev/null || true
    fi
    export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH}
fi
if ! grep -q "/usr/lib64-nvidia" /root/.bashrc 2>/dev/null; then
    echo 'export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH' >> /root/.bashrc
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ CẢNH BÁO: Chưa tìm thấy NVIDIA GPU. Hãy kiểm tra lại Runtime > Change runtime type > GPU!"
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

echo "=== [2/7] Mount Google Drive (Lưu Checkpoint & Kết quả an toàn) ==="
python3 -c "
import os
if not os.path.exists('/content/drive/MyDrive'):
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print('✅ Mount Google Drive thành công!')
    except Exception as e:
        print('⚠️ Không thể mount Drive tự động (bỏ qua nếu không dùng Colab):', e)
" || true

echo "=== [3/7] Clone Repository MyoPS-Net (Nhánh: main) ==="
if [ ! -d "/content/MyoPS-Net" ]; then
    git clone https://github.com/thanhquan123hi1/MyoPS-Net.git /content/MyoPS-Net
else
    echo "Thư mục /content/MyoPS-Net đã tồn tại. Tiến hành cập nhật nhánh main..."
    cd /content/MyoPS-Net
    git checkout main
    git pull origin main || true
fi
cd /content/MyoPS-Net

echo "=== [4/7] Cài đặt Dependencies cho MyoPS-Net ==="
python3 -m pip install --upgrade pip -q
python3 -m pip install -r requirements.txt gdown -q

echo "=== [5/7] Tải Dataset MyoPS380 từ Thư mục Google Drive ==="
# Thư mục Google Drive: https://drive.google.com/drive/folders/13T6892xy-pUwC1BSjGl4sRxoypMu-SIS
DATASET_FOLDER_ID="13T6892xy-pUwC1BSjGl4sRxoypMu-SIS"
DOWNLOAD_DIR="/content/dataset_download"
TARGET_DATASET_DIR="/content/MyoPS-Net/MyoPS380_dataset"

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$TARGET_DATASET_DIR"

# Tái sử dụng dataset nếu trước đó đã tải bên /content/SCAR/MyoPS380_dataset
if [ -d "/content/SCAR/MyoPS380_dataset/Processed_data/bSSFP" ] && [ -z "$(ls -A "$TARGET_DATASET_DIR" 2>/dev/null)" ]; then
    echo "✅ Phát hiện dataset đã có sẵn tại /content/SCAR/MyoPS380_dataset/Processed_data"
    LOCAL_DATASET="/content/SCAR/MyoPS380_dataset/Processed_data"
else
    # Tải toàn bộ thư mục Google Drive nếu chưa tải
    if [ -z "$(ls -A "$DOWNLOAD_DIR" 2>/dev/null)" ] && [ -z "$(ls -A "$TARGET_DATASET_DIR" 2>/dev/null)" ]; then
        echo "Đang tải toàn bộ thư mục dataset từ Google Drive (Folder ID: $DATASET_FOLDER_ID)..."
        gdown --folder --id "$DATASET_FOLDER_ID" -O "$DOWNLOAD_DIR" --remaining-ok || {
            echo "⚠️ Tải folder qua gdown gặp cảnh báo, tiếp tục kiểm tra các tệp đã tải..."
        }
    fi

    # Tự động giải nén bất kỳ tệp zip nào có trong thư mục tải về
    for zipfile in $(find "$DOWNLOAD_DIR" -maxdepth 2 -name "*.zip" 2>/dev/null); do
        echo "Đang giải nén $zipfile vào $TARGET_DATASET_DIR ..."
        command -v unzip &>/dev/null || (apt-get update -qq && apt-get install -y unzip)
        unzip -q -o "$zipfile" -d "$TARGET_DATASET_DIR/"
    done

    # Nếu trong thư mục tải về đã là thư mục dữ liệu giải nén sẵn
    if [ -d "$DOWNLOAD_DIR/Processed_data" ] && [ ! -d "$TARGET_DATASET_DIR/Processed_data" ]; then
        cp -r "$DOWNLOAD_DIR/Processed_data" "$TARGET_DATASET_DIR/"
    fi

    # Cơ chế tự động định vị thư mục dữ liệu chuẩn (thư mục chứa bSSFP, LGE, T2w)
    FOUND_BSSFP=$(find "$TARGET_DATASET_DIR" "$DOWNLOAD_DIR" -type d -name "bSSFP" 2>/dev/null | head -n 1)
    if [ -n "$FOUND_BSSFP" ]; then
        LOCAL_DATASET=$(dirname "$FOUND_BSSFP")
        echo "✅ Đã xác định thư mục dataset chuẩn tại: $LOCAL_DATASET"
    else
        LOCAL_DATASET="/content/MyoPS-Net/MyoPS380_dataset/Processed_data"
        echo "⚠️ Chưa phát hiện thư mục bSSFP, đặt mặc định: $LOCAL_DATASET"
    fi
fi

echo "=== [6/7] Kiểm tra Split Manifest (304 Train / 76 Test - No Validation) ==="
SPLIT_DIR="/content/MyoPS-Net/preprocessing/splits/myops380_no_val"
if [ ! -f "$SPLIT_DIR/train.txt" ] || [ ! -f "$SPLIT_DIR/test_vol.txt" ]; then
    echo "❌ Không tìm thấy train.txt hoặc test_vol.txt trong $SPLIT_DIR"
    exit 1
fi
echo "✅ Đã xác nhận bộ split chuẩn tại: $SPLIT_DIR"
echo "   - Số lát cắt train: $(wc -l < "$SPLIT_DIR/train.txt")"
echo "   - Số ca bệnh test:  $(wc -l < "$SPLIT_DIR/test_vol.txt")"

echo "=== [7/7] Cài đặt tmux và Kiểm tra PyTorch ==="
command -v tmux &> /dev/null || (apt-get update -qq && apt-get install -y tmux)
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Định cấu hình thư mục lưu kết quả (ưu tiên lưu vào Google Drive nếu đã mount)
SEED=1234
if [ -d "/content/drive/MyDrive" ]; then
    RUN_DIR="/content/drive/MyDrive/Quan/MyoPS-Net/myopsnet_seed${SEED}"
else
    RUN_DIR="/content/MyoPS-Net/runs/myopsnet_seed${SEED}"
fi
CKPT_PATH="$RUN_DIR/checkpoints/myopsnet_seed${SEED}.pt"

echo "=== HOÀN TẤT THIẾT LẬP MÔI TRƯỜNG CHO MYOPS-NET ==="
echo ""
echo "💡 HƯỚNG DẪN DÙNG TMUX TRÊN COLAB CLI (Không lo mất kết nối):"
echo "  1. Tạo session tmux mới:"
echo "     tmux new -s train_myopsnet"
echo "  2. Chạy lệnh train bên trong tmux:"
echo "     cd /content/MyoPS-Net"
echo "     python3 train.py --data_root \"$LOCAL_DATASET\" --split_dir \"$SPLIT_DIR\" --run_dir \"$RUN_DIR\" --seed $SEED --epochs 300 --batch_size 16 --base_lr 0.0003333 --weight_decay 0.0001 --amp auto"
echo "  3. Tách ra khỏi tmux (detach): Nhấn Ctrl+B rồi thả ra, sau đó nhấn D"
echo "  4. Xem lại tiến trình: tmux attach -t train_myopsnet"
echo ""

# === CHẠY HUẤN LUYỆN 300 EPOCHS + SUY LUẬN 76 CA + ĐÁNH GIÁ TỰ ĐỘNG ===
cd /content/MyoPS-Net

if [ -f "$CKPT_PATH" ]; then
    echo "🔄 Phát hiện checkpoint tại $CKPT_PATH -> Tự động RESUME huấn luyện..."
    python3 /content/MyoPS-Net/train.py \
        --resume "$CKPT_PATH" \
        --data_root "$LOCAL_DATASET" \
        --split_dir "$SPLIT_DIR" \
        --run_dir "$RUN_DIR" \
        --seed "$SEED" \
        --epochs 300 \
        --batch_size 16 \
        --base_lr 0.0003333 \
        --weight_decay 0.0001 \
        --amp auto
else
    # Nếu thư mục RUN_DIR từng tạo dở dang mà chưa kịp lưu checkpoint đầu tiên thì dọn sạch để tránh lỗi FileExistsError
    if [ -d "$RUN_DIR" ] && [ -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]; then
        echo "⚠️ Thư mục $RUN_DIR có tệp log cũ nhưng chưa có checkpoint, tiến hành dọn dẹp để chạy mới..."
        rm -rf "${RUN_DIR:?}"/*
    fi
    mkdir -p "$RUN_DIR"
    echo "🚀 Bắt đầu huấn luyện mới MyoPS-Net (300 Epochs, No Validation)..."
    python3 /content/MyoPS-Net/train.py \
        --data_root "$LOCAL_DATASET" \
        --split_dir "$SPLIT_DIR" \
        --run_dir "$RUN_DIR" \
        --seed "$SEED" \
        --epochs 300 \
        --batch_size 16 \
        --base_lr 0.0003333 \
        --weight_decay 0.0001 \
        --amp auto
fi

echo "=== [BƯỚC TIẾP THEO] Suy luận trên 76 ca kiểm thử (predict.py) ==="
python3 /content/MyoPS-Net/predict.py \
    --checkpoint "$CKPT_PATH" \
    --run_dir "$RUN_DIR" \
    --data_root "$LOCAL_DATASET" \
    --split_dir "$SPLIT_DIR" \
    --batch_size 8 \
    --amp auto \
    --require_complete \
    --strict

echo "=== [BƯỚC CUỐI] Đánh giá kết quả theo chuẩn SCAR (myops380_voxel_v3) ==="
python3 /content/MyoPS-Net/evaluate_predictions.py \
    --pred_dir "$RUN_DIR/test_predictions" \
    --output_dir "$RUN_DIR/evaluation" \
    --data_root "$LOCAL_DATASET" \
    --split_dir "$SPLIT_DIR/test_vol.txt"

echo "🎉 HOÀN TẤT TOÀN BỘ QUY TRÌNH BENCHMARK MYOPS-NET!"
echo "📂 Checkpoint: $CKPT_PATH"
echo "📊 Báo cáo kết quả: $RUN_DIR/evaluation/metrics.json và $RUN_DIR/evaluation/per_case.csv"
