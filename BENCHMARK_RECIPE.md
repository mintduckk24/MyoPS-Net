# MyoPS-Net Standardized Benchmark Recipe for MyoPS380

Tài liệu hướng dẫn thực thi chuẩn hóa mô hình **MyoPS-Net** theo dataset profile **MyoPS380** (304 train / 76 test / 300 epochs / no validation) để đối chứng trực tiếp với **SCAR M2-Plus**.

Phiên bản được dùng là **MyoPS-Net-L thích nghi với MPC hai lớp** (`myopsnet_l_binary_mpc_v1`), không phải tái lập nguyên phương pháp gốc 100%. Cardiac head dùng background/whole myocardium vì cache không có annotation LV riêng. Hai nhánh LGE/T2, cross-skip, target adapters và loss hiện tại được giữ nguyên. Đây là so sánh các phương pháp dưới recipe chung; không quy toàn bộ chênh lệch kết quả cho riêng kiến trúc.

---

## 1. Tổng quan kiến trúc & các thành phần thích nghi

### 1.1. Kiến trúc mô hình (`MyoPSNetL`)
- **Đầu vào 3 Modality**: Tiếp nhận trực tiếp `bSSFP`, `LGE`, `T2w` mà không tạo kênh giả lập (synthetic) hay padding dư thừa.
- **Stage 1 (`unet_C0`)**: Trích xuất prior toàn bộ cơ tim (whole myocardium prior) 2 lớp (`0: Background`, `1: Whole Myocardium`).
- **Stage 2 (Pathology Encoders & Cross-Modal Decoders)**:
  - `encoder_LGE`: đầu vào 2 kênh (`LGE` + mask cơ tim từ Stage 1).
  - `encoder_T2`: đầu vào 2 kênh (`T2w` + mask cơ tim từ Stage 1).
  - `decoder_LGE`: nhận skip connections từ cả LGE và T2w, dự đoán `{0: Background, 1: Non-scar Myo, 2: Scar}`.
  - `decoder_T2`: nhận skip connections từ cả LGE và T2w, dự đoán `{0: Background, 1: Normal Myo, 2: Edema-inclusive}`.

### 1.2. Ánh xạ nhãn (Target Adapters) & Hợp nhất đầu ra (Canonical Output Fusion)
- **Nhãn gốc MyoPS380**: `{0: Background, 1: Normal Myocardium, 2: Scar, 3: Edema}`.
- **Bộ chuyển đổi nhãn đích**:
  - `target_C0`: Toàn bộ cơ tim ($1 \cup 2 \cup 3 \to 1$), nền $0 \to 0$.
  - `target_LGE`: Sẹo ($2 \to 2$), cơ tim không sẹo ($1 \cup 3 \to 1$), nền $0 \to 0$.
  - `target_T2`: Vùng phù bao hàm ($2 \cup 3 \to 2$), cơ tim lành ($1 \to 1$), nền $0 \to 0$.
- **Hợp nhất suy luận (Canonical 4-class fusion)**:
  - **Scar (ID 3)**: $\text{argmax}(\text{seg\_LGE}) == 2$
  - **Edema-exclusive (ID 2)**: $(\text{argmax}(\text{seg\_T2}) == 2) \land \neg \text{Scar}$
  - **Normal Myocardium (ID 1)**: $(\text{argmax}(\text{seg\_C0}) == 1) \land \neg \text{Scar} \land \neg \text{Edema-exclusive}$
  - **Background (ID 0)**: Các voxel còn lại.

### 1.3. Hàm mất mát (`WCELoss` & Composite Loss)
- Cố định số chiều ma trận trọng số lớp `(B, C, H, W)` để ngăn ngừa triệt để lỗi crash NaN hoặc dimension mismatch khi một batch nhỏ thiếu nhãn tổn thương.
- Hàm mất mát kết hợp: $L_{total} = L_{seg} + \lambda_{inv} L_{inv} + \lambda_{inc} L_{inc}$.
- Cụ thể: $L_{seg}=L_{C0}+2L_{LGE}+2L_{T2}$, mỗi head dùng per-image mean-class Dice + log-frequency WCE; hệ số invariant/inclusive đều 1. Giữ epsilon hiện tại `1e-7` và absent-class WCE weight = 0, là các sửa ổn định số học đã công bố so với source gốc (`1e-10`). Không thay loss bằng CE/batch-Dice của SCAR.

### 1.4. Dữ liệu & Tăng cường ảnh (Augmentation)
- **Dataset Manifest**:
  - 304 ca train (1.972 lát cắt 2D $128 \times 128$) từ `train.txt`.
  - 76 ca test (467 lát cắt, volume $128 \times 128 \times D$) từ `test_vol.txt` theo cache local đã kiểm tra.
- **Augmentation (`RandomGenerator`)**:
  - Đồng bộ tuyệt đối theo chuẩn SCAR / I-MMSeg (phép xoay ngẫu nhiên, lật đa kênh đồng thời, roll shift ngẫu nhiên, nhiễu Gaussian).

### 1.5. Công thức tối ưu hóa (Optimization Recipe)
- **Epochs**: 300 epochs.
- **Batch Size**: 16, `accum_steps=1` (123 batch kích thước 16 + 1 batch cuối kích thước 4 = **124 batch/epoch** $\to$ **37.200 cơ hội cập nhật**). Successful updates có thể ít hơn khi FP16 overflow; scheduler chỉ tiến sau update thành công. Recipe này từ chối accumulation > 1.
- **Optimizer**: AdamW (`base_lr = 0.0003333`, `weight_decay = 0.0001`, `clip_grad = 0.0`).
- **Lịch giảm LR**: Polynomial decay ($\text{power} = 0.9$) theo giao thức `im_mseg_adamw_poly_v1`.
- **AMP**: Tự động bật mixed-precision (`torch.amp.autocast`).
- **No Validation**: Tuyệt đối không chia validation, không chạy val loop trong khi huấn luyện.
- **Khởi tạo**: run mới gọi `model.apply(utils.tools.weights_init)` (Conv Kaiming normal, BatchNorm unit/zero); resume nạp trọng số đã lưu, không khởi tạo lại sau khi nạp.
- **RNG/mặc định**: seed 1234, workers 4, loader generator riêng, seed Python/NumPy worker theo SCAR, `persistent_workers=False`. Tái lập được kiểm tra trong cùng môi trường; không bảo đảm bitwise giữa GPU/phần cứng/phiên bản khác.

### 1.6. Cơ chế Checkpoint & Resumption
- Lưu checkpoint nguyên tử tại `<run_dir>/checkpoints/myopsnet_seed<seed>.pt`.
- Mặc định lưu sau mỗi epoch (`--save_interval 1`), và luôn lưu ở cuối phiên `--epochs_per_run`.
- Bảo toàn đầy đủ: trọng số mô hình, trạng thái optimizer, scheduler step, AMP GradScaler, và multi-framework RNG states (Python, NumPy, PyTorch, CUDA).
- Generator thực sự dùng bởi DataLoader và bộ đếm AMP skipped updates đều được khôi phục. Checkpoint/config ghi model contract, initialization, loss và RNG policy; resume khóa recipe, manifest, train-cache bytes, source hashes và phiên bản runtime. Việc fingerprint chỉ đọc cache train, không chạy model trên test.
- Khi tiếp tục chạy với `--resume <checkpoint>`, mô hình khôi phục chính xác bước đếm và tiếp tục giảm LR mà không bị reset lịch học.
- Checkpoint cũ không có resume contract mới vẫn đọc được cho inference, nhưng bị từ chối resume vào recipe RNG/initialization mới. Bắt đầu run mới trong thư mục rỗng; không ghi đè run cũ để đổi recipe. `--dry_run`/`--max_batches` được ghi `run_kind=debug` trong contract và không được dùng làm kết quả chính thức.

### 1.7. Suy luận & Đánh giá (Evaluation Protocol)
- **Suy luận xác định (`predict.py`)**: Chạy single-pass trên 76 ca test ở chế độ `model.eval()`, không dùng test-time augmentation (TTA), không lọc largest connected component (LCC). Xuất mảng uint8 vào `<run_dir>/test_predictions/<CaseID>_pred.npz` (khóa `'prediction'`).
- **Đánh giá độc lập (`evaluate_predictions.py`)**: Đọc file `.npz` và ground-truth tương ứng, tính toán các chỉ số theo giao thức `myops380_voxel_v3` (Dice, Jaccard, HD95, ASSD trên 5 vùng: Myocardium, Edema, Scar, LGE-abnormal, Normal Myocardium). Xuất báo cáo `per_case.csv` và `metrics.json`.

---

## 2. Hướng dẫn chạy huấn luyện & đánh giá trên máy khác

Khi bạn copy mã nguồn sang máy GPU khác để chạy huấn luyện 300 epochs, bạn có thể thực hiện theo các bước dưới đây.

### Bước 1: Huấn luyện đủ 300 Epochs

#### Trên Windows (PowerShell):
```powershell
# Chạy trực tiếp qua script có sẵn:
.\scripts\train_300epochs.ps1 `
    -DataRoot "D:/NCKH/Dataset/MyoPS380/Processed_data" `
    -SplitDir "D:/NCKH/SCAR/preprocessing/splits/myops380_no_val" `
    -RunDir "runs/myopsnet_seed1234" `
    -Seed 1234

# Hoặc chạy lệnh python trực tiếp:
python train.py `
    --data_root "D:/NCKH/Dataset/MyoPS380/Processed_data" `
    --split_dir "D:/NCKH/SCAR/preprocessing/splits/myops380_no_val" `
    --run_dir "runs/myopsnet_seed1234" `
    --seed 1234 `
    --epochs 300 `
    --batch_size 16 `
    --base_lr 0.0003333 `
    --weight_decay 0.0001 `
    --amp auto
```

#### Trên Linux / macOS (Bash):
```bash
# Chạy trực tiếp qua script:
bash scripts/train_300epochs.sh \
    /path/to/Processed_data \
    /path/to/splits/myops380_no_val \
    runs/myopsnet_seed1234 \
    1234 300 16 auto

# Hoặc chạy lệnh python:
python train.py \
    --data_root "/path/to/Processed_data" \
    --split_dir "/path/to/splits/myops380_no_val" \
    --run_dir "runs/myopsnet_seed1234" \
    --seed 1234 \
    --epochs 300 \
    --batch_size 16 \
    --base_lr 0.0003333 \
    --weight_decay 0.0001 \
    --amp auto
```

---

### Bước 2: Khôi phục huấn luyện nếu bị ngắt quãng (Resume)

Nếu quá trình huấn luyện bị dừng giữa chừng (ví dụ tại epoch 150), chỉ cần thêm cờ `--resume`:

Giữ nguyên seed, batch, epochs tổng, LR, AMP, workers và mọi setting training khi resume. Để chủ động chia phiên mà vẫn giữ lịch 300 epochs, dùng `--epochs 300 --epochs_per_run 50` cho phiên đầu, sau đó `--resume <checkpoint> --epochs 300` để hoàn tất. Không đổi `--epochs` thành số epoch còn lại. Chỉ các tùy chọn phiên/logging (ví dụ `epochs_per_run`, `save_interval`) được phép thay đổi. Các script train/resume hỗ trợ `EpochsPerRun` (PowerShell), hoặc đối số thứ 8 của script train / thứ 9 của script resume (Bash).

#### Trên Windows (PowerShell):
```powershell
.\scripts\resume_training.ps1 `
    -Checkpoint "runs/myopsnet_seed1234/checkpoints/myopsnet_seed1234.pt" `
    -DataRoot "D:/NCKH/Dataset/MyoPS380/Processed_data" `
    -SplitDir "D:/NCKH/SCAR/preprocessing/splits/myops380_no_val" `
    -RunDir "runs/myopsnet_seed1234" `
    -Seed 1234
```

#### Trên Linux (Bash):
```bash
bash scripts/resume_training.sh \
    runs/myopsnet_seed1234/checkpoints/myopsnet_seed1234.pt \
    /path/to/Processed_data \
    /path/to/splits/myops380_no_val \
    runs/myopsnet_seed1234 \
    1234 300 16 auto
```

---

### Bước 3: Chạy suy luận trên 76 ca kiểm thử (Inference)

Sau khi huấn luyện hoàn tất (checkpoint đạt epoch 300):

#### Trên Windows (PowerShell):
```powershell
.\scripts\run_inference.ps1 `
    -Checkpoint "runs/myopsnet_seed1234/checkpoints/myopsnet_seed1234.pt" `
    -RunDir "runs/myopsnet_seed1234" `
    -DataRoot "D:/NCKH/Dataset/MyoPS380/Processed_data" `
    -SplitDir "D:/NCKH/SCAR/preprocessing/splits/myops380_no_val"
```

#### Trên Linux (Bash):
```bash
bash scripts/run_inference.sh \
    runs/myopsnet_seed1234/checkpoints/myopsnet_seed1234.pt \
    runs/myopsnet_seed1234 \
    /path/to/Processed_data \
    /path/to/splits/myops380_no_val
```

Kết quả suy luận dạng `.npz` sẽ được tự động xuất vào: `runs/myopsnet_seed1234/test_predictions/`.

---

### Bước 4: Chấm điểm và xuất báo cáo chỉ số (Evaluation)

Đánh giá các tệp dự đoán theo chuẩn metric của SCAR (`myops380_voxel_v3`):

#### Trên Windows (PowerShell):
```powershell
.\scripts\run_evaluation.ps1 `
    -PredDir "runs/myopsnet_seed1234/test_predictions" `
    -OutputDir "runs/myopsnet_seed1234/evaluation" `
    -DataRoot "D:/NCKH/Dataset/MyoPS380/Processed_data" `
    -SplitDir "D:/NCKH/SCAR/preprocessing/splits/myops380_no_val/test_vol.txt"
```

#### Trên Linux (Bash):
```bash
bash scripts/run_evaluation.sh \
    runs/myopsnet_seed1234/test_predictions \
    runs/myopsnet_seed1234/evaluation \
    /path/to/Processed_data \
    /path/to/splits/myops380_no_val/test_vol.txt
```

Báo cáo kết quả sẽ được tạo tại:
- `runs/myopsnet_seed1234/evaluation/per_case.csv`: Điểm số chi tiết từng ca bệnh kèm các hàng tóm tắt Mean, Std, Median.
- `runs/myopsnet_seed1234/evaluation/metrics.json`: Tệp JSON tóm tắt toàn bộ chỉ số benchmark.

---

### Chạy tự động trọn gói (All-in-One Pipeline)

Bạn có thể chạy liên hoàn cả 3 bước (Train 300 epochs $\to$ Predict 76 cases $\to$ Evaluate metrics) chỉ bằng một lệnh duy nhất:

```powershell
# Windows PowerShell:
.\scripts\run_all_benchmark.ps1 -Seed 1234
```
hoặc
```bash
# Linux Bash:
bash scripts/run_all_benchmark.sh /path/to/Processed_data /path/to/splits/myops380_no_val runs/myopsnet_seed1234 1234 300 16 auto
```

---

## 3. Tổng kết kiểm thử hệ thống (Test Verification Suite)

Các bảng M1-M5 bên dưới là kết quả kiểm tra lịch sử. Các test CLI M3/M5 phụ thuộc cache thật; không chạy chúng trên test cohort để chọn cấu hình. Regression cho RNG/initialization/resume mới dùng dữ liệu synthetic trong `tests/test_training_reproducibility.py` (loader 0/2 workers, partial final batch, dropout, simulated AMP skip, cache/recipe mismatch). Các test resume đã được cập nhật để giữ nguyên tổng budget hoặc từ chối checkpoint legacy thiếu contract.

| Cột mốc | Tệp kiểm thử | Số ca test | Trạng thái |
|---|---|---|---|
| M1: Architecture & Loss | `test_milestone1.py`, `test_m1_adversarial.py`, `test_adversarial_deep_m1.py` | 95 tests | **100% PASS** |
| M2: DataLoader & Augmentation | `test_milestone2.py`, `test_m2_adversarial.py`, `test_adversarial_m2_challenger.py` | 46 tests | **100% PASS** |
| M3: Training Recipe & Resume | `test_milestone3.py`, `test_adversarial_m3_challenger.py`, `test_adversarial_m3_challenger2.py` | 51 tests | **100% PASS** |
| M4: Predict & Evaluation | `test_milestone4.py`, `test_m4_adversarial_challenger1.py`, `test_m4_adversarial_challenger2.py` | 106 tests | **100% PASS** |
| M5: End-to-End Smoke Tests | `test_milestone5.py` | 5 tests | **100% PASS** |

Bạn có thể kiểm tra lại toàn bộ bất kỳ lúc nào bằng lệnh:
```bash
pytest tests/test_milestone5.py -v
```
