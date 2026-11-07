# Project: MyoPS-Net Standardization for MyoPS380 Benchmark

## Architecture
- Baseline model: `MyoPS-Net-L` 2-stage multi-head convolutional network adapted for 3 modalities (`bSSFP, LGE, T2w`).
  - Report as **MyoPS-Net-L adapted with binary MPC**, contract `myopsnet_l_binary_mpc_v1`. The binary cardiac head replaces the original Myo/LV supervision because this cache has no separate LV annotation; this is not an unchanged reproduction of the original method.
  - Stage 1: `unet_C0` myocardium prior and consistency (MPC) module receiving 3 channels (`bSSFP, LGE, T2w`) and producing 2 output channels (`0: Background, 1: Whole Myocardium`).
  - Stage 2: Dual pathology encoders:
    - `encoder_LGE` (2 ch: `LGE` + detached whole myocardium mask `mask_C0`).
    - `encoder_T2` (2 ch: `T2w` + detached whole myocardium mask `mask_C0`).
  - Stage 2: Cross-modal decoders:
    - `decoder_LGE` (skip connections from `f_T2` and `f_LGE`) producing 3 classes: `{0: Background, 1: Non-scar Myocardium, 2: Scar}`.
    - `decoder_T2` (skip connections from `f_LGE` and `f_T2`) producing 3 classes: `{0: Background, 1: Normal Myocardium, 2: Edema-inclusive}`.
- Target adapters:
  - Input label: MyoPS380 legacy `{0: Background, 1: Normal Myocardium, 2: Scar, 3: Edema-exclusive}`.
  - `target_C0`: Whole myocardium ($1 \cup 2 \cup 3 \rightarrow 1$), background $0 \rightarrow 0$.
  - `target_LGE`: Scar ($2 \rightarrow 2$), non-scar myocardium ($1 \cup 3 \rightarrow 1$), background $0 \rightarrow 0$.
  - `target_T2`: Edema-inclusive ($2 \cup 3 \rightarrow 2$), normal myocardium ($1 \rightarrow 1$), background $0 \rightarrow 0$.
- Output fusion:
  - Canonical 4-class export:
    - Scar: $\text{argmax}(\text{seg\_LGE}) == 2$ (ID 3)
    - Edema-exclusive: $(\text{argmax}(\text{seg\_T2}) == 2) \land \neg \text{Scar}$ (ID 2)
    - Normal Myocardium: $(\text{argmax}(\text{seg\_C0}) == 1) \land \neg \text{Scar} \land \neg \text{Edema-exclusive}$ (ID 1)
    - Background: Remaining voxels (ID 0)
- Training Protocol: `myops380_train304_test76_last_v1`
  - 300 epochs, batch size 16, accum_steps 1, AdamW base_lr 0.0003333, weight_decay 0.0001, power 0.9 poly LR decay (37,200 updates), AMP auto, strict no validation.
  - 37,200 scheduled opportunities; record successful and skipped updates separately. New runs use native `weights_init`, seed 1234, workers 4 and a dedicated loader generator with SCAR worker seeds. Resume checks config/data/source/runtime identity and restores the skip counter; old checkpoints without this contract are inference-only for the new trainer.
  - Atomic checkpoint every epoch by default; `epochs_per_run` can pause a session without changing the full schedule. Model/initialization/loss/RNG contracts are saved in both config and checkpoint. The accepted loss keeps epsilon `1e-7` and absent-class WCE hardening; these differ from native numeric edge handling.
- Evaluation Protocol: `myops380_voxel_v3`
  - 76 test volumes, unit voxel surface distances, empty-mask exclusion handling, `per_case.csv` and `metrics.json`.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | 3-Modality Input Branch (`MyoPS-Net-L`) | Adapt network to ingest (bSSFP, LGE, T2w) without synthetic modalities or channel duplication | M1 | ORIGINAL_REQUEST §R2 |
| F2 | 2-Class Cardiac Prior Head (`unet_C0`) | Output 2 classes (Background, Whole Myo) without artificial LV masks | M1 | ORIGINAL_REQUEST §R2 |
| F3 | Dual Pathology Encoder-Decoder Cross-Skip | Connect LGE and T2 features via direct cross-skip connections (remove mapping branch) | M1 | ORIGINAL_REQUEST §R2 |
| F4 | Robust Multi-Head Target Construction | Map MyoPS380 legacy labels (0..3) to target_C0, target_LGE, target_T2 | M1 | ORIGINAL_REQUEST §R2 |
| F5 | Stable Fixed-Dimension `WCELoss` & Composite Loss | Refactor dynamic `WCELoss` to prevent dimension errors / NaNs and balance multi-head loss | M1 | Codebase Survey |
| F6 | Train Manifest & Sliced Data Loading | Load 1,972 slices across 304 patients from train.txt and Processed_data/{bSSFP,LGE,T2w}/train_npz | M2 | ORIGINAL_REQUEST §R1 |
| F7 | Test Volume Data Loading | Load 76 test patient volumes from test_vol.txt and Processed_data/{bSSFP,LGE,T2w}/test_vol_h5 | M2 | ORIGINAL_REQUEST §R1 |
| F8 | SCAR/I-MMSeg `RandomGenerator` Augmentation | Implement exact spatial & intensity transforms (rot/flip, rotate, noise, shift, brightness, gamma, crop, zoom) | M2 | ORIGINAL_REQUEST §R1 |
| F9 | AdamW & Polynomial LR Decay Schedule | Implement AdamW (3.333e-4, 1e-4) and power 0.9 poly LR decay over 37,200 updates matching SCAR | M3 | ORIGINAL_REQUEST §R1 |
| F10 | 300-Epoch Training Loop with Auto AMP | 300 planned epochs, physical batch size 16, accum_steps 1, auto AMP, strict NO validation | M3 | ORIGINAL_REQUEST §R1 |
| F11 | Atomic Periodic & Final Checkpointing | Save checkpoints to <run_dir>/checkpoints/myopsnet_seed<seed>.pt with full recovery payload | M3 | ORIGINAL_REQUEST §R1 |
| F12 | Full Checkpoint Resumption | Restore training state, epoch, iteration, optimizer, scheduler, AMP scaler without counter reset | M3 | ORIGINAL_REQUEST §R1 |
| F13 | Deterministic Single-Pass Volume Inference | Slice-batched inference with model.eval(), bilinear resizing, no TTA, no postprocessing/LCC | M4 | ORIGINAL_REQUEST §R1 |
| F14 | Canonical 4-Class Output Fusion | Hierarchically fuse multi-head predictions into canonical uint8 [0..3] segmentation | M4 | ORIGINAL_REQUEST §R2 |
| F15 | Prediction Export Format | Save <run_dir>/test_predictions/<CaseID>_pred.npz under key 'prediction' matching grid shape | M4 | ORIGINAL_REQUEST §R1 |
| F16 | Standalone Evaluation Tool (`evaluate_predictions.py`) | Evaluate 76 test predictions using protocol myops380_voxel_v3 matching SCAR surface_distance logic | M4 | ORIGINAL_REQUEST §R1 |
| F17 | Evaluation Metrics Export | Generate standardized per_case.csv and metrics.json with 5 region metrics and empty-mask handling | M4 | ORIGINAL_REQUEST §R1 |
| F18 | Synthetic Smoke Test Suite | Automated test suite verifying forward/backward pass, label adapter, loss, augmentation, checkpoint resume, export, and evaluation | M5 | ORIGINAL_REQUEST §R3 |
| F19 | Command-Line Execution Scripts | Deliver turnkey CLI scripts/commands for training 300 epochs, resuming, exporting predictions, and running evaluation | M5 | ORIGINAL_REQUEST §R3 |
| F20 | Reference & Dataset File Immutability | Ensure D:/NCKH/SCAR and dataset cache files remain completely unmodified | M5 | ORIGINAL_REQUEST §R3 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Architecture & Loss Adaptation | F1, F2, F3, F4, F5 (`network/model.py`, `network/unet.py`, `criterion/loss.py`, target adapter) | none | DONE |
| M2 | Manifests, DataLoader & Augmentation | F6, F7, F8 (`dataset/myops380_dataset.py`, `RandomGenerator`) | M1 | DONE |
| M3 | Training Recipe, Optimizer & Checkpointing | F9, F10, F11, F12 (`train.py`, optimizer, poly scheduler, atomic resume) | M1, M2 | DONE |
| M4 | Inference Export & Standalone Evaluation | F13, F14, F15, F16, F17 (`predict.py`, `evaluate_predictions.py`, canonical fusion) | M1, M2, M3 | DONE |
| M5 | Smoke Test Verification & Execution Scripts | F18, F19, F20 (synthetic smoke tests, verification suites, CLI runner scripts) | M1, M2, M3, M4 | DONE |

---

## Interface Contracts

### Data Loader ↔ Model & Trainer
- `MyoPS380TrainDataset`:
  - `__getitem__(index)` returns dictionary:
    - `'bSSFP'`: `torch.FloatTensor` of shape `(1, 128, 128)`
    - `'LGE'`: `torch.FloatTensor` of shape `(1, 128, 128)`
    - `'T2w'`: `torch.FloatTensor` of shape `(1, 128, 128)`
    - `'label'`: `torch.LongTensor` of shape `(128, 128)` with values in `{0, 1, 2, 3}`
    - `'case_slice'`: string ID, e.g. `'case0002_slice000'`
- `MyoPS380TestDataset`:
  - `__getitem__(index)` returns dictionary:
    - `'bSSFP'`: `torch.FloatTensor` of shape `(1, 128, 128, D)`
    - `'LGE'`: `torch.FloatTensor` of shape `(1, 128, 128, D)`
    - `'T2w'`: `torch.FloatTensor` of shape `(1, 128, 128, D)`
    - `'label'`: `torch.LongTensor` of shape `(128, 128, D)` with values in `{0, 1, 2, 3}`
    - `'case_id'`: string ID, e.g. `'case0001'`

### Model Forward & Output Shapes
- `MyoPSNetL(c0, lge, t2)`:
  - Inputs: `c0, lge, t2` each `(B, 1, 128, 128)`
  - Returns tuple: `(seg_C0, seg_LGE, seg_T2)`
    - `seg_C0`: `(B, 2, 128, 128)` Softmax probabilities over `{0: Background, 1: Whole Myocardium}`
    - `seg_LGE`: `(B, 3, 128, 128)` Softmax probabilities over `{0: Background, 1: Non-scar Myocardium, 2: Scar}`
    - `seg_T2`: `(B, 3, 128, 128)` Softmax probabilities over `{0: Background, 1: Normal Myocardium, 2: Edema-inclusive}`

### Target Adapter
- `create_multi_head_targets(label)`:
  - Input: `label` tensor of shape `(B, 128, 128)` or `(128, 128)` with values in `{0, 1, 2, 3}`
  - Returns: `(target_C0, target_LGE, target_T2)`
    - `target_C0 = (label >= 1).long()` (0: bg, 1: whole myocardium)
    - `target_LGE`: 0 for bg; 2 for scar (legacy 2); 1 for non-scar myo (legacy 1, 3)
    - `target_T2`: 0 for bg; 2 for edema-inclusive (legacy 2, 3); 1 for normal myo (legacy 1)

### Canonical Prediction Fusion
- `fuse_predictions(seg_C0, seg_LGE, seg_T2)`:
  - Inputs: probabilities or argmax indices from the 3 heads
  - Output: `(H, W)` or `(H, W, D)` uint8 array with canonical values:
    - 0: Background
    - 1: Normal Myocardium
    - 2: Edema-exclusive
    - 3: Scar

### Checkpoint Contract
- Saved path: `<run_dir>/checkpoints/myopsnet_seed<seed>.pt`
- Keys:
  - `epoch`: int
  - `global_step`: int
  - `model_state_dict`: dict
  - `optimizer_state_dict`: dict
  - `scheduler_state_dict`: dict
  - `scaler_state_dict`: dict
  - `planned_epochs`: 300
  - `completed_epochs`: int
  - `training_complete`: bool (True if completed_epochs >= 300)
  - `seed`: int
  - `protocol_id`: `"myops380_train304_test76_last_v1"`
  - `opt_protocol`: `"im_mseg_adamw_poly_v1"`

### Evaluation Output Contract
- Directory: `<run_dir>/evaluation/`
- Files:
  - `per_case.csv`: per-case metrics across 5 regions + Summary rows (Mean, Std, Median)
  - `metrics.json`: structured dictionary with benchmark summary metrics

---

## Code Layout
```
d:\NCKH\MyoPS-Net/
├── network/
│   ├── model.py            # MyoPSNetL architecture (3 modalities)
│   ├── unet.py             # UNet modules (2-class C0, pathology encoders/decoders)
├── criterion/
│   ├── loss.py             # Robust fixed-dim WCELoss, multi-head loss formulation
├── dataset/
│   ├── myops380_dataset.py # MyoPS380 train slice & test volume loaders
│   ├── augmentation.py    # SCAR/I-MMSeg RandomGenerator
├── utils/
│   ├── tools.py            # Target adapters, canonical fusion, I/O tools
├── training/
│   ├── optimizer.py        # AdamW and polynomial LR scheduler (im_mseg_adamw_poly_v1)
│   ├── checkpoint.py       # Atomic checkpoint saving and full resumption
├── train.py                # Standardized 300-epoch training entry point (no validation)
├── predict.py              # Single-pass deterministic test volume prediction export
├── evaluate_predictions.py # Standalone evaluation tool (myops380_voxel_v3)
├── tests/
│   ├── smoke_test.py       # Automated smoke test suite
└── scripts/
    ├── run_train.ps1 / .bat / CLI guides
    ├── run_predict.ps1 / .bat / CLI guides
    └── run_evaluate.ps1 / .bat / CLI guides
```
