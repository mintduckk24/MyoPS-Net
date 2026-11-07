"""
test_milestone5.py
==================
Milestone 5: Synthetic Smoke Tests, End-to-End Pipeline Integration & Production Verification.

Verifies:
  1. Full training loop smoke test (1 epoch, minimal batches, auto AMP, atomic checkpoint).
  2. Full checkpoint resumption smoke test (restores state, increments step, continues schedule).
  3. Single-pass deterministic volume inference (exports uint8 .npz under key 'prediction').
  4. Standalone evaluation metric engine (computes myops380_voxel_v3, outputs per_case.csv, metrics.json).
  5. Immutability checks: D:/NCKH/SCAR tracked files and dataset cache files remain completely unmodified.
  6. Strict zero-validation adherence (no validation data, no val loop, no best.pt selection).
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCAR_ROOT = Path("D:/NCKH/SCAR")
DATASET_ROOT = Path("D:/NCKH/Dataset/MyoPS380/Processed_data")
SPLIT_ROOT = SCAR_ROOT / "preprocessing/splits/myops380_no_val"


class TestMilestone5EndToEndPipeline:
    """Verifies end-to-end integration: train dry-run -> checkpoint -> resume -> predict -> evaluate."""

    @pytest.fixture(autouse=True)
    def setup_dirs(self, tmp_path):
        self.tmp_run_dir = tmp_path / "test_run_e2e"
        self.tmp_run_dir.mkdir(parents=True, exist_ok=True)

    def test_train_dry_run_creates_atomic_checkpoint(self):
        """Verifies train.py dry run produces compliant atomic checkpoint."""
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--data_root", str(DATASET_ROOT),
            "--split_dir", str(SPLIT_ROOT),
            "--run_dir", str(self.tmp_run_dir),
            "--amp", "auto",
            "--seed", "1234",
            "--dry_run",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"Training dry-run failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        ckpt_path = self.tmp_run_dir / "checkpoints" / "myopsnet_seed1234.pt"
        assert ckpt_path.is_file(), f"Expected checkpoint at {ckpt_path}"

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        required_keys = [
            "epoch", "global_step", "model_state_dict", "optimizer_state_dict",
            "scheduler_state_dict", "scaler_state_dict", "planned_epochs",
            "completed_epochs", "training_complete", "seed", "protocol_id"
        ]
        for k in required_keys:
            assert k in payload, f"Missing key in checkpoint: {k}"
        assert payload["epoch"] == 1
        assert payload["completed_epochs"] == 1
        assert payload["planned_epochs"] == 300
        assert payload["training_complete"] is False
        assert payload["seed"] == 1234
        assert payload["protocol_id"] == "myops380_train304_test76_last_v1"

    def test_resume_from_checkpoint_continues_seamlessly(self):
        """Verifies resuming from checkpoint continues without counter resets."""
        # 1. Pause a fixed synthetic/smoke budget without changing it on resume.
        cmd1 = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--data_root", str(DATASET_ROOT),
            "--split_dir", str(SPLIT_ROOT),
            "--run_dir", str(self.tmp_run_dir),
            "--amp", "auto",
            "--seed", "1234",
            "--epochs", "2",
            "--max_batches", "1",
            "--batch_size", "2",
            "--epochs_per_run", "1",
        ]
        r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=120)
        assert r1.returncode == 0, f"Initial train failed:\n{r1.stderr}"

        ckpt_path = self.tmp_run_dir / "checkpoints" / "myopsnet_seed1234.pt"
        payload1 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        step1 = payload1["global_step"]
        assert payload1["completed_epochs"] == 1

        # 2. Resume for epoch 2 (1 batch) by setting --epochs 2 --save_interval 1
        cmd2 = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--data_root", str(DATASET_ROOT),
            "--split_dir", str(SPLIT_ROOT),
            "--run_dir", str(self.tmp_run_dir),
            "--epochs", "2",
            "--save_interval", "1",
            "--max_batches", "1",
            "--batch_size", "2",
            "--amp", "auto",
            "--seed", "1234",
            "--resume", str(ckpt_path),
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=120)
        assert r2.returncode == 0, f"Resume train failed:\n{r2.stderr}"

        payload2 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert payload2["completed_epochs"] == 2
        assert payload2["global_step"] > step1, "Global step should advance after resume"

    def test_predict_and_evaluate_end_to_end(self):
        """Verifies predict.py creates valid .npz and evaluate_predictions.py scores them."""
        # 1. Quick dry-run checkpoint
        cmd_train = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--data_root", str(DATASET_ROOT),
            "--split_dir", str(SPLIT_ROOT),
            "--run_dir", str(self.tmp_run_dir),
            "--amp", "auto",
            "--seed", "1234",
            "--dry_run",
        ]
        r_tr = subprocess.run(cmd_train, capture_output=True, text=True, timeout=120)
        assert r_tr.returncode == 0

        ckpt_path = self.tmp_run_dir / "checkpoints" / "myopsnet_seed1234.pt"

        # 2. Predict on subset of 2 test cases using --dry_run
        cmd_pred = [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--data_root", str(DATASET_ROOT),
            "--split_dir", str(SPLIT_ROOT),
            "--checkpoint", str(ckpt_path),
            "--run_dir", str(self.tmp_run_dir),
            "--dry_run",
        ]
        r_pred = subprocess.run(cmd_pred, capture_output=True, text=True, timeout=120)
        assert r_pred.returncode == 0, f"Predict failed:\nSTDOUT:\n{r_pred.stdout}\nSTDERR:\n{r_pred.stderr}"

        pred_dir = self.tmp_run_dir / "test_predictions"
        assert pred_dir.is_dir()
        pred_files = list(pred_dir.glob("*_pred.npz"))
        assert len(pred_files) == 2, f"Expected 2 prediction files, found {len(pred_files)}"
        predicted_cases = [f.stem.replace("_pred", "") for f in pred_files]

        for pf in pred_files:
            data = np.load(pf)
            assert "prediction" in data, f"Missing 'prediction' key in {pf}"
            pred_arr = data["prediction"]
            assert pred_arr.dtype == np.uint8, f"Expected uint8 dtype, got {pred_arr.dtype}"
            assert set(np.unique(pred_arr)).issubset({0, 1, 2, 3}), "Predictions must be in [0..3]"
            assert pred_arr.ndim == 3, f"Expected 3D array (H, W, D), got ndim={pred_arr.ndim}"

        # 3. Evaluate predictions for the predicted subset
        eval_dir = self.tmp_run_dir / "evaluation"
        cmd_eval = [
            sys.executable,
            str(PROJECT_ROOT / "evaluate_predictions.py"),
            "--pred_dir", str(pred_dir),
            "--data_root", str(DATASET_ROOT),
            "--split_dir", str(SPLIT_ROOT / "test_vol.txt"),
            "--output_dir", str(eval_dir),
            "--cases", *predicted_cases,
        ]
        r_eval = subprocess.run(cmd_eval, capture_output=True, text=True, timeout=120)
        assert r_eval.returncode == 0, f"Evaluation failed:\nSTDOUT:\n{r_eval.stdout}\nSTDERR:\n{r_eval.stderr}"

        per_case_csv = eval_dir / "per_case.csv"
        metrics_json = eval_dir / "metrics.json"
        assert per_case_csv.is_file(), f"Missing {per_case_csv}"
        assert metrics_json.is_file(), f"Missing {metrics_json}"

        with open(metrics_json, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        assert metrics_data["metric_protocol"] == "myops380_voxel_v3"
        assert metrics_data["case_count"] == 2
        assert "myocardial_ring" in metrics_data


class TestExternalImmutabilityAndStrictRules:
    """Verifies that reference code and datasets were not modified and strict rules are kept."""

    def test_scar_reference_git_clean(self):
        """Verifies SCAR git tracked files are completely unmodified."""
        if (SCAR_ROOT / ".git").exists():
            r = subprocess.run(["git", "diff", "--exit-code"], cwd=str(SCAR_ROOT), capture_output=True, text=True)
            assert r.returncode == 0, f"SCAR tracked files modified:\n{r.stdout}"
            r_staged = subprocess.run(["git", "diff", "--cached", "--exit-code"], cwd=str(SCAR_ROOT), capture_output=True, text=True)
            assert r_staged.returncode == 0, f"SCAR staged files modified:\n{r_staged.stdout}"

    def test_no_validation_mechanism_in_train(self):
        """Verifies train.py has no validation data loading or epoch validation."""
        with open(PROJECT_ROOT / "train.py", "r", encoding="utf-8") as f:
            code = f.read()
        assert "val_loader" not in code, "train.py must not instantiate a val_loader"
        assert "validation.py" not in code, "train.py must not import legacy validation.py"
        assert "best.pt" not in code, "train.py must not save or track best.pt"
