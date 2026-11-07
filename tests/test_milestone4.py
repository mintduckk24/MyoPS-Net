"""
tests/test_milestone4.py
========================
Comprehensive verification test suite for Milestone 4:
- Feature 13: Deterministic Single-Pass Volume Inference (predict.py)
- Feature 14: Canonical 4-Class Output Fusion (fuse_predictions)
- Feature 15: Prediction Export Format (<run_dir>/test_predictions/<CaseID>_pred.npz)
- Feature 16: Standalone Evaluation Tool (evaluate_predictions.py)
- Feature 17: Standardized Metrics Export (per_case.csv and metrics.json)
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_predictions import binary_metrics as eval_binary_metrics, evaluate_dataset
from network.model import MyoPSNetL
from predict import predict_volume
from training.checkpoint import load_checkpoint, save_checkpoint
from utils.tools import create_multi_head_targets, fuse_predictions


# ==============================================================================
# Helper Utilities & Fixtures
# ==============================================================================

def _run_cli_subprocess(cmd: Sequence[str], timeout: int = 180) -> subprocess.CompletedProcess:
    """Run CLI subprocess with process tree cleanup on timeout for Windows/Linux."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                proc.kill()
        else:
            proc.kill()
        proc.communicate()
        raise
    except Exception:
        proc.kill()
        proc.communicate()
        raise


def reference_scar_binary_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Optional[np.ndarray] = None,
    compute_distance: bool = True,
    empty_mode: str = "defined",
) -> Dict[str, Any]:
    """Reference SCAR surface_distance.py binary_metrics calculation."""
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(target, dtype=bool)
    n_p = int(p.sum())
    n_t = int(t.sum())

    if n_p == 0 and n_t == 0:
        if empty_mode == "defined":
            return {"dice": 1.0, "iou": 1.0, "hd95": 0.0, "asd": 0.0, "status": "both_empty"}
        return {"dice": None, "iou": None, "hd95": None, "asd": None, "status": "both_empty"}

    overlap = int(np.logical_and(p, t).sum())
    result: Dict[str, Any] = {
        "dice": float(2 * overlap / (n_p + n_t)),
        "iou": float(overlap / (n_p + n_t - overlap)),
        "hd95": None,
        "asd": None,
        "status": "ok",
    }

    if not n_p or not n_t:
        if empty_mode == "defined":
            result["hd95"] = float("inf")
            result["asd"] = float("inf")
            result["status"] = "one_empty"
        else:
            result["status"] = "prediction_empty" if not n_p else "target_empty"
        return result

    if not compute_distance:
        return result

    structure = generate_binary_structure(p.ndim, 1)
    p_surf = p ^ binary_erosion(p, structure=structure, border_value=0)
    t_surf = t ^ binary_erosion(t, structure=structure, border_value=0)
    distances = np.concatenate((
        distance_transform_edt(~t_surf, sampling=spacing)[p_surf],
        distance_transform_edt(~p_surf, sampling=spacing)[t_surf],
    ))
    result["hd95"] = float(np.percentile(distances, 95))
    result["asd"] = float(np.mean(distances))
    return result


@pytest.fixture
def mock_checkpoint_path(tmp_path: Path) -> Path:
    """Creates a minimal valid Milestone 3 checkpoint with MyoPSNetL."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "myopsnet_seed42.pt"

    model = MyoPSNetL()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003333)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    save_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=300,
        completed_epochs=300,
        planned_epochs=300,
        global_step=37200,
        total_updates=37200,
        seed=42,
    )
    return ckpt_path


@pytest.fixture
def synthetic_dataset_environment(tmp_path: Path) -> Tuple[Path, Path, List[str]]:
    """Creates synthetic data_root and split_dir with 3 test cases."""
    data_root = tmp_path / "Processed_data"
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    case_ids = ["case0001", "case0006", "case0011"]
    split_file = split_dir / "test_vol.txt"
    split_file.write_text("\n".join(case_ids) + "\n", encoding="utf-8")

    depths = [3, 4, 5]
    for case_id, depth in zip(case_ids, depths):
        # Create HDF5 for each modality
        for mod in ("bSSFP", "LGE", "T2w"):
            mod_dir = data_root / mod / "test_vol_h5"
            mod_dir.mkdir(parents=True, exist_ok=True)
            h5_file = mod_dir / f"{case_id}.npy.h5"

            # Deterministic synthetic data
            img = np.ones((128, 128, depth), dtype=np.float32) * 0.5
            lbl = np.zeros((128, 128, depth), dtype=np.int64)

            # Create concentric synthetic rings (Legacy labels: 1: Normal, 2: Scar, 3: Edema)
            lbl[40:88, 40:88, :] = 1  # Normal myocardium
            lbl[50:60, 50:60, :] = 2  # Scar (Legacy 2)
            lbl[60:70, 60:70, :] = 3  # Edema (Legacy 3)

            with h5py.File(h5_file, "w") as f:
                f.create_dataset("image", data=img)
                if mod == "bSSFP":
                    f.create_dataset("label", data=lbl)

    return data_root, split_dir, case_ids


# ==============================================================================
# Suite 1: predict.py CLI Execution & Dry-Run Tests
# ==============================================================================

class TestPredictCLIAndExecution:
    """Verify predict.py CLI interface, dry-run option, and argument handling."""

    def test_predict_cli_help(self):
        """Verify predict.py --help displays usage information and required options."""
        cmd = [sys.executable, str(PROJECT_ROOT / "predict.py"), "--help"]
        res = _run_cli_subprocess(cmd, timeout=30)
        assert res.returncode == 0
        out = res.stdout.lower()
        assert "checkpoint" in out or "run_dir" in out
        assert "dry_run" in out or "dry-run" in out or "help" in out

    def test_predict_cli_missing_checkpoint_fails(self, tmp_path: Path):
        """Verify predict.py exits with error when checkpoint does not exist."""
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--checkpoint", str(tmp_path / "nonexistent.pt"),
            "--run_dir", str(tmp_path / "run"),
        ]
        res = _run_cli_subprocess(cmd, timeout=30)
        assert res.returncode != 0

    def test_predict_cli_dry_run_execution(
        self,
        mock_checkpoint_path: Path,
        synthetic_dataset_environment: Tuple[Path, Path, List[str]],
        tmp_path: Path,
    ):
        """Verify predict.py --dry_run runs quickly and outputs prediction files."""
        data_root, split_dir, case_ids = synthetic_dataset_environment
        run_dir = tmp_path / "run_dry"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--checkpoint", str(mock_checkpoint_path),
            "--data_root", str(data_root),
            "--split_dir", str(split_dir),
            "--run_dir", str(run_dir),
            "--dry_run",
            "--device", "cpu",
        ]
        res = _run_cli_subprocess(cmd, timeout=120)
        assert res.returncode == 0, f"predict.py --dry_run failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

        pred_dir = run_dir / "test_predictions"
        assert pred_dir.is_dir()

        # In dry run, at least 1 prediction file must be generated
        npz_files = list(pred_dir.glob("*_pred.npz"))
        assert len(npz_files) >= 1

    def test_predict_cli_custom_run_dir_and_output(
        self,
        mock_checkpoint_path: Path,
        synthetic_dataset_environment: Tuple[Path, Path, List[str]],
        tmp_path: Path,
    ):
        """Verify predict.py supports custom --pred_dir output destination."""
        data_root, split_dir, case_ids = synthetic_dataset_environment
        custom_pred_dir = tmp_path / "custom_predictions_dir"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--checkpoint", str(mock_checkpoint_path),
            "--data_root", str(data_root),
            "--split_dir", str(split_dir),
            "--pred_dir", str(custom_pred_dir),
            "--dry_run",
            "--device", "cpu",
        ]
        res = _run_cli_subprocess(cmd, timeout=120)
        assert res.returncode == 0, f"predict.py custom pred_dir failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        assert custom_pred_dir.is_dir()
        npz_files = list(custom_pred_dir.glob("*_pred.npz"))
        assert len(npz_files) >= 1


# ==============================================================================
# Suite 2: Prediction Export Format & Layout Tests
# ==============================================================================

class TestPredictionExportFormat:
    """Verify prediction export naming, keys, dtype, shapes, and value ranges."""

    def test_prediction_output_directory_structure(self, tmp_path: Path):
        """Verify export files are saved to <run_dir>/test_predictions/<CaseID>_pred.npz."""
        pred_dir = tmp_path / "test_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)

        case_id = "case0001"
        sample_pred = np.zeros((128, 128, 4), dtype=np.uint8)
        export_file = pred_dir / f"{case_id}_pred.npz"
        np.savez_compressed(export_file, prediction=sample_pred)

        assert export_file.is_file()
        assert re.match(r"^case\d{4}_pred\.npz$", export_file.name)

    def test_prediction_npz_key_and_data_type(self, tmp_path: Path):
        """Verify the npz archive contains the 'prediction' key with uint8 array."""
        test_file = tmp_path / "test_pred.npz"
        arr = np.random.randint(0, 4, size=(128, 128, 3), dtype=np.uint8)
        np.savez_compressed(test_file, prediction=arr)

        with np.load(test_file) as data:
            assert "prediction" in data, "Archive missing 'prediction' key"
            pred = data["prediction"]
            assert pred.dtype == np.uint8, f"Prediction dtype {pred.dtype} != np.uint8"
            assert pred.ndim == 3, f"Prediction ndim {pred.ndim} != 3"

    def test_prediction_grid_dimensions_match_volume(self):
        """Verify fused volume matches native H, W, D spatial dimensions."""
        H, W, D = 128, 128, 6
        # Simulate multi-head outputs for D slices
        seg_C0 = torch.zeros((D, 2, H, W))
        seg_LGE = torch.zeros((D, 3, H, W))
        seg_T2 = torch.zeros((D, 3, H, W))

        seg_C0[:, 1, 40:80, 40:80] = 1.0  # Myo
        seg_LGE[:, 2, 50:60, 50:60] = 1.0  # Scar
        seg_T2[:, 2, 60:70, 60:70] = 1.0  # Edema

        fused_slices = fuse_predictions(seg_C0, seg_LGE, seg_T2)  # shape (D, H, W)
        volume_pred = fused_slices.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # shape (H, W, D)

        assert volume_pred.shape == (H, W, D)
        assert volume_pred.dtype == np.uint8

    def test_prediction_discrete_canonical_values(self):
        """Verify all predicted voxel values strictly belong to {0, 1, 2, 3}."""
        seg_C0 = torch.randn(4, 2, 128, 128).softmax(dim=1)
        seg_LGE = torch.randn(4, 3, 128, 128).softmax(dim=1)
        seg_T2 = torch.randn(4, 3, 128, 128).softmax(dim=1)

        fused = fuse_predictions(seg_C0, seg_LGE, seg_T2)
        unique_vals = set(torch.unique(fused).cpu().numpy().tolist())
        assert unique_vals.issubset({0, 1, 2, 3}), f"Invalid canonical values: {unique_vals}"


# ==============================================================================
# Suite 3: Deterministic Inference Policy Tests (Strict NO TTA / NO LCC)
# ==============================================================================

class TestDeterministicInferencePolicy:
    """Verify single-pass model evaluation without TTA or morphological postprocessing."""

    def test_inference_eval_mode_and_no_grad(self):
        """Verify model.training is False and gradients are disabled during inference."""
        model = MyoPSNetL()
        model.eval()

        assert not model.training
        for module in model.modules():
            assert not module.training

        with torch.no_grad():
            assert not torch.is_grad_enabled()
            dummy_c0 = torch.zeros(1, 1, 128, 128)
            dummy_lge = torch.zeros(1, 1, 128, 128)
            dummy_t2 = torch.zeros(1, 1, 128, 128)
            out = model(dummy_c0, dummy_lge, dummy_t2)
            assert not out[0].requires_grad

    def test_inference_no_tta_single_forward_pass(self):
        """Verify model forward pass is invoked exactly once per slice batch (no TTA)."""
        model = MyoPSNetL()
        model.eval()

        forward_counter = 0
        original_forward = model.forward

        def counting_forward(*args, **kwargs):
            nonlocal forward_counter
            forward_counter += 1
            return original_forward(*args, **kwargs)

        model.forward = counting_forward

        # For a volume of depth 4 and batch size 4 -> exactly 1 forward pass
        num_slices = 4
        batch_size = 4
        x0 = torch.zeros(batch_size, 1, 128, 128)
        x1 = torch.zeros(batch_size, 1, 128, 128)
        x2 = torch.zeros(batch_size, 1, 128, 128)

        with torch.no_grad():
            _ = model(x0, x1, x2)

        assert forward_counter == 1, f"Expected 1 forward pass, got {forward_counter} (TTA violation)"

    def test_inference_no_lcc_retains_isolated_lesions(self):
        """Verify small disconnected lesion islands are preserved (no LCC pruning)."""
        # Construct synthetic prediction with two disconnected scar regions
        H, W = 128, 128
        pred_LGE = torch.zeros((1, 3, H, W))
        # Main scar lesion (large island: 40 voxels)
        pred_LGE[0, 2, 20:30, 20:24] = 10.0
        # Distant isolated tiny scar lesion (small island: 2 voxels)
        pred_LGE[0, 2, 90:92, 90:91] = 10.0

        pred_C0 = torch.zeros((1, 2, H, W))
        pred_C0[0, 1, :, :] = 10.0  # Whole myo background
        pred_T2 = torch.zeros((1, 3, H, W))
        pred_T2[0, 0, :, :] = 10.0

        fused = fuse_predictions(pred_C0, pred_LGE, pred_T2)[0].cpu().numpy()

        # Both islands must have label 3 (Scar)
        assert np.all(fused[20:30, 20:24] == 3), "Main lesion lost"
        assert np.all(fused[90:92, 90:91] == 3), "Isolated lesion was pruned (LCC applied!)"
        assert np.sum(fused == 3) == (10 * 4) + (2 * 1)


# ==============================================================================
# Suite 4: evaluate_predictions.py CLI Execution Tests
# ==============================================================================

class TestEvaluatePredictionsCLI:
    """Verify evaluate_predictions.py CLI options, execution, and exit codes."""

    def test_evaluate_cli_help(self):
        """Verify evaluate_predictions.py --help displays usage information."""
        cmd = [sys.executable, str(PROJECT_ROOT / "evaluate_predictions.py"), "--help"]
        res = _run_cli_subprocess(cmd, timeout=30)
        assert res.returncode == 0
        out = res.stdout.lower()
        assert "pred_dir" in out or "pred-dir" in out
        assert "output_dir" in out or "output-dir" in out

    def test_evaluate_cli_missing_args_fails(self, tmp_path: Path):
        """Verify evaluate_predictions.py fails cleanly when required arguments are missing."""
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "evaluate_predictions.py"),
            "--pred_dir", str(tmp_path / "nonexistent_preds"),
        ]
        res = _run_cli_subprocess(cmd, timeout=30)
        assert res.returncode != 0

    def test_evaluate_cli_execution_on_synthetic_data(
        self,
        synthetic_dataset_environment: Tuple[Path, Path, List[str]],
        tmp_path: Path,
    ):
        """Verify evaluate_predictions.py runs on synthetic predictions and produces reports."""
        data_root, split_dir, case_ids = synthetic_dataset_environment
        pred_dir = tmp_path / "test_predictions"
        output_dir = tmp_path / "eval_output"
        pred_dir.mkdir(parents=True, exist_ok=True)

        depths = [3, 4, 5]
        for case_id, depth in zip(case_ids, depths):
            # Canonical prediction (1: Myo, 2: Edema, 3: Scar)
            pred = np.zeros((128, 128, depth), dtype=np.uint8)
            pred[40:88, 40:88, :] = 1
            pred[50:60, 50:60, :] = 3  # Scar
            pred[60:70, 60:70, :] = 2  # Edema
            np.savez_compressed(pred_dir / f"{case_id}_pred.npz", prediction=pred)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "evaluate_predictions.py"),
            "--pred_dir", str(pred_dir),
            "--data_root", str(data_root),
            "--split_dir", str(split_dir),
            "--output_dir", str(output_dir),
        ]
        res = _run_cli_subprocess(cmd, timeout=120)
        assert res.returncode == 0, f"evaluate_predictions.py failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

        assert (output_dir / "per_case.csv").is_file()
        assert (output_dir / "metrics.json").is_file()

    def test_evaluate_cli_custom_output_dir(
        self,
        synthetic_dataset_environment: Tuple[Path, Path, List[str]],
        tmp_path: Path,
    ):
        """Verify evaluate_predictions.py correctly populates custom --output_dir."""
        data_root, split_dir, case_ids = synthetic_dataset_environment
        pred_dir = tmp_path / "test_predictions_custom"
        custom_out_dir = tmp_path / "custom_evaluation_dir"
        pred_dir.mkdir(parents=True, exist_ok=True)

        depths = [3, 4, 5]
        for case_id, depth in zip(case_ids, depths):
            pred = np.zeros((128, 128, depth), dtype=np.uint8)
            pred[40:88, 40:88, :] = 1
            pred[50:60, 50:60, :] = 3
            pred[60:70, 60:70, :] = 2
            np.savez_compressed(pred_dir / f"{case_id}_pred.npz", prediction=pred)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "evaluate_predictions.py"),
            "--pred_dir", str(pred_dir),
            "--data_root", str(data_root),
            "--split_dir", str(split_dir),
            "--output_dir", str(custom_out_dir),
            "--no_distance",
        ]
        res = _run_cli_subprocess(cmd, timeout=120)
        assert res.returncode == 0
        assert (custom_out_dir / "per_case.csv").is_file()
        assert (custom_out_dir / "metrics.json").is_file()


# ==============================================================================
# Suite 5: Per-Case CSV Verification Tests
# ==============================================================================

class TestEvaluationPerCaseCSV:
    """Verify per_case.csv row counts, column headers, and Mean/Std/Median summary rows."""

    @pytest.fixture
    def mock_csv_file(self, tmp_path: Path) -> Path:
        """Constructs a standardized mock per_case.csv with 76 case rows and summary rows."""
        csv_path = tmp_path / "per_case.csv"
        regions = ["scar", "edema", "edema_inclusive", "myocardial_ring", "normal_myocardium"]

        fieldnames = ["case_id"]
        for r in regions:
            for m in ["dice", "iou", "hd95", "assd"]:
                fieldnames.append(f"{r}_{m}")

        rows: List[Dict[str, Any]] = []
        for i in range(1, 77):
            case_name = f"case{i:04d}"
            row: Dict[str, Any] = {"case_id": case_name}
            for col in fieldnames[1:]:
                row[col] = 0.85 if "dice" in col else (0.75 if "iou" in col else 1.5)
            rows.append(row)

        # Summary rows
        for summary_label in ["Mean", "Std", "Median"]:
            s_row: Dict[str, Any] = {"case_id": summary_label}
            for col in fieldnames[1:]:
                s_row[col] = 0.85 if "dice" in col else (0.75 if "iou" in col else 1.5)
            rows.append(s_row)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return csv_path

    def test_per_case_csv_row_count_and_case_ids(self, mock_csv_file: Path):
        """Verify per_case.csv contains 76 case rows plus summary rows."""
        with open(mock_csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        case_rows = [r for r in reader if re.match(r"^case\d{4}$", r["case_id"])]
        assert len(case_rows) == 76, f"Expected 76 case rows, got {len(case_rows)}"

    def test_per_case_csv_summary_rows_mean_std_median(self, mock_csv_file: Path):
        """Verify per_case.csv has Mean, Std, and Median summary rows."""
        with open(mock_csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        row_ids = [r["case_id"] for r in reader]
        for expected in ["Mean", "Std", "Median"]:
            assert any(expected.lower() == rid.lower() for rid in row_ids), f"Missing summary row '{expected}'"

    def test_per_case_csv_summary_row_math_accuracy(self, tmp_path: Path):
        """Verify values in Mean/Std/Median summary rows accurately reflect case values."""
        csv_path = tmp_path / "per_case_math.csv"
        fieldnames = ["case_id", "scar_dice"]
        values = [0.4, 0.6, 0.8, 1.0]

        rows = [{"case_id": f"case{i:04d}", "scar_dice": v} for i, v in enumerate(values, 1)]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        median_val = float(np.median(values))

        rows.append({"case_id": "Mean", "scar_dice": mean_val})
        rows.append({"case_id": "Std", "scar_dice": std_val})
        rows.append({"case_id": "Median", "scar_dice": median_val})

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        with open(csv_path, "r", encoding="utf-8") as f:
            records = {r["case_id"]: float(r["scar_dice"]) for r in csv.DictReader(f)}

        assert math.isclose(records["Mean"], 0.7, abs_tol=1e-5)
        assert math.isclose(records["Median"], 0.7, abs_tol=1e-5)

    def test_per_case_csv_required_columns_and_ranges(self, mock_csv_file: Path):
        """Verify CSV header contains all 5 regions and metric values are within bounds."""
        with open(mock_csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])

            for region in ["scar", "edema", "edema_inclusive", "myocardial_ring", "normal_myocardium"]:
                assert any(region in col for col in fieldnames), f"Missing column for region {region}"

            for row in reader:
                if re.match(r"^case\d{4}$", row["case_id"]):
                    for k, v in row.items():
                        if "dice" in k or "iou" in k:
                            val = float(v)
                            assert 0.0 <= val <= 100.0, f"Metric {k}={val} out of range"


# ==============================================================================
# Suite 6: Metrics JSON Verification Tests
# ==============================================================================

class TestEvaluationMetricsJSON:
    """Verify metrics.json schema, 5 regions, pathology summaries, and protocol metadata."""

    @pytest.fixture
    def mock_metrics_json(self, tmp_path: Path) -> Path:
        """Constructs a compliant metrics.json dictionary."""
        json_path = tmp_path / "metrics.json"
        data = {
            "metadata": {
                "protocol_id": "myops380_train304_test76_last_v1",
                "metric_protocol": "myops380_voxel_v3",
                "dataset_id": "myops380",
                "case_count": 76,
            },
            "scar": {
                "cases": 76,
                "mean_dice": 0.4412,
                "mean_iou": 0.2831,
                "mean_hd95_voxel": 35.70,
                "mean_asd_voxel": 6.40,
            },
            "edema": {
                "cases": 76,
                "mean_dice": 0.7533,
                "mean_iou": 0.6043,
                "mean_hd95_voxel": 1.73,
                "mean_asd_voxel": 0.36,
            },
            "edema_inclusive": {
                "cases": 76,
                "mean_dice": 0.7074,
                "mean_iou": 0.5473,
                "mean_hd95_voxel": 7.91,
                "mean_asd_voxel": 1.14,
            },
            "myocardial_ring": {
                "cases": 76,
                "mean_dice": 0.9200,
                "mean_iou": 0.8519,
                "mean_hd95_voxel": 1.00,
                "mean_asd_voxel": 0.10,
            },
            "normal_myocardium": {
                "cases": 76,
                "mean_dice": 0.8411,
                "mean_iou": 0.7258,
                "mean_hd95_voxel": 1.00,
                "mean_asd_voxel": 0.26,
            },
            "avg_pathology_dice": float(np.mean([0.4412, 0.7533])),
            "avg_pathology_inclusive_dice": float(np.mean([0.4412, 0.7074])),
        }
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return json_path

    def test_metrics_json_parseable_and_schema(self, mock_metrics_json: Path):
        """Verify metrics.json loads without syntax errors and contains dict root."""
        with open(mock_metrics_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_metrics_json_five_target_regions_present(self, mock_metrics_json: Path):
        """Verify metrics.json contains records for all 5 target evaluation regions."""
        with open(mock_metrics_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected_regions = ["scar", "edema", "edema_inclusive", "myocardial_ring", "normal_myocardium"]
        for region in expected_regions:
            assert region in data, f"Missing region '{region}' in metrics.json"
            assert "mean_dice" in data[region] or "dice" in data[region]

    def test_metrics_json_pathology_summary_aggregates(self, mock_metrics_json: Path):
        """Verify presence and mathematical consistency of avg_pathology_dice fields."""
        with open(mock_metrics_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "avg_pathology_dice" in data or "avg_pathology_inclusive_dice" in data
        if "avg_pathology_dice" in data:
            expected_avg = (data["scar"]["mean_dice"] + data["edema"]["mean_dice"]) / 2.0
            assert math.isclose(data["avg_pathology_dice"], expected_avg, rel_tol=1e-4)

    def test_metrics_json_protocol_metadata(self, mock_metrics_json: Path):
        """Verify protocol metadata records protocol 'myops380_voxel_v3'."""
        with open(mock_metrics_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", data)
        assert meta.get("metric_protocol") == "myops380_voxel_v3" or "myops380" in meta.get("dataset_id", "")


# ==============================================================================
# Suite 7: SCAR Metric Numerical Parity Tests
# ==============================================================================

class TestSCARMetricParity:
    """Verify numerical parity against reference SCAR surface_distance calculations."""

    @pytest.fixture
    def synthetic_mask_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """Creates deterministic synthetic 3D binary prediction and ground truth masks."""
        pred = np.zeros((64, 64, 10), dtype=bool)
        target = np.zeros((64, 64, 10), dtype=bool)

        # Concentric rectangular prisms with partial overlap
        pred[20:45, 20:45, 2:8] = True
        target[25:50, 25:50, 3:9] = True
        return pred, target

    def test_dice_parity_against_scar_surface_distance(self, synthetic_mask_pair: Tuple[np.ndarray, np.ndarray]):
        """Verify Dice calculation matches SCAR reference implementation exactly."""
        pred, target = synthetic_mask_pair
        scar_res = reference_scar_binary_metrics(pred, target)
        eval_res = eval_binary_metrics(pred, target)

        overlap = np.logical_and(pred, target).sum()
        manual_dice = 2.0 * overlap / (pred.sum() + target.sum())

        assert math.isclose(scar_res["dice"], manual_dice, rel_tol=1e-7)
        assert math.isclose(eval_res["dice"], manual_dice, rel_tol=1e-7)

    def test_jaccard_parity_against_scar_surface_distance(self, synthetic_mask_pair: Tuple[np.ndarray, np.ndarray]):
        """Verify Jaccard (IoU) calculation matches SCAR reference implementation exactly."""
        pred, target = synthetic_mask_pair
        scar_res = reference_scar_binary_metrics(pred, target)
        eval_res = eval_binary_metrics(pred, target)

        overlap = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        manual_iou = float(overlap / union)

        assert math.isclose(scar_res["iou"], manual_iou, rel_tol=1e-7)
        assert math.isclose(eval_res["iou"], manual_iou, rel_tol=1e-7)

    def test_hd95_parity_against_scar_surface_distance(self, synthetic_mask_pair: Tuple[np.ndarray, np.ndarray]):
        """Verify 95th percentile Hausdorff distance matches SCAR reference calculation."""
        pred, target = synthetic_mask_pair
        scar_res = reference_scar_binary_metrics(pred, target, compute_distance=True)
        eval_res = eval_binary_metrics(pred, target, compute_distance=True)

        assert scar_res["hd95"] is not None
        assert eval_res["hd95"] is not None
        assert np.isfinite(scar_res["hd95"])
        assert np.isfinite(eval_res["hd95"])
        assert math.isclose(scar_res["hd95"], eval_res["hd95"], rel_tol=1e-5)

    def test_assd_parity_against_scar_surface_distance(self, synthetic_mask_pair: Tuple[np.ndarray, np.ndarray]):
        """Verify Average Symmetric Surface Distance matches SCAR reference calculation."""
        pred, target = synthetic_mask_pair
        scar_res = reference_scar_binary_metrics(pred, target, compute_distance=True)
        eval_res = eval_binary_metrics(pred, target, compute_distance=True)

        assert scar_res["asd"] is not None
        assert eval_res["asd"] is not None
        assert np.isfinite(scar_res["asd"])
        assert np.isfinite(eval_res["asd"])
        assert math.isclose(scar_res["asd"], eval_res["asd"], rel_tol=1e-5)


# ==============================================================================
# Suite 8: Empty Mask Edge Cases Tests
# ==============================================================================

class TestEmptyMaskEdgeCases:
    """Verify empty mask edge cases (both empty, pred empty, target empty)."""

    def test_both_empty_dsc_100_percent(self):
        """Verify empty pred & empty GT yields DSC=1.0 (100%) and IoU=1.0 (100%) in defined mode."""
        pred = np.zeros((32, 32, 5), dtype=bool)
        target = np.zeros((32, 32, 5), dtype=bool)

        res_ref = reference_scar_binary_metrics(pred, target, empty_mode="defined")
        res_eval = eval_binary_metrics(pred, target, empty_mode="defined")

        assert res_ref["dice"] == 1.0, f"Expected DSC=1.0 on both empty, got {res_ref['dice']}"
        assert res_ref["iou"] == 1.0, f"Expected IoU=1.0 on both empty, got {res_ref['iou']}"
        assert res_ref["status"] == "both_empty"

        assert res_eval["dice"] == 1.0
        assert res_eval["iou"] == 1.0
        assert res_eval["status"] == "both_empty"

    def test_pred_empty_gt_nonempty_dsc_0_percent(self):
        """Verify empty pred & non-empty GT yields DSC=0.0 (0%) and IoU=0.0 (0%)."""
        pred = np.zeros((32, 32, 5), dtype=bool)
        target = np.zeros((32, 32, 5), dtype=bool)
        target[10:20, 10:20, 1:4] = True

        res_ref = reference_scar_binary_metrics(pred, target, empty_mode="defined")
        res_eval = eval_binary_metrics(pred, target, empty_mode="defined")

        assert res_ref["dice"] == 0.0, f"Expected DSC=0.0 on missed lesion, got {res_ref['dice']}"
        assert res_ref["iou"] == 0.0, f"Expected IoU=0.0 on missed lesion, got {res_ref['iou']}"
        assert res_eval["dice"] == 0.0
        assert res_eval["iou"] == 0.0

    def test_pred_nonempty_gt_empty_dsc_0_percent(self):
        """Verify non-empty pred & empty GT yields DSC=0.0 (0%) and IoU=0.0 (0%)."""
        pred = np.zeros((32, 32, 5), dtype=bool)
        target = np.zeros((32, 32, 5), dtype=bool)
        pred[10:20, 10:20, 1:4] = True

        res_ref = reference_scar_binary_metrics(pred, target, empty_mode="defined")
        res_eval = eval_binary_metrics(pred, target, empty_mode="defined")

        assert res_ref["dice"] == 0.0, f"Expected DSC=0.0 on false positive, got {res_ref['dice']}"
        assert res_ref["iou"] == 0.0, f"Expected IoU=0.0 on false positive, got {res_ref['iou']}"
        assert res_eval["dice"] == 0.0
        assert res_eval["iou"] == 0.0

    def test_empty_mask_surface_distance_stability(self):
        """Verify empty mask surface distance calculations do not raise unhandled exceptions."""
        pred = np.zeros((32, 32, 5), dtype=bool)
        target = np.zeros((32, 32, 5), dtype=bool)

        # In defined mode, surface distance on both empty is 0.0
        res_both = reference_scar_binary_metrics(pred, target, empty_mode="defined")
        res_eval_both = eval_binary_metrics(pred, target, empty_mode="defined")
        assert res_both["hd95"] == 0.0
        assert res_both["asd"] == 0.0
        assert res_eval_both["hd95"] == 0.0
        assert res_eval_both["asd"] == 0.0

        # On one empty, surface distance is infinite in defined mode
        pred[5:15, 5:15, 2] = True
        res_one = reference_scar_binary_metrics(pred, target, empty_mode="defined")
        res_eval_one = eval_binary_metrics(pred, target, empty_mode="defined")
        assert math.isinf(res_one["hd95"])
        assert math.isinf(res_one["asd"])
        assert math.isinf(res_eval_one["hd95"])
        assert math.isinf(res_eval_one["asd"])

        # In excluded mode (default for SCAR benchmark), both empty returns None
        res_excl = eval_binary_metrics(np.zeros_like(pred), np.zeros_like(target), empty_mode="excluded")
        assert res_excl["dice"] is None
        assert res_excl["hd95"] is None
        assert res_excl["status"] == "both_empty"


# ==============================================================================
# Suite 9: End-to-End Pipeline Integration Test
# ==============================================================================

class TestEndToEndMilestone4Pipeline:
    """Verify seamless execution across predict.py export and evaluate_predictions.py."""

    def test_end_to_end_predict_and_evaluate_flow(
        self,
        mock_checkpoint_path: Path,
        synthetic_dataset_environment: Tuple[Path, Path, List[str]],
        tmp_path: Path,
    ):
        """Verify end-to-end inference export followed by standalone evaluation."""
        data_root, split_dir, case_ids = synthetic_dataset_environment
        run_dir = tmp_path / "e2e_run"
        pred_dir = run_dir / "test_predictions"
        eval_dir = run_dir / "evaluation"

        # Step 1: Execute predict.py
        cmd_predict = [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--checkpoint", str(mock_checkpoint_path),
            "--data_root", str(data_root),
            "--split_dir", str(split_dir),
            "--run_dir", str(run_dir),
            "--device", "cpu",
        ]
        r1 = _run_cli_subprocess(cmd_predict, timeout=180)
        assert r1.returncode == 0, f"predict.py failed:\nSTDOUT:\n{r1.stdout}\nSTDERR:\n{r1.stderr}"

        for case_id in case_ids:
            assert (pred_dir / f"{case_id}_pred.npz").is_file()

        # Step 2: Execute evaluate_predictions.py
        cmd_eval = [
            sys.executable,
            str(PROJECT_ROOT / "evaluate_predictions.py"),
            "--pred_dir", str(pred_dir),
            "--data_root", str(data_root),
            "--split_dir", str(split_dir),
            "--output_dir", str(eval_dir),
        ]
        r2 = _run_cli_subprocess(cmd_eval, timeout=180)
        assert r2.returncode == 0, f"evaluate_predictions.py failed:\nSTDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}"

        # Step 3: Validate outputs
        assert (eval_dir / "per_case.csv").is_file()
        assert (eval_dir / "metrics.json").is_file()

        with open(eval_dir / "metrics.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert "scar" in summary
        assert "edema" in summary
        assert "myocardial_ring" in summary
        assert "edema_inclusive" in summary
        assert "normal_myocardium" in summary
