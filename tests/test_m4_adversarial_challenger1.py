"""
tests/test_m4_adversarial_challenger1.py
=========================================
Empirical Adversarial Stress-Test Suite for Milestone 4 (predict.py).
Target: Deterministic Single-Pass Volume Inference Engine.

Challenge Vectors:
1. Volume Depth Variations: Extreme depth values (D=1, 2, 7, 10, 17, and primes).
2. Arbitrary Batch Sizes: B=1, 3, 7, 64 and cross-batch deterministic invariance.
3. Spatial Resolution Variations: (96, 96, D), (256, 256, D), non-square (160, 144, D), prime dims.
4. Adversarial Strict NO-TTA Invariant: Forward pass count == ceil(D / B_slice) with zero flips/rotations.
5. Adversarial Strict NO-LCC Invariant: Preservation of 1-voxel and 2-voxel isolated lesion islands.
6. Corrupted Checkpoint & Error Handling: Non-existent, 0-byte, garbage-byte, invalid payload, and input validation.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from network.model import MyoPSNetL
from predict import build_parser, main as predict_main, predict_volume, resolve_paths
from training.checkpoint import load_checkpoint, save_checkpoint
from utils.tools import fuse_predictions


# ==============================================================================
# Helper Utilities & Fixtures
# ==============================================================================

def _run_cli_subprocess(cmd: Sequence[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """Execute CLI command safely across Windows/Linux with process tree kill on timeout."""
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


class FastMockModel(nn.Module):
    """Fast, deterministic mock model returning structured 3-head multi-class outputs."""

    def __init__(self, mode: str = "normal"):
        super().__init__()
        self.mode = mode
        # Dummy parameter to satisfy next(model.parameters()).device
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        c0: torch.Tensor,
        lge: torch.Tensor,
        t2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = c0.shape
        device = c0.device

        # Default: C0 -> 2 classes, LGE -> 3 classes, T2 -> 3 classes
        seg_c0 = torch.zeros((batch_size, 2, height, width), device=device)
        seg_lge = torch.zeros((batch_size, 3, height, width), device=device)
        seg_t2 = torch.zeros((batch_size, 3, height, width), device=device)

        if self.mode == "normal":
            # Class 0: background dominant
            seg_c0[:, 0, :, :] = 5.0
            seg_lge[:, 0, :, :] = 5.0
            seg_t2[:, 0, :, :] = 5.0
            # Normal myocardium in central square [40:88, 40:88]
            seg_c0[:, 1, 40:88, 40:88] = 10.0
        elif self.mode == "nan":
            seg_c0[:, 0, :, :] = float("nan")
        elif self.mode == "inf":
            seg_lge[:, 0, :, :] = float("inf")

        return seg_c0, seg_lge, seg_t2


@pytest.fixture
def mock_checkpoint_path(tmp_path: Path) -> Path:
    """Creates a minimal valid checkpoint for testing CLI loading."""
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


# ==============================================================================
# Suite 1: Volume Depth Variations (D=1, 2, 7, 10, 17, and Primes)
# ==============================================================================

class TestVolumeDepthVariations:
    """Adversarial testing of volume depth variations from D=1 to large primes."""

    @pytest.mark.parametrize("depth", [1, 2, 7, 10, 17, 13, 23])
    def test_predict_volume_various_depths_fast(self, depth: int):
        """Verify predict_volume processes volumes of arbitrary depth D without shape collapse."""
        model = FastMockModel(mode="normal")
        H, W = 128, 128
        images = [
            np.ones((H, W, depth), dtype=np.float32) * 0.5,
            np.ones((H, W, depth), dtype=np.float32) * 0.3,
            np.ones((H, W, depth), dtype=np.float32) * 0.7,
        ]

        pred = predict_volume(model, images, img_size=128, batch_size=8, device="cpu")

        assert isinstance(pred, np.ndarray), "Output must be a numpy ndarray"
        assert pred.shape == (H, W, depth), f"Expected shape {(H, W, depth)}, got {pred.shape}"
        assert pred.dtype == np.uint8, f"Expected dtype uint8, got {pred.dtype}"
        unique_vals = set(np.unique(pred))
        assert unique_vals.issubset({0, 1, 2, 3}), f"Unexpected labels in prediction: {unique_vals}"

    def test_depth_d1_single_slice_no_squeeze(self):
        """Stress-test D=1: Ensure 1-slice volume does NOT collapse into 2D array (H, W)."""
        model = FastMockModel(mode="normal")
        H, W, D = 128, 128, 1
        images = [np.ones((H, W, D), dtype=np.float32) for _ in range(3)]

        pred = predict_volume(model, images, img_size=128, batch_size=8, device="cpu")

        assert pred.ndim == 3, f"D=1 volume collapsed to ndim={pred.ndim}"
        assert pred.shape == (H, W, 1), f"Expected shape {(H, W, 1)}, got {pred.shape}"
        assert pred.dtype == np.uint8

    def test_depth_d17_multiple_batches_tail_placement(self):
        """Stress-test D=17 with batch_size=8: Verifies 8 + 8 + 1 slices across 3 batches."""
        # Create a model that tags each slice with its depth index into normal myocardium
        class DepthTaggingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
                self.slice_counter = 0

            def forward(self, c0, lge, t2):
                B, _, H, W = c0.shape
                seg_c0 = torch.zeros((B, 2, H, W))
                seg_lge = torch.zeros((B, 3, H, W))
                seg_t2 = torch.zeros((B, 3, H, W))
                seg_c0[:, 0, :, :] = 5.0
                seg_lge[:, 0, :, :] = 5.0
                seg_t2[:, 0, :, :] = 5.0
                # Use c0 value to encode slice id
                # For each slice b in batch: if c0 has tag, predict normal myo at that tag location
                for b in range(B):
                    tag = int(c0[b, 0, 0, 0].item())
                    # mark coordinate (tag, tag)
                    seg_c0[b, 1, tag:tag + 2, tag:tag + 2] = 10.0
                return seg_c0, seg_lge, seg_t2

        model = DepthTaggingModel()
        H, W, D = 128, 128, 17
        images = [np.zeros((H, W, D), dtype=np.float32) for _ in range(3)]
        for z in range(D):
            images[0][:, :, z] = float(z + 5)  # tags from 5 to 21

        pred = predict_volume(model, images, img_size=128, batch_size=8, device="cpu")

        assert pred.shape == (H, W, 17)
        for z in range(D):
            tag = z + 5
            # At slice z, voxels [tag:tag+2, tag:tag+2] must be 1 (Normal)
            assert np.all(pred[tag:tag + 2, tag:tag + 2, z] == 1), f"Slice {z} was corrupted or misplaced!"
            # Other coordinates should be 0 (Background)
            assert pred[0, 0, z] == 0

    def test_cli_volume_depth_variations_end_to_end(
        self,
        mock_checkpoint_path: Path,
        tmp_path: Path,
    ):
        """Verify predict.py CLI correctly processes test dataset with varied depths (D=1, 7, 17)."""
        data_root = tmp_path / "Processed_data"
        split_dir = tmp_path / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)

        case_depths = [("case0001", 1), ("case0002", 7), ("case0003", 17)]
        case_ids = [c[0] for c in case_depths]
        (split_dir / "test_vol.txt").write_text("\n".join(case_ids) + "\n", encoding="utf-8")

        for case_id, depth in case_depths:
            for mod in ("bSSFP", "LGE", "T2w"):
                mod_dir = data_root / mod / "test_vol_h5"
                mod_dir.mkdir(parents=True, exist_ok=True)
                h5_file = mod_dir / f"{case_id}.npy.h5"
                with h5py.File(h5_file, "w") as f:
                    f.create_dataset("image", data=np.ones((128, 128, depth), dtype=np.float32) * 0.5)
                    if mod == "bSSFP":
                        f.create_dataset("label", data=np.zeros((128, 128, depth), dtype=np.int64))

        run_dir = tmp_path / "run_cli_depths"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--checkpoint", str(mock_checkpoint_path),
            "--data_root", str(data_root),
            "--split_dir", str(split_dir),
            "--run_dir", str(run_dir),
            "--device", "cpu",
        ]
        res = _run_cli_subprocess(cmd, timeout=120)
        assert res.returncode == 0, f"predict.py CLI failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

        pred_dir = run_dir / "test_predictions"
        assert pred_dir.is_dir()

        for case_id, depth in case_depths:
            npz_path = pred_dir / f"{case_id}_pred.npz"
            assert npz_path.is_file(), f"Missing prediction file for {case_id}"
            with np.load(npz_path) as data:
                p = data["prediction"]
                assert p.shape == (128, 128, depth), f"{case_id} shape {p.shape} != (128, 128, {depth})"
                assert p.dtype == np.uint8


# ==============================================================================
# Suite 2: Arbitrary and Odd Batch Sizes (B=1, 3, 7, 64) & Invariance
# ==============================================================================

class TestArbitraryBatchSizes:
    """Adversarial testing of strange, prime, and oversized batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 3, 7, 64])
    def test_predict_volume_arbitrary_batch_sizes(self, batch_size: int):
        """Verify predict_volume runs seamlessly with B=1, B=3, B=7, and B=64."""
        model = FastMockModel(mode="normal")
        H, W, D = 128, 128, 10
        images = [np.ones((H, W, D), dtype=np.float32) * 0.4 for _ in range(3)]

        pred = predict_volume(model, images, img_size=128, batch_size=batch_size, device="cpu")
        assert pred.shape == (H, W, D)
        assert pred.dtype == np.uint8

    def test_deterministic_invariance_across_batch_sizes(self):
        """Strict Invariant: Predictions for identical inputs MUST BE IDENTICAL across batch sizes."""
        # Using real MyoPSNetL model to verify true layer invariance
        model = MyoPSNetL().eval()
        torch.manual_seed(42)

        H, W, D = 128, 128, 11
        np.random.seed(42)
        images = [np.random.randn(H, W, D).astype(np.float32) for _ in range(3)]

        pred_b1 = predict_volume(model, images, img_size=128, batch_size=1, device="cpu")
        pred_b3 = predict_volume(model, images, img_size=128, batch_size=3, device="cpu")
        pred_b8 = predict_volume(model, images, img_size=128, batch_size=8, device="cpu")
        pred_b64 = predict_volume(model, images, img_size=128, batch_size=64, device="cpu")

        assert np.array_equal(pred_b1, pred_b3), "Mismatch between batch_size=1 and batch_size=3"
        assert np.array_equal(pred_b1, pred_b8), "Mismatch between batch_size=1 and batch_size=8"
        assert np.array_equal(pred_b1, pred_b64), "Mismatch between batch_size=1 and batch_size=64"

    @pytest.mark.parametrize("invalid_b", [0, -1, -8])
    def test_invalid_batch_size_rejected(self, invalid_b: int):
        """Verify non-positive batch_size is immediately rejected with ValueError."""
        model = FastMockModel(mode="normal")
        images = [np.zeros((128, 128, 4), dtype=np.float32) for _ in range(3)]
        with pytest.raises(ValueError, match="batch_size must be positive"):
            predict_volume(model, images, batch_size=invalid_b, device="cpu")


# ==============================================================================
# Suite 3: Spatial Resolution Variations (96x96, 256x256, Non-Square, Primes)
# ==============================================================================

class TestSpatialResolutionVariations:
    """Stress-test non-standard spatial input shapes and bilinear re-interpolation."""

    @pytest.mark.parametrize("H, W, D", [
        (96, 96, 4),       # Downscaled square
        (256, 256, 4),     # Upscaled square
        (160, 144, 3),     # Rectangular non-square
        (131, 137, 2),     # Odd prime spatial dimensions
        (77, 85, 5),       # Small non-square
    ])
    def test_spatial_resolution_shapes_preserved(self, H: int, W: int, D: int):
        """Verify output predictions strictly match native (H, W, D) dimensions."""
        model = FastMockModel(mode="normal")
        images = [np.ones((H, W, D), dtype=np.float32) * 0.5 for _ in range(3)]

        pred = predict_volume(model, images, img_size=128, batch_size=4, device="cpu")

        assert pred.shape == (H, W, D), f"Expected shape {(H, W, D)}, got {pred.shape}"
        assert pred.dtype == np.uint8
        assert set(np.unique(pred)).issubset({0, 1, 2, 3})

    def test_spatial_resolution_with_real_model(self):
        """Verify real MyoPSNetL executes forward and interpolation correctly on non-square resolution."""
        model = MyoPSNetL().eval()
        H, W, D = 144, 160, 3
        images = [np.random.randn(H, W, D).astype(np.float32) for _ in range(3)]

        pred = predict_volume(model, images, img_size=128, batch_size=2, device="cpu")

        assert pred.shape == (H, W, D)
        assert pred.dtype == np.uint8


# ==============================================================================
# Suite 4: Adversarial Strict NO-TTA Verification (Counting Forward Passes)
# ==============================================================================

class TestAdversarialNoTTAVerification:
    """Adversarial spy instrumentation to prove zero test-time augmentation."""

    @pytest.mark.parametrize("D, B_slice, expected_passes", [
        (1, 8, 1),
        (2, 8, 1),
        (7, 8, 1),
        (8, 8, 1),
        (10, 8, 2),
        (16, 8, 2),
        (17, 8, 3),
        (10, 3, 4),
        (17, 1, 17),
        (17, 64, 1),
        (23, 7, 4),
    ])
    def test_exact_forward_pass_count(self, D: int, B_slice: int, expected_passes: int):
        """Adversarial Invariant: Number of model forward passes MUST strictly equal ceil(D / B_slice)."""
        forward_call_count = 0
        recorded_batch_sizes = []

        class SpyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

            def forward(self, c0, lge, t2):
                nonlocal forward_call_count, recorded_batch_sizes
                forward_call_count += 1
                B, _, H, W = c0.shape
                recorded_batch_sizes.append(B)
                seg_c0 = torch.zeros((B, 2, H, W))
                seg_lge = torch.zeros((B, 3, H, W))
                seg_t2 = torch.zeros((B, 3, H, W))
                return seg_c0, seg_lge, seg_t2

        model = SpyModel()
        images = [np.zeros((128, 128, D), dtype=np.float32) for _ in range(3)]

        _ = predict_volume(model, images, img_size=128, batch_size=B_slice, device="cpu")

        assert forward_call_count == expected_passes, (
            f"Adversarial TTA Failure for D={D}, B={B_slice}: "
            f"expected {expected_passes} forward calls, got {forward_call_count}!"
        )
        assert sum(recorded_batch_sizes) == D, (
            f"Total slices processed across forward passes {sum(recorded_batch_sizes)} != D={D}"
        )

    def test_no_hidden_transforms_or_rotations(self):
        """Adversarial check: Ensure slices are passed in natural orientation without flips or rotations."""
        captured_inputs = []

        class InspectionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

            def forward(self, c0, lge, t2):
                captured_inputs.append(c0.clone().cpu())
                B, _, H, W = c0.shape
                return (
                    torch.zeros((B, 2, H, W)),
                    torch.zeros((B, 3, H, W)),
                    torch.zeros((B, 3, H, W)),
                )

        model = InspectionModel()
        H, W, D = 128, 128, 3
        # Put an asymmetrical gradient across x and y
        ramp = np.arange(H)[:, None] * 10.0 + np.arange(W)[None, :]
        images = [np.repeat(ramp[:, :, None], D, axis=2).astype(np.float32) for _ in range(3)]

        _ = predict_volume(model, images, img_size=128, batch_size=8, device="cpu")

        assert len(captured_inputs) == 1
        tensor_batch = captured_inputs[0]  # (3, 1, 128, 128)
        assert tensor_batch.shape == (3, 1, 128, 128)

        # Check first slice matches original orientation exactly (no flip, no transpose)
        slice_np = tensor_batch[0, 0].numpy()
        assert np.allclose(slice_np, ramp), "Input slice was secretly transformed, flipped, or rotated!"


# ==============================================================================
# Suite 5: Adversarial Strict NO-LCC Verification (Preserving Tiny Lesion Islands)
# ==============================================================================

class TestAdversarialNoLCCVerification:
    """Adversarial test synthesizing 1-voxel and 2-voxel isolated islands far from main lesions."""

    def test_isolated_tiny_lesion_islands_preserved_in_volume(self):
        """Verify that predict_volume strictly does NOT filter out tiny isolated lesion clusters."""
        H, W, D = 128, 128, 4

        class IsolatedIslandModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

            def forward(self, c0, lge, t2):
                B, _, H, W = c0.shape
                seg_c0 = torch.zeros((B, 2, H, W))
                seg_lge = torch.zeros((B, 3, H, W))
                seg_t2 = torch.zeros((B, 3, H, W))

                # Background default
                seg_c0[:, 0, :, :] = 5.0
                seg_lge[:, 0, :, :] = 5.0
                seg_t2[:, 0, :, :] = 5.0

                # Normal Myocardium ring everywhere
                seg_c0[:, 1, 30:100, 30:100] = 10.0

                # Slice 1: Primary Large Scar cluster (50 voxels) at [50:60, 50:55]
                if B > 1:
                    seg_lge[1, 2, 50:60, 50:55] = 15.0
                    # Isolated tiny 1-voxel Scar island at (10, 10), far away
                    seg_lge[1, 2, 10, 10] = 15.0

                # Slice 2: Isolated tiny 2-voxel Scar island at (115:117, 115:116)
                if B > 2:
                    seg_lge[2, 2, 115:117, 115:116] = 15.0
                    # Isolated tiny 1-voxel Edema island at (15, 15)
                    seg_t2[2, 2, 15, 15] = 15.0

                # Slice 3: Isolated tiny 2-voxel Edema island at (90:92, 20:21)
                if B > 3:
                    seg_t2[3, 2, 90:92, 20:21] = 15.0

                return seg_c0, seg_lge, seg_t2

        model = IsolatedIslandModel()
        images = [np.zeros((H, W, D), dtype=np.float32) for _ in range(3)]

        pred = predict_volume(model, images, img_size=128, batch_size=4, device="cpu")

        # 1. Verify Slice 1 primary scar is present
        assert np.all(pred[50:60, 50:55, 1] == 3), "Primary scar lesion missing!"

        # 2. Verify Slice 1 isolated 1-voxel scar island at (10, 10) survives
        assert pred[10, 10, 1] == 3, "Adversarial LCC Violation: 1-voxel scar island was eliminated!"

        # 3. Verify Slice 2 isolated 2-voxel scar island at (115:117, 115:116) survives
        assert np.all(pred[115:117, 115:116, 2] == 3), "Adversarial LCC Violation: 2-voxel scar island was eliminated!"

        # 4. Verify Slice 2 isolated 1-voxel edema island at (15, 15) survives
        assert pred[15, 15, 2] == 2, "Adversarial LCC Violation: 1-voxel edema island was eliminated!"

        # 5. Verify Slice 3 isolated 2-voxel edema island at (90:92, 20:21) survives
        assert np.all(pred[90:92, 20:21, 3] == 2), "Adversarial LCC Violation: 2-voxel edema island was eliminated!"

        # 6. Total count verification
        total_scar_voxels = np.sum(pred == 3)
        expected_scar = (10 * 5) + 1 + 2  # 50 + 1 + 2 = 53
        assert total_scar_voxels == expected_scar, (
            f"Expected {expected_scar} scar voxels, got {total_scar_voxels}"
        )

        total_edema_voxels = np.sum(pred == 2)
        expected_edema = 1 + 2  # 1 + 2 = 3
        assert total_edema_voxels == expected_edema, (
            f"Expected {expected_edema} edema voxels, got {total_edema_voxels}"
        )


# ==============================================================================
# Suite 6: Corrupted Checkpoint & Error Handling
# ==============================================================================

class TestCorruptedCheckpointAndErrorHandling:
    """Adversarial stress-testing of corrupted files, invalid inputs, and error exits."""

    def test_cli_nonexistent_checkpoint_exits_nonzero(self, tmp_path: Path):
        """Verify non-existent checkpoint path fails cleanly with non-zero exit code."""
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--checkpoint", str(tmp_path / "does_not_exist_404.pt"),
            "--data_root", str(tmp_path),
            "--split_dir", str(tmp_path),
        ]
        res = _run_cli_subprocess(cmd, timeout=30)
        assert res.returncode != 0
        assert "not found" in res.stderr.lower() or "not found" in res.stdout.lower()

    def test_cli_empty_checkpoint_file_exits_nonzero(self, tmp_path: Path):
        """Verify 0-byte corrupted checkpoint file exits with non-zero code."""
        empty_ckpt = tmp_path / "empty_corrupted.pt"
        empty_ckpt.write_bytes(b"")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--checkpoint", str(empty_ckpt),
            "--data_root", str(tmp_path),
            "--split_dir", str(tmp_path),
        ]
        res = _run_cli_subprocess(cmd, timeout=30)
        assert res.returncode != 0

    def test_cli_garbage_corrupted_checkpoint_exits_nonzero(self, tmp_path: Path):
        """Verify garbage-filled checkpoint file exits cleanly with non-zero code."""
        garbage_ckpt = tmp_path / "garbage_corrupted.pt"
        garbage_ckpt.write_bytes(b"GARBAGE_BYTES_RANDOM_DATA_!@#$%^&*()_+1234567890\x00\xFF\xFE")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--checkpoint", str(garbage_ckpt),
            "--data_root", str(tmp_path),
            "--split_dir", str(tmp_path),
        ]
        res = _run_cli_subprocess(cmd, timeout=30)
        assert res.returncode != 0

    def test_cli_missing_manifest_exits_nonzero(self, mock_checkpoint_path: Path, tmp_path: Path):
        """Verify non-existent manifest split file exits with non-zero code."""
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--checkpoint", str(mock_checkpoint_path),
            "--data_root", str(tmp_path),
            "--split_dir", str(tmp_path / "nonexistent_splits_dir"),
        ]
        res = _run_cli_subprocess(cmd, timeout=30)
        assert res.returncode != 0

    def test_predict_volume_input_modality_validation(self):
        """Verify predict_volume rejects incorrect modality count (must be exactly 3)."""
        model = FastMockModel()
        vol = np.zeros((128, 128, 4), dtype=np.float32)

        # 2 modalities
        with pytest.raises(ValueError, match="Expected exactly 3 aligned modality arrays"):
            predict_volume(model, [vol, vol])

        # 4 modalities
        with pytest.raises(ValueError, match="Expected exactly 3 aligned modality arrays"):
            predict_volume(model, [vol, vol, vol, vol])

    def test_predict_volume_input_dimension_validation(self):
        """Verify predict_volume rejects non-3D input arrays."""
        model = FastMockModel()
        vol_2d = np.zeros((128, 128), dtype=np.float32)
        vol_4d = np.zeros((1, 128, 128, 4), dtype=np.float32)

        with pytest.raises(ValueError, match="Expected 3D arrays of shape"):
            predict_volume(model, [vol_2d, vol_2d, vol_2d])

        with pytest.raises(ValueError, match="Expected 3D arrays of shape"):
            predict_volume(model, [vol_4d, vol_4d, vol_4d])

    def test_predict_volume_mismatched_shapes_validation(self):
        """Verify predict_volume rejects modalities with mismatched shapes."""
        model = FastMockModel()
        vol1 = np.zeros((128, 128, 4), dtype=np.float32)
        vol2 = np.zeros((128, 128, 5), dtype=np.float32)

        with pytest.raises(ValueError, match="Modalities must have identical shapes"):
            predict_volume(model, [vol1, vol2, vol1])

    def test_predict_volume_nan_inf_input_validation(self):
        """Verify predict_volume rejects NaN and Inf input volume values."""
        model = FastMockModel()
        vol_clean = np.zeros((128, 128, 4), dtype=np.float32)

        vol_nan = vol_clean.copy()
        vol_nan[10, 10, 1] = float("nan")
        with pytest.raises(ValueError, match="Non-finite values detected"):
            predict_volume(model, [vol_nan, vol_clean, vol_clean])

        vol_inf = vol_clean.copy()
        vol_inf[10, 10, 1] = float("inf")
        with pytest.raises(ValueError, match="Non-finite values detected"):
            predict_volume(model, [vol_clean, vol_inf, vol_clean])

    def test_predict_volume_nan_inf_model_output_validation(self):
        """Verify FloatingPointError is raised if model outputs non-finite logits."""
        vol = np.zeros((128, 128, 4), dtype=np.float32)

        nan_model = FastMockModel(mode="nan")
        with pytest.raises(FloatingPointError, match="Non-finite output detected"):
            predict_volume(nan_model, [vol, vol, vol], device="cpu")

        inf_model = FastMockModel(mode="inf")
        with pytest.raises(FloatingPointError, match="Non-finite output detected"):
            predict_volume(inf_model, [vol, vol, vol], device="cpu")
