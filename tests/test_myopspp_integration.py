"""
test_myopspp_integration.py — Integration and Invariance Tests for MyoPS++
========================================================================
Tests:
  1. Model Architecture Invariance (MyoPSNetL structure, multi-branch, parameter count).
  2. Loss Function Invariance (MyoPSLoss, multi-head loss formulas).
  3. Label Mapping & Anti-Inversion (Canonical vs Legacy, raw MyoPS++ values).
  4. Real MyoPS++ Dataset Loading (train slices, test volumes, spacing, affine).
  5. Prediction & Evaluation Logic (Metric calculation, spacing, empty handling).
  6. Efficiency Benchmark Tool (Params, FLOPs, Latency measurement).
"""

import sys
from pathlib import Path

# Ensure MyoPS-Net repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch

from network.model import MyoPSNetL
from criterion.loss import MyoPSLoss
from utils.tools import create_multi_head_targets
from dataset.myops380_dataset import (
    MyoPS380TrainDataset,
    MyoPS380TestVolumeDataset,
    canonicalize_raw_myopspp,
    canonicalize_label,
)
from evaluate_predictions import evaluate_single_volume, binary_metrics
from benchmark_efficiency import compute_flops, measure_inference, run_benchmark


class TestModelArchitectureInvariance:
    """Verifies that MyoPSNetL architecture is strictly preserved."""

    def test_model_instantiation_and_heads(self):
        model = MyoPSNetL()
        model.eval()

        # Input: 3 modalities (bSSFP, LGE, T2w) cropped to 128x128
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            outputs = model(x)

        assert isinstance(outputs, (list, tuple))
        assert len(outputs) == 3, f"Expected 3 heads, got {len(outputs)}"

        myo_out, lge_out, t2_out = outputs
        # Head 1 (Myo ring): 2 classes (bg vs myo_ring)
        assert myo_out.shape == (2, 2, 128, 128)
        # Head 2 (LGE pathology): 3 classes (bg, edema, scar)
        assert lge_out.shape == (2, 3, 128, 128)
        # Head 3 (T2 pathology): 3 classes (bg, normal_myo, edema)
        assert t2_out.shape == (2, 3, 128, 128)

    def test_parameter_count_matches_original(self):
        model = MyoPSNetL()
        total_params = sum(p.numel() for p in model.parameters())
        # MyoPSNetL parameter count is exactly ~119.28M
        assert 119_000_000 < total_params < 120_000_000, f"Unexpected parameter count: {total_params}"


class TestLossFunctionInvariance:
    """Verifies that MyoPSLoss formulas and behavior are intact."""

    def test_loss_computation_and_backward(self):
        criterion = MyoPSLoss()
        model = MyoPSNetL()

        x = torch.randn(2, 3, 128, 128, requires_grad=True)
        outputs = model(x)

        # Targets for 3 heads:
        # Myo: (2, 128, 128) in [0, 1]
        # LGE: (2, 128, 128) in [0, 1, 2]
        # T2:  (2, 128, 128) in [0, 1, 2]
        target_myo = torch.randint(0, 2, (2, 128, 128), dtype=torch.long)
        target_lge = torch.randint(0, 3, (2, 128, 128), dtype=torch.long)
        target_t2 = torch.randint(0, 3, (2, 128, 128), dtype=torch.long)

        loss_total, loss_seg, loss_inv, loss_inc = criterion(outputs, (target_myo, target_lge, target_t2))
        assert torch.isfinite(loss_total), "Loss must be finite"
        assert loss_total.item() > 0, "Loss must be positive"

        # Verify gradient flow
        loss_total.backward()
        assert x.grad is not None, "Gradient must propagate to inputs"


class TestLabelStandardConsistency:
    """Verifies correct mapping between Canonical and Legacy standards to prevent label inversion."""

    def test_canonicalize_raw_myopspp(self):
        # Raw MyoPS++ labels:
        # 0: Background
        # 200: Normal Myocardium -> 1
        # 500: LV cavity -> 0
        # 600: RV cavity -> 0
        # 1220: Edema -> 2
        # 2221: Scar -> 3
        raw = np.array([0, 200, 500, 600, 1220, 2221], dtype=np.int64)
        canonical = canonicalize_raw_myopspp(raw)
        expected = np.array([0, 1, 0, 0, 2, 3], dtype=np.int64)
        np.testing.assert_array_equal(canonical, expected)

    def test_multi_head_target_generation_canonical_vs_legacy(self):
        """Crucial test: Proves that create_multi_head_targets correctly handles canonical mode."""
        # 1. Canonical label: 0:bg, 1:normal_myo, 2:edema, 3:scar
        canonical_label = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        myo_c, lge_c, t2_c = create_multi_head_targets(canonical_label, mode="canonical")

        # In canonical mode:
        # Myo head: 0->0, 1->1, 2->1, 3->1
        assert (myo_c == torch.tensor([[0, 1, 1, 1]])).all()
        # LGE head: 0->0, 1:normal->1, 2:edema->1, 3:scar->2
        assert (lge_c == torch.tensor([[0, 1, 1, 2]])).all()
        # T2 head: 0->0, 1:normal->1, 2:edema->2, 3:scar->2
        assert (t2_c == torch.tensor([[0, 1, 2, 2]])).all()

        # 2. Legacy label: 0:bg, 1:normal_myo, 2:scar, 3:edema
        legacy_label = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        myo_l, lge_l, t2_l = create_multi_head_targets(legacy_label, mode="legacy")

        # In legacy mode:
        # Myo head: 0->0, 1->1, 2->1, 3->1
        assert (myo_l == torch.tensor([[0, 1, 1, 1]])).all()
        # LGE head: 0->0, 1:normal->1, 2:scar->2, 3:edema->1
        assert (lge_l == torch.tensor([[0, 1, 2, 1]])).all()
        # T2 head: 0->0, 1:normal->1, 2:scar->2, 3:edema->2
        assert (t2_l == torch.tensor([[0, 1, 2, 2]])).all()

        # Verify that running canonical label with legacy mode INVERTS edema and scar:
        lge_inverted = create_multi_head_targets(canonical_label, mode="legacy")[1]
        assert lge_inverted[0, 2] == 2  # Edema gets mapped to 2 (scar)! Inversion prevented by mode='canonical'!
        assert lge_c[0, 2] == 1         # Under mode='canonical', Edema correctly gets mapped to 1 (non-scar)!


class TestRealDatasetLoading:
    """Verifies that MyoPS++ dataset loads properly from disk."""

    DATA_ROOT = Path(r"D:\nckh_model\Dataset\MyoPSpp_UNet2D_ROI128_1mm\MyoPSpp_UNet2D_ROI128_1mm")
    SPLIT_DIR = Path(r"D:\nckh_model\MyoPS-Net\dataset\splits\myopspp_unet2d_roi128_1mm")

    @pytest.mark.skipif(not Path(r"D:\nckh_model\Dataset\MyoPSpp_UNet2D_ROI128_1mm\MyoPSpp_UNet2D_ROI128_1mm").is_dir(), reason="Dataset directory not present")
    def test_train_slice_dataset(self):
        dataset = MyoPS380TrainDataset(
            split_file=self.SPLIT_DIR / "train.txt",
            data_root=self.DATA_ROOT,
            dataset="myopspp_unet2d_roi128_1mm",
            label_order="canonical",
        )
        assert len(dataset) == 296, f"Expected 296 slices, got {len(dataset)}"

        sample = dataset[0]
        assert "bSSFP" in sample and "LGE" in sample and "T2w" in sample and "label" in sample
        assert sample["bSSFP"].shape == (1, 128, 128)
        assert sample["LGE"].shape == (1, 128, 128)
        assert sample["T2w"].shape == (1, 128, 128)
        assert sample["label"].shape == (128, 128)
        assert sample["bSSFP"].dtype == torch.float32
        assert sample["label"].dtype == torch.long

        # Combined 3 modalities tensor
        stacked = torch.cat([sample["bSSFP"], sample["LGE"], sample["T2w"]], dim=0)
        assert stacked.shape == (3, 128, 128)

        unique_vals = set(sample["label"].unique().numpy())
        assert unique_vals.issubset({0, 1, 2, 3}), f"Unexpected label values: {unique_vals}"

    @pytest.mark.skipif(not Path(r"D:\nckh_model\Dataset\MyoPSpp_UNet2D_ROI128_1mm\MyoPSpp_UNet2D_ROI128_1mm").is_dir(), reason="Dataset directory not present")
    def test_test_volume_dataset(self):
        dataset = MyoPS380TestVolumeDataset(
            split_file=self.SPLIT_DIR / "test_vol.txt",
            data_root=self.DATA_ROOT,
            dataset="myopspp_unet2d_roi128_1mm",
        )
        assert len(dataset) == 16, f"Expected 16 test volumes, got {len(dataset)}"

        sample = dataset[0]
        assert "bSSFP" in sample and "LGE" in sample and "T2w" in sample and "label" in sample
        # Each modality shape: (1, 128, 128, D)
        assert sample["bSSFP"].dim() == 4
        assert sample["bSSFP"].shape[0] == 1
        assert sample["bSSFP"].shape[1:3] == (128, 128)
        assert sample["label"].shape[0:2] == (128, 128)
        assert sample["spacing"] is not None
        assert len(sample["spacing"]) == 3


class TestEvaluationAndMetrics:
    """Verifies evaluation metric calculation and empty mask handling."""

    def test_binary_metrics_exact(self):
        pred = np.zeros((10, 10, 10), dtype=np.uint8)
        target = np.zeros((10, 10, 10), dtype=np.uint8)
        pred[2:6, 2:6, 2:6] = 1
        target[2:6, 2:6, 2:6] = 1

        m = binary_metrics(pred, target, spacing=(1.0, 1.0, 1.0), compute_distance=True)
        assert m["dice"] == 1.0
        assert m["iou"] == 1.0
        assert m["hd95"] == 0.0
        assert m["asd"] == 0.0

    def test_evaluate_single_volume_canonical_gt(self):
        pred = np.zeros((128, 128, 5), dtype=np.uint8)
        gt = np.zeros((128, 128, 5), dtype=np.uint8)

        # Region 1: Myo
        gt[30:70, 30:70, :] = 1
        pred[30:70, 30:70, :] = 1

        # Region 2: Edema
        gt[40:50, 40:50, :] = 2
        pred[40:50, 40:50, :] = 2

        # Region 3: Scar
        gt[45:50, 45:50, :] = 3
        pred[45:50, 45:50, :] = 3

        detailed, summary = evaluate_single_volume(
            pred,
            gt,
            case_id="Case_test",
            spacing=(1.0, 1.0, 1.0),
            is_canonical_gt=True,
        )
        assert summary["edema_dice"] == 1.0
        assert summary["scar_dice"] == 1.0
        assert summary["myocardial_ring_dice"] == 1.0


class TestBenchmarkEfficiencyTool:
    """Verifies that the benchmark efficiency tool runs without error."""

    def test_compute_flops_and_inference(self):
        model = MyoPSNetL()
        device = torch.device("cpu")
        flops = compute_flops(model, input_size=(1, 3, 128, 128), device=device)
        assert flops > 0, "FLOPs must be positive"

        inf_res = measure_inference(model, device=device, warmup_runs=2, measure_runs=3)
        assert "latency_ms_per_slice" in inf_res
        assert inf_res["latency_ms_per_slice"] > 0


class TestAugmentationInvariance:
    """Verifies that RandomGenerator augmentation strictly preserves label values and dimensions."""

    def test_random_generator_preserves_label_set(self):
        from dataset.augmentation import RandomGenerator
        aug = RandomGenerator(output_size=[128, 128])
        sample = {
            "image": np.random.rand(128, 128).astype(np.float32),
            "image1": np.random.rand(128, 128).astype(np.float32),
            "image2": np.random.rand(128, 128).astype(np.float32),
            "label": np.random.choice([0, 1, 2, 3], size=(128, 128)).astype(np.int64),
        }
        res = aug(sample)
        assert set(np.unique(res["label"])).issubset({0, 1, 2, 3})
        assert res["image"].shape == (1, 128, 128)
        assert res["label"].shape == (128, 128)


class TestPredictPathFallback:
    """Verifies that predict.py resolve_paths handles nested checkpoints and missing dirs."""

    def test_resolve_paths_fallback(self, tmp_path):
        from predict import resolve_paths
        import argparse

        run_dir = tmp_path / "test_run"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt_file = ckpt_dir / "myopsnet_seed1234.pt"
        ckpt_file.write_text("dummy", encoding="utf-8")

        # Passing checkpoint pointing to run_dir/<name> directly when it only exists in checkpoints/<name>
        dummy_cp = run_dir / "myopsnet_seed1234.pt"
        args = argparse.Namespace(
            checkpoint=str(dummy_cp),
            run_dir=None,
            seed=1234,
            pred_dir=None,
            dataset="auto",
            data_root=str(tmp_path),
            split_dir=str(tmp_path),
            split="test_vol",
            test_manifest=None,
        )
        resolved_cp, _, _, _, _, _, _ = resolve_paths(args)
        assert resolved_cp == ckpt_file.resolve()

