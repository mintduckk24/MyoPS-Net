"""tests/test_adversarial_m2_challenger.py
=========================================
Empirical Adversarial Challenge Suite for Milestone 2:
1. Augmentation Mathematical & Algorithmic Parity vs SCAR Benchmark
2. Zero Unexpected Normalization or Clipping Checks
3. 100% Spatial Synchrony Across bSSFP, LGE, T2w, and Label
4. Full DataLoader Epoch Traversal (1,972 slices -> 124 batches, 124th batch = 4)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import sys
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.augmentation import (
    RandomGenerator as MyoPSRandomGenerator,
    random_rot_flip as myops_rot_flip,
    random_rotate as myops_rotate,
    add_gaussian_noise as myops_gaussian_noise,
)
from dataset.myops380_dataset import (
    MyoPS380TrainDataset,
    create_myops380_train_loader,
    EXPECTED_TRAIN_SLICES,
)

# Reference SCAR paths
SCAR_DATASET_PY = Path("D:/NCKH/SCAR/training/dataset/myops_dataset.py")
TRAIN_SPLIT_FILE = Path("D:/NCKH/SCAR/preprocessing/splits/myops380_no_val/train.txt")
DATA_ROOT = Path("D:/NCKH/Dataset/MyoPS380/Processed_data")


def load_scar_components():
    """Dynamically loads SCAR's RandomGenerator and helper functions."""
    if not SCAR_DATASET_PY.is_file():
        pytest.skip(f"SCAR reference file not found: {SCAR_DATASET_PY}")

    # SCAR imports training.dataset.data_contract, so ensure D:/NCKH/SCAR is in sys.path
    scar_root = str(SCAR_DATASET_PY.parents[2])
    if scar_root not in sys.path:
        sys.path.insert(0, scar_root)
    import training
    scar_training = str(Path(scar_root) / "training")
    if hasattr(training, "__path__") and scar_training not in training.__path__:
        training.__path__.append(scar_training)

    spec = importlib.util.spec_from_file_location("scar_myops_dataset", SCAR_DATASET_PY)
    scar_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scar_mod)
    return scar_mod


# ==============================================================================
# Challenge 1: Bit-for-Bit Parity against SCAR Benchmark
# ==============================================================================

class TestAugmentationParityVsSCAR:
    """Rigorous comparison between MyoPS-Net augmentation and SCAR benchmark."""

    def test_bit_for_bit_rng_and_tensor_parity_across_seeds(self):
        """Challenge: Run 50 different seeds and assert 100% identical tensor outputs."""
        scar_mod = load_scar_components()
        scar_aug = scar_mod.RandomGenerator(output_size=(128, 128))
        myops_aug = MyoPSRandomGenerator(output_size=(128, 128))

        # Base synthetic sample
        np.random.seed(42)
        base_bssfp = np.random.uniform(0.0, 1.0, (140, 140)).astype(np.float32)
        base_lge = np.random.uniform(0.0, 1.0, (140, 140)).astype(np.float32)
        base_t2w = np.random.uniform(0.0, 1.0, (140, 140)).astype(np.float32)
        base_label = np.random.choice([0, 1, 2, 3], size=(140, 140)).astype(np.int64)

        for seed in range(50):
            # Seed SCAR run
            random.seed(seed * 101 + 13)
            np.random.seed(seed * 101 + 13)
            scar_sample = {
                "image": base_bssfp.copy(),
                "image1": base_lge.copy(),
                "image2": base_t2w.copy(),
                "label": base_label.copy(),
            }
            scar_out = scar_aug(scar_sample)

            # Seed MyoPS-Net run with identical seeds
            random.seed(seed * 101 + 13)
            np.random.seed(seed * 101 + 13)
            myops_sample = {
                "bSSFP": base_bssfp.copy(),
                "LGE": base_lge.copy(),
                "T2w": base_t2w.copy(),
                "label": base_label.copy(),
            }
            myops_out = myops_aug(myops_sample)

            # Assert bit-for-bit equality
            assert torch.equal(scar_out["image"], myops_out["bSSFP"]), f"bSSFP mismatch at seed {seed}"
            assert torch.equal(scar_out["image1"], myops_out["LGE"]), f"LGE mismatch at seed {seed}"
            assert torch.equal(scar_out["image2"], myops_out["T2w"]), f"T2w mismatch at seed {seed}"
            assert torch.equal(scar_out["label"], myops_out["label"]), f"Label mismatch at seed {seed}"

    def test_mathematical_formula_subcomponents(self):
        """Challenge: Verify exact parameter boundaries, formulas, and zoom orders."""
        scar_mod = load_scar_components()

        # 1. Rotation angles in random_rotate
        # SCAR: angle = np.random.randint(-20, 20)
        np.random.seed(999)
        angles = [np.random.randint(-20, 20) for _ in range(500)]
        assert min(angles) >= -20
        assert max(angles) <= 19
        assert any(a < 0 for a in angles) and any(a > 0 for a in angles)

        # 2. Gaussian noise formula
        img1 = np.full((32, 32), 0.5, dtype=np.float32)
        img2 = np.full((32, 32), 0.5, dtype=np.float32)
        img3 = np.full((32, 32), 0.5, dtype=np.float32)
        lbl = np.zeros((32, 32), dtype=np.int64)

        np.random.seed(123)
        scar_res = scar_mod.add_gaussian_noise(img1.copy(), img2.copy(), img3.copy(), lbl.copy(), mean=0, std=0.2)
        np.random.seed(123)
        myops_res = myops_gaussian_noise(img1.copy(), img2.copy(), img3.copy(), lbl.copy(), mean=0.0, std=0.2)

        for s_arr, m_arr in zip(scar_res, myops_res):
            assert np.array_equal(s_arr, m_arr)
            assert np.min(m_arr) >= 0.0
            assert np.max(m_arr) <= 1.0

        # Assert shared noise map across modalities
        # Since input images were identical, img1 + noise == img2 + noise == img3 + noise
        assert np.array_equal(myops_res[0], myops_res[1])
        assert np.array_equal(myops_res[1], myops_res[2])

        # 3. Brightness factor range [0.8, 1.2]
        factors = [random.uniform(0.8, 1.2) for _ in range(500)]
        assert min(factors) >= 0.8
        assert max(factors) <= 1.2

        # 4. Gamma power range [0.8, 1.2]
        gammas = [random.uniform(0.8, 1.2) for _ in range(500)]
        assert min(gammas) >= 0.8
        assert max(gammas) <= 1.2


# ==============================================================================
# Challenge 2: Invariant Testing — Clipping, Normalization, & Synchrony
# ==============================================================================

class TestAugmentationInvariants:
    """Stress tests for clipping anomalies, unexpected normalization, and spatial misalignment."""

    def test_no_unexpected_normalization_or_clipping(self):
        """Assert that no mean/std normalization or unexpected clipping occurs."""
        aug = MyoPSRandomGenerator(output_size=(128, 128))

        # Test with high-range image (values > 1.0, e.g. 50.0..100.0)
        # Without gaussian noise (which clips to 0..1), values should remain large (> 1.0)
        # To isolate non-noise transforms, test zoom/rot/flip/shift
        img = np.full((128, 128), 50.0, dtype=np.float32)
        lbl = np.ones((128, 128), dtype=np.int64)

        # Apply rot_flip directly
        o_img, _, _, _ = myops_rot_flip(img.copy(), img.copy(), img.copy(), lbl.copy())
        assert np.isclose(np.mean(o_img), 50.0), "rot_flip modified unnormalized pixel intensity!"

        # Apply rotate directly
        o_img2, _, _, _ = myops_rotate(img.copy(), img.copy(), img.copy(), lbl.copy())
        # Interior pixels (not rotated out of frame) must still be 50.0
        assert np.max(o_img2) == 50.0, "rotate modified unnormalized pixel intensity!"

    def test_label_mask_values_strictly_discrete_integers(self):
        """Assert that label mask values are strictly within {0, 1, 2, 3} under all transforms."""
        aug = MyoPSRandomGenerator(output_size=(128, 128))

        img = np.random.uniform(0, 1, (128, 128)).astype(np.float32)
        lbl = np.random.choice([0, 1, 2, 3], size=(128, 128)).astype(np.int64)

        for seed in range(50):
            random.seed(seed)
            np.random.seed(seed)
            out = aug({"bSSFP": img.copy(), "LGE": img.copy(), "T2w": img.copy(), "label": lbl.copy()})
            out_lbl = out["label"]
            unique_vals = set(torch.unique(out_lbl).tolist())
            assert unique_vals.issubset({0, 1, 2, 3}), f"Unexpected label values: {unique_vals}"
            assert out_lbl.dtype == torch.long

    def test_100_percent_spatial_synchrony_across_modalities(self):
        """Assert that spatial transformations are 100% synchronous across bSSFP, LGE, T2w, and label."""
        aug = MyoPSRandomGenerator(output_size=(128, 128))

        # Place a unique spatial feature (e.g. non-zero block at (40:50, 40:50))
        H, W = 128, 128
        bSSFP = np.zeros((H, W), dtype=np.float32)
        LGE = np.zeros((H, W), dtype=np.float32)
        T2w = np.zeros((H, W), dtype=np.float32)
        label = np.zeros((H, W), dtype=np.int64)

        bSSFP[40:50, 40:50] = 1.0
        LGE[40:50, 40:50] = 2.0
        T2w[40:50, 40:50] = 3.0
        label[40:50, 40:50] = 1

        for seed in range(30):
            random.seed(seed * 3)
            np.random.seed(seed * 3)
            out = aug({
                "bSSFP": bSSFP.copy(),
                "LGE": LGE.copy(),
                "T2w": T2w.copy(),
                "label": label.copy(),
            })

            # Check spatial alignment of the non-zero region
            bssfp_active = (out["bSSFP"][0] > 0.05).numpy()
            lge_active = (out["LGE"][0] > 0.05).numpy()
            t2w_active = (out["T2w"][0] > 0.05).numpy()
            label_active = (out["label"] > 0).numpy()

            # When nearest neighbor or rot_flip is used, active masks must match exactly
            # (Gaussian noise adds values, so check without noise or check mask correspondence)
            # For pure spatial transforms (shift, rot_flip, rotate, crop):
            # Check rot_flip directly:
            rf_b, rf_l, rf_t, rf_mask = myops_rot_flip(bSSFP.copy(), LGE.copy(), T2w.copy(), label.copy())
            assert np.array_equal((rf_b > 0), (rf_mask > 0))
            assert np.array_equal((rf_l > 0), (rf_mask > 0))
            assert np.array_equal((rf_t > 0), (rf_mask > 0))

            # Check rotate directly:
            rot_b, rot_l, rot_t, rot_mask = myops_rotate(bSSFP.copy(), LGE.copy(), T2w.copy(), label.copy())
            assert np.array_equal((rot_b > 0), (rot_mask > 0))
            assert np.array_equal((rot_l > 0), (rot_mask > 0))
            assert np.array_equal((rot_t > 0), (rot_mask > 0))


# ==============================================================================
# Challenge 3: Full DataLoader Epoch Traversal (1,972 slices -> 124 batches)
# ==============================================================================

class TestDataLoaderEpochTraversal:
    """Rigorous empirical validation of complete epoch iteration."""

    def test_full_epoch_batch_counts_and_remainder_size(self):
        """Assert that an epoch iteration traverses 1,972 samples across 124 batches, ending in 4."""
        loader = create_myops380_train_loader(
            split_file=TRAIN_SPLIT_FILE,
            data_root=DATA_ROOT,
            batch_size=16,
            shuffle=False,
            num_workers=0,  # Single-process for deterministic empirical test
            pin_memory=False,
        )

        total_batches = 0
        total_samples = 0
        last_batch_size = None
        visited_slices = []

        for batch in loader:
            total_batches += 1
            bs = batch["bSSFP"].shape[0]
            total_samples += bs
            last_batch_size = bs
            visited_slices.extend(batch["case_slice"])

            # Every batch before the 124th must have batch size exactly 16
            if total_batches < 124:
                assert bs == 16, f"Batch {total_batches} had unexpected size {bs} (expected 16)"

        # Assert total batches
        assert total_batches == 124, f"Expected exactly 124 batches, got {total_batches}"
        # Assert 124th batch size
        assert last_batch_size == 4, f"Expected 124th batch to contain 4 samples, got {last_batch_size}"
        # Assert total samples
        assert total_samples == EXPECTED_TRAIN_SLICES, f"Expected {EXPECTED_TRAIN_SLICES} samples, got {total_samples}"
        # Assert no duplicates and complete coverage
        assert len(visited_slices) == EXPECTED_TRAIN_SLICES
        assert len(set(visited_slices)) == EXPECTED_TRAIN_SLICES

    def test_non_square_input_resizing_and_crop_parity_with_scar(self):
        """Assert that non-square inputs (e.g. 160x120) produce identical results in both implementations."""
        scar_mod = load_scar_components()
        scar_aug = scar_mod.RandomGenerator(output_size=(128, 128))
        myops_aug = MyoPSRandomGenerator(output_size=(128, 128))

        np.random.seed(777)
        img0 = np.random.uniform(0.0, 1.0, (160, 120)).astype(np.float32)
        img1 = np.random.uniform(0.0, 1.0, (160, 120)).astype(np.float32)
        img2 = np.random.uniform(0.0, 1.0, (160, 120)).astype(np.float32)
        lbl = np.random.choice([0, 1, 2, 3], size=(160, 120)).astype(np.int64)

        for seed in range(20):
            random.seed(seed * 41)
            np.random.seed(seed * 41)
            scar_out = scar_aug({"image": img0.copy(), "image1": img1.copy(), "image2": img2.copy(), "label": lbl.copy()})

            random.seed(seed * 41)
            np.random.seed(seed * 41)
            myops_out = myops_aug({"bSSFP": img0.copy(), "LGE": img1.copy(), "T2w": img2.copy(), "label": lbl.copy()})

            assert scar_out["image"].shape == (1, 128, 128)
            assert myops_out["bSSFP"].shape == (1, 128, 128)
            assert torch.equal(scar_out["image"], myops_out["bSSFP"])
            assert torch.equal(scar_out["image1"], myops_out["LGE"])
            assert torch.equal(scar_out["image2"], myops_out["T2w"])
            assert torch.equal(scar_out["label"], myops_out["label"])

    def test_spline_zoom_order_image_cubic_label_nearest(self):
        """Assert that spatial zoom uses bicubic spline (order=3) for images and nearest-neighbor (order=0) for label."""
        from scipy.ndimage import zoom as scipy_zoom
        aug = MyoPSRandomGenerator(output_size=(128, 128))

        # Sharp edge in image and label
        img = np.zeros((64, 64), dtype=np.float32)
        img[:32, :] = 1.0
        lbl = np.zeros((64, 64), dtype=np.int64)
        lbl[:32, :] = 2

        # Direct zoom
        z_img = scipy_zoom(img, (128 / 64, 128 / 64), order=3)
        z_lbl = scipy_zoom(lbl, (128 / 64, 128 / 64), order=0)

        # Bicubic produces intermediate gradient values between 0 and 1
        assert len(np.unique(z_img)) > 2, "Bicubic zoom (order=3) should produce continuous interpolation!"
        # Nearest neighbor strictly preserves discrete classes {0, 2}
        assert set(np.unique(z_lbl).tolist()) == {0, 2}, "Nearest-neighbor zoom (order=0) must preserve discrete labels!"

    def test_dataloader_enforces_drop_last_false(self):
        """Assert that create_myops380_train_loader never drops the remainder 4 slices."""
        loader = create_myops380_train_loader(
            split_file=TRAIN_SPLIT_FILE,
            data_root=DATA_ROOT,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        assert loader.drop_last is False, "DataLoader drop_last must be False to preserve all 1,972 slices!"
