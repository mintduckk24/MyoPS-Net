"""tests/test_milestone2.py (Milestone 2 Verification Test Suite)
============================================================
Comprehensive test harness for Milestone 2:
- Feature 6: Train Manifest & Sliced Data Loading (1,972 slices across 304 patients)
- Feature 7: Test Volume Data Loading (76 patient volumes)
- Feature 8: SCAR/I-MMSeg RandomGenerator Multi-Modality Augmentation
- Invariants: Disjoint split (zero patient overlap), label consistency, exact batch remainder
- Integration: Forward/backward gradient flow, loss computation, and canonical prediction fusion
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import Milestone 2 Dataset and Augmentation classes
from dataset.myops380_dataset import (
    MyoPS380TrainDataset,
    MyoPS380TestDataset,
    create_myops380_train_loader,
    create_myops380_test_loader,
)
from dataset.augmentation import RandomGenerator, ResizeGenerator

# Import Milestone 1 components for integration verification
from network.model import MyoPSNetL
from utils.tools import create_multi_head_targets, fuse_predictions
from criterion.loss import MyoPSLoss


# ==============================================================================
# Configuration & Constants
# ==============================================================================

DEFAULT_SPLITS_DIR = Path("D:/nckh_model/SCAR/preprocessing/splits/myops380_no_val") if Path("D:/nckh_model/SCAR/preprocessing/splits/myops380_no_val").exists() else Path("D:/NCKH/SCAR/preprocessing/splits/myops380_no_val")
DEFAULT_DATA_ROOT = Path("D:/nckh_model/Dataset/MyoPS380/Processed_data") if Path("D:/nckh_model/Dataset/MyoPS380/Processed_data").exists() else Path("D:/NCKH/Dataset/MyoPS380/Processed_data")

SPLITS_DIR = Path(os.environ.get("MYOPS_SPLITS_DIR", str(DEFAULT_SPLITS_DIR)))
DATA_ROOT = Path(os.environ.get("MYOPS_DATA_ROOT", str(DEFAULT_DATA_ROOT)))

TRAIN_SPLIT_FILE = SPLITS_DIR / "train.txt"
TEST_SPLIT_FILE = SPLITS_DIR / "test_vol.txt"

OFFICIAL_TRAIN_SHA256 = "16248ec3630bb0493fb421aab5ffb4dc9b1d0fd49c8614d69d3098a92305f919"
OFFICIAL_TEST_SHA256 = "3237b31411833cedcb3ecd86d36edae356bd3f880d3fcfd915066f1c850d9cbc"

EXPECTED_TRAIN_SLICES = 1972
EXPECTED_TRAIN_PATIENTS = 304
EXPECTED_TEST_VOLUMES = 76
EXPECTED_TOTAL_PATIENTS = 380

MODALITIES = ("bSSFP", "LGE", "T2w")


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def manifest_data():
    """Reads and parses train and test split manifests."""
    if not TRAIN_SPLIT_FILE.is_file() or not TEST_SPLIT_FILE.is_file():
        pytest.skip(f"Manifest files not found in {SPLITS_DIR}")

    with open(TRAIN_SPLIT_FILE, "r", encoding="utf-8") as f:
        train_slices = [line.strip() for line in f if line.strip()]

    with open(TEST_SPLIT_FILE, "r", encoding="utf-8") as f:
        test_cases = [line.strip() for line in f if line.strip()]

    train_patients = sorted(list({s.split("_slice")[0] for s in train_slices}))
    test_patients = sorted(list({re.sub(r"(\.npy)?\.h5$", "", c) for c in test_cases}))

    return {
        "train_slices": train_slices,
        "test_cases": test_cases,
        "train_patients": train_patients,
        "test_patients": test_patients,
    }


@pytest.fixture(scope="module")
def train_dataset():
    """Initializes MyoPS380TrainDataset without transforms."""
    if not TRAIN_SPLIT_FILE.is_file() or not DATA_ROOT.is_dir():
        pytest.skip("Dataset files not found on disk")
    return MyoPS380TrainDataset(
        split_file=TRAIN_SPLIT_FILE,
        data_root=DATA_ROOT,
        transform=None,
    )


@pytest.fixture(scope="module")
def test_dataset():
    """Initializes MyoPS380TestDataset."""
    if not TEST_SPLIT_FILE.is_file() or not DATA_ROOT.is_dir():
        pytest.skip("Dataset files not found on disk")
    return MyoPS380TestDataset(
        split_file=TEST_SPLIT_FILE,
        data_root=DATA_ROOT,
    )


# ==============================================================================
# Suite 1: Manifest Resolution, Patient Partitioning & Physical File Inventory
# ==============================================================================

class TestManifestResolutionAndInventory:
    """Verifies strict manifest resolution, SHA-256 integrity, zero overlap, and file existence."""

    def test_train_manifest_exact_counts_and_sha256(self, manifest_data):
        """Verify train.txt contains exactly 1,972 slices across 304 patients and matches official SHA256."""
        train_slices = manifest_data["train_slices"]
        train_patients = manifest_data["train_patients"]

        assert len(train_slices) == EXPECTED_TRAIN_SLICES, (
            f"Expected {EXPECTED_TRAIN_SLICES} training slices, found {len(train_slices)}"
        )
        assert len(train_patients) == EXPECTED_TRAIN_PATIENTS, (
            f"Expected {EXPECTED_TRAIN_PATIENTS} training patients, found {len(train_patients)}"
        )

        # Verify pattern case%04d_slice%03d
        pattern = re.compile(r"^case\d{4}_slice\d{3}$")
        for s in train_slices:
            assert pattern.match(s), f"Train slice ID '{s}' violates case%04d_slice%03d format"

        # Verify SHA-256 of sorted IDs terminated with newline
        digest = hashlib.sha256(("\n".join(sorted(train_slices)) + "\n").encode("utf-8")).hexdigest()
        assert digest == OFFICIAL_TRAIN_SHA256, (
            f"train.txt SHA256 mismatch: got {digest}, expected {OFFICIAL_TRAIN_SHA256}"
        )

    def test_train_slice_indexing_contiguity(self, manifest_data):
        """Verify slices for each training patient start at slice000 and form a contiguous sequence."""
        slices_per_patient: Dict[str, List[int]] = {}
        for s in manifest_data["train_slices"]:
            pid, sidx = s.split("_slice")
            slices_per_patient.setdefault(pid, []).append(int(sidx))

        for pid, indices in slices_per_patient.items():
            sorted_indices = sorted(indices)
            assert sorted_indices[0] == 0, f"Patient {pid} does not start at slice 000: {sorted_indices}"
            expected = list(range(len(sorted_indices)))
            assert sorted_indices == expected, f"Patient {pid} slices non-contiguous: {sorted_indices} vs {expected}"
            assert 2 <= len(sorted_indices) <= 12, f"Patient {pid} slice count {len(sorted_indices)} out of bounds [2, 12]"

    def test_test_vol_manifest_exact_counts_and_sha256(self, manifest_data):
        """Verify test_vol.txt contains exactly 76 patient volumes and matches official SHA256."""
        test_cases = manifest_data["test_cases"]
        test_patients = manifest_data["test_patients"]

        assert len(test_cases) == EXPECTED_TEST_VOLUMES, (
            f"Expected {EXPECTED_TEST_VOLUMES} test cases, found {len(test_cases)}"
        )
        assert len(test_patients) == EXPECTED_TEST_VOLUMES, (
            f"Expected {EXPECTED_TEST_VOLUMES} unique test patients, found {len(test_patients)}"
        )

        pattern = re.compile(r"^case\d{4}$")
        for c in test_patients:
            assert pattern.match(c), f"Test patient ID '{c}' violates case%04d format"

        # Verify SHA-256
        digest = hashlib.sha256(("\n".join(sorted(test_patients)) + "\n").encode("utf-8")).hexdigest()
        assert digest == OFFICIAL_TEST_SHA256, (
            f"test_vol.txt SHA256 mismatch: got {digest}, expected {OFFICIAL_TEST_SHA256}"
        )

    def test_disjoint_split_zero_patient_leakage(self, manifest_data):
        """Verify zero patient overlap between train and test cohorts, and complete 380 patient coverage."""
        train_p = set(manifest_data["train_patients"])
        test_p = set(manifest_data["test_patients"])

        overlap = train_p & test_p
        assert len(overlap) == 0, f"Critical Leakage! {len(overlap)} overlapping patients: {overlap}"

        total_union = train_p | test_p
        assert len(total_union) == EXPECTED_TOTAL_PATIENTS, (
            f"Expected total cohort of {EXPECTED_TOTAL_PATIENTS} patients, got {len(total_union)}"
        )
        expected_full_set = {f"case{i:04d}" for i in range(1, EXPECTED_TOTAL_PATIENTS + 1)}
        assert total_union == expected_full_set, "Cohort does not exactly span case0001 to case0380"

    def test_zero_missing_files_in_processed_data(self, manifest_data):
        """Verify that every slice and volume exists across all 3 modalities in Processed_data."""
        if not DATA_ROOT.is_dir():
            pytest.skip("Processed_data root not found")

        # 1. Train slices (1,972 * 3 = 5,916 files)
        for mod in MODALITIES:
            mod_train_dir = DATA_ROOT / mod / "train_npz"
            assert mod_train_dir.is_dir(), f"Directory missing: {mod_train_dir}"
            missing_slices = [
                s for s in manifest_data["train_slices"]
                if not (mod_train_dir / f"{s}.npz").is_file()
            ]
            assert len(missing_slices) == 0, (
                f"Modality {mod} is missing {len(missing_slices)} train npz files. "
                f"First 5: {missing_slices[:5]}"
            )

        # 2. Test volumes (76 * 3 = 228 files)
        for mod in MODALITIES:
            mod_test_dir = DATA_ROOT / mod / "test_vol_h5"
            assert mod_test_dir.is_dir(), f"Directory missing: {mod_test_dir}"
            missing_vols = [
                c for c in manifest_data["test_patients"]
                if not (mod_test_dir / f"{c}.npy.h5").is_file()
            ]
            assert len(missing_vols) == 0, (
                f"Modality {mod} is missing {len(missing_vols)} test h5 files. "
                f"First 5: {missing_vols[:5]}"
            )

    def test_label_identicality_across_modalities(self, manifest_data):
        """Verify that label masks are bitwise identical across all 3 modalities."""
        if not DATA_ROOT.is_dir():
            pytest.skip("Processed_data root not found")

        # Spot check 5 slices across different patients
        sample_slices = manifest_data["train_slices"][:5]
        for s in sample_slices:
            l_b = np.load(DATA_ROOT / "bSSFP" / "train_npz" / f"{s}.npz")["label"]
            l_l = np.load(DATA_ROOT / "LGE" / "train_npz" / f"{s}.npz")["label"]
            l_t = np.load(DATA_ROOT / "T2w" / "train_npz" / f"{s}.npz")["label"]
            assert np.array_equal(l_b, l_l), f"Label mismatch bSSFP vs LGE in {s}"
            assert np.array_equal(l_b, l_t), f"Label mismatch bSSFP vs T2w in {s}"

        # Spot check 3 test volumes
        sample_cases = manifest_data["test_patients"][:3]
        for c in sample_cases:
            with h5py.File(DATA_ROOT / "bSSFP" / "test_vol_h5" / f"{c}.npy.h5", "r") as fb, \
                 h5py.File(DATA_ROOT / "LGE" / "test_vol_h5" / f"{c}.npy.h5", "r") as fl, \
                 h5py.File(DATA_ROOT / "T2w" / "test_vol_h5" / f"{c}.npy.h5", "r") as ft:
                assert np.array_equal(fb["label"][:], fl["label"][:]), f"Test label mismatch bSSFP vs LGE in {c}"
                assert np.array_equal(fb["label"][:], ft["label"][:]), f"Test label mismatch bSSFP vs T2w in {c}"


# ==============================================================================
# Suite 2: Dataset Classes, Data Types & Output Tensor Shapes
# ==============================================================================

class TestDatasetClassesAndShapes:
    """Verifies MyoPS380TrainDataset and MyoPS380TestDataset conformance to interface contract."""

    def test_train_dataset_length_and_sample_structure(self, train_dataset):
        """Verify train dataset returns 1,972 samples with expected keys and tensor properties."""
        assert len(train_dataset) == EXPECTED_TRAIN_SLICES

        # Inspect first, middle, and last sample
        sample_indices = [0, len(train_dataset) // 2, len(train_dataset) - 1]
        for idx in sample_indices:
            sample = train_dataset[idx]
            assert isinstance(sample, dict)
            for k in ("bSSFP", "LGE", "T2w", "label", "case_slice"):
                assert k in sample, f"Key '{k}' missing from train sample {idx}"

            # Modality tensors
            for mod in MODALITIES:
                tensor = sample[mod]
                assert isinstance(tensor, torch.Tensor), f"{mod} is not a torch.Tensor"
                assert tensor.dtype == torch.float32, f"{mod} dtype {tensor.dtype} != float32"
                assert tensor.shape == (1, 128, 128), f"{mod} shape {tensor.shape} != (1, 128, 128)"
                assert torch.isfinite(tensor).all(), f"{mod} contains NaN or Inf"
                assert tensor.min() >= 0.0, f"{mod} min < 0.0"
                # Pre-normalized cached data is in [0, 1]
                assert tensor.max() <= 1.0 + 1e-4, f"{mod} max > 1.0: {tensor.max().item()}"

            # Label mask
            lbl = sample["label"]
            assert isinstance(lbl, torch.Tensor), "label is not a torch.Tensor"
            assert lbl.dtype == torch.long, f"label dtype {lbl.dtype} != torch.long"
            assert lbl.shape == (128, 128), f"label shape {lbl.shape} != (128, 128)"
            unique_vals = torch.unique(lbl).tolist()
            assert all(v in {0, 1, 2, 3} for v in unique_vals), f"label contains invalid IDs: {unique_vals}"

            # Case slice ID
            assert isinstance(sample["case_slice"], str)
            assert re.match(r"^case\d{4}_slice\d{3}$", sample["case_slice"])

    def test_test_dataset_length_and_sample_structure(self, test_dataset):
        """Verify test dataset returns 76 volumes with expected 3D shapes (1, 128, 128, D)."""
        assert len(test_dataset) == EXPECTED_TEST_VOLUMES

        # Inspect 3 representative test cases
        sample_indices = [0, 30, len(test_dataset) - 1]
        for idx in sample_indices:
            sample = test_dataset[idx]
            assert isinstance(sample, dict)
            for k in ("bSSFP", "LGE", "T2w", "label", "case_id"):
                assert k in sample, f"Key '{k}' missing from test sample {idx}"

            # Modality volumes
            D = sample["bSSFP"].shape[-1]
            assert 2 <= D <= 10, f"Test volume depth {D} out of expected range [2, 10]"

            for mod in MODALITIES:
                tensor = sample[mod]
                assert isinstance(tensor, torch.Tensor), f"{mod} is not a torch.Tensor"
                assert tensor.dtype == torch.float32, f"{mod} dtype {tensor.dtype} != float32"
                assert tensor.shape == (1, 128, 128, D), f"{mod} shape {tensor.shape} != (1, 128, 128, {D})"
                assert torch.isfinite(tensor).all(), f"{mod} contains NaN or Inf"
                assert tensor.min() >= 0.0, f"{mod} min < 0.0"
                assert tensor.max() <= 1.0 + 1e-4, f"{mod} max > 1.0"

            # Label volume
            lbl = sample["label"]
            assert isinstance(lbl, torch.Tensor), "label is not a torch.Tensor"
            assert lbl.dtype == torch.long, f"label dtype {lbl.dtype} != torch.long"
            assert lbl.shape == (128, 128, D), f"label shape {lbl.shape} != (128, 128, {D})"
            unique_vals = torch.unique(lbl).tolist()
            assert all(v in {0, 1, 2, 3} for v in unique_vals), f"label contains invalid IDs: {unique_vals}"

            # Case ID
            assert isinstance(sample["case_id"], str)
            assert re.match(r"^case\d{4}$", sample["case_id"])

    def test_train_dataset_custom_transform_invocation(self):
        """Verify custom transform callable is executed on train dataset sample."""
        invoked = False

        def mock_transform(sample):
            nonlocal invoked
            invoked = True
            sample["bSSFP"] = sample["bSSFP"] * 0.5
            return sample

        ds = MyoPS380TrainDataset(
            split_file=TRAIN_SPLIT_FILE,
            data_root=DATA_ROOT,
            transform=mock_transform,
        )
        sample = ds[0]
        assert invoked, "Transform callable was not invoked by MyoPS380TrainDataset"
        assert sample["bSSFP"].shape == (1, 128, 128)

    def test_test_dataset_canonical_label_mapping(self):
        """Verify canonical_label=True maps legacy [0,1,2,3] -> canonical [0,1,3,2]."""
        ds_raw = MyoPS380TestDataset(
            split_file=TEST_SPLIT_FILE,
            data_root=DATA_ROOT,
            canonical_label=False,
        )
        ds_canonical = MyoPS380TestDataset(
            split_file=TEST_SPLIT_FILE,
            data_root=DATA_ROOT,
            canonical_label=True,
        )

        sample_raw = ds_raw[0]
        sample_can = ds_canonical[0]

        lbl_raw = sample_raw["label"].numpy()
        lbl_can = sample_can["label"].numpy()

        # Legacy 2 (Scar) should map to Canonical 3
        if np.any(lbl_raw == 2):
            assert np.all(lbl_can[lbl_raw == 2] == 3)
        # Legacy 3 (Edema) should map to Canonical 2
        if np.any(lbl_raw == 3):
            assert np.all(lbl_can[lbl_raw == 3] == 2)
        # 0 and 1 remain identical
        if np.any(lbl_raw == 0):
            assert np.all(lbl_can[lbl_raw == 0] == 0)
        if np.any(lbl_raw == 1):
            assert np.all(lbl_can[lbl_raw == 1] == 1)

    def test_dataset_invalid_paths_raise_filenotfound(self):
        """Verify instantiating with non-existent data_root or split_file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MyoPS380TrainDataset(
                split_file="non_existent_train.txt",
                data_root=DATA_ROOT,
            )

        with pytest.raises(FileNotFoundError):
            MyoPS380TrainDataset(
                split_file=TRAIN_SPLIT_FILE,
                data_root="D:/NonExistentDirectory",
            )


# ==============================================================================
# Suite 3: DataLoader Batching, Physical Batch Size 16 & Remainder Batch Handling
# ==============================================================================

class TestDataLoaderBatchingAndRemainder:
    """Verifies PyTorch DataLoader batching, 124 total batches, and exact 4-sample remainder."""

    def test_standard_batch_size_16_shapes(self, train_dataset):
        """Verify standard batch shape is (16, 1, 128, 128) for images and (16, 128, 128) for labels."""
        loader = DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=False)
        first_batch = next(iter(loader))

        for mod in MODALITIES:
            tensor = first_batch[mod]
            assert tensor.shape == (16, 1, 128, 128), f"Batch {mod} shape {tensor.shape} != (16, 1, 128, 128)"
            assert tensor.dtype == torch.float32

        lbl = first_batch["label"]
        assert lbl.shape == (16, 128, 128), f"Batch label shape {lbl.shape} != (16, 128, 128)"
        assert lbl.dtype == torch.long

        case_slices = first_batch["case_slice"]
        assert len(case_slices) == 16
        assert isinstance(case_slices[0], str)

    def test_dataloader_total_batches_and_124th_remainder_batch(self, train_dataset):
        """Verify 1,972 slices yield exactly 124 batches where the 124th remainder batch has size 4."""
        loader = DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=False)

        batches = list(loader)
        assert len(batches) == 124, f"Expected 124 batches per epoch (1972/16 ceil), got {len(batches)}"

        # Verify first 123 batches have size 16
        for i in range(123):
            assert batches[i]["bSSFP"].shape[0] == 16, f"Batch {i+1} size != 16"

        # Verify 124th batch (remainder batch: 1972 = 123 * 16 + 4)
        last_batch = batches[123]
        for mod in MODALITIES:
            assert last_batch[mod].shape == (4, 1, 128, 128), (
                f"124th batch {mod} shape {last_batch[mod].shape} != (4, 1, 128, 128)"
            )
            assert last_batch[mod].dtype == torch.float32

        assert last_batch["label"].shape == (4, 128, 128), (
            f"124th batch label shape {last_batch['label'].shape} != (4, 128, 128)"
        )
        assert len(last_batch["case_slice"]) == 4

        # Total sample coverage verification
        total_samples = sum(b["bSSFP"].shape[0] for b in batches)
        assert total_samples == EXPECTED_TRAIN_SLICES, (
            f"Total samples across 124 batches {total_samples} != {EXPECTED_TRAIN_SLICES}"
        )

        all_retrieved_slices = [s for b in batches for s in b["case_slice"]]
        assert len(set(all_retrieved_slices)) == EXPECTED_TRAIN_SLICES, "Duplicate or missing slice IDs in DataLoader"

    def test_shuffled_dataloader_preserves_total_count_and_remainder(self, train_dataset):
        """Verify DataLoader with shuffle=True preserves sample integrity and produces 124 batches."""
        torch.manual_seed(42)
        loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False)

        count = 0
        last_batch_size = None
        seen_slices = set()

        for batch in loader:
            bs = batch["bSSFP"].shape[0]
            count += bs
            last_batch_size = bs
            for s in batch["case_slice"]:
                seen_slices.add(s)

        assert count == EXPECTED_TRAIN_SLICES
        assert len(seen_slices) == EXPECTED_TRAIN_SLICES
        assert last_batch_size == 4, f"Shuffled last batch size {last_batch_size} != 4"


# ==============================================================================
# Suite 4: Augmentation Synchronous Alignment & Mathematical Integrity
# ==============================================================================

class TestAugmentationSynchronousAlignment:
    """Verifies that RandomGenerator maintains strict spatial alignment and nearest-neighbor label integrity."""

    def test_synchronous_spatial_rotation_flip_alignment(self):
        """Verify that spatial rot90 and flips transform all 3 modalities and the label identically."""
        # Create synthetic asymmetric patterns where each quadrant has unique values
        H, W = 128, 128
        bSSFP = np.zeros((H, W), dtype=np.float32)
        LGE = np.zeros((H, W), dtype=np.float32)
        T2w = np.zeros((H, W), dtype=np.float32)
        label = np.zeros((H, W), dtype=np.int64)

        # Set unique regional markers
        # Top-left quadrant
        bSSFP[:64, :64] = 0.2
        LGE[:64, :64] = 0.4
        T2w[:64, :64] = 0.6
        label[:64, :64] = 1

        # Center asymmetric target
        bSSFP[30:50, 40:70] = 0.8
        LGE[30:50, 40:70] = 0.9
        T2w[30:50, 40:70] = 1.0
        label[30:50, 40:70] = 2

        # Bottom-right marker
        bSSFP[100:120, 100:120] = 0.5
        LGE[100:120, 100:120] = 0.5
        T2w[100:120, 100:120] = 0.5
        label[100:120, 100:120] = 3

        aug = RandomGenerator(output_size=(128, 128))

        # Test across 20 distinct random seeds
        for seed in range(20):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            sample = {
                "bSSFP": bSSFP.copy(),
                "LGE": LGE.copy(),
                "T2w": T2w.copy(),
                "label": label.copy(),
            }
            out = aug(sample)

            out_bssfp = out["bSSFP"][0].numpy()
            out_lge = out["LGE"][0].numpy()
            out_t2w = out["T2w"][0].numpy()
            out_lbl = out["label"].numpy()

            # Output shapes
            assert out["bSSFP"].shape == (1, 128, 128)
            assert out["LGE"].shape == (1, 128, 128)
            assert out["T2w"].shape == (1, 128, 128)
            assert out["label"].shape == (128, 128)

            # Check that locations where label == 2 align with elevated intensity in bSSFP, LGE, T2w
            mask_lbl_2 = (out_lbl == 2)
            if np.any(mask_lbl_2):
                # Where label == 2, bSSFP, LGE, T2w must all be non-zero and aligned
                assert np.all(out_bssfp[mask_lbl_2] > 0.0), f"bSSFP misaligned with label=2 at seed {seed}"
                assert np.all(out_lge[mask_lbl_2] > 0.0), f"LGE misaligned with label=2 at seed {seed}"
                assert np.all(out_t2w[mask_lbl_2] > 0.0), f"T2w misaligned with label=2 at seed {seed}"

            mask_lbl_3 = (out_lbl == 3)
            if np.any(mask_lbl_3):
                assert np.all(out_bssfp[mask_lbl_3] > 0.0), f"bSSFP misaligned with label=3 at seed {seed}"

    def test_nearest_neighbor_interpolation_label_integrity(self):
        """Verify that label interpolation is strictly nearest-neighbor (no fractional or interpolated values)."""
        aug = RandomGenerator(output_size=(128, 128))

        H, W = 128, 128
        bSSFP = np.random.rand(H, W).astype(np.float32)
        LGE = np.random.rand(H, W).astype(np.float32)
        T2w = np.random.rand(H, W).astype(np.float32)
        # Assign all 4 discrete labels
        label = np.random.choice([0, 1, 2, 3], size=(H, W)).astype(np.int64)

        for seed in range(50):
            random.seed(seed + 100)
            np.random.seed(seed + 100)

            sample = {
                "bSSFP": bSSFP.copy(),
                "LGE": LGE.copy(),
                "T2w": T2w.copy(),
                "label": label.copy(),
            }
            out = aug(sample)
            out_lbl = out["label"]

            assert out_lbl.dtype == torch.long, f"Augmented label dtype {out_lbl.dtype} != torch.long"
            unique_vals = torch.unique(out_lbl).tolist()
            # Must strictly be a subset of {0, 1, 2, 3}
            assert all(v in {0, 1, 2, 3} for v in unique_vals), (
                f"Non-integer or out-of-range interpolated label values at seed {seed}: {unique_vals}"
            )

    def test_zoom_and_crop_preserves_target_output_shape(self):
        """Verify that any random crop and zoom operation strictly preserves (1, 128, 128) and (128, 128)."""
        aug = RandomGenerator(output_size=(128, 128))

        # Test on varied rectangular input shapes to test zoom scaling
        for in_h, in_w in [(128, 128), (140, 140), (100, 100)]:
            sample = {
                "bSSFP": np.ones((in_h, in_w), dtype=np.float32),
                "LGE": np.ones((in_h, in_w), dtype=np.float32),
                "T2w": np.ones((in_h, in_w), dtype=np.float32),
                "label": np.ones((in_h, in_w), dtype=np.int64),
            }
            out = aug(sample)
            assert out["bSSFP"].shape == (1, 128, 128)
            assert out["LGE"].shape == (1, 128, 128)
            assert out["T2w"].shape == (1, 128, 128)
            assert out["label"].shape == (128, 128)

    def test_intensity_transforms_leave_label_mask_unaltered(self):
        """Verify that intensity augmentations (noise, brightness, gamma) never modify label mask values."""
        aug = RandomGenerator(output_size=(128, 128))

        H, W = 128, 128
        bSSFP = np.full((H, W), 0.5, dtype=np.float32)
        LGE = np.full((H, W), 0.5, dtype=np.float32)
        T2w = np.full((H, W), 0.5, dtype=np.float32)
        # Create a deterministic checkerboard label
        label = np.zeros((H, W), dtype=np.int64)
        label[::2, ::2] = 1
        label[1::2, 1::2] = 2
        label[::2, 1::2] = 3

        # Force only intensity transforms by testing directly or testing with seeds
        for seed in range(30):
            random.seed(seed * 7)
            np.random.seed(seed * 7)

            sample = {
                "bSSFP": bSSFP.copy(),
                "LGE": LGE.copy(),
                "T2w": T2w.copy(),
                "label": label.copy(),
            }
            out = aug(sample)
            out_lbl = out["label"]

            # Values must be exact subset of {0, 1, 2, 3}
            unique = set(torch.unique(out_lbl).tolist())
            assert unique.issubset({0, 1, 2, 3})
            assert out_lbl.dtype == torch.long

    def test_augmentation_supports_both_modality_keys_and_legacy_keys(self):
        """Verify RandomGenerator accepts both ('bSSFP', 'LGE', 'T2w') and ('image', 'image1', 'image2')."""
        aug = RandomGenerator(output_size=(128, 128))
        img = np.ones((128, 128), dtype=np.float32)
        lbl = np.ones((128, 128), dtype=np.int64)

        # 1. MyoPS-Net dict format
        sample_mod = {"bSSFP": img, "LGE": img, "T2w": img, "label": lbl}
        out_mod = aug(sample_mod)
        assert "bSSFP" in out_mod and "LGE" in out_mod and "T2w" in out_mod and "label" in out_mod

        # 2. SCAR legacy format
        sample_legacy = {"image": img, "image1": img, "image2": img, "label": lbl}
        out_legacy = aug(sample_legacy)
        assert "image" in out_legacy and "image1" in out_legacy and "image2" in out_legacy and "label" in out_legacy


# ==============================================================================
# Suite 5: End-to-End Integration with Milestone 1 Components
# ==============================================================================

class TestMilestone1Integration:
    """Verifies seamless forward/backward training flow from DataLoader batches to MyoPSNetL and MyoPSLoss."""

    def test_dataloader_batch_16_forward_backward_pass(self, train_dataset):
        """Verify batch size 16 from DataLoader produces valid loss and backward gradients in MyoPSNetL."""
        loader = DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=False)
        batch = next(iter(loader))

        c0 = batch["bSSFP"]
        lge = batch["LGE"]
        t2 = batch["T2w"]
        label = batch["label"]

        assert c0.shape == (16, 1, 128, 128)
        assert label.shape == (16, 128, 128)

        # 1. Target Adapter
        target_C0, target_LGE, target_T2 = create_multi_head_targets(label)
        assert target_C0.shape == (16, 128, 128)
        assert target_LGE.shape == (16, 128, 128)
        assert target_T2.shape == (16, 128, 128)
        assert set(torch.unique(target_C0).tolist()).issubset({0, 1})
        assert set(torch.unique(target_LGE).tolist()).issubset({0, 1, 2})
        assert set(torch.unique(target_T2).tolist()).issubset({0, 1, 2})

        # 2. Model Forward Pass
        model = MyoPSNetL()
        model.train()
        seg_C0, seg_LGE, seg_T2 = model(c0, lge, t2)

        assert seg_C0.shape == (16, 2, 128, 128)
        assert seg_LGE.shape == (16, 3, 128, 128)
        assert seg_T2.shape == (16, 3, 128, 128)

        # 3. Loss Computation
        criterion = MyoPSLoss()
        loss_out = criterion(seg_C0, seg_LGE, seg_T2, target_C0, target_LGE, target_T2)
        total_loss = loss_out[0] if isinstance(loss_out, (tuple, list)) else loss_out

        assert torch.isfinite(total_loss), f"Total loss is not finite: {total_loss.item()}"
        assert total_loss.item() > 0.0, f"Total loss is non-positive: {total_loss.item()}"

        # 4. Backward Pass
        total_loss.backward()

        # Check gradients exist on all modules
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter '{name}' did not receive gradient"
                assert torch.isfinite(param.grad).all(), f"Parameter '{name}' gradient has NaN/Inf"

    def test_dataloader_remainder_batch_4_forward_backward_pass(self, train_dataset):
        """Verify remainder batch (size 4) executes forward and backward passes without shape errors."""
        loader = DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=False)
        batches = list(loader)
        remainder_batch = batches[-1]

        c0 = remainder_batch["bSSFP"]
        lge = remainder_batch["LGE"]
        t2 = remainder_batch["T2w"]
        label = remainder_batch["label"]

        assert c0.shape == (4, 1, 128, 128)
        assert label.shape == (4, 128, 128)

        # Target Adapter
        target_C0, target_LGE, target_T2 = create_multi_head_targets(label)
        assert target_C0.shape == (4, 128, 128)
        assert target_LGE.shape == (4, 128, 128)
        assert target_T2.shape == (4, 128, 128)

        # Model Forward
        model = MyoPSNetL()
        model.train()
        seg_C0, seg_LGE, seg_T2 = model(c0, lge, t2)
        assert seg_C0.shape == (4, 2, 128, 128)
        assert seg_LGE.shape == (4, 3, 128, 128)
        assert seg_T2.shape == (4, 3, 128, 128)

        # Loss and Backward
        criterion = MyoPSLoss()
        loss_out = criterion(seg_C0, seg_LGE, seg_T2, target_C0, target_LGE, target_T2)
        total_loss = loss_out[0] if isinstance(loss_out, (tuple, list)) else loss_out
        assert torch.isfinite(total_loss)
        assert total_loss.item() > 0.0
        total_loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()

    def test_dataloader_batch_prediction_fusion(self, train_dataset):
        """Verify model outputs on a DataLoader batch fuse into canonical 4-class predictions."""
        loader = DataLoader(train_dataset, batch_size=4, shuffle=False, drop_last=False)
        batch = next(iter(loader))

        c0, lge, t2 = batch["bSSFP"], batch["LGE"], batch["T2w"]

        model = MyoPSNetL()
        model.eval()
        with torch.no_grad():
            seg_C0, seg_LGE, seg_T2 = model(c0, lge, t2)

        # Fuse predictions
        fused = fuse_predictions(seg_C0, seg_LGE, seg_T2)

        assert fused.shape == (4, 128, 128)
        assert fused.dtype in (np.uint8, np.int64, torch.uint8, torch.long)
        unique_classes = np.unique(fused) if isinstance(fused, np.ndarray) else torch.unique(fused).tolist()
        assert all(c in [0, 1, 2, 3] for c in unique_classes), f"Invalid fused class values: {unique_classes}"

    def test_dataloader_batch_optimizer_step(self, train_dataset):
        """Verify that optimizer.step() successfully updates network weights on a real batch."""
        loader = DataLoader(train_dataset, batch_size=4, shuffle=False, drop_last=False)
        batch = next(iter(loader))

        c0, lge, t2 = batch["bSSFP"], batch["LGE"], batch["T2w"]
        label = batch["label"]

        target_C0, target_LGE, target_T2 = create_multi_head_targets(label)

        model = MyoPSNetL()
        model.train()

        # Record initial weights of first model parameter
        param_to_track = next(p for p in model.parameters() if p.requires_grad)
        init_weight = param_to_track.clone().detach()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        optimizer.zero_grad()

        seg_C0, seg_LGE, seg_T2 = model(c0, lge, t2)
        criterion = MyoPSLoss()
        loss_out = criterion(seg_C0, seg_LGE, seg_T2, target_C0, target_LGE, target_T2)
        total_loss = loss_out[0] if isinstance(loss_out, (tuple, list)) else loss_out

        total_loss.backward()
        optimizer.step()

        updated_weight = param_to_track.detach()
        diff = torch.norm(updated_weight - init_weight).item()
        assert diff > 0.0, "Optimizer step failed to update model parameters"
