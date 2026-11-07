"""MyoPS380 Dataset Classes for Standardized Benchmark Protocol.
============================================================
Protocol: myops380_train304_test76_last_v1
Interface contracts:
  - MyoPS380TrainDataset:
      Input: split_file (train.txt), data_root (Processed_data), transform=None
      Yields: dict with keys:
        'bSSFP': torch.FloatTensor (1, 128, 128)
        'LGE':   torch.FloatTensor (1, 128, 128)
        'T2w':   torch.FloatTensor (1, 128, 128)
        'label': torch.LongTensor (128, 128) with values in {0, 1, 2, 3}
        'case_slice': str ID, e.g. 'case0002_slice000'
  - MyoPS380TestDataset:
      Input: split_file (test_vol.txt), data_root (Processed_data)
      Yields: dict with keys:
        'bSSFP': torch.FloatTensor (1, 128, 128, D)
        'LGE':   torch.FloatTensor (1, 128, 128, D)
        'T2w':   torch.FloatTensor (1, 128, 128, D)
        'label': torch.LongTensor (128, 128, D) with values in {0, 1, 2, 3}
        'case_id': str ID, e.g. 'case0001'
"""

from __future__ import annotations

import os
from pathlib import Path
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


EXPECTED_TRAIN_SLICES = 1972
EXPECTED_TRAIN_PATIENTS = 304
EXPECTED_TEST_VOLUMES = 76
CANONICAL_LUT = np.array([0, 1, 3, 2], dtype=np.int64)


def seed_worker(worker_id: int) -> None:
    """Use SCAR's Python/NumPy worker seeding policy (also picklable on Windows)."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def resolve_split_file(split_file: Union[str, Path], default_filename: str) -> Path:
    """Resolves split file path whether a directory or file path is passed."""
    path = Path(split_file)
    if path.is_dir():
        path = path / default_filename
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    return path


def extract_patient_id(case_slice_or_vol: str) -> str:
    """Extracts normalized patient ID from slice name or volume filename."""
    clean = re.sub(r"(\.npy)?\.h5$", "", str(case_slice_or_vol).strip())
    return clean.split("_slice")[0]


def read_manifest_lines(file_path: Union[str, Path]) -> List[str]:
    """Reads non-empty, stripped lines from a manifest file."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def _to_image_tensor(img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Ensures 2D slice image is a torch.FloatTensor of shape (1, 128, 128)."""
    if isinstance(img, torch.Tensor):
        t = img.float()
    else:
        t = torch.from_numpy(np.asarray(img, dtype=np.float32))
    if t.dim() == 2:
        t = t.unsqueeze(0)
    elif t.dim() == 3 and t.shape[0] != 1 and t.shape[-1] == 1:
        t = t.permute(2, 0, 1)
    return t


def _to_label_tensor(lbl: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Ensures 2D slice label is a torch.LongTensor of shape (128, 128)."""
    if isinstance(lbl, torch.Tensor):
        t = lbl.long()
    else:
        t = torch.from_numpy(np.asarray(lbl, dtype=np.int64))
    if t.dim() == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    elif t.dim() == 3 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    return t


def _to_volume_image_tensor(img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Ensures 3D volume image is a torch.FloatTensor of shape (1, 128, 128, D)."""
    if isinstance(img, torch.Tensor):
        t = img.float()
    else:
        t = torch.from_numpy(np.asarray(img, dtype=np.float32))
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t


def _to_volume_label_tensor(lbl: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Ensures 3D volume label is a torch.LongTensor of shape (128, 128, D)."""
    if isinstance(lbl, torch.Tensor):
        t = lbl.long()
    else:
        t = torch.from_numpy(np.asarray(lbl, dtype=np.int64))
    if t.dim() == 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t


class MyoPS380TrainDataset(Dataset):
    """Slice-level training dataset for MyoPS380 benchmark.
    
    Loads aligned bSSFP, LGE, and T2w 2D image slices along with ground-truth
    segmentation masks across 1,972 slices from 304 patients specified in train.txt.
    """

    def __init__(
        self,
        split_file: Union[str, Path],
        data_root: Union[str, Path],
        transform: Optional[Callable] = None,
        validate_files: bool = False,
        strict: bool = False,
    ):
        self.split_file = resolve_split_file(split_file, "train.txt")
        self.data_root = Path(data_root)
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"data_root directory not found: {self.data_root}")

        self.transform = transform
        self.slices: List[str] = read_manifest_lines(self.split_file)

        if not self.slices:
            raise ValueError(f"Train manifest is empty: {self.split_file}")

        self.patients = sorted(list({extract_patient_id(s) for s in self.slices}))

        if strict:
            if len(self.slices) != EXPECTED_TRAIN_SLICES:
                raise ValueError(
                    f"Expected exactly {EXPECTED_TRAIN_SLICES} training slices, "
                    f"got {len(self.slices)} in {self.split_file}"
                )
            if len(self.patients) != EXPECTED_TRAIN_PATIENTS:
                raise ValueError(
                    f"Expected exactly {EXPECTED_TRAIN_PATIENTS} unique training patients, "
                    f"got {len(self.patients)} in {self.split_file}"
                )

        if validate_files:
            self._validate_all_files()

    def _validate_all_files(self) -> None:
        missing = []
        for modality in ("bSSFP", "LGE", "T2w"):
            mod_dir = self.data_root / modality / "train_npz"
            for slice_id in self.slices:
                npz_path = mod_dir / f"{slice_id}.npz"
                if not npz_path.is_file():
                    missing.append(str(npz_path))
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} train slice files; first 5: {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        slice_id = self.slices[idx]

        bssfp_path = self.data_root / "bSSFP" / "train_npz" / f"{slice_id}.npz"
        lge_path = self.data_root / "LGE" / "train_npz" / f"{slice_id}.npz"
        t2w_path = self.data_root / "T2w" / "train_npz" / f"{slice_id}.npz"

        with np.load(bssfp_path) as d_bssfp:
            bssfp = np.asarray(d_bssfp["image"], dtype=np.float32)
            label = np.asarray(d_bssfp["label"])

        with np.load(lge_path) as d_lge:
            lge = np.asarray(d_lge["image"], dtype=np.float32)

        with np.load(t2w_path) as d_t2w:
            t2w = np.asarray(d_t2w["image"], dtype=np.float32)

        if self.transform is not None:
            sample_in = {
                "bSSFP": bssfp,
                "LGE": lge,
                "T2w": t2w,
                "image": bssfp,
                "image1": lge,
                "image2": t2w,
                "label": label,
                "case_slice": slice_id,
            }
            sample_out = self.transform(sample_in)

            if isinstance(sample_out, dict):
                if "bSSFP" in sample_out and (
                    isinstance(sample_out["bSSFP"], torch.Tensor)
                    or sample_out["bSSFP"] is not bssfp
                ):
                    bssfp = sample_out["bSSFP"]
                elif "image" in sample_out:
                    bssfp = sample_out["image"]
                else:
                    bssfp = sample_out.get("bSSFP", bssfp)

                if "LGE" in sample_out and (
                    isinstance(sample_out["LGE"], torch.Tensor)
                    or sample_out["LGE"] is not lge
                ):
                    lge = sample_out["LGE"]
                elif "image1" in sample_out:
                    lge = sample_out["image1"]
                else:
                    lge = sample_out.get("LGE", lge)

                if "T2w" in sample_out and (
                    isinstance(sample_out["T2w"], torch.Tensor)
                    or sample_out["T2w"] is not t2w
                ):
                    t2w = sample_out["T2w"]
                elif "image2" in sample_out:
                    t2w = sample_out["image2"]
                else:
                    t2w = sample_out.get("T2w", t2w)

                label = sample_out.get("label", label)
            elif isinstance(sample_out, (list, tuple)) and len(sample_out) >= 4:
                bssfp, lge, t2w, label = sample_out[0], sample_out[1], sample_out[2], sample_out[3]

        return {
            "bSSFP": _to_image_tensor(bssfp),
            "LGE": _to_image_tensor(lge),
            "T2w": _to_image_tensor(t2w),
            "label": _to_label_tensor(label),
            "case_slice": slice_id,
        }


class MyoPS380TestDataset(Dataset):
    """Volume-level test evaluation dataset for MyoPS380 benchmark.
    
    Loads aligned 3D volumes (128, 128, D) for bSSFP, LGE, and T2w modalities,
    along with 3D ground truth segmentation masks across the 76 official test cases.
    """

    def __init__(
        self,
        split_file: Union[str, Path],
        data_root: Union[str, Path],
        canonical_label: bool = False,
        validate_files: bool = False,
        strict: bool = False,
    ):
        self.split_file = resolve_split_file(split_file, "test_vol.txt")
        self.data_root = Path(data_root)
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"data_root directory not found: {self.data_root}")

        self.canonical_label = canonical_label
        raw_cases = read_manifest_lines(self.split_file)
        self.cases: List[str] = [re.sub(r"(\.npy)?\.h5$", "", c) for c in raw_cases]

        if not self.cases:
            raise ValueError(f"Test manifest is empty: {self.split_file}")

        if strict and len(self.cases) != EXPECTED_TEST_VOLUMES:
            raise ValueError(
                f"Expected exactly {EXPECTED_TEST_VOLUMES} test volumes, "
                f"got {len(self.cases)} in {self.split_file}"
            )

        if validate_files:
            self._validate_all_files()

    def _validate_all_files(self) -> None:
        missing = []
        for modality in ("bSSFP", "LGE", "T2w"):
            mod_dir = self.data_root / modality / "test_vol_h5"
            for case_id in self.cases:
                h5_path = mod_dir / f"{case_id}.npy.h5"
                if not h5_path.is_file():
                    missing.append(str(h5_path))
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} test volume files; first 5: {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        case_id = self.cases[idx]

        bssfp_path = self.data_root / "bSSFP" / "test_vol_h5" / f"{case_id}.npy.h5"
        lge_path = self.data_root / "LGE" / "test_vol_h5" / f"{case_id}.npy.h5"
        t2w_path = self.data_root / "T2w" / "test_vol_h5" / f"{case_id}.npy.h5"

        with h5py.File(bssfp_path, "r") as f_bssfp:
            bssfp = f_bssfp["image"][:]
            label = f_bssfp["label"][:]

        with h5py.File(lge_path, "r") as f_lge:
            lge = f_lge["image"][:]

        with h5py.File(t2w_path, "r") as f_t2w:
            t2w = f_t2w["image"][:]

        label_np = np.asarray(label, dtype=np.int64)
        if self.canonical_label:
            label_np = CANONICAL_LUT[label_np]

        return {
            "bSSFP": _to_volume_image_tensor(bssfp),
            "LGE": _to_volume_image_tensor(lge),
            "T2w": _to_volume_image_tensor(t2w),
            "label": _to_volume_label_tensor(label_np),
            "case_id": case_id,
        }


def create_myops380_train_loader(
    split_file: Union[str, Path],
    data_root: Union[str, Path],
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Optional[Callable] = None,
    strict: bool = False,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """Factory helper to construct standardized MyoPS380 train DataLoader."""
    dataset = MyoPS380TrainDataset(
        split_file=split_file,
        data_root=data_root,
        transform=transform,
        strict=strict,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )


def create_myops380_test_loader(
    split_file: Union[str, Path],
    data_root: Union[str, Path],
    canonical_label: bool = False,
    num_workers: int = 0,
    strict: bool = False,
) -> DataLoader:
    """Factory helper to construct standardized MyoPS380 test DataLoader."""
    dataset = MyoPS380TestDataset(
        split_file=split_file,
        data_root=data_root,
        canonical_label=canonical_label,
        strict=strict,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )
