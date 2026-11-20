"""Dataset and DataLoader implementations for MyoPS-Net on MyoPS380 and MyoPS++ benchmarks.

Strictly supports:
1. MyoPS380 Benchmark:
   - Protocol: myops380_train304_test76_last_v1
   - 304 train patients (1,972 slices) from train.txt, zero validation
   - 76 test patients from test_vol.txt (unit grid, HD95 in voxels)
   - Source labels in legacy order {0:bg, 1:myo, 2:scar, 3:edema} canonicalized to
     {0:bg, 1:myo, 2:edema, 3:scar}

2. MyoPS++ Benchmark (Profile myopspp_unet2d_roi128_1mm):
   - 64 train patients (296 slices) from train.txt, zero validation
   - 16 test patients from test_vol.txt (physical grid, HD95 in mm)
   - Source labels:
     - Raw NIfTI: {0:bg, 200:myo, 500:LV, 600:RV, 1220:edema, 2221:scar} mapped to
       canonical {0:bg, 1:myo, 2:edema, 3:scar}
     - Cached NPZ/H5: already in canonical order {0:bg, 1:myo, 2:edema, 3:scar}
   - Preserves physical voxel spacing in mm (e.g. [1.0, 1.0, slice_thickness])

Consistent Across Both Benchmarks:
- Input channels: 3 (C0, LGE, T2w) in [0.0, 1.0] float32
- Target spatial shape: 128x128
- Canonical 4 classes: 0: background, 1: normal myocardium, 2: edema, 3: scar
- Data augmentation: RandomGenerator (flips, rotations, scaling, noise)
- Model architecture and loss function: 100% unchanged
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import nibabel as nib
except ImportError:
    nib = None


EXPECTED_TRAIN_SLICES = 1972
EXPECTED_TRAIN_PATIENTS = 304
EXPECTED_TEST_VOLUMES = 76
EXPECTED_MYOPSPP_TRAIN_SLICES = 296
EXPECTED_MYOPSPP_TRAIN_PATIENTS = 64
EXPECTED_MYOPSPP_TEST_VOLUMES = 16

CANONICAL_LUT = np.array([0, 1, 3, 2], dtype=np.int64)


def seed_worker(worker_id: int) -> None:
    """Use Python/NumPy worker seeding policy (picklable on Windows)."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def canonicalize_raw_myopspp(label: np.ndarray) -> np.ndarray:
    """Maps raw MyoPS++ labels (0, 200, 500, 600, 1220, 2221) to canonical 0..3."""
    raw = np.asarray(label, dtype=np.int64)
    out = np.zeros_like(raw, dtype=np.int64)
    out[raw == 200] = 1   # Normal myocardium
    out[raw == 1220] = 2  # Edema
    out[raw == 2221] = 3  # Scar
    # 0, 500 (LV), 600 (RV) remain 0 (background)
    return out


def canonicalize_label(label: np.ndarray, source_order: str = "legacy") -> np.ndarray:
    """Converts labels to canonical order: {0:bg, 1:myo, 2:edema, 3:scar}."""
    arr = np.asarray(label, dtype=np.int64)
    if source_order == "legacy":
        lut = np.array([0, 1, 3, 2], dtype=np.int64)
        return lut[arr]
    elif source_order == "canonical":
        return arr
    elif source_order == "raw_myopspp":
        return canonicalize_raw_myopspp(arr)
    elif source_order == "auto":
        if arr.max() > 3:
            return canonicalize_raw_myopspp(arr)
        return arr
    else:
        raise ValueError(f"Unsupported source label order: {source_order}")


def ingest_image_array(image: np.ndarray, name: str = "sample") -> np.ndarray:
    """Ingests and validates preprocessed image array into [0.0, 1.0] float32."""
    img = np.asarray(image, dtype=np.float32)
    if not np.isfinite(img).all():
        raise ValueError(f"Non-finite values (NaN or Inf) encountered in image array for {name}.")
    if img.min() < -1e-5 or img.max() > 1.0 + 1e-5:
        img = np.clip(img, 0.0, 1.0)
    return img


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
    clean = re.sub(r"(\.nii)?\.gz$", "", clean)
    return clean.split("_slice")[0]


def read_manifest_lines(file_path: Union[str, Path]) -> List[str]:
    """Reads non-empty, stripped lines from a manifest file."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8-sig") as f:
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


def _find_cases_directory(path: Path) -> Optional[Path]:
    """Finds directory containing cases/CenterB or cases/CenterC for MyoPS++."""
    candidates = [
        path,
        path / "cases",
        path / "MyoPSpp_UNet2D_ROI128_1mm" / "cases",
        path / "MyoPSpp_UNet2D_ROI128_1mm",
        path.parent / "cases",
        path.parent / "MyoPSpp_UNet2D_ROI128_1mm" / "cases",
        path.parent / "MyoPSpp_UNet2D_ROI128_1mm",
    ]
    for c in candidates:
        if c.is_dir() and ((c / "CenterB").is_dir() or (c / "CenterC").is_dir()):
            return c
        if (c / "cases").is_dir() and (((c / "cases") / "CenterB").is_dir() or ((c / "cases") / "CenterC").is_dir()):
            return c / "cases"
    return None


class MyoPS380TrainDataset(Dataset):
    """Slice-level training dataset for MyoPS380 and MyoPS++ benchmarks.

    Supports:
      1. MyoPS++ direct NIfTI cases (CenterB & CenterC, 64 patients, 296 slices).
      2. MyoPS380 preprocessed NPZ slices (bSSFP, LGE, T2w, 304 patients, 1,972 slices).
    """

    def __init__(
        self,
        split_file: Union[str, Path],
        data_root: Union[str, Path],
        transform: Optional[Callable] = None,
        validate_files: bool = False,
        strict: bool = False,
        dataset: str = "auto",
        label_order: str = "auto",
        **kwargs: Any,
    ):
        p_data = Path(data_root)
        if not p_data.is_dir():
            raise FileNotFoundError(f"data_root directory not found: {p_data}")

        # Detect MyoPS++ NIfTI export format
        cases_dir = _find_cases_directory(p_data)
        self.is_nifti_export = (cases_dir is not None)
        self.nifti_cases_dir = cases_dir

        is_myopspp = (
            self.is_nifti_export
            or "myopspp" in str(p_data).lower()
            or "myopspp" in str(dataset).lower()
            or "myopspp" in str(split_file).lower()
        )
        self.dataset_id = "myopspp_unet2d_roi128_1mm" if is_myopspp else "myops380"

        # Resolve split file
        try:
            self.split_file = resolve_split_file(split_file, "train.txt")
        except FileNotFoundError:
            repo_root = Path(__file__).resolve().parent.parent
            if is_myopspp:
                cand_split = repo_root / "dataset" / "splits" / "myopspp_unet2d_roi128_1mm" / "train.txt"
                if not cand_split.is_file():
                    cand_split = repo_root / "preprocessing" / "splits" / "myopspp_unet2d_roi128_1mm" / "train.txt"
            else:
                cand_split = repo_root / "preprocessing" / "splits" / "myops380_no_val" / "train.txt"
                if not cand_split.is_file():
                    cand_split = repo_root / "dataset" / "splits" / "myops380_no_val" / "train.txt"
            if cand_split.is_file():
                self.split_file = cand_split
            else:
                raise

        self.transform = transform
        self.slices: List[str] = read_manifest_lines(self.split_file)

        if not self.slices:
            raise ValueError(f"Train manifest is empty: {self.split_file}")

        self.patients = sorted(list({extract_patient_id(s) for s in self.slices}))

        if strict:
            expected_slices = EXPECTED_MYOPSPP_TRAIN_SLICES if is_myopspp else EXPECTED_TRAIN_SLICES
            expected_patients = EXPECTED_MYOPSPP_TRAIN_PATIENTS if is_myopspp else EXPECTED_TRAIN_PATIENTS
            if len(self.slices) != expected_slices:
                raise ValueError(
                    f"Expected exactly {expected_slices} training slices, "
                    f"got {len(self.slices)} in {self.split_file}"
                )
            if len(self.patients) != expected_patients:
                raise ValueError(
                    f"Expected exactly {expected_patients} unique training patients, "
                    f"got {len(self.patients)} in {self.split_file}"
                )

        # Volume memory cache for fast reading of NIfTI slices
        self._vol_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

        if not self.is_nifti_export:
            # Handle NPZ directory resolution
            if (p_data / "Processed_data" / "bSSFP" / "train_npz").is_dir():
                p_data = p_data / "Processed_data"
            elif (p_data / "bSSFP" / "train_npz").is_dir():
                pass
            else:
                for cand in [p_data / "Processed_data", p_data.parent / "Processed_data"]:
                    if (cand / "bSSFP" / "train_npz").is_dir():
                        p_data = cand
                        break

            self.data_root = p_data
            self.modality_dirs = {
                "bSSFP": self.data_root / "bSSFP" / "train_npz",
                "LGE": self.data_root / "LGE" / "train_npz",
                "T2w": self.data_root / "T2w" / "train_npz",
            }

            # Determine source label order for NPZ cache
            if label_order != "auto":
                self.source_order = label_order
            elif is_myopspp:
                self.source_order = "canonical"
            else:
                self.source_order = "legacy"

            if validate_files:
                self._validate_npz_files()
        else:
            self.data_root = cases_dir.parent
            self.source_order = "raw_myopspp"

            if validate_files:
                self._validate_nifti_files()

    def _validate_npz_files(self) -> None:
        missing = []
        for modality, mod_dir in self.modality_dirs.items():
            for slice_id in self.slices:
                npz_path = mod_dir / f"{slice_id}.npz"
                if not npz_path.is_file():
                    missing.append(str(npz_path))
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} train slice NPZ files; first 5: {missing[:5]}"
            )

    def _validate_nifti_files(self) -> None:
        missing = []
        for pat in self.patients:
            center = "CenterB" if pat.startswith("Case2") else "CenterC"
            case_dir = self.nifti_cases_dir / center / pat
            if not case_dir.is_dir():
                case_dir = self.nifti_cases_dir / pat
            for suffix in ("_C0.nii.gz", "_LGE.nii.gz", "_T2.nii.gz", "_gd.nii.gz"):
                p = case_dir / f"{pat}{suffix}"
                if not p.is_file():
                    missing.append(str(p))
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} train NIfTI files; first 5: {missing[:5]}"
            )

    def _load_nifti_case(self, patient_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads and caches 3D volumes (C0, LGE, T2, gd) for a patient."""
        if patient_id in self._vol_cache:
            return self._vol_cache[patient_id]

        if nib is None:
            raise ImportError("nibabel is required to read NIfTI cases directly. Run: pip install nibabel")

        center = "CenterB" if patient_id.startswith("Case2") else "CenterC"
        case_dir = self.nifti_cases_dir / center / patient_id
        if not case_dir.is_dir():
            if (self.nifti_cases_dir / patient_id).is_dir():
                case_dir = self.nifti_cases_dir / patient_id
            else:
                for cand in self.nifti_cases_dir.glob(f"*/{patient_id}"):
                    if cand.is_dir():
                        case_dir = cand
                        break

        p_c0 = case_dir / f"{patient_id}_C0.nii.gz"
        p_lge = case_dir / f"{patient_id}_LGE.nii.gz"
        p_t2 = case_dir / f"{patient_id}_T2.nii.gz"
        p_gd = case_dir / f"{patient_id}_gd.nii.gz"

        for p in (p_c0, p_lge, p_t2, p_gd):
            if not p.is_file():
                raise FileNotFoundError(f"Missing MyoPS++ modality file: {p}")

        c0_vol = ingest_image_array(nib.load(str(p_c0)).get_fdata(dtype=np.float32), name=f"{patient_id}_c0")
        lge_vol = ingest_image_array(nib.load(str(p_lge)).get_fdata(dtype=np.float32), name=f"{patient_id}_lge")
        t2_vol = ingest_image_array(nib.load(str(p_t2)).get_fdata(dtype=np.float32), name=f"{patient_id}_t2")
        raw_label = np.asarray(nib.load(str(p_gd)).dataobj)
        label_vol = canonicalize_raw_myopspp(raw_label)

        vol_tuple = (c0_vol, lge_vol, t2_vol, label_vol)
        self._vol_cache[patient_id] = vol_tuple
        return vol_tuple

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        slice_id = self.slices[idx]

        if self.is_nifti_export:
            match = re.match(r"^(Case\d+)_slice(\d+)$", slice_id)
            if not match:
                raise ValueError(f"Invalid MyoPS++ slice format: {slice_id}; expected CaseXXXX_sliceYYY")
            patient_id, slice_idx_str = match.group(1), match.group(2)
            slice_idx = int(slice_idx_str)

            c0_vol, lge_vol, t2_vol, label_vol = self._load_nifti_case(patient_id)
            bssfp = c0_vol[:, :, slice_idx]
            lge = lge_vol[:, :, slice_idx]
            t2w = t2_vol[:, :, slice_idx]
            label = label_vol[:, :, slice_idx]
        else:
            bssfp_path = self.modality_dirs["bSSFP"] / f"{slice_id}.npz"
            lge_path = self.modality_dirs["LGE"] / f"{slice_id}.npz"
            t2w_path = self.modality_dirs["T2w"] / f"{slice_id}.npz"

            with np.load(bssfp_path) as d_bssfp:
                bssfp = ingest_image_array(d_bssfp["image"], name=f"{slice_id}_bssfp")
                label = np.asarray(d_bssfp["label"])

            with np.load(lge_path) as d_lge:
                lge = ingest_image_array(d_lge["image"], name=f"{slice_id}_lge")

            with np.load(t2w_path) as d_t2w:
                t2w = ingest_image_array(d_t2w["image"], name=f"{slice_id}_t2w")

            label = np.round(label).astype(np.int64)
            if self.source_order == "legacy":
                label = canonicalize_label(label, source_order="legacy")

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
                if "bSSFP" in sample_out:
                    bssfp = sample_out["bSSFP"]
                elif "image" in sample_out:
                    bssfp = sample_out["image"]

                if "LGE" in sample_out:
                    lge = sample_out["LGE"]
                elif "image1" in sample_out:
                    lge = sample_out["image1"]

                if "T2w" in sample_out:
                    t2w = sample_out["T2w"]
                elif "image2" in sample_out:
                    t2w = sample_out["image2"]

                label = sample_out.get("label", label)
            elif isinstance(sample_out, (list, tuple)) and len(sample_out) >= 4:
                bssfp, lge, t2w, label = sample_out[0], sample_out[1], sample_out[2], sample_out[3]

        t_bssfp = _to_image_tensor(bssfp)
        t_lge = _to_image_tensor(lge)
        t_t2w = _to_image_tensor(t2w)
        t_label = _to_label_tensor(label)

        return {
            "bSSFP": t_bssfp,
            "LGE": t_lge,
            "T2w": t_t2w,
            "image": t_bssfp,
            "image1": t_lge,
            "image2": t_t2w,
            "label": t_label,
            "case_slice": slice_id,
        }


class MyoPS380TestDataset(Dataset):
    """Volume-level test evaluation dataset for MyoPS380 and MyoPS++ benchmarks.

    Supports:
      1. MyoPS++ 3D NIfTI volumes across 16 official test cases (HD95 in mm).
      2. MyoPS380 3D H5 volumes across 76 official test cases (HD95 in voxels).
    """

    def __init__(
        self,
        split_file: Union[str, Path],
        data_root: Union[str, Path],
        canonical_label: bool = True,
        validate_files: bool = False,
        strict: bool = False,
        dataset: str = "auto",
        **kwargs: Any,
    ):
        p_data = Path(data_root)
        if not p_data.is_dir():
            raise FileNotFoundError(f"data_root directory not found: {p_data}")

        cases_dir = _find_cases_directory(p_data)
        self.is_nifti_export = (cases_dir is not None)
        self.nifti_cases_dir = cases_dir

        is_myopspp = (
            self.is_nifti_export
            or "myopspp" in str(p_data).lower()
            or "myopspp" in str(dataset).lower()
            or "myopspp" in str(split_file).lower()
        )
        self.dataset_id = "myopspp_unet2d_roi128_1mm" if is_myopspp else "myops380"

        try:
            self.split_file = resolve_split_file(split_file, "test_vol.txt")
        except FileNotFoundError:
            repo_root = Path(__file__).resolve().parent.parent
            if is_myopspp:
                cand_split = repo_root / "dataset" / "splits" / "myopspp_unet2d_roi128_1mm" / "test_vol.txt"
                if not cand_split.is_file():
                    cand_split = repo_root / "preprocessing" / "splits" / "myopspp_unet2d_roi128_1mm" / "test_vol.txt"
            else:
                cand_split = repo_root / "preprocessing" / "splits" / "myops380_no_val" / "test_vol.txt"
                if not cand_split.is_file():
                    cand_split = repo_root / "dataset" / "splits" / "myops380_no_val" / "test_vol.txt"
            if cand_split.is_file():
                self.split_file = cand_split
            else:
                raise

        self.canonical_label = canonical_label
        raw_cases = read_manifest_lines(self.split_file)
        self.cases: List[str] = [
            re.sub(r"(\.npy)?\.h5$", "", c).replace(".nii.gz", "").replace(".nii", "")
            for c in raw_cases
        ]

        if not self.cases:
            raise ValueError(f"Test manifest is empty: {self.split_file}")

        if strict:
            expected_volumes = EXPECTED_MYOPSPP_TEST_VOLUMES if is_myopspp else EXPECTED_TEST_VOLUMES
            if len(self.cases) != expected_volumes:
                raise ValueError(
                    f"Expected exactly {expected_volumes} test volumes, "
                    f"got {len(self.cases)} in {self.split_file}"
                )

        if not self.is_nifti_export:
            if (p_data / "Processed_data" / "bSSFP" / "test_vol_h5").is_dir():
                p_data = p_data / "Processed_data"
            elif (p_data / "bSSFP" / "test_vol_h5").is_dir():
                pass
            else:
                for cand in [p_data / "Processed_data", p_data.parent / "Processed_data"]:
                    if (cand / "bSSFP" / "test_vol_h5").is_dir():
                        p_data = cand
                        break

            self.data_root = p_data
            self.modality_dirs = {
                "bSSFP": self.data_root / "bSSFP" / "test_vol_h5",
                "LGE": self.data_root / "LGE" / "test_vol_h5",
                "T2w": self.data_root / "T2w" / "test_vol_h5",
            }
            if validate_files:
                self._validate_h5_files()
        else:
            self.data_root = cases_dir.parent
            if validate_files:
                self._validate_nifti_test_files()

    def _validate_h5_files(self) -> None:
        missing = []
        for modality, mod_dir in self.modality_dirs.items():
            for case_id in self.cases:
                h5_path = mod_dir / f"{case_id}.npy.h5"
                if not h5_path.is_file():
                    h5_alt = mod_dir / f"{case_id}.h5"
                    if not h5_alt.is_file():
                        missing.append(str(h5_path))
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} test volume files; first 5: {missing[:5]}"
            )

    def _validate_nifti_test_files(self) -> None:
        missing = []
        for case_id in self.cases:
            center = "CenterB" if case_id.startswith("Case2") else "CenterC"
            case_dir = self.nifti_cases_dir / center / case_id
            if not case_dir.is_dir():
                case_dir = self.nifti_cases_dir / case_id
            for suffix in ("_C0.nii.gz", "_LGE.nii.gz", "_T2.nii.gz", "_gd.nii.gz"):
                p = case_dir / f"{case_id}{suffix}"
                if not p.is_file():
                    missing.append(str(p))
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} test NIfTI files; first 5: {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, np.ndarray]]:
        case_id = self.cases[idx]

        if self.is_nifti_export:
            if nib is None:
                raise ImportError("nibabel is required to load MyoPS++ NIfTI volumes directly.")
            center = "CenterB" if case_id.startswith("Case2") else "CenterC"
            case_dir = self.nifti_cases_dir / center / case_id
            if not case_dir.is_dir():
                if (self.nifti_cases_dir / case_id).is_dir():
                    case_dir = self.nifti_cases_dir / case_id
                else:
                    for cand in self.nifti_cases_dir.glob(f"*/{case_id}"):
                        if cand.is_dir():
                            case_dir = cand
                            break

            p_c0 = case_dir / f"{case_id}_C0.nii.gz"
            p_lge = case_dir / f"{case_id}_LGE.nii.gz"
            p_t2 = case_dir / f"{case_id}_T2.nii.gz"
            p_gd = case_dir / f"{case_id}_gd.nii.gz"

            c0_nib = nib.load(str(p_c0))
            bssfp = ingest_image_array(c0_nib.get_fdata(dtype=np.float32), name=f"{case_id}_c0")
            lge = ingest_image_array(nib.load(str(p_lge)).get_fdata(dtype=np.float32), name=f"{case_id}_lge")
            t2w = ingest_image_array(nib.load(str(p_t2)).get_fdata(dtype=np.float32), name=f"{case_id}_t2")
            raw_label = np.asarray(nib.load(str(p_gd)).dataobj)
            label_np = canonicalize_raw_myopspp(raw_label)

            meta_json = case_dir / "metadata.json"
            spacing = np.array([1.0, 1.0, 10.0], dtype=np.float64)
            affine = np.asarray(c0_nib.affine, dtype=np.float64)
            if meta_json.is_file():
                try:
                    meta = json.loads(meta_json.read_text(encoding="utf-8"))
                    if "output_spacing_mm" in meta:
                        spacing = np.asarray(meta["output_spacing_mm"], dtype=np.float64)
                    elif "native_spacing_mm" in meta:
                        spacing = np.asarray(meta["native_spacing_mm"], dtype=np.float64)
                    if "output_affine" in meta:
                        affine = np.asarray(meta["output_affine"], dtype=np.float64)
                except Exception:
                    pass
            else:
                spacing = np.asarray(c0_nib.header.get_zooms()[:3], dtype=np.float64)

            spacing_unit = "mm"
            has_geometry = True
        else:
            bssfp_path = self.modality_dirs["bSSFP"] / f"{case_id}.npy.h5"
            if not bssfp_path.is_file():
                bssfp_path = self.modality_dirs["bSSFP"] / f"{case_id}.h5"

            lge_path = self.modality_dirs["LGE"] / f"{case_id}.npy.h5"
            if not lge_path.is_file():
                lge_path = self.modality_dirs["LGE"] / f"{case_id}.h5"

            t2w_path = self.modality_dirs["T2w"] / f"{case_id}.npy.h5"
            if not t2w_path.is_file():
                t2w_path = self.modality_dirs["T2w"] / f"{case_id}.h5"

            with h5py.File(bssfp_path, "r") as f_bssfp:
                bssfp = ingest_image_array(f_bssfp["image"][:], name=f"{case_id}_bssfp")
                label = f_bssfp["label"][:]

            with h5py.File(lge_path, "r") as f_lge:
                lge = ingest_image_array(f_lge["image"][:], name=f"{case_id}_lge")

            with h5py.File(t2w_path, "r") as f_t2w:
                t2w = ingest_image_array(f_t2w["image"][:], name=f"{case_id}_t2w")

            label_np = np.asarray(label, dtype=np.int64)
            if self.canonical_label:
                label_np = CANONICAL_LUT[label_np]

            spacing = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            affine = np.eye(4, dtype=np.float64)
            spacing_unit = "voxel"
            has_geometry = False

        t_bssfp = _to_volume_image_tensor(bssfp)
        t_lge = _to_volume_image_tensor(lge)
        t_t2w = _to_volume_image_tensor(t2w)
        t_label = _to_volume_label_tensor(label_np)

        return {
            "bSSFP": t_bssfp,
            "LGE": t_lge,
            "T2w": t_t2w,
            "image": t_bssfp,
            "image1": t_lge,
            "image2": t_t2w,
            "label": t_label,
            "case_id": case_id,
            "case_name": case_id,
            "spacing": spacing,
            "affine": affine,
            "spacing_unit": spacing_unit,
            "has_geometry": has_geometry,
        }


MyoPS380TestVolumeDataset = MyoPS380TestDataset


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
    dataset: str = "auto",
    label_order: str = "auto",
) -> DataLoader:
    """Factory helper to construct standardized train DataLoader for MyoPS380 and MyoPS++."""
    dataset_inst = MyoPS380TrainDataset(
        split_file=split_file,
        data_root=data_root,
        transform=transform,
        strict=strict,
        dataset=dataset,
        label_order=label_order,
    )
    return DataLoader(
        dataset_inst,
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
    canonical_label: bool = True,
    num_workers: int = 0,
    strict: bool = False,
    dataset: str = "auto",
) -> DataLoader:
    """Factory helper to construct standardized test DataLoader for MyoPS380 and MyoPS++."""
    dataset_inst = MyoPS380TestDataset(
        split_file=split_file,
        data_root=data_root,
        canonical_label=canonical_label,
        strict=strict,
        dataset=dataset,
    )
    return DataLoader(
        dataset_inst,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )
