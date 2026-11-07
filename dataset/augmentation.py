"""dataset/augmentation.py
=======================
Standardized Multi-Modality Data Augmentation for MyoPS380 Benchmark.

Reproduces I-MMSeg-Fork and SCAR M2-Plus augmentation with 100% RNG and
tensor fidelity, ensuring scientifically valid benchmark comparisons.

Key Features:
-------------
1. Synchronous Spatial Transforms:
   - Rotation/Flip, small-angle rotation, spatial shift, random crop, and zoom
     are applied identically across all 3 CMR modalities (bSSFP, LGE, T2w) and
     the segmentation label mask to preserve multi-modal anatomical alignment.
2. Synchronous Intensity Transforms:
   - Shared additive Gaussian noise (mean=0, std=0.2), shared brightness scaling
     factor in [0.8, 1.2], and shared 255-based gamma power correction in [0.8, 1.2]
     matching the reference SCAR / I-MMSeg recipe.
3. Dual-Schema Key Compatibility:
   - Ingests MyoPS-Net format: {'bSSFP': img, 'LGE': img, 'T2w': img, 'label': mask}
   - Ingests SCAR format:      {'image': img, 'image1': img, 'image2': img, 'label': mask}
   - Ingests tuple format:     (img_bSSFP, img_LGE, img_T2w, mask)
   - Preserves all metadata keys without duplicating tensors in DataLoader batches.
4. Deterministic ResizeGenerator:
   - Provides alignment-preserving bilinear (images) and nearest-neighbor (label)
     resizing for validation/inference slices.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom
import torch
from torch.nn import functional as F


def random_rot_flip(
    image: np.ndarray,
    image1: np.ndarray,
    image2: np.ndarray,
    label: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply synchronized 90-degree rotation and random axis flip."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    image1 = np.rot90(image1, k)
    image2 = np.rot90(image2, k)
    label = np.rot90(label, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    image1 = np.flip(image1, axis=axis).copy()
    image2 = np.flip(image2, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, image1, image2, label


def random_rotate(
    image: np.ndarray,
    image1: np.ndarray,
    image2: np.ndarray,
    label: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply synchronized small-angle rotation with nearest-neighbor interpolation."""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    image1 = ndimage.rotate(image1, angle, order=0, reshape=False)
    image2 = ndimage.rotate(image2, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, image1, image2, label


def add_gaussian_noise(
    image: np.ndarray,
    image1: np.ndarray,
    image2: np.ndarray,
    label: np.ndarray,
    mean: float = 0.0,
    std: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply a single shared Gaussian noise map across all 3 modalities."""
    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    image1 = image1 + noise
    image2 = image2 + noise
    image = np.clip(image, 0.0, 1.0)
    image1 = np.clip(image1, 0.0, 1.0)
    image2 = np.clip(image2, 0.0, 1.0)
    return image, image1, image2, label


class RandomGenerator:
    """Standardized multi-modality data augmentation for MyoPS380 benchmark.

    Reproduces I-MMSeg-Fork and SCAR M2-Plus augmentation with 100% RNG and
    tensor fidelity. Preserves identical spatial transformations across all
    modalities (bSSFP, LGE, T2w) and label mask. Supports both MyoPS-Net
    modality keys ('bSSFP', 'LGE', 'T2w') and legacy SCAR keys ('image',
    'image1', 'image2'), as well as tuple/list inputs.
    """

    def __init__(self, output_size: Union[int, Sequence[int]]):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif len(output_size) == 2 and all(int(v) > 0 for v in output_size):
            self.output_size = (int(output_size[0]), int(output_size[1]))
        else:
            raise ValueError(
                f"output_size must be an int or a sequence of two positive ints, got {output_size!r}"
            )

    def __call__(
        self, sample: Union[Dict[str, Any], Sequence[Any]]
    ) -> Union[Dict[str, Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # 1. Unpack sample inputs
        is_dict = isinstance(sample, dict)
        if is_dict:
            has_bssfp = "bSSFP" in sample
            has_image = "image" in sample
            if has_bssfp:
                image = sample["bSSFP"]
                image1 = sample["LGE"]
                image2 = sample["T2w"]
            elif has_image:
                image = sample["image"]
                image1 = sample["image1"]
                image2 = sample["image2"]
            else:
                raise KeyError(
                    f"Sample dict must contain either ('bSSFP', 'LGE', 'T2w') or ('image', 'image1', 'image2'). "
                    f"Found keys: {list(sample.keys())}"
                )
            if "label" not in sample:
                raise KeyError(
                    f"Sample dict must contain 'label' key. Found keys: {list(sample.keys())}"
                )
            label = sample["label"]
        elif isinstance(sample, (list, tuple)) and len(sample) >= 4:
            has_bssfp = False
            has_image = False
            image, image1, image2, label = sample[0], sample[1], sample[2], sample[3]
        else:
            raise TypeError(
                f"Sample must be a dict or a sequence with at least 4 items, got {type(sample)}"
            )

        # Convert tensors to numpy if passed as PyTorch tensors
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(image1, torch.Tensor):
            image1 = image1.detach().cpu().numpy()
        if isinstance(image2, torch.Tensor):
            image2 = image2.detach().cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()

        # Ensure arrays are 2D (H, W) for spatial transforms
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)
        if image1.ndim == 3 and image1.shape[0] == 1:
            image1 = image1.squeeze(0)
        if image2.ndim == 3 and image2.shape[0] == 1:
            image2 = image2.squeeze(0)
        if label.ndim == 3 and label.shape[0] == 1:
            label = label.squeeze(0)

        # 2. Sequence of Augmentations (exact I-MMSeg / SCAR order and RNG calls)
        # Step 1: Branching rotation/flip, small-angle rotation, or Gaussian noise
        if random.random() > 0.5:
            image, image1, image2, label = random_rot_flip(image, image1, image2, label)
        elif random.random() > 0.5:
            image, image1, image2, label = random_rotate(image, image1, image2, label)
        elif random.random() > 0.5:
            image, image1, image2, label = add_gaussian_noise(image, image1, image2, label)

        # Step 2: Cyclic spatial shift
        if random.random() > 0.5:
            shift_x = random.randint(-5, 5)
            shift_y = random.randint(-5, 5)
            image = np.roll(image, (shift_x, shift_y), axis=(0, 1))
            image1 = np.roll(image1, (shift_x, shift_y), axis=(0, 1))
            image2 = np.roll(image2, (shift_x, shift_y), axis=(0, 1))
            label = np.roll(label, (shift_x, shift_y), axis=(0, 1))

        # Step 3: Brightness scaling (255-based clip)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255)
            image1 = np.clip(image1 * factor, 0, 255)
            image2 = np.clip(image2 * factor, 0, 255)

        # Step 4: Gamma correction (255-based power)
        if random.random() > 0.5:
            gamma = random.uniform(0.8, 1.2)
            image = np.power(image / 255.0, gamma) * 255.0
            image1 = np.power(image1 / 255.0, gamma) * 255.0
            image2 = np.power(image2 / 255.0, gamma) * 255.0

        # Step 5: Random crop
        if random.random() > 0.5:
            top = random.randint(0, image.shape[0] // 5)
            left = random.randint(0, image.shape[1] // 5)
            image = image[top : top + self.output_size[0], left : left + self.output_size[1]]
            image1 = image1[top : top + self.output_size[0], left : left + self.output_size[1]]
            image2 = image2[top : top + self.output_size[0], left : left + self.output_size[1]]
            label = label[top : top + self.output_size[0], left : left + self.output_size[1]]

        # Step 6: Spatial zoom / resize if shape differs from output_size
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image1 = zoom(image1, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image2 = zoom(image2, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # Step 7: Packaging as PyTorch tensors
        image_t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image1_t = torch.from_numpy(image1.astype(np.float32)).unsqueeze(0)
        image2_t = torch.from_numpy(image2.astype(np.float32)).unsqueeze(0)
        label_t = torch.from_numpy(label.astype(np.float32)).long()

        if is_dict:
            result = dict(sample)
            if has_bssfp or not has_image:
                result["bSSFP"] = image_t
                result["LGE"] = image1_t
                result["T2w"] = image2_t
            if has_image:
                result["image"] = image_t
                result["image1"] = image1_t
                result["image2"] = image2_t
            result["label"] = label_t
            return result
        else:
            return image_t, image1_t, image2_t, label_t


class ResizeGenerator:
    """Deterministic, alignment-preserving image resize and tensor conversion."""

    def __init__(self, output_size: Union[int, Sequence[int]]):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif len(output_size) == 2 and all(int(v) > 0 for v in output_size):
            self.output_size = (int(output_size[0]), int(output_size[1]))
        else:
            raise ValueError(
                f"output_size must contain two positive dimensions, got {output_size!r}"
            )

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(sample)
        shape = sample["label"].shape
        if len(shape) != 2:
            raise ValueError(
                "Slice transforms require 2D samples; volume resize belongs in inference."
            )

        has_bssfp = "bSSFP" in sample
        has_image = "image" in sample

        keys = []
        if has_bssfp:
            keys.extend(["bSSFP", "LGE", "T2w"])
        if has_image:
            keys.extend(["image", "image1", "image2"])
        keys.append("label")

        for key in keys:
            value = np.asarray(sample[key])
            if value.shape != shape:
                raise ValueError(
                    f"Modalities and mask must have identical spatial shapes. "
                    f"Key {key} has shape {value.shape} vs label {shape}"
                )
            dtype = np.int64 if key == "label" else np.float32
            tensor = torch.from_numpy(np.array(value, dtype=dtype, copy=True, order="C"))
            if shape != self.output_size:
                if key == "label":
                    tensor = F.interpolate(
                        tensor[None, None].float(), size=self.output_size, mode="nearest"
                    )[0, 0].long()
                else:
                    tensor = F.interpolate(
                        tensor[None, None],
                        size=self.output_size,
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
            result[key] = tensor if key == "label" else tensor.unsqueeze(0)
        return result
