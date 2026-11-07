"""Describe the accepted binary-MPC adaptation and lock resumable training inputs."""
from __future__ import annotations

import hashlib
from pathlib import Path
import platform
from typing import Any

import numpy as np
import scipy
import torch

from training.optimizer import OPTIMIZATION_PROTOCOL


MODEL_CONTRACT = {
    "id": "myopsnet_l_binary_mpc_v1",
    "variant": "MyoPS-Net-L adapted with binary MPC",
    "adaptation": "LV supervision unavailable; MPC predicts background/whole myocardium",
    "modalities": ["bSSFP", "LGE", "T2w"],
    "input_channels": [3, 2, 2],
    "head_classes": {
        "C0": ["background", "whole_myocardium"],
        "LGE": ["background", "non_scar_myocardium", "scar"],
        "T2": ["background", "normal_myocardium", "edema_inclusive"],
    },
    "source_label_order": "legacy_background_normal_scar_edema",
    "exported_label_order": "canonical_background_normal_edema_scar",
    "adapter_version": "binary_mpc_scar_priority_v1",
    "activation": "softmax",
    "detach_prior": True,
    "detach_reference_features": True,
}
INITIALIZATION = {
    "id": "native_kaiming_normal_v1",
    "function": "utils.tools.weights_init",
    "conv": "kaiming_normal_fan_in_relu; zero_bias",
    "batchnorm": "unit_weight; zero_bias",
    "linear": "xavier_uniform; zero_bias",
}
LOSS_CONTRACT = {
    "id": "myopsnet_binary_mpc_multihead_v1",
    "segmentation": "per_image_mean_class_dice_plus_log_frequency_wce",
    "include_background": True,
    "absent_class_wce_weight": 0.0,
    "epsilon": 1e-7,
    "lambda_s": 2.0,
    "lambda_e": 2.0,
    "lambda_inv": 1.0,
    "lambda_inc": 1.0,
}
RNG_POLICY = {
    "id": "scar_loader_generator_worker_seed_v1",
    "worker_seed": "torch.initial_seed_mod_2**32_for_python_and_numpy",
    "persistent_workers": False,
}

_SOURCE_FILES = (
    "train.py", "dataset/myops380_dataset.py", "dataset/augmentation.py",
    "network/model.py", "network/unet.py", "criterion/loss.py", "utils/tools.py",
    "training/optimizer.py", "training/checkpoint.py", "training/provenance.py",
)
_SETTING_KEYS = (
    "seed", "epochs", "batch_size", "accum_steps", "base_lr", "weight_decay",
    "clip_grad", "num_workers", "deterministic", "dry_run", "max_batches",
)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_resume_contract(args, dataset, *, device, amp_dtype, pin_memory: bool) -> dict:
    """Hash training cache bytes read-only; never load or run the model on test data."""
    root = Path(args.data_root)
    digest = hashlib.sha256()
    for modality in MODEL_CONTRACT["modalities"]:
        for name in sorted(dataset.slices):
            relative = f"{modality}/train_npz/{name}.npz"
            digest.update(f"{relative}\0{_file_sha256(root / relative)}\n".encode("utf-8"))
    source_root = Path(__file__).resolve().parents[1]
    settings = {key: getattr(args, key) for key in _SETTING_KEYS}
    settings.update(device_type=device.type, amp_dtype=str(amp_dtype), pin_memory=pin_memory)
    batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    processed = min(batches, 2) if args.dry_run else batches
    if args.max_batches:
        processed = min(processed, args.max_batches)
    return {
        "version": 1,
        "settings": settings,
        "run_kind": "debug" if args.dry_run or args.max_batches else "training",
        "batches_per_epoch": batches,
        "processed_batches_per_epoch": processed,
        "data": {
            "train_slices": len(dataset),
            "train_manifest_sha256": _file_sha256(Path(dataset.split_file)),
            "train_cache_sha256": digest.hexdigest(),
        },
        "source_sha256": {name: _file_sha256(source_root / name) for name in _SOURCE_FILES},
        "runtime": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "scipy": scipy.__version__, "cuda": torch.version.cuda,
        },
        "model_contract": MODEL_CONTRACT,
        "initialization": INITIALIZATION,
        "loss_contract": LOSS_CONTRACT,
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "rng_policy": RNG_POLICY,
    }


def _differences(saved: Any, expected: Any, prefix: str = "") -> list[str]:
    if isinstance(saved, dict) and isinstance(expected, dict):
        result = []
        for key in sorted(saved.keys() | expected.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in saved or key not in expected:
                result.append(path)
            else:
                result.extend(_differences(saved[key], expected[key], path))
        return result
    return [] if saved == expected else [prefix]


def validate_resume_contract(checkpoint: dict, expected: dict) -> None:
    saved = checkpoint.get("resume_contract")
    if not isinstance(saved, dict):
        raise ValueError(
            "Checkpoint lacks the current RNG/initialization resume contract. "
            "Start a new run; legacy checkpoints remain readable for inference."
        )
    differences = _differences(saved, expected)
    if differences:
        raise ValueError("Resume changes training contract: " + ", ".join(differences[:10]))
    if not isinstance(checkpoint.get("loader_rng"), torch.Tensor):
        raise ValueError("Resume checkpoint is missing the DataLoader generator state")
    settings = expected["settings"]
    successful = checkpoint["global_step"]
    skipped = checkpoint.get("amp_skipped_updates")
    opportunities = checkpoint["completed_epochs"] * expected["processed_batches_per_epoch"]
    if (
        checkpoint["planned_epochs"] != settings["epochs"]
        or checkpoint.get("seed") != settings["seed"]
        or checkpoint.get("total_updates") != settings["epochs"] * expected["batches_per_epoch"]
        or type(successful) is not int or successful < 0
        or type(skipped) is not int or skipped < 0
        or successful + skipped != opportunities
        or checkpoint["scheduler_state_dict"].get("last_epoch") != successful
    ):
        raise ValueError("Inconsistent resume budget, successful/skipped updates or scheduler state")
