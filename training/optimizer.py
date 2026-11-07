"""
training/optimizer.py
=====================
AdamW Optimizer and Polynomial Learning Rate Schedule for MyoPS-Net.
Protocol: im_mseg_adamw_poly_v1
Benchmark: MyoPS380 (304 Train / 76 Test / 300 Epochs / Strict NO Validation)
"""

from __future__ import annotations
import math
from typing import Any, Dict, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


OPTIMIZATION_PROTOCOL: Dict[str, Any] = {
    "id": "im_mseg_adamw_poly_v1",
    "optimizer": "AdamW",
    "weight_decay_scope": "all_parameters",
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "scheduler": "polynomial_after_update_zero_based_iteration",
    "power": 0.9,
}


def build_optimizer(
    model: nn.Module,
    base_lr: float = 0.0003333,
    weight_decay: float = 0.0001,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> AdamW:
    """Build AdamW optimizer strictly matching im_mseg_adamw_poly_v1 protocol.
    
    All trainable parameters are placed into a single parameter group and receive
    weight decay (biases and normalization parameters are strictly NOT excluded).
    """
    params = list(model.parameters())
    return AdamW(
        params,
        lr=base_lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_updates: int = 37200,
    power: float = 0.9,
) -> LambdaLR:
    """Build polynomial learning rate decay scheduler stepped per batch update.
    
    Formula:
        lr(step) = base_lr * max(0.0, 1.0 - max(0, step - 1) / total_updates) ** power
    
    Note on step numbering:
        LambdaLR starts with step = 0 (before the first update, factor = 1.0).
        After 1st update, step = 1 (factor = 1.0 due to max(0, 1 - 1)).
        Subsequent updates (step >= 2) decay smoothly according to polynomial power 0.9.
    """
    if not isinstance(total_updates, int) or total_updates < 1:
        raise ValueError(f"total_updates must be a positive integer, got {total_updates}")

    return LambdaLR(
        optimizer,
        lambda step: max(0.0, 1.0 - max(0, step - 1) / total_updates) ** power,
    )


def compute_gradient_norm(
    model: nn.Module,
    clip_grad: float = 0.0,
    error_if_nonfinite: bool = True,
) -> float:
    """Calculate Euclidean gradient norm across model parameters and optionally clip.
    
    When clip_grad <= 0.0 (benchmark standard clip_grad = 0.0), gradients are not
    scaled down (max_norm = inf), but the Euclidean L2 norm is calculated.
    
    Parameters:
        model: nn.Module whose parameter gradients are evaluated.
        clip_grad: Maximum allowable gradient norm. If <= 0.0, clipping is disabled.
        error_if_nonfinite: If True (default), raises FloatingPointError when the
            computed gradient norm is non-finite (Inf/NaN). If False, returns the
            non-finite norm value (e.g. inf, nan), allowing callers (such as
            torch.amp.GradScaler in train.py) to skip the optimizer step and reduce
            the loss scale without crashing the training process.
            
    Returns:
        float: Computed Euclidean gradient norm.
    """
    max_norm = clip_grad if clip_grad > 0.0 else float("inf")
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        error_if_nonfinite=False,
    )
    norm_val = float(norm)
    if error_if_nonfinite and not math.isfinite(norm_val):
        raise FloatingPointError(f"Non-finite gradient norm detected: {norm_val}")
    return norm_val


def calculate_total_updates(epochs: int, batches_per_epoch: int, accum_steps: int = 1) -> int:
    """Calculate total scheduled optimizer updates across the entire training run."""
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")
    return epochs * math.ceil(batches_per_epoch / accum_steps)


def validate_optimization_checkpoint(checkpoint: Dict[str, Any]) -> None:
    """Validate that checkpoint optimization protocol matches im_mseg_adamw_poly_v1."""
    protocol = checkpoint.get("opt_protocol")
    if protocol is None and "optimization_protocol" in checkpoint:
        opt_p = checkpoint["optimization_protocol"]
        if isinstance(opt_p, dict):
            protocol = opt_p.get("id")
        elif isinstance(opt_p, str):
            protocol = opt_p
            
    if protocol != OPTIMIZATION_PROTOCOL["id"]:
        raise ValueError(
            f"Checkpoint optimization protocol '{protocol}' does not match "
            f"required '{OPTIMIZATION_PROTOCOL['id']}'"
        )
