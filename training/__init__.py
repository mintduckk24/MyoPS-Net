"""
training
========
Optimization, learning rate scheduling, and atomic checkpoint persistence
for MyoPS-Net baseline standardization.
Protocol: myops380_train304_test76_last_v1
Optimization: im_mseg_adamw_poly_v1
"""

from training.optimizer import (
    OPTIMIZATION_PROTOCOL,
    build_optimizer,
    build_scheduler,
    compute_gradient_norm,
    calculate_total_updates,
    validate_optimization_checkpoint,
)

from training.checkpoint import (
    capture_rng_state,
    restore_rng_state,
    safe_atomic_save,
    atomic_save_checkpoint,
    validate_checkpoint_payload,
    save_checkpoint,
    load_checkpoint,
    resume_checkpoint,
    resume_training_state,
)

__all__ = [
    # Optimizer & Scheduler
    "OPTIMIZATION_PROTOCOL",
    "build_optimizer",
    "build_scheduler",
    "compute_gradient_norm",
    "calculate_total_updates",
    "validate_optimization_checkpoint",
    # Checkpointing
    "capture_rng_state",
    "restore_rng_state",
    "safe_atomic_save",
    "atomic_save_checkpoint",
    "validate_checkpoint_payload",
    "save_checkpoint",
    "load_checkpoint",
    "resume_checkpoint",
    "resume_training_state",
]
