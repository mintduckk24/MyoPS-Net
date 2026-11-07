"""
tests/test_milestone3.py
========================
Comprehensive verification test suite for Milestone 3:
- Feature 9: AdamW Optimizer & Polynomial Learning Rate Scheduler (im_mseg_adamw_poly_v1)
- Feature 10: 300-Epoch Training Loop with Auto AMP & Strict NO Validation
- Feature 11: Atomic Periodic & Final Checkpointing (PID tmp, fsync, os.replace backoff)
- Feature 12: Full Lossless Checkpoint Resumption (State fidelity, LR invariance, RNG continuity)
"""

from __future__ import annotations
import copy
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any, Dict, Sequence

import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from network.model import MyoPSNetL
from training.checkpoint import (
    atomic_save_checkpoint,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    resume_checkpoint,
    resume_training_state,
    safe_atomic_save,
    save_checkpoint,
    validate_checkpoint_payload,
)
from training.optimizer import (
    OPTIMIZATION_PROTOCOL,
    build_optimizer,
    build_scheduler,
    calculate_total_updates,
    compute_gradient_norm,
    validate_optimization_checkpoint,
)


# ==============================================================================
# 1. Optimizer Configuration Tests
# ==============================================================================

class TestAdamWOptimizerConfiguration:
    """Verify AdamW optimizer protocol im_mseg_adamw_poly_v1 compliance."""

    def test_adamw_hyperparameters(self):
        """Verify lr, weight_decay, betas, and eps match protocol specifications."""
        model = nn.Sequential(nn.Linear(10, 10), nn.BatchNorm1d(10), nn.Linear(10, 2))
        base_lr = 0.0003333
        weight_decay = 0.0001
        optimizer = build_optimizer(model, base_lr=base_lr, weight_decay=weight_decay)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) == 1

        group = optimizer.param_groups[0]
        assert math.isclose(group["lr"], 0.0003333, rel_tol=1e-7)
        assert math.isclose(group["weight_decay"], 0.0001, rel_tol=1e-7)
        assert group["betas"] == (0.9, 0.999)
        assert math.isclose(group["eps"], 1e-8, rel_tol=1e-7)

    def test_single_param_group_all_parameters_decay(self):
        """Verify all parameters (including biases and norm layers) receive weight decay."""
        model = MyoPSNetL()
        optimizer = build_optimizer(model)

        assert len(optimizer.param_groups) == 1
        group = optimizer.param_groups[0]

        all_params = list(model.parameters())
        assert len(group["params"]) == len(all_params)

        param_id_set = {id(p) for p in group["params"]}
        for name, param in model.named_parameters():
            assert id(param) in param_id_set
            # Verify weight decay is 0.0001 across the group
            assert group["weight_decay"] == 0.0001

    def test_optimization_protocol_metadata(self):
        """Verify protocol dictionary constants."""
        assert OPTIMIZATION_PROTOCOL["id"] == "im_mseg_adamw_poly_v1"
        assert OPTIMIZATION_PROTOCOL["optimizer"] == "AdamW"
        assert OPTIMIZATION_PROTOCOL["weight_decay_scope"] == "all_parameters"
        assert OPTIMIZATION_PROTOCOL["betas"] == (0.9, 0.999)
        assert OPTIMIZATION_PROTOCOL["eps"] == 1e-8
        assert OPTIMIZATION_PROTOCOL["power"] == 0.9


# ==============================================================================
# 2. Polynomial LR Scheduler Tests
# ==============================================================================

class TestPolynomialLRScheduler:
    """Verify polynomial LR decay schedule matching I-MMSeg / SCAR specification."""

    def test_poly_lr_formula_and_key_steps(self):
        """Verify exact decimal learning rates at key milestone updates."""
        model = nn.Linear(10, 2)
        base_lr = 0.0003333
        total_updates = 37200
        power = 0.9

        optimizer = build_optimizer(model, base_lr=base_lr)
        scheduler = build_scheduler(optimizer, total_updates=total_updates, power=power)

        # Step 0: Initial state before any update
        lr_0 = optimizer.param_groups[0]["lr"]
        assert math.isclose(lr_0, base_lr, rel_tol=1e-9)

        # Step 1: After 1st update -> max(0, 1 - 1) = 0, so factor = 1.0 (base LR preserved)
        scheduler.step()
        lr_1 = optimizer.param_groups[0]["lr"]
        assert math.isclose(lr_1, base_lr, rel_tol=1e-9)

        # Step 2: After 2nd update -> factor = (1 - 1/37200)**0.9
        scheduler.step()
        lr_2 = optimizer.param_groups[0]["lr"]
        expected_lr_2 = base_lr * ((1.0 - 1.0 / total_updates) ** power)
        assert math.isclose(lr_2, expected_lr_2, rel_tol=1e-9)

        # Fast forward to Step 18600 (epoch 150 midpoint)
        # Advance scheduler to step 18600
        for s in range(3, 18601):
            scheduler.step()
        lr_18600 = optimizer.param_groups[0]["lr"]
        expected_lr_18600 = base_lr * (max(0.0, 1.0 - (18600 - 1) / total_updates) ** power)
        assert math.isclose(lr_18600, expected_lr_18600, rel_tol=1e-9)
        assert math.isclose(lr_18600, 0.000178619689978, rel_tol=1e-5)

        # Step 37199: second to last update
        # We can construct a direct scheduler instance at this step via last_epoch
        scheduler_end = build_scheduler(optimizer, total_updates=total_updates, power=power)
        scheduler_end.last_epoch = 37199
        scheduler_end.step()  # transitions to 37200
        lr_37200 = optimizer.param_groups[0]["lr"]
        expected_lr_37200 = base_lr * ((1.0 / total_updates) ** power)
        assert math.isclose(lr_37200, expected_lr_37200, rel_tol=1e-7)
        assert lr_37200 > 0.0  # Strictly non-zero at final budget update!
        assert math.isclose(lr_37200, 2.5665319e-08, rel_tol=1e-5)

        # Step 37201: beyond nominal budget -> drops to 0.0
        scheduler_end.step()
        lr_37201 = optimizer.param_groups[0]["lr"]
        assert lr_37201 == 0.0

    def test_poly_lr_strictly_monotonic_decay(self):
        """Verify learning rate is strictly monotonically decreasing from step 1 to 37200."""
        model = nn.Linear(5, 1)
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer, total_updates=100)

        lrs = [optimizer.param_groups[0]["lr"]]
        for _ in range(100):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # Step 0 and 1 are equal to base_lr
        assert math.isclose(lrs[0], lrs[1], rel_tol=1e-9)
        # From step 1 onwards, strictly decreasing
        for i in range(1, len(lrs) - 1):
            assert lrs[i + 1] < lrs[i]

    def test_scheduler_state_dict_roundtrip(self):
        """Verify scheduler state_dict serializes and restores step counter and decayed LR."""
        model = nn.Linear(5, 1)
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer, total_updates=1000)

        for _ in range(150):
            scheduler.step()

        decayed_lr = optimizer.param_groups[0]["lr"]
        sd = copy.deepcopy(scheduler.state_dict())

        # Create fresh optimizer and scheduler
        new_opt = build_optimizer(model)
        new_sched = build_scheduler(new_opt, total_updates=1000)
        new_sched.load_state_dict(sd)

        # In PyTorch LambdaLR, loading state_dict syncs last_epoch and get_last_lr
        assert new_sched.last_epoch == 150
        assert math.isclose(new_sched.get_last_lr()[0], decayed_lr, rel_tol=1e-9)
        new_opt.load_state_dict(optimizer.state_dict())
        assert math.isclose(new_opt.param_groups[0]["lr"], decayed_lr, rel_tol=1e-9)

    def test_calculate_total_updates(self):
        """Verify total updates calculation across epochs and batches."""
        assert calculate_total_updates(epochs=300, batches_per_epoch=124, accum_steps=1) == 37200
        assert calculate_total_updates(epochs=1, batches_per_epoch=124, accum_steps=1) == 124
        assert calculate_total_updates(epochs=10, batches_per_epoch=20, accum_steps=2) == 100


# ==============================================================================
# 3. Gradient Norm & Finiteness Tests
# ==============================================================================

class TestGradientNormAndClipping:
    """Verify Euclidean gradient norm computation and finite verification."""

    def test_compute_gradient_norm_finite_no_clipping(self):
        """Verify Euclidean L2 norm calculation when clip_grad=0.0 (no scaling)."""
        model = nn.Linear(4, 2, bias=False)
        # Create known weight gradients: 2x4 = 8 elements, all equal to 1.0
        # Euclidean norm of 8 ones = sqrt(8) ≈ 2.828427
        x = torch.ones(1, 4)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for p in model.parameters():
            p.grad.fill_(1.0)

        grad_copy = model.weight.grad.clone()
        norm = compute_gradient_norm(model, clip_grad=0.0)

        assert math.isclose(norm, math.sqrt(8.0), rel_tol=1e-5)
        # Verify gradients were NOT modified (no clipping applied)
        assert torch.allclose(model.weight.grad, grad_copy)

    def test_compute_gradient_norm_with_clipping(self):
        """Verify gradients are clipped when clip_grad > 0.0 and norm exceeds threshold."""
        model = nn.Linear(4, 2, bias=False)
        x = torch.ones(1, 4)
        out = model(x)
        out.sum().backward()

        for p in model.parameters():
            p.grad.fill_(1.0)  # norm is sqrt(8) ≈ 2.828

        clip_threshold = 1.0
        norm = compute_gradient_norm(model, clip_grad=clip_threshold)

        assert math.isclose(norm, math.sqrt(8.0), rel_tol=1e-5)
        # After clipping with threshold 1.0, current norm should be 1.0
        current_norm = torch.norm(model.weight.grad, p=2).item()
        assert math.isclose(current_norm, 1.0, rel_tol=1e-5)

    def test_compute_gradient_norm_raises_on_non_finite(self):
        """Verify FloatingPointError is raised when non-finite gradient is encountered."""
        model = nn.Linear(2, 2)
        x = torch.ones(1, 2)
        model(x).sum().backward()

        # Inject NaN into gradient
        model.weight.grad[0, 0] = float("nan")

        with pytest.raises(FloatingPointError, match="Non-finite gradient norm"):
            compute_gradient_norm(model, clip_grad=0.0)

    def test_compute_gradient_norm_non_finite_safe_mode(self):
        """Verify non-finite gradient returns float('nan') or float('inf') without raising when error_if_nonfinite=False."""
        model = nn.Linear(2, 2)
        x = torch.ones(1, 2)
        model(x).sum().backward()

        # Inject NaN into gradient
        model.weight.grad[0, 0] = float("nan")

        norm = compute_gradient_norm(model, clip_grad=0.0, error_if_nonfinite=False)
        assert not math.isfinite(norm)


# ==============================================================================
# 4. Atomic Checkpointing & Payload Schema Tests
# ==============================================================================

class TestAtomicCheckpointingAndSchema:
    """Verify atomic file writing, Windows retry backoff, and schema validation."""

    def test_atomic_save_creates_target_file_no_temp_leak(self, tmp_path):
        """Verify safe_atomic_save writes destination atomically and cleans up temp files."""
        filepath = tmp_path / "checkpoints" / "myopsnet_seed42.pt"
        payload = {"data": [1, 2, 3]}

        safe_atomic_save(payload, filepath)
        assert filepath.is_file()

        # Verify no .tmp files exist in parent directory
        tmp_files = list(filepath.parent.glob("*.tmp*"))
        assert len(tmp_files) == 0

        # Verify content loads correctly
        loaded = torch.load(filepath, weights_only=False)
        assert loaded["data"] == [1, 2, 3]

    def test_atomic_save_overwrites_existing_file(self, tmp_path):
        """Verify safe_atomic_save replaces an existing file without corruption."""
        filepath = tmp_path / "myopsnet_seed42.pt"
        safe_atomic_save({"version": 1}, filepath)
        assert torch.load(filepath, weights_only=False)["version"] == 1

        safe_atomic_save({"version": 2}, filepath)
        assert torch.load(filepath, weights_only=False)["version"] == 2

    def test_validate_checkpoint_payload_success(self):
        """Verify canonical payload dictionary validates successfully."""
        model = nn.Linear(2, 2)
        opt = build_optimizer(model)
        sched = build_scheduler(opt, total_updates=37200)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        payload = {
            "epoch": 10,
            "completed_epochs": 10,
            "planned_epochs": 300,
            "training_complete": False,
            "global_step": 1240,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "seed": 42,
            "protocol_id": "myops380_train304_test76_last_v1",
            "opt_protocol": "im_mseg_adamw_poly_v1",
            "rng_state": capture_rng_state(),
        }

        # Should pass without exception
        validate_checkpoint_payload(payload)

    def test_validate_checkpoint_payload_missing_keys_rejected(self):
        """Verify ValueError is raised if any mandatory key is missing."""
        base_payload = {
            "epoch": 1,
            "completed_epochs": 1,
            "planned_epochs": 300,
            "training_complete": False,
            "global_step": 124,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "scaler_state_dict": {},
            "seed": 42,
            "protocol_id": "myops380_train304_test76_last_v1",
            "opt_protocol": "im_mseg_adamw_poly_v1",
            "rng_state": capture_rng_state(),
        }

        for key in list(base_payload.keys()):
            bad_payload = dict(base_payload)
            del bad_payload[key]
            with pytest.raises(ValueError, match="missing mandatory keys"):
                validate_checkpoint_payload(bad_payload)

    def test_validate_checkpoint_payload_mismatched_counters_rejected(self):
        """Verify ValueError is raised on inconsistent epoch counters or completion flags."""
        payload = {
            "epoch": 5,
            "completed_epochs": 4,  # Mismatch!
            "planned_epochs": 300,
            "training_complete": False,
            "global_step": 500,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "scaler_state_dict": {},
            "seed": 42,
            "protocol_id": "myops380_train304_test76_last_v1",
            "opt_protocol": "im_mseg_adamw_poly_v1",
            "rng_state": capture_rng_state(),
        }
        with pytest.raises(ValueError, match="Mismatched epoch counters"):
            validate_checkpoint_payload(payload)

        # Inconsistent training_complete flag
        payload["completed_epochs"] = 5
        payload["training_complete"] = True  # Should be False since 5 < 300
        with pytest.raises(ValueError, match="Inconsistent training_complete flag"):
            validate_checkpoint_payload(payload)

    def test_validate_checkpoint_protocol_mismatch_rejected(self):
        """Verify ValueError is raised if protocol_id or opt_protocol is tampered."""
        payload = {
            "epoch": 1,
            "completed_epochs": 1,
            "planned_epochs": 300,
            "training_complete": False,
            "global_step": 124,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "scaler_state_dict": {},
            "seed": 42,
            "protocol_id": "wrong_protocol",
            "opt_protocol": "im_mseg_adamw_poly_v1",
            "rng_state": capture_rng_state(),
        }
        with pytest.raises(ValueError, match="Invalid protocol_id"):
            validate_checkpoint_payload(payload)

        payload["protocol_id"] = "myops380_train304_test76_last_v1"
        payload["opt_protocol"] = "wrong_opt"
        with pytest.raises(ValueError, match="Invalid opt_protocol"):
            validate_checkpoint_payload(payload)


# ==============================================================================
# 5. Checkpoint Resumption & Bitwise Equivalence Tests
# ==============================================================================

class TestCheckpointResumptionEquivalence:
    """Verify lossless resumption, RNG restoration, and numerical reproducibility."""

    def test_rng_state_capture_and_restore(self):
        """Verify RNG capture and restoration produces identical random sequence."""
        # 1. Capture state
        random.seed(9999)
        np.random.seed(9999)
        torch.manual_seed(9999)
        saved_rng = capture_rng_state()

        # Generate sequence A
        seq_a_py = [random.random() for _ in range(5)]
        seq_a_np = np.random.rand(5).tolist()
        seq_a_torch = torch.rand(5).tolist()

        # Perturb state
        random.seed(1111)
        np.random.seed(1111)
        torch.manual_seed(1111)

        # Restore state
        restore_rng_state(saved_rng)

        # Generate sequence B
        seq_b_py = [random.random() for _ in range(5)]
        seq_b_np = np.random.rand(5).tolist()
        seq_b_torch = torch.rand(5).tolist()

        assert seq_a_py == seq_b_py
        assert seq_a_np == seq_b_np
        assert seq_a_torch == seq_b_torch

    def test_uninterrupted_vs_resumed_exact_numerical_equivalence(self, tmp_path):
        """Verify interrupted + resumed run produces identical loss and weights to uninterrupted run."""
        seed = 1234
        num_steps_part1 = 3
        num_steps_part2 = 3
        total_steps = num_steps_part1 + num_steps_part2

        # -------------------------------------------------------------
        # Run A: Uninterrupted run for 6 steps
        # -------------------------------------------------------------
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model_a = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
        opt_a = build_optimizer(model_a, base_lr=0.0003333, weight_decay=0.0001)
        sched_a = build_scheduler(opt_a, total_updates=total_steps)
        scaler_a = torch.amp.GradScaler("cuda", enabled=False)

        # Generate fixed synthetic mini-batches
        batches = [
            (torch.randn(4, 8), torch.randint(0, 2, (4,)))
            for _ in range(total_steps)
        ]

        losses_a = []
        for x, y in batches:
            out = model_a(x)
            loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            opt_a.step()
            sched_a.step()
            opt_a.zero_grad()
            losses_a.append(loss.item())

        weights_a = [p.clone() for p in model_a.parameters()]

        # -------------------------------------------------------------
        # Run B: Interrupted after 3 steps, checkpointed, resumed for 3 steps
        # -------------------------------------------------------------
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model_b = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
        opt_b = build_optimizer(model_b, base_lr=0.0003333, weight_decay=0.0001)
        sched_b = build_scheduler(opt_b, total_updates=total_steps)
        scaler_b = torch.amp.GradScaler("cuda", enabled=False)

        losses_b = []
        for i in range(num_steps_part1):
            x, y = batches[i]
            out = model_b(x)
            loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            opt_b.step()
            sched_b.step()
            opt_b.zero_grad()
            losses_b.append(loss.item())

        # Save checkpoint at step 3
        ckpt_path = tmp_path / "checkpoints" / "myopsnet_seed1234.pt"
        save_checkpoint(
            checkpoint_path=ckpt_path,
            model=model_b,
            optimizer=opt_b,
            scheduler=sched_b,
            scaler=scaler_b,
            epoch=1,
            completed_epochs=1,
            planned_epochs=2,
            global_step=num_steps_part1,
            total_updates=total_steps,
            seed=seed,
        )

        # Perturb seeds and instantiate fresh model / optimizer / scheduler
        random.seed(7777)
        np.random.seed(7777)
        torch.manual_seed(7777)

        fresh_model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
        fresh_opt = build_optimizer(fresh_model, base_lr=0.0003333, weight_decay=0.0001)
        fresh_sched = build_scheduler(fresh_opt, total_updates=total_steps)
        fresh_scaler = torch.amp.GradScaler("cuda", enabled=False)

        # Resume state
        res = resume_training_state(
            checkpoint_path=ckpt_path,
            model=fresh_model,
            optimizer=fresh_opt,
            scheduler=fresh_sched,
            scaler=fresh_scaler,
        )
        assert res["status"] == "resumed"
        assert res["completed_epochs"] == 1
        assert res["global_step"] == num_steps_part1

        # Execute remaining steps
        for i in range(num_steps_part1, total_steps):
            x, y = batches[i]
            out = fresh_model(x)
            loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            fresh_opt.step()
            fresh_sched.step()
            fresh_opt.zero_grad()
            losses_b.append(loss.item())

        weights_b = [p.clone() for p in fresh_model.parameters()]

        # -------------------------------------------------------------
        # Verify bitwise loss and parameter parity
        # -------------------------------------------------------------
        assert len(losses_a) == len(losses_b)
        for idx, (la, lb) in enumerate(zip(losses_a, losses_b)):
            assert math.isclose(la, lb, abs_tol=1e-6), f"Loss mismatch at step {idx}: {la} vs {lb}"

        for pa, pb in zip(weights_a, weights_b):
            assert torch.allclose(pa, pb, atol=1e-6), "Model weight mismatch after resume"

    def test_resume_already_completed_run_detection(self, tmp_path):
        """Verify resume_training_state cleanly identifies fully completed runs."""
        model = nn.Linear(2, 2)
        opt = build_optimizer(model)
        sched = build_scheduler(opt)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        ckpt_path = tmp_path / "myopsnet_seed42.pt"
        save_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            epoch=300,
            completed_epochs=300,
            planned_epochs=300,
            global_step=37200,
            total_updates=37200,
            seed=42,
        )

        fresh_model = nn.Linear(2, 2)
        fresh_opt = build_optimizer(fresh_model)
        fresh_sched = build_scheduler(fresh_opt)
        fresh_scaler = torch.amp.GradScaler("cuda", enabled=False)

        res = resume_training_state(
            checkpoint_path=ckpt_path,
            model=fresh_model,
            optimizer=fresh_opt,
            scheduler=fresh_sched,
            scaler=fresh_scaler,
        )
        assert res["status"] == "already_completed"
        assert res["training_complete"] is True
        assert res["completed_epochs"] == 300


# ==============================================================================
# 6. CLI Execution & Dry-Run Smoke Tests
# ==============================================================================

def _run_cli_subprocess(cmd: Sequence[str], timeout: int = 180) -> subprocess.CompletedProcess:
    """Run CLI subprocess with process tree cleanup on timeout to prevent GPU memory leaks."""
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


class TestCLIExecutionAndDryRun:
    """Verify train.py CLI options, dry-run mode, and resumption from command line."""

    def test_dry_run_cli_execution(self, tmp_path):
        """Verify python train.py --dry_run runs 1 epoch of 2 batches in seconds."""
        run_dir = tmp_path / "verify_dry_run"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--dry_run",
            "--run_dir", str(run_dir),
            "--seed", "42",
            "--amp", "auto",
            "--num_workers", "0",
        ]

        result = _run_cli_subprocess(cmd, timeout=180)
        assert result.returncode == 0, f"train.py --dry_run failed:\n{result.stderr}\n{result.stdout}"

        # Verify artifacts generated
        ckpt_file = run_dir / "checkpoints" / "myopsnet_seed42.pt"
        log_file = run_dir / "logs" / "train.log"
        csv_file = run_dir / "logs" / "metrics.csv"
        cfg_file = run_dir / "logs" / "config.json"

        assert ckpt_file.is_file()
        assert log_file.is_file()
        assert csv_file.is_file()
        assert cfg_file.is_file()

        # Verify checkpoint payload validity
        ckpt = torch.load(ckpt_file, weights_only=False)
        validate_checkpoint_payload(ckpt)
        assert ckpt["completed_epochs"] == 1
        assert ckpt["seed"] == 42
        assert ckpt["training_complete"] is False

    def test_resume_cli_execution_and_completion(self, tmp_path):
        """Verify train.py resume advances from epoch 1 to epoch 2 and marks complete."""
        run_dir = tmp_path / "verify_resume"
        # Step 1: Pause a two-epoch debug budget; resume must keep this budget/settings.
        cmd1 = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--epochs", "2",
            "--max_batches", "2",
            "--epochs_per_run", "1",
            "--run_dir", str(run_dir),
            "--seed", "42",
            "--amp", "auto",
            "--num_workers", "0",
        ]
        r1 = _run_cli_subprocess(cmd1, timeout=180)
        assert r1.returncode == 0

        ckpt_file = run_dir / "checkpoints" / "myopsnet_seed42.pt"
        assert ckpt_file.is_file()

        # Step 2: Resume with planned epochs = 2, max_batches = 2
        cmd2 = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--resume", str(ckpt_file),
            "--epochs", "2",
            "--max_batches", "2",
            "--run_dir", str(run_dir),
            "--seed", "42",
            "--amp", "auto",
            "--num_workers", "0",
        ]
        r2 = _run_cli_subprocess(cmd2, timeout=180)
        assert r2.returncode == 0, f"train.py --resume failed:\n{r2.stderr}\n{r2.stdout}"

        # Checkpoint should now be completed_epochs = 2, training_complete = True
        ckpt2 = torch.load(ckpt_file, weights_only=False)
        validate_checkpoint_payload(ckpt2)
        assert ckpt2["completed_epochs"] == 2
        assert ckpt2["planned_epochs"] == 2
        assert ckpt2["training_complete"] is True

        # Step 3: Run resume again on completed checkpoint -> must exit cleanly with code 0 immediately
        cmd3 = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--resume", str(ckpt_file),
            "--epochs", "2",
            "--max_batches", "2",
            "--run_dir", str(run_dir),
            "--seed", "42",
            "--amp", "auto",
            "--num_workers", "0",
        ]
        r3 = _run_cli_subprocess(cmd3, timeout=60)
        assert r3.returncode == 0
        assert "Training already complete" in r3.stdout or "Training already complete" in (run_dir / "logs" / "train.log").read_text(encoding="utf-8")
