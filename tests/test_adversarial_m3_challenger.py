"""
tests/test_adversarial_m3_challenger.py
=======================================
Adversarial stress-test suite for Milestone 3 (Optimization & Schedule Invariants).
Author: m3_challenger_1 (Empirical Challenger)

Challenge Dimensions:
1. Exact LR formula parity and boundary conditions (steps 0, 1, 2, 100, 18600, 37199, 37200, 37201, 37202).
2. Weight decay uniformity across all model parameters (weights, biases, norms).
3. AMP GradScaler gradient overflow and update-skipping dynamics.
4. Discovery and empirical verification of train.py's compute_gradient_norm crash on AMP FP16 overflow.
5. Checkpoint schema validation and lossless resumption invariants.
"""

from __future__ import annotations
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict

import numpy as np
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from network.model import MyoPSNetL
from training.checkpoint import (
    atomic_save_checkpoint,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
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


class TestLRParityAndBoundaryAdversarial:
    """Challenge 1: Stress-test polynomial LR decay schedule across critical boundary conditions."""

    def test_lr_boundary_conditions_and_exact_parity(self):
        """Test scheduler across steps 0, 1, 2, 100, 18600, 37199, 37200, 37201, 37202."""
        base_lr = 0.0003333
        total_updates = 37200
        power = 0.9

        model = nn.Linear(4, 2)
        optimizer = build_optimizer(model, base_lr=base_lr)
        scheduler = build_scheduler(optimizer, total_updates=total_updates, power=power)

        # Step 0 (initial before any updates)
        lr_0 = optimizer.param_groups[0]["lr"]
        assert lr_0 == base_lr, f"Step 0 LR mismatch: {lr_0} != {base_lr}"
        assert math.isclose(lr_0, 0.0003333, rel_tol=1e-12)

        # Step 1 (after 1st scheduler.step)
        scheduler.step()
        lr_1 = optimizer.param_groups[0]["lr"]
        assert lr_1 == base_lr, f"Step 1 LR mismatch: {lr_1} != {base_lr}"
        assert math.isclose(lr_1, 0.0003333, rel_tol=1e-12)

        # Step 2 (after 2nd scheduler.step)
        scheduler.step()
        lr_2 = optimizer.param_groups[0]["lr"]
        expected_lr_2 = base_lr * ((1.0 - 1.0 / total_updates) ** power)
        assert math.isclose(lr_2, expected_lr_2, rel_tol=1e-12)
        # Formatted to 7 decimal places (standard logging precision), step 2 rounds to 0.0003333
        assert f"{lr_2:.7f}" == "0.0003333"

        # Verify that Update 1 and Update 2 in training both execute with base_lr
        # Emulate optimizer step 1 and step 2
        opt_test = build_optimizer(model, base_lr=base_lr)
        sched_test = build_scheduler(opt_test, total_updates=total_updates, power=power)
        # Update 1 uses opt_test LR:
        lr_used_update_1 = opt_test.param_groups[0]["lr"]
        opt_test.step()
        sched_test.step()
        # Update 2 uses opt_test LR:
        lr_used_update_2 = opt_test.param_groups[0]["lr"]
        opt_test.step()
        sched_test.step()
        assert lr_used_update_1 == base_lr
        assert lr_used_update_2 == base_lr

        # Test Step 100
        sched_100 = build_scheduler(optimizer, total_updates=total_updates, power=power)
        sched_100.last_epoch = 99
        sched_100.step()
        lr_100 = optimizer.param_groups[0]["lr"]
        expected_lr_100 = base_lr * (max(0.0, 1.0 - (100 - 1) / total_updates) ** power)
        assert math.isclose(lr_100, expected_lr_100, rel_tol=1e-12)
        assert math.isclose(lr_100, 0.0003325015864118, rel_tol=1e-7)

        # Test Step 18600 (epoch 150 midpoint)
        sched_mid = build_scheduler(optimizer, total_updates=total_updates, power=power)
        sched_mid.last_epoch = 18599
        sched_mid.step()
        lr_18600 = optimizer.param_groups[0]["lr"]
        expected_lr_18600 = base_lr * (max(0.0, 1.0 - (18600 - 1) / total_updates) ** power)
        assert math.isclose(lr_18600, expected_lr_18600, rel_tol=1e-12)
        assert math.isclose(lr_18600, 0.0001786196899785, rel_tol=1e-7)

        # Test Step 37199 (penultimate update)
        sched_pen = build_scheduler(optimizer, total_updates=total_updates, power=power)
        sched_pen.last_epoch = 37198
        sched_pen.step()
        lr_37199 = optimizer.param_groups[0]["lr"]
        expected_lr_37199 = base_lr * (max(0.0, 1.0 - (37199 - 1) / total_updates) ** power)
        assert math.isclose(lr_37199, expected_lr_37199, rel_tol=1e-12)
        assert math.isclose(lr_37199, 4.789317886216e-08, rel_tol=1e-6)

        # Test Step 37200 (final planned update)
        sched_final = build_scheduler(optimizer, total_updates=total_updates, power=power)
        sched_final.last_epoch = 37199
        sched_final.step()
        lr_37200 = optimizer.param_groups[0]["lr"]
        expected_lr_37200 = base_lr * ((1.0 / total_updates) ** power)
        assert math.isclose(lr_37200, expected_lr_37200, rel_tol=1e-12)
        assert lr_37200 > 0.0, "LR at step 37200 must be strictly positive!"
        assert math.isclose(lr_37200, 2.566531907049e-08, rel_tol=1e-6)

        # Test Step 37201 (over budget: 1.0 - 37200/37200 = 0.0)
        sched_final.step()
        lr_37201 = optimizer.param_groups[0]["lr"]
        assert lr_37201 == 0.0, f"LR at step 37201 must be 0.0, got {lr_37201}"

        # Test Step 37202 (over budget: clamped to 0.0, non-negative)
        sched_final.step()
        lr_37202 = optimizer.param_groups[0]["lr"]
        assert lr_37202 == 0.0, f"LR at step 37202 must be 0.0, got {lr_37202}"
        assert lr_37202 >= 0.0, "LR must never be negative!"

    def test_lr_strict_monotonicity_full_profile(self):
        """Verify strict monotonic decay from step 1 to 37200."""
        model = nn.Linear(2, 2)
        optimizer = build_optimizer(model, base_lr=0.0003333)
        scheduler = build_scheduler(optimizer, total_updates=37200)

        # Sample across range: 1, 10, 100, 1000, 5000, 10000, 20000, 30000, 37200
        probe_steps = [1, 2, 10, 100, 1000, 5000, 10000, 20000, 30000, 37199, 37200]
        lrs = []
        for s in probe_steps:
            sched = build_scheduler(optimizer, total_updates=37200)
            sched.last_epoch = s - 1
            sched.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1], f"Monotonicity violated between steps {probe_steps[i]} and {probe_steps[i+1]}"
            if probe_steps[i] >= 1:
                assert lrs[i] > lrs[i + 1], f"Strict decay violated between steps {probe_steps[i]} and {probe_steps[i+1]}"


class TestWeightDecayUniformityAdversarial:
    """Challenge 2: Stress-test weight decay uniformity on MyoPSNetL."""

    def test_all_parameters_in_single_param_group_with_exact_weight_decay(self):
        """Verify 100% of parameters (including all biases and norms) receive weight_decay=0.0001."""
        model = MyoPSNetL()
        base_lr = 0.0003333
        weight_decay = 0.0001
        optimizer = build_optimizer(model, base_lr=base_lr, weight_decay=weight_decay)

        # Must have exactly 1 parameter group
        assert len(optimizer.param_groups) == 1, f"Expected 1 group, got {len(optimizer.param_groups)}"
        group = optimizer.param_groups[0]

        # Verify hyperparameter values in group
        assert group["lr"] == base_lr
        assert group["weight_decay"] == weight_decay
        assert group["betas"] == (0.9, 0.999)
        assert group["eps"] == 1e-8

        # Verify param count and identity
        model_params = list(model.parameters())
        group_params = group["params"]
        assert len(model_params) == len(group_params), (
            f"Param count mismatch: model has {len(model_params)}, optimizer has {len(group_params)}"
        )
        assert len(model_params) == 270

        group_id_set = {id(p) for p in group_params}
        bias_count = 0
        weight_count = 0

        for name, param in model.named_parameters():
            assert id(param) in group_id_set, f"Parameter {name} omitted from optimizer!"
            if "bias" in name:
                bias_count += 1
            else:
                weight_count += 1

        assert bias_count == 135
        assert weight_count == 135
        assert bias_count + weight_count == 270


class TestAMPGradScalerSkipAdversarial:
    """Challenge 3 & 4: Stress-test AMP GradScaler update-skipping and train.py crash vulnerability."""

    def test_gradscaler_step_skip_on_overflow_isolated(self):
        """Verify that when GradScaler detects overflow, scaler.step skips optimizer update and scheduler doesn't step."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            pytest.skip("CUDA required for torch.amp.GradScaler('cuda')")

        model = nn.Linear(4, 2).to(device)
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer, total_updates=100)
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        initial_scale = scaler.get_scale()

        # Step 1: Normal step
        x = torch.randn(2, 4, device=device)
        loss = model(x).sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        skipped = (scaler.get_scale() < old_scale)
        assert not skipped
        scheduler.step()
        assert scheduler.last_epoch == 1
        optimizer.zero_grad(set_to_none=True)

        # Step 2: Inject simulated FP16 overflow (Inf in gradient)
        x = torch.randn(2, 4, device=device)
        loss = model(x).sum()
        scaler.scale(loss).backward()
        # Inject non-finite gradient
        model.weight.grad[0, 0] = float("inf")

        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        # Scale should be halved
        new_scale = scaler.get_scale()
        assert new_scale < old_scale, f"Scale should have reduced: {new_scale} vs {old_scale}"
        skipped = (new_scale < old_scale)
        assert skipped, "Update should be flagged as skipped!"

        # Scheduler step MUST be skipped
        if not skipped:
            scheduler.step()
        assert scheduler.last_epoch == 1, (
            f"Scheduler step counter should remain 1 after skipped update, got {scheduler.last_epoch}"
        )
        optimizer.zero_grad(set_to_none=True)

        # Step 3: Normal step after recovery
        x = torch.randn(2, 4, device=device)
        loss = model(x).sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        skipped = (scaler.get_scale() < old_scale)
        assert not skipped
        scheduler.step()
        assert scheduler.last_epoch == 2, f"Scheduler should now be at step 2, got {scheduler.last_epoch}"

    def test_reproduce_train_py_compute_gradient_norm_crash_on_amp_overflow(self):
        """Empirically reproduce the bug in train.py: compute_gradient_norm crashes before scaler.step on overflow."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            pytest.skip("CUDA required for torch.amp.GradScaler('cuda')")

        model = nn.Linear(4, 2).to(device)
        optimizer = build_optimizer(model)
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        x = torch.randn(2, 4, device=device)
        loss = model(x).sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Simulate FP16 gradient overflow
        model.weight.grad[0, 0] = float("inf")

        # In train.py line 331, compute_gradient_norm is called before scaler.step:
        # norm = compute_gradient_norm(model, clip_grad=0.0)
        # This raises FloatingPointError when error_if_nonfinite=True (default), preventing scaler.step(optimizer) and scaler.update()!
        with pytest.raises(FloatingPointError, match="Non-finite gradient norm detected: inf"):
            compute_gradient_norm(model, clip_grad=0.0, error_if_nonfinite=True)

    def test_amp_gradient_overflow_does_not_crash_and_skips_step(self):
        """Verify that when error_if_nonfinite=False (AMP enabled), non-finite gradient does NOT crash,
        triggers scale reduction, and skips optimizer/scheduler updates for both Inf and NaN."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            pytest.skip("CUDA required for torch.amp.GradScaler('cuda')")

        for bad_val, val_name in [(float("inf"), "inf"), (float("nan"), "nan")]:
            model = nn.Linear(4, 2).to(device)
            optimizer = build_optimizer(model)
            scheduler = build_scheduler(optimizer, total_updates=100)
            scaler = torch.amp.GradScaler("cuda", enabled=True)

            w_before = model.weight.clone()
            initial_scale = scaler.get_scale()

            x = torch.randn(2, 4, device=device)
            loss = model(x).sum()
            scaler.scale(loss).backward()

            # Inject simulated overflow into gradient before unscale_
            model.weight.grad[0, 0] = bad_val

            # Call unscale_ and compute_gradient_norm matching train.py:330-335
            scaler.unscale_(optimizer)
            norm = compute_gradient_norm(
                model,
                clip_grad=0.0,
                error_if_nonfinite=not scaler.is_enabled(),
            )
            assert not math.isfinite(norm), f"Computed norm should be non-finite for {val_name}, got {norm}"

            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            # Verify scale reduction
            new_scale = scaler.get_scale()
            skipped = (new_scale < old_scale)
            assert skipped, f"Step was not skipped for {val_name}! Old scale: {old_scale}, New scale: {new_scale}"
            assert new_scale == old_scale * 0.5, f"Scale should be halved: {new_scale} vs {old_scale * 0.5}"

            # Verify optimizer step skipped (weights unchanged)
            w_after = model.weight.clone()
            assert torch.equal(w_before, w_after), f"Weights were modified during skipped step for {val_name}!"

            # Verify scheduler step skipped
            if not skipped:
                scheduler.step()
            assert scheduler.last_epoch == 0, f"Scheduler advanced on skipped step for {val_name}!"

    def test_amp_gradient_overflow_with_clipping_enabled(self):
        """Verify that gradient clipping (clip_grad > 0.0) with non-finite gradients under AMP
        does not crash and correctly triggers update skipping."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            pytest.skip("CUDA required for torch.amp.GradScaler('cuda')")

        model = nn.Linear(4, 2).to(device)
        optimizer = build_optimizer(model)
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        w_before = model.weight.clone()
        x = torch.randn(2, 4, device=device)
        loss = model(x).sum()
        scaler.scale(loss).backward()
        model.weight.grad[0, 0] = float("inf")

        scaler.unscale_(optimizer)
        norm = compute_gradient_norm(model, clip_grad=1.0, error_if_nonfinite=False)
        assert not math.isfinite(norm)

        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        assert scaler.get_scale() < old_scale
        assert torch.equal(w_before, model.weight)

    def test_amp_gradient_overflow_multi_step_recovery_myopsnet(self):
        """End-to-end 3-step test on full MyoPSNetL architecture: clean step -> overflow step -> recovery step."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            pytest.skip("CUDA required for torch.amp.GradScaler('cuda')")

        model = MyoPSNetL().to(device)
        optimizer = build_optimizer(model, base_lr=0.0003333)
        scheduler = build_scheduler(optimizer, total_updates=100)
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        global_step = 0
        amp_skipped_updates = 0

        def run_step(inject_overflow: bool = False):
            nonlocal global_step, amp_skipped_updates
            c0 = torch.randn(2, 1, 128, 128, device=device)
            lge = torch.randn(2, 1, 128, 128, device=device)
            t2 = torch.randn(2, 1, 128, 128, device=device)

            with torch.autocast(device_type=device, dtype=torch.float16):
                s_c0, s_lge, s_t2 = model(c0, lge, t2)
                loss = s_c0.sum() + s_lge.sum() + s_t2.sum()

            scaler.scale(loss).backward()
            if inject_overflow:
                list(model.parameters())[0].grad.view(-1)[0] = float("inf")

            scaler.unscale_(optimizer)
            norm = compute_gradient_norm(model, clip_grad=0.0, error_if_nonfinite=not scaler.is_enabled())

            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            skipped = (scaler.get_scale() < old_scale)
            if not skipped:
                scheduler.step()
                global_step += 1
            else:
                amp_skipped_updates += 1

            optimizer.zero_grad(set_to_none=True)
            return skipped, norm

        # Step 1: Clean update
        skipped_1, _ = run_step(inject_overflow=False)
        assert not skipped_1
        assert global_step == 1
        assert amp_skipped_updates == 0
        w_after_step1 = list(model.parameters())[0].clone()

        # Step 2: Injected FP16 overflow
        skipped_2, norm_2 = run_step(inject_overflow=True)
        assert skipped_2
        assert not math.isfinite(norm_2)
        assert global_step == 1, "global_step should not increment on skipped update"
        assert amp_skipped_updates == 1
        assert scaler.get_scale() == 32768.0
        w_after_step2 = list(model.parameters())[0].clone()
        assert torch.equal(w_after_step1, w_after_step2), "Weights changed during skipped overflow step!"

        # Step 3: Clean recovery update
        skipped_3, _ = run_step(inject_overflow=False)
        assert not skipped_3
        assert global_step == 2
        assert amp_skipped_updates == 1
        w_after_step3 = list(model.parameters())[0].clone()
        assert not torch.equal(w_after_step2, w_after_step3), "Weights failed to update on recovery step!"

        # Verify all parameters remain finite
        for name, p in model.named_parameters():
            assert torch.all(torch.isfinite(p)), f"Parameter {name} has non-finite values after recovery!"



class TestCheckpointResumptionInvariantsAdversarial:
    """Challenge 5: Stress-test checkpoint resumption fidelity and rejection of corrupted states."""

    def test_checkpoint_rejection_of_corrupted_payloads(self, tmp_path):
        """Verify validate_checkpoint_payload rejects altered protocol IDs and counters."""
        model = nn.Linear(2, 2)
        opt = build_optimizer(model)
        sched = build_scheduler(opt)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        valid_payload = {
            "protocol_id": "myops380_train304_test76_last_v1",
            "opt_protocol": "im_mseg_adamw_poly_v1",
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
            "rng_state": capture_rng_state(),
        }

        # Should pass
        validate_checkpoint_payload(valid_payload)

        # 1. Tampered protocol_id
        bad_p = dict(valid_payload, protocol_id="custom_protocol")
        with pytest.raises(ValueError, match="Invalid protocol_id"):
            validate_checkpoint_payload(bad_p)

        # 2. Tampered opt_protocol
        bad_opt = dict(valid_payload, opt_protocol="sgd_cosine")
        with pytest.raises(ValueError, match="Invalid opt_protocol"):
            validate_checkpoint_payload(bad_opt)

        # 3. Mismatched epoch counters
        bad_counters = dict(valid_payload, completed_epochs=9)
        with pytest.raises(ValueError, match="Mismatched epoch counters"):
            validate_checkpoint_payload(bad_counters)

        # 4. Inconsistent training_complete flag
        bad_flag = dict(valid_payload, training_complete=True)  # 10 < 300, so complete must be False
        with pytest.raises(ValueError, match="Inconsistent training_complete flag"):
            validate_checkpoint_payload(bad_flag)

    def test_resumption_restores_exact_optimizer_lr_and_scheduler_step(self, tmp_path):
        """Verify resuming from checkpoint restores exact decayed LR and scheduler step."""
        model = nn.Linear(4, 2)
        opt = build_optimizer(model, base_lr=0.0003333)
        sched = build_scheduler(opt, total_updates=37200)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        # Advance 500 steps
        for _ in range(500):
            sched.step()

        expected_lr = opt.param_groups[0]["lr"]
        expected_step = 500

        ckpt_path = tmp_path / "ckpt_test.pt"
        save_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            epoch=4,
            completed_epochs=4,
            planned_epochs=300,
            global_step=expected_step,
            total_updates=37200,
            seed=42,
        )

        # Instantiate fresh objects
        fresh_model = nn.Linear(4, 2)
        fresh_opt = build_optimizer(fresh_model, base_lr=0.0003333)
        fresh_sched = build_scheduler(fresh_opt, total_updates=37200)
        fresh_scaler = torch.amp.GradScaler("cuda", enabled=False)

        res = resume_training_state(
            checkpoint_path=ckpt_path,
            model=fresh_model,
            optimizer=fresh_opt,
            scheduler=fresh_sched,
            scaler=fresh_scaler,
        )

        assert res["status"] == "resumed"
        assert res["completed_epochs"] == 4
        assert res["global_step"] == 500
        assert fresh_sched.last_epoch == 500
        assert math.isclose(fresh_opt.param_groups[0]["lr"], expected_lr, rel_tol=1e-12)
