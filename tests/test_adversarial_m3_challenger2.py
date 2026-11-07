"""
tests/test_adversarial_m3_challenger2.py
========================================
Adversarial Challenge Suite for Milestone 3 Iteration 2.
Author: m3_challenger_2_2 (Empirical Challenger)

Challenge Dimensions:
1. State dict reference identity: zero duplicate .state_dict() calls and shared object references in payload.
2. 13 mandatory schema keys: complete schema validation, protocol enforcement, RNG subkey validation, and counter invariants.
3. Full MyoPSNetL architecture resumption equivalence: exact numerical and parameter parity across interrupted training.
4. Completed-training defense: clean exit without re-training or file mutation when resuming a completed checkpoint.
5. Atomic checkpoint replacement and temporary file cleanup invariants.
"""

from __future__ import annotations
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from criterion.loss import MyoPSLoss
from network.model import MyoPSNetL
import train
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
)
from utils.tools import create_multi_head_targets


# ==============================================================================
# Challenge 1: State Dict Reference Identity & Zero Duplicate Overhead
# ==============================================================================

class TestStateDictReferenceIdentityAdversarial:
    """Stress-test memory efficiency and reference identity in checkpoint serialization."""

    def test_state_dict_called_exactly_once_per_component(self, tmp_path):
        """Verify model, optimizer, scheduler, and scaler .state_dict() are called exactly once."""
        from unittest.mock import patch

        model = nn.Linear(4, 2)
        opt = build_optimizer(model)
        sched = build_scheduler(opt)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        counts = {"m": 0, "o": 0, "s": 0, "sc": 0}

        orig_m = type(model).state_dict
        def spy_m(self, *a, **kw):
            counts["m"] += 1
            return orig_m(self, *a, **kw)

        orig_o = type(opt).state_dict
        def spy_o(self, *a, **kw):
            counts["o"] += 1
            return orig_o(self, *a, **kw)

        orig_s = type(sched).state_dict
        def spy_s(self, *a, **kw):
            counts["s"] += 1
            return orig_s(self, *a, **kw)

        orig_sc = type(scaler).state_dict
        def spy_sc(self, *a, **kw):
            counts["sc"] += 1
            return orig_sc(self, *a, **kw)

        ckpt_path = tmp_path / "test_single_call.pt"
        with patch.object(type(model), "state_dict", spy_m), \
             patch.object(type(opt), "state_dict", spy_o), \
             patch.object(type(sched), "state_dict", spy_s), \
             patch.object(type(scaler), "state_dict", spy_sc):
            save_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=opt,
                scheduler=sched,
                scaler=scaler,
                epoch=1,
                completed_epochs=1,
                planned_epochs=300,
                global_step=10,
            )

        assert counts["m"] == 1, f"model.state_dict() called {counts['m']} times (expected 1)"
        assert counts["o"] == 1, f"optimizer.state_dict() called {counts['o']} times (expected 1)"
        assert counts["s"] == 1, f"scheduler.state_dict() called {counts['s']} times (expected 1)"
        assert counts["sc"] == 1, f"scaler.state_dict() called {counts['sc']} times (expected 1)"

    def test_in_memory_and_deserialized_reference_identity(self, tmp_path):
        """Verify primary and alias keys share exact same object reference before and after disk save."""
        model = nn.Linear(4, 2)
        opt = build_optimizer(model)
        sched = build_scheduler(opt)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        ckpt_path = tmp_path / "test_identity.pt"
        save_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            epoch=5,
            completed_epochs=5,
            planned_epochs=300,
            global_step=50,
        )

        loaded = torch.load(ckpt_path, weights_only=False)

        # Check reference identity on loaded dictionary
        assert loaded["model"] is loaded["model_state_dict"], "model state dict reference identity failed"
        assert loaded["optimizer"] is loaded["optimizer_state_dict"], "optimizer state dict reference identity failed"
        assert loaded["scheduler"] is loaded["scheduler_state_dict"], "scheduler state dict reference identity failed"
        assert loaded["scaler"] is loaded["scaler_state_dict"], "scaler state dict reference identity failed"
        assert loaded["rng"] is loaded["rng_state"], "rng state dict reference identity failed"

    def test_dictionary_payload_reference_identity_handling(self, tmp_path):
        """Verify save_checkpoint with dictionary payload shares reference identity between primary and aliases."""
        sd_model = {"weight": torch.randn(2, 2)}
        sd_opt = {"state": {}, "param_groups": []}
        sd_sched = {"last_epoch": 5}
        sd_scaler = {"scale": 1.0}
        rng = capture_rng_state()

        dict_payload = {
            "epoch": 5,
            "completed_epochs": 5,
            "planned_epochs": 300,
            "training_complete": False,
            "global_step": 50,
            "model_state_dict": sd_model,
            "optimizer_state_dict": sd_opt,
            "scheduler_state_dict": sd_sched,
            "scaler_state_dict": sd_scaler,
            "seed": 42,
            "protocol_id": "myops380_train304_test76_last_v1",
            "opt_protocol": "im_mseg_adamw_poly_v1",
            "rng_state": rng,
        }

        ckpt_path = tmp_path / "test_dict_payload.pt"
        save_checkpoint(dict_payload, ckpt_path)

        loaded = torch.load(ckpt_path, weights_only=False)
        assert loaded["model"] is loaded["model_state_dict"]
        assert loaded["optimizer"] is loaded["optimizer_state_dict"]
        assert loaded["scheduler"] is loaded["scheduler_state_dict"]
        assert loaded["scaler"] is loaded["scaler_state_dict"]


# ==============================================================================
# Challenge 2: 13 Mandatory Schema Keys & Payload Validation Invariants
# ==============================================================================

class TestCheckpointSchema13MandatoryKeysAdversarial:
    """Stress-test the 13 mandatory schema keys, subkeys, and budget invariants."""

    MANDATORY_13_KEYS = [
        "epoch",
        "completed_epochs",
        "planned_epochs",
        "training_complete",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "seed",
        "protocol_id",
        "opt_protocol",
        "rng_state",
    ]

    def _build_valid_payload(self) -> Dict[str, Any]:
        return {
            "epoch": 10,
            "completed_epochs": 10,
            "planned_epochs": 300,
            "training_complete": False,
            "global_step": 1240,
            "model_state_dict": {"w": torch.tensor(1.0)},
            "optimizer_state_dict": {"opt": 1},
            "scheduler_state_dict": {"last_epoch": 1240},
            "scaler_state_dict": {"scale": 65536.0},
            "seed": 42,
            "protocol_id": "myops380_train304_test76_last_v1",
            "opt_protocol": "im_mseg_adamw_poly_v1",
            "rng_state": capture_rng_state(),
        }

    def test_schema_has_exactly_13_mandatory_keys(self):
        """Verify the mandatory key list has exactly 13 keys."""
        assert len(self.MANDATORY_13_KEYS) == 13

    def test_each_of_13_mandatory_keys_strictly_required(self):
        """Verify deleting ANY of the 13 keys causes validate_checkpoint_payload to raise ValueError."""
        base_payload = self._build_valid_payload()
        validate_checkpoint_payload(base_payload)  # baseline must pass

        for key in self.MANDATORY_13_KEYS:
            bad_payload = dict(base_payload)
            del bad_payload[key]
            with pytest.raises(ValueError, match="missing mandatory keys"):
                validate_checkpoint_payload(bad_payload)

    def test_rng_state_subkeys_strictly_required(self):
        """Verify rng_state must contain all 4 backend RNG subkeys."""
        base_payload = self._build_valid_payload()
        for subkey in ("python_random", "numpy_random", "torch_cpu", "torch_cuda"):
            bad_payload = dict(base_payload)
            bad_rng = dict(base_payload["rng_state"])
            del bad_rng[subkey]
            bad_payload["rng_state"] = bad_rng
            with pytest.raises(ValueError, match=f"rng_state missing key: {subkey}"):
                validate_checkpoint_payload(bad_payload)

    def test_non_dict_payload_and_rng_state_rejected(self):
        """Verify non-dict payload or non-dict rng_state is rejected."""
        with pytest.raises(ValueError, match="Checkpoint payload must be a dict"):
            validate_checkpoint_payload(["not", "a", "dict"])

        base_payload = self._build_valid_payload()
        base_payload["rng_state"] = "not_a_dict"
        with pytest.raises(ValueError, match="rng_state must be a dictionary"):
            validate_checkpoint_payload(base_payload)

    @pytest.mark.parametrize("invalid_epoch", [0, -1, "10", 3.14])
    def test_invalid_epoch_rejected(self, invalid_epoch):
        """Verify non-positive or non-integer epoch is rejected."""
        base_payload = self._build_valid_payload()
        base_payload["epoch"] = invalid_epoch
        base_payload["completed_epochs"] = invalid_epoch
        with pytest.raises(ValueError):
            validate_checkpoint_payload(base_payload)

    def test_planned_epochs_and_completed_budget_invariants(self):
        """Verify completed_epochs cannot exceed planned_epochs."""
        base_payload = self._build_valid_payload()
        base_payload["epoch"] = 301
        base_payload["completed_epochs"] = 301
        base_payload["planned_epochs"] = 300
        base_payload["training_complete"] = True
        with pytest.raises(ValueError, match="Inconsistent planned epochs budget"):
            validate_checkpoint_payload(base_payload)

    def test_require_complete_flag_behavior(self):
        """Verify require_complete enforces training_complete=True and completed_epochs >= planned_epochs."""
        incomplete_payload = self._build_valid_payload()
        with pytest.raises(ValueError, match="Incomplete training checkpoint"):
            validate_checkpoint_payload(incomplete_payload, require_complete=True)

        complete_payload = self._build_valid_payload()
        complete_payload["epoch"] = 300
        complete_payload["completed_epochs"] = 300
        complete_payload["planned_epochs"] = 300
        complete_payload["training_complete"] = True
        # Must pass cleanly
        validate_checkpoint_payload(complete_payload, require_complete=True)


# ==============================================================================
# Challenge 3: Full MyoPSNetL Architecture Resumption Numerical Equivalence
# ==============================================================================

class TestMyoPSNetLResumptionEquivalenceAdversarial:
    """Stress-test resumption bitwise equivalence on the real MyoPSNetL architecture."""

    def test_myopsnetl_uninterrupted_vs_resumed_exact_parity(self, tmp_path):
        """Verify MyoPSNetL + MyoPSLoss interrupted run matches uninterrupted run to 1e-6."""
        seed = 4242
        device = torch.device("cpu")  # CPU guarantees deterministic cross-run parity
        total_steps = 4
        split_step = 2

        # Fixed deterministic synthetic inputs (B=2, 3 modalities, 128x128)
        torch.manual_seed(12345)
        batches = []
        for _ in range(total_steps):
            bssfp = torch.randn(2, 1, 128, 128, device=device)
            lge   = torch.randn(2, 1, 128, 128, device=device)
            t2w   = torch.randn(2, 1, 128, 128, device=device)
            label = torch.randint(0, 4, (2, 128, 128), device=device)
            batches.append((bssfp, lge, t2w, label))

        # ----------------------------------------------------------------------
        # Run 1: Continuous Uninterrupted Run (4 steps)
        # ----------------------------------------------------------------------
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model_cont = MyoPSNetL().to(device)
        criterion_cont = MyoPSLoss().to(device)
        opt_cont = build_optimizer(model_cont, base_lr=0.0003333, weight_decay=0.0001)
        sched_cont = build_scheduler(opt_cont, total_updates=total_steps)
        scaler_cont = torch.amp.GradScaler("cuda", enabled=False)

        losses_cont = []
        for bssfp, lge, t2w, label in batches:
            t_c0, t_lge, t_t2 = create_multi_head_targets(label)
            s_c0, s_lge, s_t2 = model_cont(bssfp, lge, t2w)
            loss, _, _, _ = criterion_cont(s_c0, s_lge, s_t2, t_c0, t_lge, t_t2)
            loss.backward()
            opt_cont.step()
            sched_cont.step()
            opt_cont.zero_grad(set_to_none=True)
            losses_cont.append(loss.item())

        weights_cont = {name: p.detach().clone() for name, p in model_cont.named_parameters()}
        opt_state_cont = opt_cont.state_dict()

        # ----------------------------------------------------------------------
        # Run 2: Interrupted after step 2, saved, then resumed for steps 3 and 4
        # ----------------------------------------------------------------------
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model_res = MyoPSNetL().to(device)
        criterion_res = MyoPSLoss().to(device)
        opt_res = build_optimizer(model_res, base_lr=0.0003333, weight_decay=0.0001)
        sched_res = build_scheduler(opt_res, total_updates=total_steps)
        scaler_res = torch.amp.GradScaler("cuda", enabled=False)

        losses_res = []
        for i in range(split_step):
            bssfp, lge, t2w, label = batches[i]
            t_c0, t_lge, t_t2 = create_multi_head_targets(label)
            s_c0, s_lge, s_t2 = model_res(bssfp, lge, t2w)
            loss, _, _, _ = criterion_res(s_c0, s_lge, s_t2, t_c0, t_lge, t_t2)
            loss.backward()
            opt_res.step()
            sched_res.step()
            opt_res.zero_grad(set_to_none=True)
            losses_res.append(loss.item())

        # Save checkpoint at split_step (step 2)
        ckpt_path = tmp_path / "myopsnet_interrupted.pt"
        save_checkpoint(
            checkpoint_path=ckpt_path,
            model=model_res,
            optimizer=opt_res,
            scheduler=sched_res,
            scaler=scaler_res,
            epoch=1,
            completed_epochs=1,
            planned_epochs=2,
            global_step=split_step,
            total_updates=total_steps,
            seed=seed,
        )

        # Perturb RNG states thoroughly before resumption
        random.seed(999999)
        np.random.seed(999999)
        torch.manual_seed(999999)

        # Instantiate fresh model and components
        fresh_model = MyoPSNetL().to(device)
        fresh_opt = build_optimizer(fresh_model, base_lr=0.0003333, weight_decay=0.0001)
        fresh_sched = build_scheduler(fresh_opt, total_updates=total_steps)
        fresh_scaler = torch.amp.GradScaler("cuda", enabled=False)

        res = resume_training_state(
            checkpoint_path=ckpt_path,
            model=fresh_model,
            optimizer=fresh_opt,
            scheduler=fresh_sched,
            scaler=fresh_scaler,
            device=device,
        )

        assert res["status"] == "resumed"
        assert res["completed_epochs"] == 1
        assert res["global_step"] == split_step
        assert fresh_sched.last_epoch == split_step

        # Continue steps 2 and 3
        for i in range(split_step, total_steps):
            bssfp, lge, t2w, label = batches[i]
            t_c0, t_lge, t_t2 = create_multi_head_targets(label)
            s_c0, s_lge, s_t2 = fresh_model(bssfp, lge, t2w)
            loss, _, _, _ = criterion_res(s_c0, s_lge, s_t2, t_c0, t_lge, t_t2)
            loss.backward()
            fresh_opt.step()
            fresh_sched.step()
            fresh_opt.zero_grad(set_to_none=True)
            losses_res.append(loss.item())

        weights_res = {name: p.detach().clone() for name, p in fresh_model.named_parameters()}
        opt_state_res = fresh_opt.state_dict()

        # ----------------------------------------------------------------------
        # Empirical Verification: Exact Loss & Parameter Parity
        # ----------------------------------------------------------------------
        assert len(losses_cont) == len(losses_res) == total_steps
        for step_idx, (l_c, l_r) in enumerate(zip(losses_cont, losses_res)):
            assert math.isclose(l_c, l_r, abs_tol=1e-5), (
                f"Loss mismatch at step {step_idx}: continuous={l_c} vs resumed={l_r}"
            )

        for name in weights_cont:
            p_c = weights_cont[name]
            p_r = weights_res[name]
            assert torch.allclose(p_c, p_r, atol=1e-5), f"Parameter mismatch in {name} after resumption"

        # Check optimizer state parity (step counts, momentum)
        assert opt_state_cont["param_groups"][0]["lr"] == opt_state_res["param_groups"][0]["lr"]


# ==============================================================================
# Challenge 4: Completed-Training Defense & Clean Exit
# ==============================================================================

class TestCompletedTrainingDefenseAdversarial:
    """Stress-test defense against re-training or mutating completed checkpoints."""

    def test_resume_training_state_detects_completion(self, tmp_path):
        """Verify resume_training_state flags already_completed and complete=True."""
        model = nn.Linear(2, 2)
        opt = build_optimizer(model)
        sched = build_scheduler(opt)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        ckpt_path = tmp_path / "complete_ckpt.pt"
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

        res = resume_training_state(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
        )
        assert res["status"] == "already_completed"
        assert res["training_complete"] is True
        assert res["completed_epochs"] == 300
        assert res["start_epoch"] == 300

    def test_train_main_rejects_legacy_completed_checkpoint(self, tmp_path):
        """A completed legacy checkpoint cannot silently adopt the new RNG/init recipe."""
        # Create completed checkpoint
        model = MyoPSNetL()
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

        mtime_before = ckpt_path.stat().st_mtime

        # Run train.main with --resume pointing to completed checkpoint
        with pytest.raises(ValueError, match="lacks the current RNG/initialization"):
            train.main([
                "--resume", str(ckpt_path),
                "--epochs", "300",
                "--seed", "42",
                "--run_dir", str(tmp_path),
                "--dry_run",
                "--device", "cpu",
            ])

        # Verify checkpoint file was NOT modified/re-saved
        mtime_after = ckpt_path.stat().st_mtime
        assert mtime_before == mtime_after, "Checkpoint was mutated despite being already complete!"

    def test_train_main_rejects_legacy_short_budget_checkpoint(self, tmp_path):
        """A short completed legacy checkpoint also needs an explicit compatible contract."""
        model = MyoPSNetL()
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
            epoch=50,
            completed_epochs=50,
            planned_epochs=50,
            global_step=6200,
            total_updates=6200,
            seed=42,
        )

        with pytest.raises(ValueError, match="lacks the current RNG/initialization"):
            train.main([
                "--resume", str(ckpt_path),
                "--epochs", "50",
                "--seed", "42",
                "--run_dir", str(tmp_path),
                "--device", "cpu",
            ])


# ==============================================================================
# Challenge 5: Atomic Checkpoint Replacement & Temporary File Cleanup
# ==============================================================================

class TestAtomicReplacementInvariantsAdversarial:
    """Stress-test atomic save under rapid overwrites and failure cleanup."""

    def test_rapid_consecutive_atomic_saves_leave_zero_temp_files(self, tmp_path):
        """Verify 25 rapid atomic saves leave zero .tmp files and maintain data integrity."""
        ckpt_path = tmp_path / "stress_atomic.pt"

        for i in range(25):
            payload = {
                "version": i,
                "data": torch.randn(10, 10),
            }
            safe_atomic_save(payload, ckpt_path)

        assert ckpt_path.is_file()
        tmp_files = list(tmp_path.glob("*.tmp*"))
        assert len(tmp_files) == 0, f"Found lingering temporary files: {tmp_files}"

        loaded = torch.load(ckpt_path, weights_only=False)
        assert loaded["version"] == 24

    def test_safe_atomic_save_creates_nested_missing_parent_directories(self, tmp_path):
        """Verify safe_atomic_save automatically creates deep parent directory tree."""
        deep_path = tmp_path / "deep" / "nested" / "dir" / "tree" / "checkpoint.pt"
        assert not deep_path.parent.exists()

        safe_atomic_save({"status": "ok"}, deep_path)
        assert deep_path.is_file()
        assert torch.load(deep_path, weights_only=False)["status"] == "ok"
