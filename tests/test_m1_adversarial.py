"""
tests/test_m1_adversarial.py
============================
Adversarial and empirical verification suite for Milestone 1.
Authored by: m1_challenger_2 (teamwork_challenger)

Scope:
1. Floating-point precision underflow/overflow: FP16 mixed precision
   (torch.amp.autocast('cuda')) vs FP32 across all loss functions:
   - RobustWCELoss / WCELoss
   - MultiClassDiceLoss / DiceLoss
   - InvariantLoss
   - InclusiveLossScar / InclusiveLossEdema
   - MyoPSLoss (composite)
2. Zero probability NaN immunity: verifying no `0.0 * (-inf) = NaN` occurs
   when predicted probabilities contain zeros.
3. Prediction fusion conflict resolution: verifying priority hierarchy
   (Scar > Edema > Normal > Background) across conflicting head predictions.
4. Normal myocardium preservation: verifying normal myocardium is never
   erased when scar and edema are absent.
5. End-to-end AMP training step with GradScaler.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from network.model import MyoPSNetL
from utils.tools import create_multi_head_targets, fuse_predictions
from criterion.loss import (
    RobustWCELoss,
    WCELoss,
    MultiClassDiceLoss,
    DiceLoss,
    BinaryDiceLoss,
    InvariantLoss,
    InclusiveLossScar,
    InclusiveLossEdema,
    SegmentationLoss,
    MyoPSLoss,
)


CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"


# ==============================================================================
# Suite 1: Floating-Point Precision (FP32 vs FP16 Autocast) Stability
# ==============================================================================

class TestFloatingPointPrecision:
    """Stress tests loss functions under FP32 and FP16 mixed precision (autocast)."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16] if CUDA_AVAILABLE else [torch.float32])
    def test_wce_loss_precision(self, dtype):
        """Test RobustWCELoss under FP32 and FP16."""
        device = "cuda" if (dtype == torch.float16 and CUDA_AVAILABLE) else "cpu"
        loss_fn = RobustWCELoss()

        B, C, H, W = 4, 3, 64, 64
        torch.manual_seed(42)

        # Probabilities
        logits = torch.randn(B, C, H, W, device=device)
        probs = torch.softmax(logits, dim=1)
        targets = torch.randint(0, C, (B, H, W), device=device, dtype=torch.long)

        # FP32
        loss_fp32 = loss_fn(probs, targets)
        assert torch.isfinite(loss_fp32), f"WCE FP32 failed: {loss_fp32}"
        assert loss_fp32.item() >= 0.0

        # FP16 with autocast
        if CUDA_AVAILABLE and device == "cuda":
            with torch.amp.autocast("cuda"):
                loss_amp = loss_fn(probs, targets)
            assert torch.isfinite(loss_amp), f"WCE AMP failed: {loss_amp}"
            assert loss_amp.item() >= 0.0
            assert abs(loss_fp32.item() - loss_amp.item()) < 0.05, "AMP deviation too high"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16] if CUDA_AVAILABLE else [torch.float32])
    def test_dice_loss_precision(self, dtype):
        """Test MultiClassDiceLoss under FP32 and FP16."""
        device = "cuda" if (dtype == torch.float16 and CUDA_AVAILABLE) else "cpu"
        loss_fn = MultiClassDiceLoss()

        B, C, H, W = 4, 3, 64, 64
        torch.manual_seed(42)
        probs = torch.softmax(torch.randn(B, C, H, W, device=device), dim=1)
        targets = torch.randint(0, C, (B, H, W), device=device, dtype=torch.long)

        loss_fp32 = loss_fn(probs, targets)
        assert torch.isfinite(loss_fp32)
        assert 0.0 <= loss_fp32.item() <= 1.0

        if CUDA_AVAILABLE and device == "cuda":
            with torch.amp.autocast("cuda"):
                loss_amp = loss_fn(probs, targets)
            assert torch.isfinite(loss_amp)
            assert 0.0 <= loss_amp.item() <= 1.0
            assert abs(loss_fp32.item() - loss_amp.item()) < 0.01

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16] if CUDA_AVAILABLE else [torch.float32])
    def test_invariant_loss_precision(self, dtype):
        """Test InvariantLoss under FP32 and FP16."""
        device = "cuda" if (dtype == torch.float16 and CUDA_AVAILABLE) else "cpu"
        loss_fn = InvariantLoss()

        B, C, H, W = 4, 2, 64, 64
        torch.manual_seed(42)
        feat1 = torch.softmax(torch.randn(B, C, H, W, device=device), dim=1)
        feat2 = torch.softmax(torch.randn(B, C, H, W, device=device), dim=1)

        loss_fp32 = loss_fn(feat1, feat2)
        assert torch.isfinite(loss_fp32)
        assert 0.0 <= loss_fp32.item() <= 2.0

        if CUDA_AVAILABLE and device == "cuda":
            with torch.amp.autocast("cuda"):
                loss_amp = loss_fn(feat1, feat2)
            assert torch.isfinite(loss_amp)
            assert 0.0 <= loss_amp.item() <= 2.0
            assert abs(loss_fp32.item() - loss_amp.item()) < 0.01

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16] if CUDA_AVAILABLE else [torch.float32])
    def test_inclusive_losses_precision(self, dtype):
        """Test InclusiveLossScar and InclusiveLossEdema under FP32 and FP16."""
        device = "cuda" if (dtype == torch.float16 and CUDA_AVAILABLE) else "cpu"
        loss_scar_fn = InclusiveLossScar()
        loss_edema_fn = InclusiveLossEdema()

        B, H, W = 4, 64, 64
        torch.manual_seed(42)
        p_scar = torch.rand(B, H, W, device=device)
        p_edema = torch.rand(B, H, W, device=device)
        gd_scar = (torch.rand(B, H, W, device=device) > 0.7).float()
        gd_edema = (torch.rand(B, H, W, device=device) > 0.5).float()

        # FP32
        l_scar = loss_scar_fn(p_scar, gd_edema)
        l_edema = loss_edema_fn(p_edema, gd_scar)
        assert torch.isfinite(l_scar) and l_scar.item() >= 0.0
        assert torch.isfinite(l_edema) and l_edema.item() >= 0.0

        # FP16 with autocast
        if CUDA_AVAILABLE and device == "cuda":
            with torch.amp.autocast("cuda"):
                l_scar_amp = loss_scar_fn(p_scar, gd_edema)
                l_edema_amp = loss_edema_fn(p_edema, gd_scar)
            assert torch.isfinite(l_scar_amp) and l_scar_amp.item() >= 0.0
            assert torch.isfinite(l_edema_amp) and l_edema_amp.item() >= 0.0

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available for AMP training test")
    def test_end_to_end_amp_training_step(self):
        """Full forward + backward + GradScaler step with MyoPSNetL and MyoPSLoss."""
        model = MyoPSNetL().cuda()
        model.train()
        criterion = MyoPSLoss().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")

        B = 2
        c0 = torch.randn(B, 1, 128, 128, device="cuda")
        lge = torch.randn(B, 1, 128, 128, device="cuda")
        t2 = torch.randn(B, 1, 128, 128, device="cuda")
        raw_labels = torch.randint(0, 4, (B, 128, 128), device="cuda", dtype=torch.long)
        t_c0, t_lge, t_t2 = create_multi_head_targets(raw_labels)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)
            total_loss, l_seg, l_inv, l_inc = criterion(
                seg_c0, seg_lge, seg_t2, t_c0, t_lge, t_t2
            )

        assert torch.isfinite(total_loss)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Check all gradients are finite
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"


# ==============================================================================
# Suite 2: Zero Probability NaN Immunity (0.0 * (-inf) Check)
# ==============================================================================

class TestZeroProbabilityNaNImmunity:
    """Verifies that exact 0.0 probabilities never trigger 0.0 * (-inf) = NaN."""

    def test_wce_loss_exact_zeros(self):
        """RobustWCELoss with exact 0.0 probabilities across multiple classes."""
        loss_fn = RobustWCELoss()
        B, C, H, W = 2, 3, 16, 16

        # Construct exact 0.0 probabilities (one-hot probability distribution)
        probs = torch.zeros(B, C, H, W, dtype=torch.float32)
        probs[:, 0, :, :] = 1.0  # Class 0 has 1.0; Classes 1 and 2 are EXACTLY 0.0

        # Scenario A: Target matches class 0 (target=0 for classes 1 and 2 where prob=0.0)
        # Tests: 0.0 * log(0.0)
        target_a = torch.zeros(B, H, W, dtype=torch.long)
        loss_a = loss_fn(probs, target_a)
        assert torch.isfinite(loss_a), f"WCE failed on matching target with zeros: {loss_a}"
        assert not torch.isnan(loss_a), "WCE produced NaN on 0.0 * log(0.0)"

        # Scenario B: Target is class 1 (target=1 where prob=0.0)
        # Tests: 1.0 * log(0.0) -> must be clamped to eps without inf
        target_b = torch.ones(B, H, W, dtype=torch.long)
        loss_b = loss_fn(probs, target_b)
        assert torch.isfinite(loss_b), f"WCE failed on non-matching target with zeros: {loss_b}"
        assert not torch.isinf(loss_b), "WCE produced Inf on log(0.0)"

        # Scenario C: Target has absent classes
        target_c = torch.full((B, H, W), 2, dtype=torch.long)
        loss_c = loss_fn(probs, target_c)
        assert torch.isfinite(loss_c), f"WCE failed on absent class target: {loss_c}"

    def test_dice_loss_exact_zeros(self):
        """MultiClassDiceLoss with exact 0.0 probabilities and exact 0.0 targets."""
        loss_fn = MultiClassDiceLoss()
        B, C, H, W = 2, 3, 16, 16

        probs = torch.zeros(B, C, H, W, dtype=torch.float32)
        probs[:, 0, :, :] = 1.0

        # Both predict and target have 0 voxels for class 1 and 2 (0/0 situation)
        target = torch.zeros(B, H, W, dtype=torch.long)
        loss = loss_fn(probs, target)
        assert torch.isfinite(loss), f"Dice loss produced non-finite on 0/0: {loss}"
        assert not torch.isnan(loss)
        assert 0.0 <= loss.item() <= 1.0

    def test_inclusive_losses_exact_zeros_and_ones(self):
        """Inclusive losses under extreme boundaries (scar=0, edema=0, scar=1, edema=1)."""
        loss_scar = InclusiveLossScar()
        loss_edema = InclusiveLossEdema()

        B, H, W = 2, 16, 16

        # Test A: Scar=1 everywhere, Edema GT=0 everywhere (maximum penalty)
        scar_all1 = torch.ones(B, H, W)
        edema_gt0 = torch.zeros(B, H, W)
        l_scar = loss_scar(scar_all1, edema_gt0)
        assert torch.isfinite(l_scar)
        assert not torch.isnan(l_scar)
        assert l_scar.item() > 0.0

        # Test B: Scar=0 everywhere, Edema GT=0 everywhere (zero penalty)
        scar_all0 = torch.zeros(B, H, W)
        l_scar_zero = loss_scar(scar_all0, edema_gt0)
        assert torch.isfinite(l_scar_zero)
        assert l_scar_zero.item() == pytest.approx(0.0, abs=1e-5)

        # Test C: Edema=0 everywhere, Scar GT=1 everywhere (maximum penalty)
        edema_all0 = torch.zeros(B, H, W)
        scar_gt1 = torch.ones(B, H, W)
        l_edema = loss_edema(edema_all0, scar_gt1)
        assert torch.isfinite(l_edema)
        assert not torch.isnan(l_edema)
        assert l_edema.item() > 0.0

        # Test D: Edema=0 everywhere, Scar GT=0 everywhere (denom is eps)
        scar_gt0 = torch.zeros(B, H, W)
        l_edema_zero = loss_edema(edema_all0, scar_gt0)
        assert torch.isfinite(l_edema_zero)
        assert not torch.isnan(l_edema_zero)
        assert l_edema_zero.item() == pytest.approx(0.0, abs=1e-5)


# ==============================================================================
# Suite 3: fuse_predictions Conflict Resolution & Priority Hierarchy
# ==============================================================================

class TestFusePredictionsConflictHierarchy:
    """
    Stress-tests fuse_predictions against all conflicting cross-head outputs.
    Priority hierarchy: Scar (3) > Edema (2) > Normal (1) > Background (0).
    """

    def test_conflict_scar_wins_over_background(self):
        """seg_LGE predicts Scar (2), seg_T2 predicts BG (0), seg_C0 predicts BG (0)."""
        # Head values: C0=0, LGE=2, T2=0
        seg_c0 = torch.tensor([[0]], dtype=torch.long)
        seg_lge = torch.tensor([[2]], dtype=torch.long)
        seg_t2 = torch.tensor([[0]], dtype=torch.long)

        out = fuse_predictions(seg_c0, seg_lge, seg_t2)
        assert out.item() == 3, f"Expected Scar (3), got {out.item()}"

    def test_conflict_scar_wins_over_edema(self):
        """seg_LGE predicts Scar (2), seg_T2 predicts Edema (2) -> Scar MUST win."""
        # Head values: C0=1, LGE=2, T2=2
        seg_c0 = torch.tensor([[1]], dtype=torch.long)
        seg_lge = torch.tensor([[2]], dtype=torch.long)
        seg_t2 = torch.tensor([[2]], dtype=torch.long)

        out = fuse_predictions(seg_c0, seg_lge, seg_t2)
        assert out.item() == 3, f"Expected Scar (3), got {out.item()} (Edema should be overridden)"

    def test_conflict_scar_wins_over_normal(self):
        """seg_LGE predicts Scar (2), seg_T2 predicts Normal (1), seg_C0 predicts Myo (1)."""
        seg_c0 = torch.tensor([[1]], dtype=torch.long)
        seg_lge = torch.tensor([[2]], dtype=torch.long)
        seg_t2 = torch.tensor([[1]], dtype=torch.long)

        out = fuse_predictions(seg_c0, seg_lge, seg_t2)
        assert out.item() == 3, f"Expected Scar (3), got {out.item()}"

    def test_conflict_edema_wins_over_background(self):
        """seg_LGE predicts BG (0), seg_T2 predicts Edema (2), seg_C0 predicts BG (0)."""
        seg_c0 = torch.tensor([[0]], dtype=torch.long)
        seg_lge = torch.tensor([[0]], dtype=torch.long)
        seg_t2 = torch.tensor([[2]], dtype=torch.long)

        out = fuse_predictions(seg_c0, seg_lge, seg_t2)
        assert out.item() == 2, f"Expected Edema (2), got {out.item()}"

    def test_conflict_edema_wins_over_normal(self):
        """seg_LGE predicts Non-scar (1), seg_T2 predicts Edema (2), seg_C0 predicts Myo (1)."""
        seg_c0 = torch.tensor([[1]], dtype=torch.long)
        seg_lge = torch.tensor([[1]], dtype=torch.long)
        seg_t2 = torch.tensor([[2]], dtype=torch.long)

        out = fuse_predictions(seg_c0, seg_lge, seg_t2)
        assert out.item() == 2, f"Expected Edema (2), got {out.item()}"

    def test_exhaustive_18_combinations_truth_table(self):
        """
        Exhaustively test all 2 * 3 * 3 = 18 combinations of:
          C0 in {0, 1}
          LGE in {0, 1, 2}
          T2 in {0, 1, 2}
        Verifies truth table matches priority hierarchy:
          - If LGE == 2: ALWAYS 3 (Scar)
          - Elif T2 == 2: ALWAYS 2 (Edema-exclusive)
          - Elif C0 == 1: ALWAYS 1 (Normal Myocardium)
          - Else: 0 (Background)
        """
        for c0 in [0, 1]:
            for lge in [0, 1, 2]:
                for t2 in [0, 1, 2]:
                    s_c0 = torch.tensor([[c0]], dtype=torch.long)
                    s_lge = torch.tensor([[lge]], dtype=torch.long)
                    s_t2 = torch.tensor([[t2]], dtype=torch.long)

                    result = fuse_predictions(s_c0, s_lge, s_t2).item()

                    # Oracle prediction
                    if lge == 2:
                        expected = 3
                    elif t2 == 2:
                        expected = 2
                    elif c0 == 1:
                        expected = 1
                    else:
                        expected = 0

                    assert result == expected, (
                        f"Truth table mismatch for (C0={c0}, LGE={lge}, T2={t2}): "
                        f"expected {expected}, got {result}"
                    )


# ==============================================================================
# Suite 4: Normal Myocardium Preservation
# ==============================================================================

class TestNormalMyocardiumPreservation:
    """Verifies that normal myocardium is NEVER erased when scar and edema are absent."""

    def test_normal_myo_never_erased_with_various_background_agreements(self):
        """When Scar (LGE=2) and Edema (T2=2) are absent, C0=1 must ALWAYS produce 1."""
        # Non-pathological combinations for LGE in {0, 1} and T2 in {0, 1}
        non_pathology_cases = [
            (1, 1),  # LGE=1 (non-scar myo), T2=1 (normal myo)
            (1, 0),  # LGE=1 (non-scar myo), T2=0 (bg)
            (0, 1),  # LGE=0 (bg),           T2=1 (normal myo)
            (0, 0),  # LGE=0 (bg),           T2=0 (bg)
        ]

        for lge, t2 in non_pathology_cases:
            s_c0 = torch.ones((1, 32, 32), dtype=torch.long)  # All normal myocardium
            s_lge = torch.full((1, 32, 32), lge, dtype=torch.long)
            s_t2 = torch.full((1, 32, 32), t2, dtype=torch.long)

            fused = fuse_predictions(s_c0, s_lge, s_t2)
            assert torch.all(fused == 1), (
                f"Normal myocardium was corrupted/erased when C0=1, LGE={lge}, T2={t2}! "
                f"Found unique values: {torch.unique(fused).tolist()}"
            )

    def test_realistic_spatial_slice_preservation(self):
        """Realistic spatial test: annular myocardium ring with central/outer background."""
        H, W = 128, 128
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        r = torch.sqrt((x - 64.0)**2 + (y - 64.0)**2)

        # Ring: radius between 20 and 40
        myo_ring = (r >= 20) & (r <= 40)

        # Normal myocardium: C0=1 on ring, 0 elsewhere
        s_c0 = myo_ring.long().unsqueeze(0)
        # LGE has no scar: 1 on ring, 0 elsewhere
        s_lge = myo_ring.long().unsqueeze(0)
        # T2 has no edema: 1 on ring, 0 elsewhere
        s_t2 = myo_ring.long().unsqueeze(0)

        fused = fuse_predictions(s_c0, s_lge, s_t2)

        # Ring must be exactly 1, outside must be exactly 0
        assert torch.all(fused[0][myo_ring] == 1), "Ring voxels failed to preserve Normal Myocardium"
        assert torch.all(fused[0][~myo_ring] == 0), "Background voxels corrupted"
        assert torch.sum(fused == 2) == 0, "Phantom edema appeared"
        assert torch.sum(fused == 3) == 0, "Phantom scar appeared"


# ==============================================================================
# Suite 5: Input Format Robustness in fuse_predictions
# ==============================================================================

class TestFusePredictionsInputFormats:
    """Verifies fuse_predictions supports various shapes and container types."""

    def test_4d_probability_tensors(self):
        """Batched 4D probability tensors (B, C, H, W)."""
        B, H, W = 2, 32, 32
        p_c0 = torch.zeros(B, 2, H, W)
        p_c0[:, 1] = 1.0  # Class 1

        p_lge = torch.zeros(B, 3, H, W)
        p_lge[:, 2] = 1.0  # Class 2 (Scar)

        p_t2 = torch.zeros(B, 3, H, W)
        p_t2[:, 0] = 1.0  # Class 0

        out = fuse_predictions(p_c0, p_lge, p_t2)
        assert out.shape == (B, H, W)
        assert out.dtype == torch.uint8
        assert torch.all(out == 3)

    def test_numpy_array_inputs(self):
        """NumPy array inputs."""
        c0 = np.ones((16, 16), dtype=np.int64)
        lge = np.zeros((16, 16), dtype=np.int64)
        t2 = np.zeros((16, 16), dtype=np.int64)

        out = fuse_predictions(c0, lge, t2)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.uint8
        assert torch.all(out == 1)



# ==============================================================================
# Suite 6: Extreme Values & Denormals under FP16 Autocast
# ==============================================================================

class TestExtremeValuesAndPathologicalBatchesFP16:
    """Stress-tests extreme numbers (denormals, massive logits, zero-pathology batches) in AMP."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for AMP")
    @pytest.mark.parametrize("weight_mode", ["log", "inverse"])
    def test_wce_loss_from_logits_extreme_values_amp(self, weight_mode):
        """Test RobustWCELoss(from_logits=True) with extreme logits (-100, +100) under AMP."""
        loss_fn = RobustWCELoss(from_logits=True, weight_mode=weight_mode)
        B, C, H, W = 2, 3, 32, 32

        # Extreme logits
        logits = torch.randn(B, C, H, W, device="cuda") * 20.0  # large dynamic range
        logits[:, 0, :16, :16] = 100.0
        logits[:, 1, 16:, 16:] = -100.0
        targets = torch.randint(0, C, (B, H, W), device="cuda", dtype=torch.long)

        with torch.amp.autocast("cuda"):
            loss = loss_fn(logits, targets)

        assert torch.isfinite(loss), f"Extreme logits produced non-finite loss: {loss}"
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for AMP")
    @pytest.mark.parametrize(
        "batch_type",
        ["all_classes", "zero_scar", "zero_edema", "zero_pathology", "all_background"],
    )
    def test_myops_loss_all_pathological_batches_in_amp(self, batch_type):
        """Verify composite MyoPSLoss forward & backward in FP16 AMP on all clinical slice types."""
        device = "cuda"
        criterion = MyoPSLoss().to(device)
        B = 2
        torch.manual_seed(999)

        if batch_type == "all_classes":
            lbl = torch.randint(0, 4, (B, 64, 64), device=device, dtype=torch.long)
        elif batch_type == "zero_scar":
            lbl = torch.randint(0, 3, (B, 64, 64), device=device, dtype=torch.long)
            lbl[lbl == 2] = 3
        elif batch_type == "zero_edema":
            lbl = torch.randint(0, 3, (B, 64, 64), device=device, dtype=torch.long)
        elif batch_type == "zero_pathology":
            lbl = torch.randint(0, 2, (B, 64, 64), device=device, dtype=torch.long)
        elif batch_type == "all_background":
            lbl = torch.zeros((B, 64, 64), device=device, dtype=torch.long)

        t_c0, t_lge, t_t2 = create_multi_head_targets(lbl)

        # Generate predictions requiring gradients
        logits_c0 = torch.randn(B, 2, 64, 64, device=device, requires_grad=True)
        logits_lge = torch.randn(B, 3, 64, 64, device=device, requires_grad=True)
        logits_t2 = torch.randn(B, 3, 64, 64, device=device, requires_grad=True)

        with torch.amp.autocast("cuda"):
            seg_c0 = torch.softmax(logits_c0, dim=1)
            seg_lge = torch.softmax(logits_lge, dim=1)
            seg_t2 = torch.softmax(logits_t2, dim=1)

            total_loss, l_seg, l_inv, l_inc = criterion(
                seg_c0, seg_lge, seg_t2, t_c0, t_lge, t_t2
            )

        assert torch.isfinite(total_loss), f"NaN/Inf loss on {batch_type}: {total_loss}"
        assert torch.isfinite(l_seg), f"NaN/Inf seg loss on {batch_type}: {l_seg}"
        assert torch.isfinite(l_inv), f"NaN/Inf inv loss on {batch_type}: {l_inv}"
        assert torch.isfinite(l_inc), f"NaN/Inf inc loss on {batch_type}: {l_inc}"

        # Test backward pass
        total_loss.backward()

        for name, grad in [("logits_c0", logits_c0.grad), ("logits_lge", logits_lge.grad), ("logits_t2", logits_t2.grad)]:
            assert grad is not None, f"Grad is None for {name} on {batch_type}"
            assert torch.isfinite(grad).all(), f"Non-finite gradient in {name} on {batch_type}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

