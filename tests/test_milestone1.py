"""
tests/test_milestone1.py (Milestone 1 Verification Test Suite)
============================================================
Comprehensive test harness for Milestone 1:
- Feature 1 & 2 & 3: MyoPSNetL 3-Modality Architecture (bSSFP, LGE, T2w) & 2-Class Cardiac Head
- Feature 4: Target Adapter (Legacy MyoPS380 0..3 -> target_C0, target_LGE, target_T2) & Prediction Fusion
- Feature 5: Loss Hardening (RobustWCELoss, DiceLoss, InvariantLoss, InclusiveLoss, Composite MyoPSLoss)
- Gradient Propagation, Multi-Device Execution (CPU/CUDA), and Arbitrary Batch Sizes (B=16, B=4 remainder)
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Imports from codebase
from network.model import MyoPSNetL
from utils.tools import create_multi_head_targets, fuse_predictions
from criterion.loss import (
    RobustWCELoss,
    WCELoss,
    DiceLoss,
    MultiClassDiceLoss,
    BinaryDiceLoss,
    SegmentationLoss,
    InvariantLoss,
    InclusiveLossScar,
    InclusiveLossEdema,
    MyoPSLoss,
)


# ==============================================================================
# Helper Oracles / Reference Implementations (Used for Verification & Fallbacks)
# ==============================================================================

def ref_create_multi_head_targets(label: torch.Tensor):
    """
    Reference target adapter implementation.
    Input: label tensor with values in {0, 1, 2, 3}.
    Returns: (target_C0, target_LGE, target_T2)
      - target_C0: 0: bg, 1: whole myocardium (1, 2, 3 -> 1)
      - target_LGE: 0: bg, 1: non-scar myo (1, 3 -> 1), 2: scar (2 -> 2)
      - target_T2: 0: bg, 1: normal myo (1 -> 1), 2: edema-inclusive (2, 3 -> 2)
    """
    target_C0 = (label >= 1).long()

    target_LGE = torch.zeros_like(label, dtype=torch.long)
    target_LGE[(label == 1) | (label == 3)] = 1
    target_LGE[label == 2] = 2

    target_T2 = torch.zeros_like(label, dtype=torch.long)
    target_T2[label == 1] = 1
    target_T2[(label == 2) | (label == 3)] = 2

    return target_C0, target_LGE, target_T2


def ref_to_one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Converts (B, H, W) or (H, W) class indices into one-hot float tensor."""
    if target.dim() == 2:
        one_hot = torch.zeros(
            (num_classes, target.shape[0], target.shape[1]),
            device=target.device,
            dtype=torch.float32,
        )
        one_hot.scatter_(0, target.unsqueeze(0), 1.0)
    elif target.dim() == 3:
        one_hot = torch.zeros(
            (target.shape[0], num_classes, target.shape[1], target.shape[2]),
            device=target.device,
            dtype=torch.float32,
        )
        one_hot.scatter_(1, target.unsqueeze(1), 1.0)
    else:
        raise ValueError(f"Expected 2D or 3D target, got shape {target.shape}")
    return one_hot


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def target_adapter_fn():
    return create_multi_head_targets


@pytest.fixture
def synthetic_inputs():
    """Returns 3-modality synthetic batch (c0, lge, t2) of shape (4, 1, 128, 128)."""
    torch.manual_seed(42)
    B = 4
    c0 = torch.randn(B, 1, 128, 128)
    lge = torch.randn(B, 1, 128, 128)
    t2 = torch.randn(B, 1, 128, 128)
    return c0, lge, t2


# ==============================================================================
# 1. Target Adapter Verification Suite
# ==============================================================================

class TestTargetAdapter:
    """Verifies label mapping correctness and mathematical consistency."""

    def test_all_classes_mapping(self, target_adapter_fn):
        """Verify mapping when all legacy labels {0, 1, 2, 3} are present."""
        label = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        t_c0, t_lge, t_t2 = target_adapter_fn(label)

        # target_C0: Whole Myocardium (1, 2, 3 -> 1)
        expected_c0 = torch.tensor([[0, 1], [1, 1]], dtype=torch.long)
        assert torch.equal(t_c0, expected_c0), f"target_C0 failed: got {t_c0.tolist()}"

        # target_LGE: Scar (2 -> 2), non-scar myo (1, 3 -> 1), bg -> 0
        expected_lge = torch.tensor([[0, 1], [2, 1]], dtype=torch.long)
        assert torch.equal(t_lge, expected_lge), f"target_LGE failed: got {t_lge.tolist()}"

        # target_T2: Edema-inclusive (2, 3 -> 2), normal myo (1 -> 1), bg -> 0
        expected_t2 = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
        assert torch.equal(t_t2, expected_t2), f"target_T2 failed: got {t_t2.tolist()}"

    def test_only_background(self, target_adapter_fn):
        """Verify mapping on an all-background mask (e.g., non-cardiac slice)."""
        label = torch.zeros((2, 128, 128), dtype=torch.long)
        t_c0, t_lge, t_t2 = target_adapter_fn(label)

        assert torch.all(t_c0 == 0)
        assert torch.all(t_lge == 0)
        assert torch.all(t_t2 == 0)

    def test_only_normal_myocardium(self, target_adapter_fn):
        """Verify mapping on slices with normal myocardium but no pathology {0, 1}."""
        label = torch.zeros((2, 128, 128), dtype=torch.long)
        label[:, 40:80, 40:80] = 1

        t_c0, t_lge, t_t2 = target_adapter_fn(label)
        mask_normal = (label == 1)

        assert torch.all(t_c0[mask_normal] == 1)
        assert torch.all(t_lge[mask_normal] == 1)
        assert torch.all(t_t2[mask_normal] == 1)
        # Scar and edema classes (2) must be completely absent
        assert torch.sum(t_lge == 2) == 0
        assert torch.sum(t_t2 == 2) == 0

    def test_only_scar(self, target_adapter_fn):
        """Verify mapping on slices with scar but no edema-exclusive {0, 2}."""
        label = torch.zeros((2, 128, 128), dtype=torch.long)
        label[:, 50:70, 50:70] = 2

        t_c0, t_lge, t_t2 = target_adapter_fn(label)
        mask_scar = (label == 2)

        assert torch.all(t_c0[mask_scar] == 1)
        assert torch.all(t_lge[mask_scar] == 2)
        assert torch.all(t_t2[mask_scar] == 2)  # Scar is included in edema-inclusive!

    def test_only_edema(self, target_adapter_fn):
        """Verify mapping on slices with edema-exclusive but no scar {0, 3}."""
        label = torch.zeros((2, 128, 128), dtype=torch.long)
        label[:, 50:70, 50:70] = 3

        t_c0, t_lge, t_t2 = target_adapter_fn(label)
        mask_edema = (label == 3)

        assert torch.all(t_c0[mask_edema] == 1)
        assert torch.all(t_lge[mask_edema] == 1)  # Edema is non-scar myocardium for LGE
        assert torch.all(t_t2[mask_edema] == 2)  # Edema is edema-inclusive for T2

    def test_dimension_preservation(self, target_adapter_fn):
        """Verify target adapter handles both 2D (128, 128) and batched 3D (B, 128, 128)."""
        label_2d = torch.randint(0, 4, (128, 128), dtype=torch.long)
        t_c0_2d, t_lge_2d, t_t2_2d = target_adapter_fn(label_2d)
        assert t_c0_2d.shape == (128, 128)
        assert t_lge_2d.shape == (128, 128)
        assert t_t2_2d.shape == (128, 128)

        label_3d = torch.randint(0, 4, (8, 128, 128), dtype=torch.long)
        t_c0_3d, t_lge_3d, t_t2_3d = target_adapter_fn(label_3d)
        assert t_c0_3d.shape == (8, 128, 128)
        assert t_lge_3d.shape == (8, 128, 128)
        assert t_t2_3d.shape == (8, 128, 128)

    def test_prediction_fusion_roundtrip(self, target_adapter_fn):
        """Verify that fusing perfect target predictions reconstructs canonical 4-class labels."""
        # Canonical: 0: bg, 1: normal myo, 2: edema, 3: scar
        # Legacy:    0: bg, 1: normal myo, 2: scar, 3: edema
        legacy_label = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        t_c0, t_lge, t_t2 = target_adapter_fn(legacy_label)

        # One-hot representations
        p_c0 = ref_to_one_hot(t_c0, 2)
        p_lge = ref_to_one_hot(t_lge, 3)
        p_t2 = ref_to_one_hot(t_t2, 3)

        fused = fuse_predictions(p_c0, p_lge, p_t2)
        expected_canonical = torch.tensor([[0, 1], [3, 2]], dtype=torch.uint8)
        assert torch.equal(fused, expected_canonical), f"Fusion failed: got {fused.tolist()}"


# ==============================================================================
# 2. MyoPSNetL Architecture & Shape Invariants Suite
# ==============================================================================

class TestMyoPSNetLArchitecture:
    """Verifies 3-modality ingestion, multi-head output shapes, and probability axioms."""

    def test_model_instantiation_and_parameter_count(self):
        """Verify model instantiates with in_chs=(3, 2, 2) and out_chs=(2, 3, 3) (~119.3M params)."""
        model = MyoPSNetL(in_chs=(3, 2, 2), out_chs=(2, 3, 3))
        assert hasattr(model, "unet_C0"), "Missing unet_C0 submodule"
        assert hasattr(model, "encoder_LGE"), "Missing encoder_LGE submodule"
        assert hasattr(model, "decoder_LGE"), "Missing decoder_LGE submodule"
        assert hasattr(model, "encoder_T2"), "Missing encoder_T2 submodule"
        assert hasattr(model, "decoder_T2"), "Missing decoder_T2 submodule"

        # Mapping branch must NOT exist in MyoPSNetL
        assert not hasattr(model, "encoder_mapping"), "Orphaned encoder_mapping found"
        assert not hasattr(model, "decoder_mapping"), "Orphaned decoder_mapping found"

        total_params = sum(p.numel() for p in model.parameters())
        # Expected exact parameter count is 119,284,360
        assert total_params == 119284360, f"Unexpected parameter count: {total_params}"

    def test_forward_pass_output_shapes(self, synthetic_inputs):
        """Verify forward pass returns exactly 3 heads with specified channel dimensions."""
        c0, lge, t2 = synthetic_inputs
        model = MyoPSNetL(in_chs=(3, 2, 2), out_chs=(2, 3, 3))
        model.eval()

        with torch.no_grad():
            outputs = model(c0, lge, t2)

        assert isinstance(outputs, tuple) and len(outputs) == 3, "Forward must return 3-tuple"
        seg_c0, seg_lge, seg_t2 = outputs

        B = c0.shape[0]
        assert seg_c0.shape == (B, 2, 128, 128), f"seg_C0 shape mismatch: {seg_c0.shape}"
        assert seg_lge.shape == (B, 3, 128, 128), f"seg_LGE shape mismatch: {seg_lge.shape}"
        assert seg_t2.shape == (B, 3, 128, 128), f"seg_T2 shape mismatch: {seg_t2.shape}"

    def test_softmax_probability_axioms(self, synthetic_inputs):
        """Verify outputs are valid probability distributions in [0, 1] and sum to 1."""
        c0, lge, t2 = synthetic_inputs
        model = MyoPSNetL(in_chs=(3, 2, 2), out_chs=(2, 3, 3))
        model.eval()

        with torch.no_grad():
            seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)

        for name, head in [("seg_C0", seg_c0), ("seg_LGE", seg_lge), ("seg_T2", seg_t2)]:
            # Axiom 1: 0 <= p <= 1
            assert head.min() >= 0.0 - 1e-6, f"{name} min probability < 0"
            assert head.max() <= 1.0 + 1e-6, f"{name} max probability > 1"

            # Axiom 2: Sum over channel dimension must equal 1.0
            channel_sum = head.sum(dim=1)
            assert torch.allclose(channel_sum, torch.ones_like(channel_sum), atol=1e-5), (
                f"{name} probabilities do not sum to 1.0"
            )

            # Axiom 3: No NaN or Inf values
            assert torch.isfinite(head).all(), f"{name} contains NaN or Inf"

    def test_cpu_execution(self, synthetic_inputs):
        """Verify model executes cleanly on CPU without device mismatch."""
        c0, lge, t2 = synthetic_inputs
        model = MyoPSNetL(in_chs=(3, 2, 2), out_chs=(2, 3, 3)).to("cpu")
        model.eval()

        with torch.no_grad():
            out_c0, out_lge, out_t2 = model(c0.to("cpu"), lge.to("cpu"), t2.to("cpu"))

        assert out_c0.device.type == "cpu"
        assert out_lge.device.type == "cpu"
        assert out_t2.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_cuda_execution(self, synthetic_inputs):
        """Verify model executes cleanly on CUDA device without runtime errors."""
        c0, lge, t2 = synthetic_inputs
        device = torch.device("cuda")
        model = MyoPSNetL(in_chs=(3, 2, 2), out_chs=(2, 3, 3)).to(device)
        model.eval()

        with torch.no_grad():
            out_c0, out_lge, out_t2 = model(c0.to(device), lge.to(device), t2.to(device))

        assert out_c0.device.type == "cuda"
        assert out_lge.device.type == "cuda"
        assert out_t2.device.type == "cuda"


# ==============================================================================
# 3. Arbitrary Batch Sizes Suite (Boundary & Remainder Handling)
# ==============================================================================

class TestBatchSizeFlexibility:
    """Verifies that MyoPSNetL supports standard and remainder batch sizes."""

    @pytest.mark.parametrize("batch_size", [16, 4, 1, 7])
    def test_forward_with_various_batch_sizes(self, batch_size):
        """
        Tests forward pass on:
        - B = 16 (standard training batch size)
        - B = 4  (boundary remainder batch in MyoPS380: 1972 = 123 * 16 + 4)
        - B = 1  (single slice inference)
        - B = 7  (arbitrary odd batch)
        """
        # Run on CPU for fast testing of various shapes
        model = MyoPSNetL(in_chs=(3, 2, 2), out_chs=(2, 3, 3)).to("cpu")
        model.eval()

        c0 = torch.randn(batch_size, 1, 128, 128)
        lge = torch.randn(batch_size, 1, 128, 128)
        t2 = torch.randn(batch_size, 1, 128, 128)

        with torch.no_grad():
            seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)

        assert seg_c0.shape[0] == batch_size
        assert seg_lge.shape[0] == batch_size
        assert seg_t2.shape[0] == batch_size


# ==============================================================================
# 4. Loss Function Hardening & Stability Suite
# ==============================================================================

class TestLossFunctionStability:
    """Verifies that WCELoss, DiceLoss, and MyoPSLoss are hardened against zero classes."""

    def test_wce_loss_dimension_robustness(self):
        """
        Crucial bug fix test: Native WCELoss crashed when a class was absent
        because range(mask.max() + 1) yielded fewer weights than target channels.
        """
        wce_loss = RobustWCELoss()
        pred = torch.softmax(torch.randn(4, 3, 128, 128), dim=1)

        # Target has only classes 0 and 1; class 2 (scar) is COMPLETELY ABSENT
        target = torch.randint(0, 2, (4, 128, 128), dtype=torch.long)
        target_onehot = ref_to_one_hot(target, num_classes=3)

        # Must compute without size mismatch or runtime error
        loss = wce_loss(pred, target_onehot)
        assert torch.isfinite(loss), f"WCE loss produced NaN/Inf: {loss.item()}"
        assert loss.item() >= 0.0, f"WCE loss must be non-negative: {loss.item()}"

    def test_wce_loss_all_background(self):
        """Verify WCELoss when entire batch is background (0 target voxels for non-bg)."""
        wce_loss = RobustWCELoss()
        pred = torch.softmax(torch.randn(2, 3, 128, 128), dim=1)
        target = torch.zeros((2, 128, 128), dtype=torch.long)
        target_onehot = ref_to_one_hot(target, num_classes=3)

        loss = wce_loss(pred, target_onehot)
        assert torch.isfinite(loss), f"WCE loss produced NaN/Inf on empty batch: {loss.item()}"

    def test_dice_loss_empty_classes(self):
        """Verify MultiClassDiceLoss produces finite, bounded values when classes are absent."""
        dice_loss = DiceLoss()
        pred = torch.softmax(torch.randn(4, 3, 128, 128), dim=1)
        target = torch.zeros((4, 128, 128), dtype=torch.long)  # All background
        target_onehot = ref_to_one_hot(target, num_classes=3)

        loss = dice_loss(pred, target_onehot)
        assert torch.isfinite(loss), f"Dice loss produced NaN/Inf: {loss.item()}"
        assert 0.0 <= loss.item() <= 1.0, f"Dice loss out of bounds: {loss.item()}"

    def test_invariant_loss_identity_and_bounds(self):
        """Verify InvariantLoss is 0 for identical features and bounded in [0, 2]."""
        inv_loss = InvariantLoss()
        feat = torch.rand(2, 2, 128, 128)

        # Identity test: L(feat, feat) == 0.0
        loss_identical = inv_loss(feat, feat)
        assert loss_identical.item() < 1e-5, f"Invariant loss not zero for identical inputs: {loss_identical.item()}"

        # Boundedness test
        feat_other = torch.rand(2, 2, 128, 128)
        loss_rand = inv_loss(feat, feat_other)
        assert 0.0 <= loss_rand.item() <= 2.0, f"Invariant loss out of [0, 2]: {loss_rand.item()}"

    def test_inclusive_loss_scar_empty_edema(self):
        """Verify InclusiveLossScar does not divide by zero when ground truth edema is 0."""
        loss_fn = InclusiveLossScar()
        p_scar = torch.rand(4, 128, 128)
        gd_edema_empty = torch.zeros(4, 128, 128)

        loss = loss_fn(p_scar, gd_edema_empty)
        assert torch.isfinite(loss), f"InclusiveLossScar NaN on empty edema: {loss.item()}"

    def test_inclusive_loss_edema_empty_scar(self):
        """Verify InclusiveLossEdema evaluates cleanly to 0.0 when ground truth scar is 0."""
        loss_fn = InclusiveLossEdema()
        p_edema = torch.rand(4, 128, 128)
        gd_scar_empty = torch.zeros(4, 128, 128)

        loss = loss_fn(p_edema, gd_scar_empty)
        assert torch.isfinite(loss), f"InclusiveLossEdema NaN on empty scar: {loss.item()}"
        assert loss.item() < 1e-4, f"InclusiveLossEdema must be ~0 when no scar: {loss.item()}"

    @pytest.mark.parametrize(
        "batch_type",
        ["all_classes", "zero_scar", "zero_edema", "zero_pathology", "all_background"],
    )
    def test_composite_myops_loss_pathological_batches(self, batch_type):
        """
        Verifies that composite MyoPSLoss never produces NaN or Inf across
        all 5 distinct clinical data scenarios:
        1. all_classes: {0, 1, 2, 3}
        2. zero_scar: {0, 1, 3} (0 scar voxels)
        3. zero_edema: {0, 1, 2} (0 edema voxels)
        4. zero_pathology: {0, 1} (healthy normal myocardium only)
        5. all_background: {0} (empty non-cardiac slices)
        """
        B = 2
        torch.manual_seed(123)

        if batch_type == "all_classes":
            lbl = torch.randint(0, 4, (B, 128, 128), dtype=torch.long)
        elif batch_type == "zero_scar":
            lbl = torch.randint(0, 3, (B, 128, 128), dtype=torch.long)
            lbl[lbl == 2] = 3  # replace scar with edema
        elif batch_type == "zero_edema":
            lbl = torch.randint(0, 3, (B, 128, 128), dtype=torch.long)  # only {0, 1, 2}
        elif batch_type == "zero_pathology":
            lbl = torch.randint(0, 2, (B, 128, 128), dtype=torch.long)  # only {0, 1}
        elif batch_type == "all_background":
            lbl = torch.zeros((B, 128, 128), dtype=torch.long)

        t_c0, t_lge, t_t2 = ref_create_multi_head_targets(lbl)
        t_c0_oh = ref_to_one_hot(t_c0, 2)
        t_lge_oh = ref_to_one_hot(t_lge, 3)
        t_t2_oh = ref_to_one_hot(t_t2, 3)

        seg_c0 = torch.softmax(torch.randn(B, 2, 128, 128), dim=1)
        seg_lge = torch.softmax(torch.randn(B, 3, 128, 128), dim=1)
        seg_t2 = torch.softmax(torch.randn(B, 3, 128, 128), dim=1)

        criterion = MyoPSLoss()

        # Handle both dict-based and arg-based forward signatures
        try:
            total_loss, l_seg, l_inv, l_inc = criterion(
                seg_c0, seg_lge, seg_t2, t_c0_oh, t_lge_oh, t_t2_oh
            )
        except TypeError:
            seg_dict = {"C0": seg_c0, "LGE": seg_lge, "T2": seg_t2}
            label_dict = {"cardiac": t_c0_oh, "scar": t_lge_oh, "edema": t_t2_oh}
            l_seg, l_inv, l_inc, total_loss = criterion(seg_dict, label_dict)

        assert torch.isfinite(total_loss), f"Total loss NaN/Inf on {batch_type}: {total_loss.item()}"
        assert torch.isfinite(l_seg), f"Seg loss NaN/Inf on {batch_type}: {l_seg.item()}"
        assert torch.isfinite(l_inv), f"Invariant loss NaN/Inf on {batch_type}: {l_inv.item()}"
        assert torch.isfinite(l_inc), f"Inclusive loss NaN/Inf on {batch_type}: {l_inc.item()}"
        assert total_loss.item() >= 0.0, f"Total loss must be non-negative: {total_loss.item()}"


# ==============================================================================
# 5. Gradient Propagation & End-to-End Training Dynamics Suite
# ==============================================================================

class TestGradientFlowAndTrainingDynamics:
    """Verifies that gradients propagate to ALL model parameters without vanishing or NaNs."""

    def test_backward_pass_all_parameters_receive_gradients(self, synthetic_inputs):
        """
        CRITICAL TEST: Ensures no detached or disconnected subnetworks.
        Verifies that every single parameter in unet_C0, encoder_LGE, decoder_LGE,
        encoder_T2, and decoder_T2 receives non-zero, finite gradients.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MyoPSNetL(in_chs=(3, 2, 2), out_chs=(2, 3, 3)).to(device)
        model.train()

        criterion = MyoPSLoss().to(device)

        c0, lge, t2 = [x.to(device) for x in synthetic_inputs]
        B = c0.shape[0]

        # Target containing all classes
        lbl = torch.randint(0, 4, (B, 128, 128), device=device, dtype=torch.long)
        t_c0, t_lge, t_t2 = ref_create_multi_head_targets(lbl)
        t_c0_oh = ref_to_one_hot(t_c0, 2)
        t_lge_oh = ref_to_one_hot(t_lge, 3)
        t_t2_oh = ref_to_one_hot(t_t2, 3)

        model.zero_grad()
        seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)

        try:
            total_loss, _, _, _ = criterion(seg_c0, seg_lge, seg_t2, t_c0_oh, t_lge_oh, t_t2_oh)
        except TypeError:
            seg_dict = {"C0": seg_c0, "LGE": seg_lge, "T2": seg_t2}
            label_dict = {"cardiac": t_c0_oh, "scar": t_lge_oh, "edema": t_t2_oh}
            _, _, _, total_loss = criterion(seg_dict, label_dict)

        total_loss.backward()

        no_grad_params = []
        nan_grad_params = []
        zero_grad_params = []

        for name, p in model.named_parameters():
            if p.grad is None:
                no_grad_params.append(name)
            elif torch.isnan(p.grad).any():
                nan_grad_params.append(name)
            elif p.grad.abs().sum() == 0:
                zero_grad_params.append(name)

        assert len(no_grad_params) == 0, f"Parameters with None grad ({len(no_grad_params)}): {no_grad_params[:5]}"
        assert len(nan_grad_params) == 0, f"Parameters with NaN grad ({len(nan_grad_params)}): {nan_grad_params[:5]}"
        assert len(zero_grad_params) == 0, f"Parameters with Zero grad ({len(zero_grad_params)}): {zero_grad_params[:5]}"

    def test_backward_pass_on_zero_pathology_batch(self, synthetic_inputs):
        """Verify backward pass does not produce NaN gradients when scar and edema are 0."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MyoPSNetL(in_chs=(3, 2, 2), out_chs=(2, 3, 3)).to(device)
        model.train()
        criterion = MyoPSLoss().to(device)

        c0, lge, t2 = [x.to(device) for x in synthetic_inputs]
        B = c0.shape[0]

        # Slices with only normal myocardium and background
        lbl = torch.randint(0, 2, (B, 128, 128), device=device, dtype=torch.long)
        t_c0, t_lge, t_t2 = ref_create_multi_head_targets(lbl)
        t_c0_oh = ref_to_one_hot(t_c0, 2)
        t_lge_oh = ref_to_one_hot(t_lge, 3)
        t_t2_oh = ref_to_one_hot(t_t2, 3)

        model.zero_grad()
        seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)

        try:
            total_loss, _, _, _ = criterion(seg_c0, seg_lge, seg_t2, t_c0_oh, t_lge_oh, t_t2_oh)
        except TypeError:
            seg_dict = {"C0": seg_c0, "LGE": seg_lge, "T2": seg_t2}
            label_dict = {"cardiac": t_c0_oh, "scar": t_lge_oh, "edema": t_t2_oh}
            _, _, _, total_loss = criterion(seg_dict, label_dict)

        total_loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no grad"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN grad on zero-pathology batch"
            assert not torch.isinf(p.grad).any(), f"Parameter {name} has Inf grad on zero-pathology batch"

    def test_optimizer_parameter_update(self, synthetic_inputs):
        """Verify an AdamW optimizer step successfully modifies model weights."""
        device = torch.device("cpu")
        model = MyoPSNetL(in_chs=(3, 2, 2), out_chs=(2, 3, 3)).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = MyoPSLoss().to(device)

        c0, lge, t2 = [x.to(device) for x in synthetic_inputs]
        B = c0.shape[0]
        lbl = torch.randint(0, 4, (B, 128, 128), device=device, dtype=torch.long)
        t_c0, t_lge, t_t2 = ref_create_multi_head_targets(lbl)
        t_c0_oh = ref_to_one_hot(t_c0, 2)
        t_lge_oh = ref_to_one_hot(t_lge, 3)
        t_t2_oh = ref_to_one_hot(t_t2, 3)

        # Snapshot a subset of weights before step
        param_sample = model.unet_C0.Conv1.conv[0].weight.clone()

        optimizer.zero_grad()
        seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)
        try:
            total_loss, _, _, _ = criterion(seg_c0, seg_lge, seg_t2, t_c0_oh, t_lge_oh, t_t2_oh)
        except TypeError:
            seg_dict = {"C0": seg_c0, "LGE": seg_lge, "T2": seg_t2}
            label_dict = {"cardiac": t_c0_oh, "scar": t_lge_oh, "edema": t_t2_oh}
            _, _, _, total_loss = criterion(seg_dict, label_dict)

        total_loss.backward()
        optimizer.step()

        param_after = model.unet_C0.Conv1.conv[0].weight
        diff = torch.norm(param_after - param_sample).item()
        assert diff > 0.0, "Weights failed to update after optimizer step"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
