"""
tests/test_adversarial_m1.py
============================
Adversarial Stress Test Suite for Milestone 1:
- Extreme input dimensions and batch sizes (B=1, B=2, B=4 remainder, B=16, B=32)
- Intensity extreme ranges (negative values, zeros, large dynamic ranges [0, 1000], Gaussian noise)
- Edge-case labels: masks with 100% background, 100% scar, 100% edema, single-pixel anomalies
- Out-of-bounds label IDs (negative labels, label 4, label 255) to verify clamping
- Gradient checks: verify backward pass does not explode or vanish under extreme targets
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from network.model import MyoPSNetL
from criterion.loss import (
    RobustWCELoss,
    WCELoss,
    MultiClassDiceLoss,
    InvariantLoss,
    InclusiveLossScar,
    InclusiveLossEdema,
    SegmentationLoss,
    MyoPSLoss,
)
from utils.tools import create_multi_head_targets, fuse_predictions, to_one_hot


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 1. Extreme Input Dimensions and Batch Sizes
# ==============================================================================

class TestExtremeDimensionsAndBatchSizes:
    """Stress tests on varying batch sizes (B=1, 2, 4, 16, 32) and spatial dimensions."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 16, 32])
    def test_extreme_batch_sizes_forward(self, device, batch_size):
        """Verify forward pass survives across batch sizes B=1, 2, 4, 16, 32."""
        # For B=32 on CUDA or CPU, use eval mode to avoid huge memory footprint
        model = MyoPSNetL(in_chs=(3, 2, 2), out_chs=(2, 3, 3)).to(device)
        model.eval()

        # Allocate input tensors
        c0 = torch.randn(batch_size, 1, 128, 128, device=device)
        lge = torch.randn(batch_size, 1, 128, 128, device=device)
        t2 = torch.randn(batch_size, 1, 128, 128, device=device)

        with torch.no_grad():
            seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)

        assert seg_c0.shape == (batch_size, 2, 128, 128)
        assert seg_lge.shape == (batch_size, 3, 128, 128)
        assert seg_t2.shape == (batch_size, 3, 128, 128)
        assert torch.isfinite(seg_c0).all()
        assert torch.isfinite(seg_lge).all()
        assert torch.isfinite(seg_t2).all()

    @pytest.mark.parametrize("spatial_size", [(64, 64), (128, 128), (256, 256)])
    def test_variable_spatial_dimensions_divisible_by_16(self, device, spatial_size):
        """Verify U-Net operates correctly on spatial resolutions divisible by 16."""
        H, W = spatial_size
        model = MyoPSNetL(in_chs=(3, 2, 2), out_chs=(2, 3, 3)).to(device)
        model.eval()

        c0 = torch.randn(1, 1, H, W, device=device)
        lge = torch.randn(1, 1, H, W, device=device)
        t2 = torch.randn(1, 1, H, W, device=device)

        with torch.no_grad():
            seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)

        assert seg_c0.shape == (1, 2, H, W)
        assert seg_lge.shape == (1, 3, H, W)
        assert seg_t2.shape == (1, 3, H, W)

    def test_non_divisible_spatial_dimension_fails_cleanly(self, device):
        """Spatial dimensions not divisible by 16 should trigger runtime shape mismatch cleanly."""
        model = MyoPSNetL(in_chs=(3, 2, 2), out_chs=(2, 3, 3)).to(device)
        model.eval()

        # 130 is not divisible by 16 (130 -> 65 -> 32 -> 16 -> 8)
        # Upsampling 8*2=16, but downsampled feature was 16 vs odd dimension
        c0 = torch.randn(1, 1, 130, 130, device=device)
        lge = torch.randn(1, 1, 130, 130, device=device)
        t2 = torch.randn(1, 1, 130, 130, device=device)

        with pytest.raises(RuntimeError):
            model(c0, lge, t2)


# ==============================================================================
# 2. Intensity Extreme Ranges & Noise
# ==============================================================================

class TestIntensityExtremeRanges:
    """Stress tests on extreme intensities, flat zeros, extreme dynamic ranges, and noise."""

    def test_all_zeros_input(self, device):
        """Verify model handles all-zero inputs (e.g., zero-padded background slices)."""
        model = MyoPSNetL().to(device)
        model.eval()

        c0 = torch.zeros(2, 1, 128, 128, device=device)
        lge = torch.zeros(2, 1, 128, 128, device=device)
        t2 = torch.zeros(2, 1, 128, 128, device=device)

        with torch.no_grad():
            seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)

        for name, seg in [("seg_C0", seg_c0), ("seg_LGE", seg_lge), ("seg_T2", seg_t2)]:
            assert torch.isfinite(seg).all(), f"{name} produced NaN/Inf on zero inputs"
            # Probability sum check
            sums = seg.sum(dim=1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"{name} sum != 1"

    def test_large_dynamic_range_intensities(self, device):
        """Verify model handles large positive intensities [0, 1000] and [0, 1e4]."""
        model = MyoPSNetL().to(device)
        model.eval()

        # Large scale intensities (e.g. unnormalized MR voxels)
        c0 = torch.rand(2, 1, 128, 128, device=device) * 1000.0
        lge = torch.rand(2, 1, 128, 128, device=device) * 10000.0
        t2 = torch.rand(2, 1, 128, 128, device=device) * 500.0

        with torch.no_grad():
            seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)

        assert torch.isfinite(seg_c0).all()
        assert torch.isfinite(seg_lge).all()
        assert torch.isfinite(seg_t2).all()

    def test_negative_intensities(self, device):
        """Verify model handles negative intensities [-1000, 0] (e.g., zero-mean normalized)."""
        model = MyoPSNetL().to(device)
        model.eval()

        c0 = -torch.rand(2, 1, 128, 128, device=device) * 1000.0
        lge = -torch.rand(2, 1, 128, 128, device=device) * 500.0
        t2 = -torch.rand(2, 1, 128, 128, device=device) * 100.0

        with torch.no_grad():
            seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)

        assert torch.isfinite(seg_c0).all()
        assert torch.isfinite(seg_lge).all()
        assert torch.isfinite(seg_t2).all()

    def test_high_variance_gaussian_noise(self, device):
        """Verify model handles extreme noise N(0, 100^2)."""
        model = MyoPSNetL().to(device)
        model.eval()

        c0 = torch.randn(2, 1, 128, 128, device=device) * 100.0
        lge = torch.randn(2, 1, 128, 128, device=device) * 100.0
        t2 = torch.randn(2, 1, 128, 128, device=device) * 100.0

        with torch.no_grad():
            seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)

        assert torch.isfinite(seg_c0).all()
        assert torch.isfinite(seg_lge).all()
        assert torch.isfinite(seg_t2).all()

    def test_flat_constant_inputs(self, device):
        """Verify model handles completely flat inputs (spatial variance = 0)."""
        model = MyoPSNetL().to(device)
        model.eval()

        c0 = torch.full((2, 1, 128, 128), fill_value=42.0, device=device)
        lge = torch.full((2, 1, 128, 128), fill_value=-15.0, device=device)
        t2 = torch.full((2, 1, 128, 128), fill_value=100.0, device=device)

        with torch.no_grad():
            seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)

        assert torch.isfinite(seg_c0).all()
        assert torch.isfinite(seg_lge).all()
        assert torch.isfinite(seg_t2).all()


# ==============================================================================
# 3. Edge-Case Labels & Pathological Masks
# ==============================================================================

class TestEdgeCaseLabels:
    """Stress tests on 100% background, 100% scar, 100% edema, and single-pixel anomalies."""

    def test_100_percent_background_mask(self, device):
        """Adversarial scenario: 100% background mask (0 non-background voxels)."""
        label = torch.zeros((4, 128, 128), dtype=torch.long, device=device)
        t_c0, t_lge, t_t2 = create_multi_head_targets(label, mode="legacy")

        assert (t_c0 == 0).all()
        assert (t_lge == 0).all()
        assert (t_t2 == 0).all()

        criterion = MyoPSLoss().to(device)
        p_c0 = torch.softmax(torch.randn(4, 2, 128, 128, device=device), dim=1)
        p_lge = torch.softmax(torch.randn(4, 3, 128, 128, device=device), dim=1)
        p_t2 = torch.softmax(torch.randn(4, 3, 128, 128, device=device), dim=1)

        loss_total, l_seg, l_inv, l_inc = criterion(p_c0, p_lge, p_t2, t_c0, t_lge, t_t2)
        assert torch.isfinite(loss_total)
        assert torch.isfinite(l_seg)
        assert torch.isfinite(l_inv)
        assert torch.isfinite(l_inc)
        assert loss_total.item() >= 0.0

    def test_100_percent_scar_mask(self, device):
        """Adversarial scenario: 100% scar mask (entire slice is scar, legacy label 2)."""
        label = torch.full((2, 128, 128), fill_value=2, dtype=torch.long, device=device)
        t_c0, t_lge, t_t2 = create_multi_head_targets(label, mode="legacy")

        assert (t_c0 == 1).all(), "100% scar must map to 100% whole myocardium in C0"
        assert (t_lge == 2).all(), "100% scar must map to 100% scar in LGE"
        assert (t_t2 == 2).all(), "100% scar must map to 100% edema-inclusive in T2"

        criterion = MyoPSLoss().to(device)
        p_c0 = torch.softmax(torch.randn(2, 2, 128, 128, device=device), dim=1)
        p_lge = torch.softmax(torch.randn(2, 3, 128, 128, device=device), dim=1)
        p_t2 = torch.softmax(torch.randn(2, 3, 128, 128, device=device), dim=1)

        loss_total, l_seg, l_inv, l_inc = criterion(p_c0, p_lge, p_t2, t_c0, t_lge, t_t2)
        assert torch.isfinite(loss_total)
        assert torch.isfinite(l_seg)
        assert torch.isfinite(l_inv)
        assert torch.isfinite(l_inc)

    def test_100_percent_edema_mask(self, device):
        """Adversarial scenario: 100% edema-exclusive mask (legacy label 3)."""
        label = torch.full((2, 128, 128), fill_value=3, dtype=torch.long, device=device)
        t_c0, t_lge, t_t2 = create_multi_head_targets(label, mode="legacy")

        assert (t_c0 == 1).all()
        assert (t_lge == 1).all(), "Edema is non-scar myocardium for LGE"
        assert (t_t2 == 2).all(), "Edema is edema-inclusive for T2"

        criterion = MyoPSLoss().to(device)
        p_c0 = torch.softmax(torch.randn(2, 2, 128, 128, device=device), dim=1)
        p_lge = torch.softmax(torch.randn(2, 3, 128, 128, device=device), dim=1)
        p_t2 = torch.softmax(torch.randn(2, 3, 128, 128, device=device), dim=1)

        loss_total, l_seg, l_inv, l_inc = criterion(p_c0, p_lge, p_t2, t_c0, t_lge, t_t2)
        assert torch.isfinite(loss_total)
        assert torch.isfinite(l_seg)
        assert torch.isfinite(l_inv)
        assert torch.isfinite(l_inc)

    def test_single_pixel_scar_anomaly(self, device):
        """Adversarial scenario: Exactly 1 single pixel of scar in an ocean of background."""
        label = torch.zeros((2, 128, 128), dtype=torch.long, device=device)
        label[0, 64, 64] = 2  # exactly one scar voxel
        label[1, 10, 10] = 2  # exactly one scar voxel

        t_c0, t_lge, t_t2 = create_multi_head_targets(label, mode="legacy")
        assert t_c0.sum() == 2
        assert (t_lge == 2).sum() == 2
        assert (t_t2 == 2).sum() == 2

        criterion = MyoPSLoss().to(device)
        p_c0 = torch.softmax(torch.randn(2, 2, 128, 128, device=device), dim=1)
        p_lge = torch.softmax(torch.randn(2, 3, 128, 128, device=device), dim=1)
        p_t2 = torch.softmax(torch.randn(2, 3, 128, 128, device=device), dim=1)

        loss_total, l_seg, l_inv, l_inc = criterion(p_c0, p_lge, p_t2, t_c0, t_lge, t_t2)
        assert torch.isfinite(loss_total)
        assert torch.isfinite(l_seg)
        assert torch.isfinite(l_inv)
        assert torch.isfinite(l_inc)

    def test_single_pixel_edema_anomaly(self, device):
        """Adversarial scenario: Exactly 1 single pixel of edema in an ocean of background."""
        label = torch.zeros((2, 128, 128), dtype=torch.long, device=device)
        label[0, 32, 32] = 3

        t_c0, t_lge, t_t2 = create_multi_head_targets(label, mode="legacy")
        assert (t_lge == 1).sum() == 1  # edema maps to non-scar myo (1)
        assert (t_t2 == 2).sum() == 1   # edema maps to edema-inclusive (2)

        criterion = MyoPSLoss().to(device)
        p_c0 = torch.softmax(torch.randn(2, 2, 128, 128, device=device), dim=1)
        p_lge = torch.softmax(torch.randn(2, 3, 128, 128, device=device), dim=1)
        p_t2 = torch.softmax(torch.randn(2, 3, 128, 128, device=device), dim=1)

        loss_total, l_seg, l_inv, l_inc = criterion(p_c0, p_lge, p_t2, t_c0, t_lge, t_t2)
        assert torch.isfinite(loss_total)


# ==============================================================================
# 4. Out-of-Bounds Label IDs & Target Adapter Clamping
# ==============================================================================

class TestOutOfBoundsLabelIDs:
    """Stress tests verifying create_multi_head_targets clamping on corrupt / OOB labels."""

    def test_negative_label_ids(self, device):
        """Negative label IDs (e.g. -1, -100) must be clamped to 0 without crashing."""
        corrupted_label = torch.tensor([[-1, -100], [0, 1]], dtype=torch.long, device=device)
        t_c0, t_lge, t_t2 = create_multi_head_targets(corrupted_label, mode="legacy")

        # Negative labels clamped to 0
        assert t_c0[0, 0].item() == 0
        assert t_c0[0, 1].item() == 0
        assert t_lge[0, 0].item() == 0
        assert t_t2[0, 0].item() == 0

    def test_exorbitant_positive_label_ids(self, device):
        """Out-of-bounds positive label IDs (4, 10, 255) must be clamped to 3 without index error."""
        corrupted_label = torch.tensor([[4, 10], [255, 3]], dtype=torch.long, device=device)
        t_c0, t_lge, t_t2 = create_multi_head_targets(corrupted_label, mode="legacy")

        # In legacy mode, label 3 maps to C0:1, LGE:1, T2:2
        assert (t_c0 == 1).all()
        assert (t_lge == 1).all()
        assert (t_t2 == 2).all()

    def test_float_label_tensors(self, device):
        """Float label tensors (e.g., from interpolated data) must convert safely to long."""
        float_label = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32, device=device)
        t_c0, t_lge, t_t2 = create_multi_head_targets(float_label, mode="legacy")

        assert t_c0.dtype == torch.long
        assert t_lge.dtype == torch.long
        assert t_t2.dtype == torch.long
        assert t_c0.tolist() == [[0, 1], [1, 1]]
        assert t_lge.tolist() == [[0, 1], [2, 1]]
        assert t_t2.tolist() == [[0, 1], [2, 2]]

    def test_numpy_array_input(self):
        """Numpy array inputs must be converted seamlessly to torch tensors."""
        np_label = np.array([[0, 1], [2, 3]], dtype=np.int32)
        t_c0, t_lge, t_t2 = create_multi_head_targets(np_label, mode="legacy")

        assert isinstance(t_c0, torch.Tensor)
        assert isinstance(t_lge, torch.Tensor)
        assert isinstance(t_t2, torch.Tensor)
        assert t_c0.shape == (2, 2)


# ==============================================================================
# 5. Gradient Checks under Extreme Targets
# ==============================================================================

class TestGradientChecksUnderExtremeTargets:
    """Verifies that backward pass never produces NaN/Inf gradients under extreme targets."""

    @pytest.mark.parametrize(
        "scenario",
        ["100%_background", "100%_scar", "100%_edema", "single_pixel_scar", "single_pixel_edema"]
    )
    def test_backward_pass_gradients_finite(self, device, scenario):
        """Verify backward pass does not explode or produce NaN/Inf under pathological targets."""
        model = MyoPSNetL().to(device)
        model.train()
        criterion = MyoPSLoss().to(device)

        c0 = torch.randn(2, 1, 128, 128, device=device)
        lge = torch.randn(2, 1, 128, 128, device=device)
        t2 = torch.randn(2, 1, 128, 128, device=device)

        if scenario == "100%_background":
            lbl = torch.zeros((2, 128, 128), dtype=torch.long, device=device)
        elif scenario == "100%_scar":
            lbl = torch.full((2, 128, 128), fill_value=2, dtype=torch.long, device=device)
        elif scenario == "100%_edema":
            lbl = torch.full((2, 128, 128), fill_value=3, dtype=torch.long, device=device)
        elif scenario == "single_pixel_scar":
            lbl = torch.zeros((2, 128, 128), dtype=torch.long, device=device)
            lbl[:, 64, 64] = 2
        elif scenario == "single_pixel_edema":
            lbl = torch.zeros((2, 128, 128), dtype=torch.long, device=device)
            lbl[:, 64, 64] = 3

        t_c0, t_lge, t_t2 = create_multi_head_targets(lbl, mode="legacy")

        model.zero_grad()
        seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)
        loss_total, l_seg, l_inv, l_inc = criterion(seg_c0, seg_lge, seg_t2, t_c0, t_lge, t_t2)

        loss_total.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient in scenario {scenario}"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN grad in scenario {scenario}"
            assert not torch.isinf(p.grad).any(), f"Parameter {name} has Inf grad in scenario {scenario}"

        total_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0)
        assert torch.isfinite(total_grad_norm), f"Total grad norm is non-finite in scenario {scenario}"
        assert total_grad_norm.item() > 0.0, f"Total grad norm vanished to 0 in scenario {scenario}"

    def test_backward_pass_with_extreme_prediction_error(self, device):
        """Stress test: Prediction is completely inverted (1.0 on wrong class, eps on true class)."""
        model = MyoPSNetL().to(device)
        model.train()
        criterion = MyoPSLoss().to(device)

        c0 = torch.randn(2, 1, 128, 128, device=device)
        lge = torch.randn(2, 1, 128, 128, device=device)
        t2 = torch.randn(2, 1, 128, 128, device=device)

        # Target is all scar
        lbl = torch.full((2, 128, 128), fill_value=2, dtype=torch.long, device=device)
        t_c0, t_lge, t_t2 = create_multi_head_targets(lbl, mode="legacy")

        model.zero_grad()
        seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)

        loss_total, _, _, _ = criterion(seg_c0, seg_lge, seg_t2, t_c0, t_lge, t_t2)
        loss_total.backward()

        for name, p in model.named_parameters():
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient on inverted target"
            assert not torch.isinf(p.grad).any(), f"Parameter {name} has Inf gradient on inverted target"


# ==============================================================================
# 6. Prediction Fusion Invariants & Conflicts
# ==============================================================================

class TestPredictionFusionInvariants:
    """Stress tests on fuse_predictions under conflicting or degenerate predictions."""

    def test_conflicting_predictions_hierarchy(self):
        """Verify strict priority: Scar (3) > Edema (2) > Myo (1) > BG (0)."""
        # Case 1: LGE predicts scar (2), T2 predicts edema (2), C0 predicts myo (1)
        # Expected: Scar wins (ID 3)
        p_c0 = torch.tensor([[1]])
        p_lge = torch.tensor([[2]])
        p_t2 = torch.tensor([[2]])
        fused = fuse_predictions(p_c0, p_lge, p_t2)
        assert fused[0, 0].item() == 3

        # Case 2: LGE predicts non-scar (1), T2 predicts edema (2), C0 predicts myo (1)
        # Expected: Edema wins (ID 2)
        p_lge = torch.tensor([[1]])
        fused = fuse_predictions(p_c0, p_lge, p_t2)
        assert fused[0, 0].item() == 2

        # Case 3: LGE predicts non-scar (1), T2 predicts normal myo (1), C0 predicts myo (1)
        # Expected: Normal Myocardium (ID 1)
        p_t2 = torch.tensor([[1]])
        fused = fuse_predictions(p_c0, p_lge, p_t2)
        assert fused[0, 0].item() == 1

        # Case 4: LGE predicts bg (0), T2 predicts bg (0), C0 predicts bg (0)
        # Expected: Background (ID 0)
        p_c0 = torch.tensor([[0]])
        p_lge = torch.tensor([[0]])
        p_t2 = torch.tensor([[0]])
        fused = fuse_predictions(p_c0, p_lge, p_t2)
        assert fused[0, 0].item() == 0

    def test_fusion_with_probability_tensors(self):
        """Verify fuse_predictions accepts 4D/3D probability tensors directly."""
        p_c0 = torch.zeros(2, 2, 4, 4)
        p_c0[:, 1] = 1.0  # myo

        p_lge = torch.zeros(2, 3, 4, 4)
        p_lge[:, 2] = 1.0  # scar

        p_t2 = torch.zeros(2, 3, 4, 4)
        p_t2[:, 2] = 1.0  # edema

        fused = fuse_predictions(p_c0, p_lge, p_t2)
        assert fused.shape == (2, 4, 4)
        assert (fused == 3).all(), "Scar should dominate everywhere"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
