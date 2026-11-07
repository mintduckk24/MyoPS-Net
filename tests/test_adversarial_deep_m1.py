"""
tests/test_adversarial_deep_m1.py
=================================
Deep adversarial stress tests:
1. AMP autocast + GradScaler end-to-end training step under extreme pathology
2. Alternative calling conventions (stacked (B, 3, H, W) and 3-tuple)
3. Return features verification
4. Inverse weight mode and from_logits in WCELoss
5. Multi-head target adapter canonical mode under OOB labels
6. Peak VRAM profiling under AMP for B=16 training step
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from network.model import MyoPSNetL, MyoPSNet
from criterion.loss import WCELoss, RobustWCELoss, MyoPSLoss, SegmentationLoss
from utils.tools import create_multi_head_targets, fuse_predictions


def test_amp_autocast_and_grad_scaler_pathological():
    """Verify that AMP autocast + GradScaler training step succeeds on CUDA without underflow."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    model = MyoPSNetL().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.333e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', init_scale=32768.0)
    criterion = MyoPSLoss().to(device)

    B = 16
    c0 = torch.randn(B, 1, 128, 128, device=device)
    lge = torch.randn(B, 1, 128, 128, device=device)
    t2 = torch.randn(B, 1, 128, 128, device=device)

    # Pathological label: mostly background with tiny scar and edema
    lbl = torch.zeros(B, 128, 128, dtype=torch.long, device=device)
    lbl[0, 60:65, 60:65] = 2  # tiny scar
    lbl[1, 70:75, 70:75] = 3  # tiny edema

    t_c0, t_lge, t_t2 = create_multi_head_targets(lbl, mode="legacy")

    # Snapshot parameter to test update
    param_before = model.encoder_T2.Conv1.conv[0].weight.clone()

    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)
        total_loss, l_seg, l_inv, l_inc = criterion(seg_c0, seg_lge, seg_t2, t_c0, t_lge, t_t2)

    assert torch.isfinite(total_loss), "Loss must be finite under AMP"
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)

    # Verify unscaled gradients are finite
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name} under AMP"
            assert not torch.isinf(p.grad).any(), f"Inf grad in {name} under AMP"

    scaler.step(optimizer)
    scaler.update()

    # Verify weights updated
    param_after = model.encoder_T2.Conv1.conv[0].weight
    diff = torch.norm(param_after - param_before).item()
    assert diff > 0.0, "Weights failed to update under AMP"


def test_alternative_calling_conventions():
    """Verify MyoPSNetL handles 3-channel stacked tensor and 3-element tuple."""
    model = MyoPSNetL()
    model.eval()

    # Convention 1: Single stacked tensor (B, 3, H, W)
    stacked = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out1 = model(stacked)
    assert len(out1) == 3

    # Convention 2: 3-element tuple
    tup = (torch.randn(2, 1, 128, 128), torch.randn(2, 1, 128, 128), torch.randn(2, 1, 128, 128))
    with torch.no_grad():
        out2 = model(tup)
    assert len(out2) == 3

    # Convention 3: return_features=True
    with torch.no_grad():
        outs, feats = model(tup, return_features=True)
    assert 'f_LGE' in feats
    assert 'f_T2' in feats
    assert 'mask_C0' in feats


def test_wce_loss_inverse_weight_mode_and_logits():
    """Verify WCELoss with weight_mode='inverse' and from_logits=True."""
    pred_logits = torch.randn(4, 3, 64, 64)
    target = torch.randint(0, 3, (4, 64, 64), dtype=torch.long)

    loss_fn = WCELoss(weight_mode="inverse", from_logits=True)
    loss = loss_fn(pred_logits, target)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_canonical_mode_out_of_bounds_clamping():
    """Verify create_multi_head_targets in canonical mode with OOB labels."""
    # Canonical schema: 0: BG, 1: Normal, 2: Edema, 3: Scar
    oob_labels = torch.tensor([-5, 0, 1, 2, 3, 10, 255], dtype=torch.long)
    t_c0, t_lge, t_t2 = create_multi_head_targets(oob_labels, mode="canonical")

    # Clamped: -5 -> 0, 10 -> 3, 255 -> 3
    # 0 -> C0:0, LGE:0, T2:0
    # 1 -> C0:1, LGE:1, T2:1
    # 2 -> C0:1, LGE:1, T2:2
    # 3 -> C0:1, LGE:2, T2:2
    assert t_c0[0].item() == 0  # -5 clamped to 0
    assert t_lge[0].item() == 0
    assert t_t2[0].item() == 0

    assert t_c0[4].item() == 1  # 3 (Scar)
    assert t_lge[4].item() == 2  # Scar is 2 in LGE
    assert t_t2[4].item() == 2  # Scar is 2 in T2

    assert t_c0[5].item() == 1  # 10 clamped to 3
    assert t_lge[5].item() == 2
    assert t_t2[5].item() == 2


def test_vram_memory_footprint_batch_16():
    """Profile peak VRAM memory consumption during B=16 AMP training step."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda")
    model = MyoPSNetL().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.333e-4)
    scaler = torch.amp.GradScaler('cuda')
    criterion = MyoPSLoss().to(device)

    B = 16
    c0 = torch.randn(B, 1, 128, 128, device=device)
    lge = torch.randn(B, 1, 128, 128, device=device)
    t2 = torch.randn(B, 1, 128, 128, device=device)
    lbl = torch.randint(0, 4, (B, 128, 128), dtype=torch.long, device=device)
    t_c0, t_lge, t_t2 = create_multi_head_targets(lbl, mode="legacy")

    with torch.amp.autocast('cuda'):
        seg_c0, seg_lge, seg_t2 = model(c0, lge, t2)
        total_loss, _, _, _ = criterion(seg_c0, seg_lge, seg_t2, t_c0, t_lge, t_t2)

    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    peak_bytes = torch.cuda.max_memory_allocated()
    peak_gb = peak_bytes / (1024 ** 3)
    print(f"\n[VRAM Profiling] Peak VRAM allocated for B=16 AMP step: {peak_gb:.2f} GB")
    assert peak_gb < 4.0, f"Peak memory {peak_gb:.2f} GB exceeded 4.0 GB budget"


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
