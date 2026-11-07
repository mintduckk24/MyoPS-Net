"""
Robust Multi-Head Criterion and Composite Loss Functions for MyoPS-Net.
Standardized for MyoPS380 Benchmark Recipe (3 modalities: bSSFP, LGE, T2w).
Protocol: myops380_train304_test76_last_v1
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice Loss across multi-class predictions."""
    def __init__(self, eps: float = 1e-7):
        super(GeneralizedDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # predict: (B, C, H, W)
        # target:  (B, C, H, W) or (B, H, W)
        if target.dim() == 3:
            target = F.one_hot(target.long(), num_classes=predict.shape[1]).permute(0, 3, 1, 2).float()
        else:
            target = target.float()

        target_sum = torch.sum(target, dim=(2, 3))  # (B, C)
        w = 1.0 / (target_sum ** 2 + self.eps)

        intersection = w * torch.sum(predict * target, dim=(2, 3))  # (B, C)
        union = w * (torch.sum(predict, dim=(2, 3)) + torch.sum(target, dim=(2, 3)))  # (B, C)

        loss = 1.0 - 2.0 * (torch.sum(intersection, dim=1) + self.eps) / (torch.sum(union, dim=1) + self.eps)
        return loss.mean()


class BinaryDiceLoss(nn.Module):
    """Numerically stable binary Dice loss."""
    def __init__(self, eps: float = 1e-7):
        super(BinaryDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predict_flat = predict.contiguous().view(predict.shape[0], -1)
        target_flat = target.contiguous().view(target.shape[0], -1)

        num = (predict_flat * target_flat).sum(dim=1)
        den = predict_flat.sum(dim=1) + target_flat.sum(dim=1)

        loss = 1.0 - (2.0 * num + self.eps) / (den + self.eps)
        return loss.mean()


class MultiClassDiceLoss(nn.Module):
    """Vectorized multi-class Dice loss supporting 2-class (C0) and 3-class (LGE/T2) heads.
    
    Operates in a single vectorized pass without per-channel Python loops.
    Accepts target as either class indices (B, H, W) or one-hot (B, C, H, W).
    """
    def __init__(self, eps: float = 1e-7, include_background: bool = True):
        super(MultiClassDiceLoss, self).__init__()
        self.eps = eps
        self.include_background = include_background

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = predict.shape[1]
        if target.dim() == 3:
            target_oh = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        else:
            target_oh = target.float()

        B, C = predict.shape[:2]
        p = predict.contiguous().view(B, C, -1)
        t = target_oh.contiguous().view(B, C, -1)

        intersection = (p * t).sum(dim=-1)  # (B, C)
        cardinality = p.sum(dim=-1) + t.sum(dim=-1)  # (B, C)

        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        loss = 1.0 - dice  # (B, C)

        if not self.include_background and C > 1:
            loss = loss[:, 1:]

        return loss.mean()


# Backward-compatibility alias
DiceLoss = MultiClassDiceLoss


class WCELoss(nn.Module):
    """Hardened, device-agnostic Weighted Cross Entropy Loss.
    
    Eliminates the dynamic dimension mismatch bug (mask.max() + 1), prevents
    division by zero and NaNs for absent classes, avoids GPU-CPU sync bottlenecks,
    and supports both probabilities (Softmax) and raw logits.
    """
    def __init__(
        self,
        num_classes: Optional[int] = None,
        weight_mode: str = "log",
        from_logits: bool = False,
        eps: float = 1e-7,
    ):
        super(WCELoss, self).__init__()
        self.num_classes = num_classes
        self.weight_mode = weight_mode
        self.from_logits = from_logits
        self.eps = eps

    def weight_function(self, target: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Computes class weights directly on target.device without host synchronization."""
        if target.dim() == 4:
            mask = torch.argmax(target, dim=1)
            counts = target.sum(dim=(0, 2, 3)).float()
        else:
            mask = target.long()
            counts = torch.bincount(mask.view(-1), minlength=num_classes).float()[:num_classes]

        voxels_sum = float(mask.numel())

        if self.weight_mode == "log":
            # Native MyoPS-Net formulation with absent-class zeroing and clamping against div-0
            safe_counts = counts.clamp(min=1.0)
            log_weights = torch.log(voxels_sum / safe_counts)
            weights = torch.where(
                counts > 0,
                log_weights,
                torch.zeros_like(counts),
            )
        elif self.weight_mode == "inverse":
            # Smoothed inverse frequency normalized over active classes
            safe_counts = counts.clamp(min=1.0)
            inv_weights = voxels_sum / safe_counts
            weights = torch.where(
                counts > 0,
                inv_weights,
                torch.zeros_like(counts),
            )
            if (counts > 0).any():
                weights = weights / weights[counts > 0].mean()
        else:
            raise ValueError(f"Unknown weight_mode: {self.weight_mode}")

        return weights

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = self.num_classes if self.num_classes is not None else predict.shape[1]

        if target.dim() == 3:
            target_oh = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        else:
            target_oh = target.float()

        weights = self.weight_function(target, num_classes)

        if self.from_logits:
            log_p = F.log_softmax(predict, dim=1)
        else:
            log_p = torch.log(predict.clamp(min=self.eps, max=1.0))

        ce_per_class = torch.mean(-target_oh * log_p, dim=(0, 2, 3))  # (C,)
        loss = torch.sum(weights * ce_per_class)
        return loss


# Backward-compatibility alias
RobustWCELoss = WCELoss


class CELoss(nn.Module):
    """Standard Cross Entropy loss for probability or one-hot targets."""
    def __init__(self, eps: float = 1e-7):
        super(CELoss, self).__init__()
        self.eps = eps

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = F.one_hot(target.long(), num_classes=predict.shape[1]).permute(0, 3, 1, 2).float()
        else:
            target = target.float()

        ce = -target * torch.log(predict.clamp(min=self.eps, max=1.0))
        return ce.mean()


class InvariantLoss(nn.Module):
    """Cosine similarity loss regularizing whole myocardium consistency across sequences."""
    def __init__(self, eps: float = 1e-7):
        super(InvariantLoss, self).__init__()
        self.eps = eps

    def forward(self, seg_1: torch.Tensor, seg_2: torch.Tensor) -> torch.Tensor:
        # seg_1, seg_2: (B, 2, H, W) whole myocardium representations
        sim = F.cosine_similarity(seg_1, seg_2, dim=1, eps=self.eps)
        loss = 1.0 - sim
        return loss.mean()


class InclusiveLossScar(nn.Module):
    """Penalizes scar prediction outside ground truth edema region (Scar must be inside Edema)."""
    def __init__(self, eps: float = 1e-7):
        super(InclusiveLossScar, self).__init__()
        self.eps = eps

    def forward(self, scar: torch.Tensor, gd_edema: torch.Tensor) -> torch.Tensor:
        if scar.dim() == 4 and scar.shape[1] == 1:
            scar = scar.squeeze(1)
        if gd_edema.dim() == 4 and gd_edema.shape[1] == 1:
            gd_edema = gd_edema.squeeze(1)

        loss = -1.0 * (1.0 - gd_edema) * torch.log((1.0 - scar).clamp(min=self.eps, max=1.0))
        denom = (1.0 - gd_edema).sum() + self.eps
        return loss.sum() / denom


class InclusiveLossEdema(nn.Module):
    """Penalizes edema prediction failing to cover ground truth scar region."""
    def __init__(self, eps: float = 1e-7):
        super(InclusiveLossEdema, self).__init__()
        self.eps = eps

    def forward(self, edema: torch.Tensor, gd_scar: torch.Tensor) -> torch.Tensor:
        if edema.dim() == 4 and edema.shape[1] == 1:
            edema = edema.squeeze(1)
        if gd_scar.dim() == 4 and gd_scar.shape[1] == 1:
            gd_scar = gd_scar.squeeze(1)

        loss = -1.0 * gd_scar * torch.log(edema.clamp(min=self.eps, max=1.0))
        denom = gd_scar.sum() + self.eps
        return loss.sum() / denom


class SegmentationLoss(nn.Module):
    """Composite Dice + Weighted Cross-Entropy Loss for a single head."""
    def __init__(self, weight_mode: str = "log", eps: float = 1e-7):
        super(SegmentationLoss, self).__init__()
        self.dice_loss = MultiClassDiceLoss(eps=eps)
        self.wce_loss = WCELoss(weight_mode=weight_mode, eps=eps)

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_loss(predict, target) + self.wce_loss(predict, target)


class MyoPSLoss(nn.Module):
    """Unified Composite Multi-Head Loss Function for MyoPS-Net.
    
    Integrates:
      1. Multi-head Segmentation Loss (Dice + WCE) on C0, LGE, T2 heads
      2. Cross-modal Invariant Consistency Loss
      3. Anatomical Pathology Inclusiveness Loss
      
    Total Loss:
      L = L_seg_C0 + lambda_s * L_seg_LGE + lambda_e * L_seg_T2 + lambda_inv * L_inv + lambda_inc * L_inc
    """
    def __init__(
        self,
        lambda_s: float = 2.0,
        lambda_e: float = 2.0,
        lambda_inv: float = 1.0,
        lambda_inc: float = 1.0,
        weight_mode: str = "log",
        eps: float = 1e-7,
    ):
        super(MyoPSLoss, self).__init__()
        self.lambda_s = lambda_s
        self.lambda_e = lambda_e
        self.lambda_inv = lambda_inv
        self.lambda_inc = lambda_inc
        self.eps = eps

        self.seg_loss = SegmentationLoss(weight_mode=weight_mode, eps=eps)
        self.invariant_loss = InvariantLoss(eps=eps)
        self.inclusive_scar = InclusiveLossScar(eps=eps)
        self.inclusive_edema = InclusiveLossEdema(eps=eps)

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates composite multi-head loss.
        
        Supports two primary call signatures:
          1. 6-arg signature:
             criterion(seg_C0, seg_LGE, seg_T2, target_C0, target_LGE, target_T2)
             Returns: (loss_total, loss_seg, loss_invariant, loss_inclusive)
          2. 2-arg signature:
             criterion(seg_dict_or_tuple, label_dict_or_tuple)
             Returns: (loss_seg, loss_invariant, loss_inclusive, loss_total)
        """
        return_total_first = False

        if len(args) == 6:
            seg_c0, seg_lge, seg_t2, t_c0, t_lge, t_t2 = args
            seg = {"C0": seg_c0, "LGE": seg_lge, "T2": seg_t2}
            label = {"cardiac": t_c0, "scar": t_lge, "edema": t_t2}
            return_total_first = True
        elif len(args) == 2:
            seg, label = args[0], args[1]
            return_total_first = False
        elif len(args) == 1 and isinstance(args[0], dict):
            seg = args[0].get("seg")
            label = args[0].get("label")
            return_total_first = False
        else:
            seg = kwargs.get("seg")
            label = kwargs.get("label")
            return_total_first = False

        # Standardize seg container
        if isinstance(seg, (tuple, list)):
            seg_dict = {
                "C0": seg[0],
                "LGE": seg[1],
                "T2": seg[2],
            }
            if len(seg) > 3:
                seg_dict["mapping"] = seg[3]
            seg = seg_dict
        elif isinstance(seg, dict):
            seg = {
                "C0": seg.get("C0", seg.get("cardiac")),
                "LGE": seg.get("LGE", seg.get("scar")),
                "T2": seg.get("T2", seg.get("edema")),
                "mapping": seg.get("mapping", None),
            }

        # Standardize label container
        if isinstance(label, (tuple, list)):
            label = {
                "cardiac": label[0],
                "scar": label[1],
                "edema": label[2],
            }
        elif isinstance(label, dict):
            label = {
                "cardiac": label.get("cardiac", label.get("C0")),
                "scar": label.get("scar", label.get("LGE")),
                "edema": label.get("edema", label.get("T2")),
            }

        # 1. Segmentation Losses
        loss_seg_c0 = self.seg_loss(seg["C0"], label["cardiac"])
        loss_seg_lge = self.seg_loss(seg["LGE"], label["scar"])
        loss_seg_t2 = self.seg_loss(seg["T2"], label["edema"])

        loss_seg = loss_seg_c0 + self.lambda_s * loss_seg_lge + self.lambda_e * loss_seg_t2
        if seg.get("mapping") is not None:
            loss_seg = loss_seg + self.lambda_s * self.seg_loss(seg["mapping"], label["scar"])

        # 2. Invariant Consistency Loss
        # Myocardium representation for C0:
        # If 2 channels: already [bg, myo]
        # If 3 channels: [bg + lv, myo]
        if seg["C0"].shape[1] == 2:
            myo_c0 = seg["C0"]
        else:
            myo_c0 = torch.cat([seg["C0"][:, 0:1] + seg["C0"][:, 2:3], seg["C0"][:, 1:2]], dim=1)

        # Myocardium representation for LGE: [bg, non-scar + scar]
        myo_lge = torch.cat([seg["LGE"][:, 0:1], seg["LGE"][:, 1:2] + seg["LGE"][:, 2:3]], dim=1)

        # Myocardium representation for T2: [bg, normal + edema-inc]
        myo_t2 = torch.cat([seg["T2"][:, 0:1], seg["T2"][:, 1:2] + seg["T2"][:, 2:3]], dim=1)

        loss_invariant = self.invariant_loss(myo_c0, myo_lge) + self.invariant_loss(myo_c0, myo_t2)
        if seg.get("mapping") is not None:
            myo_mapping = torch.cat([seg["mapping"][:, 0:1], seg["mapping"][:, 1:2] + seg["mapping"][:, 2:3]], dim=1)
            loss_invariant = loss_invariant + self.invariant_loss(myo_c0, myo_mapping)

        loss_invariant = self.lambda_inv * loss_invariant

        # 3. Pathology Inclusiveness Loss
        # Extract ground truth edema-inclusive mask
        if label["edema"].dim() == 4:
            gd_edema = label["edema"][:, 2]
        else:
            gd_edema = (label["edema"] == 2).float()

        # Extract ground truth scar mask
        if label["scar"].dim() == 4:
            gd_scar = label["scar"][:, 2]
        else:
            gd_scar = (label["scar"] == 2).float()

        loss_inclusive = self.inclusive_scar(seg["LGE"][:, 2], gd_edema) + self.inclusive_edema(seg["T2"][:, 2], gd_scar)
        if seg.get("mapping") is not None:
            loss_inclusive = loss_inclusive + self.inclusive_scar(seg["mapping"][:, 2], gd_edema)

        loss_inclusive = self.lambda_inc * loss_inclusive

        # Total Composite Loss
        loss_total = loss_seg + loss_invariant + loss_inclusive

        if return_total_first:
            return loss_total, loss_seg, loss_invariant, loss_inclusive
        return loss_seg, loss_invariant, loss_inclusive, loss_total
