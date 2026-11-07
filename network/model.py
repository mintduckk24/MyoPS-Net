"""
MyoPS-Net and MyoPS-Net-L Model Architectures.
Reference: Qiu et al., "MyoPS-Net: Myocardial Pathology Segmentation with Flexible Combination
of Multi-Sequence CMR Images", Medical Image Analysis, 2023.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from network.unet import UNet, UNetEncoder, UNetDecoderPlus


class MyoPSNetL(nn.Module):
    """
    MyoPS-Net-L: Dedicated 3-Sequence MyoPS-Net Architecture.
    Ingests (bSSFP, LGE, T2w) without mapping sequences (T1m, T2*m).
    
    Stages:
      1. Myocardium Prior & Consistency (MPC) Module (unet_C0):
         - Ingests: Concat[bSSFP, LGE, T2w] (3 channels)
         - Outputs: seg_C0 (2 channels: 0: Background, 1: Whole Myocardium)
         - Anatomical prior: mask_C0 = argmax(seg_C0, dim=1, keepdim=True).float()
      2. Dual Pathology Encoders:
         - encoder_LGE: Ingests Concat[LGE, mask_C0.detach()] (2 channels)
         - encoder_T2:  Ingests Concat[T2w, mask_C0.detach()] (2 channels)
      3. Cross-Modal Feature Fusion (CMFF) Pathology Decoders:
         - decoder_LGE: Reference feature f_T2, primary feature f_LGE -> seg_LGE (3 classes)
         - decoder_T2:  Reference feature f_LGE, primary feature f_T2 -> seg_T2 (3 classes)
         
    Args:
        in_chs: Tuple of input channel counts for (unet_C0, encoder_LGE, encoder_T2). Default: (3, 2, 2).
        out_chs: Tuple of output channel counts for (seg_C0, seg_LGE, seg_T2). Default: (2, 3, 3).
        apply_softmax: Whether heads output Softmax probabilities (default True, matching loss contract).
        detach_ref: Whether reference features are detached in CMFF decoders (default True).
    """
    def __init__(
        self,
        in_chs: Tuple[int, int, int] = (3, 2, 2),
        out_chs: Tuple[int, int, int] = (2, 3, 3),
        apply_softmax: bool = True,
        detach_ref: bool = True,
    ):
        super(MyoPSNetL, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.apply_softmax = apply_softmax
        self.detach_ref = detach_ref

        # Stage 1: MPC module (bSSFP + LGE + T2w -> 2-class whole myocardium)
        self.unet_C0 = UNet(in_ch=in_chs[0], out_ch=out_chs[0], apply_softmax=apply_softmax)

        # Stage 2: Dual pathology encoders
        self.encoder_LGE = UNetEncoder(in_ch=in_chs[1])
        self.encoder_T2 = UNetEncoder(in_ch=in_chs[2])

        # Stage 2: Cross-modal feature fusion decoders
        self.decoder_LGE = UNetDecoderPlus(out_ch=out_chs[1], detach_ref=detach_ref, apply_softmax=apply_softmax)
        self.decoder_T2 = UNetDecoderPlus(out_ch=out_chs[2], detach_ref=detach_ref, apply_softmax=apply_softmax)

    def forward(
        self,
        c0: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
        lge: Optional[torch.Tensor] = None,
        t2: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]]
    ]:
        """
        Forward pass for MyoPSNetL.
        
        Args:
            c0: Tensor of shape (B, 1, H, W) representing bSSFP, OR
                a single tensor of shape (B, 3, H, W) with all 3 modalities stacked, OR
                a 3-element list/tuple [c0, lge, t2].
            lge: Tensor of shape (B, 1, H, W) representing LGE CMR. Optional if c0 is stacked or a sequence.
            t2:  Tensor of shape (B, 1, H, W) representing T2w CMR. Optional if c0 is stacked or a sequence.
            return_features: If True, returns ((seg_C0, seg_LGE, seg_T2), feature_dict).
        
        Returns:
            seg_C0: (B, 2, H, W) probabilities over {0: Background, 1: Whole Myocardium}
            seg_LGE: (B, 3, H, W) probabilities over {0: Background, 1: Non-scar Myocardium, 2: Scar}
            seg_T2: (B, 3, H, W) probabilities over {0: Background, 1: Normal Myocardium, 2: Edema-inclusive}
        """
        if lge is None and t2 is None:
            if isinstance(c0, (list, tuple)) and len(c0) == 3:
                c0, lge, t2 = c0[0], c0[1], c0[2]
            elif isinstance(c0, torch.Tensor) and c0.shape[1] == 3:
                img_c0_single = c0[:, 0:1, :, :]
                lge = c0[:, 1:2, :, :]
                t2 = c0[:, 2:3, :, :]
                c0 = img_c0_single
            else:
                raise ValueError(
                    f"MyoPSNetL forward expects either 3 individual tensors (c0, lge, t2), "
                    f"a 3-element list/tuple of tensors, or a single 3-channel stacked tensor. "
                    f"Got c0 with shape {getattr(c0, 'shape', type(c0))}."
                )

        # Stage 1: MPC module
        img_C0 = torch.cat([c0, lge, t2], dim=1)  # (B, 3, H, W)
        seg_C0 = self.unet_C0(img_C0)            # (B, 2, H, W)
        mask_C0 = torch.argmax(seg_C0, dim=1, keepdim=True).float()  # (B, 1, H, W)

        # Stage 2: Dual pathology encoders with detached anatomical guidance
        img_LGE = torch.cat([lge, mask_C0.detach()], dim=1)  # (B, 2, H, W)
        img_T2 = torch.cat([t2, mask_C0.detach()], dim=1)    # (B, 2, H, W)

        f_LGE = self.encoder_LGE(img_LGE)  # 5-tuple
        f_T2 = self.encoder_T2(img_T2)    # 5-tuple

        # Stage 2: Cross-modal feature skip decoders
        # For LGE: reference sequence is T2, primary is LGE
        seg_LGE = self.decoder_LGE(f_T2, f_LGE)  # (B, 3, H, W)
        # For T2: reference sequence is LGE, primary is T2
        seg_T2 = self.decoder_T2(f_LGE, f_T2)    # (B, 3, H, W)

        if return_features:
            features = {
                'f_LGE': f_LGE,
                'f_T2': f_T2,
                'mask_C0': mask_C0,
            }
            return (seg_C0, seg_LGE, seg_T2), features

        return seg_C0, seg_LGE, seg_T2


class MyoPSNet(nn.Module):
    """
    Unified MyoPS-Net supporting both:
      - 3-modality configuration (bSSFP, LGE, T2w) -> MyoPS-Net-L
      - 5-modality configuration (C0, LGE, T2, T1m, T2starm) -> Original MyoPS-Net
    
    If initialized with len(in_chs) == 3, instantiates MyoPSNetL without mapping branch.
    If initialized with len(in_chs) == 4, instantiates the native 5-modality architecture.
    """
    def __init__(
        self,
        in_chs: Tuple[int, ...] = (5, 2, 2, 3),
        out_chs: Tuple[int, ...] = (3, 3, 3, 3),
        apply_softmax: bool = True,
        detach_ref: bool = True,
    ):
        super(MyoPSNet, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.is_three_modality = (len(in_chs) == 3)
        self.apply_softmax = apply_softmax
        self.detach_ref = detach_ref

        if self.is_three_modality:
            # 3-modality MyoPS-Net-L configuration
            self.unet_C0 = UNet(in_ch=in_chs[0], out_ch=out_chs[0], apply_softmax=apply_softmax)
            self.encoder_LGE = UNetEncoder(in_ch=in_chs[1])
            self.decoder_LGE = UNetDecoderPlus(out_ch=out_chs[1], detach_ref=detach_ref, apply_softmax=apply_softmax)
            self.encoder_T2 = UNetEncoder(in_ch=in_chs[2])
            self.decoder_T2 = UNetDecoderPlus(out_ch=out_chs[2], detach_ref=detach_ref, apply_softmax=apply_softmax)
        else:
            # Legacy 5-modality configuration
            self.unet_C0 = UNet(in_ch=in_chs[0], out_ch=out_chs[0], apply_softmax=apply_softmax)
            self.encoder_LGE = UNetEncoder(in_ch=in_chs[1])
            self.decoder_LGE = UNetDecoderPlus(out_ch=out_chs[1], detach_ref=detach_ref, apply_softmax=apply_softmax)
            self.encoder_T2 = UNetEncoder(in_ch=in_chs[2])
            self.decoder_T2 = UNetDecoderPlus(out_ch=out_chs[2], detach_ref=detach_ref, apply_softmax=apply_softmax)
            self.encoder_mapping = UNetEncoder(in_ch=in_chs[3])
            self.decoder_mapping = UNetDecoderPlus(out_ch=out_chs[3], detach_ref=detach_ref, apply_softmax=apply_softmax)

    def forward(self, *args, **kwargs):
        if self.is_three_modality:
            # 3-modality forward pass
            c0 = args[0] if len(args) > 0 else kwargs.get('c0')
            lge = args[1] if len(args) > 1 else kwargs.get('lge', None)
            t2 = args[2] if len(args) > 2 else kwargs.get('t2', None)
            return_features = kwargs.get('return_features', False)

            if lge is None and t2 is None:
                if isinstance(c0, (list, tuple)) and len(c0) == 3:
                    c0, lge, t2 = c0[0], c0[1], c0[2]
                elif isinstance(c0, torch.Tensor) and c0.shape[1] == 3:
                    img_c0_single = c0[:, 0:1, :, :]
                    lge = c0[:, 1:2, :, :]
                    t2 = c0[:, 2:3, :, :]
                    c0 = img_c0_single

            img_C0 = torch.cat([c0, lge, t2], dim=1)
            seg_C0 = self.unet_C0(img_C0)
            mask_C0 = torch.argmax(seg_C0, dim=1, keepdim=True).float()

            img_LGE = torch.cat([lge, mask_C0.detach()], dim=1)
            img_T2 = torch.cat([t2, mask_C0.detach()], dim=1)

            f_LGE = self.encoder_LGE(img_LGE)
            f_T2 = self.encoder_T2(img_T2)

            seg_LGE = self.decoder_LGE(f_T2, f_LGE)
            seg_T2 = self.decoder_T2(f_LGE, f_T2)

            if return_features:
                features = {'f_LGE': f_LGE, 'f_T2': f_T2, 'mask_C0': mask_C0}
                return (seg_C0, seg_LGE, seg_T2), features

            return seg_C0, seg_LGE, seg_T2
        else:
            # Legacy 5-modality forward pass
            C0, LGE, T2, T1m, T2starm = args
            img = torch.cat([C0, LGE, T2, T1m, T2starm], dim=1)
            seg_C0 = self.unet_C0(img)
            mask_C0 = torch.argmax(seg_C0, dim=1, keepdim=True).float()

            img_LGE = torch.cat([LGE, mask_C0.detach()], dim=1)
            img_T2 = torch.cat([T2, mask_C0.detach()], dim=1)
            img_mapping = torch.cat([T1m, T2starm, mask_C0.detach()], dim=1)

            f_LGE = self.encoder_LGE(img_LGE)
            f_T2 = self.encoder_T2(img_T2)
            f_mapping = self.encoder_mapping(img_mapping)

            seg_LGE_input = [torch.max(f_mapping[i], f_T2[i]) for i in range(5)]
            seg_T2_input = [torch.max(f_LGE[i], f_mapping[i]) for i in range(5)]
            seg_mapping_input = [torch.max(f_LGE[i], f_T2[i]) for i in range(5)]

            seg_LGE = self.decoder_LGE(seg_LGE_input, f_LGE)
            seg_T2 = self.decoder_T2(seg_T2_input, f_T2)
            seg_mapping = self.decoder_mapping(seg_mapping_input, f_mapping)

            return seg_C0, seg_LGE, seg_T2, seg_mapping