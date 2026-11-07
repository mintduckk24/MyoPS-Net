"""
U-Net building blocks for MyoPS-Net and MyoPS-Net-L.
Reference: Qiu et al., Medical Image Analysis, 2023.
"""

from typing import List, Tuple, Union
import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class UNet(nn.Module):
    """
    Standard 5-level 2D U-Net.
    Used as the Stage 1 Myocardium Prior & Consistency (MPC) network (unet_C0).
    
    Supports:
      - in_ch=3 for bSSFP, LGE, T2w input concatenation.
      - out_ch=2 for Background (0) and Whole Myocardium (1).
      - apply_softmax: whether to apply nn.Softmax(dim=1) on output.
    """
    def __init__(self, in_ch: int, out_ch: int, apply_softmax: bool = True):
        super(UNet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.apply_softmax = apply_softmax

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.active = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv(d2)
        if self.apply_softmax:
            out = self.active(d1)
        else:
            out = d1
        return out


class UNetEncoder(nn.Module):
    """
    5-level U-Net Encoder.
    Returns feature pyramid tuple (e1, e2, e3, e4, e5) at spatial scales
    [1/1, 1/2, 1/4, 1/8, 1/16].
    """
    def __init__(self, in_ch: int):
        super(UNetEncoder, self).__init__()
        self.in_ch = in_ch

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        return e1, e2, e3, e4, e5


class UNetDecoder(nn.Module):
    """
    Standard single-stream 5-level U-Net Decoder.
    """
    def __init__(self, out_ch: int, apply_softmax: bool = True):
        super(UNetDecoder, self).__init__()
        self.out_ch = out_ch
        self.apply_softmax = apply_softmax

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.active = nn.Softmax(dim=1)

    def forward(self, e: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]) -> torch.Tensor:
        d5 = self.Up5(e[4])
        d5 = torch.cat((e[3], d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e[2], d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e[1], d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e[0], d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv(d2)
        if self.apply_softmax:
            out = self.active(d1)
        else:
            out = d1
        return out


class UNetDecoderPlus(nn.Module):
    """
    Dual-skip Cross-Modal Feature Fusion (CMFF) Decoder.
    Ingests complementary reference feature pyramid (e_ref) and primary sequence feature pyramid (e).
    
    Level 5: Concat [e_ref[4], e[4]] -> 2048 ch -> 512 ch upsampled
    Levels 4..1: Concat [e_ref[k], e[k], upsampled] -> 3 * filters[k] ch -> filters[k] ch
    
    Args:
        out_ch: Number of output segmentation classes.
        detach_ref: If True (default), detaches e_ref from gradient flow to prevent
                    cross-modal gradient conflicts (matches Qiu et al. MedIA 2023).
        apply_softmax: If True (default), applies Softmax(dim=1) over output classes.
    """
    def __init__(self, out_ch: int, detach_ref: bool = True, apply_softmax: bool = True):
        super(UNetDecoderPlus, self).__init__()
        self.out_ch = out_ch
        self.detach_ref = detach_ref
        self.apply_softmax = apply_softmax

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Up5 = up_conv(2 * filters[4], filters[3])
        self.Up_conv5 = conv_block(3 * filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(3 * filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(3 * filters[1], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(3 * filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.active = nn.Softmax(dim=1)

    def forward(
        self,
        e_ref: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
        e: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    ) -> torch.Tensor:
        ref4 = e_ref[4].detach() if self.detach_ref else e_ref[4]
        ref3 = e_ref[3].detach() if self.detach_ref else e_ref[3]
        ref2 = e_ref[2].detach() if self.detach_ref else e_ref[2]
        ref1 = e_ref[1].detach() if self.detach_ref else e_ref[1]
        ref0 = e_ref[0].detach() if self.detach_ref else e_ref[0]

        d5 = torch.cat((ref4, e[4]), dim=1)
        d5 = self.Up5(d5)
        d5 = torch.cat((ref3, e[3], d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((ref2, e[2], d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((ref1, e[1], d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((ref0, e[0], d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv(d2)
        if self.apply_softmax:
            out = self.active(d1)
        else:
            out = d1
        return out
