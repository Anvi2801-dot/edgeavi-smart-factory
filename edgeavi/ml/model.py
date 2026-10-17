"""MobileNetV3-Large encoder + UNet decoder -> 1-channel defect-logit map."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, cin: int, skip: int, cout: int):
        super().__init__()
        self.conv1 = conv_bn_relu(cin + skip, cout)
        self.conv2 = conv_bn_relu(cout, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv2(self.conv1(torch.cat([x, skip], dim=1)))


class MobileNetV3UNet(nn.Module):
    """Input H and W must be divisible by 32. Output: logits, shape [N,1,H,W]."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        f = mobilenet_v3_large(weights=weights).features
        # (stride, channels): s1 /2 16ch, s2 /4 24ch, s3 /8 40ch, s4 /16 112ch, s5 /32 960ch
        self.s1, self.s2, self.s3 = f[0:2], f[2:4], f[4:7]
        self.s4, self.s5 = f[7:13], f[13:17]
        self.d4 = DecoderBlock(960, 112, 128)
        self.d3 = DecoderBlock(128, 40, 64)
        self.d2 = DecoderBlock(64, 24, 32)
        self.d1 = DecoderBlock(32, 16, 24)
        self.head = nn.Sequential(conv_bn_relu(24, 16), nn.Conv2d(16, 1, 1))

    def forward(self, x):
        e1 = self.s1(x)
        e2 = self.s2(e1)
        e3 = self.s3(e2)
        e4 = self.s4(e3)
        e5 = self.s5(e4)
        d = self.d4(e5, e4)
        d = self.d3(d, e3)
        d = self.d2(d, e2)
        d = self.d1(d, e1)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(d)
