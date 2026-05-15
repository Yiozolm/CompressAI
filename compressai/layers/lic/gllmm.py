# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

from __future__ import annotations

import torch.nn as nn

from torch import Tensor

from compressai.layers.layers import conv1x1, conv3x3

__all__ = [
    "GLLMMNonLocalAttentionBlock",
    "GLLMMResidualBottleneck",
    "GLLMMResidualChain",
]


class GLLMMResidualBottleneck(nn.Module):
    """Bottleneck residual block used inside GLLMM attention modules."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError("channels must be at least 2")
        mid_channels = channels // 2
        self.conv1 = conv1x1(channels, mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_channels, mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(mid_channels, channels)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.conv1(input_tensor)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        return input_tensor + output


class GLLMMNonLocalAttentionBlock(nn.Module):
    """Trunk-mask attention block from the GLLMM transforms."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            GLLMMResidualBottleneck(channels),
            GLLMMResidualBottleneck(channels),
            GLLMMResidualBottleneck(channels),
        )
        self.attention = nn.Sequential(
            GLLMMResidualBottleneck(channels),
            GLLMMResidualBottleneck(channels),
            GLLMMResidualBottleneck(channels),
            conv1x1(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        trunk = self.trunk(input_tensor)
        attention = self.attention(input_tensor)
        return input_tensor + attention * trunk


class GLLMMResidualChain(nn.Module):
    """Four-convolution residual chain used in GLLMM transform stages."""

    def __init__(self, channels: int, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.layer0 = self._make_layer(channels, negative_slope)
        self.layer1 = self._make_layer(channels, negative_slope)
        self.layer2 = self._make_layer(channels, negative_slope)
        self.layer3 = self._make_layer(channels, negative_slope)

    @staticmethod
    def _make_layer(channels: int, negative_slope: float) -> nn.Sequential:
        return nn.Sequential(
            conv3x3(channels, channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        tensor2 = self.layer0(input_tensor)
        tensor2 = self.layer1(tensor2)
        tensor3 = input_tensor + tensor2
        tensor4 = self.layer2(tensor3)
        tensor5 = self.layer3(tensor4)
        tensor6 = tensor5 + tensor3
        return input_tensor + tensor6
