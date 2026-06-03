"""Shared LIC (learned image compression) building blocks.

Container for blocks reused across multiple LIC model files in
``compressai/models/`` that are not generic enough to live in
``compressai/layers/`` proper. Hosts the stride / upsample wrappers around
:class:`ResidualBottleneckBlock` originally introduced for DCAE (Lu et al.,
CVPR 2025) and reused by SAAF (Ma et al., CVPR 2026), plus the gated
depthwise transform blocks (:class:`GatedFFN`, :class:`DepthwiseConv5x5`,
:class:`GatedTransformCNN`) used by the AuxT-style transforms in GLIC (Chen
et al.) / CMIC.

:class:`LayerNorm2d` is re-exported from ``timm.layers`` (numerically
identical channel-dim layer norm, same ``weight`` / ``bias`` state-dict keys)
rather than re-implemented; the MLICv2 transforms already depend on it.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from timm.layers import LayerNorm2d
from torch import Tensor

from compressai.models.sensetime import ResidualBottleneckBlock
from compressai.models.utils import conv, deconv

__all__ = [
    "DepthwiseConv5x5",
    "GatedFFN",
    "GatedTransformCNN",
    "LayerNorm2d",
    "ResidualBottleneckBlockWithStride",
    "ResidualBottleneckBlockWithUpsample",
]


class ResidualBottleneckBlockWithStride(nn.Module):
    """Stride-2 5x5 conv followed by three :class:`ResidualBottleneckBlock` units."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = conv(in_ch, out_ch, kernel_size=5, stride=2)
        self.res1 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res2 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res3 = ResidualBottleneckBlock(out_ch, out_ch)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.conv(input_tensor)
        output = self.res1(output)
        output = self.res2(output)
        return self.res3(output)


class ResidualBottleneckBlockWithUpsample(nn.Module):
    """Three :class:`ResidualBottleneckBlock` units followed by a stride-2 5x5 deconv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.res1 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res2 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res3 = ResidualBottleneckBlock(in_ch, in_ch)
        self.conv = deconv(in_ch, out_ch, kernel_size=5, stride=2)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.res1(input_tensor)
        output = self.res2(output)
        output = self.res3(output)
        return self.conv(output)


class GatedFFN(nn.Module):
    """Gated feed-forward block used by recent LIC transforms."""

    def __init__(self, channels: int, expansion_factor: float = 4) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(
            channels,
            hidden_channels * 2,
            kernel_size=1,
            bias=False,
        )
        self.project_out = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        gate_tensor, value_tensor = self.project_in(input_tensor).chunk(2, dim=1)
        hidden = F.gelu(gate_tensor) * value_tensor
        return self.project_out(hidden)


class DepthwiseConv5x5(nn.Module):
    """Pointwise-depthwise-pointwise residual convolution block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        slope: float = 0.01,
        inplace: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=5,
            padding=2,
            groups=in_ch,
        )
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        if in_ch == out_ch:
            self.skip = nn.Identity()

    def forward(self, input_tensor: Tensor) -> Tensor:
        identity = self.skip(input_tensor)
        output = self.conv1(input_tensor)
        output = self.depth_conv(output)
        output = self.conv2(output)
        return output + identity


class GatedTransformCNN(nn.Module):
    """Depthwise convolution plus gated feed-forward transform block."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        expansion_factor: float = 4,
        **layer_kwargs,
    ) -> None:
        super().__init__()
        del layer_kwargs
        self.mixer = DepthwiseConv5x5(dim, dim_out)
        self.norm = LayerNorm2d(dim_out)
        self.mlp = GatedFFN(dim_out, expansion_factor=expansion_factor)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.mixer(input_tensor)
        return output + self.mlp(self.norm(output))
