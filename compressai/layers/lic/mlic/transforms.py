from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn

from torch import Tensor

from compressai.layers import GDN, conv1x1, conv3x3, subpel_conv3x3

__all__ = [
    "AnalysisTransform",
    "EntropyParameters",
    "HyperAnalysis",
    "HyperSynthesis",
    "LatentResidualPrediction",
    "SynthesisTransform",
]


class GeluResidualBlockWithStride(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.act = nn.GELU()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        self.skip = conv1x1(in_ch, out_ch, stride=stride) if stride != 1 or in_ch != out_ch else None

    def forward(self, input_tensor: Tensor) -> Tensor:
        identity = input_tensor if self.skip is None else self.skip(input_tensor)
        output = self.conv1(input_tensor)
        output = self.act(output)
        output = self.conv2(output)
        output = self.gdn(output)
        return output + identity


class GeluResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2) -> None:
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.act = nn.GELU()
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.subpel_conv(input_tensor)
        output = self.act(output)
        output = self.conv(output)
        output = self.igdn(output)
        return output + self.upsample(input_tensor)


class GeluResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.act = nn.GELU()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else None

    def forward(self, input_tensor: Tensor) -> Tensor:
        identity = input_tensor if self.skip is None else self.skip(input_tensor)
        output = self.conv1(input_tensor)
        output = self.act(output)
        output = self.conv2(output)
        output = self.act(output)
        return output + identity


class AnalysisTransform(nn.Module):
    def __init__(self, N: int, M: int) -> None:
        super().__init__()
        self.analysis_transform = nn.Sequential(
            GeluResidualBlockWithStride(3, N, stride=2),
            GeluResidualBlock(N, N),
            GeluResidualBlockWithStride(N, N, stride=2),
            GeluResidualBlock(N, N),
            GeluResidualBlockWithStride(N, N, stride=2),
            GeluResidualBlock(N, N),
            conv3x3(N, M, stride=2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.analysis_transform(input_tensor)


class HyperAnalysis(nn.Module):
    def __init__(self, M: int = 192, N: int = 192) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(
            conv3x3(M, N),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.reduction(input_tensor)


class HyperSynthesis(nn.Module):
    def __init__(self, M: int = 192, N: int = 192) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.increase = nn.Sequential(
            conv3x3(N, M),
            nn.GELU(),
            subpel_conv3x3(M, M, 2),
            nn.GELU(),
            conv3x3(M, M * 3 // 2),
            nn.GELU(),
            subpel_conv3x3(M * 3 // 2, M * 3 // 2, 2),
            nn.GELU(),
            conv3x3(M * 3 // 2, M * 2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.increase(input_tensor)


class SynthesisTransform(nn.Module):
    def __init__(self, N: int, M: int) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            GeluResidualBlock(M, M),
            GeluResidualBlockUpsample(M, N, 2),
            GeluResidualBlock(N, N),
            GeluResidualBlockUpsample(N, N, 2),
            GeluResidualBlock(N, N),
            GeluResidualBlockUpsample(N, N, 2),
            GeluResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.synthesis_transform(input_tensor)


class EntropyParameters(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, 320, kernel_size=1, stride=1, padding=0),
            act(),
            nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0),
            act(),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            act(),
            nn.Conv2d(128, out_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, params: Tensor) -> Tensor:
        return self.fusion(params)


class LatentResidualPrediction(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lrp_transform = nn.Sequential(
            conv3x3(in_dim, 224),
            act(),
            conv3x3(224, 128),
            act(),
            conv3x3(128, out_dim),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return 0.5 * torch.tanh(self.lrp_transform(input_tensor))
