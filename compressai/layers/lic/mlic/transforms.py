from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn

from torch import Tensor

from compressai.layers import (
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

__all__ = [
    "AnalysisTransform",
    "EntropyParameters",
    "HyperAnalysis",
    "HyperSynthesis",
    "LatentResidualPrediction",
    "SynthesisTransform",
]


class AnalysisTransform(nn.Module):
    def __init__(self, N: int, M: int) -> None:
        super().__init__()
        self.analysis_transform = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2, act=nn.GELU()),
            ResidualBlock(N, N, act=nn.GELU()),
            ResidualBlockWithStride(N, N, stride=2, act=nn.GELU()),
            ResidualBlock(N, N, act=nn.GELU()),
            ResidualBlockWithStride(N, N, stride=2, act=nn.GELU()),
            ResidualBlock(N, N, act=nn.GELU()),
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
            ResidualBlock(M, M, act=nn.GELU()),
            ResidualBlockUpsample(M, N, 2, act=nn.GELU()),
            ResidualBlock(N, N, act=nn.GELU()),
            ResidualBlockUpsample(N, N, 2, act=nn.GELU()),
            ResidualBlock(N, N, act=nn.GELU()),
            ResidualBlockUpsample(N, N, 2, act=nn.GELU()),
            ResidualBlock(N, N, act=nn.GELU()),
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
