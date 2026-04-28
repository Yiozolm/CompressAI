from __future__ import annotations

import torch
import torch.nn as nn

from torch import Tensor

__all__ = [
    "DWConvResBlock",
    "PartialConv3x3",
    "PConvResBlock",
]


class PartialConv3x3(nn.Module):
    """3x3 conv applied only to the first ``partial_channels`` channels.

    The remaining channels are passed through unchanged. Used as the spatial
    mixer in HPCM's ``PConvResBlock``.
    """

    def __init__(self, channels: int, partial_channels: int) -> None:
        super().__init__()
        if partial_channels <= 0 or partial_channels > channels:
            raise ValueError(
                "partial_channels must satisfy 0 < partial_channels <= channels"
            )
        self.channels = int(channels)
        self.partial_channels = int(partial_channels)
        self.pconv = nn.Conv2d(
            self.partial_channels,
            self.partial_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        head, tail = torch.split(
            input_tensor,
            [self.partial_channels, self.channels - self.partial_channels],
            dim=1,
        )
        head = self.pconv(head)
        return torch.cat((head, tail), dim=1)


class DWConvResBlock(nn.Module):
    """Depthwise-conv residual block: ``DW3x3 -> 1x1 -> act -> 1x1`` with skip."""

    def __init__(
        self,
        channels: int,
        mlp_ratio: int = 2,
        act: type[nn.Module] = nn.LeakyReLU,
    ) -> None:
        super().__init__()
        hidden_channels = channels * mlp_ratio
        self.branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
            ),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            act(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor + self.branch(input_tensor)


class PConvResBlock(nn.Module):
    """Partial-conv residual block as used in HPCM analysis / synthesis stacks."""

    def __init__(
        self,
        channels: int,
        partial_ratio: int = 4,
        mlp_ratio: int = 2,
        act: type[nn.Module] = nn.LeakyReLU,
    ) -> None:
        super().__init__()
        if partial_ratio <= 0:
            raise ValueError("partial_ratio must be positive")
        partial_channels = channels // partial_ratio
        hidden_channels = channels * mlp_ratio
        self.branch = nn.Sequential(
            PartialConv3x3(channels, partial_channels),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            act(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor + self.branch(input_tensor)
