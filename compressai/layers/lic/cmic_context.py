from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from torch import Tensor

from ..layers import CheckerboardMaskedConv2d
from .blocks import GatedFFN, LayerNorm2d

__all__ = [
    "CMICChannelContextBlock",
    "CMICSpatialContextBlock",
    "_apply_permutation",
    "_batched_bincount",
    "_inverse_permutation",
]


def _apply_permutation(input_tensor: Tensor, index: Tensor) -> Tensor:
    expanded_index = index.unsqueeze(-1).expand(-1, -1, input_tensor.size(-1))
    return torch.gather(input_tensor, dim=1, index=expanded_index)


def _inverse_permutation(index: Tensor) -> Tensor:
    inverse = torch.empty_like(index)
    positions = torch.arange(index.size(-1), device=index.device).expand_as(index)
    inverse.scatter_(1, index, positions)
    return inverse


def _batched_bincount(index: Tensor, num_classes: int, dtype: torch.dtype) -> Tensor:
    counts = torch.zeros(
        index.size(0),
        num_classes,
        device=index.device,
        dtype=dtype,
    )
    ones = torch.ones_like(index, dtype=dtype)
    counts.scatter_add_(1, index, ones)
    return counts


class CMICChannelContextBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        expansion_factor: float = 4.0,
        **layer_kwargs: Any,
    ) -> None:
        super().__init__()
        del layer_kwargs
        self.mixer = nn.Conv2d(dim, dim_out, kernel_size=1, stride=1)
        self.norm = LayerNorm2d(dim_out)
        self.mlp = GatedFFN(dim_out, expansion_factor=expansion_factor)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.mixer(input_tensor)
        return output + self.mlp(self.norm(output))


class _MaskedDepthwiseMixer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, slope: float = 0.01) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.LeakyReLU(negative_slope=slope, inplace=False),
        )
        self.masked_conv = CheckerboardMaskedConv2d(
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
        output = self.masked_conv(output)
        output = self.conv2(output)
        return output + identity


class _MaskedContextBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        expansion_factor: float = 4.0,
        **layer_kwargs: Any,
    ) -> None:
        super().__init__()
        del layer_kwargs
        self.mixer = _MaskedDepthwiseMixer(dim, dim_out)
        self.norm = LayerNorm2d(dim_out)
        self.mlp = GatedFFN(dim_out, expansion_factor=expansion_factor)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.mixer(input_tensor)
        return output + self.mlp(self.norm(output))


class CMICSpatialContextBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        expansion_factor: float = 4.0,
        **layer_kwargs: Any,
    ) -> None:
        super().__init__()
        self.layer1 = _MaskedContextBlock(
            dim,
            dim,
            expansion_factor=expansion_factor,
            **layer_kwargs,
        )
        self.layer2 = _MaskedContextBlock(
            dim,
            dim_out,
            expansion_factor=expansion_factor,
            **layer_kwargs,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.layer2(self.layer1(input_tensor))
