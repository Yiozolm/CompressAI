# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _conv_norm_act(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.GroupNorm(_num_groups(out_channels), out_channels),
        nn.ReLU(inplace=True),
    )


class ResidualDenseBlock(nn.Module):
    """RDB used by LBHIC's boundary-aware post-processing module."""

    def __init__(
        self,
        channels: int,
        growth_channels: int,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels + i * growth_channels,
                        growth_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                )
                for i in range(num_layers)
            ]
        )
        self.local_fusion = nn.Conv2d(
            channels + num_layers * growth_channels,
            channels,
            kernel_size=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return x + self.local_fusion(torch.cat(features, dim=1))


class GroupedResidualDenseBlock(nn.Module):
    """A compact GRDB approximation from the LBHIC BPM description."""

    def __init__(
        self,
        channels: int,
        growth_channels: int,
        num_blocks: int = 2,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                ResidualDenseBlock(channels, growth_channels, num_layers)
                for _ in range(num_blocks)
            ]
        )
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fusion(self.blocks(x))


class _PredictionUNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.enc1 = nn.Sequential(
            _conv_norm_act(in_channels, hidden_channels),
            _conv_norm_act(hidden_channels, hidden_channels),
        )
        self.down = _conv_norm_act(hidden_channels, hidden_channels * 2, stride=2)
        self.mid = _conv_norm_act(hidden_channels * 2, hidden_channels * 2)
        self.up = nn.ConvTranspose2d(
            hidden_channels * 2,
            hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.dec = _conv_norm_act(hidden_channels * 2, hidden_channels)
        self.out = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        skip = self.enc1(x)
        x = self.mid(self.down(skip))
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear")
        x = self.dec(torch.cat((x, skip), dim=1))
        return self.out(x)


class ContextualPredictionModule(nn.Module):
    """LBHIC contextual prediction module with vertical/horizontal strip pooling."""

    def __init__(
        self,
        input_channels: int = 3,
        feature_channels: int = 32,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.feature_channels = int(feature_channels)
        self.feature_extractor = nn.Sequential(
            _conv_norm_act(input_channels, feature_channels),
            _conv_norm_act(feature_channels, feature_channels),
        )
        self.prediction_network = _PredictionUNet(
            in_channels=feature_channels + 2 * input_channels,
            hidden_channels=feature_channels,
            out_channels=input_channels,
        )

    def forward(self, upper_block: Tensor, left_block: Tensor) -> Tensor:
        upper_feature = self.feature_extractor(upper_block)
        left_feature = self.feature_extractor(left_block)
        height, width = upper_feature.shape[-2:]

        upper_strip = upper_feature.mean(dim=2, keepdim=True).expand(
            -1, -1, height, width
        )
        left_strip = left_feature.mean(dim=3, keepdim=True).expand(
            -1, -1, height, width
        )
        fused_feature = upper_strip + left_strip
        return self.prediction_network(
            torch.cat((upper_block, left_block, fused_feature), dim=1)
        )


class BoundedNonLocalBlock(nn.Module):
    """Non-local block with pooled key/value tensors to keep memory bounded."""

    def __init__(self, channels: int, max_positions: int = 256) -> None:
        super().__init__()
        hidden_channels = max(channels // 2, 1)
        self.max_positions = int(max_positions)
        self.theta = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.phi = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.g = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.out = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _pool_context(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        if height * width <= self.max_positions:
            return x
        side = max(int(self.max_positions**0.5), 1)
        return F.adaptive_avg_pool2d(x, output_size=(side, side))

    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        query = self.theta(x).flatten(2).transpose(1, 2)
        context = self._pool_context(x)
        key = self.phi(context).flatten(2)
        value = self.g(context).flatten(2).transpose(1, 2)

        attention = torch.bmm(query, key).softmax(dim=-1)
        out = torch.bmm(attention, value).transpose(1, 2)
        out = out.reshape(batch, -1, height, width)
        return x + self.out(out)


class BoundaryAwarePostProcessing(nn.Module):
    """LBHIC BPM guided by a binary block-boundary mask."""

    def __init__(
        self,
        input_channels: int = 3,
        channels: int = 48,
        growth_channels: int = 16,
        num_grdb_blocks: int = 2,
        num_dense_layers: int = 4,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.residual_scale = float(residual_scale)

        self.stem = _conv_norm_act(input_channels + 1, channels)
        self.mask_attention = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.scale1 = GroupedResidualDenseBlock(
            channels, growth_channels, num_grdb_blocks, num_dense_layers
        )
        self.down2 = _conv_norm_act(channels, channels, stride=2)
        self.scale2 = GroupedResidualDenseBlock(
            channels, growth_channels, num_grdb_blocks, num_dense_layers
        )
        self.down4 = _conv_norm_act(channels, channels, stride=2)
        self.scale4 = GroupedResidualDenseBlock(
            channels, growth_channels, num_grdb_blocks, num_dense_layers
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            BoundedNonLocalBlock(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, input_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            mask = x.new_zeros(x.size(0), 1, x.size(2), x.size(3))
        features = self.stem(torch.cat((x, mask), dim=1))
        features = features * (1.0 + self.mask_attention(mask))

        scale1 = self.scale1(features)
        scale2 = self.scale2(self.down2(features))
        scale4 = self.scale4(self.down4(scale2))
        scale2 = F.interpolate(scale2, size=scale1.shape[-2:], mode="bilinear")
        scale4 = F.interpolate(scale4, size=scale1.shape[-2:], mode="bilinear")

        residual = self.fusion(torch.cat((scale1, scale2, scale4), dim=1))
        return x + self.residual_scale * residual


def make_boundary_mask(
    reference: Tensor,
    block_size: int,
    boundary_width: int = 4,
) -> Tensor:
    """Create a binary mask around non-overlapping block boundaries."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if boundary_width < 0:
        raise ValueError("boundary_width must be non-negative")

    batch, _, height, width = reference.shape
    mask = reference.new_zeros(batch, 1, height, width)
    if boundary_width == 0:
        return mask

    for y in range(block_size, height, block_size):
        start = max(y - boundary_width, 0)
        end = min(y + boundary_width, height)
        mask[:, :, start:end, :] = 1
    for x in range(block_size, width, block_size):
        start = max(x - boundary_width, 0)
        end = min(x + boundary_width, width)
        mask[:, :, :, start:end] = 1
    return mask


__all__ = [
    "BoundaryAwarePostProcessing",
    "BoundedNonLocalBlock",
    "ContextualPredictionModule",
    "GroupedResidualDenseBlock",
    "ResidualDenseBlock",
    "make_boundary_mask",
]
