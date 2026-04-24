from __future__ import annotations

import torch
import torch.nn as nn

from torch import Tensor

from compressai.layers.layers import conv1x1

from .blocks import LayerNorm2d

__all__ = [
    "NAFBlock",
    "NAFTransform",
    "SimpleGate",
]


class SimpleGate(nn.Module):
    def forward(self, input_tensor: Tensor) -> Tensor:
        gate_tensor, value_tensor = input_tensor.chunk(2, dim=1)
        return gate_tensor * value_tensor


class NAFBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        expanded_channels = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.pointwise_depthwise = nn.Sequential(
            conv1x1(channels, expanded_channels),
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=3,
                padding=1,
                groups=expanded_channels,
            ),
        )
        self.gate = SimpleGate()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(channels, channels),
        )
        self.project = conv1x1(channels, channels)
        self.norm2 = LayerNorm2d(channels)
        self.feed_forward = nn.Sequential(
            conv1x1(channels, expanded_channels),
            SimpleGate(),
            conv1x1(channels, channels),
        )
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.norm1(input_tensor)
        output = self.pointwise_depthwise(output)
        output = self.gate(output)
        output = output * self.channel_attention(output)
        output = self.project(output)
        output = input_tensor + self.beta * output
        return output + self.gamma * self.feed_forward(self.norm2(output))


class NAFTransform(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.input_projection = conv1x1(input_channels, hidden_channels)
        self.blocks = nn.Sequential(
            *(NAFBlock(hidden_channels) for _ in range(num_layers))
        )
        self.output_projection = conv1x1(hidden_channels, output_channels)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.input_projection(input_tensor)
        return self.output_projection(output + self.blocks(output))
