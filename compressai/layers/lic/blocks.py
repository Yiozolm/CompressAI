from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.layers.layers import conv1x1, conv3x3

__all__ = [
    "DepthwiseConv5x5",
    "GatedFFN",
    "GatedTransformCNN",
    "LayerNorm2d",
    "OLP",
    "ResidualBottleneckBlock",
]


class LayerNorm2d(nn.Module):
    """Layer normalization over the channel dimension for image tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, input_tensor: Tensor) -> Tensor:
        mean = input_tensor.mean(dim=1, keepdim=True)
        variance = (input_tensor - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (input_tensor - mean) / torch.sqrt(variance + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return normalized * weight + bias


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


class OLP(nn.Module):
    """Orthogonal linear projection with an auxiliary regularizer."""

    def __init__(self, in_features: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_dim, bias=bias)
        self.in_dim = in_features
        self.out_dim = out_dim
        identity_size = min(in_features, out_dim)
        identity_matrix = torch.eye(identity_size)
        self.register_buffer("identity_matrix", identity_matrix, persistent=False)

    def loss(self) -> Tensor:
        weight = self.linear.weight
        gram = weight @ weight.t() if self.in_dim > self.out_dim else weight.t() @ weight
        target = self.identity_matrix.to(device=gram.device, dtype=gram.dtype)
        return F.mse_loss(gram, target)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.linear(input_tensor)


class ResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block with 1x1, 3x3, then 1x1 convolutions."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        mid_ch = min(in_ch, out_ch) // 2
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_ch, mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(mid_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, input_tensor: Tensor) -> Tensor:
        identity = self.skip(input_tensor)
        output = self.conv1(input_tensor)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        return output + identity
