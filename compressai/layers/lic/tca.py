from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import to_2tuple
from torch import Tensor

__all__ = [
    "ConvPositionalEncoding",
    "MaskedSliceChannelAttention",
    "TCA",
    "TCABlock",
    "TCAEntropyModel",
]


def _pad_to_window_multiple(
    input_tensor: Tensor,
    window_size: int,
) -> Tuple[Tensor, int, int]:
    _, _, height, width = input_tensor.shape
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    if pad_height == 0 and pad_width == 0:
        return input_tensor, 0, 0
    return F.pad(input_tensor, (0, pad_width, 0, pad_height)), pad_height, pad_width


class MaskedSliceChannelAttention(nn.Module):
    def __init__(self, dim: int, slices: int = 12, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.scale = (dim // num_heads) ** -0.5
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, groups=slices)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        height: int,
        width: int,
        mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, num_tokens, channels = query.shape
        query = query.view(
            batch_size,
            num_tokens,
            channels // self.num_heads,
            self.num_heads,
        ).permute(0, 3, 2, 1)
        key = key.view(
            batch_size,
            num_tokens,
            channels // self.num_heads,
            self.num_heads,
        ).permute(0, 3, 2, 1)
        value = value.view(
            batch_size,
            num_tokens,
            channels // self.num_heads,
            self.num_heads,
        ).permute(0, 3, 2, 1)

        attention = (query * self.scale) @ key.transpose(-2, -1)
        if mask is not None:
            attention = attention.masked_fill(mask, float("-inf"))
        attention = attention.softmax(dim=-1)

        output = attention @ value
        output = output.permute(0, 3, 2, 1).reshape(batch_size, num_tokens, channels)
        output = output.permute(0, 2, 1).contiguous().view(batch_size, channels, height, width)
        return self.proj(output)


class SliceGroupedMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        slices: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_features,
            hidden_features,
            kernel_size=1,
            groups=slices,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(
            hidden_features,
            in_features,
            kernel_size=1,
            groups=slices,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(input_tensor)))


class ConvPositionalEncoding(nn.Module):
    def __init__(self, dim: int, slices: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            dim,
            dim,
            to_2tuple(kernel_size),
            to_2tuple(1),
            to_2tuple(kernel_size // 2),
            groups=dim,
        )
        self.norm = nn.GroupNorm(slices, dim)
        self.activation = nn.GELU()

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.proj(input_tensor)
        output = self.norm(output)
        return input_tensor + self.activation(output)


class TCABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        slices: int = 12,
        mlp_ratio: float = 1.0,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.slices = int(slices)
        self.window_size = int(window_size)
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, groups=slices)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, groups=slices)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, groups=slices)
        self.norm1 = nn.GroupNorm(slices, dim)
        self.norm2 = nn.GroupNorm(slices, dim)
        self.positional_encoding = ConvPositionalEncoding(dim, slices)
        self.attention = MaskedSliceChannelAttention(
            dim=dim,
            slices=slices,
            num_heads=num_heads,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SliceGroupedMLP(dim, mlp_hidden_dim, slices)
        self.register_buffer(
            "mask",
            self._generate_mask(dim, num_heads, slices),
            persistent=False,
        )

    @staticmethod
    def _generate_mask(dim: int, num_heads: int, slices: int) -> Tensor:
        head_dim = dim // num_heads
        attention_mask = torch.zeros(1, head_dim, head_dim, dtype=torch.bool)
        for index in range(slices - 1):
            start = (index + 1) * head_dim // slices
            end = (index + 2) * head_dim // slices
            attention_mask[:, :start, start:end] = True
        return attention_mask

    def forward(self, input_tensor: Tensor) -> Tensor:
        residual = input_tensor
        output, pad_h, pad_w = _pad_to_window_multiple(input_tensor, self.window_size)
        batch_size, channels, padded_height, padded_width = output.shape
        height_windows = padded_height // self.window_size
        width_windows = padded_width // self.window_size
        # Window partition first; norm/CPE then run on the (B*nW, C, ws, ws)
        # tensor so GroupNorm sees the same per-window statistics as upstream.
        output = output.view(
            batch_size,
            channels,
            height_windows,
            self.window_size,
            width_windows,
            self.window_size,
        ).permute(0, 2, 4, 1, 3, 5)
        output = output.reshape(-1, channels, self.window_size, self.window_size)

        output = self.positional_encoding(self.norm1(output))

        query = self.q_proj(output).flatten(2).transpose(1, 2)
        key = self.k_proj(output).flatten(2).transpose(1, 2)
        value = self.v_proj(output).flatten(2).transpose(1, 2)
        output = self.attention(
            query,
            key,
            value,
            self.window_size,
            self.window_size,
            self.mask.to(query.device),
        )
        output = output.view(
            batch_size,
            height_windows,
            width_windows,
            channels,
            self.window_size,
            self.window_size,
        ).permute(0, 3, 1, 4, 2, 5)
        output = output.reshape(batch_size, channels, padded_height, padded_width)
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, : residual.shape[2], : residual.shape[3]].contiguous()

        output = residual + output
        return output + self.mlp(self.norm2(output))


class TCA(nn.Module):
    def __init__(
        self,
        dim: int = 192,
        depth: int = 4,
        ratio: int = 4,
        slices: int = 12,
        window_size: int = 8,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        if dim % slices != 0:
            raise ValueError("dim must be divisible by slices")

        self.dim = int(dim)
        self.slices = int(slices)
        self.ratio = int(ratio)
        start_token_channels = self.dim // self.slices
        self.start_token_from_hyperprior = nn.Conv2d(
            self.dim * 2,
            start_token_channels,
            kernel_size=3,
            padding=1,
        )
        self.lift = nn.Conv2d(
            self.dim,
            self.dim * self.ratio,
            kernel_size=3,
            padding=1,
            groups=self.slices,
        )
        self.layers = nn.ModuleList(
            TCABlock(
                dim=self.dim * self.ratio,
                num_heads=num_heads,
                slices=self.slices,
                window_size=window_size,
            )
            for _ in range(depth)
        )

    def forward(self, hyper: Tensor, y: Tensor) -> Tensor:
        start_token = self.start_token_from_hyperprior(hyper)
        output = self.lift(
            torch.cat((start_token, y[:, : -self.dim // self.slices]), dim=1)
        )
        for layer in self.layers:
            output = layer(output)
        return output


class TCAEntropyModel(nn.Module):
    def __init__(
        self,
        dim: int = 192,
        depth: int = 4,
        ratio: int = 4,
        slices: int = 12,
        window_size: int = 8,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        if dim % slices != 0:
            raise ValueError("dim must be divisible by slices")

        self.dim = int(dim)
        self.ratio = int(ratio)
        self.slices = int(slices)
        self.tca = TCA(
            dim=dim,
            depth=depth,
            ratio=ratio,
            slices=slices,
            window_size=window_size,
            num_heads=num_heads,
        )
        self.hyper_trans = nn.Conv2d(dim * 2, dim * 2, kernel_size=1)
        self.entropy_parameters_net = nn.Sequential(
            nn.Conv2d(
                dim * (ratio + 2),
                dim * ratio // 2,
                kernel_size=3,
                padding=1,
                groups=slices,
            ),
            nn.GELU(),
            nn.Conv2d(
                dim * ratio // 2,
                dim * 3,
                kernel_size=3,
                padding=1,
                groups=slices,
            ),
            nn.GELU(),
            nn.Conv2d(
                dim * 3,
                dim * 3,
                kernel_size=3,
                padding=1,
                groups=slices,
            ),
        )

    def forward(self, hyper: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, channels, height, width = y.shape
        hyper_features = self.hyper_trans(hyper).view(batch_size, channels, 2, height, width)
        tca_features = self.tca(hyper, y).view(
            batch_size,
            channels,
            self.ratio,
            height,
            width,
        )
        output = torch.cat((tca_features, hyper_features), dim=2).view(
            batch_size,
            channels * (self.ratio + 2),
            height,
            width,
        )
        output = self.entropy_parameters_net(output).view(
            batch_size,
            channels,
            3,
            height,
            width,
        )
        return output[:, :, 0], output[:, :, 1], output[:, :, 2]
