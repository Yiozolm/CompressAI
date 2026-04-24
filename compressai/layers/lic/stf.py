from __future__ import annotations

from typing import Optional, Sequence, Type

import torch
import torch.nn as nn

from timm.layers import DropPath
from torch import Tensor

from ..layers import conv1x1, conv3x3
from ..attn.swin import PatchMerging, PatchSplit
from ..attn.swin_attention import (
    WindowAttention,
    _pad_to_window_size,
    build_window_attention_mask,
    window_partition,
    window_reverse,
)

__all__ = [
    "PatchEmbed",
    "STFBasicLayer",
    "STFSwinTransformerBlock",
    "STFWinBasedAttention",
    "STFWinNoShiftAttention",
]


class _STFResidualUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            conv1x1(channels, channels // 2),
            nn.GELU(),
            conv3x3(channels // 2, channels // 2),
            nn.GELU(),
            conv1x1(channels // 2, channels),
        )
        self.relu = nn.GELU()

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.relu(self.conv(input_tensor) + input_tensor)


class STFMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.fc1(input_tensor)
        output = self.act(output)
        output = self.drop(output)
        output = self.fc2(output)
        return self.drop(output)


class STFWinBasedAttention(nn.Module):
    def __init__(
        self,
        dim: int = 192,
        num_heads: int = 8,
        window_size: int = 8,
        shift_size: int = 0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0 <= shift_size < window_size:
            raise ValueError("shift_size must be in [0, window_size)")

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = input_tensor.permute(0, 2, 3, 1).contiguous()
        output, pad_height, pad_width = _pad_to_window_size(output, self.window_size)
        padded_height, padded_width = output.shape[1], output.shape[2]

        if self.shift_size > 0:
            attn_mask = build_window_attention_mask(
                padded_height,
                padded_width,
                self.window_size,
                self.shift_size,
                output.device,
            )
            output = torch.roll(
                output,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            attn_mask = None

        windows = window_partition(output, self.window_size).view(
            -1,
            self.window_size * self.window_size,
            output.shape[-1],
        )
        windows = self.attn(windows, mask=attn_mask)
        windows = windows.view(-1, self.window_size, self.window_size, output.shape[-1])
        output = window_reverse(windows, self.window_size, padded_height, padded_width)

        if self.shift_size > 0:
            output = torch.roll(
                output,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        if pad_height > 0 or pad_width > 0:
            output = output[:, : input_tensor.shape[2], : input_tensor.shape[3], :].contiguous()

        output = output.permute(0, 3, 1, 2).contiguous()
        return input_tensor + self.drop_path(output)


class STFWinNoShiftAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        shift_size: int = 0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv_a = nn.Sequential(
            _STFResidualUnit(dim),
            _STFResidualUnit(dim),
            _STFResidualUnit(dim),
        )
        self.conv_b = nn.Sequential(
            STFWinBasedAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                drop_path=drop_path,
            ),
            _STFResidualUnit(dim),
            _STFResidualUnit(dim),
            _STFResidualUnit(dim),
            conv1x1(dim, dim),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output_a = self.conv_a(input_tensor)
        output_b = self.conv_b(input_tensor)
        return input_tensor + output_a * torch.sigmoid(output_b)


class STFSwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        if not 0 <= shift_size < window_size:
            raise ValueError("shift_size must be in [0, window_size)")

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = STFMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.H: Optional[int] = None
        self.W: Optional[int] = None

    def forward(self, input_tensor: Tensor, mask_matrix: Optional[Tensor] = None) -> Tensor:
        batch_size, length, channels = input_tensor.shape
        height = self.H
        width = self.W
        if height is None or width is None:
            raise RuntimeError("STFSwinTransformerBlock requires H/W before forward")
        if length != height * width:
            raise ValueError("input feature has wrong size")

        shortcut = input_tensor
        output = self.norm1(input_tensor).view(batch_size, height, width, channels)
        output, pad_height, pad_width = _pad_to_window_size(output, self.window_size)
        padded_height, padded_width = output.shape[1], output.shape[2]

        if self.shift_size > 0:
            attn_mask = mask_matrix
            if attn_mask is None:
                attn_mask = build_window_attention_mask(
                    padded_height,
                    padded_width,
                    self.window_size,
                    self.shift_size,
                    output.device,
                )
            output = torch.roll(
                output,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            attn_mask = None

        windows = window_partition(output, self.window_size).view(
            -1,
            self.window_size * self.window_size,
            channels,
        )
        windows = self.attn(windows, mask=attn_mask)
        windows = windows.view(-1, self.window_size, self.window_size, channels)
        output = window_reverse(windows, self.window_size, padded_height, padded_width)

        if self.shift_size > 0:
            output = torch.roll(
                output,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        if pad_height > 0 or pad_width > 0:
            output = output[:, :height, :width, :].contiguous()

        output = output.view(batch_size, height * width, channels)
        output = shortcut + self.drop_path(output)
        return output + self.drop_path(self.mlp(self.norm2(output)))


class STFBasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | Sequence[float] = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        downsample: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        drop_path_values = (
            list(drop_path)
            if isinstance(drop_path, Sequence) and not isinstance(drop_path, (str, bytes))
            else [float(drop_path)] * depth
        )
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.blocks = nn.ModuleList(
            [
                STFSwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_values[index],
                    norm_layer=norm_layer,
                )
                for index in range(depth)
            ]
        )
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, input_tensor: Tensor, height: int, width: int) -> tuple[Tensor, int, int]:
        attn_mask = None
        if any(block.shift_size > 0 for block in self.blocks):
            padded_height = ((height + self.window_size - 1) // self.window_size) * self.window_size
            padded_width = ((width + self.window_size - 1) // self.window_size) * self.window_size
            attn_mask = build_window_attention_mask(
                padded_height,
                padded_width,
                self.window_size,
                self.shift_size,
                input_tensor.device,
            )

        output = input_tensor
        for block in self.blocks:
            block.H = height
            block.W = width
            output = block(output, attn_mask)

        if self.downsample is None:
            return output, height, width

        output = self.downsample(output, height, width)
        if isinstance(self.downsample, PatchMerging):
            return output, (height + 1) // 2, (width + 1) // 2
        return output, height * 2, width * 2


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, input_tensor: Tensor) -> Tensor:
        _, _, height, width = input_tensor.size()
        if width % self.patch_size[1] != 0:
            input_tensor = nn.functional.pad(
                input_tensor,
                (0, self.patch_size[1] - width % self.patch_size[1]),
            )
        if height % self.patch_size[0] != 0:
            input_tensor = nn.functional.pad(
                input_tensor,
                (0, 0, 0, self.patch_size[0] - height % self.patch_size[0]),
            )

        output = self.proj(input_tensor)
        if self.norm is None:
            return output

        out_height, out_width = output.size(2), output.size(3)
        output = output.flatten(2).transpose(1, 2)
        output = self.norm(output)
        return output.transpose(1, 2).view(-1, self.embed_dim, out_height, out_width)
