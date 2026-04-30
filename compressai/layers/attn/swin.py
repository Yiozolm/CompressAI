from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, Mlp
from torch import Tensor

from ..layers import AttentionBlock, ResidualBlock, conv1x1, conv3x3
from .swin_attention import (
    WindowAttention,
    _pad_to_window_size,
    build_window_attention_mask,
    window_partition,
    window_reverse,
)

__all__ = [
    "ConvTransBlock",
    "PatchMerging",
    "PatchSplit",
    "SWAtten",
    "SwinBlock",
    "WMSA",
    "WinNoShiftAttention",
    "build_window_attention_mask",
    "window_partition",
    "window_reverse",
]


class WMSA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int],
        head_dim: int,
        window_size: int,
        type: str = "W",
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if type not in {"W", "SW"}:
            raise ValueError(f"Unsupported attention type: {type}")
        if input_dim % head_dim != 0:
            raise ValueError("`input_dim` must be divisible by `head_dim`.")

        self.window_size = window_size
        self.shift_size = 0 if type == "W" else window_size // 2
        self.attn = WindowAttention(
            dim=input_dim,
            window_size=window_size,
            num_heads=input_dim // head_dim,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.output_proj = nn.Linear(input_dim, output_dim or input_dim)

    def forward(self, input_tensor: Tensor) -> Tensor:
        _, height, width, _ = input_tensor.shape
        output, pad_height, pad_width = _pad_to_window_size(
            input_tensor,
            self.window_size,
        )
        padded_height, padded_width = output.shape[1], output.shape[2]

        if self.shift_size > 0:
            mask = build_window_attention_mask(
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
            mask = None

        windows = window_partition(output, self.window_size)
        windows = windows.view(
            -1,
            self.window_size * self.window_size,
            windows.shape[-1],
        )
        windows = self.attn(windows, mask=mask)
        windows = windows.view(
            -1,
            self.window_size,
            self.window_size,
            windows.shape[-1],
        )
        output = window_reverse(windows, self.window_size, padded_height, padded_width)

        if self.shift_size > 0:
            output = torch.roll(
                output,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        if pad_height > 0 or pad_width > 0:
            output = output[:, :height, :width, :].contiguous()
        return self.output_proj(output)


class Block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int],
        head_dim: int,
        window_size: int,
        drop_path: float,
        type: str = "W",
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.norm1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, type=type)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(input_dim)
        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=int(input_dim * mlp_ratio),
            out_features=output_dim,
        )
        self.residual_proj = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = input_tensor + self.drop_path(self.msa(self.norm1(input_tensor)))
        residual = self.residual_proj(output)
        return residual + self.drop_path(self.mlp(self.norm2(output)))


class SwinBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int],
        head_dim: int,
        window_size: int,
        drop_path: float,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.block_1 = Block(
            input_dim,
            input_dim,
            head_dim,
            window_size,
            drop_path,
            type="W",
            mlp_ratio=mlp_ratio,
        )
        self.block_2 = Block(
            input_dim,
            output_dim,
            head_dim,
            window_size,
            drop_path,
            type="SW",
            mlp_ratio=mlp_ratio,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = input_tensor.permute(0, 2, 3, 1).contiguous()
        output = self.block_1(output)
        output = self.block_2(output)
        return output.permute(0, 3, 1, 2).contiguous()


class ConvTransBlock(nn.Module):
    def __init__(
        self,
        conv_dim: int,
        trans_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        type: str = "W",
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if type not in {"W", "SW"}:
            raise ValueError(f"Unsupported attention type: {type}")

        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.conv1_1 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1)
        self.conv1_2 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1)
        self.conv_block = ResidualBlock(conv_dim, conv_dim)
        self.trans_block = Block(
            trans_dim,
            trans_dim,
            head_dim,
            window_size,
            drop_path,
            type=type,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        mixed = self.conv1_1(input_tensor)
        conv_tensor, trans_tensor = torch.split(
            mixed,
            (self.conv_dim, self.trans_dim),
            dim=1,
        )
        conv_tensor = self.conv_block(conv_tensor) + conv_tensor
        trans_tensor = trans_tensor.permute(0, 2, 3, 1).contiguous()
        trans_tensor = self.trans_block(trans_tensor)
        trans_tensor = trans_tensor.permute(0, 3, 1, 2).contiguous()
        output = torch.cat((conv_tensor, trans_tensor), dim=1)
        return input_tensor + self.conv1_2(output)


class SWAtten(AttentionBlock):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        inter_dim: Optional[int] = 192,
    ) -> None:
        hidden_dim = inter_dim or input_dim
        super().__init__(N=hidden_dim)
        self.in_conv = (
            conv1x1(input_dim, hidden_dim) if inter_dim is not None else nn.Identity()
        )
        self.out_conv = (
            conv1x1(hidden_dim, output_dim) if inter_dim is not None else nn.Identity()
        )
        self.non_local_block = SwinBlock(
            hidden_dim,
            hidden_dim,
            head_dim,
            window_size,
            drop_path,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.in_conv(input_tensor)
        identity = output
        non_local = self.non_local_block(output)
        output = self.conv_a(output) * torch.sigmoid(self.conv_b(non_local))
        output = output + identity
        return self.out_conv(output)


class _WinResidualUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            conv1x1(channels, channels // 2),
            nn.GELU(),
            conv3x3(channels // 2, channels // 2),
            nn.GELU(),
            conv1x1(channels // 2, channels),
        )
        self.act = nn.GELU()

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.act(self.conv(input_tensor) + input_tensor)


class _WinBasedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        drop_path: float,
    ) -> None:
        super().__init__()
        attention_type = "SW" if shift_size > 0 else "W"
        self.attn = WMSA(
            input_dim=dim,
            output_dim=dim,
            head_dim=dim // num_heads,
            window_size=window_size,
            type=attention_type,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = input_tensor.permute(0, 2, 3, 1).contiguous()
        output = self.attn(output)
        output = output.permute(0, 3, 1, 2).contiguous()
        return input_tensor + self.drop_path(output)


class WinNoShiftAttention(nn.Module):
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
            _WinResidualUnit(dim),
            _WinResidualUnit(dim),
            _WinResidualUnit(dim),
        )
        self.conv_b = nn.Sequential(
            _WinBasedAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                drop_path=drop_path,
            ),
            _WinResidualUnit(dim),
            _WinResidualUnit(dim),
            _WinResidualUnit(dim),
            conv1x1(dim, dim),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor + self.conv_a(input_tensor) * torch.sigmoid(
            self.conv_b(input_tensor)
        )


class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer: type[nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, input_tensor: Tensor, height: int, width: int) -> Tensor:
        batch_size, length, channels = input_tensor.shape
        if length != height * width:
            raise ValueError("Input feature has wrong size.")

        output = input_tensor.view(batch_size, height, width, channels)
        if height % 2 == 1 or width % 2 == 1:
            output = F.pad(output, (0, 0, 0, width % 2, 0, height % 2))

        x0 = output[:, 0::2, 0::2, :]
        x1 = output[:, 1::2, 0::2, :]
        x2 = output[:, 0::2, 1::2, :]
        x3 = output[:, 1::2, 1::2, :]
        output = torch.cat([x0, x1, x2, x3], dim=-1)
        output = output.view(batch_size, -1, 4 * channels)
        return self.reduction(self.norm(output))


class PatchSplit(nn.Module):
    def __init__(self, dim: int, norm_layer: type[nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        self.reduction = nn.Linear(dim, dim * 2, bias=False)
        self.norm = norm_layer(dim)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, input_tensor: Tensor, height: int, width: int) -> Tensor:
        batch_size, length, channels = input_tensor.shape
        if length != height * width:
            raise ValueError("Input feature has wrong size.")

        output = self.reduction(self.norm(input_tensor))
        output = output.permute(0, 2, 1).contiguous()
        output = output.view(batch_size, 2 * channels, height, width)
        output = self.shuffle(output)
        output = output.permute(0, 2, 3, 1).contiguous()
        return output.view(batch_size, 4 * length, -1)


def __getattr__(name):
    if name == "Win_noShift_Attention":
        import warnings

        warnings.warn(
            "Win_noShift_Attention is deprecated; use WinNoShiftAttention instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return WinNoShiftAttention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
