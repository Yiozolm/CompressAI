"""Residual Swin Transformer Blocks for TIC.

Vendored from `Lu, Guo, Shi, Cao & Ma, "Transformer-based Image Compression"
<https://arxiv.org/abs/2111.06707>`_ (DCC 2022). Module / parameter naming is
kept verbatim from upstream (``residual_group.blocks.{i}.norm1`` /
``attn.qkv`` / ...) so TIC checkpoints load without translation.

The Swin block here is the *image-restoration* RSTB variant (alternating
``shift_size = 0 / window_size // 2``, dynamic mask rebuild keyed by
``x_size``), distinct from :class:`compressai.layers.attn.swin.SwinBlock`
which is the STF / WACNN flavour with ``output_proj`` / ``residual_proj``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from timm.layers import DropPath, Mlp
from torch import Tensor

from .swin_attention import (
    WindowAttention,
    build_window_attention_mask,
    window_partition,
    window_reverse,
)


__all__ = [
    "BasicSwinLayer",
    "PatchEmbed1D",
    "PatchUnEmbed1D",
    "RSTB",
    "SwinTransformerBlock",
]


class PatchEmbed1D(nn.Module):
    """Flatten 2D feature map to a sequence of tokens (B, H*W, C)."""

    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor.flatten(2).transpose(1, 2)


class PatchUnEmbed1D(nn.Module):
    """Reshape token sequence back to 2D feature map."""

    def forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        batch_size = input_tensor.shape[0]
        return input_tensor.transpose(1, 2).view(
            batch_size, -1, x_size[0], x_size[1]
        )


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with window or shifted-window attention.

    Args:
        dim: Input / output channels.
        input_resolution: Default ``(H, W)`` for the cached attention mask;
            when ``forward`` receives a different ``x_size`` the mask is
            rebuilt on the fly.
        num_heads: Attention heads.
        window_size: Window edge length.
        shift_size: 0 for W-MSA, ``window_size // 2`` for SW-MSA.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError("shift_size must be in [0, window_size).")

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        attn_mask = build_window_attention_mask(
            input_resolution[0],
            input_resolution[1],
            self.window_size,
            self.shift_size,
            torch.device("cpu"),
        )
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        height, width = x_size
        batch_size, length, channels = input_tensor.shape

        shortcut = input_tensor
        output = self.norm1(input_tensor).view(batch_size, height, width, channels)

        if self.shift_size > 0:
            output = torch.roll(
                output,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )

        windows = window_partition(output, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, channels)

        if self.input_resolution == tuple(x_size):
            mask = self.attn_mask
        else:
            mask = build_window_attention_mask(
                height,
                width,
                self.window_size,
                self.shift_size,
                input_tensor.device,
            )

        attn_windows = self.attn(windows, mask=mask)
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, channels
        )
        output = window_reverse(attn_windows, self.window_size, height, width)

        if self.shift_size > 0:
            output = torch.roll(
                output,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        output = output.view(batch_size, height * width, channels)

        output = shortcut + self.drop_path(output)
        return output + self.drop_path(self.mlp(self.norm2(output)))


class BasicSwinLayer(nn.Module):
    """A stack of Swin Transformer blocks alternating W / SW attention."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        for block in self.blocks:
            if self.use_checkpoint:
                input_tensor = checkpoint.checkpoint(block, input_tensor, x_size)
            else:
                input_tensor = block(input_tensor, x_size)
        return input_tensor


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Wraps :class:`BasicSwinLayer` with patch (un)embed and a residual
    connection so it can drop into a CNN-shaped feature map. ``forward``
    takes an explicit ``x_size`` so the block transparently handles inputs
    whose spatial extent differs from the default ``input_resolution``.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicSwinLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )

        self.patch_embed = PatchEmbed1D()
        self.patch_unembed = PatchUnEmbed1D()

    def forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        tokens = self.patch_embed(input_tensor)
        tokens = self.residual_group(tokens, x_size)
        return self.patch_unembed(tokens, x_size) + input_tensor
