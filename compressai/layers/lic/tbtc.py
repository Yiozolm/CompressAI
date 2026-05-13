from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.layers.attn.swin_attention import build_window_attention_mask
from compressai.layers.attn.swin_attention import window_partition, window_reverse

__all__ = [
    "TBTCAnalysisTransform",
    "TBTCBasicLayer",
    "TBTCHyperAnalysisTransform",
    "TBTCHyperSynthesisTransform",
    "TBTCPatchEmbed",
    "TBTCPatchMerging",
    "TBTCPatchSplitting",
    "TBTCSwinTransformerBlock",
    "TBTCSynthesisTransform",
]


def _stage_values(value: Optional[int] | Sequence[Optional[int]], count: int) -> list[Optional[int]]:
    if isinstance(value, Sequence):
        return list(value)
    return [value for _ in range(count)]


class TBTCPatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: Optional[int], norm_layer: type[nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = int(dim)
        self.out_dim = int(out_dim or 2 * dim)
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, height, width, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"expected {self.dim} channels, got {channels}")
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("TBTCPatchMerging requires even height and width")

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(batch_size, height // 2 * width // 2, 4 * channels)
        x = self.reduction(self.norm(x))
        return x.view(batch_size, height // 2, width // 2, self.out_dim)


class TBTCPatchSplitting(nn.Module):
    def __init__(self, dim: int, out_dim: Optional[int], norm_layer: type[nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = int(dim)
        self.out_dim = int(out_dim or dim)
        self.norm = norm_layer(dim)
        self.reduction = nn.Linear(dim, 4 * self.out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, height, width, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"expected {self.dim} channels, got {channels}")

        x = x.view(batch_size, height * width, channels)
        x = self.reduction(self.norm(x))
        x = x.view(batch_size, height, width, 4 * self.out_dim)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, upscale_factor=2)
        return x.permute(0, 2, 3, 1)


class _TBTCWindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        window_size: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = int(dim)
        self.window_size = (int(window_size), int(window_size))
        self.num_heads = int(num_heads)
        scale_dim = int(head_dim or dim // num_heads)
        self.scale = qk_scale or scale_dim**-0.5

        table_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))

        coords = torch.stack(
            torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_windows, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, num_tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query * self.scale) @ key.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.view(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                num_tokens,
                num_tokens,
            )
            attention = attention + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, num_tokens, num_tokens)

        attention = F.softmax(attention, dim=-1)
        attention = self.attn_drop(attention)
        x = (attention @ value).transpose(1, 2).reshape(batch_windows, num_tokens, channels)
        return self.proj_drop(self.proj(x))


class _TBTCMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class TBTCSwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        if not 0 <= shift_size < window_size:
            raise ValueError("shift_size must be in [0, window_size)")

        self.dim = int(dim)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.norm1 = norm_layer(dim)
        self.attn = _TBTCWindowAttention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = _TBTCMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, height, width, channels = x.shape
        if height % self.window_size != 0 or width % self.window_size != 0:
            raise ValueError("Swin block input must be divisible by window_size")

        shortcut = x.view(batch_size, height * width, channels)
        x = self.norm1(shortcut).view(batch_size, height, width, channels)
        if self.shift_size > 0:
            mask = build_window_attention_mask(
                height, width, self.window_size, self.shift_size, x.device
            )
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            mask = None

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)
        x_windows = self.attn(x_windows, mask=mask)
        x_windows = x_windows.view(-1, self.window_size, self.window_size, channels)
        x = window_reverse(x_windows, self.window_size, height, width)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x.view(batch_size, height * width, channels)
        x = x + self.mlp(self.norm2(x))
        return x.view(batch_size, height, width, channels)


class TBTCBasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: Optional[int],
        depth: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample: Optional[type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.depth = int(depth)
        self.blocks = nn.ModuleList(
            TBTCSwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                shift_size=0 if index % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
            )
            for index in range(depth)
        )
        self.downsample = (
            downsample(dim=dim, out_dim=out_dim, norm_layer=norm_layer) if downsample is not None else None
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class TBTCPatchEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: int = 2,
        norm_layer: Optional[type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, out_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(out_dim) if norm_layer is not None else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


def _build_layers(
    embed_dim: Sequence[int],
    embed_out_dim: Sequence[Optional[int]],
    depths: Sequence[int],
    head_dim: Optional[int] | Sequence[Optional[int]],
    window_size: int | Sequence[int],
    *,
    downsample: type[nn.Module],
    skip_last_downsample: bool,
) -> nn.ModuleList:
    head_dims = _stage_values(head_dim, len(depths))
    window_sizes = _stage_values(window_size, len(depths))
    return nn.ModuleList(
        TBTCBasicLayer(
            dim=embed_dim[index],
            out_dim=embed_out_dim[index],
            head_dim=head_dims[index],
            depth=depths[index],
            window_size=window_sizes[index],
            downsample=(
                None
                if skip_last_downsample and index == len(depths) - 1
                else downsample
            ),
        )
        for index in range(len(depths))
    )


class TBTCAnalysisTransform(nn.Module):
    def __init__(
        self,
        embed_dim: Sequence[int],
        embed_out_dim: Sequence[Optional[int]],
        depths: Sequence[int],
        head_dim: Optional[int] | Sequence[Optional[int]],
        window_size: int | Sequence[int],
        input_dim: int,
    ) -> None:
        super().__init__()
        self.patch_embed = TBTCPatchEmbed(dim=input_dim, out_dim=embed_dim[0])
        self.layers = _build_layers(embed_dim, embed_out_dim, depths, head_dim, window_size, downsample=TBTCPatchMerging, skip_last_downsample=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2)


class TBTCSynthesisTransform(nn.Module):
    def __init__(
        self,
        embed_dim: Sequence[int],
        embed_out_dim: Sequence[Optional[int]],
        depths: Sequence[int],
        head_dim: Optional[int] | Sequence[Optional[int]],
        window_size: int | Sequence[int],
    ) -> None:
        super().__init__()
        self.layers = _build_layers(embed_dim, embed_out_dim, depths, head_dim, window_size, downsample=TBTCPatchSplitting, skip_last_downsample=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2)


class TBTCHyperAnalysisTransform(nn.Module):
    def __init__(
        self,
        embed_dim: Sequence[int],
        embed_out_dim: Sequence[Optional[int]],
        depths: Sequence[int],
        head_dim: Optional[int] | Sequence[Optional[int]],
        window_size: int | Sequence[int],
        input_dim: int,
    ) -> None:
        super().__init__()
        self.patch_merger = TBTCPatchEmbed(dim=input_dim, out_dim=embed_out_dim[0])
        self.layers = _build_layers(embed_dim, embed_out_dim, depths, head_dim, window_size, downsample=TBTCPatchMerging, skip_last_downsample=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_merger(x)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2)


class TBTCHyperSynthesisTransform(TBTCSynthesisTransform):
    pass
