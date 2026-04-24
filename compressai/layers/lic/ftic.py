from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath
from torch import Tensor

from ..layers import (
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

__all__ = [
    "BranchWindowAttention",
    "FATBlock",
    "FTICAnalysisTransform",
    "FTICHyperAnalysisTransform",
    "FTICHyperSynthesisTransform",
    "FTICSynthesisTransform",
    "SwinFDWA",
    "WindowFrequencyModulation",
]


def _branch_window_size(split_size: int, branch_index: int) -> Tuple[int, int]:
    if branch_index == 0:
        return split_size * 2, split_size * 2
    if branch_index == 1:
        return max(1, split_size // 2), max(1, split_size // 2)
    if branch_index == 2:
        return max(1, split_size // 2), split_size * 2
    if branch_index == 3:
        return split_size * 2, max(1, split_size // 2)
    raise ValueError(f"Unsupported branch index: {branch_index}")


def _pad_bhwc(
    input_tensor: Tensor,
    window_height: int,
    window_width: int,
) -> Tuple[Tensor, int, int]:
    _, height, width, _ = input_tensor.shape
    pad_height = (window_height - height % window_height) % window_height
    pad_width = (window_width - width % window_width) % window_width
    if pad_height == 0 and pad_width == 0:
        return input_tensor, 0, 0

    output = input_tensor.permute(0, 3, 1, 2).contiguous()
    output = F.pad(output, (0, pad_width, 0, pad_height))
    return output.permute(0, 2, 3, 1).contiguous(), pad_height, pad_width


def _window_partition(
    input_tensor: Tensor,
    window_height: int,
    window_width: int,
) -> Tensor:
    batch_size, height, width, channels = input_tensor.shape
    output = input_tensor.view(
        batch_size,
        height // window_height,
        window_height,
        width // window_width,
        window_width,
        channels,
    )
    output = output.permute(0, 1, 3, 2, 4, 5).contiguous()
    return output.view(-1, window_height * window_width, channels)


def _window_reverse(
    windows: Tensor,
    window_height: int,
    window_width: int,
    height: int,
    width: int,
) -> Tensor:
    windows_per_image = (height // window_height) * (width // window_width)
    batch_size = windows.shape[0] // windows_per_image
    output = windows.view(
        batch_size,
        height // window_height,
        width // window_width,
        window_height,
        window_width,
        -1,
    )
    output = output.permute(0, 1, 3, 2, 4, 5).contiguous()
    return output.view(batch_size, height, width, -1)


class BranchWindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        branch_index: int,
        split_size: int = 8,
        num_heads: int = 2,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.window_height, self.window_width = _branch_window_size(
            split_size,
            branch_index,
        )
        table_size = (2 * self.window_height - 1) * (2 * self.window_width - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, self.num_heads)
        )

        coords_h = torch.arange(self.window_height)
        coords_w = torch.arange(self.window_width)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_height - 1
        relative_coords[:, :, 1] += self.window_width - 1
        relative_coords[:, :, 0] *= 2 * self.window_width - 1
        self.register_buffer(
            "relative_position_index",
            relative_coords.sum(-1),
        )
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, qkv: Tensor, spatial_size: Tuple[int, int]) -> Tensor:
        _, _, height, width, channels = qkv.shape
        del spatial_size
        query, key, value = qkv[0], qkv[1], qkv[2]
        query, pad_h, pad_w = _pad_bhwc(query, self.window_height, self.window_width)
        key, _, _ = _pad_bhwc(key, self.window_height, self.window_width)
        value, _, _ = _pad_bhwc(value, self.window_height, self.window_width)
        padded_height, padded_width = query.shape[1], query.shape[2]

        query = _window_partition(query, self.window_height, self.window_width)
        key = _window_partition(key, self.window_height, self.window_width)
        value = _window_partition(value, self.window_height, self.window_width)

        query = query.view(
            -1,
            self.window_height * self.window_width,
            self.num_heads,
            channels // self.num_heads,
        ).permute(0, 2, 1, 3)
        key = key.view(
            -1,
            self.window_height * self.window_width,
            self.num_heads,
            channels // self.num_heads,
        ).permute(0, 2, 1, 3)
        value = value.view(
            -1,
            self.window_height * self.window_width,
            self.num_heads,
            channels // self.num_heads,
        ).permute(0, 2, 1, 3)

        attention = (query * self.scale) @ key.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            self.window_height * self.window_width,
            self.window_height * self.window_width,
            -1,
        ).permute(2, 0, 1)
        attention = attention + relative_position_bias.unsqueeze(0)
        attention = self.attn_drop(attention.softmax(dim=-1))

        output = attention @ value
        output = output.permute(0, 2, 1, 3).contiguous().view(
            -1,
            self.window_height * self.window_width,
            channels,
        )
        output = _window_reverse(
            output,
            self.window_height,
            self.window_width,
            padded_height,
            padded_width,
        )
        if pad_h > 0 or pad_w > 0:
            output = output[:, :height, :width, :].contiguous()
        return output


class WindowFrequencyModulation(nn.Module):
    def __init__(self, dim: int, window_size: int) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.complex_weight = nn.Parameter(
            torch.cat(
                (
                    torch.ones(
                        self.window_size,
                        self.window_size // 2 + 1,
                        dim,
                        1,
                        dtype=torch.float32,
                    ),
                    torch.zeros(
                        self.window_size,
                        self.window_size // 2 + 1,
                        dim,
                        1,
                        dtype=torch.float32,
                    ),
                ),
                dim=-1,
            )
        )

    def forward(self, input_tensor: Tensor, height: int, width: int) -> Tensor:
        batch_size, _, channels = input_tensor.shape
        output = input_tensor.view(batch_size, height, width, channels)
        output, pad_h, pad_w = _pad_bhwc(output, self.window_size, self.window_size)
        padded_height, padded_width = output.shape[1], output.shape[2]
        output = output.view(
            batch_size,
            padded_height // self.window_size,
            self.window_size,
            padded_width // self.window_size,
            self.window_size,
            channels,
        ).permute(0, 1, 3, 2, 4, 5)
        output = torch.fft.rfft2(output.to(torch.float32), dim=(3, 4), norm="ortho")
        output = output * torch.view_as_complex(self.complex_weight)
        output = torch.fft.irfft2(
            output,
            s=(self.window_size, self.window_size),
            dim=(3, 4),
            norm="ortho",
        )
        output = output.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            batch_size,
            padded_height,
            padded_width,
            channels,
        )
        if pad_h > 0 or pad_w > 0:
            output = output[:, :height, :width, :].contiguous()
        return output.view(batch_size, -1, channels)


class SwinFDWA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        fm_window_size: int = 16,
        shift_size: int = 4,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("dim must be divisible by 4")
        if num_heads % 4 != 0:
            raise ValueError("num_heads must be divisible by 4")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        branch_dim = dim // 4
        branch_heads = num_heads // 4
        self.split_size = int(window_size)
        self.shift_size = int(shift_size)
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.branch_attentions = nn.ModuleList(
            BranchWindowAttention(
                branch_dim,
                branch_index=index,
                split_size=self.split_size,
                num_heads=branch_heads,
                attn_drop=attn_drop,
            )
            for index in range(4)
        )
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.frequency_modulation = WindowFrequencyModulation(dim, fm_window_size)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, input_tensor: Tensor, spatial_size: Tuple[int, int]) -> Tensor:
        height, width = spatial_size
        batch_size, length, channels = input_tensor.shape
        if length != height * width:
            raise ValueError("Flattened tokens do not match spatial size")

        output = self.norm1(input_tensor)
        qkv = self.qkv(output).reshape(batch_size, length, 3, channels).permute(2, 0, 1, 3)
        qkv = qkv.view(3, batch_size, height, width, channels)
        branch_qkv = qkv.chunk(4, dim=-1)
        shifts = (
            (self.split_size, self.split_size),
            (max(1, self.split_size // 4), max(1, self.split_size // 4)),
            (max(1, self.split_size // 4), self.split_size),
            (self.split_size, max(1, self.split_size // 4)),
        )

        outputs = []
        for index, qkv_branch in enumerate(branch_qkv):
            if self.shift_size > 0:
                shift_h, shift_w = shifts[index]
                qkv_branch = torch.roll(
                    qkv_branch,
                    shifts=(-shift_h, -shift_w),
                    dims=(2, 3),
                )
            branch_output = self.branch_attentions[index](qkv_branch, spatial_size)
            if self.shift_size > 0:
                shift_h, shift_w = shifts[index]
                branch_output = torch.roll(
                    branch_output,
                    shifts=(shift_h, shift_w),
                    dims=(1, 2),
                )
            outputs.append(branch_output.view(batch_size, length, channels // 4))

        attended = torch.cat(outputs, dim=-1)
        attended = self.proj_drop(self.proj(attended))
        output = input_tensor + self.drop_path(attended)
        output = output + self.frequency_modulation(self.ffn(self.norm2(output)), height, width)
        return output


class FATBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        fm_window_size: int,
        drop_path: float,
        attention_type: str = "W",
    ) -> None:
        super().__init__()
        if attention_type not in {"W", "SW"}:
            raise ValueError(f"Unsupported attention_type: {attention_type}")

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.frequency_attention = SwinFDWA(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            fm_window_size=fm_window_size,
            shift_size=0 if attention_type == "W" else window_size // 2,
            drop_path=drop_path,
        )
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.conv1(input_tensor)
        batch_size, channels, height, width = output.shape
        output = output.view(batch_size, channels, height * width).transpose(1, 2)
        output = self.frequency_attention(output, (height, width))
        output = output.transpose(1, 2).reshape(batch_size, channels, height, width)
        return input_tensor + self.conv2(output)


def _make_ftic_stage(
    channels: int,
    depth: int,
    num_heads: int,
    window_size: int,
    fm_window_size: int,
    drop_paths: Sequence[float],
    tail: nn.Module,
) -> nn.ModuleDict:
    if len(drop_paths) != depth:
        raise ValueError("drop_paths must match stage depth")
    blocks = [
        FATBlock(
            dim=channels,
            num_heads=num_heads,
            window_size=window_size,
            fm_window_size=fm_window_size,
            drop_path=float(drop_paths[index]),
            attention_type="W" if index % 2 == 0 else "SW",
        )
        for index in range(depth)
    ]
    return nn.ModuleDict(
        {
            "blocks": nn.ModuleList(blocks),
            "tail": tail,
        }
    )


def _run_ftic_stage(stage: nn.ModuleDict, input_tensor: Tensor) -> Tensor:
    output = input_tensor
    for block in stage["blocks"]:
        output = block(output)
    return stage["tail"](output)


class FTICAnalysisTransform(nn.Module):
    def __init__(
        self,
        feature_dims: Tuple[int, int, int],
        M: int,
        config: Sequence[int],
        num_heads: Sequence[int],
        drop_paths: Sequence[Sequence[float]],
        *,
        window_size: int = 8,
        fm_window_size: int = 16,
    ) -> None:
        super().__init__()
        dim0, dim1, dim2 = feature_dims
        self.input_block = ResidualBlockWithStride(3, dim0, stride=2)
        self.stage1 = _make_ftic_stage(
            dim0,
            int(config[0]),
            int(num_heads[0]),
            window_size,
            fm_window_size,
            drop_paths[0],
            ResidualBlockWithStride(dim0, dim1, stride=2),
        )
        self.stage2 = _make_ftic_stage(
            dim1,
            int(config[1]),
            int(num_heads[1]),
            window_size,
            fm_window_size,
            drop_paths[1],
            ResidualBlockWithStride(dim1, dim2, stride=2),
        )
        self.stage3 = _make_ftic_stage(
            dim2,
            int(config[2]),
            int(num_heads[2]),
            window_size,
            fm_window_size,
            drop_paths[2],
            conv3x3(dim2, M, stride=2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.input_block(input_tensor)
        output = _run_ftic_stage(self.stage1, output)
        output = _run_ftic_stage(self.stage2, output)
        return _run_ftic_stage(self.stage3, output)


class FTICSynthesisTransform(nn.Module):
    def __init__(
        self,
        feature_dims: Tuple[int, int, int],
        M: int,
        config: Sequence[int],
        num_heads: Sequence[int],
        drop_paths: Sequence[Sequence[float]],
        *,
        window_size: int = 8,
        fm_window_size: int = 16,
    ) -> None:
        super().__init__()
        dim0, dim1, dim2 = feature_dims
        self.input_block = ResidualBlockUpsample(M, dim2, upsample=2)
        self.stage1 = _make_ftic_stage(
            dim2,
            int(config[3]),
            int(num_heads[3]),
            window_size,
            fm_window_size,
            drop_paths[3],
            ResidualBlockUpsample(dim2, dim1, upsample=2),
        )
        self.stage2 = _make_ftic_stage(
            dim1,
            int(config[4]),
            int(num_heads[4]),
            window_size,
            fm_window_size,
            drop_paths[4],
            ResidualBlockUpsample(dim1, dim0, upsample=2),
        )
        self.stage3 = _make_ftic_stage(
            dim0,
            int(config[5]),
            int(num_heads[5]),
            window_size,
            fm_window_size,
            drop_paths[5],
            subpel_conv3x3(dim0, 3, 2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.input_block(input_tensor)
        output = _run_ftic_stage(self.stage1, output)
        output = _run_ftic_stage(self.stage2, output)
        return _run_ftic_stage(self.stage3, output)


class FTICHyperAnalysisTransform(nn.Module):
    def __init__(
        self,
        M: int,
        hyper_hidden_channels: int,
        hyper_channels: int,
        depth: int,
        num_heads: int,
        *,
        window_size: int = 2,
        fm_window_size: int = 4,
    ) -> None:
        super().__init__()
        self.input_block = ResidualBlockWithStride(M, hyper_hidden_channels, stride=2)
        self.stage = _make_ftic_stage(
            hyper_hidden_channels,
            depth,
            num_heads,
            window_size,
            fm_window_size,
            [0.0] * depth,
            conv3x3(hyper_hidden_channels, hyper_channels, stride=2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.input_block(input_tensor)
        return _run_ftic_stage(self.stage, output)


class FTICHyperSynthesisTransform(nn.Module):
    def __init__(
        self,
        M: int,
        hyper_hidden_channels: int,
        hyper_channels: int,
        depth: int,
        num_heads: int,
        *,
        window_size: int = 2,
        fm_window_size: int = 4,
    ) -> None:
        super().__init__()
        self.input_block = ResidualBlockUpsample(
            hyper_channels,
            hyper_hidden_channels,
            upsample=2,
        )
        self.stage = _make_ftic_stage(
            hyper_hidden_channels,
            depth,
            num_heads,
            window_size,
            fm_window_size,
            [0.0] * depth,
            subpel_conv3x3(hyper_hidden_channels, M, 2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.input_block(input_tensor)
        return _run_ftic_stage(self.stage, output)
