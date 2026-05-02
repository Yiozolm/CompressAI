from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from timm.layers import DropPath, to_2tuple
from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GsnConditionalLocScaleShift
from compressai.layers import (
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.layers.attn.swin_attention import pad_to_window_multiple
from compressai.registry import register_model

from .base import CompressionModel
from .utils import update_registered_buffers

__all__ = ["FrequencyAwareTransFormer", "convert_upstream_state_dict"]


# ---------------------------------------------------------------------------
# FTIC transforms (formerly compressai/layers/lic/ftic.py)
# ---------------------------------------------------------------------------


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
        query, pad_h, pad_w = pad_to_window_multiple(
            query, (self.window_height, self.window_width), layout="BHWC"
        )
        key, _, _ = pad_to_window_multiple(
            key, (self.window_height, self.window_width), layout="BHWC"
        )
        value, _, _ = pad_to_window_multiple(
            value, (self.window_height, self.window_width), layout="BHWC"
        )
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
        output, pad_h, pad_w = pad_to_window_multiple(output, self.window_size, layout="BHWC")
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
        # Match upstream's Python operator-precedence quirk: it writes
        # ``-split_size // 4``, which parses as ``(-split_size) // 4`` and
        # differs from ``-(split_size // 4)`` for split_size < 4 (the hyper
        # transforms use split_size=2, giving -1 vs 0). The rollback path
        # uses positive ``split_size // 4`` directly, so the asymmetry is
        # baked into the trained weights and must be preserved.
        forward_shifts = (
            (-self.split_size, -self.split_size),
            (-self.split_size // 4, -self.split_size // 4),
            (-self.split_size // 4, -self.split_size),
            (-self.split_size, -self.split_size // 4),
        )
        rollback_shifts = (
            (self.split_size, self.split_size),
            (self.split_size // 4, self.split_size // 4),
            (self.split_size // 4, self.split_size),
            (self.split_size, self.split_size // 4),
        )

        outputs = []
        for index, qkv_branch in enumerate(branch_qkv):
            if self.shift_size > 0:
                shift_h, shift_w = forward_shifts[index]
                qkv_branch = torch.roll(
                    qkv_branch,
                    shifts=(shift_h, shift_w),
                    dims=(2, 3),
                )
            branch_output = self.branch_attentions[index](qkv_branch, spatial_size)
            if self.shift_size > 0:
                shift_h, shift_w = rollback_shifts[index]
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


# ---------------------------------------------------------------------------
# T-CA entropy model (formerly compressai/layers/lic/tca.py)
# ---------------------------------------------------------------------------


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
        output, pad_h, pad_w = pad_to_window_multiple(input_tensor, self.window_size)
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


# ---------------------------------------------------------------------------
# FrequencyAwareTransFormer (top-level model + state-dict converter)
# ---------------------------------------------------------------------------


def _infer_ftic_stage_depth(state_dict: Dict[str, Tensor], prefix: str) -> int:
    indices = {
        int(key.split(".")[3])
        for key in state_dict
        if key.startswith(prefix) and ".conv1.weight" in key
    }
    return len(indices)


def _infer_ftic_num_heads(state_dict: Dict[str, Tensor], prefix: str) -> int:
    key = f"{prefix}.blocks.0.frequency_attention.branch_attentions.0.relative_position_bias_table"
    return state_dict[key].size(1) * 4


def _infer_window_size(state_dict: Dict[str, Tensor], prefix: str) -> int:
    key = f"{prefix}.blocks.0.frequency_attention.branch_attentions.0.relative_position_bias_table"
    table_size = state_dict[key].size(0)
    return (int(round(table_size**0.5)) + 1) // 4


def _infer_fm_window_size(state_dict: Dict[str, Tensor], prefix: str) -> int:
    key = f"{prefix}.blocks.0.frequency_attention.frequency_modulation.complex_weight"
    return state_dict[key].size(0)


def _infer_num_slices(state_dict: Dict[str, Tensor], M: int) -> int:
    start_token_channels = state_dict["tca.tca.start_token_from_hyperprior.weight"].size(0)
    return M // start_token_channels


def _infer_tca_depth(state_dict: Dict[str, Tensor]) -> int:
    indices = {
        int(key.split(".")[3])
        for key in state_dict
        if key.startswith("tca.tca.layers.") and ".q_proj.weight" in key
    }
    return len(indices)


def _infer_tca_ratio(state_dict: Dict[str, Tensor], M: int) -> int:
    return state_dict["tca.tca.lift.weight"].size(0) // M


def ste_round(input_tensor: Tensor) -> Tensor:
    return torch.round(input_tensor) - input_tensor.detach() + input_tensor


def _split_drop_paths(
    drop_path_rate: float,
    config: Sequence[int],
) -> Tuple[Sequence[float], ...]:
    all_paths = torch.linspace(0.0, drop_path_rate, sum(config)).tolist()
    splits = []
    offset = 0
    for depth in config:
        splits.append(tuple(float(value) for value in all_paths[offset : offset + depth]))
        offset += depth
    return tuple(splits)


def _is_upstream_state_dict(state_dict: Dict[str, Tensor]) -> bool:
    """Detect whether ``state_dict`` follows the upstream FLIC layout."""
    return "g_a.0.conv1.weight" in state_dict and "g_a.input_block.conv1.weight" not in state_dict


_FAT_BLOCK_RENAMES = {
    "conv1_1": "conv1",
    "conv1_2": "conv2",
}

_TRANS_BLOCK_RENAMES = {
    "fm": "frequency_modulation",
    "attns": "branch_attentions",
}

_TCA_LAYER_RENAMES = {
    "q1": "q_proj",
    "k1": "k_proj",
    "v1": "v_proj",
    "attn": "attention",
}

_GAUSSIAN_DROP_SUFFIXES = (
    "_indexes_table",
    "lower_bound_zero.bound",
    "upper_bound_mean.bound",
    "upper_bound_scale.bound",
)


def _rename_fat_block_inner(suffix: str) -> str:
    """Translate a sub-key inside a FAT_Block to compressai naming."""
    head, _, rest = suffix.partition(".")
    if head == "trans_block":
        sub_head, _, sub_rest = rest.partition(".")
        sub_head = _TRANS_BLOCK_RENAMES.get(sub_head, sub_head)
        new_rest = f"{sub_head}.{sub_rest}" if sub_rest else sub_head
        return f"frequency_attention.{new_rest}"
    if head in _FAT_BLOCK_RENAMES:
        new_head = _FAT_BLOCK_RENAMES[head]
        return f"{new_head}.{rest}" if rest else new_head
    return suffix


def _rename_transform_block(
    prefix: str,
    indices: Sequence[int],
    state_dict: Dict[str, Tensor],
) -> Tuple[Dict[str, Tensor], Tuple[int, ...]]:
    """Map a flat ``Sequential`` transform (``g_a``/``g_s``/``h_a``/...) onto
    the named sub-modules used by the compressai port.

    Returns the renamed sub-state-dict plus the discovered config (number of
    FAT_Blocks per stage). The number of stages is inferred from the data: each
    block is detected by the presence of a ``trans_block.qkv.weight`` key, and
    consecutive runs of blocks form a stage; the index immediately following
    each run is the stage tail. The first non-block index is always the
    ``input_block``.
    """
    sorted_indices = sorted(indices)
    if not sorted_indices:
        raise ValueError(f"no indices found under prefix {prefix!r}")

    block_flags = [
        f"{prefix}.{idx}.trans_block.qkv.weight" in state_dict for idx in sorted_indices
    ]
    if block_flags[0]:
        raise ValueError(f"unexpected FAT_Block at {prefix}.0; input_block missing")

    block_indices: List[List[int]] = []
    tail_indices: List[int] = []
    current_blocks: List[int] = []
    for idx, is_block in zip(sorted_indices[1:], block_flags[1:]):
        if is_block:
            current_blocks.append(idx)
        else:
            block_indices.append(current_blocks)
            tail_indices.append(idx)
            current_blocks = []
    if current_blocks:
        raise ValueError(
            f"trailing FAT_Blocks at {prefix} have no following tail; "
            f"unfinished stage indices: {current_blocks}"
        )

    config = tuple(len(stage) for stage in block_indices)

    out: Dict[str, Tensor] = {}

    def emit(old_idx: int, new_subprefix: str, is_block: bool) -> None:
        old_prefix = f"{prefix}.{old_idx}."
        new_prefix = f"{prefix}.{new_subprefix}."
        for old_key, value in state_dict.items():
            if not old_key.startswith(old_prefix):
                continue
            suffix = old_key[len(old_prefix) :]
            if is_block:
                suffix = _rename_fat_block_inner(suffix)
            out[new_prefix + suffix] = value

    emit(sorted_indices[0], "input_block", is_block=False)
    if len(block_indices) == 1:
        # Hyper transforms expose a single stage as ``stage`` rather than ``stage1``.
        for block_idx, source_idx in enumerate(block_indices[0]):
            emit(source_idx, f"stage.blocks.{block_idx}", is_block=True)
        emit(tail_indices[0], "stage.tail", is_block=False)
    else:
        for stage_idx, (stage_blocks, tail_idx) in enumerate(
            zip(block_indices, tail_indices), start=1
        ):
            for block_idx, source_idx in enumerate(stage_blocks):
                emit(source_idx, f"stage{stage_idx}.blocks.{block_idx}", is_block=True)
            emit(tail_idx, f"stage{stage_idx}.tail", is_block=False)

    return out, config


def _rename_tca(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Map ``tca.TCA.*`` / ``tca.{hyper_trans,entropy_parameters_net}.*`` onto
    the compressai naming."""
    out: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith("tca."):
            continue
        rest = key[len("tca.") :]
        if rest == "TCA.start_token":
            # Defined upstream but never used by ``TCA.forward``; compressai drops it.
            continue
        if rest.startswith("TCA."):
            sub = rest[len("TCA.") :]
            if sub.startswith("layers."):
                # tca.TCA.layers.{i}.{name}.{...}
                parts = sub.split(".")
                # parts = ["layers", i, name, ...]
                idx = parts[1]
                name = parts[2]
                tail = ".".join(parts[3:])
                if name == "cpe" and len(parts) >= 5 and parts[3] == "0":
                    # cpe.0.{...} -> positional_encoding.{...}
                    tail = ".".join(parts[4:])
                    new_sub = f"layers.{idx}.positional_encoding"
                else:
                    name = _TCA_LAYER_RENAMES.get(name, name)
                    new_sub = f"layers.{idx}.{name}"
                if tail:
                    new_sub = f"{new_sub}.{tail}"
                out[f"tca.tca.{new_sub}"] = value
            else:
                out[f"tca.tca.{sub}"] = value
        elif rest.startswith("hyper_trans."):
            sub = rest[len("hyper_trans.") :]
            if sub == "weight":
                # nn.Linear weight (out, in) -> nn.Conv2d 1x1 weight (out, in, 1, 1)
                out["tca.hyper_trans.weight"] = value.view(value.size(0), value.size(1), 1, 1)
            elif sub == "bias":
                out["tca.hyper_trans.bias"] = value
            else:
                out[f"tca.hyper_trans.{sub}"] = value
        else:
            out[f"tca.{rest}"] = value
    return out


def _rename_gaussian_conditional(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Strip the ``_entropy_model.`` wrapper prefix and drop upstream-only
    bookkeeping buffers."""
    out: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith("gaussian_conditional."):
            continue
        sub = key[len("gaussian_conditional.") :]
        if sub.startswith("_entropy_model."):
            sub = sub[len("_entropy_model.") :]
        if any(sub.endswith(suffix) for suffix in _GAUSSIAN_DROP_SUFFIXES):
            continue
        out[f"gaussian_conditional.{sub}"] = value
    return out


def convert_upstream_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Convert an upstream FTIC checkpoint to the compressai naming.

    Handles the renames documented inline:
    flat ``Sequential`` transforms -> ``input_block`` / ``stage{,1,2,3}.blocks`` /
    ``stage{,1,2,3}.tail``; ``conv1_1`` / ``conv1_2`` -> ``conv1`` / ``conv2``
    inside each FAT_Block; ``trans_block`` -> ``frequency_attention`` (with
    ``attns`` -> ``branch_attentions`` and ``fm`` -> ``frequency_modulation``);
    ``tca.TCA.*`` -> ``tca.tca.*`` (with ``q1``/``k1``/``v1`` ->
    ``q_proj``/``k_proj``/``v_proj``, ``cpe.0`` -> ``positional_encoding``,
    ``attn.proj`` -> ``attention.proj``, drops the unused ``start_token``
    parameter, and reshapes the ``hyper_trans`` Linear weight into a 1x1 Conv2d
    weight); ``gaussian_conditional._entropy_model.*`` ->
    ``gaussian_conditional.*`` (drops ``_indexes_table`` and the unused
    ``lower_bound_zero`` / ``upper_bound_*`` LowerBound buffers).
    """
    indices: Dict[str, List[int]] = {
        prefix: sorted(
            {int(key.split(".")[1]) for key in state_dict if key.startswith(prefix + ".")}
        )
        for prefix in ("g_a", "g_s", "h_a", "h_mean_s", "h_scale_s")
    }
    out: Dict[str, Tensor] = {}
    for prefix in ("g_a", "g_s", "h_a", "h_mean_s", "h_scale_s"):
        renamed, _ = _rename_transform_block(prefix, indices[prefix], state_dict)
        out.update(renamed)
    out.update(_rename_tca(state_dict))
    out.update(_rename_gaussian_conditional(state_dict))
    for key, value in state_dict.items():
        if key.startswith("entropy_bottleneck."):
            out[key] = value
    return out


@register_model("ftic")
class FrequencyAwareTransFormer(CompressionModel):
    r"""Frequency-aware Transformer model from H. Li, S. Li, W. Dai, C. Li,
    J. Zou, H. Xiong: `"Frequency-Aware Transformer for Learned Image
    Compression" <https://openreview.net/forum?id=HKGQDDTuvZ>`_, Int. Conf. on
    Learning Representations (ICLR), 2024.

    Uses frequency-decomposition window attention (FDWA) and frequency
    modulation FFNs together with a transformer-based channel-wise
    autoregressive entropy model (T-CA).

    Args:
        M (int): Number of channels in the latent representation.
        num_slices (int): Number of channel slices for the T-CA entropy model.
    """

    def __init__(
        self,
        config: Sequence[int] = (2, 2, 2, 2, 2, 2),
        num_heads: Sequence[int] = (8, 16, 32, 32, 16, 8),
        drop_path_rate: float = 0.0,
        feature_dims: Tuple[int, int, int] = (96, 144, 256),
        hyper_hidden_channels: int = 256,
        hyper_channels: int = 192,
        M: int = 320,
        num_slices: int = 5,
        num_scales: int = 256,
        num_means: int = 100,
        min_scale: float = 0.01,
        tail_mass: float = 2 ** (-8),
        window_size: int = 8,
        fm_window_size: int = 16,
        hyper_window_size: int = 2,
        hyper_fm_window_size: int = 4,
        hyper_num_heads: int = 32,
        tca_depth: int = 12,
        tca_ratio: int = 4,
        tca_window_size: int = 8,
        tca_num_heads: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if len(config) != 6:
            raise ValueError("config must provide six stage depths")
        if len(num_heads) != 6:
            raise ValueError("num_heads must provide six stage head counts")
        if M % num_slices != 0:
            raise ValueError("M must be divisible by num_slices")
        if M % tca_num_heads != 0:
            raise ValueError("M must be divisible by tca_num_heads")

        self.config = tuple(int(value) for value in config)
        self.num_heads = tuple(int(value) for value in num_heads)
        self.feature_dims = tuple(int(value) for value in feature_dims)
        self.hyper_hidden_channels = int(hyper_hidden_channels)
        self.hyper_channels = int(hyper_channels)
        self.M = int(M)
        self.num_slices = int(num_slices)
        self.num_scales = int(num_scales)
        self.num_means = int(num_means)
        self.min_scale = float(min_scale)
        self.tail_mass = float(tail_mass)
        self.window_size = int(window_size)
        self.fm_window_size = int(fm_window_size)
        self.hyper_window_size = int(hyper_window_size)
        self.hyper_fm_window_size = int(hyper_fm_window_size)
        self.hyper_num_heads = int(hyper_num_heads)
        self.tca_depth = int(tca_depth)
        self.tca_ratio = int(tca_ratio)
        self.tca_window_size = int(tca_window_size)
        self.tca_num_heads = int(tca_num_heads)

        drop_paths = _split_drop_paths(drop_path_rate, self.config)
        self.g_a = FTICAnalysisTransform(
            feature_dims=self.feature_dims,
            M=self.M,
            config=self.config,
            num_heads=self.num_heads,
            drop_paths=drop_paths,
            window_size=self.window_size,
            fm_window_size=self.fm_window_size,
        )
        self.g_s = FTICSynthesisTransform(
            feature_dims=self.feature_dims,
            M=self.M,
            config=self.config,
            num_heads=self.num_heads,
            drop_paths=tuple(reversed(drop_paths)),
            window_size=self.window_size,
            fm_window_size=self.fm_window_size,
        )
        self.h_a = FTICHyperAnalysisTransform(
            M=self.M,
            hyper_hidden_channels=self.hyper_hidden_channels,
            hyper_channels=self.hyper_channels,
            depth=self.config[0],
            num_heads=self.hyper_num_heads,
            window_size=self.hyper_window_size,
            fm_window_size=self.hyper_fm_window_size,
        )
        self.h_mean_s = FTICHyperSynthesisTransform(
            M=self.M,
            hyper_hidden_channels=self.hyper_hidden_channels,
            hyper_channels=self.hyper_channels,
            depth=self.config[3],
            num_heads=self.hyper_num_heads,
            window_size=self.hyper_window_size,
            fm_window_size=self.hyper_fm_window_size,
        )
        self.h_scale_s = FTICHyperSynthesisTransform(
            M=self.M,
            hyper_hidden_channels=self.hyper_hidden_channels,
            hyper_channels=self.hyper_channels,
            depth=self.config[3],
            num_heads=self.hyper_num_heads,
            window_size=self.hyper_window_size,
            fm_window_size=self.hyper_fm_window_size,
        )
        self.tca = TCAEntropyModel(
            dim=self.M,
            depth=self.tca_depth,
            ratio=self.tca_ratio,
            slices=self.num_slices,
            window_size=self.tca_window_size,
            num_heads=self.tca_num_heads,
        )
        self.entropy_bottleneck = EntropyBottleneck(self.hyper_channels)
        self.gaussian_conditional = GsnConditionalLocScaleShift(
            num_scales=self.num_scales,
            num_means=self.num_means,
            min_scale=self.min_scale,
            tail_mass=self.tail_mass,
        )

    def _hyper(self, z_hat: Tensor) -> Tensor:
        return torch.cat((self.h_mean_s(z_hat), self.h_scale_s(z_hat)), dim=1)

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor] | Tensor]:
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        hyper = self._hyper(z_hat)
        y_hat = ste_round(y)
        means, scales, lrp = self.tca(hyper, y_hat)
        _, y_likelihoods = self.gaussian_conditional(y, scales, means)
        y_hat = y_hat + 0.5 * torch.tanh(lrp)
        return {
            "x_hat": self.g_s(y_hat),
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x: Tensor) -> Dict[str, object]:
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper = self._hyper(z_hat)

        y_strings = []
        y_hat_coded = torch.round(y)
        lrp_coded = torch.zeros_like(y)
        channels_per_slice = self.M // self.num_slices
        means, scales, lrps = self.tca(hyper, y_hat_coded)

        for slice_index in range(self.num_slices):
            start = slice_index * channels_per_slice
            end = (slice_index + 1) * channels_per_slice
            mu = means[:, start:end]
            scale = scales[:, start:end]
            lrp = lrps[:, start:end]
            y_slice = y[:, start:end]
            y_hat_slice, slice_strings = self.gaussian_conditional.compress(y_slice, scale, mu)
            y_hat_coded[:, start:end] = y_hat_slice
            lrp_coded[:, start:end] = lrp
            y_strings.append(slice_strings)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "y_hat": y_hat_coded + 0.5 * torch.tanh(lrp_coded),
        }

    def decompress(
        self,
        strings: Sequence[Sequence[bytes] | Sequence[Sequence[bytes]]],
        shape: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        if len(strings) != 2:
            raise ValueError("strings must contain [y_strings, z_strings]")

        y_strings = strings[0]
        z_strings = strings[1]
        if len(y_strings) != self.num_slices:
            raise ValueError("y_strings must contain one entry per slice")

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        hyper = self._hyper(z_hat)
        y_hat_coded = torch.zeros(
            z_hat.size(0),
            self.M,
            z_hat.size(2) * 4,
            z_hat.size(3) * 4,
            device=z_hat.device,
        )
        lrp_coded = torch.zeros_like(y_hat_coded)
        channels_per_slice = self.M // self.num_slices

        for slice_index in range(self.num_slices):
            means, scales, lrps = self.tca(hyper, y_hat_coded)
            start = slice_index * channels_per_slice
            end = (slice_index + 1) * channels_per_slice
            mu = means[:, start:end]
            scale = scales[:, start:end]
            lrp = lrps[:, start:end]
            y_hat_slice = self.gaussian_conditional.decompress(y_strings[slice_index], scale, mu)
            y_hat_coded[:, start:end] = y_hat_slice
            lrp_coded[:, start:end] = lrp

        y_hat = y_hat_coded + 0.5 * torch.tanh(lrp_coded)
        return {"x_hat": self.g_s(y_hat).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "FrequencyAwareTransFormer":
        if _is_upstream_state_dict(state_dict):
            state_dict = convert_upstream_state_dict(state_dict)

        dim0 = state_dict["g_a.input_block.conv1.weight"].size(0)
        dim1 = state_dict["g_a.stage1.tail.conv1.weight"].size(0)
        dim2 = state_dict["g_a.stage2.tail.conv1.weight"].size(0)
        M = state_dict["g_a.stage3.tail.weight"].size(0)
        config = (
            _infer_ftic_stage_depth(state_dict, "g_a.stage1.blocks."),
            _infer_ftic_stage_depth(state_dict, "g_a.stage2.blocks."),
            _infer_ftic_stage_depth(state_dict, "g_a.stage3.blocks."),
            _infer_ftic_stage_depth(state_dict, "g_s.stage1.blocks."),
            _infer_ftic_stage_depth(state_dict, "g_s.stage2.blocks."),
            _infer_ftic_stage_depth(state_dict, "g_s.stage3.blocks."),
        )
        num_heads = (
            _infer_ftic_num_heads(state_dict, "g_a.stage1"),
            _infer_ftic_num_heads(state_dict, "g_a.stage2"),
            _infer_ftic_num_heads(state_dict, "g_a.stage3"),
            _infer_ftic_num_heads(state_dict, "g_s.stage1"),
            _infer_ftic_num_heads(state_dict, "g_s.stage2"),
            _infer_ftic_num_heads(state_dict, "g_s.stage3"),
        )
        net = cls(
            config=config,
            num_heads=num_heads,
            feature_dims=(dim0, dim1, dim2),
            hyper_hidden_channels=state_dict["h_a.input_block.conv1.weight"].size(0),
            hyper_channels=state_dict["entropy_bottleneck.quantiles"].size(0),
            M=M,
            num_slices=_infer_num_slices(state_dict, M),
            num_scales=state_dict["gaussian_conditional.scale_table"].numel()
            if "gaussian_conditional.scale_table" in state_dict
            else 256,
            num_means=state_dict["gaussian_conditional._prior_mean"].size(0)
            if state_dict.get(
                "gaussian_conditional._prior_mean",
                torch.empty(0),
            ).numel()
            > 0
            else 100,
            min_scale=float(state_dict["gaussian_conditional.scale_bound"].item()),
            window_size=_infer_window_size(state_dict, "g_a.stage1"),
            fm_window_size=_infer_fm_window_size(state_dict, "g_a.stage1"),
            hyper_window_size=_infer_window_size(state_dict, "h_a.stage"),
            hyper_fm_window_size=_infer_fm_window_size(state_dict, "h_a.stage"),
            hyper_num_heads=_infer_ftic_num_heads(state_dict, "h_a.stage"),
            tca_depth=_infer_tca_depth(state_dict),
            tca_ratio=_infer_tca_ratio(state_dict, M),
        )
        if (
            "gaussian_conditional._prior_mean" in state_dict
            and state_dict["gaussian_conditional._prior_mean"].numel() > 0
        ):
            update_registered_buffers(
                net.gaussian_conditional,
                "gaussian_conditional",
                ["_prior_mean", "_prior_scale"],
                state_dict,
            )
        # ``scale_table`` is recreated in the constructor from
        # ``(min_scale, scale_max, num_scales)``; upstream checkpoints don't
        # store it. Inject the freshly-built value so the base
        # ``CompressionModel.load_state_dict`` finds it during its mandatory
        # ``update_registered_buffers`` pass.
        state_dict = dict(state_dict)
        state_dict.setdefault(
            "gaussian_conditional.scale_table",
            net.gaussian_conditional.scale_table.clone(),
        )
        incompatible = net.load_state_dict(state_dict, strict=False)
        missing = set(incompatible.missing_keys)
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected incompatibility while loading FTIC state_dict: "
                f"missing={sorted(missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
        return net
