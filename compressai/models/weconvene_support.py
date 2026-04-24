from __future__ import annotations

import math

import torch.nn as nn

from torch import Tensor

from compressai.layers import (
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.layers.wave import (
    WaveletResidualBlockUpsample,
    WaveletResidualBlockWithStride,
)

__all__ = [
    "WeConveneAnalysisTransform",
    "WeConveneHyperAnalysisTransform",
    "WeConveneHyperSynthesisTransform",
    "WeConveneSynthesisTransform",
    "infer_max_support_slices",
    "infer_num_slices",
    "infer_support_attention",
    "lrp_channels",
    "make_entropy_transform",
    "support_channels",
]


def support_channels(
    latent_channels: int,
    slice_channels: int,
    index: int,
    max_support_slices: int,
) -> int:
    if max_support_slices < 0:
        return latent_channels + slice_channels * index
    return latent_channels + slice_channels * min(index, max_support_slices)


def lrp_channels(
    latent_channels: int,
    slice_channels: int,
    index: int,
    max_support_slices: int,
) -> int:
    if max_support_slices < 0:
        return latent_channels + slice_channels * (index + 1)
    return latent_channels + slice_channels * min(index + 1, max_support_slices + 1)


def make_entropy_transform(in_channels: int, out_channels: int) -> nn.Sequential:
    from .utils import conv

    return nn.Sequential(
        conv(in_channels, 224, stride=1, kernel_size=3),
        nn.GELU(),
        conv(224, 128, stride=1, kernel_size=3),
        nn.GELU(),
        conv(128, out_channels, stride=1, kernel_size=3),
    )


def infer_num_slices(state_dict: dict[str, Tensor]) -> int:
    indices = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("latent_codec.cc_mean_transforms_low.") and key.endswith(".0.weight")
    }
    if not indices:
        raise KeyError("Unable to infer num_slices from state_dict")
    return max(indices) + 1


def infer_max_support_slices(
    state_dict: dict[str, Tensor],
    M: int,
    num_slices: int,
) -> int:
    slice_channels = M // num_slices
    last_index = num_slices - 1
    key = f"latent_codec.mean_support_transforms_low.{last_index}.in_conv.weight"
    if key not in state_dict:
        return last_index
    input_channels = state_dict[key].size(1)
    return max(0, (input_channels - M) // slice_channels)


def infer_support_attention(state_dict: dict[str, Tensor]) -> tuple[int, int, int]:
    in_conv_key = "latent_codec.mean_support_transforms_low.0.in_conv.weight"
    table_key = (
        "latent_codec.mean_support_transforms_low.0."
        "non_local_block.block_1.msa.attn.relative_position_bias_table"
    )
    if in_conv_key not in state_dict or table_key not in state_dict:
        return 8, 16, 128

    hidden_dim = state_dict[in_conv_key].size(0)
    table_size, num_heads = state_dict[table_key].shape
    window_size = (math.isqrt(table_size) + 1) // 2
    head_dim = hidden_dim // num_heads
    return window_size, head_dim, hidden_dim


def _make_residual_stage(
    channels: int,
    residual_blocks: int,
    tail: nn.Module,
) -> nn.Sequential:
    return nn.Sequential(
        *(ResidualBlock(channels, channels) for _ in range(residual_blocks)),
        tail,
    )


class WeConveneAnalysisTransform(nn.Module):
    def __init__(
        self,
        N: int,
        M: int,
        *,
        residual_blocks: int = 3,
        wavelet: str = "haar",
    ) -> None:
        super().__init__()
        self.input_block = ResidualBlockWithStride(3, N, stride=1)
        self.down1 = _make_residual_stage(
            N,
            residual_blocks,
            WaveletResidualBlockWithStride(N, N, stride=2, wavelet=wavelet),
        )
        self.down2 = _make_residual_stage(
            N,
            residual_blocks,
            WaveletResidualBlockWithStride(N, N, stride=2, wavelet=wavelet),
        )
        self.down3 = _make_residual_stage(
            N,
            residual_blocks,
            conv3x3(N, M, stride=2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.input_block(input_tensor)
        output = self.down1(output)
        output = self.down2(output)
        return self.down3(output)


class WeConveneSynthesisTransform(nn.Module):
    def __init__(
        self,
        N: int,
        M: int,
        *,
        residual_blocks: int = 3,
        wavelet: str = "haar",
    ) -> None:
        super().__init__()
        self.input_block = ResidualBlockUpsample(M, N, upsample=1)
        self.up1 = _make_residual_stage(
            N,
            residual_blocks,
            WaveletResidualBlockUpsample(N, N, upsample=2, wavelet=wavelet),
        )
        self.up2 = _make_residual_stage(
            N,
            residual_blocks,
            WaveletResidualBlockUpsample(N, N, upsample=2, wavelet=wavelet),
        )
        self.up3 = _make_residual_stage(
            N,
            residual_blocks,
            subpel_conv3x3(N, 3, 2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.input_block(input_tensor)
        output = self.up1(output)
        output = self.up2(output)
        return self.up3(output)


class WeConveneHyperAnalysisTransform(nn.Module):
    def __init__(
        self,
        N: int,
        M: int,
        hyper_channels: int,
        *,
        residual_blocks: int = 3,
        wavelet: str = "haar",
    ) -> None:
        super().__init__()
        self.wavelet_block = WaveletResidualBlockWithStride(
            4 * M,
            N,
            stride=2,
            wavelet=wavelet,
        )
        self.down = _make_residual_stage(
            N,
            residual_blocks,
            conv3x3(N, hyper_channels, stride=2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.wavelet_block(input_tensor)
        return self.down(output)


class WeConveneHyperSynthesisTransform(nn.Module):
    def __init__(
        self,
        N: int,
        M: int,
        hyper_channels: int,
        *,
        residual_blocks: int = 3,
        wavelet: str = "haar",
    ) -> None:
        super().__init__()
        self.wavelet_block = WaveletResidualBlockUpsample(
            hyper_channels,
            N,
            upsample=2,
            wavelet=wavelet,
        )
        self.up = _make_residual_stage(
            N,
            residual_blocks,
            subpel_conv3x3(N, M, 2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.wavelet_block(input_tensor)
        return self.up(output)
