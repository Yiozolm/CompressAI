from __future__ import annotations

import torch
import torch.nn as nn

from torch import Tensor

from compressai.layers import (
    GDN,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv1x1,
    conv3x3,
    subpel_conv3x3,
)

from .wavelet import DWT2D, IDWT2D

__all__ = [
    "WaveletResidualBlockUpsample",
    "WaveletResidualBlockWithStride",
    "WeConveneAnalysisTransform",
    "WeConveneHyperAnalysisTransform",
    "WeConveneHyperSynthesisTransform",
    "WeConveneSynthesisTransform",
]


class WaveletResidualBlockWithStride(nn.Module):
    """Residual downsampling block with wavelet-domain subband processing."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 2,
        wavelet: str = "haar",
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.gdn_low = GDN(out_ch)
        self.gdn_high = GDN(3 * out_ch)
        self.dwt = DWT2D(wave=wavelet)
        self.idwt = IDWT2D(wave=wavelet)
        self.low_freq_conv = conv3x3(out_ch, out_ch)
        self.high_freq_conv = conv3x3(3 * out_ch, 3 * out_ch)
        self.skip = (
            conv1x1(in_ch, out_ch, stride=stride)
            if stride != 1 or in_ch != out_ch
            else None
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        identity = self.skip(input_tensor) if self.skip is not None else input_tensor
        output = self.leaky_relu(self.conv1(input_tensor))
        wavelet_tensor = self.dwt(output)
        low_freq, high_freq = wavelet_tensor.split(
            (output.size(1), output.size(1) * 3),
            dim=1,
        )
        low_freq = self.gdn_low(self.low_freq_conv(low_freq))
        high_freq = self.gdn_high(self.high_freq_conv(high_freq))
        return self.idwt(torch.cat([low_freq, high_freq], dim=1)) + identity


class WaveletResidualBlockUpsample(nn.Module):
    """Residual upsampling block with wavelet-domain subband processing."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        upsample: int = 2,
        wavelet: str = "haar",
    ) -> None:
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.igdn_low = GDN(out_ch, inverse=True)
        self.igdn_high = GDN(3 * out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)
        self.dwt = DWT2D(wave=wavelet)
        self.idwt = IDWT2D(wave=wavelet)
        self.low_freq_conv = conv3x3(out_ch, out_ch)
        self.high_freq_conv = conv3x3(3 * out_ch, 3 * out_ch)

    def forward(self, input_tensor: Tensor) -> Tensor:
        identity = self.upsample(input_tensor)
        output = self.leaky_relu(self.subpel_conv(input_tensor))
        wavelet_tensor = self.dwt(output)
        low_freq, high_freq = wavelet_tensor.split(
            (output.size(1), output.size(1) * 3),
            dim=1,
        )
        low_freq = self.igdn_low(self.low_freq_conv(low_freq))
        high_freq = self.igdn_high(self.high_freq_conv(high_freq))
        return self.idwt(torch.cat([low_freq, high_freq], dim=1)) + identity


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
