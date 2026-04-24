from __future__ import annotations

import torch
import torch.nn as nn

from torch import Tensor

from compressai.layers import GDN, conv1x1, conv3x3, subpel_conv3x3

from .wavelet import DWT2D, IDWT2D

__all__ = [
    "WaveletResidualBlockUpsample",
    "WaveletResidualBlockWithStride",
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
        self.skip = conv1x1(in_ch, out_ch, stride=stride) if stride != 1 or in_ch != out_ch else None

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
