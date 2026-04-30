# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Ballé-style 4-stage Conv + GDN encoder / decoder pair.

Used by both Entroformer (ICLR 2022) and the Reference-Based AR model
(ICLR 2021) as the synthesis / analysis transforms. The ``norm_cls`` argument
lets callers swap in GSDN for ref-AR; module attribute names (``encoder`` /
``decoder``, ``transpose``) are kept identical to upstream so converted
state dicts load 1:1.
"""
from __future__ import annotations

from typing import Type

import torch.nn as nn

from torch import Tensor

from compressai.layers.gdn import GDN

__all__ = [
    "Balle2Encoder",
    "Balle2Decoder",
    "Balle2Upsample",
]


class Balle2Upsample(nn.Module):
    """``ConvTranspose2d`` wrapper preserving the upstream ``transpose`` attribute name."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        output_padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            padding_mode="zeros",
            groups=groups,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transpose(x)


class Balle2Encoder(nn.Module):
    """4-stage 5×5 stride-2 Conv + GDN analysis transform."""

    def __init__(
        self,
        channels: int = 192,
        last_channels: int = 384,
        norm_cls: Type[nn.Module] = GDN,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 5, stride=2, padding=2, padding_mode="zeros"),
            norm_cls(channels),
            nn.Conv2d(channels, channels, 5, stride=2, padding=2, padding_mode="zeros"),
            norm_cls(channels),
            nn.Conv2d(channels, channels, 5, stride=2, padding=2, padding_mode="zeros"),
            norm_cls(channels),
            nn.Conv2d(channels, last_channels, 5, stride=2, padding=2, padding_mode="zeros"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class Balle2Decoder(nn.Module):
    """4-stage 5×5 stride-2 ConvTranspose + inverse-GDN synthesis transform."""

    def __init__(
        self,
        channels: int = 192,
        last_channels: int = 384,
        norm_cls: Type[nn.Module] = GDN,
    ) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            Balle2Upsample(last_channels, channels, 5, stride=2, padding=2, output_padding=1),
            norm_cls(channels, inverse=True),
            Balle2Upsample(channels, channels, 5, stride=2, padding=2, output_padding=1),
            norm_cls(channels, inverse=True),
            Balle2Upsample(channels, channels, 5, stride=2, padding=2, output_padding=1),
            norm_cls(channels, inverse=True),
            Balle2Upsample(channels, 3, 5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)
