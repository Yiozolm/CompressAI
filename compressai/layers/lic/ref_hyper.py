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

"""Hyperprior encoder/decoder for the Reference-Based AR model.

Three-conv chain with LeakyReLU(0.2). Module attribute names ``encoder`` /
``decoder`` and their indexed children match
``candidate_none/img-comp-reference/module/dml_model.py`` so converted
checkpoints load 1:1.
"""
from __future__ import annotations

import torch.nn as nn

from torch import Tensor

from .balle2 import Balle2Upsample

__all__ = ["RefHyperEncoder", "RefHyperDecoder"]


class RefHyperEncoder(nn.Module):
    """``y → z`` analysis: 3×3 + 5×5 stride-2 + 5×5 stride-2 (3-stage)."""

    def __init__(
        self, in_channel: int = 384, out_channel: int = 192, channel: int = 192
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, channel, 3, stride=1, padding=1, padding_mode="zeros"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, channel, 5, stride=2, padding=2, padding_mode="zeros"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, out_channel, 5, stride=2, padding=2, padding_mode="zeros"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class RefHyperDecoder(nn.Module):
    """``z_hat → z_feature`` synthesis: 5×5 deconv ×2 + 3×3 conv + 1×1 head."""

    def __init__(
        self, in_channel: int = 192, out_channel: int = 768, channel: int = 192
    ) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            Balle2Upsample(in_channel, channel, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Balle2Upsample(channel, channel, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, padding_mode="zeros"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, out_channel, 1, stride=1, padding=0, padding_mode="zeros"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)
