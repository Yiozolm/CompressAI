# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# This file adapts the Stochastic Gumbel Annealing (SGA) quantizer from the
# original TensorFlow implementation at
# https://github.com/mandt-lab/improving-inference-for-neural-image-compression
# (paper: Yang, Bamler, Mandt, "Improving Inference for Neural Image
# Compression", NeurIPS 2020). The PyTorch port at
# https://github.com/tongdaxu/pytorch-improving-inference-for-neural-image-compression
# was used as reference. Modifications by InterDigital Communications, Inc. are
# released under the BSD 3-Clause Clear License terms below.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

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

"""Stochastic Gumbel Annealing (SGA) quantizer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["SGAQuantizer"]


class SGAQuantizer(nn.Module):
    """Stochastic Gumbel Annealing quantizer.

    This module samples between ``floor(x)`` and ``ceil(x)`` using the relaxed
    Gumbel-softmax distribution from Yang, Bamler, and Mandt, "Improving
    Inference for Neural Image Compression", NeurIPS 2020.

    Call :meth:`set_iter` before each refinement step to enable relaxed
    sampling. With the iteration state unset, the module falls back to
    ``torch.round``.
    """

    def __init__(
        self,
        annealing_rate: float = 1e-3,
        upper_temperature: float = 0.5,
        lower_temperature: float = 1e-8,
        warmup: float = 700.0,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.annealing_rate = float(annealing_rate)
        self.upper_temperature = float(upper_temperature)
        self.lower_temperature = float(lower_temperature)
        self.warmup = float(warmup)
        self.epsilon = float(epsilon)
        self._iter: Optional[int] = None

    def set_iter(self, it: Optional[int], total_iter: Optional[int] = None) -> None:
        """Set the current SGA iteration.

        ``total_iter`` is accepted for compatibility with refinement loops that
        pass both values, but the default schedule follows the official TF2
        implementation and uses an absolute ``warmup`` value.
        """
        self._iter = None if it is None else int(it)

    def annealed_temperature(self, it: int) -> float:
        tau = (
            self.upper_temperature
            * torch.exp(torch.tensor(-self.annealing_rate * (it - self.warmup))).item()
        )
        return min(max(tau, self.lower_temperature), self.upper_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._iter is None:
            return torch.round(x)

        x_floor = torch.floor(x)
        x_ceil = torch.ceil(x)
        x_bds = torch.stack([x_floor, x_ceil], dim=-1)

        eps = self.epsilon
        tau = self.annealed_temperature(self._iter)
        logits = torch.stack(
            [
                -torch.atanh(torch.clamp(x - x_floor, -1 + eps, 1 - eps)) / tau,
                -torch.atanh(torch.clamp(x_ceil - x, -1 + eps, 1 - eps)) / tau,
            ],
            dim=-1,
        )
        sample = torch.distributions.RelaxedOneHotCategorical(
            tau,
            logits=logits,
        ).rsample()
        return torch.sum(x_bds * sample, dim=-1)
