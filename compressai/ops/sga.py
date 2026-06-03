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

"""Stochastic Gumbel Annealing (SGA) quantizer.

SGA replaces hard ``round`` in the inference path of a neural image codec
with a relaxed Gumbel-softmax sample between ``floor`` and ``ceil`` of the
input. The temperature is annealed during a per-image optimization loop,
starting near ``T=0.5`` and decaying toward zero so the sample converges to
the hard rounding result.

Used by the MLICv2+ inference-time refinement (latent + side-information
re-optimization), but applicable to any LIC entropy stack.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SGAQuantizer"]


class SGAQuantizer(nn.Module):
    """Stochastic Gumbel Annealing quantizer.

    A stateful nn.Module that, when ``set_iter(it, total_iter)`` has been
    called, returns a relaxed Gumbel-softmax sample between
    ``floor(x) / ceil(x)`` with an annealed temperature
    ``T = ub * exp(-r * (it - t0))``. When the iteration state is unset (the
    default after construction or after ``set_iter(None, None)``) the module
    falls back to plain ``torch.round``.

    Reference:
        Yang, Bamler, Mandt. "Improving Inference for Neural Image
        Compression", NeurIPS 2020. https://arxiv.org/abs/2006.04240

    Args:
        annealing_rate: exponential decay rate ``r`` (default ``1e-3``).
        upper_temperature: maximum temperature ``T_ub`` reached at iteration
            ``t0`` (default ``0.5``).
        warmup_fraction: ``t0`` is computed as
            ``int(total_iter * warmup_fraction)``, so the schedule scales with
            the loop length (default ``0.35``).
        lower_temperature: floor to avoid numerical issues as ``T → 0``
            (default ``1e-8``).

    Example:
        >>> sga = SGAQuantizer()
        >>> sga.set_iter(0, 2000)              # before each refine step
        >>> y_tilde = sga(y - means)           # relaxed quantization
        >>> sga.set_iter(None, None)           # fall back to hard round
        >>> y_round = sga(y - means)
    """

    def __init__(
        self,
        annealing_rate: float = 1e-3,
        upper_temperature: float = 0.5,
        warmup_fraction: float = 0.35,
        lower_temperature: float = 1e-8,
    ) -> None:
        super().__init__()
        self.annealing_rate = float(annealing_rate)
        self.upper_temperature = float(upper_temperature)
        self.warmup_fraction = float(warmup_fraction)
        self.lower_temperature = float(lower_temperature)
        self._iter: Optional[int] = None
        self._total_iter: Optional[int] = None

    def set_iter(self, it: Optional[int], total_iter: Optional[int]) -> None:
        """Set the current iteration state for the next forward pass.

        Pass ``(None, None)`` to fall back to hard rounding.
        """
        self._iter = None if it is None else int(it)
        self._total_iter = None if total_iter is None else int(total_iter)

    def annealed_temperature(self, it: int, total_iter: int) -> float:
        t0 = int(total_iter * self.warmup_fraction)
        tau = self.upper_temperature * float(
            torch.exp(torch.tensor(-self.annealing_rate * (it - t0))).item()
        )
        return min(max(tau, self.lower_temperature), self.upper_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._iter is None or self._total_iter is None:
            return torch.round(x)

        x_floor = torch.floor(x)
        x_ceil = torch.ceil(x)
        x_bds = torch.stack([x_floor, x_ceil], dim=-1)

        eps = 1e-5
        T = self.annealed_temperature(self._iter, self._total_iter)

        x_interval1 = torch.clamp(x - x_floor, -1 + eps, 1 - eps)
        x_atanh1 = torch.log((1 + x_interval1) / (1 - x_interval1)) / 2
        x_interval2 = torch.clamp(x_ceil - x, -1 + eps, 1 - eps)
        x_atanh2 = torch.log((1 + x_interval2) / (1 - x_interval2)) / 2

        rx_logits = torch.stack([-x_atanh1 / T, -x_atanh2 / T], dim=-1)
        rx = F.softmax(rx_logits, dim=-1)
        rx_dist = torch.distributions.RelaxedOneHotCategorical(T, rx)
        rx_sample = rx_dist.rsample()

        return torch.sum(x_bds * rx_sample, dim=-1)
