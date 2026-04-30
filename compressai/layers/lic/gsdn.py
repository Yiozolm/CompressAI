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

"""Generalized Subtractive and Divisive Normalization (GSDN).

GSDN extends GDN with a per-channel learnable subtractive (mean) term *before*
the divisive (variance) normalisation:

    y[i] = (x[i] − μ[i]) / sqrt(β[i] + Σ_j γ[j, i] · (x[j] − μ[j])²)

Used by the Reference-Based AR image-compression model (Qian et al.,
ICLR 2021). Only the divisive ``β`` / ``γ`` are reparametrised through
:class:`compressai.ops.parametrizers.NonNegativeParametrizer`; the subtractive
``β2`` / ``γ2`` follow the same reparametrisation but with a different
positivity floor on ``β2`` (zero by default upstream).

Module attribute names ``beta`` / ``gamma`` / ``beta2`` / ``gamma2`` and
init values are kept identical to upstream ``module/ops.py::GSDN`` so the
released checkpoints load 1:1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.ops.parametrizers import NonNegativeParametrizer

__all__ = ["GSDN"]


class GSDN(nn.Module):
    """Generalized Subtractive + Divisive Normalisation layer."""

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.inverse = bool(inverse)

        # Divisive (β, γ): identical to GDN.
        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

        # Subtractive (β2, γ2): upstream initialises β2 to zeros.
        self.beta2_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta2 = torch.zeros(in_channels)
        beta2 = self.beta2_reparam.init(beta2)
        self.beta2 = nn.Parameter(beta2)

        self.gamma2_reparam = NonNegativeParametrizer()
        gamma2 = gamma_init * torch.eye(in_channels)
        gamma2 = self.gamma2_reparam.init(gamma2)
        self.gamma2 = nn.Parameter(gamma2)

    def _norm_params(self, beta_p: nn.Parameter, gamma_p: nn.Parameter, beta_repar, gamma_repar, C: int):
        beta = beta_repar(beta_p)
        gamma = gamma_repar(gamma_p)
        gamma = gamma.reshape(C, C, 1, 1)
        return beta, gamma

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        if self.inverse:
            # Decoder side: inverse divisive (multiply) then add learned mean.
            beta, gamma = self._norm_params(
                self.beta, self.gamma, self.beta_reparam, self.gamma_reparam, C
            )
            norm = torch.sqrt(F.conv2d(x**2, gamma, beta))
            x = x * norm

            beta2, gamma2 = self._norm_params(
                self.beta2, self.gamma2, self.beta2_reparam, self.gamma2_reparam, C
            )
            mean = F.conv2d(x, gamma2, beta2)
            return x + mean

        # Encoder side: subtract learned mean then divisive normalise.
        beta2, gamma2 = self._norm_params(
            self.beta2, self.gamma2, self.beta2_reparam, self.gamma2_reparam, C
        )
        mean = F.conv2d(x, gamma2, beta2)
        x = x - mean

        beta, gamma = self._norm_params(
            self.beta, self.gamma, self.beta_reparam, self.gamma_reparam, C
        )
        norm = torch.rsqrt(F.conv2d(x**2, gamma, beta))
        return x * norm
