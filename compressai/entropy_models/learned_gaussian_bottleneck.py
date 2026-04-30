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

from typing import Any, List, Optional, Tuple

import numpy as np
import scipy.stats
import torch
import torch.nn as nn

from torch import Tensor

from compressai.ops import LowerBound

from .entropy_models import EntropyModel


class LearnedGaussianBottleneck(EntropyModel):
    r"""Per-channel learnable single-Gaussian factorized prior.

    Equivalent to the upstream ``module.prob_model.Entropy`` used by the
    Entroformer (ICLR 2022) and Reference-Based AR (ICLR 2021) image
    compression models, where the hyper-prior ``z`` is modelled as
    ``p(z_c) = Φ((z_c + 0.5 − μ_c) / σ_c) − Φ((z_c − 0.5 − μ_c) / σ_c)``
    with per-channel learnable ``μ_c`` and ``log σ_c`` (shape ``(1, C, 1, 1)``).

    This is mathematically distinct from
    :class:`compressai.entropy_models.EntropyBottleneck`, which learns a
    non-parametric factorized CDF via deep factorized matrices.
    """

    _offset: Tensor

    def __init__(
        self,
        channels: int,
        *args: Any,
        tail_mass: float = 1e-9,
        scale_bound: float = 1e-9,
        init_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.channels = int(channels)
        self.tail_mass = float(tail_mass)
        self.lower_bound_scale = LowerBound(float(scale_bound))

        # Per-channel learnable mean and log-stddev (broadcasting over (N, H, W)).
        # Shape matches upstream Entropy(channel) so converted state dicts load 1:1.
        self.mu = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.log_sigma = nn.Parameter(
            torch.full((1, channels, 1, 1), float(np.log(init_scale)))
        )

    def _scale(self) -> Tensor:
        return self.lower_bound_scale(self.log_sigma.exp())

    @staticmethod
    def _standardized_cumulative(inputs: Tensor) -> Tensor:
        # Same as GaussianConditional: half * erfc(-(2^-0.5) * x) for max precision
        const = float(-(2**-0.5))
        return 0.5 * torch.erfc(const * inputs)

    def _likelihood(self, inputs: Tensor) -> Tensor:
        scales = self._scale()
        values = (inputs - self.mu).abs()
        upper = self._standardized_cumulative((0.5 - values) / scales)
        lower = self._standardized_cumulative((-0.5 - values) / scales)
        return upper - lower

    def forward(
        self, x: Tensor, training: Optional[bool] = None
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        # Upstream NoiseQuant: z + U(-0.5, 0.5) at train, round(z + 0.5) at eval.
        # `quantize(..., means=None)` matches both modes (no mean shift, then likelihood
        # handles μ analytically).
        outputs = self.quantize(x, "noise" if training else "dequantize", means=None)
        likelihood = self._likelihood(outputs)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    @torch.no_grad()
    def update(self, force: bool = False) -> bool:
        # Skip update if CDFs are already cached and the caller did not ask for a refresh.
        if self._offset.numel() > 0 and not force:
            return False

        device = self.mu.device
        mu = self.mu.flatten().detach()  # (C,)
        scale = self._scale().flatten().detach()  # (C,)

        # Window each channel's PMF symmetrically around round(μ_c) with width
        # ⌈multiplier · σ_c⌉, where multiplier is the standard-normal quantile
        # that bounds tail mass at `tail_mass / 2` per side.
        multiplier = float(-scipy.stats.norm.ppf(self.tail_mass / 2))
        center = torch.round(mu).int()
        pmf_center = torch.ceil(scale * multiplier).int().clamp(min=1)
        pmf_length = 2 * pmf_center + 1
        max_length = int(pmf_length.max().item())

        # samples shape: (C, max_length); valid range is [center - pmf_center, center + pmf_center]
        offsets = torch.arange(max_length, device=device).int()
        samples = center.unsqueeze(1) + offsets.unsqueeze(0) - pmf_center.unsqueeze(1)
        samples = samples.float()

        values = (samples - mu.unsqueeze(1)).abs()
        scale_ = scale.unsqueeze(1)
        upper = self._standardized_cumulative((0.5 - values) / scale_)
        lower = self._standardized_cumulative((-0.5 - values) / scale_)
        pmf = upper - lower

        # Tail mass beyond the truncation window (on either side of μ).
        tail_left = self._standardized_cumulative(
            (-(pmf_center.float() - (center.float() - mu)) - 0.5) / scale
        )
        tail_right = self._standardized_cumulative(
            (-(pmf_center.float() + (center.float() - mu)) - 0.5) / scale
        )
        tail_mass = (tail_left + tail_right).unsqueeze(1)

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = center - pmf_center
        self._cdf_length = pmf_length + 2
        return True

    @staticmethod
    def _build_indexes(size: torch.Size) -> Tensor:
        dims = len(size)
        N = size[0]
        C = size[1]
        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims).int()
        return indexes.repeat(N, 1, *size[2:])

    def compress(self, x: Tensor) -> List[bytes]:
        indexes = self._build_indexes(x.size())
        return super().compress(x, indexes)

    def decompress(
        self,
        strings: List[bytes],
        size: Tuple[int, ...],
    ) -> Tensor:
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        return super().decompress(strings, indexes, dtype=torch.float32)
