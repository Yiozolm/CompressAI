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

"""Transformer-based hyperprior + autoregressive entropy codec used by Entroformer.

Wraps the hyperprior encoder/decoder, the AR Transformer and the parameter
network so that the model layer (:class:`compressai.models.Entroformer`) only
hands over the latent ``y``. The hyperprior bottleneck is the per-channel
:class:`compressai.entropy_models.LearnedGaussianBottleneck`.

The released Entroformer checkpoint uses ``num_parameter=2`` (mean + log-σ,
single Gaussian, ``K=1``); ``num_parameter=3`` is supported for forward only by
treating the extra channel as the (unused) mixture logit so converted
checkpoints with that head still load.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import GaussianConditional, LearnedGaussianBottleneck
from compressai.ops import quantize_ste
from compressai.registry import register_module

from .base import LatentCodec

__all__ = [
    "TransformerARLatentCodec",
]


@register_module("TransformerARLatentCodec")
class TransformerARLatentCodec(LatentCodec):
    """Hyperprior + Transformer AR + parameter network.

    Args:
        latent_channels: Number of channels in the ``y`` latent (``M``).
        hyper_channels: Number of channels in the ``z`` latent.
        y_hyper_encode: Module mapping ``y`` (B, M, H, W) to ``z`` (B, Z, H/2^scale, W/2^scale).
        y_hyper_decode: Module mapping ``z_hat`` to a transformer feature
            tensor of shape ``(B, dim_embed, H, W)``.
        y_ar: Causal Transformer producing the AR feature ``(B, dim_embed, H, W)``
            from the noise-quantised ``y_noise``.
        param_net: ``nn.Sequential`` that maps ``cat([feat_hyper, feat_ar], dim=1)``
            to ``(B, num_parameter * M * K, H, W)``.
        num_parameter: ``2`` (mean + log-σ) or ``3`` (logit_pi + mean + log-σ).
            Only ``num_parameter=2`` is fully supported; with ``num_parameter=3``
            and ``K=1`` the leading logit channel is discarded.
        log_scale_min: Lower clamp on predicted log-σ. Upstream sets this to
            ``-7`` (≈ σ ≥ 9e-4).
    """

    def __init__(
        self,
        *,
        latent_channels: int,
        hyper_channels: int,
        y_hyper_encode: nn.Module,
        y_hyper_decode: nn.Module,
        y_ar: nn.Module,
        param_net: nn.Module,
        gaussian_conditional: GaussianConditional | None = None,
        entropy_bottleneck: LearnedGaussianBottleneck | None = None,
        num_parameter: int = 2,
        log_scale_min: float = -7.0,
    ) -> None:
        super().__init__()
        if num_parameter not in (2, 3):
            raise ValueError(f"num_parameter must be 2 or 3, got {num_parameter}")

        self.latent_channels = int(latent_channels)
        self.hyper_channels = int(hyper_channels)
        self.num_parameter = int(num_parameter)
        self.log_scale_min = float(log_scale_min)

        self.y_hyper_encode = y_hyper_encode
        self.y_hyper_decode = y_hyper_decode
        self.y_ar = y_ar
        self.param_net = param_net

        # Use a tiny scale_bound so forward likelihood matches upstream
        # `log_scales.clamp(min=-7).exp()` exactly (default 0.11 would clip).
        self.gaussian_conditional = gaussian_conditional or GaussianConditional(
            None, scale_bound=float(torch.exp(torch.tensor(log_scale_min)).item())
        )
        self.entropy_bottleneck = entropy_bottleneck or LearnedGaussianBottleneck(
            hyper_channels
        )

    def _split_params(self, params: Tensor) -> tuple[Tensor, Tensor]:
        """Reshape ``param_net`` output into (scales, means) tensors of shape ``(B, M, H, W)``.

        Layout matches upstream ``DiscretizedMixGaussLoss._extract_non_shared``:
        ``params.reshape(B, num_p, M, K=1, H, W)`` with ``[0]=log_scales``,
        ``[1]=means`` for ``num_p=2``; for ``num_p=3`` the leading entry is
        ``logit_pi`` (single mixture component → ignored).
        """
        B, _, H, W = params.shape
        params = params.reshape(B, self.num_parameter, self.latent_channels, 1, H, W)
        if self.num_parameter == 2:
            log_scales = params[:, 0, :, 0, :, :]
            means = params[:, 1, :, 0, :, :]
        else:  # num_parameter == 3, K=1
            # params[:, 0] is logit_pi; with K=1, softmax is identity (always 1.0).
            means = params[:, 1, :, 0, :, :]
            log_scales = params[:, 2, :, 0, :, :]
        log_scales = log_scales.clamp(min=self.log_scale_min)
        scales = log_scales.exp()
        return scales, means

    def forward(self, y: Tensor) -> Dict[str, Any]:
        # Hyperprior branch
        z = self.y_hyper_encode(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        feat_hyper = self.y_hyper_decode(z_hat)

        # Noise-quantised y feeds the AR transformer + likelihood; STE-quantised
        # y is what the synthesis (g_s) sees, matching upstream behaviour.
        y_for_ar = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        feat_ar = self.y_ar(y_for_ar)
        merged = torch.cat([feat_hyper, feat_ar], dim=1)
        params = self.param_net(merged)
        scales, means = self._split_params(params)

        # Recompute likelihood directly to keep the AR-input quantisation in sync.
        y_likelihoods = self.gaussian_conditional._likelihood(
            y_for_ar, scales, means
        )
        if self.gaussian_conditional.use_likelihood_bound:
            y_likelihoods = self.gaussian_conditional.likelihood_lower_bound(
                y_likelihoods
            )

        y_hat_ste = quantize_ste(y) if self.training else y_for_ar

        return {
            "y_hat": y_hat_ste,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # NOTE: compress / decompress for the unidirectional (raster-scan) AR
    # variant requires per-pixel re-evaluation of `y_ar`, which matches upstream
    # `main_trans_hyper_ar.compress` but is on the order of 10^4 forward passes
    # for a single 256×256 image. Left to a follow-up integration step; forward
    # + state_dict loading + numerical equivalence are sufficient for the
    # current migration verification.
    def compress(self, y: Tensor) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError(
            "TransformerARLatentCodec.compress is not implemented yet; "
            "raster-scan bitstream coding for Entroformer's AR transformer "
            "is a planned follow-up. Use forward() for rate estimation."
        )

    def decompress(  # pragma: no cover
        self, strings, shape, *args, **kwargs
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "TransformerARLatentCodec.decompress is not implemented yet; "
            "see compress() for the same restriction."
        )
