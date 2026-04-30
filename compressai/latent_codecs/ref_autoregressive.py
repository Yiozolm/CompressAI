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

"""Reference-based 3-cascade auto-regressive entropy codec.

Used by the Reference-Based AR image-compression model
(`Qian et al., ICLR 2021 <https://arxiv.org/abs/2010.08321>`_).

Structure (matching upstream ``module/autoregressive_model.RefAutoRegressiveModel``):

* ``mask_conv`` (5×5 mask-A) → local context feature ``2M`` channels;
* a first 1×1 conv head (``conv_1x1_1``) that produces a single-mixture
  Gaussian (``num_p · M`` channels) used as the ``y_prob1`` input of
  :class:`compressai.layers.lic.SearchTransfer`;
* the search returns a per-pixel similarity ``S``, uncertainty ``U`` and a
  gathered reference patch tensor;
* ``mask_conv_ref`` (3×3 unfolded mask) → reference context feature ``2M``;
* three 1×1 cascade heads (``conv_1x1_1`` / ``_2`` / ``_3``) produce one
  GMM component each (``[local], [local, ref], [local, ref, hyper]``); the
  three are concatenated along the mixture dimension to form a ``K=3`` GMM;
* the second cascade adds ``log(S) + log(U)`` to its mixture-weight logit so
  similar / confident references contribute more (the upstream
  log-sum-exp "soft attention" trick).

The likelihood is the per-channel weighted sum of three discretised
Gaussians, evaluated by
:class:`compressai.entropy_models.GaussianMixtureConditional` (already in
the library, ``K=3`` interprets ``scales/means/weights`` as
``[K0·M, K1·M, K2·M]`` along the channel axis).
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import (
    GaussianMixtureConditional,
    LearnedGaussianBottleneck,
)
from compressai.layers import MaskedConv2d
from compressai.layers.lic import Conv2dUnfold, SearchTransfer
from compressai.ops import quantize_ste
from compressai.registry import register_module

from .base import LatentCodec

__all__ = ["RefAutoregressiveLatentCodec"]


def _make_param_head(in_ch: int, hidden_ch: int, out_ch: int, bias: bool = True) -> nn.Sequential:
    """Three-conv 1×1 cascade head; matches upstream ``conv_1x1_{1,2,3}``."""
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden_ch, 1, 1, 0, bias=bias),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hidden_ch, hidden_ch, 1, 1, 0, bias=bias),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hidden_ch, out_ch, 1, 1, 0, bias=bias),
    )


def _gauss_cdf(x: Tensor) -> Tensor:
    """``Φ(x)`` via the complementary error function (numerically stable)."""
    const = float(-(2 ** -0.5))
    return 0.5 * torch.erfc(const * x)


@register_module("RefAutoregressiveLatentCodec")
class RefAutoregressiveLatentCodec(LatentCodec):
    """Local context + global reference + hyperprior 3-cascade GMM codec.

    Args:
        latent_channels: Number of channels in the ``y`` latent (``M``).
        hyper_channels: Number of channels in the ``z`` latent.
        head_channels: Hidden channel count inside the three 1×1 cascade
            heads (upstream ``channels=last_channels*3``; default ``3 * M``).
        sk: Reference patch size used by :class:`SearchTransfer` and
            :class:`Conv2dUnfold` (default 3, matches upstream).
        num_parameter: Number of distribution parameters per pixel; only
            ``3`` (``logit_pi`` + ``mean`` + ``log-σ``) is supported.
        bias: Whether the 1×1 cascade convs use bias (default True).
        log_scale_min: Lower clamp on predicted ``log σ``. Upstream sets this
            to ``-7`` (≈ σ ≥ 9e-4).
        gaussian_mixture_conditional: Optional pre-built GMM head (``K=3``).
            A new one is created with ``scale_bound = exp(log_scale_min)`` if
            omitted, so the forward likelihood matches upstream exactly.
        entropy_bottleneck: Optional pre-built ``z`` entropy model. Defaults
            to :class:`LearnedGaussianBottleneck(hyper_channels)`.
    """

    def __init__(
        self,
        *,
        latent_channels: int,
        hyper_channels: int,
        head_channels: int | None = None,
        sk: int = 3,
        num_parameter: int = 3,
        bias: bool = True,
        log_scale_min: float = -7.0,
        gaussian_mixture_conditional: GaussianMixtureConditional | None = None,
        entropy_bottleneck: LearnedGaussianBottleneck | None = None,
    ) -> None:
        super().__init__()
        if num_parameter != 3:
            raise ValueError(
                "RefAutoregressiveLatentCodec requires num_parameter=3 "
                "(logit_pi + mean + log-σ); upstream releases use only 3."
            )

        M = int(latent_channels)
        Z = int(hyper_channels)
        Hd = int(head_channels if head_channels is not None else 3 * M)
        self.latent_channels = M
        self.hyper_channels = Z
        self.head_channels = Hd
        self.num_parameter = int(num_parameter)
        self.log_scale_min = float(log_scale_min)
        self._mixtures = 3  # local, ref, hyper

        # Local context: 5×5 mask-A conv, M → 2M.
        self.mask_conv = MaskedConv2d(M, 2 * M, 5, 1, 2, bias=bias, mask_type="A")
        # Global reference search and masked-conv on unfolded refs.
        self.search = SearchTransfer(M, k=sk)
        self.mask_conv_ref = Conv2dUnfold(True, M, 2 * M, sk, 1, sk // 2, bias=bias)

        out_ch = M * num_parameter  # one GMM component per cascade
        # Cascade 1: local only (input = 2M).
        self.conv_1x1_1 = _make_param_head(2 * M, Hd, out_ch, bias=bias)
        # Cascade 2: local + ref (input = 4M).
        self.conv_1x1_2 = _make_param_head(2 * M + 2 * M, Hd, out_ch, bias=bias)
        # Cascade 3: local + ref + hyper (input = 4M + chyper, upstream chyper=2M).
        self.conv_1x1_3 = _make_param_head(2 * M + 2 * M + 2 * M, Hd, out_ch, bias=bias)

        if gaussian_mixture_conditional is None:
            import math

            gaussian_mixture_conditional = GaussianMixtureConditional(
                K=self._mixtures,
                scale_table=None,
                scale_bound=float(math.exp(log_scale_min)),
            )
        self.gaussian_mixture_conditional = gaussian_mixture_conditional
        self.entropy_bottleneck = entropy_bottleneck or LearnedGaussianBottleneck(Z)

    # ------------------------------------------------------------------ helpers

    def _local_only_prob(self, y: Tensor, para1: Tensor) -> Tensor:
        """Per-pixel mean-of-channels probability under the *local-only* head.

        Used as the ``y_prob1`` input to :class:`SearchTransfer` so the search
        can weight references by the model's existing (local-only) confidence.
        """
        n, _, h, w = y.shape
        M = self.latent_channels
        # para1 layout: (N, num_p*C, H, W) interpreted as (N, num_p, C, K=1, H, W)
        params = para1.reshape(n, self.num_parameter, M, 1, h, w)
        means = params[:, 1, :, 0, :, :]
        log_scales = params[:, 2, :, 0, :, :].clamp(min=self.log_scale_min)
        scales = log_scales.exp()
        values = (y - means).abs()
        upper = _gauss_cdf((0.5 - values) / scales)
        lower = _gauss_cdf((-0.5 - values) / scales)
        prob = (upper - lower).clamp(min=1e-12)
        return prob.mean(dim=1, keepdim=True)

    def _split_to_mixture(
        self, paras: tuple[Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Stack three single-mixture cascades into ``K=3`` mixture inputs.

        Upstream layout (after `cat([para_i.reshape(N, num_p, C, 1, H, W) for i in 1..3], dim=3)`)
        is ``(N, num_p, C, K=3, H, W)`` with ``[p, c, k]`` outer-to-inner. Compressai's
        :class:`GaussianMixtureConditional` expects ``scales / means / weights`` of shape
        ``(N, K * C, H, W)`` with channel layout ``[K0·C, K1·C, K2·C]``, i.e. ``[k, c]``.
        We therefore transpose ``c`` and ``k`` after stacking.

        The second cascade additionally has ``log(S) + log(U)`` added to its
        mixture-weight (logit) channel — this reproduces the upstream
        log-sum-exp soft-attention trick.
        """
        n, _, h, w = paras[0].shape
        M = self.latent_channels
        K = self._mixtures
        # Stack along K axis: (N, num_p, K, C, H, W)
        stacked = torch.stack(
            [p.reshape(n, self.num_parameter, M, h, w) for p in paras], dim=2
        )
        logit_pi = stacked[:, 0]  # (N, K, C, H, W)
        means = stacked[:, 1]
        log_scales = stacked[:, 2]
        return logit_pi, means, log_scales

    # ------------------------------------------------------------------ forward

    def forward(self, y: Tensor, z_feature: Tensor) -> Dict[str, Any]:
        # Upstream noise-quant for AR input + likelihood, STE for downstream g_s.
        y_for_ar = self.gaussian_mixture_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )

        n, _, h, w = y.shape
        M = self.latent_channels

        # --- search prep (uses a local-only `para1` to bootstrap probabilities)
        local = self.mask_conv(y_for_ar)
        para1_for_search = self.conv_1x1_1(local.detach())
        y_prob1 = self._local_only_prob(y_for_ar, para1_for_search.clone())
        S, U, ref_unfold, _ = self.search(y_for_ar.detach(), y_prob1.detach())
        ref = self.mask_conv_ref(ref_unfold, h, w)

        # --- 3 cascade heads
        para1 = self.conv_1x1_1(local)
        para2 = self.conv_1x1_2(torch.cat([local, ref], dim=1))
        para3 = self.conv_1x1_3(torch.cat([local, ref, z_feature], dim=1))

        # --- merge to K=3 GMM
        logit_pi, means, log_scales = self._split_to_mixture((para1, para2, para3))
        # Soft attention via log-sum-exp on cascade-2 mixture weight.
        log_su = ((S + 1e-8).log() + (U + 1e-8).log()).expand_as(logit_pi[:, 1])
        logit_pi = logit_pi.clone()
        logit_pi[:, 1] = logit_pi[:, 1] + log_su

        # softmax over K to get mixture weights
        weights = torch.softmax(logit_pi, dim=1)
        scales = log_scales.clamp(min=self.log_scale_min).exp()

        # Compressai GMM expects (N, K*M, H, W) layout `[k, m]`; reshape directly.
        K = self._mixtures
        weights_flat = weights.reshape(n, K * M, h, w)
        means_flat = means.reshape(n, K * M, h, w)
        scales_flat = scales.reshape(n, K * M, h, w)

        y_lik = self.gaussian_mixture_conditional._likelihood(
            y_for_ar, scales_flat, means_flat, weights_flat
        )
        if self.gaussian_mixture_conditional.use_likelihood_bound:
            y_lik = self.gaussian_mixture_conditional.likelihood_lower_bound(y_lik)

        y_hat = quantize_ste(y) if self.training else y_for_ar
        return {
            "y_hat": y_hat,
            "likelihoods": {"y": y_lik},
        }

    # --- compress / decompress: per-pixel raster-scan AR, planned follow-up.
    def compress(self, y: Tensor, z_feature: Tensor) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError(
            "RefAutoregressiveLatentCodec.compress is not implemented yet; the "
            "upstream raster-scan AR bitstream coding requires per-pixel "
            "re-evaluation of the search/cascade heads. Use forward() for rate "
            "estimation."
        )

    def decompress(  # pragma: no cover
        self, strings, shape, z_feature: Tensor, *args, **kwargs
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "RefAutoregressiveLatentCodec.decompress is not implemented yet; "
            "see compress()."
        )
