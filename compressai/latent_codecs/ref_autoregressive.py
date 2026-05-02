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

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.entropy_models import (
    GaussianMixtureConditional,
    LearnedGaussianBottleneck,
)
from compressai.layers import MaskedConv2d
from compressai.ops import quantize_ste
from compressai.registry import register_module

from .base import LatentCodec

__all__ = ["RefAutoregressiveLatentCodec"]


# ---------------------------------------------------------------------------
# Search + masked-conv-on-unfolded-refs (formerly compressai/layers/lic/ref_search.py)
# ---------------------------------------------------------------------------


class _Conv2dUnfold(nn.Conv2d):
    """Masked k x k convolution acting on already-unfolded references.

    The trainable ``weight`` and the ``mask`` buffer share the same shape as a
    standard :class:`nn.Conv2d`. The ``mask`` zeroes out the contribution of
    the centre and below-centre kernel positions (mask type "A"), so the
    masked conv only "sees" causal neighbours of each reference patch.
    """

    def __init__(self, mask: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Buffer name 'mask' matches upstream so checkpoint keys load 1:1.
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.is_mask = bool(mask)
        if mask:
            self.mask.fill_(1)
            self.mask[:, :, kH // 2, kW // 2 + 1 :] = 0
            self.mask[:, :, kH // 2 + 1 :] = 0

    def forward(self, x_unfold: Tensor, h: int, w: int) -> Tensor:  # type: ignore[override]
        if self.is_mask:
            self.weight.data *= self.mask
        out_unfold = (
            x_unfold.transpose(1, 2)
            .matmul(self.weight.view(self.weight.size(0), -1).t())
            .transpose(1, 2)
        )
        return F.fold(out_unfold, (h, w), (1, 1))

    def forward_origin(self, x: Tensor) -> Tensor:
        """Standard masked-conv forward (operates on dense feature maps)."""
        if self.is_mask:
            self.weight.data *= self.mask
        return super().forward(x)


class _SearchTransfer(nn.Module):
    """Causal global-reference search + transfer.

    For each spatial position the module finds the index of the most-similar
    already-decoded location based on cosine similarity over masked k x k
    patches, then gathers the corresponding patch and a per-position
    probability tensor. Returns ``(S, U, ref_unfold, R_arg)`` where:

    * ``S`` -- similarity score map ``(N, 1, H, W)``;
    * ``U`` -- gathered probability ``(N, 1, H, W)``;
    * ``ref_unfold`` -- gathered k x k patch unfolded ``(N, C * k * k, H * W)``,
      ready to be consumed by :class:`_Conv2dUnfold`;
    * ``R_arg`` -- argmax indices ``(N, H * W)``.
    """

    def __init__(self, channels: int, k: int = 3, split: int = 1) -> None:
        super().__init__()
        # Mask Type "A": zero out centre + lower-half so the search only
        # references causal neighbours of each reference patch.
        mask = torch.ones((channels // split, k, k))
        mask[:, k // 2, k // 2 :] = 0
        mask[:, k // 2 + 1 :, :] = 0
        mask_unfold = F.unfold(mask.unsqueeze(0), kernel_size=(k, k), padding=0)
        # Stored as a non-trainable Parameter (matches upstream key
        # `search.mask_unfold` so converted checkpoints load by name).
        self.mask_unfold = nn.Parameter(mask_unfold, requires_grad=False)
        self.k = int(k)
        self.split = int(split)

    def forward(
        self, y_hat: Tensor, y_prob: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        k = self.k
        n, c, h, w = y_hat.shape

        unfold = F.unfold(y_hat, kernel_size=(k, k), padding=k // 2) * self.mask_unfold
        unfold = F.normalize(unfold, dim=1)  # (N, C*k*k, H*W)
        unfold_T = unfold.permute(0, 2, 1)  # (N, H*W, C*k*k)
        R = torch.bmm(unfold_T, unfold)  # (N, H*W, H*W)

        # Training: bidirectional reference (drop diagonal). Eval: causal only.
        if self.training:
            R = torch.triu(R, diagonal=1) + torch.tril(R, diagonal=-1)
        else:
            R = torch.triu(R, diagonal=1)

        R_star, R_star_arg = torch.max(R, dim=1)  # (N, H*W)

        y_hat_unfold = F.unfold(y_hat, kernel_size=(k, k), padding=k // 2)
        ref_unfold = self._batch_index_select(y_hat_unfold, 2, R_star_arg)
        unfold_prob = F.unfold(y_prob, kernel_size=(1, 1), padding=0)
        U_unfold = self._batch_index_select(unfold_prob, 2, R_star_arg)

        S = R_star.view(n, 1, h, w)
        U = F.fold(U_unfold, output_size=(h, w), kernel_size=(1, 1), padding=0)

        # First pixel has no causal reference; tag as zero/identity.
        if not self.training:
            S[:, :, 0, 0] = 1e-8
            U[:, :, 0, 0] = 1e-8
            ref_unfold[:, :, 0] = 0.0
            R_star_arg[:, 0] = -1

        S = torch.clamp(S, min=1e-8, max=1.0)
        U = torch.clamp(U, min=1e-8, max=1.0)
        return S, U, ref_unfold, R_star_arg

    @staticmethod
    def _batch_index_select(input_: Tensor, dim: int, index: Tensor) -> Tensor:
        """Per-batch ``torch.gather`` along ``dim``."""
        views = [input_.size(0)] + [
            1 if i != dim else -1 for i in range(1, len(input_.size()))
        ]
        expanse = list(input_.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input_, dim, index)


# ---------------------------------------------------------------------------
# Latent codec
# ---------------------------------------------------------------------------


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

        # Local context: 5x5 mask-A conv, M -> 2M.
        self.mask_conv = MaskedConv2d(M, 2 * M, 5, 1, 2, bias=bias, mask_type="A")
        # Global reference search and masked-conv on unfolded refs.
        self.search = _SearchTransfer(M, k=sk)
        self.mask_conv_ref = _Conv2dUnfold(True, M, 2 * M, sk, 1, sk // 2, bias=bias)

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
