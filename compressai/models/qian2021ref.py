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

"""Reference-based auto-regressive image compression model.

Reference: Yichen Qian, Zhiyu Tan, Xiuyu Sun, Ming Lin, Dongyang Li, Zhenhong
Sun, Hao Li, Rong Jin, *"Learning Accurate Entropy Model with Global Reference
for Image Compression"*, ICLR 2021
(`arXiv:2010.08321 <https://arxiv.org/abs/2010.08321>`_).

Re-implementation of the official release at ``candidate_none/img-comp-reference``.
The architecture is a Ballé-style 4-stage Conv+GSDN auto-encoder with a
3-conv hyper-prior and a 3-cascade auto-regressive entropy model that
combines a local masked-conv context, a global reference search and a
hyperprior-derived feature into a discretised K=3 Gaussian mixture
likelihood.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.entropy_models import (
    GaussianMixtureConditional,
    LearnedGaussianBottleneck,
)
from compressai.latent_codecs import RefAutoregressiveLatentCodec
from compressai.layers.lic import (
    Balle2Decoder,
    Balle2Encoder,
    RefHyperDecoder,
    RefHyperEncoder,
)
from compressai.ops.parametrizers import NonNegativeParametrizer
from compressai.registry import register_model

from .base import CompressionModel

__all__ = ["RefBasedAR"]


class _GSDN(nn.Module):
    """Generalized Subtractive + Divisive Normalisation layer.

    GSDN extends GDN with a per-channel learnable subtractive (mean) term
    *before* the divisive (variance) normalisation::

        y[i] = (x[i] - mu[i]) / sqrt(beta[i] + sum_j gamma[j, i] * (x[j] - mu[j])**2)

    Only the divisive ``beta`` / ``gamma`` are reparametrised through
    :class:`compressai.ops.parametrizers.NonNegativeParametrizer`; the
    subtractive ``beta2`` / ``gamma2`` follow the same reparametrisation but
    with a different positivity floor on ``beta2`` (zero by default upstream).

    Module attribute names ``beta`` / ``gamma`` / ``beta2`` / ``gamma2`` and
    init values are kept identical to upstream ``module/ops.py::GSDN`` so the
    released checkpoints load 1:1.
    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.inverse = bool(inverse)

        # Divisive (beta, gamma): identical to GDN.
        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

        # Subtractive (beta2, gamma2): upstream initialises beta2 to zeros.
        self.beta2_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta2 = torch.zeros(in_channels)
        beta2 = self.beta2_reparam.init(beta2)
        self.beta2 = nn.Parameter(beta2)

        self.gamma2_reparam = NonNegativeParametrizer()
        gamma2 = gamma_init * torch.eye(in_channels)
        gamma2 = self.gamma2_reparam.init(gamma2)
        self.gamma2 = nn.Parameter(gamma2)

    def _norm_params(self, beta_p, gamma_p, beta_repar, gamma_repar, C):
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


@register_model("qian2021-ref")
class RefBasedAR(CompressionModel):
    r"""Reference-Based AR (Qian et al., ICLR 2021).

    Args:
        N: ``g_a`` intermediate channel count (upstream ``channels``,
            default 192).
        M: latent ``y`` channel count (upstream ``last_channels``, default
            384).
        Z: hyper latent ``z`` channel count (upstream ``hyper_channels``,
            default 192).
        norm: Either ``"GSDN"`` (default, matches the released MSE config) or
            ``"GDN"`` (compressai's standard GDN; mathematically the
            divisive-only variant).
        sk: Reference patch size for the global-reference search (default 3,
            matches upstream).
        head_channels: Hidden width of the three 1×1 cascade heads inside the
            AR codec (upstream ``channels = last_channels * 3``;
            default ``3 * M``).
        num_parameter: Number of distribution parameters per pixel; only
            ``3`` (logit_pi + mean + log-σ) is supported.
        log_scale_min: Lower clamp on predicted ``log σ``. Upstream sets
            this to ``-7`` (≈ σ ≥ 9e-4).
    """

    g_a: nn.Module
    g_s: nn.Module

    def __init__(
        self,
        N: int = 192,
        M: int = 384,
        Z: int = 192,
        *,
        norm: str = "GSDN",
        sk: int = 3,
        head_channels: int | None = None,
        num_parameter: int = 3,
        log_scale_min: float = -7.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if num_parameter != 3:
            raise ValueError(
                "RefBasedAR migration supports num_parameter=3 only "
                "(matches the released MSE config)."
            )
        if norm not in ("GSDN", "GDN"):
            raise ValueError(f"unknown norm {norm!r}; expected GSDN or GDN")

        from compressai.layers.gdn import GDN as _GDN

        norm_cls = _GSDN if norm == "GSDN" else _GDN

        self.N = int(N)
        self.M = int(M)
        self.Z = int(Z)
        self.norm = norm
        self.sk = int(sk)
        self.head_channels = (
            int(head_channels) if head_channels is not None else 3 * M
        )

        self.g_a = Balle2Encoder(channels=N, last_channels=M, norm_cls=norm_cls)
        self.g_s = Balle2Decoder(channels=N, last_channels=M, norm_cls=norm_cls)
        self.h_a = RefHyperEncoder(in_channel=M, out_channel=Z, channel=N)
        self.h_s = RefHyperDecoder(in_channel=Z, out_channel=2 * M, channel=N)

        self.entropy_bottleneck = LearnedGaussianBottleneck(Z)
        self.latent_codec = RefAutoregressiveLatentCodec(
            latent_channels=M,
            hyper_channels=Z,
            head_channels=self.head_channels,
            sk=self.sk,
            num_parameter=num_parameter,
            log_scale_min=log_scale_min,
            entropy_bottleneck=self.entropy_bottleneck,
        )

        self._hparams = dict(
            N=N,
            M=M,
            Z=Z,
            norm=norm,
            sk=sk,
            head_channels=self.head_channels,
            num_parameter=num_parameter,
        )

    @property
    def downsampling_factor(self) -> int:
        # 4 stride-2 stages in g_a + 2 stride-2 stages in h_a.
        return 2 ** (4 + 2)

    def forward(self, x: Tensor) -> Dict[str, Any]:
        # Upstream's ref model trains directly on [0, 1] images (g_a takes the
        # image as-is, g_s outputs in [0, 1] before clamp), so no [-1, 1] shell
        # like Entroformer.
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_feature = self.h_s(z_hat)
        codec_out = self.latent_codec(y, z_feature)
        x_hat = self.g_s(codec_out["y_hat"])
        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": codec_out["likelihoods"]["y"],
                "z": z_likelihoods,
            },
        }

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "RefBasedAR":
        """Instantiate ``RefBasedAR`` from a converted compressai-layout state dict.

        ``N`` is inferred from ``g_a.encoder.0.weight`` (out channels), ``M``
        from ``g_a.encoder.6.weight`` and ``Z`` from
        ``entropy_bottleneck.mu``. The presence of ``encoder.1.beta2`` (i.e.
        the GSDN subtractive parameter) selects ``norm='GSDN'`` versus
        ``norm='GDN'``.
        """
        N = state_dict["g_a.encoder.0.weight"].size(0)
        M = state_dict["g_a.encoder.6.weight"].size(0)
        Z = state_dict["entropy_bottleneck.mu"].size(1)
        norm = "GSDN" if "g_a.encoder.1.beta2" in state_dict else "GDN"

        # Infer head_channels from the cascade-1 head's hidden width.
        # `latent_codec.conv_1x1_1.0.weight` shape: (head, 2*M, 1, 1) → head.
        head_w = state_dict["latent_codec.conv_1x1_1.0.weight"]
        head_channels = head_w.size(0)

        # `mask_conv_ref.weight` shape (2M, M, sk, sk) → infer sk.
        sk = state_dict["latent_codec.mask_conv_ref.weight"].size(-1)

        net = cls(
            N=N,
            M=M,
            Z=Z,
            norm=norm,
            sk=sk,
            head_channels=head_channels,
            num_parameter=3,
        )
        net.load_state_dict(state_dict)
        return net

    def compress(self, x: Tensor):
        raise NotImplementedError(
            "RefBasedAR.compress is not implemented yet; the upstream raster-scan AR "
            "bitstream coding requires per-pixel re-evaluation of the search/cascade "
            "heads, and is a planned follow-up. Use forward() for rate estimation."
        )

    def decompress(self, *args, **kwargs):
        raise NotImplementedError(
            "RefBasedAR.decompress is not implemented yet; see compress()."
        )
