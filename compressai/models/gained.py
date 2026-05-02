# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0.

"""Gained variable-rate hyperprior models.

Per-channel gain modulation on top of the bmshj2018 / mbt2018 hyperprior
families, from:

    Z. Cui et al., "Asymmetric Gained Deep Image Compression With Continuous
    Rate Adaptation", CVPR 2021.

This is a port of the unofficial reference implementation from
``https://github.com/mmSir/GainedVAE``. As noted in that repo, the asymmetric
Gaussian entropy model from the paper is *not* implemented; the contribution
here is the per-channel learned ``Gain`` / ``InverseGain`` vectors and the
power-mean interpolation that lets one trained model serve a continuous rate
range.

Three variants share a single file (mirroring ``vbr.py``):

* ``bmshj2018-hyperprior-gained`` — :class:`GainedScaleHyperprior`
* ``mbt2018-mean-gained``         — :class:`GainedMSHyperprior`
* ``mbt2018-mean-gained-sc``      — :class:`SCGainedMSHyperprior`
  (adds SPADE-style SFT modulation conditioned on a quality map ``qmap``;
  no upstream checkpoint exists for this variant).

The gain forward/inverse pair is asymmetric and learned independently, so the
modules expect the user to pass a level index ``s`` and an interpolation
factor ``l`` ∈ [0, 1] that mixes between adjacent levels ``s`` and ``s+1``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.entropy_models import GaussianConditional
from compressai.layers import GDN
from compressai.registry import register_model

from .base import get_scale_table
from .google import MeanScaleHyperprior, ScaleHyperprior
from .utils import conv, deconv

__all__ = [
    "GainedScaleHyperprior",
    "GainedMSHyperprior",
    "SCGainedMSHyperprior",
]


class _SFT(nn.Module):
    """Spatially-adaptive feature transform (SPADE-style).

    Predicts per-pixel ``(gamma, beta)`` from a conditioning map and applies
    ``out = x * (1 + gamma) + beta``. The conditioning map is adaptive-avg-pooled
    to ``x``'s spatial size, so resolution mismatches are tolerated.

    Args:
        x_nc: number of channels of the modulated feature ``x``.
        prior_nc: number of channels of the conditioning map.
        ks: kernel size for the gamma / beta predictor convs.
        nhidden: hidden width of the shared MLP.
    """

    def __init__(self, x_nc: int, prior_nc: int = 1, ks: int = 3, nhidden: int = 128):
        super().__init__()
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(prior_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU(),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)

    def forward(self, x: torch.Tensor, qmap: torch.Tensor) -> torch.Tensor:
        qmap = F.adaptive_avg_pool2d(qmap, x.size()[2:])
        actv = self.mlp_shared(qmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return x * (1 + gamma) + beta


# Default lambda set from the upstream implementation (HUAWEI CVPR 2021).
_DEFAULT_LMBDA = (0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003)


def _expand(vec: torch.Tensor) -> torch.Tensor:
    """Reshape a length-C gain vector to ``(1, C, 1, 1)`` for broadcasting."""
    return vec.unsqueeze(0).unsqueeze(2).unsqueeze(3)


def _interpolate_pow(g0: torch.Tensor, g1: torch.Tensor, l: float) -> torch.Tensor:
    """Geometric (power-mean) interpolation: ``g0^(1-l) * g1^l``."""
    return torch.abs(g0).pow(1 - l) * torch.abs(g1).pow(l)


def _interpolate_linear(g0: torch.Tensor, g1: torch.Tensor, l: float) -> torch.Tensor:
    """Plain linear interpolation: ``(1-l)*|g0| + l*|g1|``."""
    return torch.abs(g0) * (1 - l) + torch.abs(g1) * l


@register_model("bmshj2018-hyperprior-gained")
class GainedScaleHyperprior(ScaleHyperprior):
    r"""Variable bitrate (gained) version of bmshj2018-hyperprior.

    Per-channel gain modulation from
    `"Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation"
    <https://openaccess.thecvf.com/content/CVPR2021/html/Cui_Asymmetric_Gained_Deep_Image_Compression_With_Continuous_Rate_Adaptation_CVPR_2021_paper.html>`_,
    CVPR 2021. Wraps :class:`~compressai.models.ScaleHyperprior` with four
    learned per-channel gain tables (``Gain``/``InverseGain`` of width ``M``;
    ``HyperGain``/``InverseHyperGain`` of width ``N``), one row per discrete
    rate level.

    Args:
        N (int): Number of channels in the main backbone.
        M (int): Number of channels in the latent / last hyper layer.
        lmbda (Sequence[float]): Per-level rate-distortion trade-off targets.
            Determines the number of discrete levels (``len(lmbda)``).
    """

    def __init__(self, N: int = 192, M: int = 320, lmbda=_DEFAULT_LMBDA, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.lmbda = list(lmbda)
        self.levels = len(self.lmbda)
        self.Gain = nn.Parameter(torch.ones(self.levels, M), requires_grad=True)
        self.InverseGain = nn.Parameter(torch.ones(self.levels, M), requires_grad=True)
        self.HyperGain = nn.Parameter(torch.ones(self.levels, N), requires_grad=True)
        self.InverseHyperGain = nn.Parameter(
            torch.ones(self.levels, N), requires_grad=True
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        levels = state_dict["Gain"].size(0)
        net = cls(N=N, M=M, lmbda=[0.0] * levels)
        net.load_state_dict(state_dict)
        return net

    # --- gain helpers ---------------------------------------------------------

    def _check_level(self, s: int):
        if s not in range(0, self.levels - 1):
            raise ValueError(
                f"s should be in range(0, {self.levels - 1}), got {s}"
            )

    def _check_lambda(self, l: float):
        if not (0.0 <= l <= 1.0):
            raise ValueError(f"l should be in [0, 1], got {l}")

    def forward(self, x: torch.Tensor, s: int):
        """Train-time forward at discrete level ``s``."""
        gain = _expand(torch.abs(self.Gain[s]))
        hyper_gain = _expand(torch.abs(self.HyperGain[s]))
        inv_gain = _expand(torch.abs(self.InverseGain[s]))
        inv_hyper = _expand(torch.abs(self.InverseHyperGain[s]))

        y = self.g_a(x) * gain
        z = self.h_a(y) * hyper_gain
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = z_hat * inv_hyper
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        y_hat = y_hat * inv_gain
        x_hat = self.g_s(y_hat)

        return {
            "y": y,
            "y_hat": y_hat,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x: torch.Tensor, s: int, l: float):
        self._check_level(s)
        self._check_lambda(l)

        gain = _expand(_interpolate_pow(self.Gain[s], self.Gain[s + 1], l))
        hyper_gain = _expand(
            _interpolate_pow(self.HyperGain[s], self.HyperGain[s + 1], l)
        )
        inv_hyper = _expand(
            _interpolate_pow(self.InverseHyperGain[s], self.InverseHyperGain[s + 1], l)
        )

        y = self.g_a(x) * gain
        z = self.h_a(y) * hyper_gain

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = z_hat * inv_hyper

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, s: int, l: float):
        assert isinstance(strings, list) and len(strings) == 2
        self._check_level(s)
        self._check_lambda(l)

        # Upstream uses linear interpolation for inverse gains in decompress
        # (the pow form is overwritten); preserved for bit-exact behaviour.
        inv_gain = _expand(
            _interpolate_linear(self.InverseGain[s], self.InverseGain[s + 1], l)
        )
        inv_hyper = _expand(
            _interpolate_linear(
                self.InverseHyperGain[s], self.InverseHyperGain[s + 1], l
            )
        )

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = z_hat * inv_hyper
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        y_hat = y_hat * inv_gain
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def update(self, scale_table=None, force: bool = False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )
        updated |= super(ScaleHyperprior, self).update(force=force)
        return updated


@register_model("mbt2018-mean-gained")
class GainedMSHyperprior(MeanScaleHyperprior):
    r"""Variable bitrate (gained) version of mbt2018-mean (MeanScaleHyperprior)."""

    def __init__(self, N: int = 128, M: int = 192, lmbda=_DEFAULT_LMBDA, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.lmbda = list(lmbda)
        self.levels = len(self.lmbda)
        self.Gain = nn.Parameter(torch.ones(self.levels, M), requires_grad=True)
        self.InverseGain = nn.Parameter(torch.ones(self.levels, M), requires_grad=True)
        self.HyperGain = nn.Parameter(torch.ones(self.levels, N), requires_grad=True)
        self.InverseHyperGain = nn.Parameter(
            torch.ones(self.levels, N), requires_grad=True
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        levels = state_dict["Gain"].size(0)
        net = cls(N=N, M=M, lmbda=[0.0] * levels)
        net.load_state_dict(state_dict)
        return net

    def _check_level(self, s: int):
        if s not in range(0, self.levels - 1):
            raise ValueError(
                f"s should be in range(0, {self.levels - 1}), got {s}"
            )

    def _check_lambda(self, l: float):
        if not (0.0 <= l <= 1.0):
            raise ValueError(f"l should be in [0, 1], got {l}")

    def forward(self, x: torch.Tensor, s: int):
        gain = _expand(torch.abs(self.Gain[s]))
        hyper_gain = _expand(torch.abs(self.HyperGain[s]))
        inv_gain = _expand(torch.abs(self.InverseGain[s]))
        inv_hyper = _expand(torch.abs(self.InverseHyperGain[s]))

        y = self.g_a(x) * gain
        z = self.h_a(y) * hyper_gain
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = z_hat * inv_hyper
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(
            y, scales_hat, means=means_hat
        )
        y_hat = y_hat * inv_gain
        x_hat = self.g_s(y_hat)

        return {
            "y": y,
            "y_hat": y_hat,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x: torch.Tensor, s: int, l: float):
        self._check_level(s)
        self._check_lambda(l)

        # Upstream MS variant uses linear interpolation for *all* gains in
        # compress() — the initial pow form is immediately overwritten.
        gain = _expand(_interpolate_linear(self.Gain[s], self.Gain[s + 1], l))
        hyper_gain = _expand(
            _interpolate_linear(self.HyperGain[s], self.HyperGain[s + 1], l)
        )
        inv_hyper = _expand(
            _interpolate_linear(
                self.InverseHyperGain[s], self.InverseHyperGain[s + 1], l
            )
        )

        y = self.g_a(x) * gain
        z = self.h_a(y) * hyper_gain

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = z_hat * inv_hyper

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, s: int, l: float):
        assert isinstance(strings, list) and len(strings) == 2
        self._check_level(s)
        self._check_lambda(l)

        inv_gain = _expand(
            _interpolate_linear(self.InverseGain[s], self.InverseGain[s + 1], l)
        )
        inv_hyper = _expand(
            _interpolate_linear(
                self.InverseHyperGain[s], self.InverseHyperGain[s + 1], l
            )
        )

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = z_hat * inv_hyper
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        y_hat = y_hat * inv_gain
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def update(self, scale_table=None, force: bool = False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )
        updated |= super(ScaleHyperprior, self).update(force=force)
        return updated


@register_model("mbt2018-mean-gained-sc")
class SCGainedMSHyperprior(GainedMSHyperprior):
    r"""Spatial-channel gained MS hyperprior with SPADE-style SFT modulation.

    Adds an extra ``qmap`` (quality map) input that is fused into the encoder
    via SPADE-style :class:`_SFT` blocks at three intermediate stages,
    and re-injected on the decoder side from the hyper-latent ``z_hat``. Used
    for spatially-varying rate control.

    No upstream pretrained checkpoint exists for this variant. Forward / state
    dict round-trip are still verified by the test suite.
    """

    def __init__(self, N: int = 128, M: int = 192, lmbda=_DEFAULT_LMBDA, **kwargs):
        super().__init__(N=N, M=M, lmbda=lmbda, **kwargs)

        # Replace the inherited monolithic g_a / g_s with the staged versions.
        del self.g_a
        del self.g_s

        # ---- encoder qmap path ----
        self.qmap_feature_ga0 = nn.Sequential(
            conv(4, N * 2, kernel_size=3, stride=1),
            nn.LeakyReLU(0.1, True),
            conv(N * 2, N, kernel_size=3, stride=1),
            nn.LeakyReLU(0.1, True),
            conv(N, N, kernel_size=3, stride=1),
        )
        self.qmap_feature_ga1 = nn.Sequential(
            conv(N, N, kernel_size=3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, kernel_size=1, stride=1),
        )
        self.ga_SFT1 = _SFT(N, prior_nc=N, ks=3)
        self.qmap_feature_ga2 = nn.Sequential(
            conv(N, N, kernel_size=3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, kernel_size=1, stride=1),
        )
        self.ga_SFT2 = _SFT(N, prior_nc=N, ks=3)
        self.qmap_feature_ga3 = nn.Sequential(
            conv(N, N, kernel_size=3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, kernel_size=1, stride=1),
        )
        self.ga_SFT3 = _SFT(N, prior_nc=N, ks=3)

        # ---- encoder backbone (4 stages) ----
        self.g_a1 = nn.Sequential(conv(3, N), GDN(N))
        self.g_a2 = nn.Sequential(conv(N, N), GDN(N))
        self.g_a3 = nn.Sequential(conv(N, N), GDN(N))
        self.g_a4 = nn.Sequential(conv(N, M))

        # ---- decoder qmap path (regenerated from z_hat) ----
        self.qmap_feature_generation = nn.Sequential(
            deconv(N, N // 2, kernel_size=3),
            nn.LeakyReLU(0.1, True),
            deconv(N // 2, N // 4),
            nn.LeakyReLU(0.1, True),
            conv(N // 4, N // 4, kernel_size=3, stride=1),
        )
        self.qmap_feature_gs0 = nn.Sequential(
            conv(M + N // 4, N * 4, kernel_size=3, stride=1),
            nn.LeakyReLU(0.1, True),
            conv(N * 4, N * 2, kernel_size=3, stride=1),
            nn.LeakyReLU(0.1, True),
            conv(N * 2, N, kernel_size=3, stride=1),
        )
        self.gs_SFT0 = _SFT(M, prior_nc=N, ks=3)
        self.qmap_feature_gs1 = nn.Sequential(
            deconv(N, N, kernel_size=3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, kernel_size=1, stride=1),
        )
        self.gs_SFT1 = _SFT(N, prior_nc=N, ks=3)
        self.qmap_feature_gs2 = nn.Sequential(
            deconv(N, N, kernel_size=3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, kernel_size=1, stride=1),
        )
        self.gs_SFT2 = _SFT(N, prior_nc=N, ks=3)
        self.qmap_feature_gs3 = nn.Sequential(
            deconv(N, N, kernel_size=3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, kernel_size=1, stride=1),
        )
        self.gs_SFT3 = _SFT(N, prior_nc=N, ks=3)

        # ---- decoder backbone (4 stages) ----
        self.g_s1 = nn.Sequential(deconv(M, N), GDN(N, inverse=True))
        self.g_s2 = nn.Sequential(deconv(N, N), GDN(N, inverse=True))
        self.g_s3 = nn.Sequential(deconv(N, N), GDN(N, inverse=True))
        self.g_s4 = nn.Sequential(deconv(N, 3))

    @classmethod
    def from_state_dict(cls, state_dict):
        # SC variant has split encoder; M comes from the last stage,
        # N from the first stage.
        N = state_dict["g_a1.0.weight"].size(0)
        M = state_dict["g_a4.0.weight"].size(0)
        levels = state_dict["Gain"].size(0)
        net = cls(N=N, M=M, lmbda=[0.0] * levels)
        net.load_state_dict(state_dict)
        return net

    def _check_level_compress(self, s: int):
        if s not in range(0, self.levels):
            raise ValueError(f"s should be in range(0, {self.levels}), got {s}")

    # --- staged encoder / decoder ---------------------------------------------

    def g_a(self, x: torch.Tensor, qmap: torch.Tensor) -> torch.Tensor:
        qmap = self.qmap_feature_ga0(torch.cat([qmap, x], dim=1))
        qmap = self.qmap_feature_ga1(qmap)
        x = self.ga_SFT1(self.g_a1(x), qmap)

        qmap = self.qmap_feature_ga2(qmap)
        x = self.ga_SFT2(self.g_a2(x), qmap)

        qmap = self.qmap_feature_ga3(qmap)
        x = self.ga_SFT3(self.g_a3(x), qmap)

        return self.g_a4(x)

    def g_s(self, y_hat: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
        w = self.qmap_feature_generation(z_hat)
        w = self.qmap_feature_gs0(torch.cat([w, y_hat], dim=1))
        x = self.gs_SFT0(y_hat, w)

        w = self.qmap_feature_gs1(w)
        x = self.gs_SFT1(self.g_s1(x), w)

        w = self.qmap_feature_gs2(w)
        x = self.gs_SFT2(self.g_s2(x), w)

        w = self.qmap_feature_gs3(w)
        x = self.gs_SFT3(self.g_s3(x), w)

        return self.g_s4(x)

    def forward(self, x: torch.Tensor, s: int, qmap: torch.Tensor):
        gain = _expand(torch.abs(self.Gain[s]))
        hyper_gain = _expand(torch.abs(self.HyperGain[s]))
        inv_gain = _expand(torch.abs(self.InverseGain[s]))
        inv_hyper = _expand(torch.abs(self.InverseHyperGain[s]))

        y = self.g_a(x, qmap) * gain
        z = self.h_a(y) * hyper_gain
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = z_hat * inv_hyper
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(
            y, scales_hat, means=means_hat
        )
        y_hat = y_hat * inv_gain
        x_hat = self.g_s(y_hat, z_hat)

        return {
            "y": y,
            "y_hat": y_hat,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x: torch.Tensor, s: int, l: float, qmap: torch.Tensor):
        self._check_level_compress(s)
        self._check_lambda(l)
        if s + l > self.levels - 1:
            raise ValueError(
                f"s + l must be <= {self.levels - 1}, got s={s}, l={l}"
            )

        if s == self.levels - 1:
            gain_v = torch.abs(self.Gain[s])
            hyper_v = torch.abs(self.HyperGain[s])
            inv_hyper_v = torch.abs(self.InverseHyperGain[s])
        else:
            gain_v = _interpolate_pow(self.Gain[s], self.Gain[s + 1], l)
            hyper_v = _interpolate_pow(self.HyperGain[s], self.HyperGain[s + 1], l)
            inv_hyper_v = _interpolate_pow(
                self.InverseHyperGain[s], self.InverseHyperGain[s + 1], l
            )

        gain = _expand(gain_v)
        hyper_gain = _expand(hyper_v)
        inv_hyper = _expand(inv_hyper_v)

        y = self.g_a(x, qmap) * gain
        z = self.h_a(y) * hyper_gain

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = z_hat * inv_hyper

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, s: int, l: float):
        assert isinstance(strings, list) and len(strings) == 2
        self._check_level_compress(s)
        self._check_lambda(l)
        if s + l > self.levels - 1:
            raise ValueError(
                f"s + l must be <= {self.levels - 1}, got s={s}, l={l}"
            )

        if s == self.levels - 1:
            inv_gain_v = torch.abs(self.InverseGain[s])
            inv_hyper_v = torch.abs(self.InverseHyperGain[s])
        else:
            inv_gain_v = _interpolate_pow(
                self.InverseGain[s], self.InverseGain[s + 1], l
            )
            inv_hyper_v = _interpolate_pow(
                self.InverseHyperGain[s], self.InverseHyperGain[s + 1], l
            )

        inv_gain = _expand(inv_gain_v)
        inv_hyper = _expand(inv_hyper_v)

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = z_hat * inv_hyper
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        y_hat = y_hat * inv_gain
        x_hat = self.g_s(y_hat, z_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
