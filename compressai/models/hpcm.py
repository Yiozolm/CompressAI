# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# This file adapts code from https://github.com/lyq133/LIC-HPCM
# (originally distributed under the MIT License). The upstream copyright
# notice is preserved in that repository; modifications by InterDigital
# Communications, Inc. are released under the BSD 3-Clause Clear License
# terms below.

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

"""HPCM (Hierarchical Progressive Context Modeling) image compression model.

Reference: Lyu et al., "Learned Image Compression with Hierarchical Progressive
Context Modeling", ICCV 2025 (`arXiv:2507.19125`_).

This module exposes a single ``HPCM`` class parameterised to cover the three
upstream variants:

    * ``HPCM_Base``: ``g_a_depth=2``, ``g_s_depth=2``, ``y_prior_depth=2``,
      ``use_attention=True``.
    * ``HPCM_Large``: ``g_a_depth=6``, ``g_s_depth=6``, ``y_prior_depth=3``,
      ``use_attention=True``.
    * ``HPCM_Base_PhiContext``: same shape as ``HPCM_Base`` but
      ``use_attention=False`` — the published ``HPCM_Phi_Context`` checkpoint
      uses this configuration.

The 3-stage hierarchical progressive entropy modelling is delegated to
:class:`compressai.latent_codecs.HierarchicalProgressiveLatentCodec`, and the
hyperprior uses :class:`compressai.entropy_models.GeneralizedGaussianConditional`
(``β=1.5``).

.. _arXiv:2507.19125: https://arxiv.org/abs/2507.19125
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import GeneralizedGaussianConditional
from compressai.latent_codecs import HierarchicalProgressiveLatentCodec
from compressai.layers.attn import WindowedCrossAttention
from compressai.registry import register_model

from .base import CompressionModel

__all__ = ["HPCM", "convert_upstream_state_dict"]


# ----------------------------------------------------------------------------
# HPCM building blocks
# (formerly compressai/layers/lic/hpcm.py; private to the HPCM model)
# ----------------------------------------------------------------------------


class _PartialConv3x3(nn.Module):
    """3x3 conv applied only to the first ``partial_channels`` channels.

    The remaining channels are passed through unchanged. Used as the spatial
    mixer in HPCM's ``_PConvResBlock``.
    """

    def __init__(self, channels: int, partial_channels: int) -> None:
        super().__init__()
        if partial_channels <= 0 or partial_channels > channels:
            raise ValueError(
                "partial_channels must satisfy 0 < partial_channels <= channels"
            )
        self.channels = int(channels)
        self.partial_channels = int(partial_channels)
        self.pconv = nn.Conv2d(
            self.partial_channels,
            self.partial_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        head, tail = torch.split(
            input_tensor,
            [self.partial_channels, self.channels - self.partial_channels],
            dim=1,
        )
        head = self.pconv(head)
        return torch.cat((head, tail), dim=1)


class _DWConvResBlock(nn.Module):
    """Depthwise-conv residual block: ``DW3x3 -> 1x1 -> act -> 1x1`` with skip."""

    def __init__(
        self,
        channels: int,
        mlp_ratio: int = 2,
        act: type[nn.Module] = nn.LeakyReLU,
    ) -> None:
        super().__init__()
        hidden_channels = channels * mlp_ratio
        self.branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
            ),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            act(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor + self.branch(input_tensor)


class _PConvResBlock(nn.Module):
    """Partial-conv residual block as used in HPCM analysis / synthesis stacks."""

    def __init__(
        self,
        channels: int,
        partial_ratio: int = 4,
        mlp_ratio: int = 2,
        act: type[nn.Module] = nn.LeakyReLU,
    ) -> None:
        super().__init__()
        if partial_ratio <= 0:
            raise ValueError("partial_ratio must be positive")
        partial_channels = channels // partial_ratio
        hidden_channels = channels * mlp_ratio
        self.branch = nn.Sequential(
            _PartialConv3x3(channels, partial_channels),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            act(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor + self.branch(input_tensor)


# ----------------------------------------------------------------------------
# Transform building blocks
# ----------------------------------------------------------------------------
def _conv2x2_down(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)


def _deconv2x2_up(in_ch: int, out_ch: int) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_ch, out_ch, kernel_size=2, stride=2, output_padding=0, padding=0
    )


def _conv4x4_down(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)


def _deconv4x4_up(in_ch: int, out_ch: int) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_ch, out_ch, kernel_size=4, stride=2, output_padding=0, padding=1
    )


def _build_pconv_stack(
    channels: int,
    depth: int,
    mlp_ratio: int = 4,
    partial_ratio: int = 4,
) -> nn.Sequential:
    return nn.Sequential(
        *[
            _PConvResBlock(channels, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio)
            for _ in range(depth)
        ]
    )


def _build_g_a(
    M: int,
    inner_depth: int,
    mlp_ratio: int = 4,
    partial_ratio: int = 4,
) -> nn.Sequential:
    """Analysis transform; ``inner_depth`` controls the depth of the 384-ch stage."""
    return nn.Sequential(
        _conv4x4_down(3, 96),
        *_build_pconv_stack(96, 2, mlp_ratio, partial_ratio),
        _conv2x2_down(96, 192),
        *_build_pconv_stack(192, 2, mlp_ratio, partial_ratio),
        _conv2x2_down(192, 384),
        *_build_pconv_stack(384, inner_depth, mlp_ratio, partial_ratio),
        _conv2x2_down(384, M),
    )


def _build_g_s(
    M: int,
    inner_depth: int,
    mlp_ratio: int = 4,
    partial_ratio: int = 4,
) -> nn.Sequential:
    return nn.Sequential(
        _deconv2x2_up(M, 384),
        *_build_pconv_stack(384, inner_depth, mlp_ratio, partial_ratio),
        _deconv2x2_up(384, 192),
        *_build_pconv_stack(192, 2, mlp_ratio, partial_ratio),
        _deconv2x2_up(192, 96),
        *_build_pconv_stack(96, 2, mlp_ratio, partial_ratio),
        _deconv4x4_up(96, 3),
    )


def _build_h_a(
    M: int,
    N: int,
    mlp_ratio: int = 4,
    partial_ratio: int = 4,
) -> nn.Sequential:
    return nn.Sequential(
        _PConvResBlock(M, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
        _conv2x2_down(M, N),
        *_build_pconv_stack(N, 3, mlp_ratio, partial_ratio),
        _conv2x2_down(N, N),
    )


def _build_h_s(
    M: int,
    N: int,
    mlp_ratio: int = 4,
    partial_ratio: int = 4,
) -> nn.Sequential:
    return nn.Sequential(
        _deconv2x2_up(N, N),
        *_build_pconv_stack(N, 3, mlp_ratio, partial_ratio),
        _deconv2x2_up(N, M * 2),
        _PConvResBlock(M * 2, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
    )


class _SpatialPrior(nn.Module):
    """Two-branch DWConv stack producing 2*M channels from 3*M conditioning."""

    def __init__(self, M: int, branch1_depth: int = 2, branch2_depth: int = 1) -> None:
        super().__init__()
        self.branch_1 = nn.Sequential(
            *[_DWConvResBlock(M * 3) for _ in range(branch1_depth)]
        )
        tail: list[nn.Module] = [_DWConvResBlock(M * 3) for _ in range(branch2_depth)]
        tail.append(nn.Conv2d(M * 3, M * 2, kernel_size=1))
        self.branch_2 = nn.Sequential(*tail)

    def forward(self, x: Tensor, quant_step: Tensor) -> Tensor:
        return self.branch_2(self.branch_1(x) * quant_step)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@register_model("hpcm")
class HPCM(CompressionModel):
    r"""HPCM model from J. Lyu et al.: `"Learned Image Compression with
    Hierarchical Progressive Context Modeling"
    <https://arxiv.org/abs/2507.19125>`_, IEEE/CVF Int. Conf. on Computer
    Vision (ICCV), 2025.

    Unified implementation covering the published Base / Large / PhiContext
    variants. The 3-stage hierarchical progressive entropy modelling is
    delegated to
    :class:`compressai.latent_codecs.HierarchicalProgressiveLatentCodec`, and
    the hyperprior uses
    :class:`compressai.entropy_models.GeneralizedGaussianConditional`
    (``β=1.5``).

    Args:
        N (int): Number of channels in the hyperprior.
        M (int): Number of channels in the latent representation.
        g_a_depth (int): Depth of the inner stage of the analysis transform.
        g_s_depth (int): Depth of the inner stage of the synthesis transform.
        y_prior_depth (int): Depth of the y-prior network.
        use_attention (bool): Enable windowed cross-attention blocks.
    """

    def __init__(
        self,
        N: int = 256,
        M: int = 320,
        g_a_depth: int = 2,
        g_s_depth: int = 2,
        y_prior_depth: int = 2,
        use_attention: bool = True,
        attn_window_s1: int = 4,
        attn_window_s2: int = 8,
        attn_window_s3: int = 8,
        attn_num_heads: int = 32,
    ) -> None:
        super().__init__()
        self.N = int(N)
        self.M = int(M)
        self.g_a_depth = int(g_a_depth)
        self.g_s_depth = int(g_s_depth)
        self.y_prior_depth = int(y_prior_depth)
        self.use_attention = bool(use_attention)

        self.g_a = _build_g_a(M, g_a_depth)
        self.g_s = _build_g_s(M, g_s_depth)
        self.h_a = _build_h_a(M, N)
        self.h_s = _build_h_s(M, N)

        spatial_prior_s1_s2 = _SpatialPrior(M, branch1_depth=2, branch2_depth=1)
        spatial_prior_s3 = _SpatialPrior(
            M, branch1_depth=y_prior_depth + 1, branch2_depth=y_prior_depth
        )

        adaptor_s1 = nn.ModuleList(
            [nn.Conv2d(3 * M, 3 * M, kernel_size=1) for _ in range(1)]
        )
        adaptor_s2 = nn.ModuleList(
            [nn.Conv2d(3 * M, 3 * M, kernel_size=1) for _ in range(3)]
        )
        adaptor_s3 = nn.ModuleList(
            [nn.Conv2d(3 * M, 3 * M, kernel_size=1) for _ in range(6)]
        )
        adaptive_params = nn.ParameterList(
            [nn.Parameter(torch.ones(1, M * 3, 1, 1)) for _ in range(10)]
        )
        context_net = nn.ModuleList(
            [nn.Conv2d(2 * M, 2 * M, kernel_size=1) for _ in range(2)]
        )

        if use_attention:
            attn_s1 = WindowedCrossAttention(
                M * 2, M * 2, window_size=attn_window_s1, num_heads=attn_num_heads
            )
            attn_s2 = WindowedCrossAttention(
                M * 2, M * 2, window_size=attn_window_s2, num_heads=attn_num_heads
            )
            attn_s3 = WindowedCrossAttention(
                M * 2, M * 2, window_size=attn_window_s3, num_heads=attn_num_heads
            )
        else:
            attn_s1 = attn_s2 = attn_s3 = None

        self.latent_codec = HierarchicalProgressiveLatentCodec(
            latent_channels=M,
            hyper_channels=N,
            spatial_prior_s1_s2=spatial_prior_s1_s2,
            spatial_prior_s3=spatial_prior_s3,
            adaptor_s1=adaptor_s1,
            adaptor_s2=adaptor_s2,
            adaptor_s3=adaptor_s3,
            adaptive_params=adaptive_params,
            context_net=context_net,
            attn_s1=attn_s1,
            attn_s2=attn_s2,
            attn_s3=attn_s3,
            gaussian_conditional=GeneralizedGaussianConditional(
                scale_table=None, beta=1.5
            ),
        )

    def forward(self, x: Tensor) -> Dict[str, Any]:
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat_for_synth = self.latent_codec.quantize_z(z) if not self.training else (
            self._ste_round(z - self.latent_codec.means_hyper)
            + self.latent_codec.means_hyper
        )
        params = self.h_s(z_hat_for_synth)
        out = self.latent_codec(y, z, params)
        x_hat = self.g_s(out["y_hat"])
        return {
            "x_hat": x_hat,
            "likelihoods": out["likelihoods"],
        }

    @staticmethod
    def _ste_round(x: Tensor) -> Tensor:
        return (torch.round(x) - x).detach() + x

    @classmethod
    def from_state_dict(
        cls, state_dict: Dict[str, Tensor], **overrides: Any
    ) -> "HPCM":
        """Reconstruct an HPCM model from a state dict (any variant).

        Accepts either the compressai layout (keys under ``g_a.*`` /
        ``latent_codec.*``) or the upstream LIC-HPCM layout (flat keys like
        ``g_a.branch.*`` / ``y_spatial_prior_adaptor_list_s1.*`` /
        ``entropy_estimation.*``). Upstream dicts are auto-detected and
        translated via :func:`convert_upstream_state_dict` before loading.

        Structural fields that can be inferred from tensor shapes (``N``,
        ``M``, ``g_a_depth``, ``g_s_depth``, ``y_prior_depth``,
        ``use_attention``) are auto-detected. Window sizes / head counts of
        the cross-attention modules are not encoded in tensor shapes, so they
        fall back to ``__init__`` defaults; pass them via ``overrides`` to
        match a non-default checkpoint.
        """
        if _is_upstream_state_dict(state_dict):
            state_dict = convert_upstream_state_dict(state_dict)
        # The first conv4x4_down weight is g_a.0.weight: (96, 3, 4, 4)
        # The last g_a entry conv2x2_down(384, M) is g_a[N_blocks-1].weight: (M, 384, 2, 2)
        # M:
        last_g_a_keys = sorted(
            (k for k in state_dict if k.startswith("g_a.") and k.endswith(".weight")),
            key=lambda k: int(k.split(".")[1]),
        )
        last_key = last_g_a_keys[-1]
        M = state_dict[last_key].size(0)

        # N from h_a final conv2x2_down (N, N, 2, 2) — the last h_a layer
        last_h_a_keys = sorted(
            (k for k in state_dict if k.startswith("h_a.") and k.endswith(".weight")),
            key=lambda k: int(k.split(".")[1]),
        )
        N = state_dict[last_h_a_keys[-1]].size(0)

        # g_a inner depth: count _PConvResBlocks in the 384-ch stage. Each
        # _PConvResBlock contributes 4 sub-modules; the 384-ch range is between
        # the 192->384 conv2x2_down and the 384->M conv2x2_down.
        # Easier: derive from total g_a children count (index 0..5+inner_depth+...).
        g_a_indices = sorted({int(k.split(".")[1]) for k in state_dict if k.startswith("g_a.")})
        # Fixed structure: 1 + 2 + 1 + 2 + 1 + inner_depth + 1 = 8 + inner_depth modules.
        g_a_depth = len(g_a_indices) - 8
        if g_a_depth < 1:
            raise ValueError("Could not infer g_a_depth from state_dict")

        # g_s inner depth: same analysis on g_s children (1 + inner + 1 + 2 + 1 + 2 + 1 = 8 + inner).
        g_s_indices = sorted({int(k.split(".")[1]) for k in state_dict if k.startswith("g_s.")})
        g_s_depth = len(g_s_indices) - 8

        # y_prior_depth: spatial_prior_s3 lives under latent_codec.
        # branch_1 length = y_prior_depth + 1; we count Conv2d weights in branch_1.
        s3_branch1_indices = sorted(
            {
                int(k.split(".")[3])
                for k in state_dict
                if k.startswith("latent_codec.spatial_prior_s3.branch_1.")
            }
        )
        # Each _DWConvResBlock contributes 1 index. So len = y_prior_depth + 1.
        y_prior_depth = max(1, len(s3_branch1_indices) - 1)

        # use_attention: presence of attn_s1.* keys in latent_codec
        use_attention = any(
            k.startswith("latent_codec.attn_s1.conv_q.") for k in state_dict
        )

        net = cls(
            N=N,
            M=M,
            g_a_depth=g_a_depth,
            g_s_depth=g_s_depth,
            y_prior_depth=y_prior_depth,
            use_attention=use_attention,
            **overrides,
        )
        # The upstream PhiContext checkpoint omits `adaptive_params` (defaults
        # to ones, equivalent to no scaling) and `attn_*` (no attention). Allow
        # missing keys but reject any unexpected extras so genuine errors still
        # surface.
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        unexpected = [
            key
            for key in unexpected
            if not key.startswith("latent_codec.attn_")
            and not key.startswith("latent_codec.adaptive_params")
        ]
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys in HPCM state dict: {unexpected[:10]}"
                + ("..." if len(unexpected) > 10 else "")
            )
        allowed_missing_prefixes = (
            "latent_codec.adaptive_params",
            "latent_codec.attn_s1",
            "latent_codec.attn_s2",
            "latent_codec.attn_s3",
            # Initialised from __init__ defaults; upstream checkpoints don't ship it.
            "latent_codec.gaussian_conditional.scale_bound",
        )
        unwelcome_missing = [
            key
            for key in missing
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if unwelcome_missing:
            raise RuntimeError(
                f"Missing keys in HPCM state dict: {unwelcome_missing[:10]}"
                + ("..." if len(unwelcome_missing) > 10 else "")
            )
        return net


# ----------------------------------------------------------------------------
# Upstream-checkpoint conversion
# ----------------------------------------------------------------------------
def _is_upstream_state_dict(state_dict: Dict[str, Tensor]) -> bool:
    """Heuristic: upstream checkpoints carry the candidate-only key prefixes."""
    sentinel_keys = (
        "entropy_estimation.beta",
        "y_spatial_prior_adaptor_list_s1.0.weight",
        "y_spatial_prior_s1_s2.branch_1.0.branch.0.weight",
    )
    return any(key in state_dict for key in sentinel_keys)


_UPSTREAM_TOPLEVEL_RENAMES = {
    "means_hyper": "latent_codec.means_hyper",
    "scales_hyper": "latent_codec.scales_hyper",
    "scale_table": "latent_codec.gaussian_conditional.scale_table",
    "quantized_cdf_y": "latent_codec.gaussian_conditional._quantized_cdf",
    "cdf_length_y": "latent_codec.gaussian_conditional._cdf_length",
    "offset_y": "latent_codec.gaussian_conditional._offset",
    "entropy_estimation.beta": "latent_codec.gaussian_conditional.beta",
    "entropy_estimation.scale_lower_bound.bound": (
        "latent_codec.gaussian_conditional.lower_bound_scale.bound"
    ),
    "entropy_estimation.likelihood_lower_bound.bound": (
        "latent_codec.gaussian_conditional.likelihood_lower_bound.bound"
    ),
}

# Hyperprior-side CDFs and the upstream-only adaptive_params_list are dropped:
#   - quantized_cdf_z / cdf_length_z / offset_z were used by the upstream rANS
#     encoder for ``z``; compressai's GeneralizedGaussianConditional regenerates
#     them on demand and the forward pass doesn't need them.
#   - adaptive_params_list defaults to all-ones in our model, equivalent to the
#     upstream variants that omit it (e.g. the published HPCM_Phi checkpoint).
_UPSTREAM_DROP_KEYS = (
    "quantized_cdf_z",
    "cdf_length_z",
    "offset_z",
)


def _strip_branch_prefix(key: str, root: str) -> Optional[str]:
    """``g_a.branch.X`` → ``g_a.X`` for a fixed ``root`` like ``g_a``.

    Returns ``None`` when the key does not start with ``f"{root}.branch."``.
    """
    prefix = f"{root}.branch."
    if not key.startswith(prefix):
        return None
    return f"{root}." + key[len(prefix):]


def convert_upstream_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Translate a candidate ``LIC-HPCM`` state dict into compressai layout.

    The upstream layout flattens everything onto the model root and wraps each
    transform in a ``self.branch = nn.Sequential(...)`` attribute. compressai's
    HPCM mirrors the same module structure but: (1) drops the ``branch.``
    indirection on the four transforms, (2) houses the per-stage spatial
    priors / context net / hyper buffers / GGM under ``latent_codec.*``, and
    (3) renames ``entropy_estimation``'s lower-bound buffer to match
    :class:`GaussianConditional`'s naming.
    """
    converted: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key in _UPSTREAM_DROP_KEYS:
            continue
        new_key = _UPSTREAM_TOPLEVEL_RENAMES.get(key)
        if new_key is not None:
            converted[new_key] = value
            continue

        for root in ("g_a", "g_s", "h_a", "h_s"):
            stripped = _strip_branch_prefix(key, root)
            if stripped is not None:
                converted[stripped] = value
                break
        else:
            # Latent-codec-side keys: prefix every remaining upstream key.
            if key.startswith("y_spatial_prior_adaptor_list_s1"):
                converted[
                    key.replace(
                        "y_spatial_prior_adaptor_list_s1",
                        "latent_codec.adaptor_s1",
                        1,
                    )
                ] = value
            elif key.startswith("y_spatial_prior_adaptor_list_s2"):
                converted[
                    key.replace(
                        "y_spatial_prior_adaptor_list_s2",
                        "latent_codec.adaptor_s2",
                        1,
                    )
                ] = value
            elif key.startswith("y_spatial_prior_adaptor_list_s3"):
                converted[
                    key.replace(
                        "y_spatial_prior_adaptor_list_s3",
                        "latent_codec.adaptor_s3",
                        1,
                    )
                ] = value
            elif key.startswith("y_spatial_prior_s1_s2"):
                converted[
                    key.replace(
                        "y_spatial_prior_s1_s2",
                        "latent_codec.spatial_prior_s1_s2",
                        1,
                    )
                ] = value
            elif key.startswith("y_spatial_prior_s3"):
                converted[
                    key.replace(
                        "y_spatial_prior_s3",
                        "latent_codec.spatial_prior_s3",
                        1,
                    )
                ] = value
            elif key.startswith("context_net"):
                converted[
                    key.replace("context_net", "latent_codec.context_net", 1)
                ] = value
            elif key.startswith("attn_s1"):
                converted[
                    key.replace("attn_s1", "latent_codec.attn_s1", 1)
                ] = value
            elif key.startswith("attn_s2"):
                converted[
                    key.replace("attn_s2", "latent_codec.attn_s2", 1)
                ] = value
            elif key.startswith("attn_s3"):
                converted[
                    key.replace("attn_s3", "latent_codec.attn_s3", 1)
                ] = value
            elif key.startswith("adaptive_params_list"):
                converted[
                    key.replace(
                        "adaptive_params_list",
                        "latent_codec.adaptive_params",
                        1,
                    )
                ] = value
            else:
                # Pass through unknown keys; load_state_dict will surface them
                # as `unexpected_keys` if the model doesn't claim them.
                converted[key] = value
    return converted
