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

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import GeneralizedGaussianConditional
from compressai.latent_codecs import HierarchicalProgressiveLatentCodec
from compressai.layers.attn import WindowedCrossAttention
from compressai.layers.lic import DWConvResBlock, PConvResBlock
from compressai.registry import register_model

from .base import CompressionModel

__all__ = ["HPCM"]


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
            PConvResBlock(channels, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio)
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
        PConvResBlock(M, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
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
        PConvResBlock(M * 2, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
    )


class _SpatialPrior(nn.Module):
    """Two-branch DWConv stack producing 2*M channels from 3*M conditioning."""

    def __init__(self, M: int, branch1_depth: int = 2, branch2_depth: int = 1) -> None:
        super().__init__()
        self.branch_1 = nn.Sequential(
            *[DWConvResBlock(M * 3) for _ in range(branch1_depth)]
        )
        tail: list[nn.Module] = [DWConvResBlock(M * 3) for _ in range(branch2_depth)]
        tail.append(nn.Conv2d(M * 3, M * 2, kernel_size=1))
        self.branch_2 = nn.Sequential(*tail)

    def forward(self, x: Tensor, quant_step: Tensor) -> Tensor:
        return self.branch_2(self.branch_1(x) * quant_step)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@register_model("hpcm")
class HPCM(CompressionModel):
    """Unified HPCM model covering Base / Large / PhiContext variants."""

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

        Structural fields that can be inferred from tensor shapes (``N``,
        ``M``, ``g_a_depth``, ``g_s_depth``, ``y_prior_depth``,
        ``use_attention``) are auto-detected. Window sizes / head counts of
        the cross-attention modules are not encoded in tensor shapes, so they
        fall back to ``__init__`` defaults; pass them via ``overrides`` to
        match a non-default checkpoint.
        """
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

        # g_a inner depth: count PConvResBlocks in the 384-ch stage. Each
        # PConvResBlock contributes 4 sub-modules; the 384-ch range is between
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
        # Each DWConvResBlock contributes 1 index. So len = y_prior_depth + 1.
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
        net.load_state_dict(state_dict)
        return net
