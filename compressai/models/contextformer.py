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
#   this list of conditions and/or other materials provided with the
#   distribution.
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

"""ContextFormer image compression model.

Koyuncu et al., `"Contextformer: A Transformer with Spatio-Channel Attention
for Context Modeling in Learned Image Compression"
<https://arxiv.org/abs/2203.02452>`_.

This CompressAI-native reproduction follows the paper's architecture at the
forward/rate-estimation level:

* 4-stage 3x3 Conv+GDN analysis/synthesis transforms with residual
  non-local attention blocks approximated by CompressAI's ``AttentionBlock``;
* factorized hyperprior with a 2-stride hyper-analysis / hyper-synthesis path;
* causal spatio-channel Transformer context model with ``Ncs`` channel
  segments and ``cfo``/``sfo`` sequence orders;
* K=3 Gaussian mixture likelihood through
  :class:`compressai.entropy_models.GaussianMixtureConditional`.

The practical bitstream path is intentionally left unimplemented because the
paper's decoder requires token-wise causal Transformer evaluation and
sliding-window / wavefront runtime scheduling.
"""

from __future__ import annotations

import math

from collections import OrderedDict
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianMixtureConditional
from compressai.layers import AttentionBlock, GDN, ResidualBlock, conv, deconv
from compressai.layers.attn import ContextFormerContextModel
from compressai.registry import register_model

from .base import CompressionModel

__all__ = ["ContextFormer"]


class _ContextFormerEntropyParameters(nn.Module):
    """Dense entropy-parameter head from Table A1."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        hidden1 = (2 * input_channels + output_channels) // 3
        hidden2 = (input_channels + 2 * output_channels) // 3
        self.net = nn.Sequential(
            nn.Linear(input_channels, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, output_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _make_analysis_transform(N: int, M: int) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            [
                ("conv1", conv(3, N, kernel_size=3, stride=2)),
                ("gdn1", GDN(N)),
                ("rnab", AttentionBlock(N)),
                ("conv2", conv(N, N, kernel_size=3, stride=2)),
                ("gdn2", GDN(N)),
                ("conv3", conv(N, N, kernel_size=3, stride=2)),
                ("gdn3", GDN(N)),
                ("conv4", conv(N, N, kernel_size=3, stride=2)),
                ("latent", conv(N, M, kernel_size=1, stride=1)),
            ]
        )
    )


def _make_synthesis_transform(N: int, M: int) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            [
                ("resblock1", ResidualBlock(M, M)),
                ("resblock2", ResidualBlock(M, M)),
                ("deconv1", deconv(M, N, kernel_size=3, stride=2)),
                ("igdn1", GDN(N, inverse=True)),
                ("deconv2", deconv(N, N, kernel_size=3, stride=2)),
                ("igdn2", GDN(N, inverse=True)),
                ("deconv3", deconv(N, N, kernel_size=3, stride=2)),
                ("rnab", AttentionBlock(N)),
                ("igdn3", GDN(N, inverse=True)),
                ("deconv4", deconv(N, 3, kernel_size=3, stride=2)),
            ]
        )
    )


def _make_hyper_analysis(N: int, M: int) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            [
                ("conv1", conv(M, N, kernel_size=3, stride=1)),
                ("act1", nn.LeakyReLU(inplace=True)),
                ("conv2", conv(N, N, kernel_size=3, stride=1)),
                ("conv3", conv(N, N, kernel_size=3, stride=2)),
                ("act3", nn.LeakyReLU(inplace=True)),
                ("conv4", conv(N, N, kernel_size=3, stride=1)),
                ("conv5", conv(N, N, kernel_size=3, stride=2)),
            ]
        )
    )


def _make_hyper_synthesis(N: int, M: int) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            [
                ("conv1", conv(N, M, kernel_size=3, stride=1)),
                ("act1", nn.LeakyReLU(inplace=True)),
                ("deconv1", deconv(M, M, kernel_size=3, stride=2)),
                ("act2", nn.LeakyReLU(inplace=True)),
                ("conv2", conv(M, M, kernel_size=3, stride=1)),
                ("act3", nn.LeakyReLU(inplace=True)),
                ("deconv2", deconv(M, M * 3 // 2, kernel_size=3, stride=2)),
                ("act4", nn.LeakyReLU(inplace=True)),
                ("conv3", conv(M * 3 // 2, 2 * M, kernel_size=3, stride=1)),
            ]
        )
    )


@register_model("contextformer")
class ContextFormer(CompressionModel):
    """ContextFormer end-to-end image compression model.

    Args:
        N: Transform and hyperprior width. Default ``192``.
        M: Latent channel count. Default ``192``.
        num_segments: Channel segments ``Ncs``. Default ``4``.
        embed_dim: ContextFormer embedding dimension ``de``. Default ``384``.
        depth: Number of causal Transformer layers ``L``. Default ``8``.
        num_heads: Multi-head attention heads ``h``. Default ``12``.
        mlp_ratio: Transformer feed-forward expansion. Default ``4``.
        order: Coding order, ``"cfo"`` or ``"sfo"``. Default ``"cfo"``.
        mixtures: Gaussian mixture components. Default ``3``.
        max_spatial_size: Maximum latent height/width for learned position
            embeddings. Default ``64`` (1024x1024 input at 16x downsampling).
        log_scale_min: Lower clamp for predicted log scales. Default ``-7``.
    """

    def __init__(
        self,
        N: int = 192,
        M: int = 192,
        *,
        num_segments: int = 4,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        order: str = "cfo",
        mixtures: int = 3,
        max_spatial_size: int = 64,
        dropout: float = 0.0,
        log_scale_min: float = -7.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        if M % num_segments != 0:
            raise ValueError(
                f"M ({M}) must be divisible by num_segments ({num_segments})"
            )
        if mixtures < 1:
            raise ValueError(f"mixtures must be positive, got {mixtures}")

        self.N = int(N)
        self.M = int(M)
        self.num_segments = int(num_segments)
        self.segment_channels = self.M // self.num_segments
        self.embed_dim = int(embed_dim)
        self.mixtures = int(mixtures)
        self.log_scale_min = float(log_scale_min)

        self.g_a = _make_analysis_transform(N, M)
        self.g_s = _make_synthesis_transform(N, M)
        self.h_a = _make_hyper_analysis(N, M)
        self.h_s = _make_hyper_synthesis(N, M)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianMixtureConditional(
            K=mixtures,
            scale_bound=float(math.exp(log_scale_min)),
        )
        self.context_model = ContextFormerContextModel(
            latent_channels=M,
            num_segments=num_segments,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            order=order,
            max_spatial_size=max_spatial_size,
            dropout=dropout,
        )
        output_channels = 3 * mixtures * self.segment_channels
        self.entropy_parameters = _ContextFormerEntropyParameters(
            input_channels=2 * M + embed_dim,
            output_channels=output_channels,
        )

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def _sequence_to_gmm_params(
        self, seq: Tensor, y_size: Tuple[int, int]
    ) -> Tensor:
        batch, length, channels = seq.shape
        height, width = y_size
        expected = height * width * self.num_segments
        if length != expected:
            raise ValueError(
                f"Expected sequence length {expected} for latent size "
                f"{y_size}, got {length}."
            )
        if channels != self.mixtures * self.segment_channels:
            raise ValueError(
                f"Expected {self.mixtures * self.segment_channels} sequence "
                f"channels, got {channels}."
            )

        if self.context_model.order == "cfo":
            seq = seq.reshape(
                batch,
                height,
                width,
                self.num_segments,
                self.mixtures,
                self.segment_channels,
            )
            return seq.permute(0, 4, 3, 5, 1, 2).reshape(
                batch, self.mixtures * self.M, height, width
            )

        seq = seq.reshape(
            batch,
            self.num_segments,
            height,
            width,
            self.mixtures,
            self.segment_channels,
        )
        return seq.permute(0, 4, 1, 5, 2, 3).reshape(
            batch, self.mixtures * self.M, height, width
        )

    def _split_gmm_parameters(
        self, seq: Tensor, y_size: Tuple[int, int]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        logits, means, log_scales = seq.chunk(3, dim=-1)
        scales = torch.exp(log_scales.clamp(min=self.log_scale_min))
        weights = logits.reshape(
            logits.size(0),
            logits.size(1),
            self.mixtures,
            self.segment_channels,
        ).softmax(dim=2)
        weights = weights.reshape(
            logits.size(0), logits.size(1), self.mixtures * self.segment_channels
        )

        return (
            self._sequence_to_gmm_params(scales, y_size),
            self._sequence_to_gmm_params(means, y_size),
            self._sequence_to_gmm_params(weights, y_size),
        )

    def forward(self, x: Tensor) -> Dict[str, Any]:
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyper_params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        context_seq = self.context_model(y_hat)
        hyper_seq = self.context_model.spatial_to_sequence(hyper_params)
        gaussian_seq = self.entropy_parameters(torch.cat((hyper_seq, context_seq), -1))
        scales_hat, means_hat, weights_hat = self._split_gmm_parameters(
            gaussian_seq, y.shape[-2:]
        )
        _, y_likelihoods = self.gaussian_conditional(
            y,
            scales_hat,
            means_hat,
            weights_hat,
        )

        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "ContextFormer":
        N = int(state_dict["g_a.conv1.weight"].size(0))
        M = int(state_dict["g_a.latent.weight"].size(0))
        num_segments = int(state_dict["context_model.segment_embedding.weight"].size(0))
        embed_dim = int(state_dict["context_model.input_projection.weight"].size(0))
        max_spatial_size = int(state_dict["context_model.row_embedding.weight"].size(0))
        num_heads = int(state_dict["context_model.num_heads"].item())
        order_code = int(state_dict["context_model.order_code"].item())
        order = "cfo" if order_code == 0 else "sfo"

        depth = sum(
            1
            for key in state_dict
            if key.startswith("context_model.blocks.")
            and key.endswith(".attn.in_proj_weight")
        )
        hidden_dim = int(state_dict["context_model.blocks.0.mlp.0.weight"].size(0))
        mlp_ratio = hidden_dim // embed_dim
        segment_channels = M // num_segments
        output_dim = int(state_dict["entropy_parameters.net.4.weight"].size(0))
        mixtures = output_dim // (3 * segment_channels)

        net = cls(
            N=N,
            M=M,
            num_segments=num_segments,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            order=order,
            mixtures=mixtures,
            max_spatial_size=max_spatial_size,
        )
        net.load_state_dict(state_dict)
        return net

    def compress(self, x: Tensor) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError(
            "ContextFormer.compress is not implemented yet. The paper's "
            "bitstream path requires causal token-wise Transformer evaluation "
            "plus sliding-window / wavefront scheduling. Use forward() for "
            "rate estimation."
        )

    def decompress(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError(
            "ContextFormer.decompress is not implemented yet; see compress()."
        )
