# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# This file adapts code from https://github.com/JiangWeibeta/MLIC
# (originally distributed under the Apache License 2.0). The upstream
# copyright notice is preserved in that repository; modifications by
# InterDigital Communications, Inc. are released under the BSD 3-Clause
# Clear License terms below.

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

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.latent_codecs import MLICPlusPlusLatentCodec
from compressai.layers.lic.mlic import AnalysisTransform, SynthesisTransform
from compressai.registry import register_model

from .base import CompressionModel

__all__ = ["MLICPlusPlus"]

_LEGACY_LATENT_PREFIXES = (
    "h_a.",
    "h_s.",
    "entropy_bottleneck.",
    "gaussian_conditional.",
    "local_context.",
    "channel_context.",
    "global_inter_context.",
    "global_intra_context.",
    "entropy_parameters_anchor.",
    "entropy_parameters_nonanchor.",
    "lrp_anchor.",
    "lrp_nonanchor.",
)


@register_model("mlicpp")
class MLICPlusPlus(CompressionModel):
    r"""MLIC++ model from W. Jiang, J. Yang, Y. Zhai, F. Gao, R. Wang:
    `"MLIC++: Linear Complexity Multi-Reference Entropy Modeling for Learned
    Image Compression" <https://arxiv.org/abs/2307.15421>`_, ACM Trans.
    Multimedia Comput. Commun. Appl. (TOMM), 2025; ICML 2023 Neural
    Compression Workshop.

    Builds a multi-reference entropy model with linear-complexity local,
    channel, and global intra/inter context modules on top of a Minnen2018
    style hyperprior.

    Args:
        N (int): Number of channels in the hyperprior.
        M (int): Number of channels in the latent representation.
        slice_num (int): Number of channel slices for the entropy model.
        context_window (int): Spatial context window size (odd).
    """

    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        slice_num: int = 10,
        context_window: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if slice_num <= 0:
            raise ValueError("slice_num must be positive")
        if context_window % 2 == 0:
            raise ValueError("context_window must be odd")
        if M % slice_num != 0:
            raise ValueError("M must be divisible by slice_num")

        self.N = int(N)
        self.M = int(M)
        self.context_window = int(context_window)
        self.slice_num = int(slice_num)
        self.slice_ch = int(M // slice_num)

        self.g_a = AnalysisTransform(N=N, M=M)
        self.g_s = SynthesisTransform(N=N, M=M)
        self.latent_codec = MLICPlusPlusLatentCodec(
            N=N,
            M=M,
            slice_num=slice_num,
            context_window=context_window,
        )

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    @property
    def h_a(self) -> nn.Module:
        return self.latent_codec.h_a

    @property
    def h_s(self) -> nn.Module:
        return self.latent_codec.h_s

    @property
    def entropy_bottleneck(self) -> EntropyBottleneck:
        return self.latent_codec.entropy_bottleneck

    @property
    def gaussian_conditional(self) -> GaussianConditional:
        return self.latent_codec.gaussian_conditional

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor] | Tensor]:
        y = self.g_a(x)
        y_out = self.latent_codec(y)
        return {
            "x_hat": self.g_s(y_out["y_hat"]),
            "likelihoods": y_out["likelihoods"],
        }

    def compress(self, x: Tensor) -> Dict[str, object]:
        y = self.g_a(x)
        y_out = self.latent_codec.compress(y)
        return {"strings": y_out["strings"], "shape": y_out["shape"]}

    def decompress(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        y_out = self.latent_codec.decompress(strings, shape)
        return {"x_hat": self.g_s(y_out["y_hat"]).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "MLICPlusPlus":
        state_dict = cls._migrate_state_dict(state_dict)
        N = state_dict["g_a.analysis_transform.0.conv1.weight"].size(0)
        M = state_dict["g_a.analysis_transform.6.weight"].size(0)
        slice_indices = {
            int(key.split(".")[2])
            for key in state_dict
            if key.startswith("latent_codec.local_context.")
            and key.endswith(".relative_position_table")
        }
        slice_num = len(slice_indices) or 10
        context_tokens = state_dict[
            "latent_codec.local_context.0.relative_position_index"
        ].size(0)
        context_window = int(round(context_tokens**0.5))
        net = cls(N=N, M=M, slice_num=slice_num, context_window=context_window)
        net.load_state_dict(state_dict)
        return net

    @staticmethod
    def _migrate_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {
            (
                f"latent_codec.{key}"
                if not key.startswith("latent_codec.")
                and key.startswith(_LEGACY_LATENT_PREFIXES)
                else key
            ): value
            for key, value in state_dict.items()
        }
