# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# This file adapts code from https://github.com/lumingzzz/TinyLIC
# (originally distributed under the Apache License 2.0). The upstream copyright
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

"""TinyLIC: NAT-based learned image compression.

Lu & Ma, "High-Efficiency Lossy Image Coding Through Adaptive Neighborhood
Information Aggregation", `arXiv:2204.11448`_ (Apache 2.0).

Backbone module names ``g_a{0..7}`` / ``g_s{0..7}`` / ``h_a{0..3}`` /
``h_s{0..3}`` are preserved verbatim from upstream. The staged channel +
checkerboard entropy model is owned by
:class:`MultistageCheckerboardLatentCodec`; ``load_state_dict`` rewrites the
upstream top-level entropy keys to live under ``latent_codec.*``.

.. _arXiv:2204.11448: https://arxiv.org/abs/2204.11448
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from timm.models.layers import trunc_normal_

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import MultistageCheckerboardLatentCodec
from compressai.layers import ResViTBlock, conv, deconv
from compressai.models.base import CompressionModel
from compressai.ops import quantize_ste
from compressai.registry import register_model


__all__ = ["TinyLIC"]


_CODEC_PREFIXES = (
    "entropy_parameters_",
    "cc_transforms.",
    "sc_transform_",
    "gaussian_conditional.",
)


@register_model("tinylic")
class TinyLIC(CompressionModel):
    """TinyLIC end-to-end image compression model.

    Args:
        N: Base channel width (default 128). Determines the channel ramp of
            the encoder/decoder/hyper towers (``N``, ``3N/2``, ``2N``).
        M: Latent (``y``) channels (default 320).
    """

    def __init__(self, N: int = 128, M: int = 320):
        super().__init__()

        depths = [2, 2, 6, 2, 2, 2]
        num_heads = [8, 12, 16, 20, 12, 12]
        kernel_size = 7
        mlp_ratio = 2.0
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm

        self.N = int(N)
        self.M = int(M)

        # Stochastic depth schedule (per-block drop-path rate).
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ----- Analysis (g_a) -----
        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a1 = ResViTBlock(
            dim=N, depth=depths[0], num_heads=num_heads[0],
            kernel_size=kernel_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[: depths[0]],
            norm_layer=norm_layer,
        )
        self.g_a2 = conv(N, N * 3 // 2, kernel_size=3, stride=2)
        self.g_a3 = ResViTBlock(
            dim=N * 3 // 2, depth=depths[1], num_heads=num_heads[1],
            kernel_size=kernel_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[sum(depths[:1]) : sum(depths[:2])],
            norm_layer=norm_layer,
        )
        self.g_a4 = conv(N * 3 // 2, N * 2, kernel_size=3, stride=2)
        self.g_a5 = ResViTBlock(
            dim=N * 2, depth=depths[2], num_heads=num_heads[2],
            kernel_size=kernel_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[sum(depths[:2]) : sum(depths[:3])],
            norm_layer=norm_layer,
        )
        self.g_a6 = conv(N * 2, M, kernel_size=3, stride=2)
        self.g_a7 = ResViTBlock(
            dim=M, depth=depths[3], num_heads=num_heads[3],
            kernel_size=kernel_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[sum(depths[:3]) : sum(depths[:4])],
            norm_layer=norm_layer,
        )

        # ----- Hyper-analysis (h_a) -----
        self.h_a0 = conv(M, N * 3 // 2, kernel_size=3, stride=2)
        self.h_a1 = ResViTBlock(
            dim=N * 3 // 2, depth=depths[4], num_heads=num_heads[4],
            kernel_size=kernel_size // 2, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[sum(depths[:4]) : sum(depths[:5])],
            norm_layer=norm_layer,
        )
        self.h_a2 = conv(N * 3 // 2, N * 3 // 2, kernel_size=3, stride=2)
        self.h_a3 = ResViTBlock(
            dim=N * 3 // 2, depth=depths[5], num_heads=num_heads[5],
            kernel_size=kernel_size // 2, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[sum(depths[:5]) : sum(depths[:6])],
            norm_layer=norm_layer,
        )

        # ----- Hyper-synthesis (h_s); reverse depth/head order -----
        depths_r = depths[::-1]
        num_heads_r = num_heads[::-1]
        self.h_s0 = ResViTBlock(
            dim=N * 3 // 2, depth=depths_r[0], num_heads=num_heads_r[0],
            kernel_size=kernel_size // 2, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[: depths_r[0]],
            norm_layer=norm_layer,
        )
        self.h_s1 = deconv(N * 3 // 2, N * 3 // 2, kernel_size=3, stride=2)
        self.h_s2 = ResViTBlock(
            dim=N * 3 // 2, depth=depths_r[1], num_heads=num_heads_r[1],
            kernel_size=kernel_size // 2, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[sum(depths_r[:1]) : sum(depths_r[:2])],
            norm_layer=norm_layer,
        )
        self.h_s3 = deconv(N * 3 // 2, M * 2, kernel_size=3, stride=2)

        # ----- Synthesis (g_s) -----
        self.g_s0 = ResViTBlock(
            dim=M, depth=depths_r[2], num_heads=num_heads_r[2],
            kernel_size=kernel_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[sum(depths_r[:2]) : sum(depths_r[:3])],
            norm_layer=norm_layer,
        )
        self.g_s1 = deconv(M, N * 2, kernel_size=3, stride=2)
        self.g_s2 = ResViTBlock(
            dim=N * 2, depth=depths_r[3], num_heads=num_heads_r[3],
            kernel_size=kernel_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[sum(depths_r[:3]) : sum(depths_r[:4])],
            norm_layer=norm_layer,
        )
        self.g_s3 = deconv(N * 2, N * 3 // 2, kernel_size=3, stride=2)
        self.g_s4 = ResViTBlock(
            dim=N * 3 // 2, depth=depths_r[4], num_heads=num_heads_r[4],
            kernel_size=kernel_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[sum(depths_r[:4]) : sum(depths_r[:5])],
            norm_layer=norm_layer,
        )
        self.g_s5 = deconv(N * 3 // 2, N, kernel_size=3, stride=2)
        self.g_s6 = ResViTBlock(
            dim=N, depth=depths_r[5], num_heads=num_heads_r[5],
            kernel_size=kernel_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[sum(depths_r[:5]) : sum(depths_r[:6])],
            norm_layer=norm_layer,
        )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)

        # ----- Entropy model -----
        self.entropy_bottleneck = EntropyBottleneck(N * 3 // 2)
        self.latent_codec = MultistageCheckerboardLatentCodec(
            channels=M,
            hyper_channels=M * 2,
            num_iters=4,
            gamma_mode="cosine",
        )

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Backbone helpers (kept for parity with upstream API).
    # ------------------------------------------------------------------
    def g_a(self, x: Tensor) -> Tensor:
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a2(x)
        x = self.g_a3(x)
        x = self.g_a4(x)
        x = self.g_a5(x)
        x = self.g_a6(x)
        x = self.g_a7(x)
        return x

    def g_s(self, x: Tensor) -> Tensor:
        x = self.g_s0(x)
        x = self.g_s1(x)
        x = self.g_s2(x)
        x = self.g_s3(x)
        x = self.g_s4(x)
        x = self.g_s5(x)
        x = self.g_s6(x)
        x = self.g_s7(x)
        return x

    def h_a(self, x: Tensor) -> Tensor:
        x = self.h_a0(x)
        x = self.h_a1(x)
        x = self.h_a2(x)
        x = self.h_a3(x)
        return x

    def h_s(self, x: Tensor) -> Tensor:
        x = self.h_s0(x)
        x = self.h_s1(x)
        x = self.h_s2(x)
        x = self.h_s3(x)
        return x

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Dict[str, Any]:
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset

        params = self.h_s(z_hat)
        codec_out = self.latent_codec(y, params)
        x_hat = self.g_s(codec_out["y_hat"])

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": codec_out["likelihoods"]["y"],
                "z": z_likelihoods,
            },
        }

    def compress(self, x: Tensor) -> Dict[str, Any]:
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat)

        codec_out = self.latent_codec.compress(y, params)
        return {
            "strings": [codec_out["strings"][0], z_strings],
            "shape": z.size()[-2:],
        }

    def decompress(
        self, strings: List[List[bytes]], shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        y_strings = strings[0]
        codec_out = self.latent_codec.decompress([y_strings], shape, params)
        x_hat = self.g_s(codec_out["y_hat"]).clamp_(0, 1)
        return {"x_hat": x_hat}

    # ------------------------------------------------------------------
    # State-dict bridge: rewrite upstream top-level entropy keys to live
    # under ``latent_codec.*`` so :func:`from_state_dict` can load
    # checkpoints produced by the upstream training script with no key
    # surgery in the conversion script.
    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict, strict: bool = True):
        remapped: Dict[str, Tensor] = {}
        for key, value in state_dict.items():
            if any(key.startswith(prefix) for prefix in _CODEC_PREFIXES):
                remapped[f"latent_codec.{key}"] = value
            else:
                remapped[key] = value
        return super().load_state_dict(remapped, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "TinyLIC":
        # `g_a0.weight` shape: (N, 3, 5, 5) → N is dim 0.
        N = int(state_dict["g_a0.weight"].size(0))
        # `g_a6.weight` shape: (M, 2N, 3, 3) → M is dim 0.
        M = int(state_dict["g_a6.weight"].size(0))
        net = cls(N=N, M=M)
        net.load_state_dict(state_dict)
        return net
