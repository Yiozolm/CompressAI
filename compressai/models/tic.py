# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# This file adapts code from https://github.com/lumingzzz/TIC
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

"""TIC: Transformer-based Image Compression.

Lu, Guo, Shi, Cao & Ma, `"Transformer-based Image Compression"
<https://arxiv.org/abs/2111.06707>`_ (DCC 2022).

Architecture: Swin-RSTB encoder/decoder + Minnen2018-style joint
hyperprior with a :class:`CausalAttentionModule` (CAM) replacing the usual
:class:`MaskedConv2d` context model. Module / parameter names
(``g_a0`` / ``g_a1`` / ... / ``context_prediction`` / ``entropy_parameters``)
are preserved verbatim from upstream so checkpoints load directly via
:meth:`from_state_dict`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import trunc_normal_
from torch import Tensor

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import RSTB, CausalAttentionModule, conv, deconv
from compressai.models.base import CompressionModel
from compressai.registry import register_model


__all__ = ["TIC"]


@register_model("tic")
class TIC(CompressionModel):
    """TIC end-to-end image compression model.

    Args:
        N: Backbone channel width (default 128).
        M: Latent (``y``) channels (default 192).
    """

    def __init__(self, N: int = 128, M: int = 192) -> None:
        super().__init__()

        depths = [2, 4, 6, 2, 2]
        num_heads = [4, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 4.0
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_path_rate = 0.2
        norm_layer = nn.LayerNorm
        use_checkpoint = False

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.N = int(N)
        self.M = int(M)

        # ----- Analysis (g_a) -----
        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a1 = RSTB(
            dim=N, input_resolution=(128, 128),
            depth=depths[0], num_heads=num_heads[0],
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[: depths[0]],
            norm_layer=norm_layer, use_checkpoint=use_checkpoint,
        )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = RSTB(
            dim=N, input_resolution=(64, 64),
            depth=depths[1], num_heads=num_heads[1],
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:1]) : sum(depths[:2])],
            norm_layer=norm_layer, use_checkpoint=use_checkpoint,
        )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(
            dim=N, input_resolution=(32, 32),
            depth=depths[2], num_heads=num_heads[2],
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:2]) : sum(depths[:3])],
            norm_layer=norm_layer, use_checkpoint=use_checkpoint,
        )
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)

        # ----- Hyper-analysis (h_a) -----
        self.h_a0 = conv(M, N, kernel_size=3, stride=1)
        self.h_a1 = RSTB(
            dim=N, input_resolution=(16, 16),
            depth=depths[3], num_heads=num_heads[3],
            window_size=window_size // 2, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:3]) : sum(depths[:4])],
            norm_layer=norm_layer, use_checkpoint=use_checkpoint,
        )
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = RSTB(
            dim=N, input_resolution=(8, 8),
            depth=depths[4], num_heads=num_heads[4],
            window_size=window_size // 2, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:4]) : sum(depths[:5])],
            norm_layer=norm_layer, use_checkpoint=use_checkpoint,
        )
        self.h_a4 = conv(N, N, kernel_size=3, stride=2)

        # Hyper-synthesis / synthesis: reverse the depth/head schedule.
        depths_r = depths[::-1]
        num_heads_r = num_heads[::-1]

        # ----- Hyper-synthesis (h_s) -----
        self.h_s0 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s1 = RSTB(
            dim=N, input_resolution=(8, 8),
            depth=depths_r[0], num_heads=num_heads_r[0],
            window_size=window_size // 2, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[: depths_r[0]],
            norm_layer=norm_layer, use_checkpoint=use_checkpoint,
        )
        self.h_s2 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s3 = RSTB(
            dim=N, input_resolution=(16, 16),
            depth=depths_r[1], num_heads=num_heads_r[1],
            window_size=window_size // 2, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths_r[:1]) : sum(depths_r[:2])],
            norm_layer=norm_layer, use_checkpoint=use_checkpoint,
        )
        self.h_s4 = conv(N, M * 2, kernel_size=3, stride=1)

        # ----- Synthesis (g_s) -----
        self.g_s0 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s1 = RSTB(
            dim=N, input_resolution=(32, 32),
            depth=depths_r[2], num_heads=num_heads_r[2],
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths_r[:2]) : sum(depths_r[:3])],
            norm_layer=norm_layer, use_checkpoint=use_checkpoint,
        )
        self.g_s2 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s3 = RSTB(
            dim=N, input_resolution=(64, 64),
            depth=depths_r[3], num_heads=num_heads_r[3],
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths_r[:3]) : sum(depths_r[:4])],
            norm_layer=norm_layer, use_checkpoint=use_checkpoint,
        )
        self.g_s4 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s5 = RSTB(
            dim=N, input_resolution=(128, 128),
            depth=depths_r[4], num_heads=num_heads_r[4],
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths_r[:4]) : sum(depths_r[:5])],
            norm_layer=norm_layer, use_checkpoint=use_checkpoint,
        )
        self.g_s6 = deconv(N, 3, kernel_size=5, stride=2)

        # ----- Entropy model -----
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        self.context_prediction = CausalAttentionModule(M, M * 2)

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.GELU(),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.GELU(),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Backbone helpers (the RSTB blocks need an explicit x_size since the
    # attention masks depend on it).
    # ------------------------------------------------------------------
    def g_a(self, x: Tensor, x_size: Tuple[int, int]) -> Tensor:
        x = self.g_a0(x)
        x = self.g_a1(x, (x_size[0] // 2, x_size[1] // 2))
        x = self.g_a2(x)
        x = self.g_a3(x, (x_size[0] // 4, x_size[1] // 4))
        x = self.g_a4(x)
        x = self.g_a5(x, (x_size[0] // 8, x_size[1] // 8))
        x = self.g_a6(x)
        return x

    def g_s(self, x: Tensor, x_size: Tuple[int, int]) -> Tensor:
        x = self.g_s0(x)
        x = self.g_s1(x, (x_size[0] // 8, x_size[1] // 8))
        x = self.g_s2(x)
        x = self.g_s3(x, (x_size[0] // 4, x_size[1] // 4))
        x = self.g_s4(x)
        x = self.g_s5(x, (x_size[0] // 2, x_size[1] // 2))
        x = self.g_s6(x)
        return x

    def h_a(self, x: Tensor, x_size: Tuple[int, int]) -> Tensor:
        x = self.h_a0(x)
        x = self.h_a1(x, (x_size[0] // 16, x_size[1] // 16))
        x = self.h_a2(x)
        x = self.h_a3(x, (x_size[0] // 32, x_size[1] // 32))
        x = self.h_a4(x)
        return x

    def h_s(self, x: Tensor, x_size: Tuple[int, int]) -> Tensor:
        x = self.h_s0(x)
        x = self.h_s1(x, (x_size[0] // 32, x_size[1] // 32))
        x = self.h_s2(x)
        x = self.h_s3(x, (x_size[0] // 16, x_size[1] // 16))
        x = self.h_s4(x)
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
        x_size = (x.shape[2], x.shape[3])
        y = self.g_a(x, x_size)
        z = self.h_a(y, x_size)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat, x_size)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, x_size)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "TIC":
        N = int(state_dict["g_a0.weight"].size(0))
        M = int(state_dict["g_a6.weight"].size(0))
        net = cls(N=N, M=M)
        net.load_state_dict(state_dict)
        return net

    # ------------------------------------------------------------------
    # Sequential autoregressive bitstream coding (slow CPU path; same
    # structure as :class:`JointAutoregressiveHierarchicalPriors._compress_ar`
    # but the per-pixel context comes from the CAM 5×5 window self-attention
    # instead of a masked convolution).
    # ------------------------------------------------------------------
    def compress(self, x: Tensor) -> Dict[str, Any]:
        x_size = (x.shape[2], x.shape[3])
        y = self.g_a(x, x_size)
        z = self.h_a(y, x_size)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat, x_size)

        kernel_size = 5
        padding = kernel_size // 2

        s = 4  # spatial scaling between z and y
        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat_padded = F.pad(y, (padding, padding, padding, padding))

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        y_strings: List[bytes] = []
        for i in range(y.size(0)):
            encoder = BufferedRansEncoder()
            symbols_list: List[int] = []
            indexes_list: List[int] = []
            for h in range(y_height):
                for w in range(y_width):
                    y_crop = y_hat_padded[
                        i : i + 1, :, h : h + kernel_size, w : w + kernel_size
                    ]
                    ctx_p = self.context_prediction(y_crop)
                    p = params[i : i + 1, :, h : h + 1, w : w + 1]
                    gaussian_params = self.entropy_parameters(
                        torch.cat((p, ctx_p[:, :, padding : padding + 1, padding : padding + 1]), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    indexes = self.gaussian_conditional.build_indexes(scales_hat)
                    y_q = torch.round(y_crop[:, :, padding, padding] - means_hat.squeeze(-1).squeeze(-1))
                    y_hat_padded[i, :, h + padding, w + padding] = (
                        y_q + means_hat.squeeze(-1).squeeze(-1)
                    ).squeeze(0)

                    symbols_list.extend(y_q.squeeze().int().tolist())
                    indexes_list.extend(indexes.squeeze().int().tolist())

            encoder.encode_with_indexes(
                symbols_list, indexes_list, cdf, cdf_lengths, offsets
            )
            y_strings.append(encoder.flush())

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(
        self, strings: List[List[bytes]], shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        # Recover x_size from z_hat (z is downsampled by 64 from input).
        x_size = (z_hat.size(2) * 64, z_hat.size(3) * 64)
        params = self.h_s(z_hat, x_size)

        kernel_size = 5
        padding = kernel_size // 2

        s = 4
        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
            dtype=z_hat.dtype,
        )

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        for i, y_string in enumerate(strings[0]):
            decoder = RansDecoder()
            decoder.set_stream(y_string)

            for h in range(y_height):
                for w in range(y_width):
                    y_crop = y_hat[
                        i : i + 1, :, h : h + kernel_size, w : w + kernel_size
                    ]
                    ctx_p = self.context_prediction(y_crop)
                    p = params[i : i + 1, :, h : h + 1, w : w + 1]
                    gaussian_params = self.entropy_parameters(
                        torch.cat((p, ctx_p[:, :, padding : padding + 1, padding : padding + 1]), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    indexes = self.gaussian_conditional.build_indexes(scales_hat)
                    rv = decoder.decode_stream(
                        indexes.squeeze().int().tolist(),
                        cdf,
                        cdf_lengths,
                        offsets,
                    )
                    rv = torch.tensor(rv, dtype=y_hat.dtype).reshape(1, -1, 1, 1)
                    rv = self.gaussian_conditional.dequantize(rv, means_hat)

                    y_hat[
                        i,
                        :,
                        h + padding : h + padding + 1,
                        w + padding : w + padding + 1,
                    ] = rv.squeeze(0)

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat, x_size).clamp_(0, 1)
        return {"x_hat": x_hat}
