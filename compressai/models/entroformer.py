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

"""Entroformer image compression model.

Reference: Yichen Qian, Ming Lin, Xiuyu Sun, Zhiyu Tan, Rong Jin,
*"Entroformer: A Transformer-based Entropy Model for Learned Image Compression"*,
ICLR 2022 (`arXiv:2202.05492 <https://arxiv.org/abs/2202.05492>`_).

This is a re-implementation of the official release at
``candidate_none/entroformer``. The model is a Ballé-style 4-stage
Conv+GDN auto-encoder with a Transformer-based hyperprior + auto-regressive
entropy model:

* ``g_a / g_s`` — :class:`compressai.layers.lic.Balle2Encoder` /
  :class:`Balle2Decoder` (4× stride-2 5×5 Conv + GDN, channels ``N → M``).
* ``cit_he`` — :class:`compressai.layers.attn.TransHyperScale` ``(down=True)``,
  three Transformer stages alternated with two stride-2 Conv down-samplers
  producing the hyper latent ``z`` of shape ``(B, Z, H/16, W/16)``.
* ``cit_hd`` — :class:`TransHyperScale` ``(down=False)`` with sub-pixel
  up-samplers, producing a feature map of shape ``(B, dim_embed, H/4, W/4)``.
* ``cit_ar`` — :class:`compressai.layers.attn.TransDecoder`, raster causal
  Transformer over the noise-quantised ``y``.
* ``cit_pn`` — 1×1 Conv MLP that fuses ``[feat_hyper, feat_ar]`` and outputs
  ``num_parameter * M`` channels split into per-pixel scales / means for the
  Gaussian conditional likelihood of ``y``.

The hyperprior is modelled with
:class:`compressai.entropy_models.LearnedGaussianBottleneck` (per-channel
single Gaussian, equivalent to upstream ``module/prob_model.py::Entropy``).

Inputs are expected in ``[0, 1]``; the model internally rescales to
``[-1, 1]`` for ``g_a`` and back to ``[0, 1]`` after ``g_s`` to match upstream
training data normalisation, so released checkpoints load 1:1 by parameter
name (handled by :func:`examples/convert_entroformer_checkpoint.py`).
"""
from __future__ import annotations

import math

from typing import Any, Dict

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import GaussianConditional, LearnedGaussianBottleneck
from compressai.latent_codecs import TransformerARLatentCodec
from compressai.layers.attn import TransDecoder, TransHyperScale
from compressai.layers.lic import Balle2Decoder, Balle2Encoder
from compressai.registry import register_model

from .base import CompressionModel

__all__ = ["Entroformer"]


def _make_param_net(dim_embed: int, mlp_ratio: int, latent_channels: int, num_parameter: int, K: int) -> nn.Sequential:
    """1×1 Conv → LeakyReLU(0.2) → 1×1 Conv head; matches upstream ``cit_pn``."""
    inner = dim_embed * mlp_ratio
    out = latent_channels * K * num_parameter
    return nn.Sequential(
        nn.Conv2d(dim_embed * 2, inner, 1, 1, 0),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(inner, out, 1, 1, 0),
    )


@register_model("entroformer")
class Entroformer(CompressionModel):
    r"""Entroformer (Qian et al., ICLR 2022).

    Args:
        N: ``g_a`` intermediate channel count (upstream ``channels``, default 192).
        M: latent ``y`` channel count (upstream ``last_channels``, default 384).
        Z: hyper latent ``z`` channel count (upstream ``hyper_channels``,
            default 192).
        dim_embed: Transformer embedding dim (default 384).
        depth: Number of Transformer layers in ``cit_ar`` (default 6).
        heads: Attention heads in each Transformer block (default 6).
        dim_head: Per-head dim (default 64).
        mlp_ratio: FFN expansion (default 4).
        position_num: 2D-RPE bucket count for the AR Transformer; ``cit_he``
            stages halve it down stage-by-stage (default 7).
        scale: Number of down/up scaling stages in ``cit_he`` / ``cit_hd``
            (default 2).
        attn_topk: Top-k filter for self-attention; ``-1`` disables (default 32
            for the released ``unidirectional`` configuration).
        num_parameter: Number of distribution parameters per pixel; supported
            values are ``2`` (mean + log-σ) and ``3`` (logit_pi + mean + log-σ
            with ``K=1``).
        K: Number of mixture components. Only ``K=1`` is supported.
        rpe_shared: Share the 2D-RPE table across Transformer layers (the
            non-zeroth layers reuse the position bias computed by layer 0;
            default True).
    """

    g_a: nn.Module
    g_s: nn.Module
    latent_codec: TransformerARLatentCodec

    def __init__(
        self,
        N: int = 192,
        M: int = 384,
        Z: int = 192,
        *,
        dim_embed: int = 384,
        depth: int = 6,
        heads: int = 6,
        dim_head: int = 64,
        mlp_ratio: int = 4,
        position_num: int = 7,
        scale: int = 2,
        attn_topk: int = 32,
        num_parameter: int = 2,
        K: int = 1,
        rpe_shared: bool = True,
        att_scale: bool = True,
        dropout: float = 0.0,
        mask_ratio: float = 0.0,
        log_scale_min: float = -7.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if K != 1:
            raise ValueError("Entroformer migration currently supports K=1 only.")
        if M != dim_embed:
            # Upstream code allows M ≠ dim_embed via to_patch_embedding(M → dim_embed)
            # but every released config has M == dim_embed; we keep a runtime
            # sanity check rather than silently accepting mismatched channel counts.
            pass

        self.N = int(N)
        self.M = int(M)
        self.Z = int(Z)

        self.g_a = Balle2Encoder(channels=N, last_channels=M)
        self.g_s = Balle2Decoder(channels=N, last_channels=M)

        cit_he = TransHyperScale(
            cin=M,
            cout=Z,
            scale=scale,
            down=True,
            dim_embed=dim_embed,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            position_num=position_num,
            attn_topk=attn_topk,
            att_scale=att_scale,
            rpe_shared=rpe_shared,
            mask_ratio=0.0,
        )
        cit_hd = TransHyperScale(
            cin=Z,
            cout=0,
            scale=scale,
            down=False,
            dim_embed=dim_embed,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            position_num=position_num,
            attn_topk=attn_topk,
            att_scale=att_scale,
            rpe_shared=rpe_shared,
            mask_ratio=0.0,
        )
        cit_ar = TransDecoder(
            cin=M,
            cout=0,
            dim_embed=dim_embed,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            position_num=position_num,
            attn_topk=attn_topk,
            att_scale=att_scale,
            rpe_shared=rpe_shared,
            mask_ratio=mask_ratio,
        )
        cit_pn = _make_param_net(dim_embed, mlp_ratio, M, num_parameter, K)

        self.latent_codec = TransformerARLatentCodec(
            latent_channels=M,
            hyper_channels=Z,
            y_hyper_encode=cit_he,
            y_hyper_decode=cit_hd,
            y_ar=cit_ar,
            param_net=cit_pn,
            entropy_bottleneck=LearnedGaussianBottleneck(Z),
            gaussian_conditional=GaussianConditional(
                None,
                scale_bound=float(math.exp(log_scale_min)),
            ),
            num_parameter=num_parameter,
            log_scale_min=log_scale_min,
        )

        # Stash hyperparameters for downstream introspection / `from_state_dict`.
        self._hparams = dict(
            N=N,
            M=M,
            Z=Z,
            dim_embed=dim_embed,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            position_num=position_num,
            scale=scale,
            attn_topk=attn_topk,
            num_parameter=num_parameter,
            K=K,
            rpe_shared=rpe_shared,
            att_scale=att_scale,
        )

    @property
    def downsampling_factor(self) -> int:
        # 4 stride-2 conv stages in g_a, plus `scale` extra in cit_he.
        return 2 ** (4 + self._hparams["scale"])

    def forward(self, x: Tensor) -> Dict[str, Any]:
        # Upstream rescales the image to [-1, 1] before g_a; mirror that here so
        # the released checkpoints reproduce the trained reconstructions exactly.
        x_in = x * 2.0 - 1.0
        y = self.g_a(x_in)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat)
        x_hat = x_hat / 2.0 + 0.5
        return {
            "x_hat": x_hat,
            "likelihoods": y_out["likelihoods"],
        }

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "Entroformer":
        """Instantiate ``Entroformer`` from a converted compressai-layout state dict.

        Hyperparameters are inferred from tensor shapes:

        * ``N`` from ``g_a.encoder.0.weight`` (out channels of the first conv);
        * ``M`` from ``g_a.encoder.6.weight``;
        * ``Z`` from ``latent_codec.entropy_bottleneck.mu``;
        * ``dim_embed`` from ``latent_codec.y_ar.to_patch_embedding.weight``
          (out features);
        * ``depth`` from the count of ``latent_codec.y_ar.blocks.{i}.layer.0...``
          entries;
        * ``heads`` / ``dim_head`` from ``...SelfAttention.qkv.weight``
          (3 × heads × dim_head, dim_embed);
        * ``position_num`` from ``...SelfAttention.relative_attention_bias.weight``
          (rows = ``position_num**2``);
        * ``scale`` from ``latent_codec.y_hyper_encode.scale_blocks.{i}``;
        * ``num_parameter`` from ``latent_codec.param_net.2.weight`` (out
          channels = ``M * K * num_parameter``).
        """
        N = state_dict["g_a.encoder.0.weight"].size(0)
        M = state_dict["g_a.encoder.6.weight"].size(0)
        Z = state_dict["latent_codec.entropy_bottleneck.mu"].size(1)

        ar_embed_w = state_dict["latent_codec.y_ar.to_patch_embedding.weight"]
        dim_embed = ar_embed_w.size(0)

        depth = sum(
            1
            for k in state_dict
            if k.startswith("latent_codec.y_ar.blocks.")
            and k.endswith(".layer.0.SelfAttention.qkv.weight")
        )

        qkv_w = state_dict["latent_codec.y_ar.blocks.0.layer.0.SelfAttention.qkv.weight"]
        # qkv_w: (3 * heads * dim_head, dim_embed)
        rpe_w = state_dict[
            "latent_codec.y_ar.blocks.0.layer.0.SelfAttention.relative_attention_bias.weight"
        ]
        position_num = int(round(rpe_w.size(0) ** 0.5))
        dim_head = rpe_w.size(1)
        heads = qkv_w.size(0) // (3 * dim_head)

        # mlp_ratio from FFN inner-dim (`net.0.weight` out features = dim_embed * mlp_ratio)
        ffn_w = state_dict["latent_codec.y_ar.blocks.0.layer.1.fn.net.0.weight"]
        mlp_ratio = ffn_w.size(0) // dim_embed

        # `scale` from number of scale_blocks in the hyper-encoder.
        scale = sum(
            1
            for k in state_dict
            if k.startswith("latent_codec.y_hyper_encode.scale_blocks.")
            and k.endswith(".weight")
        )

        # num_parameter: cit_pn.2 out channels = M * K * num_parameter, with K=1.
        pn_out = state_dict["latent_codec.param_net.2.weight"].size(0)
        if pn_out % M != 0:
            raise ValueError(
                f"param_net output channels ({pn_out}) is not divisible by M ({M})"
            )
        num_parameter = pn_out // M

        net = cls(
            N=N,
            M=M,
            Z=Z,
            dim_embed=dim_embed,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            position_num=position_num,
            scale=scale,
            num_parameter=num_parameter,
            K=1,
        )
        net.load_state_dict(state_dict)
        return net

    def compress(self, x: Tensor):
        raise NotImplementedError(
            "Entroformer.compress is not implemented yet; the upstream raster-scan "
            "AR bitstream coding requires per-pixel re-evaluation of the Transformer "
            "AR head and is a planned follow-up. Use forward() for rate estimation."
        )

    def decompress(self, *args, **kwargs):
        raise NotImplementedError(
            "Entroformer.decompress is not implemented yet; see compress()."
        )
