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

"""Transformer blocks used by the Entroformer image-compression entropy model.

Re-implementation of ``module/entroformer_helper.py`` from the official
`Entroformer (ICLR 2022) <https://arxiv.org/abs/2202.05492>`_ release.
Kept self-contained (does not pull from ``timm``) so that the upstream
checkpoints load 1:1 by name. The 2D *contextual-product* relative position
encoding has no equivalent in the existing :mod:`compressai.layers.attn`
modules (Swin uses windowed-shifted MSA with bias-style RPE; cross-attention
modules have no self-RPE).
"""

from dataclasses import dataclass

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torch import Tensor


__all__ = [
    "EntroformerConfig",
    "EntroformerAttention",
    "EntroformerAttentionBlock",
    "EntroformerFeedForward",
    "EntroformerPreNorm",
    "EntroformerBlock",
    "EntroformerUpPixelShuffle",
    "TransDecoder",
    "TransDecoder2",
    "TransHyperScale",
]


@dataclass
class EntroformerConfig:
    dim: int = 384
    num_layers: int = 6
    num_heads: int = 6
    dim_head: int = 64
    relative_attention_num_buckets: int = 5  # must be odd
    dropout_rate: float = 0.0
    scale: bool = True
    mlp_ratio: int = 4
    is_decoder: bool = True
    rpe_mode: str = "contextualproduct"
    attn_topk: int = -1


class EntroformerPreNorm(nn.Module):
    """LayerNorm applied before ``fn``; matches upstream ``PreNorm``."""

    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.fn(self.norm(x), **kwargs)


class EntroformerFeedForward(nn.Module):
    """Two-layer MLP with LeakyReLU(0.2) (upstream uses LeakyReLU, not GELU)."""

    def __init__(self, dim: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        # `net.0` / `net.2` indexing matches upstream state_dict key names.
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EntroformerAttention(nn.Module):
    """Self-attention with 2D contextual-product relative position encoding.

    The RPE table is an :class:`nn.Embedding` of size
    ``num_buckets**2 × dim_head`` indexed by the 2D L1-distance bucket.
    Distances above ``num_buckets // 2`` collapse to bucket 0 (the upstream
    "if_small" mask). When ``has_relative_attention_bias`` is False the
    position bias passed in by the wrapping :class:`EntroformerBlock` is
    reused (RPE sharing across layers).
    """

    def __init__(
        self,
        config: EntroformerConfig,
        has_relative_attention_bias: bool = False,
    ) -> None:
        super().__init__()
        assert config.relative_attention_num_buckets % 2 == 1
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.dim = config.dim
        self.key_value_proj_dim = config.dim_head
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.scale = self.dim**-0.5 if config.scale else 1.0

        # Single qkv projection (no bias) keeps key names aligned with upstream.
        self.qkv = nn.Linear(self.dim, self.inner_dim * 3, bias=False)
        self.o = nn.Linear(self.inner_dim, self.dim, bias=False)
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets**2, self.key_value_proj_dim
            )

    def compute_bias(
        self,
        query_length: Tuple[int, int],
        key_length: Tuple[int, int],
    ) -> Tensor:
        num_buckets = self.relative_attention_num_buckets
        num_buckets_half = num_buckets // 2

        ctx_v = torch.arange(query_length[0], dtype=torch.long)[:, None]
        mem_v = torch.arange(key_length[0], dtype=torch.long)[None, :]
        rel_v = mem_v - ctx_v  # (Hq, Hk)
        ctx_h = torch.arange(query_length[1], dtype=torch.long)[:, None]
        mem_h = torch.arange(key_length[1], dtype=torch.long)[None, :]
        rel_h = mem_h - ctx_h  # (Wq, Wk)

        rel_v = rel_v.repeat(query_length[1], key_length[1]).view(
            query_length[1], query_length[0], key_length[1], key_length[0]
        )
        rel_v = rel_v.permute(1, 0, 3, 2).contiguous().view(
            query_length[0] * query_length[1], -1
        )
        rel_h = rel_h.repeat(query_length[0], key_length[0]).view(
            query_length[0] * query_length[1], -1
        )

        hamming = rel_h.abs() + rel_v.abs()
        is_small = hamming <= num_buckets_half
        zero = torch.full_like(rel_v, 0)
        buckets = (rel_v + num_buckets_half) * num_buckets + (rel_h + num_buckets_half)
        buckets = torch.where(is_small, buckets, zero)
        buckets = buckets.to(self.relative_attention_bias.weight.device)
        return self.relative_attention_bias(buckets)  # (Sq, Sk, dim_head)

    def forward(
        self,
        hidden_states: Tensor,
        query_shape_2d: Tuple[int, int],
        key_shape_2d: Tuple[int, int],
        mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        topk: int = -1,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, seq_length = hidden_states.shape[:2]

        def shape(states: Tensor) -> Tensor:
            return states.view(
                batch_size, -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states: Tensor) -> Tensor:
            return (
                states.transpose(1, 2)
                .contiguous()
                .view(batch_size, -1, self.inner_dim)
            )

        qkv = self.qkv(hidden_states).reshape(batch_size, -1, 3)
        query_states = shape(qkv[..., 0])
        key_states = shape(qkv[..., 1])
        value_states = shape(qkv[..., 2])

        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            position_bias = self.compute_bias(query_shape_2d, key_shape_2d)

        # Contextual product: per-query reweighting of the position table.
        rearr_q = rearrange(query_states, "b n q d -> q (b n) d")
        contextual_position = torch.matmul(rearr_q, position_bias.transpose(1, 2))
        contextual_position = rearrange(
            contextual_position, "q (b n) k -> b n q k", b=batch_size
        )
        scores = scores + contextual_position
        scores = scores * self.scale

        if mask is not None:
            mask_value = -torch.finfo(scores.dtype).max
            assert mask.shape[-1] == scores.shape[-1], "mask has incorrect dimensions"
            mask = rearrange(mask, "b i j -> b () i j")
            scores = scores.masked_fill(~mask, mask_value)

        if topk != -1:
            real_seq_length = seq_length
            values_topk, _ = scores.topk(
                min(topk, real_seq_length), dim=-1, largest=True, sorted=True
            )
            thres = repeat(
                values_topk[..., -1:], "b h i () -> b h i j", j=real_seq_length
            )
            topk_mask = scores >= thres
            scores = scores.masked_fill(
                ~topk_mask, -torch.finfo(scores.dtype).max
            )

        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = unshape(torch.matmul(attn_weights, value_states))
        attn_output = self.o(attn_output)
        return attn_output, position_bias


class EntroformerAttentionBlock(nn.Module):
    """LayerNorm + self-attention + residual; matches upstream ``AttentionBlock``."""

    def __init__(
        self,
        config: EntroformerConfig,
        has_relative_attention_bias: bool = False,
    ) -> None:
        super().__init__()
        self.SelfAttention = EntroformerAttention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: Tensor,
        shape_2d: Tuple[int, int],
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        topk: int = -1,
    ) -> Tuple[Tensor, Tensor]:
        normed = self.layer_norm(hidden_states)
        attn_out, pb = self.SelfAttention(
            normed,
            shape_2d,
            shape_2d,
            mask=attention_mask,
            position_bias=position_bias,
            topk=topk,
        )
        hidden_states = hidden_states + self.dropout(attn_out)
        return hidden_states, pb


class EntroformerBlock(nn.Module):
    """Self-attention + PreNorm-FFN block; submodules registered as ``layer.0`` /
    ``layer.1`` to match upstream state_dict keys."""

    def __init__(
        self,
        config: EntroformerConfig,
        has_relative_attention_bias: bool = False,
    ) -> None:
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            EntroformerAttentionBlock(
                config, has_relative_attention_bias=has_relative_attention_bias
            )
        )
        self.layer.append(
            EntroformerPreNorm(
                config.dim,
                EntroformerFeedForward(
                    config.dim, config.mlp_ratio, config.dropout_rate
                ),
            )
        )

    def forward(
        self,
        hidden_states: Tensor,
        shape_2d: Tuple[int, int],
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        topk: int = -1,
    ) -> Tuple[Tensor, Tensor]:
        hidden_states, pb = self.layer[0](
            hidden_states,
            shape_2d,
            attention_mask=attention_mask,
            position_bias=position_bias,
            topk=topk,
        )
        # PreNorm + residual on FFN
        hidden_states = hidden_states + self.layer[-1](hidden_states)
        return hidden_states, pb

    def compute_bias(self, shape_2d: Tuple[int, int]) -> Tensor:
        return self.layer[0].SelfAttention.compute_bias(shape_2d, shape_2d)


def entroformer_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Produce ``n`` identical layers via :func:`copy.deepcopy`."""
    import copy

    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class EntroformerUpPixelShuffle(nn.Module):
    """Sub-pixel up-sampler used by Entroformer's transformer hyper-decoder.

    Module attribute names ``conv2d`` / ``up`` are kept verbatim from upstream
    ``module/ops.py::UpPixelShuffle`` so checkpoints load by key name.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        scale: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * (scale**2),
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="zeros",
            groups=groups,
        )
        self.up = nn.PixelShuffle(scale)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.conv2d(x))


class TransDecoder(nn.Module):
    """Causal-mask Transformer auto-regressive entropy model.

    Used by Entroformer for the ``cit_ar`` branch. The default scan mode is
    raster (``torch.tril`` causal mask); ``train_scan_mode='random'`` enables
    upstream's MLM-style pre-training mask, but is *not* required for inference
    or for loading the released checkpoints.
    """

    debug: bool = False
    train_scan_mode: str = "default"  # 'default' or 'random'
    test_scan_mode: str = "default"
    manual_init_bias: bool = True
    is_decoder: bool = True
    rpe_mode: str = "contextualproduct"

    def __init__(
        self,
        cin: int = 0,
        cout: int = 0,
        *,
        dim_embed: int = 384,
        depth: int = 6,
        heads: int = 6,
        dim_head: int = 64,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        position_num: int = 7,
        attn_topk: int = -1,
        att_scale: bool = True,
        rpe_shared: bool = True,
        mask_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.rpe_shared = rpe_shared
        self.mask_ratio = mask_ratio
        self.dim = dim_embed
        self.num_layers = depth
        self.num_heads = heads
        self.dim_head = dim_head
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.position_num = position_num
        self.attn_topk = attn_topk
        self.att_scale = att_scale

        self.config = EntroformerConfig(
            dim=self.dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dim_head=self.dim_head,
            relative_attention_num_buckets=self.position_num,
            dropout_rate=self.dropout,
            scale=self.att_scale,
            mlp_ratio=self.mlp_ratio,
            is_decoder=self.is_decoder,
            rpe_mode=self.rpe_mode,
            attn_topk=self.attn_topk,
        )
        self.build()

    def build(self) -> None:
        self.to_patch_embedding = (
            nn.Linear(self.cin, self.config.dim) if self.cin else nn.Identity()
        )
        if self.cout:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.config.dim),
                nn.Linear(self.config.dim, self.cout),
            )
            self.sos_pred_token = nn.Parameter(torch.randn(1, 1, self.cout))
        else:
            self.mlp_head = nn.Identity()
            self.sos_pred_token = nn.Parameter(torch.randn(1, 1, self.config.dim))

        if self.rpe_shared:
            self.blocks = nn.ModuleList(
                [
                    EntroformerBlock(
                        self.config, has_relative_attention_bias=bool(i == 0)
                    )
                    for i in range(self.config.num_layers)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    EntroformerBlock(self.config, has_relative_attention_bias=True)
                    for _ in range(self.config.num_layers)
                ]
            )

        if self.mask_ratio > 0:
            self.sampler = torch.distributions.uniform.Uniform(0.0, 1.0)

    def forward(
        self, x: Tensor, manual_mask: Optional[Tuple[Tensor, ...]] = None
    ) -> Tensor:
        x = x.clone()
        batch_size, _, height, width = x.shape

        if manual_mask is None:
            mask, _, input_mask, output_mask = self.get_mask(batch_size, height, width)
        else:
            mask, _, input_mask, output_mask = manual_mask
        mask = mask.to(x.device)
        input_mask = input_mask.to(x.device)
        output_mask = output_mask.to(x.device)

        x = x.masked_fill(~input_mask, 0.0)
        x = rearrange(x, "b c h w -> b (h w) c")
        inputs_embeds = self.to_patch_embedding(x)

        position_bias: Optional[Tensor] = None
        hidden_states = inputs_embeds

        topk = self.attn_topk
        if self.training and topk != -1:
            topk = int(np.random.randint(topk // 2, topk * 2))

        for layer in self.blocks:
            hidden_states, pb = layer(
                hidden_states,
                shape_2d=(height, width),
                attention_mask=mask,
                position_bias=position_bias,
                topk=topk,
            )
            if self.rpe_shared:
                position_bias = pb

        out = self.mlp_head(hidden_states)
        # Right-shift output by sos token for causal/AR scan.
        if hasattr(self, "sos_pred_token"):
            sos = repeat(self.sos_pred_token, "() n d -> b n d", b=batch_size)
            out = torch.cat((sos, out[:, :-1, :]), dim=1)
        out = rearrange(out, "b (h w) c -> b c h w", h=height)
        out = out.masked_fill(~output_mask, 0.0)
        return out

    def get_mask(
        self, b: int, h: int, w: int
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        n = h * w
        if self.training and self.train_scan_mode == "random" and hasattr(self, "sampler"):
            mask_random = (self.sampler.sample([n]) > self.mask_ratio).bool()
            input_mask = mask_random.clone().view(h, w)
            mask = (
                repeat(mask_random.unsqueeze(0), "() n -> d n", d=n)
                & torch.tril(torch.ones((n, n))).bool()
                | torch.eye(n).bool()
            )
            output_mask = torch.cat(
                (torch.ones(1).bool(), mask_random.clone()[:-1]), 0
            ).view(h, w)
        else:
            mask = torch.tril(torch.ones((n, n))).bool()
            input_mask = torch.ones(h, w).bool()
            output_mask = torch.ones(h, w).bool()

        token_mask = None
        mask = repeat(mask.unsqueeze(0), "() d n -> b d n", b=b)
        input_mask = repeat(
            input_mask.unsqueeze(0).unsqueeze(0),
            "() () h w -> b d h w",
            b=b,
            d=self.cin,
        )
        channel = self.dim if self.cout == 0 else self.cout
        output_mask = repeat(
            output_mask.unsqueeze(0).unsqueeze(0),
            "() () h w -> b d h w",
            b=b,
            d=channel,
        )
        return mask, token_mask, input_mask, output_mask


class TransDecoder2(TransDecoder):
    """Bidirectional / checkerboard Transformer entropy model.

    Used by Entroformer's ``na='bidirectional'`` variant. Splits the latent
    grid into two halves on a checkerboard pattern; the second pass conditions
    on already-decoded anchor positions.
    """

    train_scan_mode: str = "default"
    test_scan_mode: str = "checkboard"
    is_decoder: bool = False

    def __init__(self, cin: int = 0, cout: int = 0, **kwargs) -> None:
        super().__init__(cin, cout, **kwargs)
        # Bidirectional model has no SOS token.
        del self.sos_pred_token

    def forward(
        self, x: Tensor, manual_mask: Optional[Tuple[Tensor, ...]] = None
    ) -> Tensor:
        x = x.clone()
        batch_size, _, height, width = x.shape

        if manual_mask is None:
            mask, _, input_mask, output_mask = self.get_mask(batch_size, height, width)
        else:
            mask, _, input_mask, output_mask = manual_mask
        mask = mask.to(x.device)
        input_mask = input_mask.to(x.device)
        output_mask = output_mask.to(x.device)

        x = x.masked_fill(~input_mask, 0.0)
        x = rearrange(x, "b c h w -> b (h w) c")
        inputs_embeds = self.to_patch_embedding(x)

        position_bias: Optional[Tensor] = None
        hidden_states = inputs_embeds

        topk = self.attn_topk
        if self.training and topk != -1:
            topk = int(np.random.randint(topk // 2, topk * 2))

        for layer in self.blocks:
            hidden_states, pb = layer(
                hidden_states,
                shape_2d=(height, width),
                attention_mask=mask,
                position_bias=position_bias,
                topk=topk,
            )
            if self.rpe_shared:
                position_bias = pb

        out = self.mlp_head(hidden_states)
        out = rearrange(out, "b (h w) c -> b c h w", h=height)
        out = out.masked_fill(~output_mask, 0.0)
        return out

    def get_mask(
        self, b: int, h: int, w: int
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        n = h * w
        if self.training:
            if self.train_scan_mode == "random" and hasattr(self, "sampler"):
                mask_random = (self.sampler.sample([n]) > self.mask_ratio).bool()
                input_mask = mask_random.clone().view(h, w)
                output_mask = ~mask_random.clone().view(h, w)
                mask = repeat(mask_random.unsqueeze(0), "() n -> d n", d=n)
                mask = mask | torch.eye(n).bool()
            else:
                mask_cb = torch.ones((h, w)).bool()
                mask_cb[0::2, 0::2] = 0
                mask_cb[1::2, 1::2] = 0
                input_mask = mask_cb.clone()
                output_mask = ~mask_cb.clone()
                mask = repeat(mask_cb.view(1, -1), "() n -> d n", d=n)
                mask = mask | torch.eye(n).bool()
        else:
            if "checkboard" in self.test_scan_mode:
                mask_cb = torch.ones((h, w)).bool()
                if self.test_scan_mode == "checkboard":
                    mask_cb[0::2, 0::2] = 0
                    mask_cb[1::2, 1::2] = 0
                else:
                    mask_cb[0::2, 1::2] = 0
                    mask_cb[1::2, 0::2] = 0
                input_mask = mask_cb.clone()
                output_mask = ~mask_cb.clone()
                mask = repeat(mask_cb.view(1, -1), "() n -> d n", d=n)
                mask = mask | torch.eye(n).bool()
            else:
                raise ValueError(f"unknown test scan mode: {self.test_scan_mode!r}")

        token_mask = None
        mask = repeat(mask.unsqueeze(0), "() d n -> b d n", b=b)
        input_mask = repeat(
            input_mask.unsqueeze(0).unsqueeze(0),
            "() () h w -> b d h w",
            b=b,
            d=self.cin,
        )
        channel = self.dim if self.cout == 0 else self.cout
        output_mask = repeat(
            output_mask.unsqueeze(0).unsqueeze(0),
            "() () h w -> b d h w",
            b=b,
            d=channel,
        )
        return mask, token_mask, input_mask, output_mask


class TransHyperScale(TransDecoder):
    """Multi-scale Transformer used by Entroformer for the hyperprior branch.

    ``down=True`` builds the encoder (alternating Transformer stages with
    stride-2 conv down-samplers); ``down=False`` builds the decoder
    (alternating stages with sub-pixel up-samplers). The 2D-RPE bucket size
    halves each stage to fit the smaller spatial resolution of the
    hyper-latent.
    """

    is_decoder: bool = False

    def __init__(
        self,
        cin: int = 0,
        cout: int = 0,
        scale: int = 2,
        down: bool = True,
        **kwargs,
    ) -> None:
        self.scale = scale
        self.down = down
        super().__init__(cin, cout, **kwargs)

    def build(self) -> None:
        self.to_patch_embedding = (
            nn.Linear(self.cin, self.config.dim) if self.cin else nn.Identity()
        )
        self.mlp_head = (
            nn.Sequential(
                nn.LayerNorm(self.config.dim), nn.Linear(self.config.dim, self.cout)
            )
            if self.cout
            else nn.Identity()
        )

        if self.down:
            self.scale_blocks = entroformer_clones(
                nn.Conv2d(self.config.dim, self.config.dim, 3, 2, 1, groups=1),
                self.scale,
            )
        else:
            self.scale_blocks = entroformer_clones(
                EntroformerUpPixelShuffle(
                    self.config.dim, self.config.dim, kernel_size=3, scale=2
                ),
                self.scale,
            )

        self.trans_blocks = nn.ModuleList()
        num_each_stage = self.config.num_layers // 2 // (self.scale + 1)
        for _ in range(self.scale + 1):
            if self.rpe_shared:
                stage = nn.ModuleList(
                    [
                        EntroformerBlock(
                            self.config, has_relative_attention_bias=bool(i == 0)
                        )
                        for i in range(num_each_stage)
                    ]
                )
            else:
                stage = nn.ModuleList(
                    [
                        EntroformerBlock(
                            self.config, has_relative_attention_bias=True
                        )
                        for _ in range(num_each_stage)
                    ]
                )
            self.trans_blocks.append(stage)
            # Halve the RPE bucket count for the next (smaller) stage; clamp to ≥5.
            next_num = self.config.relative_attention_num_buckets // 2
            next_num = next_num if next_num % 2 == 1 else next_num + 1
            self.config.relative_attention_num_buckets = max(next_num, 5)

        if not self.down:
            self.trans_blocks = self.trans_blocks[::-1]

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, height, width = x.shape

        mask_list, _, _, _ = self.get_mask(batch_size, height, width)
        mask_list = [m.to(x.device) for m in mask_list]

        x = rearrange(x, "b c h w -> b (h w) c")
        inputs_embeds = self.to_patch_embedding(x)
        hidden_states = inputs_embeds

        topk = self.attn_topk
        if topk != -1:
            if self.training:
                topk = int(np.random.randint(topk // 2, topk * 2))
            topk_list: List[int] = [topk // (2**i) for i in range(self.scale + 1)]
            topk_list = list(np.clip(topk_list, a_min=2, a_max=None))
            if not self.down:
                topk_list = topk_list[::-1]
        else:
            topk_list = [-1 for _ in range(self.scale + 1)]

        for i, scale_layer in enumerate(self.scale_blocks):
            position_bias: Optional[Tensor] = None
            for layer in self.trans_blocks[i]:
                hidden_states, pb = layer(
                    hidden_states,
                    shape_2d=(height, width),
                    attention_mask=mask_list[i],
                    position_bias=position_bias,
                    topk=int(topk_list[i]),
                )
                if self.rpe_shared:
                    position_bias = pb

            hidden_states = rearrange(
                hidden_states, "b (h w) c -> b c h w", h=height
            )
            hidden_states = scale_layer(hidden_states)
            if self.down:
                height, width = height // 2, width // 2
            else:
                height, width = height * 2, width * 2
            hidden_states = rearrange(hidden_states, "b c h w -> b (h w) c")

        position_bias = None
        for layer in self.trans_blocks[-1]:
            hidden_states, pb = layer(
                hidden_states,
                shape_2d=(height, width),
                attention_mask=mask_list[-1],
                position_bias=position_bias,
                topk=int(topk_list[-1]),
            )
            if self.rpe_shared:
                position_bias = pb

        out = self.mlp_head(hidden_states)
        out = rearrange(out, "b (h w) c -> b c h w", h=height)
        return out

    def get_mask(
        self, b: int, h: int, w: int
    ) -> Tuple[List[Tensor], Optional[Tensor], Tensor, Tensor]:
        # Self-attention masks at each scale stage; all-ones (full attention).
        mask_list: List[Tensor] = []
        ns, hs, ws = h * w, h, w
        for _ in range(self.scale + 1):
            mask = torch.ones((hs, ws, hs, ws)).bool().view(ns, ns)
            if self.down:
                ns, hs, ws = ns // 4, hs // 2, ws // 2
            else:
                ns, hs, ws = ns * 4, hs * 2, ws * 2
            mask_list.append(mask)

        token_mask = None
        input_mask = torch.ones(h, w).bool()
        output_mask = torch.ones(h, w).bool()

        mask_list = [
            repeat(m.unsqueeze(0), "() d n -> b d n", b=b) for m in mask_list
        ]
        input_mask = repeat(
            input_mask.unsqueeze(0).unsqueeze(0),
            "() () h w -> b d h w",
            b=b,
            d=self.cin,
        )
        channel = self.dim if self.cout == 0 else self.cout
        output_mask = repeat(
            output_mask.unsqueeze(0).unsqueeze(0),
            "() () h w -> b d h w",
            b=b,
            d=channel,
        )
        return mask_list, token_mask, input_mask, output_mask
