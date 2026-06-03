# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# This file adapts code from the CMIC reference implementation (Content-Aware
# Mamba for Learned Image Compression, ICLR 2026, OpenReview WwDNiisZQm),
# integrated with the original authors' permission. The upstream copyright
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

from __future__ import annotations

import math

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
    EntropyBottleneckLatentCodec,
    GaussianConditionalLatentCodec,
    HyperpriorLatentCodec,
)
from compressai.layers import (
    CheckerboardMaskedConv2d,
    sequential_channel_ramp,
    subpel_conv3x3,
)
from compressai.layers.attn.swin import (
    WindowAttention,
    pad_to_window_multiple,
    window_partition,
    window_reverse,
)
from compressai.layers.lic.blocks import GatedFFN, GatedTransformCNN, LayerNorm2d
from compressai.layers.ssm import selective_scan
from compressai.layers.wave import is_pytorch_wavelets_available
from compressai.models._helpers.auxt import WLS, iWLS
from compressai.models._helpers.auxt import aux_loss as _aggregate_aux_loss
from compressai.models.utils import conv, deconv
from compressai.registry import register_model

from .base import SimpleVAECompressionModel

__all__ = [
    "CMIC",
    "CMICAnalysisTransform",
    "CMICSynthesisTransform",
]


# ---------------------------------------------------------------------------
# Spatial / channel context blocks (formerly compressai/layers/lic/cmic_context.py)
# ---------------------------------------------------------------------------


def _apply_permutation(input_tensor: Tensor, index: Tensor) -> Tensor:
    expanded_index = index.unsqueeze(-1).expand(-1, -1, input_tensor.size(-1))
    return torch.gather(input_tensor, dim=1, index=expanded_index)


def _inverse_permutation(index: Tensor) -> Tensor:
    inverse = torch.empty_like(index)
    positions = torch.arange(index.size(-1), device=index.device).expand_as(index)
    inverse.scatter_(1, index, positions)
    return inverse


def _batched_bincount(index: Tensor, num_classes: int, dtype: torch.dtype) -> Tensor:
    counts = torch.zeros(
        index.size(0),
        num_classes,
        device=index.device,
        dtype=dtype,
    )
    ones = torch.ones_like(index, dtype=dtype)
    counts.scatter_add_(1, index, ones)
    return counts


class CMICChannelContextBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        expansion_factor: float = 4.0,
        **layer_kwargs: Any,
    ) -> None:
        super().__init__()
        del layer_kwargs
        self.mixer = nn.Conv2d(dim, dim_out, kernel_size=1, stride=1)
        self.norm = LayerNorm2d(dim_out)
        self.mlp = GatedFFN(dim_out, expansion_factor=expansion_factor)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.mixer(input_tensor)
        return output + self.mlp(self.norm(output))


class _MaskedDepthwiseMixer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, slope: float = 0.01) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.LeakyReLU(negative_slope=slope, inplace=False),
        )
        self.masked_conv = CheckerboardMaskedConv2d(
            in_ch,
            in_ch,
            kernel_size=5,
            padding=2,
            groups=in_ch,
        )
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        if in_ch == out_ch:
            self.skip = nn.Identity()

    def forward(self, input_tensor: Tensor) -> Tensor:
        identity = self.skip(input_tensor)
        output = self.conv1(input_tensor)
        output = self.masked_conv(output)
        output = self.conv2(output)
        return output + identity


class _MaskedContextBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        expansion_factor: float = 4.0,
        **layer_kwargs: Any,
    ) -> None:
        super().__init__()
        del layer_kwargs
        self.mixer = _MaskedDepthwiseMixer(dim, dim_out)
        self.norm = LayerNorm2d(dim_out)
        self.mlp = GatedFFN(dim_out, expansion_factor=expansion_factor)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.mixer(input_tensor)
        return output + self.mlp(self.norm(output))


class CMICSpatialContextBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        expansion_factor: float = 4.0,
        **layer_kwargs: Any,
    ) -> None:
        super().__init__()
        self.layer1 = _MaskedContextBlock(
            dim,
            dim,
            expansion_factor=expansion_factor,
            **layer_kwargs,
        )
        self.layer2 = _MaskedContextBlock(
            dim,
            dim_out,
            expansion_factor=expansion_factor,
            **layer_kwargs,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.layer2(self.layer1(input_tensor))


# ---------------------------------------------------------------------------
# Content-aware Mamba stage (formerly compressai/layers/lic/cmic_stage.py)
# ---------------------------------------------------------------------------


class _ConvTokenGatedFFN(nn.Module):
    def __init__(self, channels: int, expansion_factor: float) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.channels = channels
        self.project_in = nn.Conv2d(
            channels,
            hidden_channels * 2,
            kernel_size=1,
            bias=False,
        )
        self.depthwise = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        height, width = x_size
        output = input_tensor.transpose(1, 2).reshape(
            input_tensor.size(0),
            self.channels,
            height,
            width,
        )
        gate_tensor, value_tensor = self.depthwise(self.project_in(output)).chunk(
            2,
            dim=1,
        )
        output = self.project_out(F.gelu(gate_tensor) * value_tensor)
        return output.flatten(2).transpose(1, 2).contiguous()


class _ContentAwareMamba(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        cluster_num: int = 64,
        inner_rank: int = 128,
        mlp_ratio: float = 2.0,
        n_iter: int = 5,
        ema_decay: float = 0.999,
    ) -> None:
        super().__init__()
        del inner_rank
        self.dim = dim
        self.d_state = d_state
        self.cluster_num = cluster_num
        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.hidden_dim = int(dim * mlp_ratio)
        self.dt_rank = math.ceil(self.hidden_dim / 16)

        self.in_proj = nn.Conv2d(dim, self.hidden_dim, kernel_size=1)
        self.cpe = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            padding=1,
            groups=self.hidden_dim,
        )
        self.prompt_proj = nn.Linear(dim, d_state)
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, dim)

        x_proj = nn.Linear(self.hidden_dim, self.dt_rank + d_state * 2, bias=False)
        self.x_proj_weight = nn.Parameter(x_proj.weight.unsqueeze(0))
        dt_proj = self._make_dt_proj()
        self.dt_projs_weight = nn.Parameter(dt_proj.weight.unsqueeze(0))
        self.dt_projs_bias = nn.Parameter(dt_proj.bias.unsqueeze(0))
        self.A_logs = self._make_a_logs()
        self.Ds = self._make_ds()

        self.register_buffer("means", torch.randn(cluster_num, dim))
        self.register_buffer("initted", torch.tensor(False))

    def _make_dt_proj(self) -> nn.Linear:
        layer = nn.Linear(self.dt_rank, self.hidden_dim, bias=True)
        std = self.dt_rank**-0.5
        nn.init.uniform_(layer.weight, -std, std)
        dt = torch.exp(
            torch.rand(self.hidden_dim) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        with torch.no_grad():
            layer.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        return layer

    def _make_a_logs(self) -> nn.Parameter:
        values = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        values = values.repeat(self.hidden_dim, 1)
        param = nn.Parameter(torch.log(values))
        param._no_weight_decay = True
        return param

    def _make_ds(self) -> nn.Parameter:
        param = nn.Parameter(torch.ones(self.hidden_dim))
        param._no_weight_decay = True
        return param

    def _center_iter(self, tokens: Tensor, means: Tensor) -> Tensor:
        scores = torch.einsum(
            "bnc,kc->bnk",
            F.normalize(tokens, dim=-1),
            F.normalize(means, dim=-1),
        )
        buckets = scores.argmax(dim=-1)
        counts = _batched_bincount(buckets, self.cluster_num, dtype=tokens.dtype)
        zero_mask = counts.sum(dim=0) == 0

        sums = torch.zeros(
            tokens.size(0),
            self.cluster_num,
            tokens.size(-1),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        expanded_index = buckets.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
        sums.scatter_add_(1, expanded_index, tokens)
        means_new = sums.sum(dim=0) / counts.sum(dim=0).clamp(min=1).unsqueeze(-1)
        means_new = F.normalize(means_new, dim=-1).type_as(tokens)
        return torch.where(zero_mask.unsqueeze(-1), means, means_new)

    def _update_centroids(self, tokens: Tensor) -> Tensor:
        tokens = tokens.contiguous()
        if not bool(self.initted.item()):
            pad_tokens = (
                self.cluster_num - tokens.size(1) % self.cluster_num
            ) % self.cluster_num
            if pad_tokens > 0:
                tokens = F.pad(tokens, (0, 0, 0, pad_tokens))
            grouped = tokens.view(tokens.size(0), self.cluster_num, -1, tokens.size(-1))
            means = grouped.permute(1, 0, 2, 3).reshape(self.cluster_num, -1, self.dim)
            centroids = means.mean(dim=1).detach()
        else:
            centroids = self.means.detach()

        if not self.training:
            return centroids

        with torch.no_grad():
            updated = centroids
            for _ in range(max(self.n_iter - 1, 0)):
                updated = self._center_iter(tokens, updated)
            if not bool(self.initted.item()):
                self.means.copy_(updated)
                self.initted.copy_(torch.tensor(True, device=self.initted.device))
            else:
                self.means.mul_(self.ema_decay).add_(updated, alpha=1 - self.ema_decay)
        return updated

    def _forward_scan(self, input_tensor: Tensor, prompt: Tensor) -> Tensor:
        batch_size, length, _ = input_tensor.shape
        xs = input_tensor.transpose(1, 2).reshape(
            batch_size, 1, self.hidden_dim, length
        )
        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj_weight)
        dts, b_param, c_param = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=2,
        )
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight)

        output = selective_scan(
            xs.float().view(batch_size, -1, length),
            dts.float().view(batch_size, -1, length),
            -torch.exp(self.A_logs.float()).view(-1, self.d_state),
            b_param.float().view(batch_size, 1, -1, length),
            c_param.float().view(batch_size, 1, -1, length) + prompt,
            self.Ds.float().view(-1),
            delta_bias=self.dt_projs_bias.float().view(-1),
            delta_softplus=True,
        )
        return output.view(batch_size, self.hidden_dim, length).transpose(1, 2)

    def forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        height, width = x_size
        batch_size, _, channels = input_tensor.shape
        tokens = input_tensor
        centroids = self._update_centroids(tokens.detach())
        scores = torch.einsum(
            "bnc,kc->bnk",
            F.normalize(tokens.detach(), dim=-1),
            F.normalize(centroids, dim=-1),
        )
        cluster_index = scores.argmax(dim=-1)
        prompt = torch.matmul(
            F.one_hot(cluster_index, num_classes=self.cluster_num).float(),
            self.prompt_proj(centroids),
        )

        sort_index = torch.sort(cluster_index, dim=-1, stable=True).indices
        inverse_index = _inverse_permutation(sort_index)

        output = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        output = self.in_proj(output)
        output = output * torch.sigmoid(self.cpe(output))
        output = output.flatten(2).transpose(1, 2).contiguous()
        output = _apply_permutation(output, sort_index)
        output = self._forward_scan(output, prompt.transpose(1, 2).unsqueeze(1))
        output = self.out_proj(self.out_norm(output))
        return _apply_permutation(output, inverse_index)


class _CMICBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        num_heads: int,
        window_size: int,
        inner_rank: int,
        cluster_num: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.window_attention = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
        )
        self.feed_forward1 = _ConvTokenGatedFFN(dim, expansion_factor=mlp_ratio)
        self.content_model = _ContentAwareMamba(
            dim=dim,
            d_state=d_state,
            cluster_num=cluster_num,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio,
        )
        self.feed_forward2 = _ConvTokenGatedFFN(dim, expansion_factor=mlp_ratio)

    def _window_forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        height, width = x_size
        output = input_tensor.reshape(input_tensor.size(0), height, width, self.dim)
        output, pad_height, pad_width = pad_to_window_multiple(
            output, self.window_size, layout="BHWC"
        )
        padded_height, padded_width = output.shape[1], output.shape[2]
        windows = window_partition(output, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, self.dim)
        windows = self.window_attention(windows)
        windows = windows.view(-1, self.window_size, self.window_size, self.dim)
        output = window_reverse(
            windows,
            self.window_size,
            padded_height,
            padded_width,
        )
        if pad_height > 0 or pad_width > 0:
            output = output[:, :height, :width, :].contiguous()
        return output.reshape(input_tensor.size(0), height * width, self.dim)

    def forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        output = input_tensor + self._window_forward(self.norm1(input_tensor), x_size)
        output = output + self.feed_forward1(self.norm2(output), x_size)
        output = output + self.content_model(self.norm3(output), x_size)
        return output + self.feed_forward2(self.norm4(output), x_size)


class CMICStage(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        depth: int,
        num_heads: int,
        window_size: int,
        inner_rank: int,
        cluster_num: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _CMICBlock(
                    dim=dim,
                    d_state=d_state,
                    num_heads=num_heads,
                    window_size=window_size,
                    inner_rank=inner_rank,
                    cluster_num=cluster_num,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        output = input_tensor.flatten(2).transpose(1, 2).contiguous()
        for block in self.blocks:
            output = block(output, x_size)
        height, width = x_size
        return output.transpose(1, 2).reshape(input_tensor.size(0), -1, height, width)


# ---------------------------------------------------------------------------
# CMIC transforms (formerly compressai/layers/lic/cmic.py)
# ---------------------------------------------------------------------------


class CMICAnalysisTransform(nn.Module):
    def __init__(
        self,
        M: int,
        stage_dims: Tuple[int, int, int] = (128, 192, 256),
        stage_depths: Tuple[int, int] = (2, 2),
        num_heads: Tuple[int, int] = (8, 8),
        d_state: int = 8,
        window_size: int = 8,
        inner_rank: int = 32,
        cluster_num: int = 64,
        stage_mlp_ratio: float = 3.0,
    ) -> None:
        super().__init__()
        embed_dim0, embed_dim1, embed_dim2 = stage_dims
        depth1, depth2 = stage_depths
        heads1, heads2 = num_heads

        self.AuxT_enc = nn.Sequential(
            WLS(3, embed_dim0),
            WLS(embed_dim0, embed_dim1),
            WLS(embed_dim1, embed_dim2),
            WLS(embed_dim2, M),
        )
        self.g1 = nn.Sequential(
            GatedTransformCNN(embed_dim0, embed_dim0, expansion_factor=stage_mlp_ratio),
            GatedTransformCNN(embed_dim0, embed_dim0, expansion_factor=stage_mlp_ratio),
            GatedTransformCNN(embed_dim0, embed_dim0, expansion_factor=stage_mlp_ratio),
        )
        self.g2 = CMICStage(
            dim=embed_dim1,
            d_state=d_state,
            depth=depth1,
            num_heads=heads1,
            window_size=window_size,
            inner_rank=inner_rank,
            cluster_num=cluster_num,
            mlp_ratio=stage_mlp_ratio,
        )
        self.g3 = CMICStage(
            dim=embed_dim2,
            d_state=d_state,
            depth=depth2,
            num_heads=heads2,
            window_size=window_size,
            inner_rank=inner_rank,
            cluster_num=cluster_num,
            mlp_ratio=stage_mlp_ratio,
        )
        self.down0 = nn.Conv2d(3, embed_dim0, kernel_size=3, stride=2, padding=1)
        self.down1 = nn.Conv2d(
            embed_dim0, embed_dim1, kernel_size=3, stride=2, padding=1
        )
        self.down2 = nn.Conv2d(
            embed_dim1, embed_dim2, kernel_size=3, stride=2, padding=1
        )
        self.down3 = nn.Conv2d(embed_dim2, M, kernel_size=3, stride=2, padding=1)

    def forward(self, input_tensor: Tensor) -> Tensor:
        aux_output = input_tensor

        output = self.down0(input_tensor)
        output = self.g1(output)
        aux_output = self.AuxT_enc[0](aux_output)
        output = output + aux_output

        output = self.down1(output)
        output = self.g2(output, output.shape[-2:])
        aux_output = self.AuxT_enc[1](aux_output)
        output = output + aux_output

        output = self.down2(output)
        output = self.g3(output, output.shape[-2:])
        aux_output = self.AuxT_enc[2](aux_output)
        output = output + aux_output

        output = self.down3(output)
        aux_output = self.AuxT_enc[3](aux_output)
        return output + aux_output


class CMICSynthesisTransform(nn.Module):
    def __init__(
        self,
        M: int,
        stage_dims: Tuple[int, int, int] = (128, 192, 256),
        stage_depths: Tuple[int, int] = (2, 2),
        num_heads: Tuple[int, int] = (8, 8),
        d_state: int = 8,
        window_size: int = 8,
        inner_rank: int = 32,
        cluster_num: int = 64,
        stage_mlp_ratio: float = 3.0,
    ) -> None:
        super().__init__()
        embed_dim1, embed_dim2, embed_dim3 = stage_dims
        depth1, depth2 = stage_depths
        heads1, heads2 = num_heads

        self.AuxT_dec = nn.Sequential(
            iWLS(M, embed_dim3),
            iWLS(embed_dim3, embed_dim2),
            iWLS(embed_dim2, embed_dim1),
            iWLS(embed_dim1, 3),
        )
        self.g1 = CMICStage(
            dim=embed_dim3,
            d_state=d_state,
            depth=depth2,
            num_heads=heads2,
            window_size=window_size,
            inner_rank=inner_rank,
            cluster_num=cluster_num,
            mlp_ratio=stage_mlp_ratio,
        )
        self.g2 = CMICStage(
            dim=embed_dim2,
            d_state=d_state,
            depth=depth1,
            num_heads=heads1,
            window_size=window_size,
            inner_rank=inner_rank,
            cluster_num=cluster_num,
            mlp_ratio=stage_mlp_ratio,
        )
        self.g3 = nn.Sequential(
            GatedTransformCNN(embed_dim1, embed_dim1, expansion_factor=stage_mlp_ratio),
            GatedTransformCNN(embed_dim1, embed_dim1, expansion_factor=stage_mlp_ratio),
            GatedTransformCNN(embed_dim1, embed_dim1, expansion_factor=stage_mlp_ratio),
        )
        self.up0 = deconv(M, embed_dim3, kernel_size=3)
        self.up1 = deconv(embed_dim3, embed_dim2, kernel_size=3)
        self.up2 = deconv(embed_dim2, embed_dim1, kernel_size=3)
        self.up3 = subpel_conv3x3(embed_dim1, 3, 2)

    def forward(self, input_tensor: Tensor) -> Tensor:
        aux_output = input_tensor

        output = self.up0(input_tensor)
        output = self.g1(output, output.shape[-2:])
        aux_output = self.AuxT_dec[0](aux_output)
        output = output + aux_output

        output = self.up1(output)
        output = self.g2(output, output.shape[-2:])
        aux_output = self.AuxT_dec[1](aux_output)
        output = output + aux_output

        output = self.up2(output)
        output = self.g3(output)
        aux_output = self.AuxT_dec[2](aux_output)
        output = output + aux_output

        output = self.up3(output)
        aux_output = self.AuxT_dec[3](aux_output)
        return output + aux_output


# ---------------------------------------------------------------------------
# CMIC model (top-level + state-dict converter)
# ---------------------------------------------------------------------------


def _require_wavelets() -> None:
    if is_pytorch_wavelets_available():
        return
    raise ModuleNotFoundError(
        "CMIC requires the optional dependency `pytorch_wavelets`. "
        "Install `compressai[wavelet]` to enable this model."
    )


def _default_groups(M: int) -> List[int]:
    if M > 128:
        return [16, 16, 32, 64, M - 128]
    num_groups = min(4, M)
    base, remainder = divmod(M, num_groups)
    groups = [base] * num_groups
    for index in range(remainder):
        groups[-(index + 1)] += 1
    return [group for group in groups if group > 0]


@register_model("cmic")
class CMIC(SimpleVAECompressionModel):
    r"""Content-Aware Mamba Image Compression model from Y. Chen, Z. Hu,
    et al.: `"Content-Aware Mamba for Learned Image Compression"
    <https://openreview.net/forum?id=WwDNiisZQm>`_, Int. Conf. on Learning
    Representations (ICLR), 2026.

    Combines wavelet/graph auxiliary branches with content-adaptive Mamba
    state-space blocks and a checkerboard / channel-group hyperprior.

    Args:
        N (int): Number of channels in the hyperprior.
        M (int): Number of channels in the latent representation.
    """

    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        groups: Optional[List[int]] = None,
        stage_dims: Tuple[int, int, int] = (128, 192, 256),
        stage_depths: Tuple[int, int] = (2, 2),
        num_heads: Tuple[int, int] = (8, 8),
        d_state: int = 8,
        window_size: int = 8,
        inner_rank: int = 32,
        cluster_num: int = 64,
        stage_mlp_ratio: float = 3.0,
        **kwargs: Any,
    ) -> None:
        _require_wavelets()
        super().__init__(**kwargs)

        if len(stage_dims) != 3:
            raise ValueError("`stage_dims` must contain three feature dimensions.")
        if len(stage_depths) != 2:
            raise ValueError("`stage_depths` must contain two stage depths.")
        if len(num_heads) != 2:
            raise ValueError("`num_heads` must contain two head counts.")

        self.N = int(N)
        self.M = int(M)
        self.stage_dims = tuple(int(dim) for dim in stage_dims)
        self.stage_depths = tuple(int(depth) for depth in stage_depths)
        self.num_heads = tuple(int(head) for head in num_heads)
        self.d_state = int(d_state)
        self.window_size = int(window_size)
        self.inner_rank = int(inner_rank)
        self.cluster_num = int(cluster_num)
        self.stage_mlp_ratio = float(stage_mlp_ratio)
        self.groups = list(groups) if groups is not None else _default_groups(M)
        if sum(self.groups) != M:
            raise ValueError("Channel groups must sum to M.")

        self.g_a = CMICAnalysisTransform(
            M=M,
            stage_dims=self.stage_dims,
            stage_depths=self.stage_depths,
            num_heads=self.num_heads,
            d_state=self.d_state,
            window_size=self.window_size,
            inner_rank=self.inner_rank,
            cluster_num=self.cluster_num,
            stage_mlp_ratio=self.stage_mlp_ratio,
        )
        self.g_s = CMICSynthesisTransform(
            M=M,
            stage_dims=self.stage_dims,
            stage_depths=self.stage_depths,
            num_heads=self.num_heads,
            d_state=self.d_state,
            window_size=self.window_size,
            inner_rank=self.inner_rank,
            cluster_num=self.cluster_num,
            stage_mlp_ratio=self.stage_mlp_ratio,
        )

        h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            conv(N, N, kernel_size=3, stride=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            conv(N, N, kernel_size=3, stride=2),
        )
        h_s = nn.Sequential(
            subpel_conv3x3(N, N, 2),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            subpel_conv3x3(N, N, 2),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            conv(N, N * 2, kernel_size=3, stride=1),
        )

        channel_context = {
            f"y{k}": sequential_channel_ramp(
                sum(self.groups[:k]),
                self.groups[k] * 2,
                min_ch=N,
                num_layers=3,
                make_layer=GatedTransformCNN,
                make_act=lambda: nn.Identity(),
                expansion_factor=4,
            )
            for k in range(1, len(self.groups))
        }
        spatial_context = [
            CMICSpatialContextBlock(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]
        param_aggregation = [
            sequential_channel_ramp(
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + N * 2,
                self.groups[k] * 2,
                min_ch=N * 2,
                num_layers=3,
                make_layer=CMICChannelContextBlock,
                make_act=lambda: nn.Identity(),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={"y": GaussianConditionalLatentCodec(quantizer="ste")},
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        self.latent_codec = HyperpriorLatentCodec(
            h_a=h_a,
            h_s=h_s,
            latent_codec={
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                "z": EntropyBottleneckLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    quantizer="ste",
                ),
            },
        )

    def aux_loss(self) -> Tensor:
        """Aggregate the orthogonality penalty from all :class:`OLP` modules.

        Returns a 0-d tensor; weight it into the training loss alongside the
        rate-distortion objective.
        """
        return _aggregate_aux_loss(self)

    def ortho_loss(self) -> Tensor:
        """Backward-compatible alias for :meth:`aux_loss` (upstream name)."""
        return self.aux_loss()

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "CMIC":
        N = state_dict["latent_codec.z.entropy_bottleneck.quantiles"].size(0)
        M = state_dict["g_a.down3.weight"].size(0)
        stage_dims = (
            state_dict["g_a.down0.weight"].size(0),
            state_dict["g_a.down1.weight"].size(0),
            state_dict["g_a.down2.weight"].size(0),
        )
        stage_depths = (
            cls._infer_depth(state_dict, "g_a.g2"),
            cls._infer_depth(state_dict, "g_a.g3"),
        )
        num_heads = (
            state_dict[
                "g_a.g2.blocks.0.window_attention.relative_position_bias_table"
            ].size(1),
            state_dict[
                "g_a.g3.blocks.0.window_attention.relative_position_bias_table"
            ].size(1),
        )
        table_size = state_dict[
            "g_a.g2.blocks.0.window_attention.relative_position_bias_table"
        ].size(0)
        groups = cls._infer_groups(state_dict)
        stage_mlp_ratio = cls._infer_stage_mlp_ratio(state_dict, stage_dims[1])

        net = cls(
            N=N,
            M=M,
            groups=groups or None,
            stage_dims=stage_dims,
            stage_depths=stage_depths,
            num_heads=num_heads,
            d_state=state_dict["g_a.g2.blocks.0.content_model.A_logs"].size(1),
            window_size=(math.isqrt(table_size) + 1) // 2,
            cluster_num=state_dict["g_a.g2.blocks.0.content_model.means"].size(0),
            stage_mlp_ratio=stage_mlp_ratio,
        )

        # Compressai's WindowAttention registers a `relative_position_index`
        # buffer, and `pytorch_wavelets`'s `DWTForward`/`DWTInverse` register
        # their haar coefficient buffers, that the upstream checkpoint does not
        # ship. Backfill from the freshly-initialised model so we can keep
        # `strict=True`.
        init_state_dict = net.state_dict()
        for key, value in init_state_dict.items():
            if key in state_dict:
                continue
            if (
                key.endswith(".window_attention.relative_position_index")
                or ".dwt.transform." in key
                or ".idwt.inverse." in key
            ):
                state_dict[key] = value

        net.load_state_dict(state_dict)
        return net

    @staticmethod
    def _infer_depth(state_dict: Dict[str, Tensor], prefix: str) -> int:
        depth = 0
        while f"{prefix}.blocks.{depth}.norm1.weight" in state_dict:
            depth += 1
        return depth

    @staticmethod
    def _infer_stage_mlp_ratio(
        state_dict: Dict[str, Tensor],
        stage_dim: int,
    ) -> float:
        """Recover ``stage_mlp_ratio`` from a ``_ConvTokenGatedFFN`` head.

        ``feed_forward1.project_in`` maps ``dim -> 2 * int(dim * mlp_ratio)``,
        so its output width over twice the stage dim recovers the ratio.
        """
        key = "g_a.g2.blocks.0.feed_forward1.project_in.weight"
        if key not in state_dict:
            return 3.0
        return (state_dict[key].size(0) // 2) / stage_dim

    @staticmethod
    def _infer_groups(state_dict: Dict[str, Tensor]) -> List[int]:
        groups = []
        index = 0
        while True:
            key = (
                "latent_codec.y.latent_codec."
                f"y{index}.context_prediction.layer1.mixer.conv1.0.weight"
            )
            if key not in state_dict:
                key = f"latent_codec.y.latent_codec.y{index}.context_prediction.weight"
            if key not in state_dict:
                break
            groups.append(state_dict[key].size(1))
            index += 1
        return groups
