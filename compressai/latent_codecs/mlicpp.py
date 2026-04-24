# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers.lic.mlic import (
    ChannelContext,
    EntropyParameters,
    HyperAnalysis,
    HyperSynthesis,
    LatentResidualPrediction,
    LinearGlobalInterContext,
    LinearGlobalIntraContext,
    LocalContext,
    checkerboard_anchor,
    checkerboard_merge,
    checkerboard_nonanchor,
    checkerboard_split,
)
from compressai.ops import quantize_ste
from compressai.registry import register_module

from .base import LatentCodec
from .mlicpp_support import compress_single, decompress_single, select_num_heads

__all__ = ["MLICPlusPlusLatentCodec"]


@register_module("MLICPlusPlusLatentCodec")
class MLICPlusPlusLatentCodec(LatentCodec):
    """MLIC++ multi-reference latent entropy model.

    This codec owns the MLIC++ hyperprior and channel-slice entropy model:
    equal channel slices, checkerboard anchor/nonanchor coding, local context,
    channel/global contexts, and separate anchor/nonanchor LRP heads.
    """

    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        slice_num: int = 10,
        context_window: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        if slice_num <= 0:
            raise ValueError("slice_num must be positive")
        if context_window % 2 == 0:
            raise ValueError("context_window must be odd")
        if M % slice_num != 0:
            raise ValueError("M must be divisible by slice_num")

        slice_ch = M // slice_num
        self.N = int(N)
        self.M = int(M)
        self.context_window = int(context_window)
        self.slice_num = int(slice_num)
        self.slice_ch = int(slice_ch)

        self.h_a = HyperAnalysis(M=M, N=N)
        self.h_s = HyperSynthesis(M=M, N=N)
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

        self.local_context = nn.ModuleList(
            [LocalContext(dim=slice_ch, window_size=context_window) for _ in range(slice_num)]
        )
        self.channel_context = nn.ModuleList(
            [
                ChannelContext(in_dim=slice_ch * index, out_dim=slice_ch)
                if index
                else None
                for index in range(slice_num)
            ]
        )
        self.global_inter_context = nn.ModuleList(
            [
                LinearGlobalInterContext(
                    dim=slice_ch * index,
                    out_dim=slice_ch * 2,
                    num_heads=select_num_heads(slice_ch * index),
                )
                if index
                else None
                for index in range(slice_num)
            ]
        )
        self.global_intra_context = nn.ModuleList(
            [LinearGlobalIntraContext(dim=slice_ch) if index else None for index in range(slice_num)]
        )
        self.entropy_parameters_anchor = nn.ModuleList(
            [
                EntropyParameters(in_dim=M * 2, out_dim=slice_ch * 2)
                if index == 0
                else EntropyParameters(in_dim=M * 2 + slice_ch * 6, out_dim=slice_ch * 2)
                for index in range(slice_num)
            ]
        )
        self.entropy_parameters_nonanchor = nn.ModuleList(
            [
                EntropyParameters(in_dim=M * 2 + slice_ch * 2, out_dim=slice_ch * 2)
                if index == 0
                else EntropyParameters(in_dim=M * 2 + slice_ch * 10, out_dim=slice_ch * 2)
                for index in range(slice_num)
            ]
        )
        self.lrp_anchor = nn.ModuleList(
            [
                LatentResidualPrediction(in_dim=M + (index + 1) * slice_ch, out_dim=slice_ch)
                for index in range(slice_num)
            ]
        )
        self.lrp_nonanchor = nn.ModuleList(
            [
                LatentResidualPrediction(in_dim=M + (index + 1) * slice_ch, out_dim=slice_ch)
                for index in range(slice_num)
            ]
        )

    def forward(self, y: Tensor) -> Dict[str, Any]:
        self._update_local_contexts(y.size(2), y.size(3), y.device)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyper_params = self.h_s(z_hat)
        _, hyper_means = hyper_params.chunk(2, 1)

        y_hat_slices: List[Tensor] = []
        y_likelihoods: List[Tensor] = []
        for index, y_slice in enumerate(y.chunk(self.slice_num, dim=1)):
            slice_anchor, slice_nonanchor = checkerboard_split(y_slice)
            (
                scales_anchor,
                means_anchor,
                global_inter_ctx,
                channel_ctx,
            ) = self.anchor_distribution(index, y_hat_slices, hyper_params)
            slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
            lrp_anchor = self.lrp_anchor[index](
                self.lrp_inputs(hyper_means, y_hat_slices, slice_anchor)
            )
            slice_anchor = slice_anchor + checkerboard_anchor(lrp_anchor)

            scales_nonanchor, means_nonanchor = self.nonanchor_distribution(
                index,
                y_hat_slices,
                hyper_params,
                slice_anchor,
                global_inter_ctx,
                channel_ctx,
            )
            scales_slice = checkerboard_merge(scales_anchor, scales_nonanchor)
            means_slice = checkerboard_merge(means_anchor, means_nonanchor)
            _, y_slice_likelihoods = self.gaussian_conditional(
                y_slice, scales_slice, means_slice
            )

            slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
            y_hat_slice = slice_anchor + slice_nonanchor
            lrp_nonanchor = self.lrp_nonanchor[index](
                self.lrp_inputs(hyper_means, y_hat_slices, y_hat_slice)
            )
            y_hat_slice = y_hat_slice + checkerboard_nonanchor(lrp_nonanchor)
            y_hat_slices.append(y_hat_slice)
            y_likelihoods.append(y_slice_likelihoods)

        return {
            "y_hat": torch.cat(y_hat_slices, dim=1),
            "likelihoods": {
                "y": torch.cat(y_likelihoods, dim=1),
                "z": z_likelihoods,
            },
        }

    def compress(self, y: Tensor) -> Dict[str, Any]:
        self._update_local_contexts(y.size(2), y.size(3), y.device)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)
        _, hyper_means = hyper_params.chunk(2, 1)

        y_slices = y.chunk(self.slice_num, dim=1)
        streams_and_slices = [
            compress_single(
                self,
                [y_slice[index : index + 1] for y_slice in y_slices],
                hyper_params[index : index + 1],
                hyper_means[index : index + 1],
            )
            for index in range(y.size(0))
        ]
        y_strings = [stream for stream, _ in streams_and_slices]
        y_hat = torch.cat([y_hat_slice for _, y_hat_slice in streams_and_slices], dim=0)
        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "y_hat": y_hat,
        }

    def decompress(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Tuple[int, int],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if len(strings) != 2:
            raise ValueError("strings must contain [y_strings, z_strings]")

        y_strings, z_strings = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        self._update_local_contexts(
            z_hat.size(2) * 4,
            z_hat.size(3) * 4,
            z_hat.device,
        )
        hyper_params = self.h_s(z_hat)
        _, hyper_means = hyper_params.chunk(2, 1)
        y_hat = torch.cat(
            [
                decompress_single(
                    self,
                    y_string,
                    hyper_params[index : index + 1],
                    hyper_means[index : index + 1],
                )
                for index, y_string in enumerate(y_strings)
            ],
            dim=0,
        )
        return {"y_hat": y_hat}

    def _update_local_contexts(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> None:
        base_mask = None
        for index, module in enumerate(self.local_context):
            if index == 0:
                module.update_resolution(height, width, device)
                base_mask = module.attn_mask
                continue
            module.update_resolution(height, width, device, mask=base_mask)

    @staticmethod
    def lrp_inputs(
        hyper_means: Tensor,
        y_hat_slices: Sequence[Tensor],
        current_slice: Tensor,
    ) -> Tensor:
        return torch.cat([hyper_means, *y_hat_slices, current_slice], dim=1)

    @staticmethod
    def _optional_module(modules: nn.ModuleList, index: int) -> nn.Module:
        module = modules[index]
        if module is None:
            raise RuntimeError(f"Expected module at index {index}")
        return cast(nn.Module, module)

    def anchor_distribution(
        self,
        index: int,
        y_hat_slices: Sequence[Tensor],
        hyper_params: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        if index == 0:
            params = self.entropy_parameters_anchor[index](hyper_params)
            scales, means = params.chunk(2, 1)
            return checkerboard_anchor(scales), checkerboard_anchor(means), None, None

        previous_slices = torch.cat(list(y_hat_slices), dim=1)
        global_inter_ctx = self._optional_module(self.global_inter_context, index)(
            previous_slices
        )
        channel_ctx = self._optional_module(self.channel_context, index)(previous_slices)
        params = self.entropy_parameters_anchor[index](
            torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1)
        )
        scales, means = params.chunk(2, 1)
        return (
            checkerboard_anchor(scales),
            checkerboard_anchor(means),
            global_inter_ctx,
            channel_ctx,
        )

    def nonanchor_distribution(
        self,
        index: int,
        y_hat_slices: Sequence[Tensor],
        hyper_params: Tensor,
        anchor_hat: Tensor,
        global_inter_ctx: Optional[Tensor],
        channel_ctx: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        local_ctx = self.local_context[index](anchor_hat)
        if index == 0:
            params = self.entropy_parameters_nonanchor[index](
                torch.cat([local_ctx, hyper_params], dim=1)
            )
        else:
            global_intra_ctx = self._optional_module(self.global_intra_context, index)(
                y_hat_slices[-1],
                anchor_hat,
            )
            params = self.entropy_parameters_nonanchor[index](
                torch.cat(
                    [
                        local_ctx,
                        global_intra_ctx,
                        cast(Tensor, global_inter_ctx),
                        cast(Tensor, channel_ctx),
                        hyper_params,
                    ],
                    dim=1,
                )
            )
        scales, means = params.chunk(2, 1)
        return checkerboard_nonanchor(scales), checkerboard_nonanchor(means)
