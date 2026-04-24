from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers.lic.mlic import (
    AnalysisTransform,
    ChannelContext,
    EntropyParameters,
    HyperAnalysis,
    HyperSynthesis,
    LatentResidualPrediction,
    LinearGlobalInterContext,
    LinearGlobalIntraContext,
    LocalContext,
    SynthesisTransform,
    checkerboard_anchor,
    checkerboard_merge,
    checkerboard_nonanchor,
    checkerboard_split,
)
from compressai.ops import quantize_ste
from compressai.registry import register_model

from .base import CompressionModel
from .mlicpp_support import (
    anchor_distribution,
    compress_single,
    decompress_single,
    lrp_inputs,
    nonanchor_distribution,
    select_num_heads,
    update_local_contexts,
)

__all__ = ["MLICPlusPlus"]


@register_model("mlicpp")
class MLICPlusPlus(CompressionModel):
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

        slice_ch = M // slice_num
        self.N = int(N)
        self.M = int(M)
        self.context_window = int(context_window)
        self.slice_num = int(slice_num)
        self.slice_ch = int(slice_ch)

        self.g_a = AnalysisTransform(N=N, M=M)
        self.g_s = SynthesisTransform(N=N, M=M)
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

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor] | Tensor]:
        y = self.g_a(x)
        update_local_contexts(self.local_context, y.size(2), y.size(3), y.device)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyper_params = self.h_s(z_hat)
        _, hyper_means = hyper_params.chunk(2, 1)

        y_hat_slices = []
        y_likelihoods = []
        for index, y_slice in enumerate(y.chunk(self.slice_num, dim=1)):
            slice_anchor, slice_nonanchor = checkerboard_split(y_slice)
            (
                scales_anchor,
                means_anchor,
                global_inter_ctx,
                channel_ctx,
            ) = anchor_distribution(self, index, y_hat_slices, hyper_params)
            slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
            lrp_anchor = self.lrp_anchor[index](lrp_inputs(hyper_means, y_hat_slices, slice_anchor))
            slice_anchor = slice_anchor + checkerboard_anchor(lrp_anchor)

            scales_nonanchor, means_nonanchor = nonanchor_distribution(
                self,
                index,
                y_hat_slices,
                hyper_params,
                slice_anchor,
                global_inter_ctx,
                channel_ctx,
            )
            scales_slice = checkerboard_merge(scales_anchor, scales_nonanchor)
            means_slice = checkerboard_merge(means_anchor, means_nonanchor)
            _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)

            slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
            y_hat_slice = slice_anchor + slice_nonanchor
            lrp_nonanchor = self.lrp_nonanchor[index](
                lrp_inputs(hyper_means, y_hat_slices, y_hat_slice)
            )
            y_hat_slice = y_hat_slice + checkerboard_nonanchor(lrp_nonanchor)
            y_hat_slices.append(y_hat_slice)
            y_likelihoods.append(y_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        return {
            "x_hat": self.g_s(y_hat),
            "likelihoods": {
                "y": torch.cat(y_likelihoods, dim=1),
                "z": z_likelihoods,
            },
        }

    def compress(self, x: Tensor) -> Dict[str, object]:
        y = self.g_a(x)
        update_local_contexts(self.local_context, y.size(2), y.size(3), y.device)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)
        _, hyper_means = hyper_params.chunk(2, 1)

        y_slices = y.chunk(self.slice_num, dim=1)
        y_strings = [
            compress_single(
                self,
                [y_slice[index : index + 1] for y_slice in y_slices],
                hyper_params[index : index + 1],
                hyper_means[index : index + 1],
            )
            for index in range(y.size(0))
        ]
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        if len(strings) != 2:
            raise ValueError("strings must contain [y_strings, z_strings]")

        y_strings, z_strings = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        update_local_contexts(
            self.local_context,
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
        return {"x_hat": self.g_s(y_hat).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "MLICPlusPlus":
        N = state_dict["g_a.analysis_transform.0.conv1.weight"].size(0)
        M = state_dict["g_a.analysis_transform.6.weight"].size(0)
        slice_indices = {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("local_context.") and key.endswith(".relative_position_table")
        }
        slice_num = len(slice_indices) or 10
        context_tokens = state_dict["local_context.0.relative_position_index"].size(0)
        context_window = int(round(context_tokens**0.5))
        net = cls(N=N, M=M, slice_num=slice_num, context_window=context_window)
        net.load_state_dict(state_dict)
        return net
