from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers.lic.dcae import (
    MutiScaleDictionaryCrossAttentionGLU,
    ResScaleConvolutionGateBlock,
    ResidualBottleneckBlockWithStride,
    ResidualBottleneckBlockWithUpsample,
    SwinBlockWithConvMulti,
    conv,
    deconv,
)
from compressai.registry import register_model

from .dcae_support import (
    DictionaryEntropyCompressionModel,
    infer_attention_head_dim,
    infer_dictionary_max_support_slices,
    infer_dictionary_num_slices,
    infer_stage_block_num,
    infer_window_size,
)

__all__ = ["DCAE"]


def _support_count(index: int, max_support_slices: int) -> int:
    if max_support_slices < 0:
        return index
    return min(index, max_support_slices)


def _support_count_with_current(index: int, max_support_slices: int) -> int:
    if max_support_slices < 0:
        return index + 1
    return min(index + 1, max_support_slices + 1)


@register_model("dcae")
class DCAE(DictionaryEntropyCompressionModel):
    def __init__(
        self,
        head_dim: Optional[Sequence[int]] = None,
        N: int = 192,
        M: int = 320,
        hyper_channels: int = 192,
        num_slices: int = 5,
        max_support_slices: int = 5,
        feature_dims: Optional[Sequence[int]] = None,
        block_num: Optional[Sequence[int]] = None,
        dict_num: int = 128,
        dict_head_num: int = 20,
        dictionary_dim: Optional[int] = None,
        window_size: int = 8,
        hyper_window_size: int = 4,
        hyper_head_dim: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        head_dim = tuple(head_dim or (8, 16, 32, 32, 16, 8))
        feature_dims = tuple(feature_dims or (96, 144, 256))
        block_num = tuple(block_num or (1, 2, 12))
        dictionary_dim = dictionary_dim or 32 * dict_head_num
        if len(head_dim) != 6:
            raise ValueError("head_dim must have six entries")
        if len(feature_dims) != 3:
            raise ValueError("feature_dims must have three entries")
        if len(block_num) != 3:
            raise ValueError("block_num must have three entries")
        if M % num_slices != 0:
            raise ValueError("M must be divisible by num_slices")

        input_image_channel = 3
        output_image_channel = 3
        slice_channels = M // num_slices

        self.head_dim = head_dim
        self.window_size = int(window_size)
        self.num_slices = int(num_slices)
        self.max_support_slices = int(max_support_slices)
        self.M = int(M)
        self.N = int(N)
        self.hyper_channels = int(hyper_channels)
        self.feature_dims = feature_dims
        self.block_num = block_num
        self.dict_num = int(dict_num)
        self.dict_head_num = int(dict_head_num)
        self.dictionary_dim = int(dictionary_dim)
        self.hyper_window_size = int(hyper_window_size)
        self.hyper_head_dim = int(hyper_head_dim)

        self.dt = nn.Parameter(torch.randn(dict_num, dictionary_dim), requires_grad=True)
        self.dt_cross_attention = nn.ModuleList(
            MutiScaleDictionaryCrossAttentionGLU(
                input_dim=M * 2 + slice_channels * _support_count(index, max_support_slices),
                output_dim=M,
                head_num=dict_head_num,
                mlp_rate=4,
                qkv_bias=True,
                dictionary_dim=dictionary_dim,
            )
            for index in range(num_slices)
        )

        self.m_down1 = [
            SwinBlockWithConvMulti(
                feature_dims[0],
                feature_dims[0],
                head_dim[0],
                self.window_size,
                0.0,
                ResScaleConvolutionGateBlock,
                block_num=block_num[0],
            ),
            ResidualBottleneckBlockWithStride(feature_dims[0], feature_dims[1]),
        ]
        self.m_down2 = [
            SwinBlockWithConvMulti(
                feature_dims[1],
                feature_dims[1],
                head_dim[1],
                self.window_size,
                0.0,
                ResScaleConvolutionGateBlock,
                block_num=block_num[1],
            ),
            ResidualBottleneckBlockWithStride(feature_dims[1], feature_dims[2]),
        ]
        self.m_down3 = [
            SwinBlockWithConvMulti(
                feature_dims[2],
                feature_dims[2],
                head_dim[2],
                self.window_size,
                0.0,
                ResScaleConvolutionGateBlock,
                block_num=block_num[2],
            ),
            conv(feature_dims[2], M, kernel_size=5, stride=2),
        ]
        self.g_a = nn.Sequential(
            ResidualBottleneckBlockWithStride(input_image_channel, feature_dims[0]),
            *self.m_down1,
            *self.m_down2,
            *self.m_down3,
        )

        self.m_up1 = [
            SwinBlockWithConvMulti(
                feature_dims[2],
                feature_dims[2],
                head_dim[3],
                self.window_size,
                0.0,
                ResScaleConvolutionGateBlock,
                block_num=block_num[2],
            ),
            ResidualBottleneckBlockWithUpsample(feature_dims[2], feature_dims[1]),
        ]
        self.m_up2 = [
            SwinBlockWithConvMulti(
                feature_dims[1],
                feature_dims[1],
                head_dim[4],
                self.window_size,
                0.0,
                ResScaleConvolutionGateBlock,
                block_num=block_num[1],
            ),
            ResidualBottleneckBlockWithUpsample(feature_dims[1], feature_dims[0]),
        ]
        self.m_up3 = [
            SwinBlockWithConvMulti(
                feature_dims[0],
                feature_dims[0],
                head_dim[5],
                self.window_size,
                0.0,
                ResScaleConvolutionGateBlock,
                block_num=block_num[0],
            ),
            ResidualBottleneckBlockWithUpsample(feature_dims[0], output_image_channel),
        ]
        self.g_s = nn.Sequential(
            deconv(M, feature_dims[2], kernel_size=5, stride=2),
            *self.m_up1,
            *self.m_up2,
            *self.m_up3,
        )

        self.ha_down = [
            SwinBlockWithConvMulti(
                N,
                N,
                hyper_head_dim,
                hyper_window_size,
                0.0,
                ResScaleConvolutionGateBlock,
                block_num=1,
            ),
            conv(N, hyper_channels, kernel_size=3, stride=2),
        ]
        self.h_a = nn.Sequential(
            ResidualBottleneckBlockWithStride(M, N),
            *self.ha_down,
        )
        self.hs_up1 = [
            SwinBlockWithConvMulti(
                N,
                N,
                hyper_head_dim,
                hyper_window_size,
                0.0,
                ResScaleConvolutionGateBlock,
                block_num=1,
            ),
            ResidualBottleneckBlockWithUpsample(N, M),
        ]
        self.h_z_s1 = nn.Sequential(
            deconv(hyper_channels, N, kernel_size=3, stride=2),
            *self.hs_up1,
        )
        self.hs_up2 = [
            SwinBlockWithConvMulti(
                N,
                N,
                hyper_head_dim,
                hyper_window_size,
                0.0,
                ResScaleConvolutionGateBlock,
                block_num=1,
            ),
            ResidualBottleneckBlockWithUpsample(N, M),
        ]
        self.h_z_s2 = nn.Sequential(
            deconv(hyper_channels, N, kernel_size=3, stride=2),
            *self.hs_up2,
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M * 3 + slice_channels * _support_count(index, max_support_slices), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, slice_channels, stride=1, kernel_size=3),
            )
            for index in range(num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M * 3 + slice_channels * _support_count(index, max_support_slices), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, slice_channels, stride=1, kernel_size=3),
            )
            for index in range(num_slices)
        )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M * 3 + slice_channels * _support_count_with_current(index, max_support_slices), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, slice_channels, stride=1, kernel_size=3),
            )
            for index in range(num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(hyper_channels)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x: Tensor) -> Dict[str, Union[Dict[str, Tensor], Tensor]]:
        latent = self.g_a(x)
        latent_out = self._forward_latent(latent)
        return {
            "x_hat": self.g_s(latent_out["y_hat"]),
            "likelihoods": latent_out["likelihoods"],
            "para": {
                "means": latent_out["means"],
                "scales": latent_out["scales"],
                "y": latent_out["y"],
            },
        }

    def compress(self, x: Tensor) -> Dict[str, object]:
        return self._compress_latent(self.g_a(x))

    def decompress(self, strings: Sequence[Sequence[bytes]], shape: Sequence[int]) -> Dict[str, Tensor]:
        return {"x_hat": self.g_s(self._decompress_latent(strings, tuple(shape))).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "DCAE":
        feature_dims = (
            state_dict["g_a.0.conv.weight"].size(0),
            state_dict["g_a.2.conv.weight"].size(0),
            state_dict["g_a.4.conv.weight"].size(0),
        )
        N = state_dict["h_a.0.conv.weight"].size(0)
        M = state_dict["h_a.0.conv.weight"].size(1)
        hyper_channels = state_dict["entropy_bottleneck.quantiles"].size(0)
        num_slices = infer_dictionary_num_slices(state_dict) or 5
        max_support_slices = infer_dictionary_max_support_slices(state_dict, M, num_slices)
        block_num = (
            infer_stage_block_num(state_dict, "g_a.1.layers."),
            infer_stage_block_num(state_dict, "g_a.3.layers."),
            infer_stage_block_num(state_dict, "g_a.5.layers."),
        )
        head_dim = (
            infer_attention_head_dim(state_dict, "g_a.1", feature_dims[0]),
            infer_attention_head_dim(state_dict, "g_a.3", feature_dims[1]),
            infer_attention_head_dim(state_dict, "g_a.5", feature_dims[2]),
            infer_attention_head_dim(state_dict, "g_s.1", feature_dims[2]),
            infer_attention_head_dim(state_dict, "g_s.3", feature_dims[1]),
            infer_attention_head_dim(state_dict, "g_s.5", feature_dims[0]),
        )
        net = cls(
            head_dim=head_dim,
            N=N,
            M=M,
            hyper_channels=hyper_channels,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
            feature_dims=feature_dims,
            block_num=block_num,
            dict_num=state_dict["dt"].size(0),
            dict_head_num=state_dict["dt_cross_attention.0.scale"].size(0),
            dictionary_dim=state_dict["dt"].size(1),
            window_size=infer_window_size(state_dict, "g_a.1"),
            hyper_window_size=infer_window_size(state_dict, "h_a.1"),
            hyper_head_dim=infer_attention_head_dim(state_dict, "h_a.1", N),
        )
        net.load_state_dict(state_dict)
        return net
