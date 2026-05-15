# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

from __future__ import annotations

import warnings

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import (
    EntropyBottleneck,
    GaussianLaplaceLogisticMixtureConditional,
)
from compressai.layers import (
    GDN,
    MaskedConv2d,
    conv1x1,
    conv3x3,
    deconv,
)
from compressai.layers.lic.gllmm import GLLMMNonLocalAttentionBlock, GLLMMResidualChain
from compressai.models.base import CompressionModel
from compressai.registry import register_model

__all__ = [
    "GLLMM",
    "GLLMMAnalysisTransform",
    "GLLMMHyperAnalysisTransform",
    "GLLMMHyperSynthesisTransform",
    "GLLMMSynthesisTransform",
]


def _leaky_conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    negative_slope: float = 0.2,
) -> nn.Sequential:
    return nn.Sequential(
        conv3x3(in_channels, out_channels, stride=stride),
        nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
    )


def _gdn_conv3x3(channels: int, inverse: bool = False) -> nn.Sequential:
    return nn.Sequential(
        conv3x3(channels, channels),
        GDN(channels, inverse=inverse),
    )


def _subpixel_conv3x3(
    in_channels: int,
    out_channels: int,
    upscale_factor: int = 2,
    *,
    activate: bool = True,
    negative_slope: float = 0.2,
) -> nn.Sequential:
    layers: List[nn.Module] = [
        conv3x3(in_channels, out_channels * upscale_factor**2)
    ]
    if activate:
        layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
    layers.append(nn.PixelShuffle(upscale_factor))
    return nn.Sequential(*layers)


def _subpixel_conv1x1(
    in_channels: int,
    out_channels: int,
    upscale_factor: int = 2,
) -> nn.Sequential:
    return nn.Sequential(
        conv1x1(in_channels, out_channels * upscale_factor**2),
        nn.PixelShuffle(upscale_factor),
    )


class _AnalysisDownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_gdn: bool,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.shortcut = conv1x1(in_channels, out_channels, stride=2)
        self.down = _leaky_conv3x3(in_channels, out_channels, stride=2)
        self.post = _gdn_conv3x3(out_channels) if use_gdn else conv3x3(out_channels, out_channels)
        if not use_bias:
            self.shortcut = nn.Identity()
            self.down = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            self.post = nn.Identity()

    def forward(self, input_tensor: Tensor) -> Tensor:
        if isinstance(self.shortcut, nn.Identity):
            return self.down(input_tensor)
        return self.post(self.down(input_tensor)) + self.shortcut(input_tensor)


class _SynthesisUpsampleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.shortcut = _subpixel_conv1x1(channels, channels)
        self.up = _subpixel_conv3x3(channels, channels)
        self.post = _gdn_conv3x3(channels, inverse=True)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.post(self.up(input_tensor)) + self.shortcut(input_tensor)


class GLLMMAnalysisTransform(nn.Module):
    """Analysis transform from learned image compression with GLLMM."""

    def __init__(self, channels: int, input_channels: int = 3) -> None:
        super().__init__()
        self.block0_down = _AnalysisDownsampleBlock(
            input_channels, channels, use_gdn=True
        )
        self.block1_residual = GLLMMResidualChain(channels)
        self.block1_down = _AnalysisDownsampleBlock(channels, channels, use_gdn=True)
        self.block1_attention = GLLMMNonLocalAttentionBlock(channels)
        self.block2_residual = GLLMMResidualChain(channels)
        self.block2_down = _AnalysisDownsampleBlock(channels, channels, use_gdn=True)
        self.block3_residual = GLLMMResidualChain(channels)
        self.block3_down = _AnalysisDownsampleBlock(
            channels, channels, use_gdn=False, use_bias=False
        )
        self.block3_attention = GLLMMNonLocalAttentionBlock(channels)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.block0_down(input_tensor)
        output = self.block1_residual(output)
        output = self.block1_down(output)
        output = self.block1_attention(output)
        output = self.block2_residual(output)
        output = self.block2_down(output)
        output = self.block3_residual(output)
        output = self.block3_down(output)
        return self.block3_attention(output)


class GLLMMSynthesisTransform(nn.Module):
    """Synthesis transform from learned image compression with GLLMM."""

    def __init__(self, channels: int, output_channels: int = 3) -> None:
        super().__init__()
        self.block0_attention = GLLMMNonLocalAttentionBlock(channels)
        self.block0_residual = GLLMMResidualChain(channels)
        self.block0_up = _SynthesisUpsampleBlock(channels)
        self.block1_residual = GLLMMResidualChain(channels)
        self.block1_up = _SynthesisUpsampleBlock(channels)
        self.block2_attention = GLLMMNonLocalAttentionBlock(channels)
        self.block2_residual = GLLMMResidualChain(channels)
        self.block2_up = _SynthesisUpsampleBlock(channels)
        self.block3_residual = GLLMMResidualChain(channels)
        self.block3_up = _subpixel_conv3x3(
            channels, output_channels, activate=False
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.block0_attention(input_tensor)
        output = self.block0_residual(output)
        output = self.block0_up(output)
        output = self.block1_residual(output)
        output = self.block1_up(output)
        output = self.block2_attention(output)
        output = self.block2_residual(output)
        output = self.block2_up(output)
        output = self.block3_residual(output)
        return self.block3_up(output)


class GLLMMHyperAnalysisTransform(nn.Module):
    """Hyper analysis transform for the GLLMM latent prior."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _leaky_conv3x3(channels, channels),
            _leaky_conv3x3(channels, channels),
            _leaky_conv3x3(channels, channels, stride=2),
            _leaky_conv3x3(channels, channels),
            conv3x3(channels, channels, stride=2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.layers(input_tensor)


class GLLMMHyperSynthesisTransform(nn.Module):
    """Hyper synthesis transform producing GLLMM side information phi."""

    def __init__(self, channels: int, inner_channels: Optional[int] = None) -> None:
        super().__init__()
        if inner_channels is None:
            inner_channels = int(channels * 1.5)
        self.layers = nn.Sequential(
            nn.Sequential(
                deconv(channels, channels, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                deconv(channels, channels, kernel_size=3, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                deconv(channels, inner_channels, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                deconv(inner_channels, inner_channels, kernel_size=3, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            deconv(inner_channels, 2 * channels, kernel_size=3, stride=1),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.layers(input_tensor)


def _split_gllmm_params(
    params: Tensor,
    channels: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if params.size(1) != 30 * channels:
        raise ValueError(
            f"Expected {30 * channels} entropy-parameter channels, "
            f"got {params.size(1)}"
        )

    parts = params.chunk(30, dim=1)
    gaussian_weights = torch.stack(parts[0:3], dim=1).softmax(dim=1).flatten(1, 2)
    gaussian_means = torch.cat(parts[3:6], dim=1)
    gaussian_scales = torch.cat(parts[6:9], dim=1).abs()

    laplace_weights = torch.stack(parts[9:12], dim=1).softmax(dim=1).flatten(1, 2)
    laplace_means = torch.cat(parts[12:15], dim=1)
    laplace_scales = torch.cat(parts[15:18], dim=1).abs()

    logistic_weights = torch.stack(parts[18:21], dim=1).softmax(dim=1).flatten(1, 2)
    logistic_means = torch.cat(parts[21:24], dim=1)
    logistic_scales = torch.cat(parts[24:27], dim=1).abs()

    family_weights = torch.stack(parts[27:30], dim=1).softmax(dim=1).flatten(1, 2)
    means = torch.cat((gaussian_means, laplace_means, logistic_means), dim=1)
    scales = torch.cat((gaussian_scales, laplace_scales, logistic_scales), dim=1)
    return (
        means,
        scales,
        gaussian_weights,
        laplace_weights,
        logistic_weights,
        family_weights,
    )


@register_model("gllmm")
class GLLMM(CompressionModel):
    """Learned image compression with a GLLMM latent prior."""

    def __init__(
        self,
        N: int = 128,
        M: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if M is not None and int(M) != int(N):
            raise ValueError("GLLMM currently expects M == N, matching upstream.")
        channels = int(N)
        self.N = channels
        self.M = channels

        self.g_a = GLLMMAnalysisTransform(channels)
        self.g_s = GLLMMSynthesisTransform(channels)
        self.h_a = GLLMMHyperAnalysisTransform(channels)
        self.h_s = GLLMMHyperSynthesisTransform(channels, int(channels * 1.5))
        self.entropy_bottleneck = EntropyBottleneck(channels)
        self.context_prediction = MaskedConv2d(
            channels,
            2 * channels,
            kernel_size=5,
            padding=2,
            stride=1,
            mask_type="A",
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(4 * channels, 640, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 1280, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1280, 30 * channels, kernel_size=1, bias=False),
        )
        self.gaussian_conditional = GaussianLaplaceLogisticMixtureConditional()

    @property
    def downsampling_factor(self) -> int:
        return 2**6

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor] | Tensor]:
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        phi = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y,
            "noise" if self.training else "dequantize",
            means=None,
        )
        context = self.context_prediction(y_hat)
        params = self.entropy_parameters(torch.cat((context, phi), dim=1))
        means, scales, weights_g, weights_l, weights_lo, weights_family = (
            _split_gllmm_params(params, self.M)
        )
        _, y_likelihoods = self.gaussian_conditional(
            y,
            means,
            scales,
            weights_g,
            weights_l,
            weights_lo,
            weights_family,
        )

        return {
            "x_hat": self.g_s(y_hat),
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods,
            },
        }

    def compress(self, x: Tensor) -> Dict[str, Any]:
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for autoregressive models "
                "because the entropy coder runs sequentially on CPU.",
                stacklevel=2,
            )

        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        phi = self.h_s(z_hat)

        kernel_size = 5
        padding = (kernel_size - 1) // 2
        height, width = phi.size(2), phi.size(3)
        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for index in range(y.size(0)):
            y_strings.append(
                self._compress_ar(
                    y_hat[index : index + 1],
                    phi[index : index + 1],
                    height,
                    width,
                    kernel_size,
                    padding,
                )
            )
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(
        self,
        y_hat: Tensor,
        phi: Tensor,
        height: int,
        width: int,
        kernel_size: int,
        padding: int,
    ) -> Tuple[bytes, int]:
        central = y_hat[:, :, padding : padding + height, padding : padding + width]
        abs_max = int(torch.round(central).abs().max().item()) + 1
        abs_max = max(abs_max, 1)

        encoder = BufferedRansEncoder()
        symbols_list: List[int] = []
        indexes_list: List[int] = []
        cdf_list: List[List[int]] = []
        cdf_lengths: List[int] = []
        offsets: List[int] = []
        masked_weight = self.context_prediction.weight * self.context_prediction.mask

        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                context = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )
                hyper = phi[:, :, h : h + 1, w : w + 1]
                params = self.entropy_parameters(torch.cat((context, hyper), dim=1))
                means, scales, weights_g, weights_l, weights_lo, weights_family = (
                    _split_gllmm_params(params, self.M)
                )
                cdf = self.gaussian_conditional.build_cdf(
                    means,
                    scales,
                    weights_g,
                    weights_l,
                    weights_lo,
                    weights_family,
                    abs_max,
                )

                y_center = y_crop[:, :, padding, padding]
                y_symbols = self.gaussian_conditional.quantize(
                    y_center,
                    "symbols",
                    means=None,
                )
                y_hat[:, :, h + padding, w + padding] = y_symbols.to(y_hat.dtype)

                start_index = len(cdf_list)
                cdf_list.extend(cdf.cpu().tolist())
                cdf_lengths.extend([cdf.size(1)] * cdf.size(0))
                offsets.extend([0] * cdf.size(0))
                indexes_list.extend(range(start_index, start_index + cdf.size(0)))
                symbols_list.extend((y_symbols.reshape(-1).int() + abs_max).tolist())

        encoder.encode_with_indexes(
            symbols_list,
            indexes_list,
            cdf_list,
            cdf_lengths,
            offsets,
        )
        return encoder.flush(), abs_max

    def decompress(self, strings: List[Any], shape: Tuple[int, int]) -> Dict[str, Tensor]:
        if not isinstance(strings, list) or len(strings) != 2:
            raise ValueError("Invalid strings format")

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for autoregressive models "
                "because the entropy coder runs sequentially on CPU.",
                stacklevel=2,
            )

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        phi = self.h_s(z_hat)

        kernel_size = 5
        padding = (kernel_size - 1) // 2
        height, width = phi.size(2), phi.size(3)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, height + 2 * padding, width + 2 * padding),
            device=z_hat.device,
            dtype=z_hat.dtype,
        )

        for index, payload in enumerate(strings[0]):
            if isinstance(payload, tuple):
                y_string, abs_max = payload
            else:
                y_string, abs_max = payload, 255
            self._decompress_ar(
                y_string,
                int(abs_max),
                y_hat[index : index + 1],
                phi[index : index + 1],
                height,
                width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self,
        y_string: bytes,
        abs_max: int,
        y_hat: Tensor,
        phi: Tensor,
        height: int,
        width: int,
        kernel_size: int,
        padding: int,
    ) -> None:
        decoder = RansDecoder()
        decoder.set_stream(y_string)
        masked_weight = self.context_prediction.weight * self.context_prediction.mask

        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                context = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )
                hyper = phi[:, :, h : h + 1, w : w + 1]
                params = self.entropy_parameters(torch.cat((context, hyper), dim=1))
                means, scales, weights_g, weights_l, weights_lo, weights_family = (
                    _split_gllmm_params(params, self.M)
                )
                cdf = self.gaussian_conditional.build_cdf(
                    means,
                    scales,
                    weights_g,
                    weights_l,
                    weights_lo,
                    weights_family,
                    abs_max,
                )
                indexes = list(range(cdf.size(0)))
                values = decoder.decode_stream(
                    indexes,
                    cdf.cpu().tolist(),
                    [cdf.size(1)] * cdf.size(0),
                    [0] * cdf.size(0),
                )
                decoded = (
                    torch.tensor(values, device=y_hat.device, dtype=y_hat.dtype)
                    .reshape(1, -1, 1, 1)
                    .sub_(abs_max)
                )
                y_hat[:, :, h + padding : h + padding + 1, w + padding : w + padding + 1] = decoded

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Tensor]) -> "GLLMM":
        clean_state = _strip_module_prefixes(state_dict)
        channels = int(clean_state["g_a.block0_down.shortcut.weight"].size(0))
        net = cls(N=channels)
        net.load_state_dict(clean_state)
        return net


def _strip_module_prefixes(state_dict: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    clean: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        while key.startswith("module."):
            key = key.removeprefix("module.")
        clean[key] = value
    return clean
