# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
# Adapted from https://github.com/NJUVISION/NIC.

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianMixtureConditional
from compressai.layers import conv, conv1x1, conv3x3, deconv
from compressai.models.base import CompressionModel
from compressai.registry import register_model

__all__ = ["NIC"]


_RESIZABLE_BUFFERS = {"_quantized_cdf", "_offset", "_cdf_length", "scale_table"}


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = conv3x3(channels, channels)
        self.conv2 = conv3x3(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(F.relu(self.conv1(x), inplace=True))


class _NonLocalBlock(nn.Module):
    def __init__(self, in_channels: int, inner_channels: int) -> None:
        super().__init__()
        self.in_channel = int(in_channels)
        self.out_channel = int(inner_channels)
        self.g = conv1x1(self.in_channel, self.out_channel)
        self.theta = conv1x1(self.in_channel, self.out_channel)
        self.phi = conv1x1(self.in_channel, self.out_channel)
        self.W = conv1x1(self.out_channel, self.in_channel)
        nn.init.zeros_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.out_channel, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.out_channel, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.out_channel, -1)

        attention = torch.matmul(theta_x, phi_x).softmax(dim=-1)
        y = torch.matmul(attention, g_x).permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.out_channel, *x.size()[2:])
        return x + self.W(y)


class _MaskedConv3d(nn.Conv3d):
    def __init__(
        self,
        mask_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=True,
        )
        if mask_type not in {"A", "B"}:
            raise ValueError(f'Invalid mask type "{mask_type}"')

        mask = torch.zeros_like(self.weight)
        center = kernel_size**3 // 2
        flat = mask.view(out_channels, in_channels, -1)
        flat[:, :, :center] = 1
        if mask_type == "B":
            flat[:, :, center] = 1
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv3d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _AnalysisTransform(nn.Module):
    def __init__(self, input_channels: int, N2: int, M: int, M1: int) -> None:
        super().__init__()
        self.conv1 = conv(input_channels, M1, kernel_size=5, stride=1)
        self.trunk1 = nn.Sequential(_ResBlock(M1), _ResBlock(M1), conv(M1, 2 * M1))
        self.down1 = conv(2 * M1, M)
        self.trunk2 = nn.Sequential(
            _ResBlock(2 * M1), _ResBlock(2 * M1), _ResBlock(2 * M1)
        )
        self.trunk3 = nn.Sequential(
            _ResBlock(M), _ResBlock(M), _ResBlock(M), conv(M, M)
        )
        self.trunk4 = nn.Sequential(
            _ResBlock(M), _ResBlock(M), _ResBlock(M), conv(M, M)
        )
        self.trunk5 = nn.Sequential(_ResBlock(M), _ResBlock(M), _ResBlock(M))
        self.mask2 = nn.Sequential(
            _NonLocalBlock(M, M // 2),
            _ResBlock(M),
            _ResBlock(M),
            _ResBlock(M),
            conv1x1(M, M),
        )
        self.trunk6 = nn.Sequential(_ResBlock(M), _ResBlock(M), conv(M, M))
        self.trunk7 = nn.Sequential(_ResBlock(M), _ResBlock(M), conv(M, M))
        self.trunk8 = nn.Sequential(_ResBlock(M), _ResBlock(M), _ResBlock(M))
        self.mask3 = nn.Sequential(
            _NonLocalBlock(M, M // 2),
            _ResBlock(M),
            _ResBlock(M),
            _ResBlock(M),
            conv1x1(M, M),
        )
        self.conv2 = conv3x3(M, N2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv1(x)
        x = self.trunk1(x)
        x = self.down1(self.trunk2(x) + x)
        x = self.trunk3(x)
        x = self.trunk4(x)
        y = self.trunk5(x) * torch.sigmoid(self.mask2(x)) + x

        z = self.trunk6(y)
        z = self.trunk7(z)
        z = self.trunk8(z) * torch.sigmoid(self.mask3(z)) + z
        z = self.conv2(z)
        return y, z


class _HyperSynthesisTransform(nn.Module):
    def __init__(self, N2: int, M: int) -> None:
        super().__init__()
        self.conv1 = conv3x3(N2, M)
        self.trunk1 = nn.Sequential(_ResBlock(M), _ResBlock(M), _ResBlock(M))
        self.mask1 = nn.Sequential(
            _NonLocalBlock(M, M // 2),
            _ResBlock(M),
            _ResBlock(M),
            _ResBlock(M),
            conv1x1(M, M),
        )
        self.trunk2 = nn.Sequential(_ResBlock(M), _ResBlock(M), deconv(M, M))
        self.trunk3 = nn.Sequential(_ResBlock(M), _ResBlock(M), deconv(M, M))

    def forward(self, z_hat: Tensor) -> Tensor:
        x = self.conv1(z_hat)
        x = self.trunk1(x) * torch.sigmoid(self.mask1(x)) + x
        x = self.trunk2(x)
        return self.trunk3(x)


class _HyperParameterModel(nn.Module):
    def __init__(self, M: int) -> None:
        super().__init__()
        self.context_p = nn.Sequential(
            _ResBlock(M),
            _ResBlock(M),
            _ResBlock(M),
            conv3x3(M, 2 * M),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.context_p(x)


class _WeightedGaussianContext(nn.Module):
    def __init__(self, M: int) -> None:
        super().__init__()
        self.conv1 = _MaskedConv3d("A", 1, 24, kernel_size=11, stride=1, padding=5)
        self.conv2 = nn.Sequential(
            nn.Conv3d(25, 48, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(48, 96, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 9, kernel_size=1),
        )
        self.conv3 = conv3x3(2 * M, M)

    def forward(self, y_hat: Tensor, hyper_params: Tensor) -> Tensor:
        y_context = self.conv1(y_hat.unsqueeze(1))
        hyper = self.conv3(hyper_params).unsqueeze(1)
        return self.conv2(torch.cat((y_context, hyper), dim=1))


@register_model("nic")
class NIC(CompressionModel):
    """NLAIC/NIC image compression model adapted to the CompressAI interface."""

    def __init__(
        self,
        input_channels: int = 3,
        N2: int = 128,
        M: int = 192,
        M1: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_channels = int(input_channels)
        self.N2 = int(N2)
        self.M = int(M)
        self.M1 = int(M1 if M1 is not None else M // 2)

        self.encoder = _AnalysisTransform(self.input_channels, self.N2, self.M, self.M1)
        self.entropy_bottleneck = EntropyBottleneck(self.N2, filters=(3, 3, 3))
        self.hyper_dec = _HyperSynthesisTransform(self.N2, self.M)
        self.p = _HyperParameterModel(self.M)
        self.context = _WeightedGaussianContext(self.M)
        self.gaussian_conditional = GaussianMixtureConditional(K=3)
        self.decoder = _SynthesisTransform(self.input_channels, self.M, self.M1)

    @property
    def downsampling_factor(self) -> int:
        return 2**4

    def _split_context_parameters(
        self, params: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        parts = [part.squeeze(1) for part in params.chunk(9, dim=1)]
        logits = torch.stack((parts[0], parts[3], parts[6]), dim=1)
        weights = logits.softmax(dim=1).flatten(1, 2)
        means = torch.cat((parts[1], parts[4], parts[7]), dim=1)
        scales = torch.cat((parts[2], parts[5], parts[8]), dim=1).abs()
        return scales, means, weights

    def forward(self, x: Tensor) -> Dict[str, Any]:
        y, z = self.encoder(x)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyper_params = self.p(self.hyper_dec(z_hat))

        y_hat = self.gaussian_conditional.quantize(
            y,
            "noise" if self.training else "dequantize",
        )
        context_params = self.context(y_hat, hyper_params)
        scales, means, weights = self._split_context_parameters(context_params)
        y_likelihoods = self.gaussian_conditional._likelihood(
            y_hat,
            scales,
            means,
            weights,
        )
        if self.gaussian_conditional.use_likelihood_bound:
            y_likelihoods = self.gaussian_conditional.likelihood_lower_bound(
                y_likelihoods
            )

        return {
            "x_hat": self.decoder(y_hat),
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods,
            },
        }

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Tensor]) -> "NIC":
        clean_state = _strip_module_prefixes(state_dict)
        input_channels, N2, M, M1 = _infer_model_shape(clean_state)
        net = cls(input_channels=input_channels, N2=N2, M=M, M1=M1)
        migrated = _migrate_state_dict(clean_state, net.state_dict())
        net.load_state_dict(migrated)
        return net


class _SynthesisTransform(nn.Module):
    def __init__(self, output_channels: int, M: int, M1: int) -> None:
        super().__init__()
        self.trunk1 = nn.Sequential(_ResBlock(M), _ResBlock(M), _ResBlock(M))
        self.mask1 = nn.Sequential(
            _NonLocalBlock(M, M // 2),
            _ResBlock(M),
            _ResBlock(M),
            _ResBlock(M),
            conv1x1(M, M),
        )
        self.up1 = deconv(M, M)
        self.trunk2 = nn.Sequential(
            _ResBlock(M), _ResBlock(M), _ResBlock(M), deconv(M, M)
        )
        self.trunk3 = nn.Sequential(
            _ResBlock(M), _ResBlock(M), _ResBlock(M), deconv(M, 2 * M1)
        )
        self.trunk4 = nn.Sequential(
            _ResBlock(2 * M1), _ResBlock(2 * M1), _ResBlock(2 * M1)
        )
        self.trunk5 = nn.Sequential(
            deconv(2 * M1, M1),
            _ResBlock(M1),
            _ResBlock(M1),
            _ResBlock(M1),
        )
        self.conv1 = conv(M1, output_channels, kernel_size=5, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.trunk1(x) * torch.sigmoid(self.mask1(x)) + x
        x = self.up1(x)
        x = self.trunk2(x)
        x = self.trunk3(x)
        x = self.trunk4(x) + x
        x = self.trunk5(x)
        return self.conv1(x)


def _strip_module_prefixes(state_dict: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    clean: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        while key.startswith("module."):
            key = key.removeprefix("module.")
        clean[key] = value
    return clean


def _infer_model_shape(state_dict: Mapping[str, Tensor]) -> Tuple[int, int, int, int]:
    input_channels = 3
    M1 = 96
    M = 192
    N2 = 128

    if "encoder.conv1.weight" in state_dict:
        conv1 = state_dict["encoder.conv1.weight"]
        input_channels = int(conv1.size(1))
        M1 = int(conv1.size(0))
    if "encoder.conv2.weight" in state_dict:
        conv2 = state_dict["encoder.conv2.weight"]
        N2 = int(conv2.size(0))
        M = int(conv2.size(1))
    elif "context.conv3.weight" in state_dict:
        M = int(state_dict["context.conv3.weight"].size(0))
    elif "entropy_bottleneck.matrices.0" in state_dict:
        N2 = int(state_dict["entropy_bottleneck.matrices.0"].size(0))
    return input_channels, N2, M, M1


def _migrate_state_dict(
    state_dict: Mapping[str, Tensor],
    target_state_dict: Mapping[str, Tensor],
) -> Dict[str, Tensor]:
    migrated = dict(target_state_dict)
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("factorized_entropy_func._matrices."):
            new_key = key.replace(
                "factorized_entropy_func._matrices.",
                "entropy_bottleneck.matrices.",
                1,
            )
        elif key.startswith("factorized_entropy_func._bias."):
            new_key = key.replace(
                "factorized_entropy_func._bias.",
                "entropy_bottleneck.biases.",
                1,
            )
        elif key.startswith("factorized_entropy_func._factor."):
            new_key = key.replace(
                "factorized_entropy_func._factor.",
                "entropy_bottleneck.factors.",
                1,
            )

        if new_key in migrated and (
            migrated[new_key].shape == value.shape
            or new_key.rsplit(".", 1)[-1] in _RESIZABLE_BUFFERS
        ):
            migrated[new_key] = value
    return migrated
