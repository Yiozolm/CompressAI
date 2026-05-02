from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from torch import Tensor

from compressai.layers import conv
from compressai.layers.layers import conv1x1
from compressai.layers.lic.blocks import LayerNorm2d
from compressai.ops import quantize_ste

from .entropy_models import EntropyBottleneck, GaussianConditional

__all__ = [
    "CausalContextAdjustmentEntropyModel",
    "has_cca_aux_state",
    "infer_cca_hidden_channels",
    "infer_cca_num_layers",
]


# ---------------------------------------------------------------------------
# NAF (Non-linear Activation Free) building blocks
# (formerly compressai/layers/lic/cca.py; private to the CCA model + entropy model)
# ---------------------------------------------------------------------------


class _SimpleGate(nn.Module):
    def forward(self, input_tensor: Tensor) -> Tensor:
        gate_tensor, value_tensor = input_tensor.chunk(2, dim=1)
        return gate_tensor * value_tensor


class _NAFBlock(nn.Module):
    """Non-linear Activation Free residual block.

    Used by both the CCA entropy-model auxiliary transforms and the CCA
    image-compression model's analysis / synthesis stacks. State-dict keys
    (``norm1`` / ``pointwise_depthwise`` / ``channel_attention`` / ``project``
    / ``feed_forward`` / ``beta`` / ``gamma``) match upstream so released
    checkpoints load 1:1.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        expanded_channels = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.pointwise_depthwise = nn.Sequential(
            conv1x1(channels, expanded_channels),
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=3,
                padding=1,
                groups=expanded_channels,
            ),
        )
        self.gate = _SimpleGate()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(channels, channels),
        )
        self.project = conv1x1(channels, channels)
        self.norm2 = LayerNorm2d(channels)
        self.feed_forward = nn.Sequential(
            conv1x1(channels, expanded_channels),
            _SimpleGate(),
            conv1x1(channels, channels),
        )
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.norm1(input_tensor)
        output = self.pointwise_depthwise(output)
        output = self.gate(output)
        output = output * self.channel_attention(output)
        output = self.project(output)
        output = input_tensor + self.beta * output
        return output + self.gamma * self.feed_forward(self.norm2(output))


class _NAFTransform(nn.Module):
    """``Conv1x1 -> NAFBlock x N -> Conv1x1`` per-slice support transform."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.input_projection = conv1x1(input_channels, hidden_channels)
        self.blocks = nn.Sequential(
            *(_NAFBlock(hidden_channels) for _ in range(num_layers))
        )
        self.output_projection = conv1x1(hidden_channels, output_channels)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.input_projection(input_tensor)
        return self.output_projection(output + self.blocks(output))


# ---------------------------------------------------------------------------
# CCA entropy model
# ---------------------------------------------------------------------------


_CCA_PREFIX = "cca_aux_entropy_model."


def has_cca_aux_state(state_dict: Dict[str, Tensor], prefix: str = _CCA_PREFIX) -> bool:
    return any(key.startswith(prefix) for key in state_dict)


def infer_cca_hidden_channels(
    state_dict: Dict[str, Tensor],
    prefix: str = _CCA_PREFIX,
    default: int = 224,
) -> int:
    key = f"{prefix}mean_support_transforms.0.input_projection.weight"
    if key in state_dict:
        return int(state_dict[key].size(0))
    return default


def infer_cca_num_layers(
    state_dict: Dict[str, Tensor],
    prefix: str = _CCA_PREFIX,
    default: int = 4,
) -> int:
    block_prefix = f"{prefix}mean_support_transforms.0.blocks."
    block_indices = {
        int(key[len(block_prefix) :].split(".", 1)[0])
        for key in state_dict
        if key.startswith(block_prefix) and key.endswith(".beta")
    }
    return len(block_indices) or default


def _make_prediction_head(
    input_channels: int,
    hidden_channels: int,
    output_channels: int,
) -> nn.Sequential:
    mid_channels = max(hidden_channels // 2, output_channels)
    return nn.Sequential(
        conv(input_channels, hidden_channels, kernel_size=3, stride=1),
        nn.GELU(),
        conv(hidden_channels, mid_channels, kernel_size=3, stride=1),
        nn.GELU(),
        conv(mid_channels, output_channels, kernel_size=3, stride=1),
    )


class CausalContextAdjustmentEntropyModel(nn.Module):
    r"""Causal Context Adjustment entropy model from M. Han, S. Jiang, S. Li,
    X. Deng, M. Xu, C. Zhu, S. Gu: `"Causal Context Adjustment Loss for
    Learned Image Compression" <https://arxiv.org/abs/2410.04847>`_, Adv. in
    Neural Information Processing Systems 38 (NeurIPS), 2024.

    Augments a Minnen2020-style channel-wise autoregressive entropy model
    with an auxiliary CCA branch that produces ``y_cca`` likelihoods used by
    :class:`compressai.losses.CCARateDistortionLoss` to align the causal
    context with the rate-distortion objective.
    """

    y_entropy_bottleneck: EntropyBottleneck
    gaussian_conditional: GaussianConditional

    def __init__(
        self,
        latent_channels: int,
        num_slices: int,
        hidden_channels: int = 224,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        if latent_channels % num_slices != 0:
            raise ValueError("latent_channels must be divisible by num_slices")

        self.latent_channels = int(latent_channels)
        self.num_slices = int(num_slices)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.slice_channels = self.latent_channels // self.num_slices

        def support_channels(index: int) -> int:
            return self.latent_channels + self.slice_channels * max(index - 1, 0)

        self.mean_support_transforms = nn.ModuleList(
            _NAFTransform(
                support_channels(index),
                support_channels(index),
                self.hidden_channels,
                self.num_layers,
            )
            for index in range(self.num_slices)
        )
        self.scale_support_transforms = nn.ModuleList(
            _NAFTransform(
                support_channels(index),
                support_channels(index),
                self.hidden_channels,
                self.num_layers,
            )
            for index in range(self.num_slices)
        )
        self.mean_cc_transforms = nn.ModuleList(
            _make_prediction_head(
                support_channels(index),
                self.hidden_channels,
                self.slice_channels,
            )
            for index in range(self.num_slices)
        )
        self.scale_cc_transforms = nn.ModuleList(
            _make_prediction_head(
                support_channels(index),
                self.hidden_channels,
                self.slice_channels,
            )
            for index in range(self.num_slices)
        )
        self.lrp_transforms = nn.ModuleList(
            _make_prediction_head(
                support_channels(index) + self.slice_channels,
                self.hidden_channels,
                self.slice_channels,
            )
            for index in range(max(self.num_slices - 2, 0))
        )

        self.y_entropy_bottleneck = EntropyBottleneck(self.latent_channels)
        self.gaussian_conditional = GaussianConditional(None)

    def _support_slices(
        self,
        slice_index: int,
        y_hat_slices: List[Tensor],
    ) -> List[Tensor]:
        return y_hat_slices[: max(slice_index - 1, 0)]

    def _apply_lrp(
        self,
        slice_index: int,
        mean_support: Tensor,
        y_hat_slice: Tensor,
    ) -> Tensor:
        lrp = self.lrp_transforms[slice_index](torch.cat([mean_support, y_hat_slice], dim=1))
        return y_hat_slice + 0.5 * torch.tanh(lrp)

    def forward(
        self,
        y: Tensor,
        latent_means: Tensor,
        latent_scales: Tensor,
    ) -> Dict[str, Tensor]:
        y_hat_slices: List[Tensor] = []
        y_likelihoods: List[Tensor] = []

        _, y_aux_likelihoods = self.y_entropy_bottleneck(y)

        for slice_index, y_slice in enumerate(y.chunk(self.num_slices, dim=1)):
            support_slices = self._support_slices(slice_index, y_hat_slices)
            mean_support = torch.cat([latent_means, *support_slices], dim=1)
            mean_support = self.mean_support_transforms[slice_index](mean_support)
            mu = self.mean_cc_transforms[slice_index](mean_support)

            scale_support = torch.cat([latent_scales, *support_slices], dim=1)
            scale_support = self.scale_support_transforms[slice_index](scale_support)
            scale = self.scale_cc_transforms[slice_index](scale_support)

            _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scale, means=mu)
            y_likelihoods.append(y_slice_likelihoods)

            if slice_index >= len(self.lrp_transforms):
                continue

            y_hat_slice = quantize_ste(y_slice - mu) + mu
            y_hat_slices.append(self._apply_lrp(slice_index, mean_support, y_hat_slice))

        return {
            "y_aux": y_aux_likelihoods,
            "y_cca": torch.cat(y_likelihoods, dim=1),
        }
