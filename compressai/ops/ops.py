# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

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

import math
from typing import Optional, Tuple, Union

import torch

from torch import Tensor, nn


def compute_padding(in_h: int, in_w: int, *, out_h=None, out_w=None, min_div=1):
    """Returns tuples for padding and unpadding.

    Args:
        in_h: Input height.
        in_w: Input width.
        out_h: Output height.
        out_w: Output width.
        min_div: Length that output dimensions should be divisible by.
    """
    if out_h is None:
        out_h = (in_h + min_div - 1) // min_div * min_div
    if out_w is None:
        out_w = (in_w + min_div - 1) // min_div * min_div

    if out_h % min_div != 0 or out_w % min_div != 0:
        raise ValueError(
            f"Padded output height and width are not divisible by min_div={min_div}."
        )

    left = (out_w - in_w) // 2
    right = out_w - in_w - left
    top = (out_h - in_h) // 2
    bottom = out_h - in_h - top

    pad = (left, right, top, bottom)
    unpad = (-left, -right, -top, -bottom)

    return pad, unpad


def quantize_ste(x: Tensor) -> Tensor:
    """
    Rounding with non-zero gradients. Gradients are approximated by replacing
    the derivative by the identity function.

    Used in `"Lossy Image Compression with Compressive Autoencoders"
    <https://arxiv.org/abs/1703.00395>`_

    .. note::

        Implemented with the pytorch `detach()` reparametrization trick:

        `x_round = x_round - x.detach() + x`
    """
    return (torch.round(x) - x).detach() + x


def _soft_lattice_round(x: Tensor) -> Tensor:
    return x + 0.2 * torch.cos(2 * math.pi * (x + 0.25))


def _round_lattice_values(x: Tensor, round_mode: str) -> Tensor:
    if round_mode == "hard":
        return torch.round(x)
    if round_mode == "ste":
        return quantize_ste(x)
    if round_mode == "soft":
        return _soft_lattice_round(x)
    raise ValueError(f'Invalid round_mode "{round_mode}".')


def _prepare_lattice_step(
    step: Union[float, Tensor],
    inputs: Tensor,
    block_axis: str,
) -> Tensor:
    step_tensor = torch.as_tensor(step, dtype=inputs.dtype, device=inputs.device)

    if step_tensor.ndim == 1 and block_axis == "channel":
        if inputs.ndim < 2 or step_tensor.numel() != inputs.shape[1]:
            raise ValueError(
                "A 1D step tensor for channel blocks must match inputs.shape[1]."
            )
        step_shape = [1] * inputs.ndim
        step_shape[1] = inputs.shape[1]
        step_tensor = step_tensor.reshape(step_shape)

    if torch.any(step_tensor <= 0).item():
        raise ValueError("step must be strictly positive.")

    return step_tensor


def _diamond_lattice_distance(
    inputs: Tensor,
    outputs: Tensor,
    block_axis: str,
) -> Tensor:
    distance = (inputs - outputs).pow(2)
    if block_axis == "channel":
        return distance.sum(dim=1, keepdim=True)
    if block_axis == "flat":
        return distance.sum()
    raise ValueError(f'Invalid block_axis "{block_axis}".')


def diamond_lattice_quantize(
    inputs: Tensor,
    *,
    step: Union[float, Tensor] = 1.0,
    round_mode: str = "ste",
    block_axis: str = "channel",
    return_indexes: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Quantize with a two-coset diamond lattice forward proxy.

    The lattice is represented as the union of an integer lattice and a
    half-step shifted coset. This helper is intended for training/evaluation
    forward paths and does not define entropy coding symbols by itself.
    """
    if block_axis not in ("channel", "flat"):
        raise ValueError(f'Invalid block_axis "{block_axis}".')
    if block_axis == "channel" and inputs.ndim < 2:
        raise ValueError('block_axis="channel" requires inputs.ndim >= 2.')
    if round_mode not in ("hard", "ste", "soft"):
        raise ValueError(f'Invalid round_mode "{round_mode}".')

    step_tensor = _prepare_lattice_step(step, inputs, block_axis)
    scaled_inputs = inputs / step_tensor

    base_hard = torch.round(scaled_inputs)
    shifted_hard = torch.round(scaled_inputs - 0.5) + 0.5

    base_distance = _diamond_lattice_distance(
        scaled_inputs,
        base_hard,
        block_axis,
    )
    shifted_distance = _diamond_lattice_distance(
        scaled_inputs,
        shifted_hard,
        block_axis,
    )
    use_shifted = shifted_distance < base_distance

    if round_mode == "soft":
        weights = torch.softmax(
            torch.stack((-base_distance, -shifted_distance), dim=0),
            dim=0,
        )
        base_quantized = _round_lattice_values(scaled_inputs, round_mode)
        shifted_quantized = (
            _round_lattice_values(scaled_inputs - 0.5, round_mode) + 0.5
        )
        quantized = weights[0] * base_quantized + weights[1] * shifted_quantized
    else:
        base_quantized = _round_lattice_values(scaled_inputs, round_mode)
        shifted_quantized = (
            _round_lattice_values(scaled_inputs - 0.5, round_mode) + 0.5
        )
        quantized = torch.where(use_shifted, shifted_quantized, base_quantized)

    outputs = quantized * step_tensor
    if not return_indexes:
        return outputs

    indexes = use_shifted.to(torch.int64)
    if block_axis == "channel":
        indexes = indexes.squeeze(1)
    return outputs, indexes


class DiamondLatticeQuantizer(nn.Module):
    """Module wrapper for :func:`diamond_lattice_quantize`."""

    def __init__(
        self,
        *,
        step: Union[float, Tensor] = 1.0,
        round_mode: str = "ste",
        block_axis: str = "channel",
        return_indexes: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("step", torch.as_tensor(step).clone().detach())
        self.round_mode = round_mode
        self.block_axis = block_axis
        self.return_indexes = return_indexes

    def forward(
        self,
        inputs: Tensor,
        *,
        step: Optional[Union[float, Tensor]] = None,
        return_indexes: Optional[bool] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if step is None:
            step = self.step
        if return_indexes is None:
            return_indexes = self.return_indexes
        return diamond_lattice_quantize(
            inputs,
            step=step,
            round_mode=self.round_mode,
            block_axis=self.block_axis,
            return_indexes=return_indexes,
        )
