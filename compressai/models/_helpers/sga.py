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

"""Model-side helpers for SGA inference-time refinement."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.latent_codecs import (
    CheckerboardLatentCodec,
    EntropyBottleneckLatentCodec,
    GaussianConditionalLatentCodec,
    HyperpriorLatentCodec,
    MultiContextCheckerboardLatentCodec,
)
from compressai.ops import SGAQuantizer

__all__ = [
    "SGARefinementMixin",
    "apply_sga_quantizer",
    "hyperprior_refine_extract",
    "hyperprior_refine_forward",
]


_SgaState = Dict[int, Tuple[nn.Module, str, object]]


def _sga_capable_codecs(model: nn.Module):
    types = (
        CheckerboardLatentCodec,
        EntropyBottleneckLatentCodec,
        GaussianConditionalLatentCodec,
        MultiContextCheckerboardLatentCodec,
    )
    for module in model.modules():
        if isinstance(module, types):
            yield module


def apply_sga_quantizer(model: nn.Module, sga: Optional[SGAQuantizer]) -> None:
    """Switch SGA-capable latent codecs to SGA mode and restore defaults.

    Original ``quantizer`` / ``sga`` attributes are recorded on first enable
    and restored when ``sga`` is ``None``. This keeps the helper valid for
    models whose ``z`` codec defaults to either ``"ste"`` or ``"noise"``.
    """
    state: _SgaState = getattr(model, "_sga_quantizer_state", {})

    if sga is None:
        for module, quantizer, original_sga in state.values():
            module.quantizer = quantizer
            module.sga = original_sga
        state.clear()
        model._sga_quantizer_state = state
        return

    for module in _sga_capable_codecs(model):
        if not hasattr(module, "quantizer"):
            continue
        module_id = id(module)
        if module_id not in state:
            state[module_id] = (
                module,
                module.quantizer,
                getattr(module, "sga", None),
            )
        module.quantizer = "sga"
        module.sga = sga
    model._sga_quantizer_state = state


@torch.no_grad()
def hyperprior_refine_extract(model: nn.Module, x: Tensor) -> Tuple[Tensor, Tensor]:
    """Run ``g_a`` and ``h_a`` once for a hyperprior-backed image model."""
    latent_codec = model.latent_codec
    if not isinstance(latent_codec, HyperpriorLatentCodec):
        raise TypeError("SGA refinement expects a HyperpriorLatentCodec")
    y = model.g_a(x)
    z = latent_codec.h_a(y)
    return y, z


def hyperprior_refine_forward(
    model: nn.Module,
    y: Tensor,
    z: Tensor,
) -> Dict[str, Dict[str, Tensor] | Tensor]:
    """Forward pass with externally supplied hyperprior latents."""
    latent_codec = model.latent_codec
    if not isinstance(latent_codec, HyperpriorLatentCodec):
        raise TypeError("SGA refinement expects a HyperpriorLatentCodec")
    z_out = latent_codec.latent_codec["z"](z)
    z_hat = z_out["y_hat"]
    params = latent_codec.h_s(z_hat)
    y_out = latent_codec.latent_codec["y"](y, params)
    return {
        "x_hat": model.g_s(y_out["y_hat"]),
        "likelihoods": {
            "y": y_out["likelihoods"]["y"],
            "z": z_out["likelihoods"]["y"],
        },
    }


class SGARefinementMixin:
    """Reusable SGA refinement interface for hyperprior-backed image models."""

    def refine_extract(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return hyperprior_refine_extract(self, x)

    def refine_forward(
        self,
        y: Tensor,
        z: Tensor,
    ) -> Dict[str, Dict[str, Tensor] | Tensor]:
        return hyperprior_refine_forward(self, y, z)

    def set_sga_mode(self, sga: Optional[SGAQuantizer]) -> None:
        apply_sga_quantizer(self, sga)
