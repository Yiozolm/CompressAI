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

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyModel
from compressai.layers import CheckerboardMaskedConv2d
from compressai.ops import SGAQuantizer, quantize_ste
from compressai.registry import register_module

from . import _checkerboard_helpers as _ckb
from .base import LatentCodec

__all__ = [
    "CheckerboardLatentCodec",
]


@register_module("CheckerboardLatentCodec")
class CheckerboardLatentCodec(LatentCodec):
    """Reconstructs latent using 2-pass context model with checkerboard anchors.

    Checkerboard context model introduced in [He2021].

    See :py:class:`~compressai.models.sensetime.Cheng2020AnchorCheckerboard`
    for example usage.

    - `forward_method="onepass"` is fastest, but does not use
      quantization based on the intermediate means.
      Uses noise to model quantization.
    - `forward_method="twopass"` is slightly slower, but accurately
      quantizes via STE based on the intermediate means.
      Uses the same operations as [Chandelier2023].
    - `forward_method="twopass_faster"` uses slightly fewer
      redundant operations.

    [He2021]: `"Checkerboard Context Model for Efficient Learned Image
    Compression" <https://arxiv.org/abs/2103.15306>`_, by Dailan He,
    Yaoyan Zheng, Baocheng Sun, Yan Wang, and Hongwei Qin, CVPR 2021.

    [Chandelier2023]: `"ELiC-ReImplemetation"
    <https://github.com/VincentChandelier/ELiC-ReImplemetation>`_, by
    Vincent Chandelier, 2023.

    .. warning:: This implementation assumes that ``entropy_parameters``
       is a pointwise function, e.g., a composition of 1x1 convs and
       pointwise nonlinearities.

    .. code-block:: none

        0. Input:

        □ □ □ □
        □ □ □ □
        □ □ □ □

        1. Decode anchors:

        ◌ □ ◌ □
        □ ◌ □ ◌
        ◌ □ ◌ □

        2. Decode non-anchors:

        ■ ◌ ■ ◌
        ◌ ■ ◌ ■
        ■ ◌ ■ ◌

        3. End result:

        ■ ■ ■ ■
        ■ ■ ■ ■
        ■ ■ ■ ■

        LEGEND:
        ■   decoded
        ◌   currently decoding
        □   empty
    """

    def __init__(
        self,
        latent_codec: Mapping[str, LatentCodec],
        entropy_parameters: nn.Module,
        context_prediction: CheckerboardMaskedConv2d,
        anchor_parity="even",
        forward_method="twopass",
        quantizer: str = "ste",
        sga: Optional[SGAQuantizer] = None,
        **kwargs,
    ):
        super().__init__()
        if quantizer not in ("ste", "sga"):
            raise ValueError(f'Invalid quantizer "{quantizer}"')
        if quantizer == "sga" and sga is None:
            raise ValueError('quantizer="sga" requires the `sga` argument')
        self._kwargs = kwargs
        self.anchor_parity = anchor_parity
        self.non_anchor_parity = {"odd": "even", "even": "odd"}[anchor_parity]
        self.forward_method = forward_method
        self.quantizer = quantizer
        self.sga = sga
        self.entropy_parameters = entropy_parameters
        self.context_prediction = context_prediction
        self.y = latent_codec["y"]
        self.latent_codec = latent_codec

    def forward(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        if self.forward_method == "onepass":
            return self._forward_onepass(y, side_params)
        if self.forward_method == "twopass":
            return self._forward_twopass(y, side_params)
        if self.forward_method == "twopass_faster":
            return self._forward_twopass_faster(y, side_params)
        raise ValueError(f"Unknown forward method: {self.forward_method}")

    def _forward_onepass(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Fast estimation with single pass of the entropy parameters network.

        It is faster than the twopass method (only one pass required!),
        but also less accurate.

        This method uses uniform noise to roughly model quantization.
        """
        if self.quantizer == "sga":
            raise ValueError(
                'quantizer="sga" requires forward_method="twopass" '
                'or forward_method="twopass_faster"'
            )
        y_hat = self.quantize(y)
        y_ctx = _ckb.mask_all_but_step(
            self.context_prediction(y_hat),
            "non_anchor",
            anchor_parity=self.anchor_parity,
        )
        params = self.entropy_parameters(_ckb.merge(y_ctx, side_params))
        y_out = self.latent_codec["y"](y, params)
        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            "y_hat": y_hat,
        }

    def _forward_twopass(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Runs the entropy parameters network in two passes.

        The first pass gets ``y_hat`` and ``means_hat`` for the anchors.
        This ``y_hat`` is used as context to predict the non-anchors.
        The second pass gets ``y_hat`` for the non-anchors.
        The two ``y_hat`` tensors are then combined. The resulting
        ``y_hat`` models the effects of quantization more realistically.

        To compute ``y_hat_anchors``, we need the predicted ``means_hat``:
        ``y_hat = quantize_ste(y - means_hat) + means_hat``.
        Thus, two passes of ``entropy_parameters`` are necessary.
        """
        B, C, H, W = y.shape
        params = y.new_zeros((B, C * 2, H, W))
        y_hat_ = []

        for step in ("anchor", "non_anchor"):
            # Determine y_ctx for current step.
            if step == "anchor":
                y_ctx_i = self._y_ctx_zero(y)
            else:  # step == "non_anchor"
                y_ctx_i = self.context_prediction(y_hat_[0])

            # Determine params for current step.
            params_i = self.entropy_parameters(_ckb.merge(y_ctx_i, side_params))
            params_i = _ckb.mask_all_but_step(
                params_i, step, anchor_parity=self.anchor_parity
            )
            _ckb.write_step(params, params_i, step, anchor_parity=self.anchor_parity)

            # Determine y_hat for current step.
            _, means_i = self.latent_codec["y"]._chunk(params_i)
            y_i = _ckb.mask_all_but_step(y, step, anchor_parity=self.anchor_parity)
            y_hat_i = self._quantize(y_i, means_i)
            y_hat_i = _ckb.mask_all_but_step(
                y_hat_i, step, anchor_parity=self.anchor_parity
            )
            y_hat_.append(y_hat_i)

        [y_hat_anchors, y_hat_non_anchors] = y_hat_
        y_hat = y_hat_anchors + y_hat_non_anchors
        if self.quantizer == "sga":
            y_likelihoods = self._likelihood_for_quantized(y_hat, params)
        else:
            y_out = self.latent_codec["y"](y, params)
            y_likelihoods = y_out["likelihoods"]["y"]

        return {
            "likelihoods": {
                "y": y_likelihoods,
            },
            "y_hat": y_hat,
        }

    def _forward_twopass_faster(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Runs the entropy parameters network in two passes.

        This version was written based on the paper description.
        It is a tiny bit faster than the twopass method since
        it avoids a few redundant operations. The "probably unnecessary"
        operations can likely be removed as well.
        The speedup is very small, however.
        """
        y_ctx = self._y_ctx_zero(y)
        params = self.entropy_parameters(_ckb.merge(y_ctx, side_params))
        params = _ckb.mask_all_but_step(
            params, "anchor", anchor_parity=self.anchor_parity
        )  # Probably unnecessary.
        _, means_hat = self.latent_codec["y"]._chunk(params)
        y_hat_anchors = self._quantize(y, means_hat)
        y_hat_anchors = _ckb.mask_all_but_step(
            y_hat_anchors, "anchor", anchor_parity=self.anchor_parity
        )

        y_ctx = self.context_prediction(y_hat_anchors)
        y_ctx = _ckb.mask_all_but_step(
            y_ctx, "non_anchor", anchor_parity=self.anchor_parity
        )  # Probably unnecessary.
        params = self.entropy_parameters(_ckb.merge(y_ctx, side_params))
        if self.quantizer == "sga":
            _, means_hat = self.latent_codec["y"]._chunk(params)
            y_hat = self._quantize(y, means_hat)
        else:
            y_out = self.latent_codec["y"](y, params)
            y_hat = y_out["y_hat"]
            y_likelihoods = y_out["likelihoods"]["y"]

        # Reuse quantized anchors that were used for non-anchor context prediction.
        _ckb.write_step(
            y_hat, y_hat_anchors, "anchor", anchor_parity=self.anchor_parity
        )  # Probably unnecessary.
        if self.quantizer == "sga":
            y_likelihoods = self._likelihood_for_quantized(y_hat, params)

        return {
            "likelihoods": {
                "y": y_likelihoods,
            },
            "y_hat": y_hat,
        }

    @torch.no_grad()
    def _y_ctx_zero(self, y: Tensor) -> Tensor:
        """Create a zero tensor with correct shape for y_ctx."""
        return _ckb.mask_all(self.context_prediction(y).detach())

    def compress(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        n, c, h, w = y.shape
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = _ckb.unembed(side_params, anchor_parity=self.anchor_parity)
        y_ = _ckb.unembed(y, anchor_parity=self.anchor_parity)
        y_strings_ = [None] * 2

        for i in range(2):
            y_ctx_i = _ckb.unembed(
                self.context_prediction(
                    _ckb.embed(y_hat_, anchor_parity=self.anchor_parity)
                ),
                anchor_parity=self.anchor_parity,
            )[i]
            if i == 0:
                y_ctx_i = _ckb.mask_all(y_ctx_i)
            params_i = self.entropy_parameters(_ckb.merge(y_ctx_i, side_params_[i]))
            y_out = self.latent_codec["y"].compress(y_[i], params_i)
            y_hat_[i] = y_out["y_hat"]
            [y_strings_[i]] = y_out["strings"]

        y_hat = _ckb.embed(y_hat_, anchor_parity=self.anchor_parity)

        return {
            "strings": y_strings_,
            "shape": y_hat.shape[1:],
            "y_hat": y_hat,
        }

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Tuple[int, ...],
        side_params: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        y_strings_ = strings
        n = len(y_strings_[0])
        assert len(y_strings_) == 2
        assert all(len(x) == n for x in y_strings_)

        c, h, w = shape
        y_i_shape = (h, w // 2)
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = _ckb.unembed(side_params, anchor_parity=self.anchor_parity)

        for i in range(2):
            y_ctx_i = _ckb.unembed(
                self.context_prediction(
                    _ckb.embed(y_hat_, anchor_parity=self.anchor_parity)
                ),
                anchor_parity=self.anchor_parity,
            )[i]
            if i == 0:
                y_ctx_i = _ckb.mask_all(y_ctx_i)
            params_i = self.entropy_parameters(_ckb.merge(y_ctx_i, side_params_[i]))
            y_out = self.latent_codec["y"].decompress(
                [y_strings_[i]], y_i_shape, params_i
            )
            y_hat_[i] = y_out["y_hat"]

        y_hat = _ckb.embed(y_hat_, anchor_parity=self.anchor_parity)

        return {
            "y_hat": y_hat,
        }

    def quantize(self, y: Tensor) -> Tensor:
        mode = "noise" if self.training else "dequantize"
        y_hat = EntropyModel.quantize(None, y, mode)
        return y_hat

    def _quantize(self, y: Tensor, means: Tensor) -> Tensor:
        if self.quantizer == "sga":
            assert self.sga is not None
            return self.sga(y - means) + means
        return quantize_ste(y - means) + means

    def _likelihood_for_quantized(self, y_hat: Tensor, params: Tensor) -> Tensor:
        return self.latent_codec["y"]._likelihood_for_quantized(y_hat, params)
