# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianMixtureConditional
from compressai.layers import GDN, MaskedConv2d, conv, deconv
from compressai.layers.lic.lbhic import (
    BoundaryAwarePostProcessing,
    ContextualPredictionModule,
    make_boundary_mask,
)
from compressai.models.base import CompressionModel
from compressai.registry import register_model

__all__ = ["LBHIC"]


def _ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x


def _strip_module_prefixes(state_dict: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    clean: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        while key.startswith("module."):
            key = key.removeprefix("module.")
        clean[key] = value
    return clean


def _buffer_int(state_dict: Mapping[str, Tensor], key: str, default: int) -> int:
    value = state_dict.get(key)
    if isinstance(value, Tensor) and value.numel() == 1:
        return int(value.item())
    return int(default)


def _infer_grdb_shape(
    state_dict: Mapping[str, Tensor],
    prefix: str,
) -> Tuple[int, int]:
    block_ids = set()
    layer_ids = set()
    marker = f"{prefix}.blocks."
    for key in state_dict:
        if not key.startswith(marker):
            continue
        rest = key.removeprefix(marker)
        parts = rest.split(".")
        if len(parts) >= 4 and parts[0].isdigit() and parts[2].isdigit():
            block_ids.add(int(parts[0]))
            layer_ids.add(int(parts[2]))
    return (
        max(block_ids) + 1 if block_ids else 2,
        max(layer_ids) + 1 if layer_ids else 4,
    )


@register_model("lbhic")
class LBHIC(CompressionModel):
    """Learned Block-based Hybrid Image Compression.

    Paper-only CompressAI reproduction with block scan, CPM prediction, a
    hyperprior + autoregressive GMM entropy model, and BPM post-processing.
    """

    def __init__(
        self,
        N: int = 192,
        M: int = 192,
        input_channels: int = 3,
        block_size: int = 128,
        gmm_components: int = 3,
        prediction_channels: int = 32,
        post_channels: int = 48,
        post_growth_channels: int = 16,
        post_grdb_blocks: int = 2,
        post_dense_layers: int = 4,
        boundary_width: int = 4,
        use_prediction: bool = True,
        use_postprocess: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if block_size % 64 != 0:
            raise ValueError("block_size must be divisible by 64")

        self.N = int(N)
        self.M = int(M)
        self.input_channels = int(input_channels)
        self.block_size = int(block_size)
        self.gmm_components = int(gmm_components)
        self.prediction_channels = int(prediction_channels)
        self.post_channels = int(post_channels)
        self.post_growth_channels = int(post_growth_channels)
        self.post_grdb_blocks = int(post_grdb_blocks)
        self.post_dense_layers = int(post_dense_layers)
        self.boundary_width = int(boundary_width)
        self.use_prediction = bool(use_prediction)
        self.use_postprocess = bool(use_postprocess)

        self.register_buffer("_lbhic_block_size", torch.tensor(self.block_size))
        self.register_buffer("_lbhic_boundary_width", torch.tensor(self.boundary_width))
        self.register_buffer("_lbhic_gmm_components", torch.tensor(self.gmm_components))
        self.register_buffer(
            "_lbhic_use_prediction", torch.tensor(int(self.use_prediction))
        )
        self.register_buffer(
            "_lbhic_use_postprocess", torch.tensor(int(self.use_postprocess))
        )

        self.g_a = nn.Sequential(
            conv(input_channels, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, input_channels, kernel_size=5, stride=2),
        )
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )
        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.context_prediction = MaskedConv2d(
            M,
            2 * M,
            kernel_size=5,
            padding=2,
            stride=1,
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(4 * M, 3 * M, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(3 * M, 3 * M, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(3 * M, 3 * gmm_components * M, kernel_size=1),
        )
        self.gaussian_conditional = GaussianMixtureConditional(K=gmm_components)

        self.prediction = (
            ContextualPredictionModule(input_channels, prediction_channels)
            if self.use_prediction
            else None
        )
        self.post_process = (
            BoundaryAwarePostProcessing(
                input_channels=input_channels,
                channels=post_channels,
                growth_channels=post_growth_channels,
                num_grdb_blocks=post_grdb_blocks,
                num_dense_layers=post_dense_layers,
            )
            if self.use_postprocess
            else None
        )

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def _pad_to_block_multiple(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        height, width = x.shape[-2:]
        pad_h = (-height) % self.block_size
        pad_w = (-width) % self.block_size
        if pad_h == 0 and pad_w == 0:
            return x, (height, width)
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), (height, width)

    def _split_gmm_parameters(self, params: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        scales, means, logits = params.chunk(3, dim=1)
        scales = F.softplus(scales) + 1e-3
        batch, _, height, width = logits.shape
        weights = logits.view(
            batch,
            self.gmm_components,
            self.M,
            height,
            width,
        ).softmax(dim=1)
        return (
            scales,
            means,
            weights.reshape(batch, self.gmm_components * self.M, height, width),
        )

    def _forward_residual_codec(self, residual: Tensor) -> Dict[str, Tensor]:
        y = self.g_a(residual)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyper_params = self.h_s(z_hat)

        if self.training:
            y_hat_entropy = self.gaussian_conditional.quantize(y, "noise")
            y_hat_synthesis = _ste_round(y)
        else:
            y_hat_entropy = self.gaussian_conditional.quantize(y, "dequantize")
            y_hat_synthesis = y_hat_entropy

        ctx_params = self.context_prediction(y_hat_entropy)
        gaussian_params = self.entropy_parameters(
            torch.cat((hyper_params, ctx_params), dim=1)
        )
        scales, means, weights = self._split_gmm_parameters(gaussian_params)
        y_likelihoods = self.gaussian_conditional._likelihood(
            y_hat_entropy,
            scales,
            means,
            weights,
        )
        if self.gaussian_conditional.use_likelihood_bound:
            y_likelihoods = self.gaussian_conditional.likelihood_lower_bound(
                y_likelihoods
            )

        return {
            "residual_hat": self.g_s(y_hat_synthesis),
            "y_likelihoods": y_likelihoods,
            "z_likelihoods": z_likelihoods,
        }

    def _predict_block(self, upper: Tensor, left: Tensor) -> Tensor:
        if self.prediction is None:
            return upper.new_zeros(upper.shape)
        return self.prediction(upper, left)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x_pad, original_size = self._pad_to_block_multiple(x)
        rows = x_pad.size(2) // self.block_size
        cols = x_pad.size(3) // self.block_size

        decoded_grid = []
        x_rows = []
        y_likelihood_rows = []
        z_likelihood_rows = []

        for row in range(rows):
            decoded_row = []
            x_row = []
            y_likelihood_row = []
            z_likelihood_row = []
            for col in range(cols):
                block = x_pad[
                    :,
                    :,
                    row * self.block_size : (row + 1) * self.block_size,
                    col * self.block_size : (col + 1) * self.block_size,
                ]
                zero_ref = block.new_zeros(block.shape)
                upper = decoded_grid[row - 1][col] if row > 0 else zero_ref
                left = decoded_row[col - 1] if col > 0 else zero_ref

                prediction = self._predict_block(upper, left)
                codec_out = self._forward_residual_codec(block - prediction)
                block_hat = codec_out["residual_hat"] + prediction

                decoded_row.append(block_hat)
                x_row.append(block_hat)
                y_likelihood_row.append(codec_out["y_likelihoods"])
                z_likelihood_row.append(codec_out["z_likelihoods"])

            decoded_grid.append(decoded_row)
            x_rows.append(torch.cat(x_row, dim=3))
            y_likelihood_rows.append(torch.cat(y_likelihood_row, dim=3))
            z_likelihood_rows.append(torch.cat(z_likelihood_row, dim=3))

        x_hat_blocks = torch.cat(x_rows, dim=2)
        y_likelihoods = torch.cat(y_likelihood_rows, dim=2)
        z_likelihoods = torch.cat(z_likelihood_rows, dim=2)

        boundary_mask = make_boundary_mask(
            x_hat_blocks,
            self.block_size,
            self.boundary_width,
        )
        x_hat = (
            self.post_process(x_hat_blocks, boundary_mask)
            if self.post_process is not None
            else x_hat_blocks
        )
        height, width = original_size

        return {
            "x_hat": x_hat[:, :, :height, :width],
            "x_hat_blocks": x_hat_blocks[:, :, :height, :width],
            "boundary_mask": boundary_mask[:, :, :height, :width],
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods,
            },
        }

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Tensor]) -> "LBHIC":
        clean_state = _strip_module_prefixes(state_dict)
        N = int(clean_state["g_a.0.weight"].size(0))
        M = int(clean_state["g_a.6.weight"].size(0))
        input_channels = int(clean_state["g_a.0.weight"].size(1))
        block_size = _buffer_int(clean_state, "_lbhic_block_size", 128)
        boundary_width = _buffer_int(clean_state, "_lbhic_boundary_width", 4)
        gmm_components = _buffer_int(clean_state, "_lbhic_gmm_components", 3)
        use_prediction = bool(
            _buffer_int(
                clean_state,
                "_lbhic_use_prediction",
                int(any(k.startswith("prediction.") for k in clean_state)),
            )
        )
        use_postprocess = bool(
            _buffer_int(
                clean_state,
                "_lbhic_use_postprocess",
                int(any(k.startswith("post_process.") for k in clean_state)),
            )
        )
        prediction_channels = int(
            clean_state.get(
                "prediction.feature_extractor.0.0.weight",
                torch.empty(32),
            ).shape[0]
        )
        post_channels = int(
            clean_state.get("post_process.stem.0.weight", torch.empty(48)).shape[0]
        )
        post_growth_channels = int(
            clean_state.get(
                "post_process.scale1.blocks.0.layers.0.0.weight",
                torch.empty(16),
            ).shape[0]
        )
        post_grdb_blocks, post_dense_layers = _infer_grdb_shape(
            clean_state,
            "post_process.scale1",
        )

        net = cls(
            N=N,
            M=M,
            input_channels=input_channels,
            block_size=block_size,
            gmm_components=gmm_components,
            prediction_channels=prediction_channels,
            post_channels=post_channels,
            post_growth_channels=post_growth_channels,
            post_grdb_blocks=post_grdb_blocks,
            post_dense_layers=post_dense_layers,
            boundary_width=boundary_width,
            use_prediction=use_prediction,
            use_postprocess=use_postprocess,
        )
        target_state = net.state_dict()
        migrated = dict(target_state)
        migrated.update({k: v for k, v in clean_state.items() if k in target_state})
        net.load_state_dict(migrated)
        return net

    def compress(self, x: Tensor) -> Dict[str, Any]:
        raise NotImplementedError(
            "LBHIC real bitstream coding requires block-order CPM decoding plus "
            "per-symbol GMM arithmetic coding. The current port exposes "
            "forward/rate-estimation only."
        )

    def decompress(self, strings: Any, shape: Any) -> Dict[str, Tensor]:
        raise NotImplementedError(
            "LBHIC real bitstream decoding is not wired in for this paper-only "
            "candidate."
        )
