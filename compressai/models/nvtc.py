# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0.

"""NVTC image compression model.

Port of Feng et al., "NVTC: Nonlinear Vector Transform Coding", CVPR 2023.
The upstream image repository is Lightning-based and marks practical entropy
coding as TODO; this CompressAI port provides the differentiable model,
likelihood estimates, zoo registration and checkpoint loading path.
"""

from __future__ import annotations

import math

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.layers.lic.nvtc import (
    BlockCombination,
    BlockPartition,
    ECVQLastDim,
    ResBlocks,
    VTUnit,
)
from compressai.registry import register_model

from .base import CompressionModel
from .nvtc_support import (
    as_tuple,
    convert_upstream_state_dict,
    extract_state_dict,
    infer_config_from_state_dict,
    padding_factor,
    sum_tensors,
)

__all__ = ["NVTC", "convert_upstream_state_dict"]


@register_model("nvtc")
class NVTC(CompressionModel):
    """Nonlinear Vector Transform Coding model adapted to CompressAI.

    Args mirror the upstream NVTC image config. ``forward`` returns
    ``{"x_hat", "likelihoods"}`` plus NVTC-specific rate/loss diagnostics.
    """

    def __init__(
        self,
        lmbda: float = 256,
        n_stage: int = 3,
        n_layer: Sequence[int] = (4, 6, 6),
        downscale_factor: Sequence[int] = (4, 8, 16),
        vt_dim: Sequence[int] = (192, 192, 192),
        vt_nunit: Sequence[int] = (2, 2, 2),
        block_size: Sequence[int] = (4, 4, 4),
        cb_dim: Sequence[int] = (4, 8, 16),
        cb_size: Sequence[int] = (128, 256, 512),
        param_dim: Sequence[int] = (4, 4, 4),
        param_nlevel: Sequence[int] = (128, 64, 32),
        rate_constrain: bool = True,
        conditional_prior: bool = True,
        use_vq: Optional[Sequence[Sequence[bool]]] = None,
        discretized: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if lmbda <= 0:
            raise ValueError(f"lmbda must be positive, got {lmbda}")

        self.lmbda = float(lmbda)
        self.n_stage = int(n_stage)
        self.n_layer = as_tuple(n_layer, self.n_stage, "n_layer")
        self.downscale_factor = as_tuple(
            downscale_factor, self.n_stage, "downscale_factor"
        )
        self.vt_dim = as_tuple(vt_dim, self.n_stage, "vt_dim")
        self.vt_nunit = as_tuple(vt_nunit, self.n_stage, "vt_nunit")
        self.block_size = as_tuple(block_size, self.n_stage, "block_size")
        self.cb_dim = as_tuple(cb_dim, self.n_stage, "cb_dim")
        self.cb_size = as_tuple(cb_size, self.n_stage, "cb_size")
        self.param_dim = as_tuple(param_dim, self.n_stage, "param_dim")
        self.param_nlevel = as_tuple(param_nlevel, self.n_stage, "param_nlevel")
        self.rate_constrain = bool(rate_constrain)
        self.conditional_prior = bool(conditional_prior)
        self.discretized = bool(discretized)

        self.use_vq = self._make_use_vq(use_vq)
        self._validate_downscale_factors()

        self.quantizer = nn.ModuleList()
        self.vt_encoder = nn.ModuleList()
        self.vt_decoder = nn.ModuleList()
        self.projection_in = nn.ModuleList()
        self.projection_out = nn.ModuleList()
        self.downscaling = nn.ModuleList()
        self.upscaling = nn.ModuleList()
        self.partition = nn.ModuleList()
        self.combination = nn.ModuleList()
        self.prior_estimator = nn.ModuleList()
        self._build_stages()

    @property
    def required_input_factor(self) -> int:
        return padding_factor(self.downscale_factor, self.block_size)

    def _make_use_vq(
        self,
        use_vq: Optional[Sequence[Sequence[bool]]],
    ) -> Tuple[Tuple[bool, ...], ...]:
        if use_vq is None:
            return tuple(tuple(True for _ in range(n)) for n in self.n_layer)
        if len(use_vq) != self.n_stage:
            raise ValueError(f"use_vq length must be {self.n_stage}, got {len(use_vq)}")

        result: List[Tuple[bool, ...]] = []
        for stage, (stage_flags, n_layer) in enumerate(zip(use_vq, self.n_layer)):
            flags = tuple(bool(flag) for flag in stage_flags)
            if len(flags) != n_layer:
                raise ValueError(
                    f"use_vq[{stage}] length must be {n_layer}, got {len(flags)}"
                )
            result.append(flags)
        return tuple(result)

    def _validate_downscale_factors(self) -> None:
        previous = 1
        for factor in self.downscale_factor:
            if factor <= 0:
                raise ValueError("downscale_factor values must be positive")
            if factor % previous != 0:
                raise ValueError(
                    "Each downscale factor must be divisible by the previous stage"
                )
            previous = factor

    def _build_stages(self) -> None:
        for stage in range(self.n_stage):
            resolution_factor = self.downscale_factor[stage]
            if stage > 0:
                resolution_factor //= self.downscale_factor[stage - 1]

            vt_dim_upper = 3 if stage == 0 else self.vt_dim[stage - 1]
            scaling_inner_dim = vt_dim_upper * resolution_factor**2
            current_dim = self.vt_dim[stage]
            block = self.block_size[stage]

            self.downscaling.append(
                nn.Sequential(
                    nn.PixelUnshuffle(resolution_factor),
                    nn.Conv2d(scaling_inner_dim, current_dim, kernel_size=1),
                    ResBlocks(current_dim) if stage == 0 else nn.Sequential(),
                )
            )
            self.upscaling.append(
                nn.Sequential(
                    ResBlocks(current_dim) if stage == 0 else nn.Sequential(),
                    nn.Conv2d(current_dim, scaling_inner_dim, kernel_size=1),
                    nn.PixelShuffle(resolution_factor),
                )
            )
            self.partition.append(BlockPartition(block, block))
            self.combination.append(BlockCombination(block, block))

            quantizer = nn.ModuleList()
            vt_encoder = nn.ModuleList()
            vt_decoder = nn.ModuleList()
            projection_in = nn.ModuleList()
            projection_out = nn.ModuleList()
            prior_estimator = nn.ModuleList()

            for _ in range(self.n_layer[stage]):
                vt_encoder.append(
                    nn.Sequential(
                        *[
                            VTUnit(current_dim, block)
                            for _ in range(self.vt_nunit[stage])
                        ]
                    )
                )
                vt_decoder.append(
                    nn.Sequential(
                        *[
                            VTUnit(current_dim, block)
                            for _ in range(self.vt_nunit[stage])
                        ]
                    )
                )
                projection_in.append(
                    nn.Conv2d(current_dim, self.cb_dim[stage], kernel_size=1)
                )
                projection_out.append(
                    nn.Conv2d(self.cb_dim[stage], current_dim, kernel_size=1)
                )
                quantizer.append(
                    ECVQLastDim(
                        event_shape=(block**2, self.cb_dim[stage]),
                        cb_size=self.cb_size[stage],
                        param_dim=self.param_dim[stage],
                        param_nlevel=self.param_nlevel[stage],
                        share_codebook=False,
                        rate_constrain=self.rate_constrain,
                        discretized=self.discretized,
                    )
                )
                prior_estimator.append(
                    nn.Sequential(
                        nn.Conv2d(current_dim, 64, kernel_size=1),
                        ResBlocks(64, kernel_size=1),
                        nn.Conv2d(64, self.param_dim[stage], kernel_size=1),
                    )
                )

            self.vt_encoder.append(vt_encoder)
            self.vt_decoder.append(vt_decoder)
            self.projection_in.append(projection_in)
            self.projection_out.append(projection_out)
            self.quantizer.append(quantizer)
            self.prior_estimator.append(prior_estimator)

    def pre_padding(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        height, width = x.shape[2:4]
        factor = self.required_input_factor
        dh = factor * math.ceil(height / factor) - height
        dw = factor * math.ceil(width / factor) - width
        if dh == 0 and dw == 0:
            return x, (height, width)
        x = F.pad(x, (dw // 2, dw // 2 + dw % 2, dh // 2, dh // 2 + dh % 2))
        return x, (height, width)

    def post_cropping(self, x: Tensor, shape: Tuple[int, int]) -> Tensor:
        height, width = shape
        factor = self.required_input_factor
        dh = factor * math.ceil(height / factor) - height
        dw = factor * math.ceil(width / factor) - width
        if dh == 0 and dw == 0:
            return x
        dh1, dh2 = dh // 2, -(dh // 2 + dh % 2) or None
        dw1, dw2 = dw // 2, -(dw // 2 + dw % 2) or None
        return x[..., dh1:dh2, dw1:dw2]

    def update_rate_constrain(self) -> None:
        for stage in range(self.n_stage):
            for layer in range(self.n_layer[stage]):
                self.quantizer[stage][layer].rate_constrain = self.rate_constrain

    def forward(self, x: Tensor) -> Dict[str, Any]:
        x_ori = x
        x, original_shape = self.pre_padding(x)
        height, width = x.shape[2:4]
        numel = x_ori.numel()
        num_pixels = x_ori.size(0) * x_ori.size(2) * x_ori.size(3)

        transformed_vector: List[List[Tensor]] = []
        current = x
        for stage in range(self.n_stage):
            current = self.downscaling[stage](current)
            current = self.partition[stage](current)
            vectors = []
            for layer in range(self.n_layer[stage]):
                current = self.vt_encoder[stage][layer](current)
                vectors.append(current)
            transformed_vector.append(vectors)
            output_size = (
                height // self.downscale_factor[stage],
                width // self.downscale_factor[stage],
            )
            current = self.combination[stage](current, output_size)

        vq_distances: List[Tensor] = []
        rate_unconditional: List[Tensor] = []
        rate_conditional: List[Tensor] = []
        prior_vq_distances: List[Tensor] = []
        prior_vq_rates: List[Tensor] = []
        selected_likelihoods: List[Tensor] = []

        x_hat: Optional[Tensor] = None
        for stage in reversed(range(self.n_stage)):
            if x_hat is not None:
                x_hat = self.partition[stage](x_hat)

            for layer in reversed(range(self.n_layer[stage])):
                prior_param = None
                if x_hat is not None and self.conditional_prior:
                    prior_param = self.prior_estimator[stage][layer](x_hat)
                    prior_param = prior_param.flatten(start_dim=-2)
                    prior_param = prior_param.permute(0, 2, 1).contiguous()

                if self.use_vq[stage][layer]:
                    base = 0 if x_hat is None else x_hat
                    residual = transformed_vector[stage][layer] - base
                    residual = self.projection_in[stage][layer](residual)
                    residual_shape = residual.shape
                    residual = residual.flatten(start_dim=-2)
                    residual = residual.permute(0, 2, 1).contiguous()

                    q_out = self.quantizer[stage][layer](
                        residual, prior_param, self.lmbda
                    )
                    vq_distances.append(((residual - q_out.x_hat) ** 2).sum() / numel)
                    rate_unconditional.append(q_out.rate_unconditional_bits / numel)
                    rate_conditional.append(q_out.rate_conditional_bits / numel)
                    prior_vq_distances.append(q_out.prior_distortion / numel)
                    prior_vq_rates.append(q_out.prior_bits / numel)

                    if q_out.conditional_likelihoods is not None:
                        likelihood = q_out.conditional_likelihoods
                    else:
                        likelihood = q_out.unconditional_likelihoods
                    selected_likelihoods.append(likelihood.flatten())

                    residual_hat = (q_out.x_hat - residual).detach() + residual
                    residual_hat = residual_hat.permute(0, 2, 1).contiguous()
                    residual_hat = residual_hat.view(residual_shape)
                    residual_hat = self.projection_out[stage][layer](residual_hat)
                    x_hat = residual_hat if x_hat is None else x_hat + residual_hat
                elif x_hat is None:
                    x_hat = torch.zeros_like(transformed_vector[stage][layer])

                x_hat = self.vt_decoder[stage][layer](x_hat)

            output_size = (
                height // self.downscale_factor[stage],
                width // self.downscale_factor[stage],
            )
            x_hat = self.combination[stage](x_hat, output_size)
            x_hat = self.upscaling[stage](x_hat)

        if x_hat is None:
            raise RuntimeError("NVTC decoded no tensor; at least one VQ layer is required")

        x_hat = self.post_cropping(x_hat, original_shape)
        rate = sum_tensors(rate_unconditional, x_ori)
        if self.conditional_prior and rate_conditional:
            rate = rate_unconditional[0] + sum_tensors(rate_conditional[1:], x_ori)

        likelihoods = torch.cat(selected_likelihoods).clamp_min(1e-9)
        total_bits = -torch.log2(likelihoods).sum()
        distortion_loss = self.lmbda * ((x_ori - x_hat) ** 2).sum() / numel
        vq_loss = self.lmbda * sum_tensors(vq_distances, x_ori)
        prior_vq_loss = sum_tensors(prior_vq_distances, x_ori)
        prior_vq_rate = sum_tensors(prior_vq_rates, x_ori)
        rd_loss = rate + distortion_loss
        loss = (
            sum_tensors(rate_unconditional, x_ori)
            + sum_tensors(rate_conditional, x_ori)
            + distortion_loss
            + vq_loss
            + prior_vq_loss
            + prior_vq_rate
        )

        return {
            "x_hat": x_hat,
            "likelihoods": {"vq": likelihoods},
            "rate": rate,
            "bpp": total_bits / num_pixels,
            "rd_loss": rd_loss,
            "vq_loss": vq_loss,
            "prior_vq_loss": prior_vq_loss,
            "prior_vq_rate": prior_vq_rate,
            "loss": loss,
        }

    def compress(self, x: Tensor) -> Dict[str, Any]:
        raise NotImplementedError(
            "NVTC upstream does not provide practical entropy coding; "
            "this CompressAI port currently supports forward/rate estimation only."
        )

    def decompress(self, strings: Sequence[Any], shape: Sequence[int]) -> Dict[str, Tensor]:
        raise NotImplementedError(
            "NVTC upstream does not provide practical entropy coding; "
            "this CompressAI port currently supports forward/rate estimation only."
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Any],
        **kwargs: Any,
    ) -> "NVTC":
        state, lmbda = extract_state_dict(state_dict)
        clean_state = convert_upstream_state_dict(state)
        cfg = infer_config_from_state_dict(clean_state)
        if lmbda is not None and "lmbda" not in kwargs:
            cfg["lmbda"] = lmbda
        cfg.update(kwargs)
        net = cls(**cfg)
        net.load_state_dict(clean_state)
        return net
