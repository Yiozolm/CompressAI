# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from torch import Tensor

from .entropy_models import EntropyModel, GaussianMixtureConditional
from .laplace import LaplaceMixtureConditional
from .logistic import LogisticMixtureConditional

__all__ = ["GaussianLaplaceLogisticMixtureConditional"]


class GaussianLaplaceLogisticMixtureConditional(EntropyModel):
    """Gaussian-Laplace-Logistic mixture-of-families conditional model."""

    def __init__(
        self,
        K: int = 3,
        scale_bound: float = 0.11,
        likelihood_bound: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(likelihood_bound=likelihood_bound, **kwargs)
        self.K = int(K)
        self.gaussian = GaussianMixtureConditional(
            K=self.K,
            scale_bound=scale_bound,
            likelihood_bound=likelihood_bound,
            **kwargs,
        )
        self.laplace = LaplaceMixtureConditional(
            K=self.K,
            scale_bound=scale_bound,
            likelihood_bound=likelihood_bound,
            **kwargs,
        )
        self.logistic = LogisticMixtureConditional(
            K=self.K,
            scale_bound=scale_bound,
            likelihood_bound=likelihood_bound,
            **kwargs,
        )

    def _likelihood(
        self,
        inputs: Tensor,
        means: Tensor,
        scales: Tensor,
        gaussian_weights: Tensor,
        laplace_weights: Tensor,
        logistic_weights: Tensor,
        family_weights: Tensor,
    ) -> Tensor:
        channels = inputs.size(1)
        expected_channels = 3 * self.K * channels
        if means.size(1) != expected_channels or scales.size(1) != expected_channels:
            raise ValueError(
                "Expected means/scales with "
                f"{expected_channels} channels, got {means.size(1)} and {scales.size(1)}"
            )

        family_g, family_l, family_lo = self._split_family_weights(
            family_weights,
            channels,
        )
        gaussian_slice = slice(0, self.K * channels)
        laplace_slice = slice(self.K * channels, 2 * self.K * channels)
        logistic_slice = slice(2 * self.K * channels, 3 * self.K * channels)

        gaussian_likelihood = self.gaussian._likelihood(
            inputs,
            scales[:, gaussian_slice],
            means[:, gaussian_slice],
            gaussian_weights,
        )
        laplace_likelihood = self.laplace._likelihood(
            inputs,
            scales[:, laplace_slice],
            means[:, laplace_slice],
            laplace_weights,
        )
        logistic_likelihood = self.logistic._likelihood(
            inputs,
            scales[:, logistic_slice],
            means[:, logistic_slice],
            logistic_weights,
        )
        return (
            family_g * gaussian_likelihood
            + family_l * laplace_likelihood
            + family_lo * logistic_likelihood
        )

    @staticmethod
    def _split_family_weights(
        family_weights: Tensor,
        channels: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if family_weights.dim() == 5:
            return (
                family_weights[:, 0],
                family_weights[:, 1],
                family_weights[:, 2],
            )
        if family_weights.size(1) != 3 * channels:
            raise ValueError(
                "Expected family weights with "
                f"{3 * channels} channels, got {family_weights.size(1)}"
            )
        return family_weights.chunk(3, dim=1)

    @torch.no_grad()
    def build_cdf(
        self,
        means: Tensor,
        scales: Tensor,
        gaussian_weights: Tensor,
        laplace_weights: Tensor,
        logistic_weights: Tensor,
        family_weights: Tensor,
        abs_max: int,
    ) -> Tensor:
        """Build one indexed CDF per channel for a single spatial position."""
        means = means.reshape(-1)
        scales = scales.reshape(-1).clamp(0.11, 256)
        gaussian_weights = gaussian_weights.reshape(-1)
        laplace_weights = laplace_weights.reshape(-1)
        logistic_weights = logistic_weights.reshape(-1)
        family_weights = family_weights.reshape(-1)

        channels = means.numel() // (3 * self.K)
        if channels == 0:
            raise ValueError("Expected at least one latent channel")

        means = means.reshape(3, self.K, channels)
        scales = scales.reshape(3, self.K, channels)
        family_weights = family_weights.reshape(3, channels)
        component_weights = (
            gaussian_weights.reshape(self.K, channels),
            laplace_weights.reshape(self.K, channels),
            logistic_weights.reshape(self.K, channels),
        )
        distributions = (self.gaussian, self.laplace, self.logistic)

        num_samples = abs_max * 2 + 1
        cdf_limit = 2**self.entropy_coder_precision - 1
        samples = torch.arange(
            -abs_max,
            abs_max + 1,
            device=means.device,
            dtype=means.dtype,
        ).unsqueeze(0)
        pmf = means.new_zeros(channels, num_samples)

        for family_index, distribution in enumerate(distributions):
            for component_index in range(self.K):
                component_mean = means[family_index, component_index].unsqueeze(-1)
                component_scale = scales[family_index, component_index].unsqueeze(-1)
                upper = distribution._standardized_cumulative(
                    (samples + 0.5 - component_mean) / component_scale
                )
                lower = distribution._standardized_cumulative(
                    (samples - 0.5 - component_mean) / component_scale
                )
                weights = (
                    component_weights[family_index][component_index]
                    * family_weights[family_index]
                ).unsqueeze(-1)
                pmf += (upper - lower) * weights

        pmf = torch.clamp(pmf, min=1.0 / cdf_limit, max=1.0)
        pmf_scaled = torch.round(pmf * cdf_limit)
        pmf_sum = torch.sum(pmf_scaled, dim=1, keepdim=True)
        cdf = F.pad(
            torch.cumsum(pmf_scaled * cdf_limit / pmf_sum, dim=1).int(),
            (1, 0),
            "constant",
            0,
        )
        pmf_quantized = torch.diff(cdf, dim=1)

        pmf_zero_count = num_samples - torch.count_nonzero(pmf_quantized, dim=1)
        stealable = torch.where(
            pmf_quantized > pmf_zero_count.unsqueeze(-1),
            pmf_quantized,
            torch.full_like(pmf_quantized, cdf_limit + 1),
        )
        _, first_stealable = torch.min(stealable, dim=1)

        zero_indices = (pmf_quantized == 0).nonzero().transpose(0, 1)
        if zero_indices.numel() > 0:
            pmf_quantized[zero_indices[0], zero_indices[1]] += 1

        steal_indices = torch.cat(
            (
                torch.arange(channels, device=means.device).unsqueeze(-1),
                first_stealable.unsqueeze(-1),
            ),
            dim=1,
        ).transpose(0, 1)
        pmf_quantized[steal_indices[0], steal_indices[1]] -= pmf_zero_count

        cdf = F.pad(torch.cumsum(pmf_quantized, dim=1).int(), (1, 0), "constant", 0)
        return F.pad(cdf, (0, 1), "constant", cdf_limit + 1)

    def forward(
        self,
        inputs: Tensor,
        means: Tensor,
        scales: Tensor,
        gaussian_weights: Tensor,
        laplace_weights: Tensor,
        logistic_weights: Tensor,
        family_weights: Tensor,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        outputs = self.quantize(
            inputs,
            "noise" if training else "dequantize",
            means=None,
        )
        likelihood = self._likelihood(
            outputs,
            means,
            scales,
            gaussian_weights,
            laplace_weights,
            logistic_weights,
            family_weights,
        )
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood
