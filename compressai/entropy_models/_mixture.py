# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch import Tensor

from .entropy_models import GaussianMixtureConditional


class _GenericMixtureConditional(GaussianMixtureConditional):
    """Mixture conditional using the subclass standardized CDF."""

    @torch.no_grad()
    def _build_cdf(
        self,
        scales: Tensor,
        means: Tensor,
        weights: Tensor,
        abs_max: int,
    ) -> Tensor:
        num_latents = scales.size(1)
        num_samples = abs_max * 2 + 1
        if num_latents == 0:
            return torch.empty(
                0,
                num_samples + 2,
                dtype=torch.int32,
                device=scales.device,
            )

        cdf_limit = 2**self.entropy_coder_precision - 1
        tiny = 1e-10
        device = scales.device

        scales = scales.clamp(0.11, 256)
        means = means + abs_max
        samples = torch.arange(
            num_samples,
            device=device,
            dtype=scales.dtype,
        ).unsqueeze(0)
        samples = samples.expand(num_latents, -1)

        pmf = torch.zeros_like(samples)
        for k in range(self.K):
            component_scales = scales[k].unsqueeze(-1)
            component_means = means[k].unsqueeze(-1)
            component_weights = weights[k].unsqueeze(-1)
            upper = self._standardized_cumulative(
                (samples + 0.5 - component_means) / (component_scales + tiny)
            )
            lower = self._standardized_cumulative(
                (samples - 0.5 - component_means) / (component_scales + tiny)
            )
            pmf += (upper - lower) * component_weights

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
        _, pmf_first_stealable_indices = torch.min(stealable, dim=1)

        zero_indices = (pmf_quantized == 0).nonzero().transpose(0, 1)
        if zero_indices.numel() > 0:
            pmf_quantized[zero_indices[0], zero_indices[1]] += 1

        steal_indices = torch.cat(
            (
                torch.arange(num_latents, device=device).unsqueeze(-1),
                pmf_first_stealable_indices.unsqueeze(-1),
            ),
            dim=1,
        ).transpose(0, 1)
        pmf_quantized[steal_indices[0], steal_indices[1]] -= pmf_zero_count

        cdf = F.pad(torch.cumsum(pmf_quantized, dim=1).int(), (1, 0), "constant", 0)
        return F.pad(cdf, (0, 1), "constant", cdf_limit + 1)
