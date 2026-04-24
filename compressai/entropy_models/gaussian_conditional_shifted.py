import math
from typing import Any, List, Optional, Tuple

import torch

from torch import Tensor

from .entropy_models import EntropyModel, GaussianConditional


__all__ = [
    "GsnConditionalLocScaleShift",
    "Scaler",
]


class Scaler:
    """Map positive scales to a logarithmic integer index grid."""

    def __init__(
        self,
        scales_min: float = 0.01,
        scales_max: float = 256.0,
        num_bins: int = 256,
        verify_valid_scales: bool = True,
    ) -> None:
        if scales_min <= 0 or scales_max <= 0:
            raise ValueError("Scale bounds must be positive")
        if scales_max < scales_min:
            raise ValueError("Maximum scale must be greater than minimum scale")
        if num_bins < 1:
            raise ValueError("Number of scale bins must be positive")

        self.scales_min = float(scales_min)
        self.scales_max = float(scales_max)
        self.num_bins = int(num_bins)
        self.verify_valid_scales = bool(verify_valid_scales)

    def to_scale_idx(self, scales: Tensor, training: bool = True) -> Tensor:
        if self.verify_valid_scales and torch.any(scales <= 0):
            raise ValueError("Scales must be strictly positive")

        indexes = torch.log(scales + self.scales_min)
        indexes = indexes - math.log(self.scales_min)
        normalizer = self.scales_max / self.scales_min + 1.0
        indexes = indexes / math.log(normalizer)
        indexes = indexes * (self.num_bins - 1)
        if not training:
            indexes = torch.round(indexes)
        return indexes

    def from_scale_idx(self, indexes: Tensor) -> Tensor:
        normalizer = self.scales_max / self.scales_min + 1.0
        outputs = indexes / (self.num_bins - 1)
        outputs = outputs * math.log(normalizer)
        return torch.exp(outputs + math.log(self.scales_min))


def _make_scale_table(
    min_scale: float,
    scale_max: float,
    num_scales: int,
) -> List[float]:
    if num_scales == 1:
        return [float(min_scale)]

    normalizer = scale_max / min_scale + 1.0
    log_min = math.log(min_scale)
    log_normalizer = math.log(normalizer)
    return [
        math.exp(log_min + log_normalizer * index / (num_scales - 1))
        for index in range(num_scales)
    ]


class GsnConditionalLocScaleShift(GaussianConditional):
    """Gaussian conditional with FTIC-style rounded-location shifting."""

    def __init__(
        self,
        num_scales: int = 256,
        min_scale: float = 0.01,
        num_means: int = 100,
        tail_mass: float = 2 ** (-8),
        round_idx: bool = True,
        scale_max: float = 256.0,
        **kwargs: Any,
    ) -> None:
        num_scales = int(num_scales)
        num_means = int(num_means)
        if num_scales < 1:
            raise ValueError("Number of scales must be positive")
        if num_means < 1:
            raise ValueError("Number of means must be positive")

        scale_table = _make_scale_table(min_scale, scale_max, num_scales)
        kwargs.setdefault("entropy_coder_precision", 16)
        super().__init__(
            scale_table,
            scale_bound=min_scale,
            tail_mass=tail_mass,
            **kwargs,
        )

        self._scaler = Scaler(
            scales_min=min_scale,
            scales_max=scale_max,
            num_bins=num_scales,
        )
        self._num_means = num_means
        self._num_scales = num_scales
        self._min_scale = float(min_scale)
        self._scale_max = float(scale_max)
        self._round_idx = bool(round_idx)
        self._one_mean_flag = num_means == 1

        self.register_buffer("_prior_mean", torch.Tensor())
        self.register_buffer("_prior_scale", torch.Tensor())

    @staticmethod
    def _round_st(inputs: Tensor) -> Tensor:
        return (torch.round(inputs) - inputs).detach() + inputs

    @staticmethod
    def verysoftplus(inputs: Tensor) -> Tensor:
        zeros = torch.zeros_like(inputs)
        inputs_pos = torch.maximum(inputs, zeros)
        inputs_neg = torch.minimum(inputs, zeros)
        return torch.where(inputs > 0, inputs_pos + 1.0, 1.0 / (1.0 - inputs_neg))

    def _get_indexes(
        self,
        means: Tensor,
        scales: Tensor,
        training: bool = True,
    ) -> Tensor:
        scales_i = self._scaler.to_scale_idx(
            self.verysoftplus(scales),
            training=training,
        )
        if self._one_mean_flag:
            means_i = torch.zeros_like(scales_i)
        else:
            mean_residual = means - self._round_st(means)
            means_i = (mean_residual + 0.5) * self._num_means
            if not training and self._round_idx:
                means_i = torch.round(means_i)

        indexes = torch.stack([means_i, scales_i], dim=-1)
        return self._normalize_indexes(indexes)

    def _normalize_indexes(self, indexes: Tensor) -> Tensor:
        means_i = indexes[..., 0].clamp(0, self._num_means - 1)
        scales_i = indexes[..., 1].clamp(0, self._num_scales - 1)
        return torch.stack([means_i, scales_i], dim=-1)

    def indexes_to_cdf_indexes(
        self,
        indexes_mean: Tensor,
        indexes_scale: Tensor,
    ) -> Tensor:
        return indexes_mean.int() * self._num_scales + indexes_scale.int()

    def build_indexes(self, scales: Tensor, means: Optional[Tensor] = None) -> Tensor:
        if means is None:
            means = torch.zeros_like(scales)
        indexes = self._get_indexes(means, scales, training=False)
        return self.indexes_to_cdf_indexes(indexes[..., 0], indexes[..., 1])

    def update_scale_table(
        self,
        scale_table: Optional[List[float]] = None,
        force: bool = False,
    ) -> bool:
        return self.update(force=force)

    def update(self, force: bool = True) -> bool:
        if self._offset.numel() > 0 and not force:
            return False

        device = self.scale_table.device
        dtype = self.scale_table.dtype
        mean_indexes = torch.arange(self._num_means, device=device, dtype=dtype)
        if self._one_mean_flag:
            mean_table = torch.zeros_like(mean_indexes)
        else:
            mean_table = mean_indexes / self._num_means - 0.5

        prior_mean = mean_table[:, None].expand(-1, self._num_scales)
        prior_scale = self.scale_table[None, :].expand(self._num_means, -1)
        self._prior_mean = prior_mean
        self._prior_scale = prior_scale

        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        tail_lower = prior_mean - multiplier * prior_scale
        tail_upper = prior_mean + multiplier * prior_scale

        minimas = torch.floor(tail_lower).to(torch.int32)
        maximas = torch.ceil(tail_upper).to(torch.int32)
        pmf_start = minimas.to(dtype=torch.float32)
        pmf_length = maximas - minimas + 1
        max_length = int(torch.max(pmf_length).item())

        samples = torch.arange(max_length, device=device, dtype=torch.float32)
        samples = samples[:, None, None] + pmf_start[None, :, :]
        values = torch.abs(samples - prior_mean[None, :, :])
        scales = prior_scale[None, :, :]
        upper = self._standardized_cumulative((0.5 - values) / scales)
        lower = self._standardized_cumulative((-0.5 - values) / scales)
        pmf = upper - lower

        num_pmfs = self._num_means * self._num_scales
        pmf = pmf.reshape(max_length, num_pmfs).transpose(0, 1)
        tail_mass = 2 * lower.reshape(max_length, num_pmfs).transpose(0, 1)[:, :1]
        pmf_length = pmf_length.reshape(num_pmfs)

        self._quantized_cdf = self._pmf_to_cdf(
            pmf,
            tail_mass,
            pmf_length,
            max_length,
        )
        self._offset = minimas.reshape(num_pmfs)
        self._cdf_length = pmf_length + 2
        return True

    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Tensor,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training

        outputs = self.quantize(inputs, "noise" if training else "dequantize")
        likelihood = self._likelihood(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        if training:
            outputs = self._round_st(inputs)
        return outputs, likelihood

    def compress(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Tensor,
    ) -> Tuple[Tensor, List[bytes]]:
        indexes = self.build_indexes(scales, means)
        rounded_means = torch.round(means)
        shifted_inputs = inputs - rounded_means
        strings = EntropyModel.compress(self, shifted_inputs, indexes)
        quantized = torch.round(shifted_inputs) + rounded_means
        return quantized, strings

    def decompress(
        self,
        strings: List[bytes],
        scales: Tensor,
        means: Tensor,
    ) -> Tensor:
        indexes = self.build_indexes(scales, means)
        rounded_means = torch.round(means)
        outputs = EntropyModel.decompress(
            self,
            strings,
            indexes,
            dtype=means.dtype,
        )
        return outputs + rounded_means
