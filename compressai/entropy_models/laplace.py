# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

from __future__ import annotations

import scipy.stats
import torch

from torch import Tensor

from ._mixture import _GenericMixtureConditional
from .entropy_models import GaussianConditional

__all__ = [
    "LaplaceConditional",
    "LaplaceMixtureConditional",
]


class LaplaceConditional(GaussianConditional):
    """Conditional Laplace entropy model."""

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        sign = torch.sign(inputs)
        return 0.5 + 0.5 * sign * (-torch.expm1(-inputs.abs()))

    @staticmethod
    def _standardized_quantile(quantile: float) -> float:
        return scipy.stats.laplace.ppf(quantile)


class LaplaceMixtureConditional(_GenericMixtureConditional):
    """Mixture conditional with Laplace components."""

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        sign = torch.sign(inputs)
        return 0.5 + 0.5 * sign * (-torch.expm1(-inputs.abs()))

    @staticmethod
    def _standardized_quantile(quantile: float) -> float:
        return scipy.stats.laplace.ppf(quantile)
