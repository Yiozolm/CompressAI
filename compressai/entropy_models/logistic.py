# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

from __future__ import annotations

import scipy.stats
import torch

from torch import Tensor

from ._mixture import _GenericMixtureConditional
from .entropy_models import GaussianConditional

__all__ = [
    "LogisticConditional",
    "LogisticMixtureConditional",
]


class LogisticConditional(GaussianConditional):
    """Conditional Logistic entropy model."""

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        return torch.sigmoid(inputs)

    @staticmethod
    def _standardized_quantile(quantile: float) -> float:
        return scipy.stats.logistic.ppf(quantile)


class LogisticMixtureConditional(_GenericMixtureConditional):
    """Mixture conditional with Logistic components."""

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        return torch.sigmoid(inputs)

    @staticmethod
    def _standardized_quantile(quantile: float) -> float:
        return scipy.stats.logistic.ppf(quantile)
