from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import scipy.stats
import torch

from torch import Tensor

from .entropy_models import GaussianConditional

__all__ = [
    "GeneralizedGaussianConditional",
]


class _GeneralizedNormalCDF(torch.autograd.Function):
    """Numerically-stable forward of the standard generalized-Gaussian CDF.

    The PDF is :math:`p(y) = \\frac{\\beta}{2\\Gamma(1/\\beta)} e^{-|y|^\\beta}`,
    so the CDF can be written via the regularized incomplete gamma function as
    :math:`F(y) = \\frac{1}{2} (1 + \\operatorname{sign}(y) \\, P(1/\\beta, |y|^\\beta))`.

    Backward returns the analytic PDF wrt ``y``; ``beta`` is treated as a fixed
    parameter (no gradient).
    """

    @staticmethod
    def forward(ctx, beta: Tensor, y: Tensor) -> Tensor:
        abs_y_pow_beta = torch.pow(torch.abs(y), beta)
        ctx.save_for_backward(abs_y_pow_beta, beta)
        return 0.5 * (1 + torch.sign(y) * torch.special.gammainc(1 / beta, abs_y_pow_beta))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        abs_y_pow_beta, beta = ctx.saved_tensors
        pdf = (
            torch.exp(-abs_y_pow_beta - torch.special.gammaln(1 / beta)) * beta / 2
        )
        return None, grad_output * pdf


class GeneralizedGaussianConditional(GaussianConditional):
    """Generalized Gaussian (gennorm) conditional entropy model.

    Drop-in replacement for :class:`GaussianConditional` that models latents as
    a generalized Gaussian distribution with shape parameter ``beta`` (Gaussian
    when ``beta=2``, Laplace when ``beta=1``). HPCM (`arXiv:2507.19125`_) uses
    ``beta=1.5`` for both the per-channel hyperprior and the spatially
    autoregressive y prior.

    The base class' :meth:`update`, :meth:`build_indexes`, :meth:`compress`,
    :meth:`decompress` and the entropy-coder plumbing are reused as-is; only
    the standardized cumulative and standardized quantile are overridden.

    .. _arXiv:2507.19125: https://arxiv.org/abs/2507.19125
    """

    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]],
        *args: Any,
        beta: float = 1.5,
        scale_bound: float = 0.12,
        **kwargs: Any,
    ) -> None:
        super().__init__(scale_table, *args, scale_bound=scale_bound, **kwargs)
        self.register_buffer("beta", torch.tensor([float(beta)]))

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        return _GeneralizedNormalCDF.apply(self.beta, inputs)

    def _standardized_quantile(self, quantile):
        beta_value = float(self.beta.item()) if isinstance(self.beta, Tensor) else float(self.beta)
        return scipy.stats.gennorm.ppf(quantile, beta_value)

    def _likelihood(
        self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None
    ) -> Tensor:
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs
        scales = self.lower_bound_scale(scales)
        upper = self._standardized_cumulative((values + half) / scales)
        lower = self._standardized_cumulative((values - half) / scales)
        return upper - lower
