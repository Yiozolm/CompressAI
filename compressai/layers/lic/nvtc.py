# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0.

"""Building blocks for NVTC image compression.

This module ports the pure-PyTorch pieces from the upstream NVTC image model
while removing the Lightning training shell and the unfinished range-coding
dependency.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class _LowerBoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, bound: Tensor) -> Tensor:
        ctx.save_for_backward(x, bound)
        return torch.maximum(x, bound)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        x, bound = ctx.saved_tensors
        pass_through = torch.logical_or(x >= bound, grad_output < 0).to(grad_output)
        return pass_through * grad_output, None


class _UpperBoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, bound: Tensor) -> Tensor:
        ctx.save_for_backward(x, bound)
        return torch.minimum(x, bound)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        x, bound = ctx.saved_tensors
        pass_through = torch.logical_or(x <= bound, grad_output > 0).to(grad_output)
        return pass_through * grad_output, None


def _lower_bound(x: Tensor, bound: float) -> Tensor:
    return _LowerBoundFunction.apply(x, x.new_tensor(float(bound)))


def _upper_bound(x: Tensor, bound: float) -> Tensor:
    return _UpperBoundFunction.apply(x, x.new_tensor(float(bound)))


class _SoftmaxPMF:
    def __init__(self, logits: Tensor) -> None:
        self.logits = logits

    def pmf(self) -> Tensor:
        pmf = F.softmax(self.logits, dim=-1)
        return _lower_bound(pmf, 1e-9) if pmf.requires_grad else pmf

    def log_pmf(self) -> Tensor:
        log_pmf = F.log_softmax(self.logits, dim=-1)
        if log_pmf.requires_grad:
            return _lower_bound(log_pmf, math.log(1e-9))
        return log_pmf


class VTUnit(nn.Module):
    def __init__(self, channels: int, spatial_shape: int, ratio: float = 0.5) -> None:
        super().__init__()
        hidden_channels = int(channels * ratio)
        self.intra_transform = ChannelFC(channels, hidden_channels, channels)
        self.inter_transform = DepthwiseBlockFC(channels, spatial_shape)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.intra_transform(x)
        return x + self.inter_transform(x)


class ChannelFC(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hid_channels, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hid_channels, out_channels, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DepthwiseBlockFC(nn.Module):
    def __init__(self, channels: int, block_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, block_size**2, block_size**2))
        self.bias = nn.Parameter(torch.empty(channels, block_size**2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x = torch.flatten(x.unsqueeze(1), start_dim=-2)
        x = torch.einsum("bacn,cnm->bacm", x, self.weight)
        x = x + self.bias
        return x.reshape(*shape)


class BlockPartition(nn.Module):
    """Partition ``(B, C, H, W)`` into ``(B * n_blocks, C, h_block, w_block)``."""

    def __init__(self, h_block: int, w_block: int) -> None:
        super().__init__()
        self.h_block = int(h_block)
        self.w_block = int(w_block)

    def forward(self, x: Tensor) -> Tensor:
        hb, wb = self.h_block, self.w_block
        batch, channels, height, width = x.shape
        if height % hb != 0 or width % wb != 0:
            raise ValueError(
                f"Input spatial size {(height, width)} is not divisible by {(hb, wb)}"
            )
        n_block = height // hb * width // wb
        x = x.view(batch, channels, height // hb, hb, width // wb, wb)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(batch * n_block, channels, hb, wb)


class BlockCombination(nn.Module):
    """Reverse operation of :class:`BlockPartition`."""

    def __init__(self, h_block: int, w_block: int) -> None:
        super().__init__()
        self.h_block = int(h_block)
        self.w_block = int(w_block)

    def forward(self, x: Tensor, output_size: Tuple[int, int]) -> Tensor:
        hb, wb = self.h_block, self.w_block
        height, width = output_size
        channels = x.shape[1]
        if height % hb != 0 or width % wb != 0:
            raise ValueError(
                f"Output spatial size {(height, width)} is not divisible by {(hb, wb)}"
            )
        x = x.view(-1, height // hb, width // wb, channels, hb, wb)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(-1, channels, height, width)


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.m = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.m(x)


class ResBlocks(nn.Module):
    def __init__(self, channels: int, n: int = 3, kernel_size: int = 3) -> None:
        super().__init__()
        self.m = nn.Sequential(
            *[ResBlock(channels, kernel_size=kernel_size) for _ in range(n)]
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.m(x)


class _LinearResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.m(x)


class _LinearResBlocks(nn.Module):
    def __init__(self, channels: int, n: int = 3) -> None:
        super().__init__()
        self.m = nn.Sequential(*[_LinearResBlock(channels) for _ in range(n)])

    def forward(self, x: Tensor) -> Tensor:
        return x + self.m(x)


class _DeepConditionalPriorFn(nn.Module):
    def __init__(self, param_dim: int, cb_size: int) -> None:
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(param_dim, 64),
            _LinearResBlocks(64),
            nn.Linear(64, cb_size),
        )

    def forward(self, params: Tensor) -> _SoftmaxPMF:
        return _SoftmaxPMF(self.nn(params))


class DiscreteConditionalEntropyModel(nn.Module):
    def __init__(
        self,
        param_dim: int,
        param_nlevel: int,
        cb_size: int,
        discretized: bool = False,
    ) -> None:
        super().__init__()
        self.discretized = bool(discretized)
        self.prior_fn = _DeepConditionalPriorFn(param_dim, cb_size)
        self.param_dim = int(param_dim)
        self.param_nlevel = int(param_nlevel)
        self.param_table = nn.Parameter(
            torch.empty(param_nlevel, param_dim).normal_(0, 1 / math.sqrt(param_dim))
        )
        self.logits = nn.Parameter(torch.zeros(param_nlevel))

    def _make_prior(self, params: Tensor) -> _SoftmaxPMF:
        return self.prior_fn(params)

    def _normalize_params(self, params: Tensor) -> Tensor:
        return _upper_bound(_lower_bound(params, -1), 1)

    def _quantize_params(self, params: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        param_table = self.param_table
        dist = param_table.pow(2).sum(dim=-1) + params.pow(2).sum(dim=-1).unsqueeze(-1)
        dist = dist - 2 * torch.einsum("abc,dc->abd", params, param_table)
        index = dist.argmin(dim=-1, keepdim=True)
        one_hot = torch.zeros_like(dist).scatter_(-1, index, 1.0)
        params_quantized = torch.einsum("abd,dc->abc", one_hot, param_table)
        return params_quantized, one_hot, index

    def log_pmf(self, params: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        params = self._normalize_params(params)
        if self.discretized:
            params_quantized, one_hot, _ = self._quantize_params(params)
            params_ste = (params_quantized - params).detach() + params
            prior = self._make_prior(params_ste)
            param_log_pmf = _SoftmaxPMF(self.logits).log_pmf()
            param_bit = (one_hot * param_log_pmf).sum() / (-math.log(2))
        else:
            params_quantized = params
            prior = self._make_prior(params)
            param_bit = params.new_zeros(())
        return prior.log_pmf(), params_quantized, param_bit


@dataclass(frozen=True)
class ECVQOutput:
    x_hat: Tensor
    rate_unconditional_bits: Tensor
    rate_conditional_bits: Tensor
    prior_distortion: Tensor
    prior_bits: Tensor
    unconditional_likelihoods: Tensor
    conditional_likelihoods: Optional[Tensor]


class ECVQLastDim(nn.Module):
    """Entropy-constrained vector quantization on the last tensor dimension."""

    def __init__(
        self,
        event_shape: Tuple[int, int] = (16, 4),
        cb_size: int = 1024,
        param_dim: int = 4,
        param_nlevel: int = 128,
        share_codebook: bool = False,
        rate_constrain: bool = True,
        discretized: bool = False,
    ) -> None:
        super().__init__()
        self.event_shape = tuple(int(v) for v in event_shape)
        self.cb_size = int(cb_size)
        self.cb_dim = self.event_shape[1]
        self.share_codebook = bool(share_codebook)
        self.rate_constrain = bool(rate_constrain)

        ncb = self.event_shape[0] if not self.share_codebook else 1
        self.ncb = ncb
        self.codebook = nn.Parameter(
            torch.empty(ncb, self.cb_size, self.cb_dim).normal_(
                0, 1 / math.sqrt(self.cb_dim)
            )
        )
        self.logits = nn.Parameter(torch.zeros(ncb, self.cb_size))
        self.quantization = ConditionalVectorQuantization()
        self.entropy_model = DiscreteConditionalEntropyModel(
            param_dim=param_dim,
            param_nlevel=param_nlevel,
            cb_size=self.cb_size,
            discretized=discretized,
        )

    def forward(
        self,
        x: Tensor,
        prior_param: Optional[Tensor],
        lmbda: float,
    ) -> ECVQOutput:
        if x.shape[-2:] != self.event_shape:
            raise ValueError(f"Expected event shape {self.event_shape}, got {x.shape[-2:]}")

        shape = x.shape
        x = x.view(-1, *self.event_shape)
        log_pmf_u = _SoftmaxPMF(self.logits).log_pmf()
        bits_u = log_pmf_u / (-math.log(2))
        log_pmf_for_rate = bits_u

        if prior_param is not None:
            log_pmf_c, params_quantized, param_bits = self.entropy_model.log_pmf(
                prior_param
            )
            bits_c = log_pmf_c / (-math.log(2))
            log_pmf_for_rate = bits_c
            prior_distortion = ((params_quantized - prior_param) ** 2).sum()
        else:
            log_pmf_c = None
            bits_c = None
            param_bits = x.new_zeros(())
            prior_distortion = x.new_zeros(())

        rate_bias = log_pmf_for_rate / float(lmbda) if self.rate_constrain else None
        x_hat, one_hot, index = self.quantization(x, self.codebook, rate_bias)

        rate_unconditional = (one_hot * bits_u).sum()
        if bits_c is not None:
            rate_conditional = torch.gather(bits_c, dim=-1, index=index).sum()
        else:
            rate_conditional = x.new_zeros(())

        log_pmf_u_expanded = log_pmf_u.unsqueeze(0).expand(x.size(0), -1, -1)
        likelihoods_u = torch.gather(log_pmf_u_expanded, -1, index).exp().squeeze(-1)
        likelihoods_c = None
        if log_pmf_c is not None:
            likelihoods_c = torch.gather(log_pmf_c, -1, index).exp().squeeze(-1)

        return ECVQOutput(
            x_hat=x_hat.view(shape),
            rate_unconditional_bits=rate_unconditional,
            rate_conditional_bits=rate_conditional,
            prior_distortion=prior_distortion,
            prior_bits=param_bits,
            unconditional_likelihoods=likelihoods_u,
            conditional_likelihoods=likelihoods_c,
        )


class ConditionalVectorQuantization(nn.Module):
    @staticmethod
    def l2_dist(x: Tensor, code_book: Tensor) -> Tensor:
        x_expanded = x.unsqueeze(-1)
        dist = x_expanded.pow(2).sum(dim=-2) + code_book.pow(2).sum(dim=-1)
        return dist - 2 * torch.einsum("abc,dac->dab", code_book, x)

    def forward(
        self,
        x: Tensor,
        code_book: Tensor,
        rate_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        dist = self.l2_dist(x, code_book)
        if rate_bias is not None:
            dist = dist + rate_bias
        index = dist.argmin(dim=-1, keepdim=True)
        one_hot = torch.zeros_like(dist).scatter_(-1, index, 1.0)
        x_hat = torch.einsum("abc,bcd->abd", one_hot, code_book)
        return x_hat, one_hot, index
