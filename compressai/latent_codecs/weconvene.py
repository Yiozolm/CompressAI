from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import GaussianConditional
from compressai.layers.wave import DWT2D, IDWT2D
from compressai.ops import quantize_ste
from compressai.registry import register_module

from .base import LatentCodec
from .weconvene_support import (
    apply_high_lrp,
    apply_low_lrp,
    decode_high_branch,
    decode_low_branch,
    encode_high_branch,
    encode_low_branch,
    high_params,
    low_params,
)

__all__ = ["WeChARMLatentCodec"]


@register_module("WeChARMLatentCodec")
class WeChARMLatentCodec(LatentCodec):
    cc_mean_transforms_high: nn.ModuleList
    cc_mean_transforms_low: nn.ModuleList
    cc_scale_transforms_high: nn.ModuleList
    cc_scale_transforms_low: nn.ModuleList
    gaussian_conditional_high: GaussianConditional
    gaussian_conditional_low: GaussianConditional
    lrp_transforms_high: nn.ModuleList
    lrp_transforms_low: nn.ModuleList

    def __init__(
        self,
        M: int,
        *,
        cc_mean_transforms_low: nn.ModuleList,
        cc_scale_transforms_low: nn.ModuleList,
        cc_mean_transforms_high: nn.ModuleList,
        cc_scale_transforms_high: nn.ModuleList,
        lrp_transforms_low: nn.ModuleList,
        lrp_transforms_high: nn.ModuleList,
        gaussian_conditional_low: Optional[GaussianConditional] = None,
        gaussian_conditional_high: Optional[GaussianConditional] = None,
        mean_support_transforms_low: Optional[nn.ModuleList] = None,
        scale_support_transforms_low: Optional[nn.ModuleList] = None,
        mean_support_transforms_high: Optional[nn.ModuleList] = None,
        scale_support_transforms_high: Optional[nn.ModuleList] = None,
        num_slices: int = 5,
        max_support_slices: int = -1,
        quantizer: str = "ste",
        lrp_scale: float = 0.5,
        wavelet: str = "haar",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if quantizer not in {"ste", "noise"}:
            raise ValueError(f"Unknown quantizer: {quantizer!r}")
        if M % num_slices != 0:
            raise ValueError("M must be divisible by num_slices")

        self._kwargs = kwargs
        self.M = int(M)
        self.num_slices = int(num_slices)
        self.max_support_slices = int(max_support_slices)
        self.quantizer = quantizer
        self.lrp_scale = float(lrp_scale)
        self.low_slice_channels = self.M // self.num_slices
        self.high_slice_channels = (3 * self.M) // self.num_slices

        self.cc_mean_transforms_low = cc_mean_transforms_low
        self.cc_scale_transforms_low = cc_scale_transforms_low
        self.cc_mean_transforms_high = cc_mean_transforms_high
        self.cc_scale_transforms_high = cc_scale_transforms_high
        self.lrp_transforms_low = lrp_transforms_low
        self.lrp_transforms_high = lrp_transforms_high
        self.mean_support_transforms_low = mean_support_transforms_low or nn.ModuleList(
            nn.Identity() for _ in range(self.num_slices)
        )
        self.scale_support_transforms_low = scale_support_transforms_low or nn.ModuleList(
            nn.Identity() for _ in range(self.num_slices)
        )
        self.mean_support_transforms_high = mean_support_transforms_high or nn.ModuleList(
            nn.Identity() for _ in range(self.num_slices)
        )
        self.scale_support_transforms_high = scale_support_transforms_high or nn.ModuleList(
            nn.Identity() for _ in range(self.num_slices)
        )
        self.gaussian_conditional_low = gaussian_conditional_low or GaussianConditional(None)
        self.gaussian_conditional_high = gaussian_conditional_high or GaussianConditional(None)
        self.dwt = DWT2D(wave=wavelet)
        self.idwt = IDWT2D(wave=wavelet)

        self._validate_lengths()

    def _validate_lengths(self) -> None:
        expected = self.num_slices
        modules = {
            "cc_mean_transforms_low": self.cc_mean_transforms_low,
            "cc_scale_transforms_low": self.cc_scale_transforms_low,
            "cc_mean_transforms_high": self.cc_mean_transforms_high,
            "cc_scale_transforms_high": self.cc_scale_transforms_high,
            "lrp_transforms_low": self.lrp_transforms_low,
            "lrp_transforms_high": self.lrp_transforms_high,
            "mean_support_transforms_low": self.mean_support_transforms_low,
            "scale_support_transforms_low": self.scale_support_transforms_low,
            "mean_support_transforms_high": self.mean_support_transforms_high,
            "scale_support_transforms_high": self.scale_support_transforms_high,
        }
        for name, module_list in modules.items():
            if len(module_list) != expected:
                raise ValueError(f"{name} must have {expected} entries")

    def to_wavelet(self, y: Tensor) -> Tensor:
        return self.dwt(y)

    def from_wavelet(self, wavelet_tensor: Tensor) -> Tensor:
        return self.idwt(wavelet_tensor)

    def _split_wavelet(self, wavelet_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        return wavelet_tensor.split((self.M, 3 * self.M), dim=1)

    def _merge_wavelet(self, y_low_hat: Tensor, y_high_hat: Tensor) -> Tensor:
        return torch.cat((y_low_hat, y_high_hat), dim=1)

    def _quantize(self, inputs: Tensor, means: Tensor, gaussian: GaussianConditional) -> Tensor:
        if self.quantizer == "ste":
            return quantize_ste(inputs - means) + means
        mode = "noise" if self.training else "dequantize"
        return gaussian.quantize(inputs, mode, means)

    def forward(
        self,
        y: Tensor,
        latent_means: Tensor,
        latent_scales: Tensor,
        *,
        wavelet_output: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        wavelet_tensor = wavelet_output if wavelet_output is not None else self.to_wavelet(y)
        y_low, y_high = self._split_wavelet(wavelet_tensor)
        spatial_shape = (y_low.shape[2], y_low.shape[3])

        y_low_hat_slices: List[Tensor] = []
        y_low_likelihoods: List[Tensor] = []
        for slice_index, y_low_slice in enumerate(y_low.chunk(self.num_slices, dim=1)):
            mu, scale, mean_support = low_params(
                self,
                slice_index,
                latent_means,
                latent_scales,
                y_low_hat_slices,
                spatial_shape,
            )
            _, y_low_slice_likelihood = self.gaussian_conditional_low(
                y_low_slice,
                scale,
                means=mu,
            )
            y_low_hat_slice = self._quantize(
                y_low_slice,
                mu,
                self.gaussian_conditional_low,
            )
            y_low_hat_slice = apply_low_lrp(
                self,
                slice_index,
                mean_support,
                y_low_hat_slice,
            )
            y_low_hat_slices.append(y_low_hat_slice)
            y_low_likelihoods.append(y_low_slice_likelihood)

        y_low_hat = torch.cat(y_low_hat_slices, dim=1)
        y_high_hat_slices: List[Tensor] = []
        y_high_likelihoods: List[Tensor] = []
        for slice_index, y_high_slice in enumerate(y_high.chunk(self.num_slices, dim=1)):
            mu, scale, mean_support = high_params(
                self,
                slice_index,
                latent_means,
                latent_scales,
                y_low_hat,
                y_high_hat_slices,
                spatial_shape,
            )
            _, y_high_slice_likelihood = self.gaussian_conditional_high(
                y_high_slice,
                scale,
                means=mu,
            )
            y_high_hat_slice = self._quantize(
                y_high_slice,
                mu,
                self.gaussian_conditional_high,
            )
            y_high_hat_slice = apply_high_lrp(
                self,
                slice_index,
                y_low_hat,
                mean_support,
                y_high_hat_slice,
            )
            y_high_hat_slices.append(y_high_hat_slice)
            y_high_likelihoods.append(y_high_slice_likelihood)

        y_high_hat = torch.cat(y_high_hat_slices, dim=1)
        return {
            "y_hat": self.from_wavelet(self._merge_wavelet(y_low_hat, y_high_hat)),
            "likelihoods": {
                "y_low": torch.cat(y_low_likelihoods, dim=1),
                "y_high": torch.cat(y_high_likelihoods, dim=1),
            },
        }

    def compress(
        self,
        y: Tensor,
        latent_means: Tensor,
        latent_scales: Tensor,
        *,
        wavelet_output: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        wavelet_tensor = wavelet_output if wavelet_output is not None else self.to_wavelet(y)
        y_low, y_high = self._split_wavelet(wavelet_tensor)
        y_low_string, y_low_hat = encode_low_branch(self, y_low, latent_means, latent_scales)
        y_high_string, y_high_hat = encode_high_branch(
            self,
            y_high,
            y_low_hat,
            latent_means,
            latent_scales,
        )
        return {
            "strings": [[y_low_string], [y_high_string]],
            "shape": y_low.shape[-2:],
            "y_hat": self.from_wavelet(self._merge_wavelet(y_low_hat, y_high_hat)),
        }

    def decompress(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Tuple[int, int],
        latent_means: Tensor,
        latent_scales: Tensor,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if len(strings) != 2:
            raise ValueError("strings must contain [low_strings, high_strings]")

        y_low_hat = decode_low_branch(self, strings[0], shape, latent_means, latent_scales)
        y_high_hat = decode_high_branch(
            self,
            strings[1],
            shape,
            y_low_hat,
            latent_means,
            latent_scales,
        )
        return {
            "y_hat": self.from_wavelet(self._merge_wavelet(y_low_hat, y_high_hat)),
        }
