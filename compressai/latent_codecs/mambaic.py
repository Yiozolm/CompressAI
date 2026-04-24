from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import GaussianConditional
from compressai.ops import quantize_ste
from compressai.registry import register_module

from .base import LatentCodec

__all__ = ["MambaICLatentCodec"]


def _keep_anchor(input_tensor: Tensor) -> Tensor:
    output = torch.zeros_like(input_tensor)
    output[:, :, 0::2, 0::2] = input_tensor[:, :, 0::2, 0::2]
    output[:, :, 1::2, 1::2] = input_tensor[:, :, 1::2, 1::2]
    return output


def _keep_nonanchor(input_tensor: Tensor) -> Tensor:
    output = torch.zeros_like(input_tensor)
    output[:, :, 0::2, 1::2] = input_tensor[:, :, 0::2, 1::2]
    output[:, :, 1::2, 0::2] = input_tensor[:, :, 1::2, 0::2]
    return output


def _squeeze_anchor(input_tensor: Tensor) -> Tensor:
    batch_size, channels, height, width = input_tensor.shape
    output = input_tensor.new_zeros((batch_size, channels, height, width // 2))
    output[:, :, 0::2, :] = input_tensor[:, :, 0::2, 0::2]
    output[:, :, 1::2, :] = input_tensor[:, :, 1::2, 1::2]
    return output


def _squeeze_nonanchor(input_tensor: Tensor) -> Tensor:
    batch_size, channels, height, width = input_tensor.shape
    output = input_tensor.new_zeros((batch_size, channels, height, width // 2))
    output[:, :, 0::2, :] = input_tensor[:, :, 0::2, 1::2]
    output[:, :, 1::2, :] = input_tensor[:, :, 1::2, 0::2]
    return output


def _unsqueeze_anchor(input_tensor: Tensor) -> Tensor:
    batch_size, channels, height, width = input_tensor.shape
    output = input_tensor.new_zeros((batch_size, channels, height, width * 2))
    output[:, :, 0::2, 0::2] = input_tensor[:, :, 0::2, :]
    output[:, :, 1::2, 1::2] = input_tensor[:, :, 1::2, :]
    return output


def _unsqueeze_nonanchor(input_tensor: Tensor) -> Tensor:
    batch_size, channels, height, width = input_tensor.shape
    output = input_tensor.new_zeros((batch_size, channels, height, width * 2))
    output[:, :, 0::2, 1::2] = input_tensor[:, :, 0::2, :]
    output[:, :, 1::2, 0::2] = input_tensor[:, :, 1::2, :]
    return output


@register_module("MambaICLatentCodec")
class MambaICLatentCodec(LatentCodec):
    def __init__(
        self,
        mean_support_transforms: nn.ModuleList,
        scale_support_transforms: nn.ModuleList,
        cc_mean_transforms: nn.ModuleList,
        cc_scale_transforms: nn.ModuleList,
        context_prediction: nn.ModuleList,
        context_vss: nn.ModuleList,
        context_mean_transforms: nn.ModuleList,
        context_scale_transforms: nn.ModuleList,
        lrp_transforms: nn.ModuleList,
        gaussian_conditional: GaussianConditional | None = None,
        *,
        num_slices: int,
        max_support_slices: int = -1,
        lrp_scale: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        modules = (
            mean_support_transforms,
            scale_support_transforms,
            cc_mean_transforms,
            cc_scale_transforms,
            context_prediction,
            context_vss,
            context_mean_transforms,
            context_scale_transforms,
            lrp_transforms,
        )
        if any(len(module) != num_slices for module in modules):
            raise ValueError("All per-slice module lists must have num_slices entries")

        self.num_slices = int(num_slices)
        self.max_support_slices = int(max_support_slices)
        self.lrp_scale = float(lrp_scale)
        self.mean_support_transforms = mean_support_transforms
        self.scale_support_transforms = scale_support_transforms
        self.cc_mean_transforms = cc_mean_transforms
        self.cc_scale_transforms = cc_scale_transforms
        self.context_prediction = context_prediction
        self.context_vss = context_vss
        self.context_mean_transforms = context_mean_transforms
        self.context_scale_transforms = context_scale_transforms
        self.lrp_transforms = lrp_transforms
        self.gaussian_conditional = gaussian_conditional or GaussianConditional(None)

    def _support_slices(self, y_hat_slices: Sequence[Tensor]) -> List[Tensor]:
        if self.max_support_slices < 0:
            return list(y_hat_slices)
        return list(y_hat_slices[: self.max_support_slices])

    def _base_params(
        self,
        slice_index: int,
        latent_means: Tensor,
        latent_scales: Tensor,
        y_hat_slices: Sequence[Tensor],
        spatial_shape: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        support = self._support_slices(y_hat_slices)
        mean_support = torch.cat([latent_means, *support], dim=1)
        mean_support = self.mean_support_transforms[slice_index](mean_support)
        mu = self.cc_mean_transforms[slice_index](mean_support)
        mu = mu[:, :, : spatial_shape[0], : spatial_shape[1]]

        scale_support = torch.cat([latent_scales, *support], dim=1)
        scale_support = self.scale_support_transforms[slice_index](scale_support)
        scale = self.cc_scale_transforms[slice_index](scale_support)
        scale = scale[:, :, : spatial_shape[0], : spatial_shape[1]]

        if slice_index == 0:
            support_params = torch.cat([latent_means, latent_scales], dim=1)
        else:
            support_params = torch.cat([mu, scale, latent_means, latent_scales], dim=1)
        return mu, scale, mean_support, support_params

    def _context_params(
        self,
        slice_index: int,
        support_params: Tensor,
        context_input: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        means, scales = self.context_vss[slice_index](
            torch.cat([context_input, support_params], dim=1)
        ).chunk(2, dim=1)
        means = self.context_mean_transforms[slice_index](means)
        scales = self.context_scale_transforms[slice_index](scales)
        return means, scales

    def _apply_lrp(
        self,
        slice_index: int,
        mean_support: Tensor,
        y_hat_slice: Tensor,
    ) -> Tensor:
        lrp = self.lrp_transforms[slice_index](torch.cat([mean_support, y_hat_slice], dim=1))
        return y_hat_slice + self.lrp_scale * torch.tanh(lrp)

    def _compress_step(
        self,
        input_tensor: Tensor,
        scales: Tensor,
        means: Tensor,
        *,
        squeeze_fn,
        unsqueeze_fn,
    ) -> Tuple[List[bytes], Tensor]:
        values = squeeze_fn(input_tensor)
        scales_half = squeeze_fn(scales)
        means_half = squeeze_fn(means)
        indexes = self.gaussian_conditional.build_indexes(scales_half)
        strings = self.gaussian_conditional.compress(values, indexes, means=means_half)
        quantized = self.gaussian_conditional.decompress(strings, indexes, means=means_half)
        return strings, unsqueeze_fn(quantized)

    def _decompress_step(
        self,
        strings: List[bytes],
        scales: Tensor,
        means: Tensor,
        *,
        squeeze_fn,
        unsqueeze_fn,
    ) -> Tensor:
        scales_half = squeeze_fn(scales)
        means_half = squeeze_fn(means)
        indexes = self.gaussian_conditional.build_indexes(scales_half)
        quantized = self.gaussian_conditional.decompress(strings, indexes, means=means_half)
        return unsqueeze_fn(quantized)

    def forward(
        self,
        y: Tensor,
        latent_means: Tensor,
        latent_scales: Tensor,
    ) -> Dict[str, Any]:
        spatial_shape = (y.shape[2], y.shape[3])
        slice_channels = y.shape[1] // self.num_slices
        zero_context = y.new_zeros((y.shape[0], 2 * slice_channels, *spatial_shape))
        y_hat_slices: List[Tensor] = []
        y_likelihoods_slices: List[Tensor] = []

        for slice_index, y_slice in enumerate(y.chunk(self.num_slices, dim=1)):
            _, _, mean_support, support_params = self._base_params(
                slice_index,
                latent_means,
                latent_scales,
                y_hat_slices,
                spatial_shape,
            )
            means_anchor, scales_anchor = self._context_params(
                slice_index,
                support_params,
                zero_context,
            )
            means_anchor = _keep_anchor(means_anchor)
            scales_anchor = _keep_anchor(scales_anchor)

            y_anchor_hat = _keep_anchor(quantize_ste(y_slice - means_anchor) + means_anchor)
            masked_context = self.context_prediction[slice_index](y_anchor_hat)
            means_nonanchor, scales_nonanchor = self._context_params(
                slice_index,
                support_params,
                masked_context,
            )
            means_nonanchor = _keep_nonanchor(means_nonanchor)
            scales_nonanchor = _keep_nonanchor(scales_nonanchor)

            means_slice = means_anchor + means_nonanchor
            scales_slice = scales_anchor + scales_nonanchor
            _, y_slice_likelihoods = self.gaussian_conditional(
                y_slice,
                scales_slice,
                means=means_slice,
            )

            y_nonanchor_hat = _keep_nonanchor(
                quantize_ste(y_slice - means_nonanchor) + means_nonanchor
            )
            y_hat_slice = self._apply_lrp(
                slice_index,
                mean_support,
                y_anchor_hat + y_nonanchor_hat,
            )
            y_hat_slices.append(y_hat_slice)
            y_likelihoods_slices.append(y_slice_likelihoods)

        return {
            "y_hat": torch.cat(y_hat_slices, dim=1),
            "likelihoods": {"y": torch.cat(y_likelihoods_slices, dim=1)},
        }

    def compress(
        self,
        y: Tensor,
        latent_means: Tensor,
        latent_scales: Tensor,
    ) -> Dict[str, Any]:
        spatial_shape = (y.shape[2], y.shape[3])
        slice_channels = y.shape[1] // self.num_slices
        zero_context = y.new_zeros((y.shape[0], 2 * slice_channels, *spatial_shape))
        y_hat_slices: List[Tensor] = []
        y_strings: List[List[List[bytes]]] = []

        for slice_index, y_slice in enumerate(y.chunk(self.num_slices, dim=1)):
            _, _, mean_support, support_params = self._base_params(
                slice_index,
                latent_means,
                latent_scales,
                y_hat_slices,
                spatial_shape,
            )
            means_anchor, scales_anchor = self._context_params(
                slice_index,
                support_params,
                zero_context,
            )
            anchor_strings, y_anchor_hat = self._compress_step(
                y_slice,
                means=means_anchor,
                scales=scales_anchor,
                squeeze_fn=_squeeze_anchor,
                unsqueeze_fn=_unsqueeze_anchor,
            )

            masked_context = self.context_prediction[slice_index](y_anchor_hat)
            means_nonanchor, scales_nonanchor = self._context_params(
                slice_index,
                support_params,
                masked_context,
            )
            nonanchor_strings, y_nonanchor_hat = self._compress_step(
                y_slice,
                means=means_nonanchor,
                scales=scales_nonanchor,
                squeeze_fn=_squeeze_nonanchor,
                unsqueeze_fn=_unsqueeze_nonanchor,
            )
            y_hat_slice = self._apply_lrp(
                slice_index,
                mean_support,
                y_anchor_hat + y_nonanchor_hat,
            )
            y_hat_slices.append(y_hat_slice)
            y_strings.append([anchor_strings, nonanchor_strings])

        return {
            "strings": y_strings,
            "shape": spatial_shape,
            "y_hat": torch.cat(y_hat_slices, dim=1),
        }

    def decompress(
        self,
        strings: Sequence[Sequence[List[bytes]]],
        shape: Tuple[int, int],
        latent_means: Tensor,
        latent_scales: Tensor,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if len(strings) != self.num_slices:
            raise ValueError("strings must contain one [anchor, nonanchor] pair per slice")

        slice_channels = latent_means.shape[1] // self.num_slices
        zero_context = latent_means.new_zeros(
            (latent_means.shape[0], 2 * slice_channels, shape[0], shape[1])
        )
        y_hat_slices: List[Tensor] = []

        for slice_index, slice_strings in enumerate(strings):
            if len(slice_strings) != 2:
                raise ValueError("Each slice stream must contain anchor and non-anchor strings")

            _, _, mean_support, support_params = self._base_params(
                slice_index,
                latent_means,
                latent_scales,
                y_hat_slices,
                shape,
            )
            means_anchor, scales_anchor = self._context_params(
                slice_index,
                support_params,
                zero_context,
            )
            y_anchor_hat = self._decompress_step(
                slice_strings[0],
                means=means_anchor,
                scales=scales_anchor,
                squeeze_fn=_squeeze_anchor,
                unsqueeze_fn=_unsqueeze_anchor,
            )
            masked_context = self.context_prediction[slice_index](y_anchor_hat)
            means_nonanchor, scales_nonanchor = self._context_params(
                slice_index,
                support_params,
                masked_context,
            )
            y_nonanchor_hat = self._decompress_step(
                slice_strings[1],
                means=means_nonanchor,
                scales=scales_nonanchor,
                squeeze_fn=_squeeze_nonanchor,
                unsqueeze_fn=_unsqueeze_nonanchor,
            )
            y_hat_slices.append(
                self._apply_lrp(
                    slice_index,
                    mean_support,
                    y_anchor_hat + y_nonanchor_hat,
                )
            )

        return {"y_hat": torch.cat(y_hat_slices, dim=1)}
