from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn

from torch import Tensor

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from .base import CompressionModel

__all__ = [
    "DictionaryEntropyCompressionModel",
    "infer_attention_head_dim",
    "infer_dictionary_max_support_slices",
    "infer_dictionary_num_slices",
    "infer_stage_block_num",
    "infer_window_size",
]


_CC_TRANSFORM_KEY_PREFIX = "cc_mean_transforms."
_CC_TRANSFORM_KEY_SUFFIX = ".0.weight"


def infer_dictionary_num_slices(state_dict: Dict[str, Tensor]) -> int:
    slice_indices = {
        int(key[len(_CC_TRANSFORM_KEY_PREFIX) :].split(".", 1)[0])
        for key in state_dict
        if key.startswith(_CC_TRANSFORM_KEY_PREFIX)
        and key.endswith(_CC_TRANSFORM_KEY_SUFFIX)
    }
    return len(slice_indices)


def infer_dictionary_max_support_slices(
    state_dict: Dict[str, Tensor],
    latent_channels: int,
    num_slices: int,
) -> int:
    slice_channels = latent_channels // num_slices
    max_input_channels = max(
        tensor.size(1)
        for key, tensor in state_dict.items()
        if key.startswith(_CC_TRANSFORM_KEY_PREFIX)
        and key.endswith(_CC_TRANSFORM_KEY_SUFFIX)
    )
    return max(0, (max_input_channels - 3 * latent_channels) // slice_channels)


def infer_stage_block_num(state_dict: Dict[str, Tensor], prefix: str) -> int:
    layer_indices = {
        int(key[len(prefix) :].split(".", 1)[0])
        for key in state_dict
        if key.startswith(prefix)
    }
    return len(layer_indices)


def infer_attention_head_dim(
    state_dict: Dict[str, Tensor],
    prefix: str,
    input_dim: int,
) -> int:
    relative_params_key = f"{prefix}.layers.0.msa.relative_position_params"
    if relative_params_key in state_dict:
        return input_dim // state_dict[relative_params_key].size(0)

    relative_bias_key = f"{prefix}.layers.0.msa.relative_position_bias_table"
    if relative_bias_key in state_dict:
        return input_dim // state_dict[relative_bias_key].size(1)

    raise KeyError(f"Unable to infer head_dim from prefix {prefix!r}")


def infer_window_size(state_dict: Dict[str, Tensor], prefix: str) -> int:
    relative_params_key = f"{prefix}.layers.0.msa.relative_position_params"
    if relative_params_key in state_dict:
        return (state_dict[relative_params_key].size(1) + 1) // 2

    relative_bias_key = f"{prefix}.layers.0.msa.relative_position_bias_table"
    if relative_bias_key in state_dict:
        table_side = int(round(state_dict[relative_bias_key].size(0) ** 0.5))
        return (table_side + 1) // 2

    raise KeyError(f"Unable to infer window_size from prefix {prefix!r}")


class DictionaryEntropyCompressionModel(CompressionModel):
    h_a: nn.Module
    h_z_s1: nn.Module
    h_z_s2: nn.Module
    entropy_bottleneck: EntropyBottleneck
    gaussian_conditional: GaussianConditional
    dt_cross_attention: nn.ModuleList
    cc_mean_transforms: nn.ModuleList
    cc_scale_transforms: nn.ModuleList
    lrp_transforms: nn.ModuleList
    dt: nn.Parameter
    num_slices: int
    max_support_slices: int

    def _support_slices(self, y_hat_slices: Sequence[Tensor]) -> List[Tensor]:
        if self.max_support_slices < 0:
            return list(y_hat_slices)
        return list(y_hat_slices[: self.max_support_slices])

    def _hyper_priors(self, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        return z_hat, z_likelihoods, latent_means, latent_scales

    def _slice_support(
        self,
        slice_index: int,
        latent_means: Tensor,
        latent_scales: Tensor,
        y_hat_slices: Sequence[Tensor],
        spatial_shape: Tuple[int, int],
        dictionary: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        support_slices = self._support_slices(y_hat_slices)
        query = torch.cat([latent_scales, latent_means, *support_slices], dim=1)
        dictionary_info = self.dt_cross_attention[slice_index](query, dictionary)
        support = torch.cat([query, dictionary_info], dim=1)
        mu = self.cc_mean_transforms[slice_index](support)
        mu = mu[:, :, : spatial_shape[0], : spatial_shape[1]]
        scale = self.cc_scale_transforms[slice_index](support)
        scale = scale[:, :, : spatial_shape[0], : spatial_shape[1]]
        return support, mu, scale

    def _forward_latent(
        self,
        y: Tensor,
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        spatial_shape = (y.size(2), y.size(3))
        dictionary = self.dt.unsqueeze(0).expand(y.size(0), -1, -1)
        _, z_likelihoods, latent_means, latent_scales = self._hyper_priors(y)
        y_hat_slices: List[Tensor] = []
        y_likelihood_slices: List[Tensor] = []
        means: List[Tensor] = []
        scales: List[Tensor] = []

        for slice_index, y_slice in enumerate(y.chunk(self.num_slices, dim=1)):
            support, mu, scale = self._slice_support(
                slice_index,
                latent_means,
                latent_scales,
                y_hat_slices,
                spatial_shape,
                dictionary,
            )
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_hat_slice = torch.round(y_slice - mu) - (y_slice - mu).detach() + y_slice
            y_hat_slice = y_hat_slice + mu
            lrp = self.lrp_transforms[slice_index](torch.cat([support, y_hat_slice], dim=1))
            y_hat_slice = y_hat_slice + 0.5 * torch.tanh(lrp)

            y_hat_slices.append(y_hat_slice)
            y_likelihood_slices.append(y_slice_likelihood)
            means.append(mu)
            scales.append(scale)

        return {
            "y_hat": torch.cat(y_hat_slices, dim=1),
            "likelihoods": {
                "y": torch.cat(y_likelihood_slices, dim=1),
                "z": z_likelihoods,
            },
            "means": torch.cat(means, dim=1),
            "scales": torch.cat(scales, dim=1),
            "y": y,
        }

    def _compress_latent(self, y: Tensor) -> Dict[str, object]:
        spatial_shape = (y.size(2), y.size(3))
        dictionary = self.dt.unsqueeze(0).expand(y.size(0), -1, -1)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list: List[int] = []
        indexes_list: List[int] = []
        y_hat_slices: List[Tensor] = []

        for slice_index, y_slice in enumerate(y.chunk(self.num_slices, dim=1)):
            support, mu, scale = self._slice_support(
                slice_index,
                latent_means,
                latent_scales,
                y_hat_slices,
                spatial_shape,
                dictionary,
            )
            indexes = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu
            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(indexes.reshape(-1).tolist())
            lrp = self.lrp_transforms[slice_index](torch.cat([support, y_hat_slice], dim=1))
            y_hat_slice = y_hat_slice + 0.5 * torch.tanh(lrp)
            y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        return {
            "strings": [[encoder.flush()], z_strings],
            "shape": z.size()[-2:],
        }

    def _decompress_latent(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Tuple[int, int],
    ) -> Tensor:
        if len(strings) != 2:
            raise ValueError("strings must contain [y_strings, z_strings]")

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        dictionary = self.dt.unsqueeze(0).expand(z_hat.size(0), -1, -1)
        y_shape = (z_hat.shape[2] * 4, z_hat.shape[3] * 4)

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(strings[0][0])
        y_hat_slices: List[Tensor] = []

        for slice_index in range(self.num_slices):
            support, mu, scale = self._slice_support(
                slice_index,
                latent_means,
                latent_scales,
                y_hat_slices,
                y_shape,
                dictionary,
            )
            indexes = self.gaussian_conditional.build_indexes(scale)
            values = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            y_q_slice = torch.tensor(values, device=mu.device, dtype=mu.dtype).reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(y_q_slice, mu)
            lrp = self.lrp_transforms[slice_index](torch.cat([support, y_hat_slice], dim=1))
            y_hat_slice = y_hat_slice + 0.5 * torch.tanh(lrp)
            y_hat_slices.append(y_hat_slice)

        return torch.cat(y_hat_slices, dim=1)
