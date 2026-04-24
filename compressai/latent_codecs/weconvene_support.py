from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Tuple

import torch

from torch import Tensor

from compressai.ans import BufferedRansEncoder, RansDecoder

if TYPE_CHECKING:
    from .weconvene import WeChARMLatentCodec


def support_slices(
    codec: "WeChARMLatentCodec",
    y_hat_slices: Sequence[Tensor],
) -> List[Tensor]:
    if codec.max_support_slices < 0:
        return list(y_hat_slices)
    return list(y_hat_slices[: codec.max_support_slices])


def low_params(
    codec: "WeChARMLatentCodec",
    slice_index: int,
    latent_means: Tensor,
    latent_scales: Tensor,
    y_low_hat_slices: Sequence[Tensor],
    spatial_shape: Tuple[int, int],
) -> Tuple[Tensor, Tensor, Tensor]:
    support = support_slices(codec, y_low_hat_slices)
    mean_support = torch.cat([latent_means, *support], dim=1)
    mean_support = codec.mean_support_transforms_low[slice_index](mean_support)
    mu = codec.cc_mean_transforms_low[slice_index](mean_support)
    mu = mu[:, :, : spatial_shape[0], : spatial_shape[1]]

    scale_support = torch.cat([latent_scales, *support], dim=1)
    scale_support = codec.scale_support_transforms_low[slice_index](scale_support)
    scale = codec.cc_scale_transforms_low[slice_index](scale_support)
    scale = scale[:, :, : spatial_shape[0], : spatial_shape[1]]
    return mu, scale, mean_support


def high_params(
    codec: "WeChARMLatentCodec",
    slice_index: int,
    latent_means: Tensor,
    latent_scales: Tensor,
    y_low_hat: Tensor,
    y_high_hat_slices: Sequence[Tensor],
    spatial_shape: Tuple[int, int],
) -> Tuple[Tensor, Tensor, Tensor]:
    support = support_slices(codec, y_high_hat_slices)
    mean_support = torch.cat([latent_means, y_low_hat, *support], dim=1)
    mean_support = codec.mean_support_transforms_high[slice_index](mean_support)
    mu = codec.cc_mean_transforms_high[slice_index](mean_support)
    mu = mu[:, :, : spatial_shape[0], : spatial_shape[1]]

    scale_support = torch.cat([latent_scales, y_low_hat, *support], dim=1)
    scale_support = codec.scale_support_transforms_high[slice_index](scale_support)
    scale = codec.cc_scale_transforms_high[slice_index](scale_support)
    scale = scale[:, :, : spatial_shape[0], : spatial_shape[1]]
    return mu, scale, mean_support


def apply_low_lrp(
    codec: "WeChARMLatentCodec",
    slice_index: int,
    mean_support: Tensor,
    y_low_hat_slice: Tensor,
) -> Tensor:
    lrp = codec.lrp_transforms_low[slice_index](
        torch.cat([mean_support, y_low_hat_slice], dim=1)
    )
    return y_low_hat_slice + codec.lrp_scale * torch.tanh(lrp)


def apply_high_lrp(
    codec: "WeChARMLatentCodec",
    slice_index: int,
    y_low_hat: Tensor,
    mean_support: Tensor,
    y_high_hat_slice: Tensor,
) -> Tensor:
    lrp = codec.lrp_transforms_high[slice_index](
        torch.cat([y_low_hat, mean_support, y_high_hat_slice], dim=1)
    )
    return y_high_hat_slice + codec.lrp_scale * torch.tanh(lrp)


def encode_low_branch(
    codec: "WeChARMLatentCodec",
    y_low: Tensor,
    latent_means: Tensor,
    latent_scales: Tensor,
) -> Tuple[bytes, Tensor]:
    cdf = codec.gaussian_conditional_low.quantized_cdf.tolist()
    cdf_lengths = codec.gaussian_conditional_low.cdf_length.reshape(-1).int().tolist()
    offsets = codec.gaussian_conditional_low.offset.reshape(-1).int().tolist()
    encoder = BufferedRansEncoder()
    symbols_list: List[int] = []
    indexes_list: List[int] = []
    y_low_hat_slices: List[Tensor] = []
    spatial_shape = (y_low.shape[2], y_low.shape[3])

    for slice_index, y_low_slice in enumerate(y_low.chunk(codec.num_slices, dim=1)):
        mu, scale, mean_support = low_params(
            codec,
            slice_index,
            latent_means,
            latent_scales,
            y_low_hat_slices,
            spatial_shape,
        )
        indexes = codec.gaussian_conditional_low.build_indexes(scale)
        y_q_slice = codec.gaussian_conditional_low.quantize(y_low_slice, "symbols", mu)
        y_low_hat_slice = apply_low_lrp(codec, slice_index, mean_support, y_q_slice + mu)
        symbols_list.extend(y_q_slice.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        y_low_hat_slices.append(y_low_hat_slice)

    encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
    return encoder.flush(), torch.cat(y_low_hat_slices, dim=1)


def encode_high_branch(
    codec: "WeChARMLatentCodec",
    y_high: Tensor,
    y_low_hat: Tensor,
    latent_means: Tensor,
    latent_scales: Tensor,
) -> Tuple[bytes, Tensor]:
    cdf = codec.gaussian_conditional_high.quantized_cdf.tolist()
    cdf_lengths = codec.gaussian_conditional_high.cdf_length.reshape(-1).int().tolist()
    offsets = codec.gaussian_conditional_high.offset.reshape(-1).int().tolist()
    encoder = BufferedRansEncoder()
    symbols_list: List[int] = []
    indexes_list: List[int] = []
    y_high_hat_slices: List[Tensor] = []
    spatial_shape = (y_high.shape[2], y_high.shape[3])

    for slice_index, y_high_slice in enumerate(y_high.chunk(codec.num_slices, dim=1)):
        mu, scale, mean_support = high_params(
            codec,
            slice_index,
            latent_means,
            latent_scales,
            y_low_hat,
            y_high_hat_slices,
            spatial_shape,
        )
        indexes = codec.gaussian_conditional_high.build_indexes(scale)
        y_q_slice = codec.gaussian_conditional_high.quantize(y_high_slice, "symbols", mu)
        y_high_hat_slice = apply_high_lrp(
            codec,
            slice_index,
            y_low_hat,
            mean_support,
            y_q_slice + mu,
        )
        symbols_list.extend(y_q_slice.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        y_high_hat_slices.append(y_high_hat_slice)

    encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
    return encoder.flush(), torch.cat(y_high_hat_slices, dim=1)


def decode_low_branch(
    codec: "WeChARMLatentCodec",
    low_strings: Sequence[bytes],
    shape: Tuple[int, int],
    latent_means: Tensor,
    latent_scales: Tensor,
) -> Tensor:
    if len(low_strings) != 1:
        raise ValueError("Only batch size 1 is supported for WeConvene low-band decoding")

    cdf = codec.gaussian_conditional_low.quantized_cdf.tolist()
    cdf_lengths = codec.gaussian_conditional_low.cdf_length.reshape(-1).int().tolist()
    offsets = codec.gaussian_conditional_low.offset.reshape(-1).int().tolist()
    decoder = RansDecoder()
    decoder.set_stream(low_strings[0])
    y_low_hat_slices: List[Tensor] = []

    for slice_index in range(codec.num_slices):
        mu, scale, mean_support = low_params(
            codec,
            slice_index,
            latent_means,
            latent_scales,
            y_low_hat_slices,
            shape,
        )
        indexes = codec.gaussian_conditional_low.build_indexes(scale)
        values = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        y_q_slice = torch.tensor(values, device=mu.device, dtype=mu.dtype).reshape(mu.shape)
        y_low_hat_slice = codec.gaussian_conditional_low.dequantize(y_q_slice, mu)
        y_low_hat_slices.append(apply_low_lrp(codec, slice_index, mean_support, y_low_hat_slice))

    return torch.cat(y_low_hat_slices, dim=1)


def decode_high_branch(
    codec: "WeChARMLatentCodec",
    high_strings: Sequence[bytes],
    shape: Tuple[int, int],
    y_low_hat: Tensor,
    latent_means: Tensor,
    latent_scales: Tensor,
) -> Tensor:
    if len(high_strings) != 1:
        raise ValueError("Only batch size 1 is supported for WeConvene high-band decoding")

    cdf = codec.gaussian_conditional_high.quantized_cdf.tolist()
    cdf_lengths = codec.gaussian_conditional_high.cdf_length.reshape(-1).int().tolist()
    offsets = codec.gaussian_conditional_high.offset.reshape(-1).int().tolist()
    decoder = RansDecoder()
    decoder.set_stream(high_strings[0])
    y_high_hat_slices: List[Tensor] = []

    for slice_index in range(codec.num_slices):
        mu, scale, mean_support = high_params(
            codec,
            slice_index,
            latent_means,
            latent_scales,
            y_low_hat,
            y_high_hat_slices,
            shape,
        )
        indexes = codec.gaussian_conditional_high.build_indexes(scale)
        values = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        y_q_slice = torch.tensor(values, device=mu.device, dtype=mu.dtype).reshape(mu.shape)
        y_high_hat_slice = codec.gaussian_conditional_high.dequantize(y_q_slice, mu)
        y_high_hat_slices.append(
            apply_high_lrp(codec, slice_index, y_low_hat, mean_support, y_high_hat_slice)
        )

    return torch.cat(y_high_hat_slices, dim=1)
