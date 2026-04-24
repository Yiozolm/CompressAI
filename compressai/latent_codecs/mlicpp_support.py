# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Tuple, cast

import torch

from torch import Tensor

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.layers.lic.mlic import (
    checkerboard_anchor,
    checkerboard_nonanchor,
    checkerboard_split,
    compress_anchor_symbols,
    compress_nonanchor_symbols,
    decompress_anchor_symbols,
    decompress_nonanchor_symbols,
)

if TYPE_CHECKING:
    from .mlicpp import MLICPlusPlusLatentCodec


def select_num_heads(channels: int) -> int:
    target = max(1, channels // 32)
    while channels % target != 0:
        target -= 1
    return target


def entropy_coder_state(
    codec: MLICPlusPlusLatentCodec,
) -> Tuple[List[List[int]], List[int], List[int]]:
    return (
        codec.gaussian_conditional.quantized_cdf.tolist(),
        codec.gaussian_conditional.cdf_length.reshape(-1).int().tolist(),
        codec.gaussian_conditional.offset.reshape(-1).int().tolist(),
    )


def compress_single(
    codec: MLICPlusPlusLatentCodec,
    y_slices: Sequence[Tensor],
    hyper_params: Tensor,
    hyper_means: Tensor,
) -> Tuple[bytes, Tensor]:
    encoder = BufferedRansEncoder()
    symbols_list: List[int] = []
    indexes_list: List[int] = []
    y_hat_slices: List[Tensor] = []

    for index, y_slice in enumerate(y_slices):
        slice_anchor, slice_nonanchor = checkerboard_split(y_slice)
        (
            scales_anchor,
            means_anchor,
            global_inter_ctx,
            channel_ctx,
        ) = codec.anchor_distribution(index, y_hat_slices, hyper_params)
        slice_anchor = compress_anchor_symbols(
            codec.gaussian_conditional,
            slice_anchor,
            scales_anchor,
            means_anchor,
            symbols_list,
            indexes_list,
        )
        lrp_anchor = codec.lrp_anchor[index](
            codec.lrp_inputs(hyper_means, y_hat_slices, slice_anchor)
        )
        slice_anchor = slice_anchor + checkerboard_anchor(lrp_anchor)

        scales_nonanchor, means_nonanchor = codec.nonanchor_distribution(
            index,
            y_hat_slices,
            hyper_params,
            slice_anchor,
            global_inter_ctx,
            channel_ctx,
        )
        slice_nonanchor = compress_nonanchor_symbols(
            codec.gaussian_conditional,
            slice_nonanchor,
            scales_nonanchor,
            means_nonanchor,
            symbols_list,
            indexes_list,
        )
        y_hat_slice = slice_anchor + slice_nonanchor
        lrp_nonanchor = codec.lrp_nonanchor[index](
            codec.lrp_inputs(hyper_means, y_hat_slices, y_hat_slice)
        )
        y_hat_slice = y_hat_slice + checkerboard_nonanchor(lrp_nonanchor)
        y_hat_slices.append(y_hat_slice)

    cdf, cdf_lengths, offsets = entropy_coder_state(codec)
    encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
    return cast(bytes, encoder.flush()), torch.cat(y_hat_slices, dim=1)


def decompress_single(
    codec: MLICPlusPlusLatentCodec,
    y_string: bytes,
    hyper_params: Tensor,
    hyper_means: Tensor,
) -> Tensor:
    decoder = RansDecoder()
    decoder.set_stream(y_string)
    cdf, cdf_lengths, offsets = entropy_coder_state(codec)
    y_hat_slices: List[Tensor] = []

    for index in range(codec.slice_num):
        (
            scales_anchor,
            means_anchor,
            global_inter_ctx,
            channel_ctx,
        ) = codec.anchor_distribution(index, y_hat_slices, hyper_params)
        slice_anchor = decompress_anchor_symbols(
            codec.gaussian_conditional,
            scales_anchor,
            means_anchor,
            decoder,
            cdf,
            cdf_lengths,
            offsets,
        )
        lrp_anchor = codec.lrp_anchor[index](
            codec.lrp_inputs(hyper_means, y_hat_slices, slice_anchor)
        )
        slice_anchor = slice_anchor + checkerboard_anchor(lrp_anchor)

        scales_nonanchor, means_nonanchor = codec.nonanchor_distribution(
            index,
            y_hat_slices,
            hyper_params,
            slice_anchor,
            global_inter_ctx,
            channel_ctx,
        )
        slice_nonanchor = decompress_nonanchor_symbols(
            codec.gaussian_conditional,
            scales_nonanchor,
            means_nonanchor,
            decoder,
            cdf,
            cdf_lengths,
            offsets,
        )
        y_hat_slice = slice_anchor + slice_nonanchor
        lrp_nonanchor = codec.lrp_nonanchor[index](
            codec.lrp_inputs(hyper_means, y_hat_slices, y_hat_slice)
        )
        y_hat_slice = y_hat_slice + checkerboard_nonanchor(lrp_nonanchor)
        y_hat_slices.append(y_hat_slice)

    return torch.cat(y_hat_slices, dim=1)
