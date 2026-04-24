from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn

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
    from .mlicpp import MLICPlusPlus


def select_num_heads(channels: int) -> int:
    target = max(1, channels // 32)
    while channels % target != 0:
        target -= 1
    return target


def update_local_contexts(
    local_context: nn.ModuleList,
    height: int,
    width: int,
    device: torch.device,
) -> None:
    base_mask = None
    for index, module in enumerate(local_context):
        if index == 0:
            module.update_resolution(height, width, device)
            base_mask = module.attn_mask
            continue
        module.update_resolution(height, width, device, mask=base_mask)


def lrp_inputs(
    hyper_means: Tensor,
    y_hat_slices: Sequence[Tensor],
    current_slice: Tensor,
) -> Tensor:
    return torch.cat([hyper_means, *y_hat_slices, current_slice], dim=1)


def _optional_module(modules: nn.ModuleList, index: int) -> nn.Module:
    module = modules[index]
    if module is None:
        raise RuntimeError(f"Expected module at index {index}")
    return cast(nn.Module, module)


def anchor_distribution(
    model: MLICPlusPlus,
    index: int,
    y_hat_slices: Sequence[Tensor],
    hyper_params: Tensor,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    if index == 0:
        params = model.entropy_parameters_anchor[index](hyper_params)
        scales, means = params.chunk(2, 1)
        return checkerboard_anchor(scales), checkerboard_anchor(means), None, None

    previous_slices = torch.cat(list(y_hat_slices), dim=1)
    global_inter_ctx = _optional_module(model.global_inter_context, index)(previous_slices)
    channel_ctx = _optional_module(model.channel_context, index)(previous_slices)
    params = model.entropy_parameters_anchor[index](
        torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1)
    )
    scales, means = params.chunk(2, 1)
    return (
        checkerboard_anchor(scales),
        checkerboard_anchor(means),
        global_inter_ctx,
        channel_ctx,
    )


def nonanchor_distribution(
    model: MLICPlusPlus,
    index: int,
    y_hat_slices: Sequence[Tensor],
    hyper_params: Tensor,
    anchor_hat: Tensor,
    global_inter_ctx: Optional[Tensor],
    channel_ctx: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    local_ctx = model.local_context[index](anchor_hat)
    if index == 0:
        params = model.entropy_parameters_nonanchor[index](
            torch.cat([local_ctx, hyper_params], dim=1)
        )
    else:
        global_intra_ctx = _optional_module(model.global_intra_context, index)(
            y_hat_slices[-1],
            anchor_hat,
        )
        params = model.entropy_parameters_nonanchor[index](
            torch.cat(
                [
                    local_ctx,
                    global_intra_ctx,
                    cast(Tensor, global_inter_ctx),
                    cast(Tensor, channel_ctx),
                    hyper_params,
                ],
                dim=1,
            )
        )
    scales, means = params.chunk(2, 1)
    return checkerboard_nonanchor(scales), checkerboard_nonanchor(means)


def entropy_coder_state(model: MLICPlusPlus) -> Tuple[List[List[int]], List[int], List[int]]:
    return (
        model.gaussian_conditional.quantized_cdf.tolist(),
        model.gaussian_conditional.cdf_length.reshape(-1).int().tolist(),
        model.gaussian_conditional.offset.reshape(-1).int().tolist(),
    )


def compress_single(
    model: MLICPlusPlus,
    y_slices: Sequence[Tensor],
    hyper_params: Tensor,
    hyper_means: Tensor,
) -> bytes:
    encoder = BufferedRansEncoder()
    symbols_list: List[int] = []
    indexes_list: List[int] = []
    y_hat_slices = []

    for index, y_slice in enumerate(y_slices):
        slice_anchor, slice_nonanchor = checkerboard_split(y_slice)
        (
            scales_anchor,
            means_anchor,
            global_inter_ctx,
            channel_ctx,
        ) = anchor_distribution(model, index, y_hat_slices, hyper_params)
        slice_anchor = compress_anchor_symbols(
            model.gaussian_conditional,
            slice_anchor,
            scales_anchor,
            means_anchor,
            symbols_list,
            indexes_list,
        )
        lrp_anchor = model.lrp_anchor[index](lrp_inputs(hyper_means, y_hat_slices, slice_anchor))
        slice_anchor = slice_anchor + checkerboard_anchor(lrp_anchor)

        scales_nonanchor, means_nonanchor = nonanchor_distribution(
            model,
            index,
            y_hat_slices,
            hyper_params,
            slice_anchor,
            global_inter_ctx,
            channel_ctx,
        )
        slice_nonanchor = compress_nonanchor_symbols(
            model.gaussian_conditional,
            slice_nonanchor,
            scales_nonanchor,
            means_nonanchor,
            symbols_list,
            indexes_list,
        )
        y_hat_slice = slice_anchor + slice_nonanchor
        lrp_nonanchor = model.lrp_nonanchor[index](
            lrp_inputs(hyper_means, y_hat_slices, y_hat_slice)
        )
        y_hat_slice = y_hat_slice + checkerboard_nonanchor(lrp_nonanchor)
        y_hat_slices.append(y_hat_slice)

    cdf, cdf_lengths, offsets = entropy_coder_state(model)
    encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
    return cast(bytes, encoder.flush())


def decompress_single(
    model: MLICPlusPlus,
    y_string: bytes,
    hyper_params: Tensor,
    hyper_means: Tensor,
) -> Tensor:
    decoder = RansDecoder()
    decoder.set_stream(y_string)
    cdf, cdf_lengths, offsets = entropy_coder_state(model)
    y_hat_slices = []

    for index in range(model.slice_num):
        (
            scales_anchor,
            means_anchor,
            global_inter_ctx,
            channel_ctx,
        ) = anchor_distribution(model, index, y_hat_slices, hyper_params)
        slice_anchor = decompress_anchor_symbols(
            model.gaussian_conditional,
            scales_anchor,
            means_anchor,
            decoder,
            cdf,
            cdf_lengths,
            offsets,
        )
        lrp_anchor = model.lrp_anchor[index](lrp_inputs(hyper_means, y_hat_slices, slice_anchor))
        slice_anchor = slice_anchor + checkerboard_anchor(lrp_anchor)

        scales_nonanchor, means_nonanchor = nonanchor_distribution(
            model,
            index,
            y_hat_slices,
            hyper_params,
            slice_anchor,
            global_inter_ctx,
            channel_ctx,
        )
        slice_nonanchor = decompress_nonanchor_symbols(
            model.gaussian_conditional,
            scales_nonanchor,
            means_nonanchor,
            decoder,
            cdf,
            cdf_lengths,
            offsets,
        )
        y_hat_slice = slice_anchor + slice_nonanchor
        lrp_nonanchor = model.lrp_nonanchor[index](
            lrp_inputs(hyper_means, y_hat_slices, y_hat_slice)
        )
        y_hat_slice = y_hat_slice + checkerboard_nonanchor(lrp_nonanchor)
        y_hat_slices.append(y_hat_slice)

    return torch.cat(y_hat_slices, dim=1)
