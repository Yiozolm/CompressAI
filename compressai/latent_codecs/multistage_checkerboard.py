"""Multi-stage checkerboard latent codec used by TinyLIC and ShiftLIC large.

Both upstream models share the *exact* same staged checkerboard schedule on
the latent ``y``:

* ``num_iters`` = 4 channel-wise slices, with sizes derived from a "gamma"
  schedule (``cosine`` for TinyLIC, ``linear`` for ShiftLIC).
* The first slice is decoded in 4 sub-stages (the four checkerboard
  quadrants); slices 1 and 2 each in 2 sub-stages (checkerboard halves);
  the last slice is decoded in a single one-shot Gaussian step.
* Each sub-stage uses its own ``MultistageMaskedConv2d`` (mask A / B / C /
  B / B; kernel sizes 3/5/5/5/5) for the spatial-context branch
  (``sc_transform_{1..5}``).
* A per-iteration ``entropy_parameters_{1..4}`` stack reduces concatenated
  ``[hyper_params, sc_params, cc_params]`` to ``(2 * slice_ch)`` Gaussian
  parameters.
* A per-iteration ``cc_transforms[i]`` projects the cumulative
  ``[hyper_params, prior_y_hat_slices]`` to a per-slice
  ``(2 * slice_ch)`` cross-channel prior.

Only the ``cc_transforms`` flavour differs between the two backbones — the
caller injects it via ``make_cc_transform``:

* TinyLIC:  ``conv5 -> GELU -> conv5 -> GELU -> conv3``.
* ShiftLIC: ``ResidualShiftBlock × 4`` interleaved with two ``GELU``s.

Module / parameter naming is preserved verbatim from upstream so checkpoints
load with ``module.``-stripping only.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from torch import Tensor

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import GaussianConditional
from compressai.layers import MultistageMaskedConv2d, conv
from compressai.ops import (
    demultiplex,
    demultiplex_v2,
    multiplex,
    multiplex_v2,
    quantize_ste,
)
from compressai.registry import register_module

from .base import LatentCodec


__all__ = [
    "MultistageCheckerboardLatentCodec",
    "gamma_func",
]


def gamma_func(mode: str) -> Callable[[float], float]:
    """Schedule controlling per-iteration latent slice sizes."""
    if mode == "linear":
        return lambda r: 1.0 - r
    if mode == "cosine":
        return lambda r: float(np.cos(r * np.pi / 2.0))
    if mode == "square":
        return lambda r: 1.0 - r**2
    if mode == "cubic":
        return lambda r: 1.0 - r**3
    raise ValueError(f"Unknown gamma mode: {mode!r}")


def _slice_sizes(channels: int, num_iters: int, gamma) -> List[int]:
    """Channel-wise slice sizes for the staged loop (sums to ``channels``)."""
    sizes: List[int] = []
    remaining = channels
    for t in range(1, num_iters + 1):
        n = 0 if t == num_iters else int(np.ceil(gamma(t / num_iters) * channels))
        sizes.append(remaining - n)
        remaining = n
    return sizes


def _default_make_cc_transform(in_ch: int, out_ch: int) -> nn.Module:
    """TinyLIC-style cc_transform.

    ``conv5 -> GELU -> conv5 -> GELU -> conv3``; the first conv ramps
    ``in_ch -> out_ch // 2``, the inner conv stays at ``out_ch // 2``, the
    last conv doubles to ``out_ch`` (the codec consumes ``2 * slice_size``
    scale+mean channels).
    """
    if out_ch % 2 != 0:
        raise ValueError(
            "_default_make_cc_transform expects out_ch to be even (codec "
            f"passes 2*slice_size); got {out_ch}"
        )
    inner = out_ch // 2
    return nn.Sequential(
        conv(in_ch, inner, kernel_size=5, stride=1),
        nn.GELU(),
        conv(inner, inner, kernel_size=5, stride=1),
        nn.GELU(),
        conv(inner, out_ch, kernel_size=3, stride=1),
    )


@register_module("MultistageCheckerboardLatentCodec")
class MultistageCheckerboardLatentCodec(LatentCodec):
    """Staged channel + checkerboard codec shared by TinyLIC / ShiftLIC large.

    Args:
        channels: Latent channels ``M`` (must match the encoder output).
        hyper_channels: Channels of the hyper-decoder output ``params``.
            Both TinyLIC and ShiftLIC large feed ``2 * M``; configurable for
            future variants.
        num_iters: Number of channel slices (default 4 — the only setting the
            current sc/ep masks were designed for).
        gamma_mode: Slice-size schedule. ``"cosine"`` (TinyLIC) or
            ``"linear"`` (ShiftLIC). See :func:`gamma_func`.
        make_cc_transform: Factory ``(in_ch, out_ch) -> nn.Module`` building
            one cc_transform. Defaults to TinyLIC's conv stack; pass
            :class:`compressai.layers.lic.shift.ResidualShiftStack` for
            ShiftLIC large.
        mask_kernel_sizes: Kernel sizes for the five
            ``MultistageMaskedConv2d`` instances ``sc_transform_{1..5}``.
            Fixed at ``(3, 5, 5, 5, 5)`` upstream — exposed here for parity
            with the comment in upstream code.
    """

    def __init__(
        self,
        channels: int,
        hyper_channels: int,
        num_iters: int = 4,
        gamma_mode: str = "cosine",
        make_cc_transform: Callable[[int, int], nn.Module] = _default_make_cc_transform,
        mask_kernel_sizes: Sequence[int] = (3, 5, 5, 5, 5),
    ) -> None:
        super().__init__()
        if num_iters != 4:
            raise NotImplementedError(
                "MultistageCheckerboardLatentCodec only supports num_iters=4"
            )
        if len(mask_kernel_sizes) != 5:
            raise ValueError("mask_kernel_sizes must have 5 entries")

        self.M = int(channels)
        self.hyper_channels = int(hyper_channels)
        self.num_iters = int(num_iters)
        self.gamma_mode = gamma_mode
        self.gamma = gamma_func(gamma_mode)

        slice_sizes = _slice_sizes(self.M, self.num_iters, self.gamma)
        self._slice_sizes = slice_sizes
        s0, s1, s2, _s3 = slice_sizes

        self.gaussian_conditional = GaussianConditional(None)

        # sc_transforms operate within a single slice, doubling channels
        # (scale + mean). Kernel/mask types are fixed by the upstream design.
        self.sc_transform_1 = MultistageMaskedConv2d(
            s0, s0 * 2,
            kernel_size=mask_kernel_sizes[0],
            padding=mask_kernel_sizes[0] // 2,
            stride=1, mask_type="A",
        )
        self.sc_transform_2 = MultistageMaskedConv2d(
            s0, s0 * 2,
            kernel_size=mask_kernel_sizes[1],
            padding=mask_kernel_sizes[1] // 2,
            stride=1, mask_type="B",
        )
        self.sc_transform_3 = MultistageMaskedConv2d(
            s0, s0 * 2,
            kernel_size=mask_kernel_sizes[2],
            padding=mask_kernel_sizes[2] // 2,
            stride=1, mask_type="C",
        )
        self.sc_transform_4 = MultistageMaskedConv2d(
            s1, s1 * 2,
            kernel_size=mask_kernel_sizes[3],
            padding=mask_kernel_sizes[3] // 2,
            stride=1, mask_type="B",
        )
        self.sc_transform_5 = MultistageMaskedConv2d(
            s2, s2 * 2,
            kernel_size=mask_kernel_sizes[4],
            padding=mask_kernel_sizes[4] // 2,
            stride=1, mask_type="B",
        )

        # entropy_parameters_{1..3}: input is concat(hyper, sc, cc); each of
        # sc and cc is `slice_ch * 2`, so input channel count is
        # `hyper_ch + 4*slice_ch` (== `hyper_ch + slice_ch * 12 // 3`).
        # entropy_parameters_4: no sc branch — input is concat(hyper, cc),
        # `hyper_ch + 2 * slice_ch` (== `hyper_ch + slice_ch * 6 // 3`).
        self.entropy_parameters_1 = self._build_entropy_parameters(s0, with_sc=True)
        self.entropy_parameters_2 = self._build_entropy_parameters(s1, with_sc=True)
        self.entropy_parameters_3 = self._build_entropy_parameters(s2, with_sc=True)
        # iter 3: input is `hyper + cc` only; output is `slice3 * 2`.
        self.entropy_parameters_4 = self._build_entropy_parameters(
            slice_sizes[3], with_sc=False
        )

        # cc_transforms: one per iteration; `in_ch` is the cumulative
        # `[hyper_params, prior_y_hat_slices]` channel count, `out_ch` is the
        # current slice scale + mean (== `2 * slice_size[i]`).
        cc_transforms = []
        cumulative = 0
        for i in range(self.num_iters):
            in_ch = self.hyper_channels + cumulative
            out_ch = 2 * slice_sizes[i]
            cc_transforms.append(make_cc_transform(in_ch, out_ch))
            cumulative += slice_sizes[i]
        self.cc_transforms = nn.ModuleList(cc_transforms)

    def _build_entropy_parameters(self, slice_ch: int, with_sc: bool) -> nn.Sequential:
        """Build one ``entropy_parameters_*`` tower.

        Channel ramp matches upstream: 12/3 -> 10/3 -> 8/3 -> 6/3
        (with sc), or 6/3 -> 6/3 -> 6/3 -> 6/3 (no sc).
        """
        if with_sc:
            in_ch = self.hyper_channels + slice_ch * 12 // 3
            mid1 = slice_ch * 10 // 3
            mid2 = slice_ch * 8 // 3
            out_ch = slice_ch * 6 // 3
        else:
            in_ch = self.hyper_channels + slice_ch * 6 // 3
            mid1 = slice_ch * 6 // 3
            mid2 = slice_ch * 6 // 3
            out_ch = slice_ch * 6 // 3
        return nn.Sequential(
            conv(in_ch, mid1, kernel_size=1, stride=1),
            nn.GELU(),
            conv(mid1, mid2, kernel_size=1, stride=1),
            nn.GELU(),
            conv(mid2, out_ch, kernel_size=1, stride=1),
        )

    # ------------------------------------------------------------------
    # forward (training / eval): returns y_hat + per-element y likelihood
    # ------------------------------------------------------------------
    def forward(self, y: Tensor, params: Tensor) -> Dict[str, Any]:
        slice_sizes = self._slice_sizes
        y_slices = y.split(tuple(slice_sizes), dim=1)
        y_hat_slices: List[Tensor] = []
        y_likelihood: List[Tensor] = []

        for slice_index, y_slice in enumerate(y_slices):
            support = torch.cat([params] + y_hat_slices, dim=1)
            cc_params = self.cc_transforms[slice_index](support)

            if slice_index == 0:
                # 4-stage checkerboard within slice 0.
                sc_params_1 = torch.zeros_like(cc_params)
                sc_params = sc_params_1
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_0 = y_hat_slice.clone()
                y_0[:, :, 0::2, 1::2] = 0
                y_0[:, :, 1::2, :] = 0
                sc_params_2 = self.sc_transform_1(y_0)
                sc_params_2[:, :, 0::2, :] = 0
                sc_params_2[:, :, 1::2, 0::2] = 0
                sc_params = sc_params_1 + sc_params_2
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_1 = y_hat_slice.clone()
                y_1[:, :, 0::2, 1::2] = 0
                y_1[:, :, 1::2, 0::2] = 0
                sc_params_3 = self.sc_transform_2(y_1)
                sc_params_3[:, :, 0::2, 0::2] = 0
                sc_params_3[:, :, 1::2, :] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_2 = y_hat_slice.clone()
                y_2[:, :, 1::2, 0::2] = 0
                sc_params_4 = self.sc_transform_3(y_2)
                sc_params_4[:, :, 0::2, :] = 0
                sc_params_4[:, :, 1::2, 1::2] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3 + sc_params_4
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

            elif slice_index == 1:
                sc_params = torch.zeros_like(cc_params)
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_half = y_hat_slice.clone()
                y_half[:, :, 0::2, 0::2] = 0
                y_half[:, :, 1::2, 1::2] = 0
                sc_params = self.sc_transform_4(y_half)
                sc_params[:, :, 0::2, 1::2] = 0
                sc_params[:, :, 1::2, 0::2] = 0
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

            elif slice_index == 2:
                sc_params = torch.zeros_like(cc_params)
                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_half = y_hat_slice.clone()
                y_half[:, :, 0::2, 1::2] = 0
                y_half[:, :, 1::2, 0::2] = 0
                sc_params = self.sc_transform_5(y_half)
                sc_params[:, :, 0::2, 0::2] = 0
                sc_params[:, :, 1::2, 1::2] = 0
                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

            else:  # slice_index == 3
                gaussian_params = self.entropy_parameters_4(
                    torch.cat((params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

            y_hat_slices.append(y_hat_slice)
            _, y_slice_likelihood = self.gaussian_conditional(
                y_slice, scales_hat, means=means_hat
            )
            y_likelihood.append(y_slice_likelihood)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        return {
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods},
        }

    # ------------------------------------------------------------------
    # compress / decompress: bitstream round-trip via Rans
    # ------------------------------------------------------------------
    def compress(self, y: Tensor, params: Tensor) -> Dict[str, Any]:
        slice_sizes = self._slice_sizes
        y_slices = y.split(tuple(slice_sizes), dim=1)
        y_hat_slices: List[Tensor] = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list: List[int] = []
        indexes_list: List[int] = []

        for slice_index, y_slice in enumerate(y_slices):
            support = torch.cat([params] + y_hat_slices, dim=1)
            cc_params = self.cc_transforms[slice_index](support)
            B, _, H, W = y_slice.shape

            if slice_index == 0:
                y_slice_0, y_slice_1, y_slice_2, y_slice_3 = demultiplex_v2(y_slice)

                sc_params_1 = torch.zeros(
                    B, y_slice.shape[1] * 2, H, W,
                    device=y_slice.device, dtype=y_slice.dtype,
                )
                sc_params = sc_params_1
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                scales_hat_0, _, _, _ = demultiplex_v2(scales_hat)
                means_hat_0, _, _, _ = demultiplex_v2(means_hat)
                index_0 = self.gaussian_conditional.build_indexes(scales_hat_0)
                y_q_slice_0 = self.gaussian_conditional.quantize(
                    y_slice_0, "symbols", means_hat_0
                )
                y_hat_slice_0 = y_q_slice_0 + means_hat_0
                symbols_list.extend(y_q_slice_0.reshape(-1).tolist())
                indexes_list.extend(index_0.reshape(-1).tolist())

                y_hat_slice = multiplex_v2(
                    y_hat_slice_0,
                    torch.zeros_like(y_hat_slice_0),
                    torch.zeros_like(y_hat_slice_0),
                    torch.zeros_like(y_hat_slice_0),
                )
                sc_params_2 = self.sc_transform_1(y_hat_slice)
                sc_params_2[:, :, 0::2, :] = 0
                sc_params_2[:, :, 1::2, 0::2] = 0
                sc_params = sc_params_1 + sc_params_2
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                _, scales_hat_1, _, _ = demultiplex_v2(scales_hat)
                _, means_hat_1, _, _ = demultiplex_v2(means_hat)
                index_1 = self.gaussian_conditional.build_indexes(scales_hat_1)
                y_q_slice_1 = self.gaussian_conditional.quantize(
                    y_slice_1, "symbols", means_hat_1
                )
                y_hat_slice_1 = y_q_slice_1 + means_hat_1
                symbols_list.extend(y_q_slice_1.reshape(-1).tolist())
                indexes_list.extend(index_1.reshape(-1).tolist())

                y_hat_slice = multiplex_v2(
                    y_hat_slice_0, y_hat_slice_1,
                    torch.zeros_like(y_hat_slice_0),
                    torch.zeros_like(y_hat_slice_0),
                )
                sc_params_3 = self.sc_transform_2(y_hat_slice)
                sc_params_3[:, :, 0::2, 0::2] = 0
                sc_params_3[:, :, 1::2, :] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                _, _, scales_hat_2, _ = demultiplex_v2(scales_hat)
                _, _, means_hat_2, _ = demultiplex_v2(means_hat)
                index_2 = self.gaussian_conditional.build_indexes(scales_hat_2)
                y_q_slice_2 = self.gaussian_conditional.quantize(
                    y_slice_2, "symbols", means_hat_2
                )
                y_hat_slice_2 = y_q_slice_2 + means_hat_2
                symbols_list.extend(y_q_slice_2.reshape(-1).tolist())
                indexes_list.extend(index_2.reshape(-1).tolist())

                y_hat_slice = multiplex_v2(
                    y_hat_slice_0, y_hat_slice_1, y_hat_slice_2,
                    torch.zeros_like(y_hat_slice_0),
                )
                sc_params_4 = self.sc_transform_3(y_hat_slice)
                sc_params_4[:, :, 0::2, :] = 0
                sc_params_4[:, :, 1::2, 1::2] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3 + sc_params_4
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                _, _, _, scales_hat_3 = demultiplex_v2(scales_hat)
                _, _, _, means_hat_3 = demultiplex_v2(means_hat)
                index_3 = self.gaussian_conditional.build_indexes(scales_hat_3)
                y_q_slice_3 = self.gaussian_conditional.quantize(
                    y_slice_3, "symbols", means_hat_3
                )
                y_hat_slice_3 = y_q_slice_3 + means_hat_3
                symbols_list.extend(y_q_slice_3.reshape(-1).tolist())
                indexes_list.extend(index_3.reshape(-1).tolist())

                y_hat_slice = multiplex_v2(
                    y_hat_slice_0, y_hat_slice_1, y_hat_slice_2, y_hat_slice_3
                )
                y_hat_slices.append(y_hat_slice)

            elif slice_index == 1:
                y_slice_anchor, y_slice_non_anchor = demultiplex(y_slice)
                zero_sc = torch.zeros(
                    B, y_slice.shape[1] * 2, H, W,
                    device=y_slice.device, dtype=y_slice.dtype,
                )
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, zero_sc, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                scales_hat_anchor, _ = demultiplex(scales_hat)
                means_hat_anchor, _ = demultiplex(means_hat)
                index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
                y_q_anchor = self.gaussian_conditional.quantize(
                    y_slice_anchor, "symbols", means_hat_anchor
                )
                y_hat_anchor = y_q_anchor + means_hat_anchor
                symbols_list.extend(y_q_anchor.reshape(-1).tolist())
                indexes_list.extend(index_anchor.reshape(-1).tolist())

                y_hat_full = multiplex(y_hat_anchor, torch.zeros_like(y_hat_anchor))
                sc_params = self.sc_transform_4(y_hat_full)
                sc_params[:, :, 0::2, 1::2] = 0
                sc_params[:, :, 1::2, 0::2] = 0
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                _, scales_hat_non = demultiplex(scales_hat)
                _, means_hat_non = demultiplex(means_hat)
                index_non = self.gaussian_conditional.build_indexes(scales_hat_non)
                y_q_non = self.gaussian_conditional.quantize(
                    y_slice_non_anchor, "symbols", means_hat_non
                )
                y_hat_non = y_q_non + means_hat_non
                symbols_list.extend(y_q_non.reshape(-1).tolist())
                indexes_list.extend(index_non.reshape(-1).tolist())

                y_hat_slices.append(multiplex(y_hat_anchor, y_hat_non))

            elif slice_index == 2:
                # Note: the upstream demultiplex order for slice 2 is
                # (non_anchor, anchor) — kept for state_dict / bitstream
                # parity.
                y_slice_non_anchor, y_slice_anchor = demultiplex(y_slice)
                zero_sc = torch.zeros(
                    B, y_slice.shape[1] * 2, H, W,
                    device=y_slice.device, dtype=y_slice.dtype,
                )
                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, zero_sc, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                _, scales_hat_anchor = demultiplex(scales_hat)
                _, means_hat_anchor = demultiplex(means_hat)
                index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
                y_q_anchor = self.gaussian_conditional.quantize(
                    y_slice_anchor, "symbols", means_hat_anchor
                )
                y_hat_anchor = y_q_anchor + means_hat_anchor
                symbols_list.extend(y_q_anchor.reshape(-1).tolist())
                indexes_list.extend(index_anchor.reshape(-1).tolist())

                y_hat_full = multiplex(torch.zeros_like(y_hat_anchor), y_hat_anchor)
                sc_params = self.sc_transform_5(y_hat_full)
                sc_params[:, :, 0::2, 0::2] = 0
                sc_params[:, :, 1::2, 1::2] = 0
                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                scales_hat_non, _ = demultiplex(scales_hat)
                means_hat_non, _ = demultiplex(means_hat)
                index_non = self.gaussian_conditional.build_indexes(scales_hat_non)
                y_q_non = self.gaussian_conditional.quantize(
                    y_slice_non_anchor, "symbols", means_hat_non
                )
                y_hat_non = y_q_non + means_hat_non
                symbols_list.extend(y_q_non.reshape(-1).tolist())
                indexes_list.extend(index_non.reshape(-1).tolist())

                y_hat_slices.append(multiplex(y_hat_non, y_hat_anchor))

            else:  # slice_index == 3
                gaussian_params = self.entropy_parameters_4(
                    torch.cat((params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                index = self.gaussian_conditional.build_indexes(scales_hat)
                y_q = self.gaussian_conditional.quantize(
                    y_slice, "symbols", means_hat
                )
                y_hat_slice = y_q + means_hat
                symbols_list.extend(y_q.reshape(-1).tolist())
                indexes_list.extend(index.reshape(-1).tolist())
                y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )
        y_string = encoder.flush()
        return {"strings": [[y_string]], "shape": None}

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Tuple[int, int],
        params: Tensor,
    ) -> Dict[str, Any]:
        """Decode latent ``y_hat`` from the per-slice bitstream.

        Args:
            strings: ``[[y_bytes]]`` (single-element outer list, since the
                staged loop emits a single concatenated byte buffer).
            shape: ``(h, w)`` of the *hyper* latent (i.e. ``z`` resolution);
                used to derive the latent ``y`` resolution as ``(4h, 4w)``.
            params: ``h_s(z_hat)`` of shape ``(B, hyper_channels, 4h, 4w)``.
        """
        assert isinstance(strings, list) and len(strings) == 1
        y_string = strings[0][0]
        y_hat_slices: List[Tensor] = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)
        slice_sizes = self._slice_sizes
        zh, zw = shape
        device = params.device
        dtype = params.dtype

        for slice_index in range(self.num_iters):
            support = torch.cat([params] + y_hat_slices, dim=1)
            cc_params = self.cc_transforms[slice_index](support)
            slice_ch = slice_sizes[slice_index]

            if slice_index == 0:
                sc_params_1 = torch.zeros(
                    1, slice_ch * 2, zh * 4, zw * 4,
                    device=device, dtype=dtype,
                )
                sc_params = sc_params_1
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                scales_hat_0, _, _, _ = demultiplex_v2(scales_hat)
                means_hat_0, _, _, _ = demultiplex_v2(means_hat)
                index_0 = self.gaussian_conditional.build_indexes(scales_hat_0)
                rv = decoder.decode_stream(
                    index_0.reshape(-1).tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, zh * 2, zw * 2)
                y_hat_slice_0 = self.gaussian_conditional.dequantize(rv, means_hat_0)

                y_hat_full = multiplex_v2(
                    y_hat_slice_0,
                    torch.zeros_like(y_hat_slice_0),
                    torch.zeros_like(y_hat_slice_0),
                    torch.zeros_like(y_hat_slice_0),
                )
                sc_params_2 = self.sc_transform_1(y_hat_full)
                sc_params_2[:, :, 0::2, :] = 0
                sc_params_2[:, :, 1::2, 0::2] = 0
                sc_params = sc_params_1 + sc_params_2
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                _, scales_hat_1, _, _ = demultiplex_v2(scales_hat)
                _, means_hat_1, _, _ = demultiplex_v2(means_hat)
                index_1 = self.gaussian_conditional.build_indexes(scales_hat_1)
                rv = decoder.decode_stream(
                    index_1.reshape(-1).tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, zh * 2, zw * 2)
                y_hat_slice_1 = self.gaussian_conditional.dequantize(rv, means_hat_1)

                y_hat_full = multiplex_v2(
                    y_hat_slice_0, y_hat_slice_1,
                    torch.zeros_like(y_hat_slice_0),
                    torch.zeros_like(y_hat_slice_0),
                )
                sc_params_3 = self.sc_transform_2(y_hat_full)
                sc_params_3[:, :, 0::2, 0::2] = 0
                sc_params_3[:, :, 1::2, :] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                _, _, scales_hat_2, _ = demultiplex_v2(scales_hat)
                _, _, means_hat_2, _ = demultiplex_v2(means_hat)
                index_2 = self.gaussian_conditional.build_indexes(scales_hat_2)
                rv = decoder.decode_stream(
                    index_2.reshape(-1).tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, zh * 2, zw * 2)
                y_hat_slice_2 = self.gaussian_conditional.dequantize(rv, means_hat_2)

                y_hat_full = multiplex_v2(
                    y_hat_slice_0, y_hat_slice_1, y_hat_slice_2,
                    torch.zeros_like(y_hat_slice_0),
                )
                sc_params_4 = self.sc_transform_3(y_hat_full)
                sc_params_4[:, :, 0::2, :] = 0
                sc_params_4[:, :, 1::2, 1::2] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3 + sc_params_4
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                _, _, _, scales_hat_3 = demultiplex_v2(scales_hat)
                _, _, _, means_hat_3 = demultiplex_v2(means_hat)
                index_3 = self.gaussian_conditional.build_indexes(scales_hat_3)
                rv = decoder.decode_stream(
                    index_3.reshape(-1).tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, zh * 2, zw * 2)
                y_hat_slice_3 = self.gaussian_conditional.dequantize(rv, means_hat_3)

                y_hat_slices.append(
                    multiplex_v2(
                        y_hat_slice_0, y_hat_slice_1, y_hat_slice_2, y_hat_slice_3
                    )
                )

            elif slice_index == 1:
                zero_sc = torch.zeros(
                    1, slice_ch * 2, zh * 4, zw * 4,
                    device=device, dtype=dtype,
                )
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, zero_sc, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                scales_hat_anchor, _ = demultiplex(scales_hat)
                means_hat_anchor, _ = demultiplex(means_hat)
                index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
                rv = decoder.decode_stream(
                    index_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, zh * 2, zw * 2)
                y_hat_anchor = self.gaussian_conditional.dequantize(rv, means_hat_anchor)

                y_hat_full = multiplex(y_hat_anchor, torch.zeros_like(y_hat_anchor))
                sc_params = self.sc_transform_4(y_hat_full)
                sc_params[:, :, 0::2, 1::2] = 0
                sc_params[:, :, 1::2, 0::2] = 0
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                _, scales_hat_non = demultiplex(scales_hat)
                _, means_hat_non = demultiplex(means_hat)
                index_non = self.gaussian_conditional.build_indexes(scales_hat_non)
                rv = decoder.decode_stream(
                    index_non.reshape(-1).tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, zh * 2, zw * 2)
                y_hat_non = self.gaussian_conditional.dequantize(rv, means_hat_non)
                y_hat_slices.append(multiplex(y_hat_anchor, y_hat_non))

            elif slice_index == 2:
                zero_sc = torch.zeros(
                    1, slice_ch * 2, zh * 4, zw * 4,
                    device=device, dtype=dtype,
                )
                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, zero_sc, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                _, scales_hat_anchor = demultiplex(scales_hat)
                _, means_hat_anchor = demultiplex(means_hat)
                index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
                rv = decoder.decode_stream(
                    index_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, zh * 2, zw * 2)
                y_hat_anchor = self.gaussian_conditional.dequantize(rv, means_hat_anchor)

                y_hat_full = multiplex(torch.zeros_like(y_hat_anchor), y_hat_anchor)
                sc_params = self.sc_transform_5(y_hat_full)
                sc_params[:, :, 0::2, 0::2] = 0
                sc_params[:, :, 1::2, 1::2] = 0
                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                scales_hat_non, _ = demultiplex(scales_hat)
                means_hat_non, _ = demultiplex(means_hat)
                index_non = self.gaussian_conditional.build_indexes(scales_hat_non)
                rv = decoder.decode_stream(
                    index_non.reshape(-1).tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, zh * 2, zw * 2)
                y_hat_non = self.gaussian_conditional.dequantize(rv, means_hat_non)
                y_hat_slices.append(multiplex(y_hat_non, y_hat_anchor))

            else:  # slice_index == 3
                gaussian_params = self.entropy_parameters_4(
                    torch.cat((params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
                index = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    index.reshape(-1).tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, zh * 4, zw * 4)
                y_hat_slice = self.gaussian_conditional.dequantize(rv, means_hat)
                y_hat_slices.append(y_hat_slice)

        return {"y_hat": torch.cat(y_hat_slices, dim=1)}
