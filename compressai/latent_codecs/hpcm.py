"""Hierarchical Progressive Context Modeling (HPCM) latent codec.

Reference: Lyu et al., "Learned Image Compression with Hierarchical Progressive
Context Modeling", ICCV 2025 (`arXiv:2507.19125`_).

This codec encapsulates the three-stage spatial autoregressive prior used by
HPCM. Each call sees a single ``y`` latent tensor plus the synthesised hyper
parameters from the model and runs the full s1 / s2 / s3 schedule:

    * ``s1`` — 4× downsampled, two anchor / non-anchor steps
    * ``s2`` — 2× downsampled, four steps
    * ``s3`` — full resolution, eight steps

The hyperprior side (``means_hyper`` / ``scales_hyper`` learnt parameters and
the GGM hyperprior bottleneck) is also owned here so the model layer only
hands over ``y`` and the synthesised ``params``. The codec is compatible with
both the attention-equipped variant ("HPCM_Base" / "HPCM_Large") and the
attention-free PhiContext variant.

.. _arXiv:2507.19125: https://arxiv.org/abs/2507.19125
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import GeneralizedGaussianConditional
from compressai.ops import quantize_ste
from compressai.registry import register_module

from .base import LatentCodec

__all__ = [
    "HierarchicalProgressiveLatentCodec",
]


def _identity_module() -> nn.Module:
    return nn.Identity()


@register_module("HierarchicalProgressiveLatentCodec")
class HierarchicalProgressiveLatentCodec(LatentCodec):
    """Three-stage hierarchical progressive context codec for HPCM.

    Args:
        latent_channels: Number of channels in the ``y`` latent (``M``).
        hyper_channels: Number of channels in the ``z`` latent (``N``); used to
            shape the per-channel learnt hyper buffers.
        spatial_prior_s1_s2: Module producing the (μ, σ) tensor of shape
            ``(B, 2*M, H, W)`` from the concatenated ``[context, y_hat_so_far]``
            tensor of shape ``(B, 3*M, H, W)`` for s1 and s2 stages.
        spatial_prior_s3: Same contract as ``spatial_prior_s1_s2`` but used in
            the s3 stage; HPCM_Large makes this a deeper module.
        adaptor_s1 / adaptor_s2 / adaptor_s3: ``ModuleList`` of ``1x1`` convs
            (``3M -> 3M``) applied before each spatial-prior call.
        adaptive_params: ``ParameterList`` of ten ``(1, 3M, 1, 1)`` modulation
            scales (1 for s1, 3 for s2, 6 for s3 — matches HPCM upstream).
        context_net: ``ModuleList`` of two ``1x1`` convs mixing the
            cross-stage context.
        attn_s1 / attn_s2 / attn_s3: Optional cross-attention modules
            implementing :class:`WindowedCrossAttention`-style ``forward(query,
            context)``. Pass ``None`` (or :class:`nn.Identity`) for the
            PhiContext variant — the codec then uses the spatial-prior output
            directly as the next-step context, matching upstream behaviour.
        gaussian_conditional: Optional pre-built
            :class:`GeneralizedGaussianConditional`. Defaults to a fresh GGM
            with ``beta=1.5``.
    """

    def __init__(
        self,
        latent_channels: int,
        hyper_channels: int,
        spatial_prior_s1_s2: nn.Module,
        spatial_prior_s3: nn.Module,
        adaptor_s1: nn.ModuleList,
        adaptor_s2: nn.ModuleList,
        adaptor_s3: nn.ModuleList,
        adaptive_params: nn.ParameterList,
        context_net: nn.ModuleList,
        attn_s1: Optional[nn.Module] = None,
        attn_s2: Optional[nn.Module] = None,
        attn_s3: Optional[nn.Module] = None,
        gaussian_conditional: Optional[GeneralizedGaussianConditional] = None,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.hyper_channels = int(hyper_channels)

        self.spatial_prior_s1_s2 = spatial_prior_s1_s2
        self.spatial_prior_s3 = spatial_prior_s3
        self.adaptor_s1 = adaptor_s1
        self.adaptor_s2 = adaptor_s2
        self.adaptor_s3 = adaptor_s3
        self.adaptive_params = adaptive_params
        self.context_net = context_net

        self.attn_s1 = attn_s1 if attn_s1 is not None else _identity_module()
        self.attn_s2 = attn_s2 if attn_s2 is not None else _identity_module()
        self.attn_s3 = attn_s3 if attn_s3 is not None else _identity_module()
        self.use_attention = not (
            isinstance(self.attn_s1, nn.Identity)
            and isinstance(self.attn_s2, nn.Identity)
            and isinstance(self.attn_s3, nn.Identity)
        )

        self.means_hyper = nn.Parameter(torch.zeros(1, hyper_channels, 1, 1))
        self.scales_hyper = nn.Parameter(torch.ones(1, hyper_channels, 1, 1))

        self.gaussian_conditional = (
            gaussian_conditional
            if gaussian_conditional is not None
            else GeneralizedGaussianConditional(scale_table=None, beta=1.5)
        )

        self._mask_cache_two: Dict[str, List[Tensor]] = {}
        self._mask_cache_four: Dict[str, List[Tensor]] = {}
        self._mask_cache_eight: Dict[str, List[Tensor]] = {}
        self._mask_cache_rec_s2: Dict[str, List[Tensor]] = {}

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _one_channel_two_part_masks(
        height: int, width: int, dtype: torch.dtype, device: torch.device
    ) -> List[Tensor]:
        micro = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
        mask_0 = micro.repeat(height // 2, width // 2).unsqueeze(0).unsqueeze(0)
        mask_1 = torch.ones_like(mask_0) - mask_0
        return [mask_0, mask_1]

    @staticmethod
    def _one_channel_four_part_masks(
        height: int, width: int, dtype: torch.dtype, device: torch.device
    ) -> List[Tensor]:
        patterns = [
            ((1, 0), (0, 0)),
            ((0, 1), (0, 0)),
            ((0, 0), (1, 0)),
            ((0, 0), (0, 1)),
        ]
        masks = []
        for pattern in patterns:
            micro = torch.tensor(pattern, dtype=dtype, device=device)
            tile = micro.repeat((height + 1) // 2, (width + 1) // 2)
            masks.append(tile[:height, :width].unsqueeze(0).unsqueeze(0))
        return masks

    @staticmethod
    def _one_channel_eight_part_masks(
        height: int, width: int, dtype: torch.dtype, device: torch.device
    ) -> List[Tensor]:
        patterns = [
            ((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0)),
            ((0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)),
            ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)),
            ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0)),
            ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0)),
            ((0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0)),
            ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0)),
            ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0)),
        ]
        masks = []
        for pattern in patterns:
            micro = torch.tensor(pattern, dtype=dtype, device=device).repeat(2, 2)
            tile = micro.repeat((height + 1) // 8, (width + 1) // 8)
            masks.append(tile[:height, :width].unsqueeze(0).unsqueeze(0))
        return masks

    @staticmethod
    def _one_channel_eight_part_masks_for_s1(
        height: int, width: int, dtype: torch.dtype, device: torch.device
    ) -> List[Tensor]:
        patterns = [
            ((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
            ((0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
            ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
            ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0)),
            ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
            ((0, 0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
            ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
            ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        ]
        masks = []
        for pattern in patterns:
            micro = torch.tensor(pattern, dtype=dtype, device=device)
            tile = micro.repeat((height + 1) // 4, (width + 1) // 4)
            masks.append(tile[:height, :width].unsqueeze(0).unsqueeze(0))
        return masks

    def _mask_two(
        self,
        batch: int,
        channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> List[Tensor]:
        key = f"{batch}_{channels}x{width}x{height}"
        cache = self._mask_cache_two
        if key not in cache:
            assert channels % 2 == 0
            ones = torch.ones(
                (batch, channels // 2, height, width), dtype=dtype, device=device
            )
            m0, m1 = self._one_channel_two_part_masks(height, width, dtype, device)
            cache[key] = [
                torch.cat((ones * m0, ones * m1), dim=1),
                torch.cat((ones * m1, ones * m0), dim=1),
            ]
        return [mask.to(device) for mask in cache[key]]

    def _mask_four(
        self,
        batch: int,
        channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> List[Tensor]:
        key = f"{batch}_{channels}x{width}x{height}"
        cache = self._mask_cache_four
        if key not in cache:
            assert channels % 4 == 0
            ones = torch.ones(
                (batch, channels // 4, height, width), dtype=dtype, device=device
            )
            m0, m1, m2, m3 = self._one_channel_four_part_masks(
                height, width, dtype, device
            )
            cache[key] = [
                torch.cat((ones * m0, ones * m1, ones * m2, ones * m3), dim=1),
                torch.cat((ones * m3, ones * m2, ones * m1, ones * m0), dim=1),
                torch.cat((ones * m2, ones * m3, ones * m0, ones * m1), dim=1),
                torch.cat((ones * m1, ones * m0, ones * m3, ones * m2), dim=1),
            ]
        return [mask.to(device) for mask in cache[key]]

    def _mask_eight(
        self,
        batch: int,
        channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> List[Tensor]:
        key = f"{batch}_{channels}x{width}x{height}"
        cache = self._mask_cache_eight
        if key not in cache:
            assert channels % 8 == 0
            ones = torch.ones(
                (batch, channels // 8, height, width), dtype=dtype, device=device
            )
            single = self._one_channel_eight_part_masks(height, width, dtype, device)
            cat_indices = [
                [0, 2, 4, 6, 1, 3, 5, 7],
                [1, 3, 5, 7, 0, 2, 4, 6],
                [2, 4, 6, 0, 3, 5, 7, 1],
                [3, 5, 7, 1, 2, 4, 6, 0],
                [4, 6, 0, 2, 5, 7, 1, 3],
                [5, 7, 1, 3, 4, 6, 0, 2],
                [6, 0, 2, 4, 7, 1, 3, 5],
                [7, 1, 3, 5, 6, 0, 2, 4],
            ]
            cache[key] = [
                torch.cat([ones * single[index] for index in row], dim=1)
                for row in cat_indices
            ]
        return [mask.to(device) for mask in cache[key]]

    def _mask_rec_s2(
        self,
        batch: int,
        channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> List[Tensor]:
        key = f"{batch}_{channels}x{width}x{height}"
        cache = self._mask_cache_rec_s2
        if key not in cache:
            assert channels % 4 == 0
            ones = torch.ones(
                (batch, channels // 4, height, width), dtype=dtype, device=device
            )
            m0, m1, m2, m3 = self._one_channel_four_part_masks(
                height, width, dtype, device
            )
            cache[key] = [ones * m0, ones * m1, ones * m2, ones * m3]
        return [mask.to(device) for mask in cache[key]]

    def _mask_for_s1(
        self,
        batch: int,
        channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> List[Tensor]:
        assert channels % 8 == 0
        ones = torch.ones(
            (batch, channels // 8, height, width), dtype=dtype, device=device
        )
        single = self._one_channel_eight_part_masks_for_s1(height, width, dtype, device)
        return [ones * single[i] for i in [0, 2, 4, 6, 1, 3, 5, 7]]

    # ------------------------------------------------------------------
    # Mask-driven gather / scatter helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _gather(
        y: Tensor,
        masks: Sequence[Tensor],
        batch: int,
        channels: int,
        height: int,
        width: int,
        reduce: int,
    ) -> Tensor:
        slice_size = channels // reduce
        gathered = []
        for index in range(reduce):
            chunk = y[:, slice_size * index : slice_size * (index + 1), :, :]
            gathered.append(
                chunk.masked_select(masks[index].bool()).view(
                    batch, slice_size, height, width
                )
            )
        return torch.cat(gathered, dim=1)

    @staticmethod
    def _scatter(
        y_curr: Tensor,
        masks: Sequence[Tensor],
        batch: int,
        channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        recon = torch.zeros(
            (batch, channels, height, width), dtype=dtype, device=device
        )
        full_mask = torch.cat(list(masks), dim=1)
        recon[full_mask.bool()] = y_curr.reshape(-1)
        return recon

    @staticmethod
    def _scatter_s2_hyper(
        y: Tensor,
        masks_s1: Sequence[Tensor],
        masks_s2: Sequence[Tensor],
        masks_rec_s2: Sequence[Tensor],
        batch: int,
        channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        recon = torch.zeros(
            (batch, channels, height, width), dtype=dtype, device=device
        )
        diff = torch.cat(
            [masks_s2[index] - masks_s1[index] for index in range(len(masks_s1))],
            dim=1,
        )
        rec_full = torch.cat(list(masks_rec_s2), dim=1)
        recon[~rec_full.bool()] = y[diff.bool()]
        return recon

    @staticmethod
    def _scatter_s3_hyper(
        common_params: Tensor,
        masks: Sequence[Tensor],
        batch: int,
        channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        recon = torch.zeros(
            (batch, channels, height, width), dtype=dtype, device=device
        )
        full_mask = torch.cat(list(masks), dim=1)
        recon[~full_mask.bool()] = common_params[~full_mask.bool()]
        return recon

    # ------------------------------------------------------------------
    # Per-element processing
    # ------------------------------------------------------------------
    def _process_with_mask(
        self,
        y: Tensor,
        scales: Tensor,
        means: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        scales_hat = scales * mask
        means_hat = means * mask
        y_res = (y - means_hat) * mask
        if self.training:
            y_q = quantize_ste(y_res)
        else:
            y_q = torch.round(y_res)
        y_hat = y_q + means_hat
        return y_res, y_q, y_hat, scales_hat

    def _maybe_attend(
        self, attn: nn.Module, query: Tensor, context: Tensor
    ) -> Tensor:
        if isinstance(attn, nn.Identity):
            return query
        return attn(query, context)

    # ------------------------------------------------------------------
    # Forward (training / eval)
    # ------------------------------------------------------------------
    def forward(self, y: Tensor, z: Tensor, synth_params: Tensor) -> Dict[str, Any]:
        """Runs the full HPCM forward.

        Args:
            y: Latent of shape ``(B, M, H, W)``.
            z: Hyper latent of shape ``(B, N, h, w)`` (raw output of ``h_a``;
                the codec owns the per-channel ``means_hyper`` shift and the
                rounding).
            synth_params: Output of ``h_s(z_hat)`` of shape ``(B, 2*M, H, W)``;
                the model is responsible for running ``h_s`` because it owns
                that transform.

        Returns:
            Dict with ``y_hat`` and a nested ``likelihoods`` dict.
        """
        z_likelihoods, _z_hat = self._hyper_likelihood_and_hat(z)
        y_res, _y_q, y_hat, scales_y = self._run_hpcm(y, synth_params)
        if self.training:
            noisy = y_res + torch.empty_like(y_res).uniform_(-0.5, 0.5)
            y_likelihoods = self.gaussian_conditional._likelihood(noisy, scales_y)
        else:
            y_likelihoods = self.gaussian_conditional._likelihood(
                torch.round(y_res), scales_y
            )
        if self.gaussian_conditional.use_likelihood_bound:
            y_likelihoods = self.gaussian_conditional.likelihood_lower_bound(
                y_likelihoods
            )
        return {
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def _hyper_likelihood_and_hat(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            z_res = z - self.means_hyper
            z_hat = quantize_ste(z_res) + self.means_hyper
            noisy = z_res + torch.empty_like(z_res).uniform_(-0.5, 0.5)
            z_likelihoods = self.gaussian_conditional._likelihood(
                noisy, self.scales_hyper.expand_as(noisy)
            )
        else:
            z_res_hat = torch.round(z - self.means_hyper)
            z_hat = z_res_hat + self.means_hyper
            z_likelihoods = self.gaussian_conditional._likelihood(
                z_res_hat, self.scales_hyper.expand_as(z_res_hat)
            )
        if self.gaussian_conditional.use_likelihood_bound:
            z_likelihoods = self.gaussian_conditional.likelihood_lower_bound(
                z_likelihoods
            )
        return z_likelihoods, z_hat

    # ------------------------------------------------------------------
    # Hyper helpers exposed to the model
    # ------------------------------------------------------------------
    def quantize_z(self, z: Tensor) -> Tensor:
        """Round-then-add-back-mean quantization used at inference time."""
        return torch.round(z - self.means_hyper) + self.means_hyper

    # ------------------------------------------------------------------
    # Core HPCM schedule
    # ------------------------------------------------------------------
    def _run_hpcm(
        self,
        y: Tensor,
        common_params: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, channels, height, width = y.shape
        dtype = common_params.dtype
        device = common_params.device

        # Pre-build masks used for both stages.
        masks_s2 = self._mask_for_s2(batch, channels, height, width, dtype, device)
        y_s2 = self._gather(
            y, masks_s2, batch, channels, height // 2, width // 2, reduce=8
        )
        masks_rec_s2 = self._mask_rec_s2(
            batch, channels, height // 2, width // 2, dtype, device
        )
        y_s1 = self._gather(
            y_s2, masks_rec_s2, batch, channels, height // 4, width // 4, reduce=4
        )

        scales_all, means_all = common_params.chunk(2, dim=1)
        scales_s2 = self._gather(
            scales_all, masks_s2, batch, channels, height // 2, width // 2, reduce=8
        )
        scales_s1 = self._gather(
            scales_s2, masks_rec_s2, batch, channels, height // 4, width // 4, reduce=4
        )
        means_s2 = self._gather(
            means_all, masks_s2, batch, channels, height // 2, width // 2, reduce=8
        )
        means_s1 = self._gather(
            means_s2, masks_rec_s2, batch, channels, height // 4, width // 4, reduce=4
        )
        common_params_s1 = torch.cat((scales_s1, means_s1), dim=1)
        context_next = common_params_s1

        # ---------------- Stage s1: 2 steps over 4× downsampled grid ---------
        masks_two = self._mask_two(
            batch, channels, height // 4, width // 4, dtype, device
        )
        y_res_list, y_q_list, y_hat_list, scale_list = [], [], [], []
        for index in range(2):
            if index == 0:
                y_res, y_q, y_hat, s_hat = self._process_with_mask(
                    y_s1, scales_s1, means_s1, masks_two[index]
                )
            else:
                y_hat_so_far = torch.sum(torch.stack(y_hat_list), dim=0)
                concat = torch.cat((context_next, y_hat_so_far), dim=1)
                spatial_out = self.spatial_prior_s1_s2(
                    self.adaptor_s1[index - 1](concat),
                    self.adaptive_params[index - 1],
                )
                if self.use_attention:
                    context_next = self._maybe_attend(
                        self.attn_s1, spatial_out, context_next
                    )
                    scales, means = spatial_out.chunk(2, dim=1)
                else:
                    context_next = spatial_out
                    scales, means = context_next.chunk(2, dim=1)
                y_res, y_q, y_hat, s_hat = self._process_with_mask(
                    y_s1, scales, means, masks_two[index]
                )
            y_res_list.append(y_res)
            y_q_list.append(y_q)
            y_hat_list.append(y_hat)
            scale_list.append(s_hat)

        y_res_total = torch.sum(torch.stack(y_res_list), dim=0)
        y_q_total = torch.sum(torch.stack(y_q_list), dim=0)
        y_hat_total = torch.sum(torch.stack(y_hat_list), dim=0)
        scales_total = torch.sum(torch.stack(scale_list), dim=0)

        # Up-scale s1 outputs to s2 grid
        y_res_total = self._scatter(
            y_res_total, masks_rec_s2, batch, channels, height // 2, width // 2, dtype, device
        )
        y_q_total = self._scatter(
            y_q_total, masks_rec_s2, batch, channels, height // 2, width // 2, dtype, device
        )
        y_hat_total = self._scatter(
            y_hat_total, masks_rec_s2, batch, channels, height // 2, width // 2, dtype, device
        )
        scales_total = self._scatter(
            scales_total, masks_rec_s2, batch, channels, height // 2, width // 2, dtype, device
        )

        ctx_scales, ctx_means = context_next.chunk(2, dim=1)
        ctx_scales = self._scatter(
            ctx_scales, masks_rec_s2, batch, channels, height // 2, width // 2, dtype, device
        )
        ctx_means = self._scatter(
            ctx_means, masks_rec_s2, batch, channels, height // 2, width // 2, dtype, device
        )
        context = torch.cat((ctx_scales, ctx_means), dim=1)

        # ---------------- Stage s2: 4 steps over 2× downsampled grid ---------
        masks_s1 = self._mask_for_s1(batch, channels, height, width, dtype, device)
        scales_s2 = self._scatter_s2_hyper(
            scales_all, masks_s1, masks_s2, masks_rec_s2,
            batch, channels, height // 2, width // 2, dtype, device,
        )
        means_s2 = self._scatter_s2_hyper(
            means_all, masks_s1, masks_s2, masks_rec_s2,
            batch, channels, height // 2, width // 2, dtype, device,
        )
        common_params_s2 = torch.cat((scales_s2, means_s2), dim=1)
        context = context + common_params_s2
        context_next = self.context_net[0](context)

        masks_four = self._mask_four(
            batch, channels, height // 2, width // 2, dtype, device
        )[1:]
        y_res_list = [y_res_total]
        y_q_list = [y_q_total]
        y_hat_list = [y_hat_total]
        scale_list = [scales_total]

        for index in range(3):
            y_hat_so_far = torch.sum(torch.stack(y_hat_list), dim=0)
            concat = torch.cat((context_next, y_hat_so_far), dim=1)
            spatial_out = self.spatial_prior_s1_s2(
                self.adaptor_s2[index - 1](concat),
                self.adaptive_params[index + 1],
            )
            if self.use_attention:
                context_next = self._maybe_attend(
                    self.attn_s2, spatial_out, context_next
                )
                scales, means = spatial_out.chunk(2, dim=1)
            else:
                context_next = spatial_out
                scales, means = context_next.chunk(2, dim=1)
            y_res, y_q, y_hat, s_hat = self._process_with_mask(
                y_s2, scales, means, masks_four[index]
            )
            y_res_list.append(y_res)
            y_q_list.append(y_q)
            y_hat_list.append(y_hat)
            scale_list.append(s_hat)

        y_res_total = torch.sum(torch.stack(y_res_list), dim=0)
        y_q_total = torch.sum(torch.stack(y_q_list), dim=0)
        y_hat_total = torch.sum(torch.stack(y_hat_list), dim=0)
        scales_total = torch.sum(torch.stack(scale_list), dim=0)

        # Up-scale s2 outputs to s3 grid
        y_res_total = self._scatter(
            y_res_total, masks_s2, batch, channels, height, width, dtype, device
        )
        y_q_total = self._scatter(
            y_q_total, masks_s2, batch, channels, height, width, dtype, device
        )
        y_hat_total = self._scatter(
            y_hat_total, masks_s2, batch, channels, height, width, dtype, device
        )
        scales_total = self._scatter(
            scales_total, masks_s2, batch, channels, height, width, dtype, device
        )

        ctx_scales, ctx_means = context_next.chunk(2, dim=1)
        ctx_scales = self._scatter(
            ctx_scales, masks_s2, batch, channels, height, width, dtype, device
        )
        ctx_means = self._scatter(
            ctx_means, masks_s2, batch, channels, height, width, dtype, device
        )
        context = torch.cat((ctx_scales, ctx_means), dim=1)

        # ---------------- Stage s3: 6 steps over full-resolution grid --------
        scales_s3 = self._scatter_s3_hyper(
            scales_all, masks_s2, batch, channels, height, width, dtype, device
        )
        means_s3 = self._scatter_s3_hyper(
            means_all, masks_s2, batch, channels, height, width, dtype, device
        )
        common_params_s3 = torch.cat((scales_s3, means_s3), dim=1)
        context = context + common_params_s3
        context_next = self.context_net[1](context)

        masks_eight = self._mask_eight(
            batch, channels, height, width, dtype, device
        )[2:]
        y_res_list = [y_res_total]
        y_q_list = [y_q_total]
        y_hat_list = [y_hat_total]
        scale_list = [scales_total]

        for index in range(6):
            y_hat_so_far = torch.sum(torch.stack(y_hat_list), dim=0)
            concat = torch.cat((context_next, y_hat_so_far), dim=1)
            spatial_out = self.spatial_prior_s3(
                self.adaptor_s3[index - 1](concat),
                self.adaptive_params[index + 4],
            )
            if self.use_attention:
                context_next = self._maybe_attend(
                    self.attn_s3, spatial_out, context_next
                )
                scales, means = spatial_out.chunk(2, dim=1)
            else:
                context_next = spatial_out
                scales, means = context_next.chunk(2, dim=1)
            y_res, y_q, y_hat, s_hat = self._process_with_mask(
                y, scales, means, masks_eight[index]
            )
            y_res_list.append(y_res)
            y_q_list.append(y_q)
            y_hat_list.append(y_hat)
            scale_list.append(s_hat)

        y_res_total = torch.sum(torch.stack(y_res_list), dim=0)
        y_q_total = torch.sum(torch.stack(y_q_list), dim=0)
        y_hat_total = torch.sum(torch.stack(y_hat_list), dim=0)
        scales_total = torch.sum(torch.stack(scale_list), dim=0)

        return y_res_total, y_q_total, y_hat_total, scales_total

    def _mask_for_s2(
        self,
        batch: int,
        channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> List[Tensor]:
        # Returns 8 masks of shape (B, C, H, W); each mask has 1/4 of pixels
        # active (sum of two 4×4 patterns), so the gather to (H/2, W/2) is
        # exact. This matches candidate's `get_mask_for_s2` exactly.
        assert channels % 8 == 0
        ones = torch.ones(
            (batch, channels // 8, height, width), dtype=dtype, device=device
        )
        single = self._one_channel_eight_part_masks_for_s2(
            height, width, dtype, device
        )
        indices_a = [0, 2, 4, 6, 1, 3, 5, 7]
        indices_b = [1, 3, 5, 7, 0, 2, 4, 6]
        return [
            ones * single[indices_a[i]] + ones * single[indices_b[i]]
            for i in range(8)
        ]

    @staticmethod
    def _one_channel_eight_part_masks_for_s2(
        height: int, width: int, dtype: torch.dtype, device: torch.device
    ) -> List[Tensor]:
        patterns = [
            ((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0)),
            ((0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)),
            ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)),
            ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0)),
            ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0)),
            ((0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0)),
            ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0)),
            ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0)),
        ]
        masks = []
        for pattern in patterns:
            micro = torch.tensor(pattern, dtype=dtype, device=device)
            tile = micro.repeat((height + 1) // 4, (width + 1) // 4)
            masks.append(tile[:height, :width].unsqueeze(0).unsqueeze(0))
        return masks

    # ------------------------------------------------------------------
    # compress / decompress are deferred — current scope is forward only
    # ------------------------------------------------------------------
    def compress(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError(
            "Real-bitstream compress/decompress for HPCM is not yet wired in. "
            "Forward pass / state-dict roundtrip / rate estimation are supported."
        )

    def decompress(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError(
            "Real-bitstream compress/decompress for HPCM is not yet wired in."
        )
