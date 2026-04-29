"""Causal Context Adjustment (CCA) standalone autoencoder.

Mirror of the upstream ``LICAutoencoder`` from
M. Han, S. Jiang, S. Li, X. Deng, M. Xu, C. Zhu, S. Gu:
`"Causal Context Adjustment Loss for Learned Image Compression"
<https://arxiv.org/abs/2410.04847>`_, NeurIPS 2024.

The shared CCA bits (NAFBlock, NAFTransform) live in
``compressai/layers/lic/cca.py`` and the auxiliary CCA-loss entropy module
lives in ``compressai/entropy_models/cca.py``. This module wires the
NAFBlock-based analysis/synthesis pair to a 5-slice channel-conditional
entropy model (with full previous-slice support) and an optional auxiliary
entropy module that produces ``y_aux`` / ``y_cca`` likelihoods for
``CCARateDistortionLoss``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import math

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers.lic.blocks import ResidualBottleneckBlock
from compressai.layers.lic.cca import NAFBlock, NAFTransform
from compressai.models.base import CompressionModel, get_scale_table
from compressai.ops import quantize_ste
from compressai.registry import register_model

__all__ = ["CCAModel", "convert_upstream_state_dict"]


def _conv2d(in_channels: int, out_channels: int, kernel_size: int, stride: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def _convt2d(in_channels: int, out_channels: int, kernel_size: int, stride: int) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


def _make_cc_head(input_channels: int, hidden_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        _conv2d(input_channels, hidden_channels, kernel_size=3, stride=1),
        nn.GELU(),
        _conv2d(hidden_channels, 128, kernel_size=3, stride=1),
        nn.GELU(),
        _conv2d(128, output_channels, kernel_size=3, stride=1),
    )


def _resolve_slice_sizes(latent_channels: int, slice_proportions: Sequence[int]) -> List[int]:
    if len(slice_proportions) == 0:
        raise ValueError("slice_proportions must contain at least one entry")
    total = sum(slice_proportions)
    if total <= 0:
        raise ValueError("slice_proportions must sum to a positive integer")
    sizes = [
        int(math.floor(latent_channels * proportion / total))
        for proportion in slice_proportions
    ]
    sizes[-1] += latent_channels - sum(sizes)
    if any(size <= 0 for size in sizes):
        raise ValueError("resolved slice sizes must all be positive")
    return sizes


class _CCAEncoder(nn.Module):
    """NAFBlock + ResidualBottleneckBlock analysis transform (4 strides)."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        stage_dims: Sequence[int],
        stage_layers: Sequence[int],
    ) -> None:
        super().__init__()
        if len(stage_dims) != len(stage_layers):
            raise ValueError("stage_dims and stage_layers must have matching length")
        self.depth = len(stage_dims)
        all_dims = [in_channels, *stage_dims, latent_channels]
        self.down = nn.ModuleList(
            _conv2d(all_dims[index], all_dims[index + 1], kernel_size=5, stride=2)
            for index in range(self.depth + 1)
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                *(ResidualBottleneckBlock(stage_dims[index], stage_dims[index]) for _ in range(3)),
                *(NAFBlock(stage_dims[index]) for _ in range(stage_layers[index])),
            )
            for index in range(self.depth)
        )

    def forward(self, x: Tensor) -> Tensor:
        for index in range(self.depth):
            x = self.down[index](x)
            x = self.blocks[index](x)
        return self.down[self.depth](x)


class _CCADecoder(nn.Module):
    """NAFBlock + ResidualBottleneckBlock synthesis transform (4 strides)."""

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        stage_dims: Sequence[int],
        stage_layers: Sequence[int],
    ) -> None:
        super().__init__()
        if len(stage_dims) != len(stage_layers):
            raise ValueError("stage_dims and stage_layers must have matching length")
        self.depth = len(stage_dims)
        all_dims = [out_channels, *stage_dims, latent_channels]
        self.up = nn.ModuleList(
            _convt2d(all_dims[index + 1], all_dims[index], kernel_size=5, stride=2)
            for index in reversed(range(self.depth + 1))
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                *(NAFBlock(stage_dims[index]) for _ in range(stage_layers[index])),
                *(ResidualBottleneckBlock(stage_dims[index], stage_dims[index]) for _ in range(3)),
            )
            for index in reversed(range(self.depth))
        )

    def forward(self, x: Tensor) -> Tensor:
        for index in range(self.depth):
            x = self.up[index](x)
            x = self.blocks[index](x)
        return self.up[self.depth](x)


class _CCAAuxEntropyModel(nn.Module):
    """Auxiliary CCA entropy branch (skip-most-recent-slice support).

    Produces the ``y_aux`` (factorized) and ``y_cca`` (Gaussian-conditional)
    likelihoods used by :class:`compressai.losses.CCARateDistortionLoss`.

    Mirrors the upstream ``AuxEntropyModel`` in
    ``candidate/CCA/models/aux_em.py``: support for slice ``i`` consists of
    ``latent_*`` concatenated with ``y_hat_slices[: max(i - 1, 0)]``. The
    upstream init only allocates LRP heads for the first ``num_slices - 2``
    slices, but published checkpoints carry all ``num_slices`` heads, so this
    class allocates them all and only uses the first ``num_slices - 2`` in
    ``forward``.
    """

    def __init__(
        self,
        latent_channels: int,
        slice_sizes: Sequence[int],
        hidden_channels: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.slice_sizes: List[int] = list(map(int, slice_sizes))
        self.num_slices = len(self.slice_sizes)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)

        def support_channels(index: int) -> int:
            previous = sum(self.slice_sizes[: max(index - 1, 0)])
            return self.latent_channels + previous

        def lrp_in_channels(index: int) -> int:
            return support_channels(index) + self.slice_sizes[index]

        self.mean_support_transforms = nn.ModuleList(
            NAFTransform(
                support_channels(index),
                support_channels(index),
                self.hidden_channels,
                self.num_layers,
            )
            for index in range(self.num_slices)
        )
        self.scale_support_transforms = nn.ModuleList(
            NAFTransform(
                support_channels(index),
                support_channels(index),
                self.hidden_channels,
                self.num_layers,
            )
            for index in range(self.num_slices)
        )
        self.mean_cc_transforms = nn.ModuleList(
            _make_cc_head(support_channels(index), self.hidden_channels, self.slice_sizes[index])
            for index in range(self.num_slices)
        )
        self.scale_cc_transforms = nn.ModuleList(
            _make_cc_head(support_channels(index), self.hidden_channels, self.slice_sizes[index])
            for index in range(self.num_slices)
        )
        self.lrp_transforms = nn.ModuleList(
            _make_cc_head(lrp_in_channels(index), self.hidden_channels, self.slice_sizes[index])
            for index in range(self.num_slices)
        )

        self.y_entropy_bottleneck = EntropyBottleneck(self.latent_channels)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(
        self,
        y: Tensor,
        latent_means: Tensor,
        latent_scales: Tensor,
    ) -> Dict[str, Tensor]:
        _, y_aux_likelihoods = self.y_entropy_bottleneck(y)

        y_hat_slices: List[Tensor] = []
        cca_likelihoods: List[Tensor] = []
        usable_lrp = max(self.num_slices - 2, 0)

        for slice_index, y_slice in enumerate(y.split(self.slice_sizes, dim=1)):
            support_slices = y_hat_slices[: max(slice_index - 1, 0)]
            mean_support = torch.cat([latent_means, *support_slices], dim=1)
            mean_support = self.mean_support_transforms[slice_index](mean_support)
            mu = self.mean_cc_transforms[slice_index](mean_support)

            scale_support = torch.cat([latent_scales, *support_slices], dim=1)
            scale_support = self.scale_support_transforms[slice_index](scale_support)
            scale = self.scale_cc_transforms[slice_index](scale_support)

            _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scale, means=mu)
            cca_likelihoods.append(y_slice_likelihoods)

            if slice_index >= usable_lrp:
                continue

            y_hat_slice = quantize_ste(y_slice - mu) + mu
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = 0.5 * torch.tanh(self.lrp_transforms[slice_index](lrp_support))
            y_hat_slices.append(y_hat_slice + lrp)

        return {
            "y_aux": y_aux_likelihoods,
            "y_cca": torch.cat(cca_likelihoods, dim=1),
        }


@register_model("cca")
class CCAModel(CompressionModel):
    r"""Causal Context Adjustment standalone autoencoder.

    Mirrors the upstream ``LICAutoencoder`` from M. Han et al., NeurIPS 2024
    (`Causal Context Adjustment Loss for Learned Image Compression
    <https://arxiv.org/abs/2410.04847>`_).

    Args:
        latent_channels: Number of channels in the latent (``M``).
        hyper_channels: Number of channels in the hyper-latent (``N_z``).
        slice_proportions: Per-slice channel proportions; the actual slice
            channel widths are computed as
            ``floor(latent_channels * p / sum(p))`` with the residual added to
            the last slice. Pass ``[1] * num_slices`` for equal-sized slices.
        encoder_dims: Per-stage feature widths for the analysis transform
            (3 stages by default).
        encoder_layers: Per-stage NAFBlock counts for the analysis transform.
        em_hidden_channels: Hidden width inside the per-slice NAFTransforms.
        em_num_layers: NAFBlock count inside each per-slice NAFTransform.
        cca_training: When ``True``, allocate the auxiliary CCA entropy branch
            so that ``forward`` populates ``aux_likelihoods``.
    """

    def __init__(
        self,
        latent_channels: int = 320,
        hyper_channels: int = 192,
        slice_proportions: Sequence[int] = (1, 1, 1, 1, 1),
        encoder_dims: Sequence[int] = (192, 224, 256),
        encoder_layers: Sequence[int] = (4, 4, 4),
        em_hidden_channels: int = 224,
        em_num_layers: int = 4,
        cca_training: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        encoder_dims = tuple(encoder_dims)
        encoder_layers = tuple(encoder_layers)
        slice_proportions = tuple(int(value) for value in slice_proportions)

        self.M = int(latent_channels)
        self.N = int(hyper_channels)
        self.encoder_dims = encoder_dims
        self.encoder_layers = encoder_layers
        self.slice_proportions = slice_proportions
        self.em_hidden_channels = int(em_hidden_channels)
        self.em_num_layers = int(em_num_layers)
        self.cca_training = bool(cca_training)

        self.slice_sizes: List[int] = _resolve_slice_sizes(self.M, slice_proportions)
        self.num_slices = len(self.slice_sizes)

        self.g_a = _CCAEncoder(3, self.M, encoder_dims, encoder_layers)
        self.g_s = _CCADecoder(3, self.M, encoder_dims, encoder_layers)

        last_encoder_dim = encoder_dims[-1]
        self.h_a = nn.Sequential(
            _conv2d(self.M, last_encoder_dim, kernel_size=3, stride=1),
            nn.GELU(),
            _conv2d(last_encoder_dim, last_encoder_dim, kernel_size=5, stride=2),
            nn.GELU(),
            _conv2d(last_encoder_dim, self.N, kernel_size=5, stride=2),
        )
        self.h_mean_s = nn.Sequential(
            _convt2d(self.N, last_encoder_dim, kernel_size=5, stride=2),
            nn.GELU(),
            _convt2d(last_encoder_dim, last_encoder_dim, kernel_size=5, stride=2),
            nn.GELU(),
            _convt2d(last_encoder_dim, self.M, kernel_size=3, stride=1),
        )
        self.h_scale_s = nn.Sequential(
            _convt2d(self.N, last_encoder_dim, kernel_size=5, stride=2),
            nn.GELU(),
            _convt2d(last_encoder_dim, last_encoder_dim, kernel_size=5, stride=2),
            nn.GELU(),
            _convt2d(last_encoder_dim, self.M, kernel_size=3, stride=1),
        )

        def support_channels(index: int) -> int:
            return self.M + sum(self.slice_sizes[:index])

        def lrp_in_channels(index: int) -> int:
            return support_channels(index) + self.slice_sizes[index]

        self.mean_support_transforms = nn.ModuleList(
            NAFTransform(
                support_channels(index),
                support_channels(index),
                self.em_hidden_channels,
                self.em_num_layers,
            )
            for index in range(self.num_slices)
        )
        self.scale_support_transforms = nn.ModuleList(
            NAFTransform(
                support_channels(index),
                support_channels(index),
                self.em_hidden_channels,
                self.em_num_layers,
            )
            for index in range(self.num_slices)
        )
        self.mean_cc_transforms = nn.ModuleList(
            _make_cc_head(support_channels(index), self.em_hidden_channels, self.slice_sizes[index])
            for index in range(self.num_slices)
        )
        self.scale_cc_transforms = nn.ModuleList(
            _make_cc_head(support_channels(index), self.em_hidden_channels, self.slice_sizes[index])
            for index in range(self.num_slices)
        )
        self.lrp_transforms = nn.ModuleList(
            _make_cc_head(lrp_in_channels(index), self.em_hidden_channels, self.slice_sizes[index])
            for index in range(self.num_slices)
        )

        if self.cca_training:
            self.aux_entropy_model = _CCAAuxEntropyModel(
                self.M,
                self.slice_sizes,
                self.em_hidden_channels,
                self.em_num_layers,
            )

        self.z_entropy_bottleneck = EntropyBottleneck(self.N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x: Tensor) -> Dict[str, object]:
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.z_entropy_bottleneck(z)
        z_offset = self.z_entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset

        latent_means = self.h_mean_s(z_hat)
        latent_scales = self.h_scale_s(z_hat)

        y_hat_slices: List[Tensor] = []
        y_likelihoods: List[Tensor] = []
        for slice_index, y_slice in enumerate(y.split(self.slice_sizes, dim=1)):
            mean_support = torch.cat([latent_means, *y_hat_slices], dim=1)
            mean_support = self.mean_support_transforms[slice_index](mean_support)
            mu = self.mean_cc_transforms[slice_index](mean_support)

            scale_support = torch.cat([latent_scales, *y_hat_slices], dim=1)
            scale_support = self.scale_support_transforms[slice_index](scale_support)
            scale = self.scale_cc_transforms[slice_index](scale_support)

            _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scale, means=mu)
            y_likelihoods.append(y_slice_likelihoods)

            y_hat_slice = quantize_ste(y_slice - mu) + mu
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = 0.5 * torch.tanh(self.lrp_transforms[slice_index](lrp_support))
            y_hat_slices.append(y_hat_slice + lrp)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat)

        result: Dict[str, object] = {
            "y": y,
            "x_hat": x_hat,
            "likelihoods": {
                "y": torch.cat(y_likelihoods, dim=1),
                "z": z_likelihoods,
            },
        }
        if self.cca_training:
            result["aux_likelihoods"] = self.aux_entropy_model(y, latent_means, latent_scales)
        else:
            result["aux_likelihoods"] = None
        return result

    def update(self, scale_table: Optional[Tensor] = None, force: bool = False, **kwargs) -> bool:
        if scale_table is None:
            scale_table = get_scale_table()
        return super().update(scale_table=scale_table, force=force, **kwargs)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True):
        if _looks_like_upstream(state_dict):
            state_dict = convert_upstream_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "CCAModel":
        if _looks_like_upstream(state_dict):
            state_dict = convert_upstream_state_dict(state_dict)
        cfg = _infer_config_from_state_dict(state_dict)
        net = cls(**cfg)
        net.load_state_dict(state_dict)
        return net


# ---------------------------------------------------------------------------
# Upstream → compressai state-dict conversion.
# ---------------------------------------------------------------------------


_NAF_BLOCK_RENAMES = {
    "dwconv.": "pointwise_depthwise.",
    "sca.": "channel_attention.",
    "FFN.": "feed_forward.",
    "conv1.": "project.",
}
_NAF_TRANSFORM_RENAMES = {
    "in_conv.": "input_projection.",
    "out_conv.": "output_projection.",
}
_TOPLEVEL_RENAMES = {
    "aux_entropymodel.": "aux_entropy_model.",
}
_NAMED_PART_RENAMES = {
    "mean_NAF_transforms.": "mean_support_transforms.",
    "scale_NAF_transforms.": "scale_support_transforms.",
}


def _looks_like_upstream(state_dict: Dict[str, Tensor]) -> bool:
    for key in state_dict:
        if key.startswith("mean_NAF_transforms.") or key.startswith("scale_NAF_transforms."):
            return True
        if key.startswith("aux_entropymodel."):
            return True
    return False


def _find_naf_block_prefixes(state_dict: Dict[str, Tensor]) -> List[str]:
    suffix = ".beta"
    out: List[str] = []
    for key in state_dict:
        if not key.endswith(suffix):
            continue
        base = key[: -len(suffix)]
        if (
            f"{base}.gamma" in state_dict
            and f"{base}.dwconv.0.weight" in state_dict
            and f"{base}.FFN.0.weight" in state_dict
        ):
            out.append(base)
    return out


def _find_naf_transform_prefixes(state_dict: Dict[str, Tensor]) -> List[str]:
    suffix = ".in_conv.weight"
    out: List[str] = []
    for key in state_dict:
        if not key.endswith(suffix):
            continue
        base = key[: -len(suffix)]
        if (
            f"{base}.out_conv.weight" in state_dict
            and f"{base}.blocks.0.beta" in state_dict
        ):
            out.append(base)
    return out


def _strip_prefix(key: str, prefix: str) -> Optional[str]:
    return key[len(prefix):] if key.startswith(prefix) else None


def _rename_with_table(
    key: str,
    base_prefixes: Sequence[str],
    rename_map: Dict[str, str],
) -> str:
    for base in base_prefixes:
        head = base + "."
        rest = _strip_prefix(key, head)
        if rest is None:
            continue
        for old, new in rename_map.items():
            inner = _strip_prefix(rest, old)
            if inner is not None:
                return head + new + inner
        return key
    return key


def convert_upstream_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Rename upstream ``LICAutoencoder`` keys to compressai layout.

    Renames cover four scopes, applied in order so that nested renames
    compose cleanly:

    1. ``NAFBlock`` interior: ``dwconv``/``sca``/``FFN``/``conv1`` →
       ``pointwise_depthwise``/``channel_attention``/``feed_forward``/``project``.
       NAFBlock locations are detected by the presence of a ``.beta`` /
       ``.gamma`` / ``.dwconv.0.weight`` triple at the same scope, so the same
       rename safely runs on NAFBlocks inside ``g_a`` / ``g_s`` / aux module.
    2. ``NAFTransform`` interior: ``in_conv`` / ``out_conv`` →
       ``input_projection`` / ``output_projection``. Detected by sibling
       ``.in_conv.weight`` / ``.out_conv.weight`` / ``.blocks.0.beta`` triple.
    3. Top-level: ``mean_NAF_transforms`` → ``mean_support_transforms``,
       ``scale_NAF_transforms`` → ``scale_support_transforms``,
       ``aux_entropymodel`` → ``aux_entropy_model``.
    4. ``ResidualBottleneckBlock`` (``g_a`` / ``g_s`` interior) needs no
       renames: compressai's ``conv1`` / ``conv2`` / ``conv3`` / ``skip``
       names match upstream verbatim.
    """
    naf_blocks = _find_naf_block_prefixes(state_dict)
    naf_transforms = _find_naf_transform_prefixes(state_dict)

    converted: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        new_key = _rename_with_table(key, naf_blocks, _NAF_BLOCK_RENAMES)
        new_key = _rename_with_table(new_key, naf_transforms, _NAF_TRANSFORM_RENAMES)
        for old, new in _TOPLEVEL_RENAMES.items():
            if new_key.startswith(old):
                new_key = new + new_key[len(old):]
                break
        for old, new in _NAMED_PART_RENAMES.items():
            new_key = new_key.replace(old, new)
        converted[new_key] = value
    return converted


def _infer_config_from_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, object]:
    """Recover constructor kwargs from a compressai-layout CCA state dict."""
    encoder_dims = (
        state_dict["g_a.down.0.weight"].size(0),
        state_dict["g_a.down.1.weight"].size(0),
        state_dict["g_a.down.2.weight"].size(0),
    )
    latent_channels = state_dict["g_a.down.3.weight"].size(0)
    hyper_channels = state_dict["h_a.4.weight"].size(0)

    encoder_layers = []
    for stage in range(3):
        index = 0
        while f"g_a.blocks.{stage}.{index}.beta" in state_dict or _has_resblock(state_dict, stage, index):
            index += 1
        encoder_layers.append(index - 3)

    cc_keys = [
        key
        for key in state_dict
        if key.startswith("mean_cc_transforms.") and key.endswith(".4.weight")
    ]
    cc_keys.sort(key=lambda key: int(key.split(".")[1]))
    if not cc_keys:
        raise RuntimeError("state dict does not contain mean_cc_transforms heads")
    slice_sizes = [int(state_dict[key].size(0)) for key in cc_keys]

    em_hidden_channels = int(
        state_dict["mean_support_transforms.0.input_projection.weight"].size(0)
    )

    em_num_layers = 0
    while f"mean_support_transforms.0.blocks.{em_num_layers}.beta" in state_dict:
        em_num_layers += 1

    cca_training = any(key.startswith("aux_entropy_model.") for key in state_dict)

    return {
        "latent_channels": int(latent_channels),
        "hyper_channels": int(hyper_channels),
        "slice_proportions": tuple(slice_sizes),
        "encoder_dims": tuple(int(value) for value in encoder_dims),
        "encoder_layers": tuple(int(value) for value in encoder_layers),
        "em_hidden_channels": em_hidden_channels,
        "em_num_layers": em_num_layers,
        "cca_training": cca_training,
    }


def _has_resblock(state_dict: Dict[str, Tensor], stage: int, sub_index: int) -> bool:
    return f"g_a.blocks.{stage}.{sub_index}.conv2.weight" in state_dict and (
        f"g_a.blocks.{stage}.{sub_index}.beta" not in state_dict
    )
