"""VSS backbone factory used by MambaIC / MambaVC.

Builds analysis/synthesis/hyper transforms as ``nn.Sequential`` stacks of
``VSSBlock`` mixed with stride convolutions. Splits drop-path budget evenly
across stages following the VMamba reference convention.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from compressai.models.utils import conv, deconv

from .ssm import VSSBlock

__all__ = [
    "build_vss_backbone",
    "build_vss_context_stage",
]


def _partition_drop_paths(
    drop_paths: Sequence[float],
    depths: Sequence[int],
) -> List[List[float]]:
    offset = 0
    partitions: List[List[float]] = []
    for depth in depths:
        partitions.append(
            [float(value) for value in drop_paths[offset : offset + depth]]
        )
        offset += depth
    return partitions


def _normalize_vss_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    aliases = {
        "ssm_d_state": "d_state",
        "ssm_conv": "d_conv",
        "ssm_act_layer": "act_layer",
        "ssm_conv_bias": "conv_bias",
        "ssm_drop_rate": "dropout",
        "ssm_init": "initialize",
    }
    normalized = dict(kwargs)
    for source, target in aliases.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized.pop(source)
    return normalized


def _make_vss_stage(
    depth: int,
    hidden_dim: int,
    drop_paths: Sequence[float],
    tail: nn.Module,
    *,
    vss_kwargs: Dict[str, Any],
) -> List[nn.Module]:
    if len(drop_paths) != depth:
        raise ValueError("drop_paths must match stage depth")
    blocks = [
        VSSBlock(
            hidden_dim=hidden_dim,
            drop_path=float(drop_paths[index]),
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            **vss_kwargs,
        )
        for index in range(depth)
    ]
    return [*blocks, tail]


def build_vss_backbone(
    depths: Sequence[int],
    drop_path_rate: float,
    N: int,
    M: int,
    hyper_channels: int = 192,
    **vss_kwargs: Any,
) -> Tuple[nn.Sequential, nn.Sequential, nn.Sequential, nn.Sequential, nn.Sequential]:
    depths = tuple(int(depth) for depth in depths)
    if len(depths) != 4:
        raise ValueError("depths must contain four stage depths")
    if any(depth < 0 for depth in depths):
        raise ValueError("depths must be non-negative")

    drop_paths = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
    encoder_drops = _partition_drop_paths(drop_paths, depths)
    decoder_drops = _partition_drop_paths(drop_paths, list(reversed(depths)))
    vss_kwargs = _normalize_vss_kwargs(dict(vss_kwargs))

    g_a = nn.Sequential(
        conv(3, 2 * N, kernel_size=5, stride=2),
        *_make_vss_stage(
            depths[0],
            2 * N,
            encoder_drops[0],
            conv(2 * N, 2 * N, kernel_size=3, stride=2),
            vss_kwargs=vss_kwargs,
        ),
        *_make_vss_stage(
            depths[1],
            2 * N,
            encoder_drops[1],
            conv(2 * N, 2 * N, kernel_size=3, stride=2),
            vss_kwargs=vss_kwargs,
        ),
        *_make_vss_stage(
            depths[2],
            2 * N,
            encoder_drops[2],
            conv(2 * N, M, kernel_size=3, stride=2),
            vss_kwargs=vss_kwargs,
        ),
    )
    h_a = nn.Sequential(
        conv(M, 2 * N, kernel_size=3, stride=2),
        *_make_vss_stage(
            depths[3],
            2 * N,
            encoder_drops[3],
            conv(2 * N, hyper_channels, kernel_size=3, stride=2),
            vss_kwargs=vss_kwargs,
        ),
    )
    h_mean_s = nn.Sequential(
        deconv(hyper_channels, 2 * N, kernel_size=3, stride=2),
        *_make_vss_stage(
            depths[3],
            2 * N,
            decoder_drops[0],
            deconv(2 * N, M, kernel_size=3, stride=2),
            vss_kwargs=vss_kwargs,
        ),
    )
    h_scale_s = nn.Sequential(
        deconv(hyper_channels, 2 * N, kernel_size=3, stride=2),
        *_make_vss_stage(
            depths[3],
            2 * N,
            decoder_drops[0],
            deconv(2 * N, M, kernel_size=3, stride=2),
            vss_kwargs=vss_kwargs,
        ),
    )
    g_s = nn.Sequential(
        deconv(M, 2 * N, kernel_size=3, stride=2),
        *_make_vss_stage(
            depths[2],
            2 * N,
            decoder_drops[1],
            deconv(2 * N, 2 * N, kernel_size=3, stride=2),
            vss_kwargs=vss_kwargs,
        ),
        *_make_vss_stage(
            depths[1],
            2 * N,
            decoder_drops[2],
            deconv(2 * N, 2 * N, kernel_size=3, stride=2),
            vss_kwargs=vss_kwargs,
        ),
        *_make_vss_stage(
            depths[0],
            2 * N,
            decoder_drops[3],
            deconv(2 * N, 3, kernel_size=5, stride=2),
            vss_kwargs=vss_kwargs,
        ),
    )
    return g_a, g_s, h_a, h_mean_s, h_scale_s


def build_vss_context_stage(
    in_channels: int,
    out_channels: int,
    depth: int,
    drop_paths: Sequence[float],
    **vss_kwargs: Any,
) -> nn.Sequential:
    return nn.Sequential(
        *_make_vss_stage(
            depth,
            in_channels,
            drop_paths,
            conv(in_channels, out_channels, kernel_size=3, stride=1),
            vss_kwargs=_normalize_vss_kwargs(dict(vss_kwargs)),
        )
    )
