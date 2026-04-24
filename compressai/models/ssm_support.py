from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.layers import conv, deconv
from compressai.layers.ssm import VSSBlock

__all__ = [
    "build_vss_backbone",
    "build_vss_context_stage",
    "infer_vss_block_kwargs",
    "infer_vss_depths",
    "lrp_support_channels",
    "make_entropy_transform",
    "slice_support_channels",
]


def _group_consecutive(indices: Sequence[int]) -> List[List[int]]:
    groups: List[List[int]] = []
    for index in sorted(indices):
        if not groups or index != groups[-1][-1] + 1:
            groups.append([index])
            continue
        groups[-1].append(index)
    return groups


def _infer_stage_groups(state_dict: Dict[str, Tensor], prefix: str) -> List[List[int]]:
    indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith(f"{prefix}.") and key.endswith(".norm.weight")
    }
    return _group_consecutive(indices)


def _partition_drop_paths(
    drop_paths: Sequence[float],
    depths: Sequence[int],
) -> List[List[float]]:
    offset = 0
    partitions: List[List[float]] = []
    for depth in depths:
        partitions.append([float(value) for value in drop_paths[offset : offset + depth]])
        offset += depth
    return partitions


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


def slice_support_channels(
    latent_channels: int,
    slice_channels: int,
    index: int,
    max_support_slices: int,
) -> int:
    if max_support_slices < 0:
        return latent_channels + slice_channels * index
    return latent_channels + slice_channels * min(index, max_support_slices)


def lrp_support_channels(
    latent_channels: int,
    slice_channels: int,
    index: int,
    max_support_slices: int,
) -> int:
    if max_support_slices < 0:
        return latent_channels + slice_channels * (index + 1)
    return latent_channels + slice_channels * min(index + 1, max_support_slices + 1)


def make_entropy_transform(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        conv(in_channels, 224, stride=1, kernel_size=3),
        nn.GELU(),
        conv(224, 128, stride=1, kernel_size=3),
        nn.GELU(),
        conv(128, out_channels, stride=1, kernel_size=3),
    )


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
    vss_kwargs = dict(vss_kwargs)

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
            vss_kwargs=dict(vss_kwargs),
        )
    )


def infer_vss_depths(state_dict: Dict[str, Tensor]) -> Tuple[int, int, int, int]:
    g_a_groups = _infer_stage_groups(state_dict, "g_a")
    h_a_groups = _infer_stage_groups(state_dict, "h_a")
    if len(g_a_groups) != 3 or len(h_a_groups) != 1:
        raise ValueError("Unable to infer VSS stage depths from state_dict")
    return (
        len(g_a_groups[0]),
        len(g_a_groups[1]),
        len(g_a_groups[2]),
        len(h_a_groups[0]),
    )


def infer_vss_block_kwargs(state_dict: Dict[str, Tensor]) -> Dict[str, Any]:
    for prefix in ("g_a", "h_a", "g_s", "h_mean_s"):
        groups = _infer_stage_groups(state_dict, prefix)
        if not groups or not groups[0]:
            continue
        index = groups[0][0]
        norm_key = f"{prefix}.{index}.norm.weight"
        a_logs_key = f"{prefix}.{index}.op.A_logs"
        out_proj_key = f"{prefix}.{index}.op.out_proj.weight"
        conv_key = f"{prefix}.{index}.op.conv2d.weight"
        hidden_dim = state_dict[norm_key].numel()
        d_inner = state_dict[out_proj_key].size(1)
        return {
            "ssm_d_state": state_dict[a_logs_key].size(1),
            "ssm_ratio": d_inner / hidden_dim,
            "ssm_conv": state_dict[conv_key].size(2) if conv_key in state_dict else 1,
        }
    return {"ssm_d_state": 16, "ssm_ratio": 2.0, "ssm_conv": 3}
