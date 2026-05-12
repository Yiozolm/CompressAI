# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0.

"""Utility helpers for the NVTC model port."""

from __future__ import annotations

import math
import re

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch import Tensor


def sum_tensors(values: Sequence[Tensor], like: Tensor) -> Tensor:
    total = like.new_zeros(())
    for value in values:
        total = total + value
    return total


def as_tuple(values: Sequence[int], length: int, name: str) -> Tuple[int, ...]:
    values = tuple(int(v) for v in values)
    if len(values) != length:
        raise ValueError(f"{name} length must be {length}, got {len(values)}")
    return values


def padding_factor(
    downscale_factor: Sequence[int],
    block_size: Sequence[int],
) -> int:
    factors = [int(d) * int(b) for d, b in zip(downscale_factor, block_size)]
    return math.lcm(*factors)


def extract_state_dict(
    checkpoint: Mapping[str, Any],
) -> Tuple[Mapping[str, Tensor], Optional[float]]:
    lmbda = checkpoint.get("lmbda")
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        return checkpoint["state_dict"], float(lmbda) if lmbda is not None else None
    return checkpoint, float(lmbda) if isinstance(lmbda, (float, int)) else None


def convert_upstream_state_dict(state_dict: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    """Strip common DataParallel/Lightning prefixes from NVTC checkpoints."""

    converted: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, Tensor):
            continue
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model.", "net."):
                if new_key.startswith(prefix):
                    new_key = new_key.removeprefix(prefix)
                    changed = True
        converted[new_key] = value
    return converted


def infer_config_from_state_dict(state_dict: Mapping[str, Tensor]) -> Dict[str, Any]:
    layers_by_stage: Dict[int, List[int]] = {}
    codebook_pattern = re.compile(r"^quantizer\.(\d+)\.(\d+)\.codebook$")
    for key in state_dict:
        match = codebook_pattern.match(key)
        if match is None:
            continue
        stage, layer = (int(match.group(1)), int(match.group(2)))
        layers_by_stage.setdefault(stage, []).append(layer)

    if not layers_by_stage:
        raise ValueError("Could not infer NVTC config: missing quantizer codebooks")

    n_stage = max(layers_by_stage) + 1
    n_layer = tuple(max(layers_by_stage[stage]) + 1 for stage in range(n_stage))
    vt_dim: List[int] = []
    vt_nunit: List[int] = []
    block_size: List[int] = []
    cb_dim: List[int] = []
    cb_size: List[int] = []
    param_dim: List[int] = []
    param_nlevel: List[int] = []
    downscale_factor: List[int] = []

    cumulative_downscale = 1
    previous_channels = 3
    for stage in range(n_stage):
        codebook = state_dict[f"quantizer.{stage}.0.codebook"]
        ncb, cb_size_stage, cb_dim_stage = codebook.shape
        block = math.isqrt(int(ncb))
        if block * block != int(ncb):
            raise ValueError(f"Cannot infer block size from codebook ncb={ncb}")
        block_size.append(block)
        cb_size.append(int(cb_size_stage))
        cb_dim.append(int(cb_dim_stage))

        param_table = state_dict[f"quantizer.{stage}.0.entropy_model.param_table"]
        param_nlevel.append(int(param_table.size(0)))
        param_dim.append(int(param_table.size(1)))

        projection = state_dict[f"projection_in.{stage}.0.weight"]
        vt_dim_stage = int(projection.size(1))
        vt_dim.append(vt_dim_stage)

        unit_pattern = re.compile(
            rf"^vt_encoder\.{stage}\.0\.(\d+)\.intra_transform\.fc1\.weight$"
        )
        units = [
            int(match.group(1))
            for key in state_dict
            if (match := unit_pattern.match(key)) is not None
        ]
        vt_nunit.append(max(units) + 1 if units else 0)

        down_conv = state_dict[f"downscaling.{stage}.1.weight"]
        factor_sq = int(down_conv.size(1)) // previous_channels
        resolution_factor = math.isqrt(factor_sq)
        if resolution_factor * resolution_factor != factor_sq:
            raise ValueError(
                f"Cannot infer downscale factor for stage {stage}: {factor_sq}"
            )
        cumulative_downscale *= resolution_factor
        downscale_factor.append(cumulative_downscale)
        previous_channels = vt_dim_stage

    return {
        "n_stage": n_stage,
        "n_layer": n_layer,
        "downscale_factor": tuple(downscale_factor),
        "vt_dim": tuple(vt_dim),
        "vt_nunit": tuple(vt_nunit),
        "block_size": tuple(block_size),
        "cb_dim": tuple(cb_dim),
        "cb_size": tuple(cb_size),
        "param_dim": tuple(param_dim),
        "param_nlevel": tuple(param_nlevel),
    }
