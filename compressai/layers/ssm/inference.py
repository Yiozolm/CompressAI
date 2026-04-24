"""State-dict introspection helpers for VSS-based models (MambaIC / MambaVC)."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from torch import Tensor

__all__ = [
    "infer_vss_block_kwargs",
    "infer_vss_depths",
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
