from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch.nn as nn

from torch import Tensor

from compressai.layers import ResidualBlockUpsample, ResidualBlockWithStride, conv3x3, subpel_conv3x3
from compressai.layers.lic import FATBlock

__all__ = [
    "FTICAnalysisTransform",
    "FTICHyperAnalysisTransform",
    "FTICHyperSynthesisTransform",
    "FTICSynthesisTransform",
    "infer_fm_window_size",
    "infer_ftic_num_heads",
    "infer_ftic_stage_depth",
    "infer_num_slices",
    "infer_tca_depth",
    "infer_tca_ratio",
    "infer_window_size",
]


def _make_stage(
    channels: int,
    depth: int,
    num_heads: int,
    window_size: int,
    fm_window_size: int,
    drop_paths: Sequence[float],
    tail: nn.Module,
) -> nn.Module:
    if len(drop_paths) != depth:
        raise ValueError("drop_paths must match stage depth")
    blocks = [
        FATBlock(
            dim=channels,
            num_heads=num_heads,
            window_size=window_size,
            fm_window_size=fm_window_size,
            drop_path=float(drop_paths[index]),
            attention_type="W" if index % 2 == 0 else "SW",
        )
        for index in range(depth)
    ]
    return nn.ModuleDict(
        {
            "blocks": nn.ModuleList(blocks),
            "tail": tail,
        }
    )


def _run_stage(stage: nn.ModuleDict, input_tensor: Tensor) -> Tensor:
    output = input_tensor
    for block in stage["blocks"]:
        output = block(output)
    return stage["tail"](output)


class FTICAnalysisTransform(nn.Module):
    def __init__(
        self,
        feature_dims: Tuple[int, int, int],
        M: int,
        config: Sequence[int],
        num_heads: Sequence[int],
        drop_paths: Sequence[Sequence[float]],
        *,
        window_size: int = 8,
        fm_window_size: int = 16,
    ) -> None:
        super().__init__()
        dim0, dim1, dim2 = feature_dims
        self.input_block = ResidualBlockWithStride(3, dim0, stride=2)
        self.stage1 = _make_stage(
            dim0,
            int(config[0]),
            int(num_heads[0]),
            window_size,
            fm_window_size,
            drop_paths[0],
            ResidualBlockWithStride(dim0, dim1, stride=2),
        )
        self.stage2 = _make_stage(
            dim1,
            int(config[1]),
            int(num_heads[1]),
            window_size,
            fm_window_size,
            drop_paths[1],
            ResidualBlockWithStride(dim1, dim2, stride=2),
        )
        self.stage3 = _make_stage(
            dim2,
            int(config[2]),
            int(num_heads[2]),
            window_size,
            fm_window_size,
            drop_paths[2],
            conv3x3(dim2, M, stride=2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.input_block(input_tensor)
        output = _run_stage(self.stage1, output)
        output = _run_stage(self.stage2, output)
        return _run_stage(self.stage3, output)


class FTICSynthesisTransform(nn.Module):
    def __init__(
        self,
        feature_dims: Tuple[int, int, int],
        M: int,
        config: Sequence[int],
        num_heads: Sequence[int],
        drop_paths: Sequence[Sequence[float]],
        *,
        window_size: int = 8,
        fm_window_size: int = 16,
    ) -> None:
        super().__init__()
        dim0, dim1, dim2 = feature_dims
        self.input_block = ResidualBlockUpsample(M, dim2, upsample=2)
        self.stage1 = _make_stage(
            dim2,
            int(config[3]),
            int(num_heads[3]),
            window_size,
            fm_window_size,
            drop_paths[3],
            ResidualBlockUpsample(dim2, dim1, upsample=2),
        )
        self.stage2 = _make_stage(
            dim1,
            int(config[4]),
            int(num_heads[4]),
            window_size,
            fm_window_size,
            drop_paths[4],
            ResidualBlockUpsample(dim1, dim0, upsample=2),
        )
        self.stage3 = _make_stage(
            dim0,
            int(config[5]),
            int(num_heads[5]),
            window_size,
            fm_window_size,
            drop_paths[5],
            subpel_conv3x3(dim0, 3, 2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.input_block(input_tensor)
        output = _run_stage(self.stage1, output)
        output = _run_stage(self.stage2, output)
        return _run_stage(self.stage3, output)


class FTICHyperAnalysisTransform(nn.Module):
    def __init__(
        self,
        M: int,
        hyper_hidden_channels: int,
        hyper_channels: int,
        depth: int,
        num_heads: int,
        *,
        window_size: int = 2,
        fm_window_size: int = 4,
    ) -> None:
        super().__init__()
        self.input_block = ResidualBlockWithStride(M, hyper_hidden_channels, stride=2)
        self.stage = _make_stage(
            hyper_hidden_channels,
            depth,
            num_heads,
            window_size,
            fm_window_size,
            [0.0] * depth,
            conv3x3(hyper_hidden_channels, hyper_channels, stride=2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.input_block(input_tensor)
        return _run_stage(self.stage, output)


class FTICHyperSynthesisTransform(nn.Module):
    def __init__(
        self,
        M: int,
        hyper_hidden_channels: int,
        hyper_channels: int,
        depth: int,
        num_heads: int,
        *,
        window_size: int = 2,
        fm_window_size: int = 4,
    ) -> None:
        super().__init__()
        self.input_block = ResidualBlockUpsample(
            hyper_channels,
            hyper_hidden_channels,
            upsample=2,
        )
        self.stage = _make_stage(
            hyper_hidden_channels,
            depth,
            num_heads,
            window_size,
            fm_window_size,
            [0.0] * depth,
            subpel_conv3x3(hyper_hidden_channels, M, 2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.input_block(input_tensor)
        return _run_stage(self.stage, output)


def infer_ftic_stage_depth(state_dict: Dict[str, Tensor], prefix: str) -> int:
    indices = {
        int(key.split(".")[3])
        for key in state_dict
        if key.startswith(prefix) and ".conv1.weight" in key
    }
    return len(indices)


def infer_ftic_num_heads(state_dict: Dict[str, Tensor], prefix: str) -> int:
    key = f"{prefix}.blocks.0.frequency_attention.branch_attentions.0.relative_position_bias_table"
    return state_dict[key].size(1) * 4


def infer_window_size(state_dict: Dict[str, Tensor], prefix: str) -> int:
    key = f"{prefix}.blocks.0.frequency_attention.branch_attentions.0.relative_position_bias_table"
    table_size = state_dict[key].size(0)
    return (int(round(table_size**0.5)) + 1) // 4


def infer_fm_window_size(state_dict: Dict[str, Tensor], prefix: str) -> int:
    key = f"{prefix}.blocks.0.frequency_attention.frequency_modulation.complex_weight"
    return state_dict[key].size(0)


def infer_num_slices(state_dict: Dict[str, Tensor], M: int) -> int:
    start_token_channels = state_dict["tca.tca.start_token_from_hyperprior.weight"].size(0)
    return M // start_token_channels


def infer_tca_depth(state_dict: Dict[str, Tensor]) -> int:
    indices = {
        int(key.split(".")[3])
        for key in state_dict
        if key.startswith("tca.tca.layers.") and ".q_proj.weight" in key
    }
    return len(indices)


def infer_tca_ratio(state_dict: Dict[str, Tensor], M: int) -> int:
    return state_dict["tca.tca.lift.weight"].size(0) // M
