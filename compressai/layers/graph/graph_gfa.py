from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import einops
import torch
import torch.nn as nn

from torch import Tensor

from .graph import GraphAttentionLayer
from .graph_ops import (
    compute_sobel_gradients,
    cosine_similarity,
    global_sampling,
    local_sampling,
)

__all__ = [
    "FeatureReshape",
    "FeatureRestore",
    "GFA",
    "GraphLayerStack",
    "MGB",
]


class GraphLayerStack(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        stages: Sequence[str],
        unfold_dict: dict,
        mlp_ratio: float = 4.0,
        bias: bool = True,
        drop_path: Union[float, Sequence[float]] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        inner_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        blocks = []
        for block_index in range(depth):
            stage = stages[block_index]
            if stage not in {"GN", "GS"}:
                raise ValueError(f"Unsupported graph stage: {stage}")
            block_drop_path = (
                drop_path[block_index]
                if isinstance(drop_path, Sequence)
                else drop_path
            )
            blocks.append(
                GraphAttentionLayer(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    sampling_method=0 if stage == "GN" else 1,
                    expansion_factor=mlp_ratio,
                    bias=bias,
                    drop_path=float(block_drop_path),
                    norm_layer=norm_layer,
                    unfold_dict=unfold_dict,
                    inner_dim=inner_dim,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        input_tensor: Tensor,
        x_size: Tuple[int, int],
        graph: Tuple[Optional[Tensor], Optional[Tensor]],
    ) -> Tensor:
        output = input_tensor
        for block in self.blocks:
            output = block(output, x_size, graph)
        return output


class FeatureReshape(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        norm_layer: Optional[type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = input_tensor.flatten(2).transpose(1, 2)
        if self.norm is not None:
            output = self.norm(output)
        return output


class FeatureRestore(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        height, width = x_size
        output = input_tensor.transpose(1, 2)
        return output.view(input_tensor.shape[0], self.embed_dim, height, width)


class GFA(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        sample_size: int,
        graph_flags: bool,
        top_k: int,
        diff_scales: Optional[float],
        stages: Sequence[str],
        mlp_ratio: float = 4.0,
        bias: bool = True,
        drop_path: Union[float, Sequence[float]] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        inner_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.sample_size = sample_size
        padding_size = (sample_size - window_size) // 2
        self.unfold_dict = {
            "kernel_size": (sample_size, sample_size),
            "stride": (window_size, window_size),
            "padding": (padding_size, padding_size),
        }
        self.num_head = num_heads
        self.graph_flag = graph_flags
        self.top_k = top_k
        self.diff_scale = diff_scales
        self.graph_switch = True
        self.fast_graph = True
        self.tensors: Optional[Tuple[Tensor, Tensor]] = None
        self.tolerance = 5

        self.residual_group = GraphLayerStack(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            bias=bias,
            drop_path=drop_path,
            norm_layer=norm_layer,
            unfold_dict=self.unfold_dict,
            stages=stages,
            inner_dim=inner_dim,
        )
        self.patch_embed = FeatureReshape(embed_dim=dim)
        self.patch_unembed = FeatureRestore(embed_dim=dim)

    @torch.no_grad()
    def _calc_graph(
        self,
        input_tensor: Tensor,
        x_size: Tuple[int, int],
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if not self.graph_switch:
            return None, None

        if self.fast_graph and self.tensors is None:
            self.tensors = (
                torch.tensor(
                    [[0.5, 1.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 1.0]],
                    dtype=torch.float32,
                    device=input_tensor.device,
                ),
                torch.tensor(
                    [[0.5, 0.0, 1.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                    device=input_tensor.device,
                ),
            )

        diff_map = compute_sobel_gradients(input_tensor, shape=x_size)
        if self.diff_scale is not None and self.diff_scale != 0:
            mean = diff_map.mean(dim=(-2, -1), keepdim=True)
            diff_map = mean + (diff_map - mean) / self.diff_scale

        diff_local = einops.rearrange(
            diff_map,
            "b (nh wh) (nw ww) -> (b nh nw) (wh ww)",
            wh=self.window_size,
            ww=self.window_size,
        )
        diff_global = einops.rearrange(
            diff_map,
            "b (wh nh) (ww nw) -> (b nh nw) (wh ww)",
            wh=self.window_size,
            ww=self.window_size,
        )
        graph_local = self._calc_graph_for_sampling(
            input_tensor,
            x_size,
            0,
            diff_local,
        )
        graph_global = self._calc_graph_for_sampling(
            input_tensor,
            x_size,
            1,
            diff_global,
        )
        return graph_local, graph_global

    @torch.no_grad()
    def _calc_graph_for_sampling(
        self,
        input_tensor: Tensor,
        x_size: Tuple[int, int],
        sampling_method: int,
        diff_map: Tensor,
    ) -> Tensor:
        feature_map = einops.rearrange(
            input_tensor,
            "b (h w) c -> b c h w",
            h=x_size[0],
            w=x_size[1],
        )
        if sampling_method == 1:
            query_sample, key_sample = global_sampling(
                feature_map,
                group_size=self.window_size,
                sample_size=self.sample_size,
                output=2,
                tensor_format="bchw",
            )
        else:
            query_sample, key_sample = local_sampling(
                feature_map,
                group_size=self.window_size,
                unfold_dict=self.unfold_dict,
                output=2,
                tensor_format="bchw",
            )

        distance = cosine_similarity(query_sample.unsqueeze(1), key_sample.unsqueeze(1))
        distance = distance.squeeze(1)
        mask_count = diff_map / diff_map.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mask_count = mask_count * distance.size(1) * self.top_k
        mask_count = torch.clamp(mask_count, 1, distance.size(-1))

        min_bound = torch.min(distance, dim=-1, keepdim=True)[0]
        max_bound = torch.ones_like(min_bound)
        wall = distance.mean(dim=-1, keepdim=True)
        threshold_state = torch.cat([wall, min_bound, max_bound], dim=-1)
        if self.tensors is None:
            raise RuntimeError("Graph threshold tensors were not initialized.")

        for _ in range(self.tolerance):
            allocated = (distance > threshold_state[..., 0:1]).sum(dim=-1)
            threshold_state = torch.where(
                (allocated > mask_count).unsqueeze(-1),
                threshold_state @ self.tensors[0],
                threshold_state @ self.tensors[1],
            )
        return (distance > threshold_state[..., 0:1]).unsqueeze(1)

    def forward(
        self,
        input_tensor: Tensor,
        x_size: Tuple[int, int],
        prev_graph: Optional[Tuple[Optional[Tensor], Optional[Tensor]]] = None,
    ) -> Tensor:
        output = self.patch_embed(input_tensor)
        if self.graph_flag:
            graph = self._calc_graph(output, x_size)
        else:
            graph = prev_graph or (None, None)
        output = self.residual_group(output, x_size, graph)
        return self.patch_unembed(output, x_size)


MGB = GFA
