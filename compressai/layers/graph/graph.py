from __future__ import annotations

from typing import Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath
from torch import Tensor

from .graph_ops import cosine_similarity, global_sampling, local_sampling

__all__ = [
    "GDFN",
    "GAL",
    "GraphAggregator",
    "GraphAttentionLayer",
    "GraphDepthwiseFeedForward",
    "IPGGrapher",
]


class GraphAggregator(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        unfold_dict: dict,
        bias: bool = True,
        inner_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.group_size = window_size
        self.num_heads = num_heads
        self.inner_dim = inner_dim or dim
        self.unfold_dict = unfold_dict
        self.sample_size = unfold_dict["kernel_size"]
        self.graph_switch = True

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.proj_group = nn.Linear(dim, self.inner_dim, bias=bias)
        self.proj_sample = nn.Linear(dim, self.inner_dim * 2, bias=bias)
        self.proj = nn.Linear(self.inner_dim, dim)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        relative_coords_table = self._build_relative_coords_table()
        self.register_buffer("relative_coords_table", relative_coords_table)
        self.register_buffer("relative_position_index", self._get_rel_pos_index())
        self.relative_position_bias_table: Optional[Tensor] = None

    def _build_relative_coords_table(self) -> Tensor:
        relative_coords_h = torch.arange(
            -(self.sample_size[0] - 1),
            self.group_size[0],
            dtype=torch.float32,
        )
        relative_coords_w = torch.arange(
            -(self.sample_size[1] - 1),
            self.group_size[1],
            dtype=torch.float32,
        )
        table = torch.stack(
            torch.meshgrid(relative_coords_h, relative_coords_w, indexing="ij")
        )
        table = table.permute(1, 2, 0).contiguous().unsqueeze(0)
        table[:, :, :, 0] /= self.group_size[0] - 1
        table[:, :, :, 1] /= self.group_size[1] - 1
        table = table * 8
        return torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / 3

    def _get_rel_pos_index(self) -> Tensor:
        coords_grid = torch.stack(
            torch.meshgrid(
                torch.arange(self.group_size[0]),
                torch.arange(self.group_size[1]),
                indexing="ij",
            )
        )
        coords_sample = torch.stack(
            torch.meshgrid(
                torch.arange(self.sample_size[0]),
                torch.arange(self.sample_size[1]),
                indexing="ij",
            )
        )
        coords_grid_flatten = torch.flatten(coords_grid, 1)
        coords_sample_flatten = torch.flatten(coords_sample, 1)
        relative_coords = (
            coords_sample_flatten[:, None, :] - coords_grid_flatten[:, :, None]
        )
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.group_size[0] - self.sample_size[0] + 1
        relative_coords[:, :, 0] *= self.group_size[1] + self.sample_size[1] - 1
        relative_coords[:, :, 1] += self.group_size[1] - self.sample_size[1] + 1
        return relative_coords.sum(-1)

    def _rel_pos_bias(self) -> Tensor:
        if self.training:
            self.relative_position_bias_table = None
        if not self.training and self.relative_position_bias_table is not None:
            bias_table = self.relative_position_bias_table
        else:
            bias_table = self.cpb_mlp(self.relative_coords_table).view(
                -1,
                self.num_heads,
            )
        if not self.training and self.relative_position_bias_table is None:
            self.relative_position_bias_table = bias_table

        bias = bias_table[self.relative_position_index.view(-1)].view(
            self.group_size[0] * self.group_size[1],
            self.sample_size[0] * self.sample_size[1],
            -1,
        )
        bias = bias.permute(2, 0, 1).contiguous()
        return (16 * torch.sigmoid(bias)).unsqueeze(0)

    def _get_correlation(
        self,
        query: Tensor,
        key: Tensor,
        graph: Optional[Tensor],
    ) -> Tensor:
        scale = torch.exp(torch.clamp(self.logit_scale, max=4.6052))
        if self.graph_switch and graph is not None:
            same_query = query.size(-2) == graph.size(-2)
            same_key = key.size(-2) == graph.size(-1)
            if not same_query or not same_key:
                raise ValueError("Graph shape is incompatible with query/key.")
        similarity = cosine_similarity(query, key, graph if self.graph_switch else None)
        return F.softmax(similarity * scale + self._rel_pos_bias(), dim=-1)

    def forward(
        self,
        input_tensor: Tensor,
        graph: Optional[Tensor] = None,
        sampling_method: int = 0,
    ) -> Tensor:
        if sampling_method == 0:
            grouped = local_sampling(
                input_tensor,
                group_size=self.group_size,
                unfold_dict=None,
                output=0,
                tensor_format="bhwc",
            )
            sampled_input = local_sampling(
                self.proj_sample(input_tensor),
                group_size=self.group_size,
                unfold_dict=self.unfold_dict,
                output=1,
                tensor_format="bhwc",
            )
        else:
            grouped = global_sampling(
                input_tensor,
                group_size=self.group_size,
                sample_size=self.sample_size,
                output=0,
                tensor_format="bhwc",
            )
            sampled_input = global_sampling(
                self.proj_sample(input_tensor),
                group_size=self.group_size,
                sample_size=self.sample_size,
                output=1,
                tensor_format="bhwc",
            )

        batch_windows, num_tokens, _ = grouped.shape
        query = einops.rearrange(
            self.proj_group(grouped),
            "b n (h c) -> b h n c",
            b=batch_windows,
            n=num_tokens,
            h=self.num_heads,
        )
        key, value = einops.rearrange(
            sampled_input,
            "b n (two h c) -> two b h n c",
            two=2,
            h=self.num_heads,
            c=self.inner_dim // self.num_heads,
        )
        correlation = self._get_correlation(query, key, graph)
        output = (correlation @ value).transpose(1, 2)
        output = output.reshape(batch_windows, num_tokens, self.inner_dim)
        return self.proj(output)


class GraphDepthwiseFeedForward(nn.Module):
    def __init__(self, channels: int, expansion_factor: float) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.channels = channels
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, 1, bias=False)
        self.conv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, 1, bias=False)

    def forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        height, width = x_size
        output = input_tensor.transpose(1, 2).view(
            input_tensor.shape[0],
            self.channels,
            height,
            width,
        )
        gate, value = self.conv(self.project_in(output)).chunk(2, dim=1)
        output = self.project_out(F.gelu(gate) * value)
        return output.flatten(2).transpose(1, 2).contiguous()


class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        unfold_dict: dict,
        window_size: int = 7,
        sampling_method: int = 0,
        expansion_factor: float = 4.0,
        bias: bool = True,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        inner_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.sampling_method = sampling_method
        self.norm1 = norm_layer(dim)
        self.grapher = GraphAggregator(
            dim,
            window_size=(window_size, window_size),
            num_heads=num_heads,
            bias=bias,
            unfold_dict=unfold_dict,
            inner_dim=inner_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = GraphDepthwiseFeedForward(dim, expansion_factor=expansion_factor)

    def forward(
        self,
        input_tensor: Tensor,
        x_size: Tuple[int, int],
        graph: Tuple[Optional[Tensor], Optional[Tensor]],
    ) -> Tensor:
        height, width = x_size
        batch_size, _, channels = input_tensor.shape
        graph_index = 0 if self.sampling_method == 0 else 1
        output = input_tensor.view(batch_size, height, width, channels)
        output = self.grapher(
            self.norm1(output),
            graph=graph[graph_index],
            sampling_method=self.sampling_method,
        )

        if self.sampling_method:
            output = einops.rearrange(
                output,
                "(b nh nw) (sh sw) c -> b (sh nh sw nw) c",
                nh=height // self.window_size,
                nw=width // self.window_size,
                sh=self.window_size,
                sw=self.window_size,
            )
        else:
            output = einops.rearrange(
                output,
                "(b nh nw) (sh sw) c -> b (nh sh nw sw) c",
                nh=height // self.window_size,
                nw=width // self.window_size,
                sh=self.window_size,
                sw=self.window_size,
            )
        output = input_tensor + self.drop_path(output)
        return output + self.mlp(self.norm2(output), x_size)


IPGGrapher = GraphAggregator
GDFN = GraphDepthwiseFeedForward
GAL = GraphAttentionLayer
