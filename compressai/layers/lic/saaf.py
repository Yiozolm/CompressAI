from __future__ import annotations

import math

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.layers import DropPath
from torch import Tensor

from .blocks import OLP, ResidualBottleneckBlock
from .dcae import ConvolutionalGLU, Scale, _pad_to_window_multiple, conv

__all__ = [
    "AdaptiveFrequencyBlock",
    "CrossSparseWindowAttention",
    "DenoisingAsRegularizer",
    "InverseAdaptiveFrequencyBlock",
    "SpatialAttentionBlock",
    "SpatialAttentionLayer",
]


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class AdaptiveFrequencyBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.olp = OLP(in_dim, out_dim)
        mid_dim = max(in_dim // 4, 4)
        self.freq_attn = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1),
            nn.GELU(),
            nn.Conv2d(mid_dim, 4, 1),
            nn.Softmax(dim=1),
        )
        self.freq_weights = nn.Parameter(torch.tensor([1.0, 0.8, 0.8, 0.6]))

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, _, height, width = input_tensor.shape
        frequency_attention = self.freq_attn(input_tensor)
        frequency_weights = torch.exp(self.freq_weights).view(1, 4, 1, 1)
        output = input_tensor.unsqueeze(1) * frequency_attention.unsqueeze(2)
        output = output * frequency_weights.unsqueeze(2)
        output = output.sum(dim=1)
        output = output.flatten(2).permute(0, 2, 1)
        output = self.olp(output)
        return output.permute(0, 2, 1).view(batch_size, -1, height, width)


class InverseAdaptiveFrequencyBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.olp = OLP(in_dim, out_dim)
        mid_dim = max(in_dim // 4, 4)
        self.freq_attn = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1),
            nn.GELU(),
            nn.Conv2d(mid_dim, 4, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, _, height, width = input_tensor.shape
        frequency_weights = self.freq_attn(input_tensor)
        output = input_tensor.flatten(2).permute(0, 2, 1)
        output = self.olp(output)
        output = output.permute(0, 2, 1).view(batch_size, -1, height, width)
        enhanced = output * frequency_weights.mean(dim=1, keepdim=True)
        return output + 0.1 * enhanced


class DenoisingAsRegularizer(nn.Module):
    def __init__(self, latent_dim: int = 320, hyper_channels: int = 192) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.noise_predictor = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(_group_count(latent_dim), latent_dim),
            nn.SiLU(),
            ResidualBottleneckBlock(latent_dim, latent_dim),
            ResidualBottleneckBlock(latent_dim, latent_dim),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(_group_count(latent_dim), latent_dim),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, 1),
        )
        condition_channels = max(latent_dim * 4 // 5, 4)
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(hyper_channels, condition_channels, 1),
            nn.GroupNorm(_group_count(condition_channels), condition_channels),
            nn.GELU(),
            nn.Conv2d(condition_channels, latent_dim, 3, padding=1),
            nn.Dropout(0.1),
            nn.GELU(),
        )

    def forward(self, latent: Tensor, hyper_latent: Tensor) -> Tensor:
        batch_size, channels, height, width = latent.size()
        condition = self.condition_encoder(hyper_latent)
        condition = F.interpolate(condition, size=(height, width), mode="bilinear", align_corners=False)
        time = torch.rand(batch_size, 1, device=latent.device, dtype=latent.dtype)
        noise = torch.randn_like(latent)
        noisy_latent = latent + noise * time.view(batch_size, 1, 1, 1)
        time_embedding = self.time_embed(time).view(batch_size, channels, 1, 1)
        prediction = self.noise_predictor(noisy_latent + time_embedding + condition)
        return F.mse_loss(prediction, noise)


class CrossSparseWindowAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        num_global_tokens: int = 2,
    ) -> None:
        super().__init__()
        if input_dim % head_dim != 0:
            raise ValueError("input_dim must be divisible by head_dim")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.embedding_layer = nn.Linear(input_dim, 3 * input_dim, bias=True)
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, input_dim))
        nn.init.trunc_normal_(self.global_tokens, std=0.02)
        self.global_kv = nn.Linear(input_dim, input_dim * 2, bias=False)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        self.linear = nn.Linear(input_dim, output_dim)
        self.register_buffer("global_alpha", torch.tensor(0.25))

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, height, width, channels = input_tensor.shape
        window_size = self.window_size
        height_windows = height // window_size
        width_windows = width // window_size
        output = input_tensor.view(
            batch_size,
            height_windows,
            window_size,
            width_windows,
            window_size,
            channels,
        )
        output = output.permute(0, 1, 3, 2, 4, 5).contiguous()
        output = output.view(batch_size * height_windows * width_windows, window_size * window_size, channels)
        num_windows = height_windows * width_windows

        qkv = self.embedding_layer(output).reshape(
            batch_size * num_windows,
            window_size * window_size,
            3,
            self.n_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        query, key, value = qkv[0], qkv[1], qkv[2]

        similarity = torch.einsum("bhpc,bhqc->bhpq", query * self.scale, key)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(window_size * window_size, window_size * window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        similarity = similarity + relative_position_bias.unsqueeze(0)
        probabilities = similarity.softmax(dim=-1)
        output_local = torch.einsum("bhij,bhjc->bhic", probabilities, value)

        global_tokens = self.global_tokens.expand(batch_size * num_windows, -1, -1)
        global_tokens = global_tokens + output.mean(dim=1, keepdim=True)
        global_kv = self.global_kv(global_tokens).reshape(
            batch_size * num_windows,
            self.num_global_tokens,
            2,
            self.n_heads,
            self.head_dim,
        )
        global_kv = global_kv.permute(2, 0, 3, 1, 4).contiguous()
        key_global, value_global = global_kv[0], global_kv[1]
        similarity_global = torch.einsum("bhpc,bhgc->bhpg", query * self.scale, key_global)
        probabilities_global = similarity_global.softmax(dim=-1)
        output_global = torch.einsum("bhpg,bhgc->bhpc", probabilities_global, value_global)

        output = (1 - self.global_alpha) * output_local + self.global_alpha * output_global
        output = output.transpose(1, 2).reshape(batch_size * num_windows, window_size * window_size, channels)
        output = self.linear(output)
        output = output.view(batch_size, height_windows, width_windows, window_size, window_size, channels)
        output = output.permute(0, 1, 3, 2, 4, 5).contiguous()
        return output.view(batch_size, height, width, channels)


class SpatialAttentionLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        input_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        del output_dim, input_resolution
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = CrossSparseWindowAttention(input_dim, input_dim, head_dim, window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = ConvolutionalGLU(input_dim, input_dim * 4)
        self.res_scale_1 = Scale(input_dim, init_value=1.0)
        self.res_scale_2 = Scale(input_dim, init_value=1.0)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.res_scale_1(input_tensor) + self.drop_path(self.msa(self.ln1(input_tensor)))
        return self.res_scale_2(output) + self.drop_path(self.mlp(self.ln2(output)))


class SpatialAttentionBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        block: type[nn.Module] = SpatialAttentionLayer,
        block_num: int = 2,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__()
        self.layers = nn.ModuleList(
            block(input_dim, input_dim, head_dim, window_size, drop_path)
            for _ in range(block_num)
        )
        self.block_num = block_num
        self.conv = conv(input_dim, output_dim, 3, 1)
        self.window_size = window_size

    def forward(self, input_tensor: Tensor) -> Tensor:
        output, pad_height, pad_width = _pad_to_window_multiple(input_tensor, self.window_size)
        output = rearrange(output, "b c h w -> b h w c")
        for layer in self.layers:
            output = layer(output)
        output = rearrange(output, "b h w c -> b c h w")
        output = self.conv(output) + F.pad(input_tensor, (0, pad_width, 0, pad_height))
        if pad_height > 0 or pad_width > 0:
            output = output[:, :, : input_tensor.size(2), : input_tensor.size(3)]
        return output.contiguous()
