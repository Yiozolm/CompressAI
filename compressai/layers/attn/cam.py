"""Causal Attention Module (CAM) for TIC.

Vendored from `Lu, Guo, Shi, Cao & Ma, "Transformer-based Image Compression"
<https://arxiv.org/abs/2111.06707>`_ (DCC 2022). The CAM replaces the
:class:`compressai.layers.MaskedConv2d` context model used by Minnen2018-style
joint autoregressive priors. It unfolds a 5×5 local window per pixel, masks
out the lower half (so only causal neighbours contribute), runs a single
masked window self-attention + MLP block, and projects back to the entropy
parameters channel count.

Module / parameter naming matches upstream verbatim (``norm1`` / ``qkv`` /
``norm2`` / ``mlp.fc1`` / ``mlp.fc2`` / ``proj``) so TIC checkpoints load
without translation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import Mlp
from torch import Tensor


__all__ = ["CausalAttentionModule"]


class CausalAttentionModule(nn.Module):
    """Causal multi-head self-attention over a 5×5 local window.

    Args:
        dim: Input channels (latent ``y`` width).
        out_dim: Output channels (typically ``2 * dim`` for mean+scale).
        block_len: Spatial extent of the local window (default 5; the
            mask is hard-coded for ``block_len == 5``).
        num_heads: Attention heads.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        block_len: int = 5,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("`dim` must be divisible by `num_heads`.")
        if block_len != 5:
            raise ValueError(
                "Only block_len == 5 is supported (causal mask is hard-coded)."
            )

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.block_size = block_len * block_len
        self.scale = qk_scale or head_dim**-0.5
        self.attn_drop = nn.Dropout(attn_drop)

        # Causal mask over the 5×5 local window: keep the 12 strictly causal
        # positions (rows 0–1 plus row 2 cols 0–1), zero out the rest. Stored
        # as a non-persistent buffer so it follows .to(device) but doesn't
        # land in the state dict.
        causal_mask = torch.tensor(
            [
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
            dtype=torch.float32,
        ).view(1, self.block_size, 1)
        self.register_buffer("mask", causal_mask, persistent=False)

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Relative position bias table for the 5×5 window (Swin-style).
        table_size = (2 * block_len - 1) * (2 * block_len - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads)
        )
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(block_len),
                torch.arange(block_len),
                indexing="ij",
            )
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += block_len - 1
        relative_coords[:, :, 1] += block_len - 1
        relative_coords[:, :, 0] *= 2 * block_len - 1
        self.register_buffer(
            "relative_position_index", relative_coords.sum(-1)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=attn_drop,
        )
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, channels, height, width = input_tensor.shape

        # Unfold to per-pixel 5×5 patches (B, C*PP, H*W) → (B*H*W, PP, C).
        unfolded = F.unfold(input_tensor, kernel_size=5, padding=2)
        unfolded = (
            unfolded.reshape(batch_size, channels, self.block_size, height * width)
            .permute(0, 3, 2, 1)
            .contiguous()
            .view(-1, self.block_size, channels)
        )

        masked = unfolded * self.mask
        normed = self.norm1(masked)
        qkv = (
            self.qkv(normed)
            .reshape(
                batch_size * height * width,
                self.block_size,
                3,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query * self.scale) @ key.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.block_size, self.block_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_position_bias.unsqueeze(0)

        attention = self.attn_drop(self.softmax(attention))
        out = (
            (attention @ value)
            .transpose(1, 2)
            .reshape(batch_size * height * width, self.block_size, channels)
        )
        out = out + masked
        summed = torch.sum(out, dim=1).reshape(batch_size, height * width, channels)

        out = self.mlp(self.norm2(summed)) + summed
        out = self.proj(out)
        return out.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2)
