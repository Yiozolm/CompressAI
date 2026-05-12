# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Spatio-channel causal Transformer context model for ContextFormer.

Koyuncu et al. split the latent tensor into channel segments and feed those
segments to a causal Transformer either in channel-first order (``cfo``) or
spatial-first order (``sfo``). This module only implements the context-model
forward path; practical bitstream coding still needs token-wise decoding and
the paper's sliding-window / wavefront runtime optimizations.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from torch import Tensor

__all__ = ["ContextFormerBlock", "ContextFormerContextModel"]


def _check_order(order: str) -> str:
    if order not in ("cfo", "sfo"):
        raise ValueError(f'Invalid ContextFormer coding order "{order}"')
    return order


class ContextFormerBlock(nn.Module):
    """Pre-norm Transformer block used by the ContextFormer context model."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.mlp(self.norm2(x)))
        return x


class ContextFormerContextModel(nn.Module):
    """Causal spatio-channel Transformer over segmented latents.

    Args:
        latent_channels: Number of channels in the latent ``y`` tensor.
        num_segments: Number of contiguous channel segments ``Ncs``.
        embed_dim: Transformer embedding dimension ``de``.
        depth: Number of Transformer layers ``L``.
        num_heads: Number of attention heads ``h``.
        mlp_ratio: Feed-forward expansion ratio. The paper uses ``4``.
        order: ``"cfo"`` for channel-first-order or ``"sfo"`` for
            spatial-first-order.
        max_spatial_size: Maximum latent height/width supported by learned
            row/column position embeddings.
        dropout: Dropout probability in attention and MLP layers.
    """

    def __init__(
        self,
        latent_channels: int,
        num_segments: int = 4,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        order: str = "cfo",
        max_spatial_size: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        order = _check_order(order)
        if latent_channels % num_segments != 0:
            raise ValueError(
                f"latent_channels ({latent_channels}) must be divisible by "
                f"num_segments ({num_segments})"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads})"
            )

        self.latent_channels = int(latent_channels)
        self.num_segments = int(num_segments)
        self.segment_channels = self.latent_channels // self.num_segments
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.order = order
        self.max_spatial_size = int(max_spatial_size)

        self.input_projection = nn.Linear(self.segment_channels, self.embed_dim)
        self.row_embedding = nn.Embedding(self.max_spatial_size, self.embed_dim)
        self.col_embedding = nn.Embedding(self.max_spatial_size, self.embed_dim)
        self.segment_embedding = nn.Embedding(self.num_segments, self.embed_dim)
        self.blocks = nn.ModuleList(
            [
                ContextFormerBlock(
                    embed_dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        self.register_buffer(
            "order_code",
            torch.tensor(0 if order == "cfo" else 1, dtype=torch.long),
        )
        self.register_buffer(
            "num_heads",
            torch.tensor(int(num_heads), dtype=torch.long),
        )

    def _check_spatial_size(self, height: int, width: int) -> None:
        if height > self.max_spatial_size or width > self.max_spatial_size:
            raise ValueError(
                "ContextFormerContextModel received latent spatial size "
                f"({height}, {width}), but max_spatial_size is "
                f"{self.max_spatial_size}."
            )

    def _sequence_position_indices(
        self, height: int, width: int, device: torch.device
    ) -> Tuple[Tensor, Tensor, Tensor]:
        self._check_spatial_size(height, width)
        rows = torch.arange(height, device=device)
        cols = torch.arange(width, device=device)
        segs = torch.arange(self.num_segments, device=device)

        if self.order == "cfo":
            row_idx = rows.view(height, 1, 1).expand(
                height, width, self.num_segments
            )
            col_idx = cols.view(1, width, 1).expand(
                height, width, self.num_segments
            )
            seg_idx = segs.view(1, 1, self.num_segments).expand(
                height, width, self.num_segments
            )
        else:
            row_idx = rows.view(1, height, 1).expand(
                self.num_segments, height, width
            )
            col_idx = cols.view(1, 1, width).expand(
                self.num_segments, height, width
            )
            seg_idx = segs.view(self.num_segments, 1, 1).expand(
                self.num_segments, height, width
            )

        return row_idx.reshape(-1), col_idx.reshape(-1), seg_idx.reshape(-1)

    def _position_encoding(self, height: int, width: int, device: torch.device) -> Tensor:
        row_idx, col_idx, seg_idx = self._sequence_position_indices(
            height, width, device
        )
        return (
            self.row_embedding(row_idx)
            + self.col_embedding(col_idx)
            + self.segment_embedding(seg_idx)
        )

    def _causal_mask(self, length: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(length, length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def latent_to_sequence(self, y: Tensor) -> Tensor:
        """Convert ``(B, M, H, W)`` to segmented sequence tokens."""
        batch, channels, height, width = y.shape
        if channels != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels, got {channels}."
            )

        y = y.reshape(
            batch, self.num_segments, self.segment_channels, height, width
        )
        if self.order == "cfo":
            return y.permute(0, 3, 4, 1, 2).reshape(
                batch,
                height * width * self.num_segments,
                self.segment_channels,
            )
        return y.permute(0, 1, 3, 4, 2).reshape(
            batch,
            height * width * self.num_segments,
            self.segment_channels,
        )

    def spatial_to_sequence(self, x: Tensor) -> Tensor:
        """Repeat spatial hyperprior features for every channel segment."""
        batch, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1)
        if self.order == "cfo":
            return x.unsqueeze(3).expand(
                batch, height, width, self.num_segments, channels
            ).reshape(batch, height * width * self.num_segments, channels)
        return x.unsqueeze(1).expand(
            batch, self.num_segments, height, width, channels
        ).reshape(batch, height * width * self.num_segments, channels)

    def forward(self, y_hat: Tensor) -> Tensor:
        batch, _, height, width = y_hat.shape
        tokens = self.latent_to_sequence(y_hat)
        start = tokens.new_zeros(batch, 1, self.segment_channels)
        shifted = torch.cat((start, tokens[:, :-1]), dim=1)

        hidden = self.input_projection(shifted)
        hidden = hidden + self._position_encoding(
            height, width, y_hat.device
        ).to(dtype=hidden.dtype).unsqueeze(0)

        attn_mask = self._causal_mask(hidden.size(1), y_hat.device)
        for block in self.blocks:
            hidden = block(hidden, attn_mask)
        return self.norm(hidden)
