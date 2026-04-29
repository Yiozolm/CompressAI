"""Token-based cross-attention helpers.

Vanilla scaled dot-product cross-attention over flat token sequences plus a
pre-norm transformer block (cross-attn + MLP). Used by Informer's global
hyperprior and parameter model. Distinct from
:class:`compressai.layers.attn.WindowedCrossAttention` which operates on 2-D
feature maps with non-overlapping windows.
"""

from __future__ import annotations

from typing import Optional, Type

import torch.nn as nn

from timm.layers import Mlp
from torch import Tensor

__all__ = ["CrossAttention", "CrossAttentionBlock"]


class CrossAttention(nn.Module):
    """Multi-head scaled dot-product cross-attention over token sequences.

    Args:
        dim: Channel dimension of both query and key/value tokens.
        num_heads: Number of attention heads. ``dim`` must be divisible.
        qkv_bias: Whether the q / kv linear projections include a bias term.
        qk_scale: Optional override for the default ``head_dim ** -0.5`` scale.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        head_dim = dim // num_heads
        self.scale = qk_scale if qk_scale is not None else head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        B, M, C = x.shape
        q = self.q(x).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        _, N, _ = y.shape
        kv = (
            self.kv(y)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, M, C)
        return self.proj(out)


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention transformer block: ``cross_attn → MLP``.

    The block matches the canonical ViT layout but uses
    :class:`CrossAttention` (queries from ``x``, keys/values from ``y``) in
    place of self-attention. Norms / MLP follow the standard pre-norm scheme
    with residual connections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale
        )
        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x + self.cross_attn(self.norm_q(x), self.norm_kv(y))
        x = x + self.mlp(self.norm_mlp(x))
        return x
