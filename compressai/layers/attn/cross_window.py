from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch import Tensor

__all__ = ["WindowedCrossAttention"]


class WindowedCrossAttention(nn.Module):
    """Windowed cross-attention used by HPCM's progressive context fusion.

    Splits both the query and key/value tensors into non-overlapping
    ``window_size``-sized patches, performs multi-head scaled dot-product
    attention inside each window with the previous step's context as keys and
    values, and merges windows back into a 2-D feature map. A residual
    projection of the original query is added to the attention output.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        window_size: int,
        kernel_size: int = 1,
        num_heads: int = 32,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.window_size = int(window_size)

        padding = kernel_size // 2
        self.conv_q = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_out = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        residual = query
        height, width = query.shape[-2:]
        ws = self.window_size
        q_windows = rearrange(
            query,
            "b c (w1 p1) (w2 p2) -> (b w1 w2) c p1 p2",
            p1=ws,
            p2=ws,
        )
        ctx_windows = rearrange(
            context,
            "b c (w1 p1) (w2 p2) -> (b w1 w2) c p1 p2",
            p1=ws,
            p2=ws,
        )
        n = q_windows.size(0)

        q = self.conv_q(q_windows)
        k = self.conv_k(ctx_windows)
        v = self.conv_v(ctx_windows)

        q = q.view(n, self.num_heads, self.head_dim, ws * ws).permute(0, 1, 3, 2)
        k = k.view(n, self.num_heads, self.head_dim, ws * ws)
        v = v.view(n, self.num_heads, self.head_dim, ws * ws).permute(0, 1, 3, 2)

        attn_scores = torch.matmul(q, k) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.permute(0, 1, 3, 2)
            .contiguous()
            .view(n, self.hidden_dim, ws, ws)
        )
        attn_output = rearrange(
            attn_output,
            "(b w1 w2) c p1 p2 -> b c (w1 p1) (w2 p2)",
            w1=height // ws,
            w2=width // ws,
        )
        return attn_output + self.conv_out(residual)
