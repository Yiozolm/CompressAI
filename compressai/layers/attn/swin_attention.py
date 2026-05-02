from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import to_2tuple, trunc_normal_
from torch import Tensor

__all__ = [
    "WindowAttention",
    "build_window_attention_mask",
    "pad_to_window_multiple",
    "window_partition",
    "window_reverse",
]


def window_partition(input_tensor: Tensor, window_size: int) -> Tensor:
    batch_size, height, width, channels = input_tensor.shape
    output = input_tensor.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    output = output.permute(0, 1, 3, 2, 4, 5).contiguous()
    return output.view(-1, window_size, window_size, channels)


def window_reverse(
    windows: Tensor,
    window_size: int,
    height: int,
    width: int,
) -> Tensor:
    windows_per_image = (height // window_size) * (width // window_size)
    batch_size = windows.shape[0] // windows_per_image
    output = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    output = output.permute(0, 1, 3, 2, 4, 5).contiguous()
    return output.view(batch_size, height, width, -1)


def build_window_attention_mask(
    height: int,
    width: int,
    window_size: int,
    shift_size: int,
    device: torch.device,
) -> Optional[Tensor]:
    if shift_size == 0:
        return None

    img_mask = torch.zeros((1, height, width, 1), device=device)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )

    count = 0
    for h_index in h_slices:
        for w_index in w_slices:
            img_mask[:, h_index, w_index, :] = count
            count += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-100.0))
    return attention_mask.masked_fill(attention_mask == 0, float(0.0))


def pad_to_window_multiple(
    input_tensor: Tensor,
    window_size: Union[int, Tuple[int, int]],
    *,
    layout: str = "BCHW",
) -> Tuple[Tensor, int, int]:
    """Right/bottom-pad a 4D tensor so its spatial dims are multiples of
    ``window_size``.

    Args:
        input_tensor: 4D tensor in either ``BCHW`` or ``BHWC`` layout.
        window_size: ``int`` (square window) or ``(window_h, window_w)``.
        layout: ``"BCHW"`` (default, PyTorch convention) or ``"BHWC"``
            (Swin / FTIC token-major layout).

    Returns:
        ``(padded_tensor, pad_h, pad_w)``, where ``pad_h`` / ``pad_w`` are
        the bottom / right padding widths added to the height / width
        dimension respectively.
    """
    if isinstance(window_size, int):
        win_h = win_w = int(window_size)
    else:
        win_h, win_w = (int(s) for s in window_size)

    if layout == "BCHW":
        height, width = input_tensor.shape[-2], input_tensor.shape[-1]
    elif layout == "BHWC":
        height, width = input_tensor.shape[1], input_tensor.shape[2]
    else:
        raise ValueError(f"layout must be 'BCHW' or 'BHWC', got {layout!r}")

    pad_h = (win_h - height % win_h) % win_h
    pad_w = (win_w - width % win_w) % win_w
    if pad_h == 0 and pad_w == 0:
        return input_tensor, 0, 0

    if layout == "BCHW":
        # F.pad on BCHW: (W_left, W_right, H_left, H_right)
        return F.pad(input_tensor, (0, pad_w, 0, pad_h)), pad_h, pad_w
    # F.pad on BHWC: (C_left, C_right, W_left, W_right, H_left, H_right)
    return F.pad(input_tensor, (0, 0, 0, pad_w, 0, pad_h)), pad_h, pad_w


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        table_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads)
        )
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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, input_tensor: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_windows, num_tokens, channels = input_tensor.shape
        qkv = self.qkv(input_tensor).reshape(
            batch_windows,
            num_tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query * self.scale) @ key.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.view(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                num_tokens,
                num_tokens,
            )
            attention = attention + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, num_tokens, num_tokens)

        attention = self.attn_drop(self.softmax(attention))
        output = (attention @ value).transpose(1, 2)
        output = output.reshape(batch_windows, num_tokens, channels)
        output = self.proj(output)
        return self.proj_drop(output)
