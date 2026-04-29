"""Directional shift building blocks for ShiftLIC.

Vendored from `Bao et al., "Lossy Image Compression with Stochastic
Quantization" / "ShiftLIC", arXiv:2503.23052`_ (TCSVT 2025). License is not
declared upstream; this fork ships per existing repo policy.

Module / parameter naming preserves the upstream convention so converted
state_dicts load 1:1.

.. _arXiv:2503.23052: https://arxiv.org/abs/2503.23052
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init


__all__ = [
    "Shift4",
    "ResidualBlockShift",
    "channel_shuffle",
    "CheapChannelV1",
    "CheapCS1",
    "ResidualShiftStack",
]


def _default_init_conv(module_list, scale: float = 0.1, bias_fill: float = 0.0) -> None:
    """Match upstream's per-conv Kaiming init scaled for residual blocks."""
    if not isinstance(module_list, (list, tuple)):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class Shift4(nn.Module):
    """Four-direction channel-grouped shift (up / down / left / right)."""

    def __init__(
        self,
        groups: int = 4,
        stride: int = 1,
        mode: str = "constant",
    ) -> None:
        super().__init__()
        self.g = int(groups)
        self.stride = int(stride)
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        assert c == self.g * 4, (
            f"Shift4 expects channels = 4 * groups; got C={c}, groups={self.g}"
        )

        pad_x = F.pad(x, [self.stride] * 4, mode=self.mode)
        out = torch.zeros_like(x)
        cx = cy = self.stride
        s = self.stride
        out[:, 0 * self.g : 1 * self.g] = pad_x[
            :, 0 * self.g : 1 * self.g, cx - s : cx - s + h, cy : cy + w
        ]
        out[:, 1 * self.g : 2 * self.g] = pad_x[
            :, 1 * self.g : 2 * self.g, cx + s : cx + s + h, cy : cy + w
        ]
        out[:, 2 * self.g : 3 * self.g] = pad_x[
            :, 2 * self.g : 3 * self.g, cx : cx + h, cy - s : cy - s + w
        ]
        out[:, 3 * self.g : 4 * self.g] = pad_x[
            :, 3 * self.g : 4 * self.g, cx : cx + h, cy + s : cy + s + w
        ]
        return out


class ResidualBlockShift(nn.Module):
    """1×1 conv → ReLU → Shift4 → 1×1 conv, with a 1×1 skip if needed."""

    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        res_scale: float = 1.0,
        pytorch_init: bool = False,
    ) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_feat, in_feat, kernel_size=1)
        self.conv2 = nn.Conv2d(in_feat, out_feat, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.shift = Shift4(groups=in_feat // 4, stride=1)

        if not pytorch_init:
            _default_init_conv([self.conv1, self.conv2], scale=0.1)

        if in_feat != out_feat:
            self.skip = nn.Conv2d(in_feat, out_feat, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        out = self.conv2(self.shift(self.relu(self.conv1(x))))
        return identity + out * self.res_scale


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """Pixel-shuffle-style channel permutation used by CheapCS."""
    batch, channels, height, width = x.size()
    assert channels % groups == 0, (
        f"channels ({channels}) must be divisible by groups ({groups})"
    )
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.view(batch, -1, height, width)


class CheapChannelV1(nn.Module):
    """Multi-resolution depthwise context fused via channel-shuffled 1×1s."""

    def __init__(self, dim: int, n_levels: int = 4) -> None:
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        self.mfr = nn.ModuleList(
            [
                nn.Conv2d(
                    chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim
                )
                for _ in range(n_levels)
            ]
        )
        self.act = nn.GELU()
        self.fusion1 = nn.Conv2d(chunk_dim * 2, chunk_dim * 2, 1)
        self.fusion2 = nn.Conv2d(chunk_dim * 3, chunk_dim * 3, 1)
        self.fusion3 = nn.Conv2d(chunk_dim * 4, chunk_dim * 4, 1)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        s = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2**i, w // 2**i)
                t = F.adaptive_max_pool2d(xc[i], p_size)
                t = self.mfr[i](t)
                t = F.interpolate(t, size=(h, w), mode="nearest")
            else:
                t = self.mfr[i](xc[i])
            s.append(t)

        res1 = self.fusion1(channel_shuffle(torch.cat([s[0], s[1]], dim=1), 8))
        res2 = self.fusion2(channel_shuffle(torch.cat([res1, s[2]], dim=1), 8))
        res3 = self.fusion3(channel_shuffle(torch.cat([res2, s[3]], dim=1), 8))
        return self.act(res3) * x


class CheapCS1(nn.Module):
    """Cheap Spatial-Channel attention used in ShiftLIC middle/large."""

    def __init__(self, dim: int, n_levels: int = 4) -> None:
        del n_levels  # kept for signature parity with upstream
        super().__init__()
        self.CheapChannel = CheapChannelV1(dim)
        self.CheapSpatial = nn.Sequential(
            ResidualBlockShift(dim, dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.CheapChannel(x) + x
        y = self.CheapSpatial(y) + y
        return y


def ResidualShiftStack(in_ch: int, out_ch: int) -> nn.Module:
    """ShiftLIC large's ``cc_transform`` factory.

    Seven-module sequential ``ResidualBlockShift × 5`` interleaved with two
    ``GELU``s. The first block ramps ``in_ch -> out_ch // 2``, the inner
    blocks stay at ``out_ch // 2``, and the final block doubles to
    ``out_ch`` (the codec consumes ``2 * slice_size`` scale+mean channels).

    Pass to
    :class:`compressai.latent_codecs.MultistageCheckerboardLatentCodec` as
    ``make_cc_transform``.
    """
    if out_ch % 2 != 0:
        raise ValueError(
            "ResidualShiftStack expects out_ch to be even (codec passes "
            "2*slice_size); got {out_ch}"
        )
    inner = out_ch // 2
    return nn.Sequential(
        ResidualBlockShift(in_ch, inner),
        ResidualBlockShift(inner, inner),
        nn.GELU(),
        ResidualBlockShift(inner, inner),
        ResidualBlockShift(inner, inner),
        nn.GELU(),
        ResidualBlockShift(inner, out_ch),
    )
