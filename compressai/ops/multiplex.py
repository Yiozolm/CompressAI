# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# This file adapts code from https://github.com/lumingzzz/TinyLIC
# (originally distributed under the Apache License 2.0). The upstream copyright
# notice is preserved in that repository; modifications by InterDigital
# Communications, Inc. are released under the BSD 3-Clause Clear License terms
# below.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

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

"""Space/depth shuffles + checkerboard (de)multiplexers.

Vendored helpers from TinyLIC. Used by both TinyLIC and ShiftLIC large during
``compress``/``decompress`` to interleave / split the staged checkerboard
slices when building bit-exact bitstreams.

- ``space2depth`` / ``depth2space``: pixel-shuffle style spatial->channel and
  inverse, with rate ``r`` (default 2).
- ``demultiplex`` / ``multiplex``: 2-way (anchor / non-anchor) split used for
  iterations 1..3 of TinyLIC.
- ``demultiplex_v2`` / ``multiplex_v2``: 4-way split used for iteration 0 of
  TinyLIC (the four checkerboard quadrants).
"""

import torch

__all__ = [
    "space2depth",
    "depth2space",
    "demultiplex",
    "multiplex",
    "demultiplex_v2",
    "multiplex_v2",
]


def space2depth(x: torch.Tensor, r: int = 2) -> torch.Tensor:
    b, c, h, w = x.size()
    out_c = c * (r**2)
    out_h = h // r
    out_w = w // r
    x_view = x.view(b, c, out_h, r, out_w, r)
    return x_view.permute(0, 3, 5, 1, 2, 4).contiguous().view(b, out_c, out_h, out_w)


def depth2space(x: torch.Tensor, r: int = 2) -> torch.Tensor:
    b, c, h, w = x.size()
    out_c = c // (r**2)
    out_h = h * r
    out_w = w * r
    x_view = x.view(b, r, r, out_c, h, w)
    return x_view.permute(0, 3, 4, 1, 5, 2).contiguous().view(b, out_c, out_h, out_w)


def demultiplex(x: torch.Tensor):
    """Split ``x`` into ``(anchor, non_anchor)`` checkerboard halves."""
    x_prime = space2depth(x, r=2)
    _, C, _, _ = x_prime.shape
    anchor_index = tuple(range(C // 4, C * 3 // 4))
    non_anchor_index = tuple(range(0, C // 4)) + tuple(range(C * 3 // 4, C))
    anchor = x_prime[:, anchor_index, :, :]
    non_anchor = x_prime[:, non_anchor_index, :, :]
    return anchor, non_anchor


def multiplex(anchor: torch.Tensor, non_anchor: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`demultiplex`."""
    _, C, _, _ = non_anchor.shape
    x_prime = torch.cat(
        (non_anchor[:, : C // 2, :, :], anchor, non_anchor[:, C // 2 :, :, :]),
        dim=1,
    )
    return depth2space(x_prime, r=2)


def demultiplex_v2(x: torch.Tensor):
    """Split ``x`` into the four checkerboard quadrants ``(y1, y2, y3, y4)``.

    Quadrant ordering matches TinyLIC's iteration-0 staged checkerboard.
    """
    x_prime = space2depth(x, r=2)
    _, C, _, _ = x_prime.shape
    y1_index = tuple(range(0, C // 4))
    y2_index = tuple(range(C * 3 // 4, C))
    y3_index = tuple(range(C // 4, C // 2))
    y4_index = tuple(range(C // 2, C * 3 // 4))
    return (
        x_prime[:, y1_index, :, :],
        x_prime[:, y2_index, :, :],
        x_prime[:, y3_index, :, :],
        x_prime[:, y4_index, :, :],
    )


def multiplex_v2(
    y1: torch.Tensor,
    y2: torch.Tensor,
    y3: torch.Tensor,
    y4: torch.Tensor,
) -> torch.Tensor:
    """Inverse of :func:`demultiplex_v2`."""
    x_prime = torch.cat((y1, y3, y4, y2), dim=1)
    return depth2space(x_prime, r=2)
