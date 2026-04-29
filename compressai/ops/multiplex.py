"""Space/depth shuffles + checkerboard (de)multiplexers.

Vendored helpers from TinyLIC. Used by both TinyLIC and ShiftLIC large during
``compress``/``decompress`` to interleave / split the staged checkerboard
slices when building bit-exact bitstreams.

- ``space2depth`` / ``depth2space``: pixel-shuffle style spatial→channel and
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
    return (
        x_view.permute(0, 3, 5, 1, 2, 4)
        .contiguous()
        .view(b, out_c, out_h, out_w)
    )


def depth2space(x: torch.Tensor, r: int = 2) -> torch.Tensor:
    b, c, h, w = x.size()
    out_c = c // (r**2)
    out_h = h * r
    out_w = w * r
    x_view = x.view(b, r, r, out_c, h, w)
    return (
        x_view.permute(0, 3, 4, 1, 5, 2)
        .contiguous()
        .view(b, out_c, out_h, out_w)
    )


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
