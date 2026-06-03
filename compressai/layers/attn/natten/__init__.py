"""Lazy router for Neighborhood Attention.

Strategy
--------
1. If the optional ``natten`` PyPI package (Linux+CUDA wheel from NVIDIA SHI Lab)
   is importable, mark CUDA fast-path available. Currently we only use the
   detection helper; the actual ``NeighborhoodAttention`` returned is the
   vendored PyTorch reference, because the upstream TinyLIC checkpoint relies
   on the ``qkv``/``proj``/``rpb`` parameter layout which differs from the
   ``natten.NeighborhoodAttention2D`` module API.
2. Otherwise fall back to the vendored ``NeighborhoodAttentionTorch``.

The vendored pure-PyTorch path is always used today, so no ``natten`` extra is
declared in ``pyproject.toml``. A future revision can wire
``natten.functional.na2d_qk{rpb}_with_bias`` / ``na2d_av`` while keeping the
same parameter layout, providing a CUDA fast path without breaking state_dicts.
"""

import importlib.util

from ._torch_impl import NeighborhoodAttentionTorch, _warn_torch_fallback

__all__ = [
    "NeighborhoodAttention",
    "NeighborhoodAttentionTorch",
    "is_natten_available",
]


def is_natten_available() -> bool:
    """Return True if the optional NATTEN PyPI package is importable."""
    return importlib.util.find_spec("natten") is not None


class NeighborhoodAttention(NeighborhoodAttentionTorch):
    """Neighborhood Attention with state_dict layout compatible with
    upstream TinyLIC checkpoints (``qkv`` / ``proj`` / ``rpb``).

    Currently always uses the vendored PyTorch reference implementation.
    """

    def __init__(self, *args, **kwargs):
        if not is_natten_available():
            _warn_torch_fallback()
        super().__init__(*args, **kwargs)
