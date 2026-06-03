"""Convert an upstream WeConvene checkpoint to compressai layout.

Loads the published candidate weight file (e.g. ``0.0025175_checkpoint.pth.tar``
from the WeConvene repo,
https://github.com/fengyurenpingsheng/WeConvene-Learned-Image-Compression-with-Wavelet-Domain-Convolution-and-Entropy-Model),
translates it to compressai's containerized module layout, and writes a state
dict that ``compressai.models.weconvene.WeConvene.from_state_dict`` can load
directly. Optionally reports forward-pass sanity numbers (PSNR / bpp) on a
synthetic input.

The upstream-vs-compressai key differences are:

- Each transform (``g_a`` / ``g_s`` / ``h_a`` / ``h_mean_s`` / ``h_scale_s``)
  is stored as a flat ``nn.Sequential``; we map indices back to the named
  sub-stages (``input_block``, ``down1``/``down2``/``down3``,
  ``wavelet_block``, ``up.0`` ... etc.).
- The wavelet-domain entropy branches are split as ``_real`` / ``_imag`` and
  live at the top level; we move them under ``latent_codec.*_low`` /
  ``latent_codec.*_high``.
- Each ``SWAtten`` is wrapped in a one-element ``nn.Sequential`` (an extra
  ``.0.`` segment) which we strip.
- ``DWT_2D`` / ``IDWT_2D`` register ``w_{ll,lh,hl,hh}`` / ``filters`` buffers,
  whereas compressai wraps ``pytorch_wavelets`` (separable
  ``transform.h{0,1}_{col,row}`` / ``inverse.g{0,1}_{col,row}`` buffers
  rebuilt at construction time); we drop the upstream buffers.
- The unused ``conv2`` weights inside upstream's wavelet residual blocks are
  dropped (the upstream forward never reads them).
- The upstream ``SwinBlock`` inside ``SWAtten`` is two stacked ``ResidualBlock``s
  (the transformer-style ``Block`` pair is commented out in the published
  code), so the converted state dict carries the residual-attention layout and
  ``WeConvene.from_state_dict`` builds with ``use_residual_attention=True``.

All renames are handled inside ``convert_upstream_weconvene_state_dict``; this
script is a thin CLI around it.

Example::

    python examples/convert_weconvene_checkpoint.py \\
        --src candidate/WeConvene/0.0025175_checkpoint.pth.tar \\
        --dst /tmp/weconvene_compressai.pth \\
        --smoke
"""

from __future__ import annotations

import argparse

from pathlib import Path
from typing import Dict, Sequence

import torch

from torch import Tensor

from compressai.models.weconvene import WeConvene

# ----------------------------------------------------------------------------
# Upstream → compressai state-dict conversion.
#
# Lives here (not in compressai/models/weconvene.py) so the model module stays
# a clean compressai-native definition — ``WeConvene.from_state_dict`` only
# loads already-converted state dicts. Run this script once to translate a
# published upstream checkpoint into compressai layout, then load the result
# via ``from_state_dict``.
# ----------------------------------------------------------------------------

# upstream ``g_a`` / ``g_s`` are flat ``nn.Sequential`` of 13 children:
#     0       -> input_block (ResidualBlockWithStride / ResidualBlockUpsample)
#     1..3    -> 3x ResidualBlock  (down1 / up1)
#     4       -> WaveletResidualBlockWithStride / Upsample (tail of down1 / up1)
#     5..7    -> 3x ResidualBlock  (down2 / up2)
#     8       -> Wavelet*          (tail of down2 / up2)
#     9..11   -> 3x ResidualBlock  (down3 / up3)
#     12      -> conv3x3 / subpel_conv3x3 (tail of down3 / up3)
_GA_INDEX_MAP: Dict[int, str] = {0: "input_block"}
_GA_INDEX_MAP.update({i: f"down1.{i - 1}" for i in range(1, 5)})
_GA_INDEX_MAP.update({i: f"down2.{i - 5}" for i in range(5, 9)})
_GA_INDEX_MAP.update({i: f"down3.{i - 9}" for i in range(9, 13)})

_GS_INDEX_MAP: Dict[int, str] = {0: "input_block"}
_GS_INDEX_MAP.update({i: f"up1.{i - 1}" for i in range(1, 5)})
_GS_INDEX_MAP.update({i: f"up2.{i - 5}" for i in range(5, 9)})
_GS_INDEX_MAP.update({i: f"up3.{i - 9}" for i in range(9, 13)})

# upstream ``h_a`` / ``h_mean_s`` / ``h_scale_s`` are flat ``nn.Sequential`` of
# 5 children: ``[wavelet_block, ResidualBlock x 3, conv3x3 / subpel_conv3x3]``.
_HA_INDEX_MAP: Dict[int, str] = {0: "wavelet_block"}
_HA_INDEX_MAP.update({i: f"down.{i - 1}" for i in range(1, 5)})

_HS_INDEX_MAP: Dict[int, str] = {0: "wavelet_block"}
_HS_INDEX_MAP.update({i: f"up.{i - 1}" for i in range(1, 5)})

# Latent codec key prefix renames. Upstream stores the wavelet-domain
# low/high-pass branches as ``_real`` / ``_imag``.
_CODEC_PREFIX_RENAMES: Dict[str, str] = {
    "cc_mean_transforms_real.": "latent_codec.cc_mean_transforms_low.",
    "cc_mean_transforms_imag.": "latent_codec.cc_mean_transforms_high.",
    "cc_scale_transforms_real.": "latent_codec.cc_scale_transforms_low.",
    "cc_scale_transforms_imag.": "latent_codec.cc_scale_transforms_high.",
    "lrp_transforms_real.": "latent_codec.lrp_transforms_low.",
    "lrp_transforms_imag.": "latent_codec.lrp_transforms_high.",
    "gaussian_conditional_real.": "latent_codec.gaussian_conditional_low.",
    "gaussian_conditional_imag.": "latent_codec.gaussian_conditional_high.",
}

# Atten key renames. Upstream wraps each ``SWAtten`` in a one-element
# ``nn.Sequential`` (extra ``.0.`` segment) under ``atten_*_{real,imag}``.
_ATTEN_PREFIX_RENAMES: Dict[str, str] = {
    "atten_mean_real.": "latent_codec.mean_support_transforms_low.",
    "atten_mean_imag.": "latent_codec.mean_support_transforms_high.",
    "atten_scale_real.": "latent_codec.scale_support_transforms_low.",
    "atten_scale_imag.": "latent_codec.scale_support_transforms_high.",
}


def _is_upstream_layout(state_dict: Dict[str, Tensor]) -> bool:
    """Heuristic: upstream WeConvene checkpoints carry the wavelet-domain split
    under ``_real`` / ``_imag`` suffixes at the top level rather than under
    ``latent_codec.*_low`` / ``latent_codec.*_high``.
    """
    for key in state_dict:
        if key.startswith("cc_mean_transforms_real.") or key.startswith(
            "gaussian_conditional_real."
        ):
            return True
    return False


def _migrate_transform_block(
    key: str,
    prefix: str,
    index_map: Dict[int, str],
    *,
    drop_unused_conv2_indices: Sequence[int] = (),
) -> str | None:
    """Translate one ``g_a/g_s/h_a/h_mean_s/h_scale_s`` key to compressai layout.

    Returns ``None`` when the key should be dropped (DWT/IDWT buffers, unused
    upstream ``conv2`` weights inside wavelet blocks).
    """
    rest = key[len(prefix) :]
    head, _, tail = rest.partition(".")
    if not head.isdigit():
        return key
    idx = int(head)
    if idx not in index_map:
        return key
    if tail.startswith("dwt.") or tail.startswith("idwt."):
        return None  # buffers regenerated at construction (different layout)
    if idx in drop_unused_conv2_indices and tail.startswith("conv2."):
        return None  # upstream wavelet block has unused conv2
    return f"{prefix}{index_map[idx]}.{tail}"


def convert_upstream_weconvene_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Translate an upstream WeConvene checkpoint to compressai key layout.

    The upstream model (``candidate/WeConvene``) stores parameters in a flat
    ``nn.Sequential`` for each transform, splits the wavelet-domain entropy
    branches as ``_real`` / ``_imag``, wraps each ``SWAtten`` in a
    one-element ``nn.Sequential``, and registers DWT/IDWT kernels as
    ``w_{ll,lh,hl,hh}`` / ``filters`` buffers. This helper rewrites those
    keys to the nested compressai layout (and drops buffers that are
    re-registered by ``pytorch_wavelets`` at construction time).
    """
    migrated: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        # Drop top-level DWT/IDWT (constructed inside the latent codec).
        if key.startswith("dwt.") or key.startswith("idwt."):
            continue

        # Codec prefix renames: cc_*/lrp_*/gaussian_conditional_* (no extra
        # wrapper to strip — match longest prefix).
        renamed = None
        for src_prefix, dst_prefix in _CODEC_PREFIX_RENAMES.items():
            if key.startswith(src_prefix):
                renamed = dst_prefix + key[len(src_prefix) :]
                break
        if renamed is not None:
            migrated[renamed] = value
            continue

        # SWAtten renames: drop the `.0.` Sequential wrapper between the
        # ModuleList index and the SWAtten internals.
        for src_prefix, dst_prefix in _ATTEN_PREFIX_RENAMES.items():
            if key.startswith(src_prefix):
                rest = key[len(src_prefix) :]
                slice_idx, _, after_slice = rest.partition(".")
                wrapper, _, after_wrapper = after_slice.partition(".")
                if wrapper != "0":
                    raise RuntimeError(
                        f"Unexpected upstream attention key (no `.0.` wrapper): {key}"
                    )
                renamed = f"{dst_prefix}{slice_idx}.{after_wrapper}"
                break
        if renamed is not None:
            migrated[renamed] = value
            continue

        # Transform blocks: g_a / g_s / h_a / h_mean_s / h_scale_s.
        if key.startswith("g_a."):
            renamed = _migrate_transform_block(
                key, "g_a.", _GA_INDEX_MAP, drop_unused_conv2_indices=(4, 8)
            )
        elif key.startswith("g_s."):
            renamed = _migrate_transform_block(key, "g_s.", _GS_INDEX_MAP)
        elif key.startswith("h_a."):
            renamed = _migrate_transform_block(
                key, "h_a.", _HA_INDEX_MAP, drop_unused_conv2_indices=(0,)
            )
        elif key.startswith("h_mean_s."):
            renamed = _migrate_transform_block(key, "h_mean_s.", _HS_INDEX_MAP)
        elif key.startswith("h_scale_s."):
            renamed = _migrate_transform_block(key, "h_scale_s.", _HS_INDEX_MAP)
        else:
            renamed = key

        if renamed is None:
            continue
        migrated[renamed] = value

    return migrated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help=(
            "Path to the upstream WeConvene checkpoint "
            "(e.g. 0.0025175_checkpoint.pth.tar)."
        ),
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help=(
            "Optional output path for the converted state dict. If omitted, "
            "the script only verifies that the checkpoint loads cleanly."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a forward smoke test on a synthetic 256x256 image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise SystemExit(f"checkpoint not found: {args.src}")

    upstream = torch.load(args.src, map_location="cpu", weights_only=False)
    state_dict = (
        upstream["state_dict"]
        if isinstance(upstream, dict) and "state_dict" in upstream
        else upstream
    )
    print(f"loaded {len(state_dict)} state-dict keys")

    if _is_upstream_layout(state_dict):
        state_dict = convert_upstream_weconvene_state_dict(state_dict)
        print(f"converted to compressai layout: {len(state_dict)} keys")

    net = WeConvene.from_state_dict(state_dict)
    net.eval()
    print(
        "variant: "
        f"N={net.N}, M={net.M}, num_slices={net.num_slices}, "
        f"hyper_channels={net.hyper_channels}, "
        f"use_residual_attention={net.use_residual_attention}"
    )
    print(f"parameters: {sum(p.numel() for p in net.parameters()):,}")

    if args.dst is not None:
        args.dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), args.dst)
        print(f"wrote converted state dict -> {args.dst}")

    if args.smoke:
        height = width = 256
        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, height),
            torch.linspace(0, 1, width),
            indexing="ij",
        )
        img = (
            torch.stack(
                [
                    0.5 + 0.3 * torch.sin(8 * xs),
                    0.5 + 0.3 * torch.sin(8 * ys),
                    0.5 + 0.3 * torch.cos(8 * (xs + ys)),
                ],
                dim=0,
            )
            .unsqueeze(0)
            .clamp(0, 1)
        )

        with torch.no_grad():
            out = net(img)
        n_pix = height * width
        psnr = -10 * torch.log10(((out["x_hat"].clamp(0, 1) - img) ** 2).mean()).item()
        y_low_bpp = -torch.log2(out["likelihoods"]["y_low"]).sum().item() / n_pix
        y_high_bpp = -torch.log2(out["likelihoods"]["y_high"]).sum().item() / n_pix
        z_bpp = -torch.log2(out["likelihoods"]["z"]).sum().item() / n_pix
        print(
            f"smoke: PSNR={psnr:.2f}dB  "
            f"y_low_bpp={y_low_bpp:.4f}  y_high_bpp={y_high_bpp:.4f}  "
            f"z_bpp={z_bpp:.4f}  "
            f"total_bpp={y_low_bpp + y_high_bpp + z_bpp:.4f}"
        )


if __name__ == "__main__":
    main()
