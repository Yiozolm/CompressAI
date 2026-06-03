"""Convert an upstream GLIC checkpoint to compressai layout.

Loads the published GLIC weight file (e.g. ``0.025checkpoint_best.pth.tar``
from https://github.com/UnoC-727/GLIC), translates it to compressai's
containerized module layout, and writes a state dict that
``compressai.models.glic.GLIC.from_state_dict`` can load directly. Optionally
reports forward-pass sanity numbers (PSNR / bpp) on a synthetic image.

The upstream-vs-compressai key differences (all handled inside
``convert_upstream_glic_state_dict``; this script is a thin CLI around it):

- ``latent_codec.hyper.*`` → ``latent_codec.{h_a,h_s}.*`` /
  ``latent_codec.z.*`` (compressai's modern ``HyperpriorLatentCodec`` keeps
  ``h_a`` / ``h_s`` at the outer level and only stores the
  ``EntropyBottleneckLatentCodec`` under the ``z`` slot).
- ``OLP`` submodule renamed to ``olp`` inside :class:`WLS` / :class:`iWLS`.
- ``GatedTransformCNN`` / ``GLICParameterAggregationBlock`` use ``norm``
  instead of upstream's ``norm2``; renamed everywhere except inside
  ``residual_group.blocks.*`` (GFA blocks legitimately have both ``norm1`` /
  ``norm2``).
- ``DepthwiseConv5x5.skip`` replaces upstream's ``mixer.adaptor`` (only
  present when ``in_ch != out_ch``); ``mixer.adaptor`` → ``mixer.skip``.
- The upstream wavelet buffers ``dwt.w_{ll,lh,hl,hh}`` / ``idwt.filters``
  (raw 2-D Haar kernels) are dropped: compressai's :class:`DWT2D` /
  :class:`IDWT2D` rebuild equivalent separable ``pytorch_wavelets`` buffers
  in their constructors and produce numerically identical outputs.

Example::

    python examples/convert_glic_checkpoint.py \\
        --src candidate/GLIC/0.025checkpoint_best.pth.tar \\
        --dst /tmp/glic_compressai.pth \\
        --smoke
"""

from __future__ import annotations

import argparse

from pathlib import Path
from typing import Dict

import torch

from torch import Tensor

from compressai.models.glic import GLIC

# ----------------------------------------------------------------------------
# Upstream → compressai state-dict conversion.
#
# Lives here (not in compressai/models/glic.py) so the model module stays a
# clean compressai-native definition — ``GLIC.from_state_dict`` only loads
# already-converted state dicts. Run this script once to translate a
# published upstream checkpoint into compressai layout, then load the result
# via ``from_state_dict``.
# ----------------------------------------------------------------------------

_UPSTREAM_HYPER_PREFIX = "latent_codec.hyper."


def _is_upstream_layout(state_dict: Dict[str, Tensor]) -> bool:
    """Detect the published GLIC checkpoint layout (``latent_codec.hyper.*`` +
    raw ``dwt.w_*`` / ``idwt.filters`` buffers + uppercase ``OLP``)."""
    for key in state_dict:
        if (
            key.startswith(_UPSTREAM_HYPER_PREFIX)
            or ".OLP." in key
            or key.endswith(".dwt.w_ll")
            or key.endswith(".idwt.filters")
        ):
            return True
    return False


def _rewrite_hyper_key(key: str) -> str:
    """Map upstream ``latent_codec.hyper.{h_a,h_s,entropy_bottleneck}.*`` keys
    to the compressai ``HyperpriorLatentCodec`` layout where ``h_a`` / ``h_s``
    sit at the outer level and only ``entropy_bottleneck`` lives under ``z``.
    """
    suffix = key[len(_UPSTREAM_HYPER_PREFIX) :]
    if suffix.startswith("h_a.") or suffix.startswith("h_s."):
        return "latent_codec." + suffix
    return "latent_codec.z." + suffix


def convert_upstream_glic_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Translate a published GLIC checkpoint into the compressai key layout.

    See the module docstring for the full list of key differences. The
    function is idempotent on already-converted dicts only insofar as the
    rename rules are no-ops there; call :func:`_is_upstream_layout` first to
    decide whether conversion is needed.
    """
    converted: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith((".dwt.w_ll", ".dwt.w_lh", ".dwt.w_hl", ".dwt.w_hh")):
            continue
        if key.endswith(".idwt.filters"):
            continue

        new_key = key
        if new_key.startswith(_UPSTREAM_HYPER_PREFIX):
            new_key = _rewrite_hyper_key(new_key)
        new_key = new_key.replace(".OLP.", ".olp.")
        new_key = new_key.replace(".mixer.adaptor.", ".mixer.skip.")
        if ".residual_group.blocks." not in new_key:
            new_key = new_key.replace(".norm2.", ".norm.")
        converted[new_key] = value
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help=(
            "Path to the upstream GLIC checkpoint "
            "(e.g. 0.025checkpoint_best.pth.tar)."
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
        state_dict = convert_upstream_glic_state_dict(state_dict)
        print(f"converted upstream layout -> {len(state_dict)} keys")

    net = GLIC.from_state_dict(state_dict)
    net.eval()
    print(f"variant: N=192, M={net.g_a.down3.weight.size(0)}, groups={net.groups}")
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
        likelihoods = out["likelihoods"]
        per_part = {
            name: -torch.log2(t).sum().item() / n_pix for name, t in likelihoods.items()
        }
        total_bpp = sum(per_part.values())
        parts = "  ".join(f"{name}_bpp={v:.4f}" for name, v in per_part.items())
        print(f"smoke: PSNR={psnr:.2f}dB  {parts}  total_bpp={total_bpp:.4f}")


if __name__ == "__main__":
    main()
