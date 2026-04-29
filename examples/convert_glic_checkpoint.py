"""Convert an upstream GLIC checkpoint to compressai layout.

Loads the published GLIC weight file (e.g. ``0.025checkpoint_best.pth.tar``
from https://github.com/UnoC-727/GLIC) and writes a state dict that
``compressai.models.GLIC.from_state_dict`` can load directly. Optionally
reports forward-pass sanity numbers (PSNR / bpp) on a synthetic image.

Upstream-vs-compressai key differences (handled by
``compressai.models.glic.convert_upstream_state_dict``):

- ``latent_codec.hyper.*`` → ``latent_codec.z.*`` (HyperpriorLatentCodec
  stores the hyper sub-codec under the ``z`` slot).
- ``OLP`` submodule renamed to ``olp`` inside :class:`WLS` / :class:`iWLS`.
- ``GatedTransformCNN`` / :class:`GLICParameterAggregationBlock` use ``norm``
  instead of upstream's ``norm2``; renamed everywhere except inside
  ``residual_group.blocks.*`` (GFA blocks have both ``norm1`` / ``norm2``).
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

import torch

from compressai.models import GLIC


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

    net = GLIC.from_state_dict(state_dict)
    net.eval()
    print(f"variant: N={192}, M={net.g_a.down3.weight.size(0)}, groups={net.groups}")
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
        img = torch.stack(
            [
                0.5 + 0.3 * torch.sin(8 * xs),
                0.5 + 0.3 * torch.sin(8 * ys),
                0.5 + 0.3 * torch.cos(8 * (xs + ys)),
            ],
            dim=0,
        ).unsqueeze(0).clamp(0, 1)

        with torch.no_grad():
            out = net(img)
        n_pix = height * width
        psnr = -10 * torch.log10(
            ((out["x_hat"].clamp(0, 1) - img) ** 2).mean()
        ).item()
        likelihoods = out["likelihoods"]
        per_part = {
            name: -torch.log2(t).sum().item() / n_pix
            for name, t in likelihoods.items()
        }
        total_bpp = sum(per_part.values())
        parts = "  ".join(f"{name}_bpp={v:.4f}" for name, v in per_part.items())
        print(f"smoke: PSNR={psnr:.2f}dB  {parts}  total_bpp={total_bpp:.4f}")


if __name__ == "__main__":
    main()
