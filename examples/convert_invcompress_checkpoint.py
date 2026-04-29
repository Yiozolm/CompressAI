"""Convert an upstream InvCompress checkpoint to compressai layout.

Loads the published candidate weight file (e.g.
``checkpoint_best_loss-c32dfce8.pth.tar`` from the InvCompress repo,
https://github.com/xyq7/InvCompress), and writes a state dict that
``compressai.models.InvCompress.from_state_dict`` can load directly.
Optionally reports forward-pass sanity numbers (PSNR / bpp) on a synthetic
input.

Compared to the candidate sources, the upstream-vs-compressai key
differences are:

- ``inv.operations.{i}.weight`` (a trainable ``[C, C]`` matrix in upstream's
  custom ``InvertibleConv1x1``) is mapped to FrEIA's ``Fixed1x1Conv`` buffers
  ``inv.operations.{i}.backend.{M, M_inv, logDetM}``. The transform is
  numerically lossless for inference; ``Fixed1x1Conv`` is non-trainable in
  FrEIA 0.2 so re-finetuning these convs is not supported by the conversion.
- The four parallel coupling sub-bottlenecks ``G1`` / ``G2`` / ``H1`` / ``H2``
  are fused into FrEIA's two ``GLOWCouplingBlock`` subnets (output channels
  ordered ``[scale; translation]`` per the SIGMOID clamp activation). The
  fusion stacks ``conv1`` along the output dim and places ``conv2`` /
  ``conv3`` on the block diagonal with zero cross-coupling, preserving the
  upstream forward bit-for-bit.
- ``enh.*`` and ``attention.*`` keys already match between layouts.

Both renames are handled inside ``InvCompress.from_state_dict`` /
``compressai.models.invcompress.convert_upstream_state_dict``; this script
is a thin CLI around them.

Example::

    python examples/convert_invcompress_checkpoint.py \\
        --src candidate/InvCompress/checkpoint_best_loss-c32dfce8.pth.tar \\
        --dst /tmp/invcompress_compressai.pth \\
        --smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from compressai.models import InvCompress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help=(
            "Path to the upstream InvCompress checkpoint "
            "(e.g. checkpoint_best_loss-c32dfce8.pth.tar)."
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

    net = InvCompress.from_state_dict(state_dict)
    net.eval()
    print(f"variant: N={net.N}, M={net.M}")
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
        y_bpp = -torch.log2(out["likelihoods"]["y"]).sum().item() / n_pix
        z_bpp = -torch.log2(out["likelihoods"]["z"]).sum().item() / n_pix
        print(
            f"smoke: PSNR={psnr:.2f}dB  y_bpp={y_bpp:.4f}  z_bpp={z_bpp:.4f}  "
            f"total_bpp={y_bpp + z_bpp:.4f}"
        )


if __name__ == "__main__":
    main()
