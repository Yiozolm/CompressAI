"""Convert an upstream LIC-HPCM checkpoint to compressai layout.

Loads the published candidate weight file (e.g. ``0.013.pth.tar`` from the
LIC-HPCM repo), translates it to compressai's HPCM module layout, and writes
a state dict that ``compressai.models.HPCM.from_state_dict`` can load
directly. Optionally reports forward-pass sanity numbers (PSNR / bpp) on a
synthetic input.

Example::

    python examples/convert_hpcm_checkpoint.py \\
        --src candidate/LIC-HPCM/0.013.pth.tar \\
        --dst /tmp/hpcm_phi_compressai.pth \\
        --smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from compressai.models import HPCM, convert_upstream_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the upstream LIC-HPCM checkpoint (e.g. 0.013.pth.tar).",
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
    converted = convert_upstream_state_dict(upstream)
    print(f"loaded {len(upstream)} upstream keys → {len(converted)} compressai keys")

    net = HPCM.from_state_dict(upstream)
    net.eval()
    print(
        "variant: "
        f"N={net.N}, M={net.M}, "
        f"g_a_depth={net.g_a_depth}, g_s_depth={net.g_s_depth}, "
        f"y_prior_depth={net.y_prior_depth}, "
        f"use_attention={net.use_attention}"
    )

    if args.dst is not None:
        args.dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), args.dst)
        print(f"wrote converted state dict → {args.dst}")

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
