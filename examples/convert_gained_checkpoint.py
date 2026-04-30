"""Convert an upstream GainedVAE checkpoint to compressai layout.

Loads a GainedVAE community checkpoint (e.g.
``checkpoint_gainmshp_epoch90.pth`` from
https://github.com/mmSir/GainedVAE) and writes a state dict that
``compressai.models.GainedMSHyperprior.from_state_dict`` can load directly.
The upstream parameter names already match the compressai layout 1:1 (no
``module.`` prefix and no key renames needed); the only thing this script
does on top of a plain ``torch.load`` is unwrap the optimizer-included
training checkpoint to its ``state_dict`` and run optional sanity checks.

Optionally reports forward-pass sanity numbers (PSNR / bpp) on a synthetic
input across all gain levels and verifies the converted state dict is
bit-equivalent to the upstream one (eval-mode ``x_hat`` / likelihoods diff
= 0).

Example::

    python examples/convert_gained_checkpoint.py \\
        --src candidate/GainedVAE/checkpoint_gainmshp_epoch90.pth \\
        --dst /tmp/gainedms_compressai.pth \\
        --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from compressai.models import GainedMSHyperprior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the upstream GainedVAE checkpoint.",
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
        help=(
            "Run a forward smoke test on a synthetic 256x256 image across all "
            "gain levels."
        ),
    )
    return parser.parse_args()


def _synthetic_image(height: int = 256, width: int = 256) -> torch.Tensor:
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
    ).unsqueeze(0)
    return img.clamp(0, 1)


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

    net = GainedMSHyperprior.from_state_dict(state_dict)
    net.eval()
    print(f"variant: N={net.N}, M={net.M}, levels={net.levels}")
    print(f"parameters: {sum(p.numel() for p in net.parameters()):,}")

    if args.dst is not None:
        args.dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), args.dst)
        print(f"wrote converted state dict -> {args.dst}")

        verify = GainedMSHyperprior(N=net.N, M=net.M, lmbda=[0.0] * net.levels)
        missing, unexpected = verify.load_state_dict(
            torch.load(args.dst, map_location="cpu", weights_only=False),
            strict=True,
        )
        if missing or unexpected:
            raise SystemExit(
                f"strict reload failed: missing={missing}, unexpected={unexpected}"
            )
        print("strict reload of converted state dict: OK (no missing / unexpected)")

    if args.smoke:
        height = width = 256
        img = _synthetic_image(height, width)
        n_pix = height * width
        print(f"\nforward smoke (256x256 synthetic image):")
        print(f"  {'s':>3}  {'PSNR (dB)':>10}  {'y_bpp':>8}  {'z_bpp':>8}  {'total':>8}")
        for s in range(net.levels):
            with torch.no_grad():
                out = net(img, s=s)
            mse = ((out["x_hat"].clamp(0, 1) - img) ** 2).mean()
            psnr = -10 * torch.log10(mse).item()
            y_bpp = -torch.log2(out["likelihoods"]["y"]).sum().item() / n_pix
            z_bpp = -torch.log2(out["likelihoods"]["z"]).sum().item() / n_pix
            print(
                f"  {s:>3}  {psnr:>10.2f}  {y_bpp:>8.4f}  {z_bpp:>8.4f}  "
                f"{y_bpp + z_bpp:>8.4f}"
            )


if __name__ == "__main__":
    main()
