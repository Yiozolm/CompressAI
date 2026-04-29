"""Convert an upstream WeConvene checkpoint to compressai layout.

Loads the published candidate weight file (e.g. ``0.0025175_checkpoint.pth.tar``
from the WeConvene repo,
https://github.com/fengyurenpingsheng/WeConvene-Learned-Image-Compression-with-Wavelet-Domain-Convolution-and-Entropy-Model),
and writes a state dict that ``compressai.models.WeConvene.from_state_dict``
can load directly. Optionally reports forward-pass sanity numbers (PSNR / bpp)
on a synthetic input.

Compared to the candidate sources, the upstream-vs-compressai key differences
are:

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
  code), so the converted model is built with ``use_residual_attention=True``.

All renames are handled inside ``WeConvene.from_state_dict`` /
``compressai.models.weconvene.convert_upstream_state_dict``; this script is a
thin CLI around them.

Example::

    python examples/convert_weconvene_checkpoint.py \\
        --src candidate/WeConvene/0.0025175_checkpoint.pth.tar \\
        --dst /tmp/weconvene_compressai.pth \\
        --smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from compressai.models import WeConvene


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
