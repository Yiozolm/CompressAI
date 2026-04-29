"""Convert an upstream TinyLIC checkpoint into compressai layout.

The upstream checkpoints (NJU Box, Q1–Q8) carry the same parameter shapes as
this fork's :class:`compressai.models.TinyLIC` — the only key difference is
the optional ``module.`` prefix from ``torch.nn.DataParallel``. The codec's
top-level entropy parameters
(``entropy_parameters_*`` / ``cc_transforms.*`` / ``sc_transform_*`` /
``gaussian_conditional.*``) are remapped to live under ``latent_codec.*``
inside :func:`TinyLIC.load_state_dict`, so this script only needs to strip
the ``module.`` prefix.

Example::

    python examples/convert_tinylic_checkpoint.py \\
        --src /path/to/tinylic_q5_mse.pth.tar \\
        --dst /tmp/tinylic_q5_mse_compressai.pth \\
        --smoke
"""
from __future__ import annotations

import argparse
import math

from pathlib import Path

import torch

from compressai.models import TinyLIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the upstream TinyLIC checkpoint (.pth / .pth.tar).",
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


def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove the optional ``module.`` prefix added by ``DataParallel``."""
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[len("module.") :] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


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
    state_dict = _strip_module_prefix(state_dict)
    print(f"loaded {len(state_dict)} state-dict keys")

    net = TinyLIC.from_state_dict(state_dict)
    net.eval()
    print(f"variant: N={net.N}, M={net.M}")
    print(f"parameters: {sum(p.numel() for p in net.parameters()):,}")

    if args.dst is not None:
        args.dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), args.dst)
        print(f"wrote converted state dict -> {args.dst}")

        verify = TinyLIC(N=net.N, M=net.M)
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
        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, height),
            torch.linspace(0, 1, width),
            indexing="ij",
        )
        image = torch.stack([xs, ys, (xs + ys) * 0.5]).unsqueeze(0)

        with torch.no_grad():
            output = net(image)

        n_pixels = image.size(0) * image.size(2) * image.size(3)
        y_bpp = -output["likelihoods"]["y"].log2().sum().item() / n_pixels
        z_bpp = -output["likelihoods"]["z"].log2().sum().item() / n_pixels
        mse = (output["x_hat"].clamp(0, 1) - image).pow(2).mean().item()
        psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float("inf")
        print(
            f"smoke: PSNR={psnr:.2f} dB  total_bpp={y_bpp + z_bpp:.4f}  "
            f"(y={y_bpp:.4f} z={z_bpp:.4f})"
        )


if __name__ == "__main__":
    main()
