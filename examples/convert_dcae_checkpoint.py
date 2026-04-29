"""Convert an upstream DCAE checkpoint to compressai layout.

Loads the published DCAE weight file (e.g. ``0.0018checkpoint_best.pth.tar``
from https://github.com/LabShuHangGU/DCAE) and writes a state dict that
``compressai.models.DCAE.from_state_dict`` can load directly. Optionally
reports forward-pass sanity numbers (PSNR / bpp) on a synthetic input.

Upstream stores the model under ``state_dict`` with a ``module.`` prefix from
``DataParallel``; the parameter / buffer names already match compressai's
DCAE module tree (the Python implementation was migrated structurally
identical, see ``compressai/models/dcae.py``), so the conversion is a thin
prefix strip plus reuse of ``DCAE.from_state_dict`` to auto-infer the
N / M / feature_dims / block_num / head_dim / num_slices / dictionary
hyperparameters.

Example::

    python examples/convert_dcae_checkpoint.py \\
        --src candidate/DCAE/0.0018checkpoint_best.pth.tar \\
        --dst /tmp/dcae_compressai.pth \\
        --smoke
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from compressai.models import DCAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the upstream DCAE checkpoint (e.g. 0.0018checkpoint_best.pth.tar).",
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


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise SystemExit(f"checkpoint not found: {args.src}")

    upstream = torch.load(args.src, map_location="cpu", weights_only=False)
    raw_state_dict = (
        upstream["state_dict"]
        if isinstance(upstream, dict) and "state_dict" in upstream
        else upstream
    )
    state_dict = _strip_module_prefix(raw_state_dict)
    print(f"loaded {len(state_dict)} state-dict keys")

    net = DCAE.from_state_dict(state_dict)
    net.eval()
    print(
        "variant: "
        f"N={net.N}, M={net.M}, num_slices={net.num_slices}, "
        f"max_support_slices={net.max_support_slices}, "
        f"feature_dims={tuple(net.feature_dims)}, "
        f"block_num={tuple(net.block_num)}, head_dim={tuple(net.head_dim)}, "
        f"window_size={net.window_size}, "
        f"hyper_channels={net.hyper_channels}, "
        f"dict_num={net.dict_num}, dict_head_num={net.dict_head_num}, "
        f"dictionary_dim={net.dictionary_dim}"
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
        mse = ((out["x_hat"].clamp(0, 1) - img) ** 2).mean().item()
        psnr = -10 * math.log10(mse)
        y_bpp = -torch.log2(out["likelihoods"]["y"]).sum().item() / n_pix
        z_bpp = -torch.log2(out["likelihoods"]["z"]).sum().item() / n_pix
        print(
            f"smoke: PSNR={psnr:.2f}dB  y_bpp={y_bpp:.4f}  z_bpp={z_bpp:.4f}  "
            f"total_bpp={y_bpp + z_bpp:.4f}"
        )


if __name__ == "__main__":
    main()
