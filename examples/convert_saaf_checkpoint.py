"""Convert an upstream SAAF checkpoint to compressai layout.

Loads the published SAAF weight file (e.g. ``mse_0.0018.pth`` from
https://drive.google.com/drive/folders/1TZlDDxYhMyRKiQeCbDgtr6W-S-lm7tiz)
and writes a state dict that ``compressai.models.SAAF.from_state_dict`` can
load directly. Optionally reports forward-pass sanity numbers (PSNR / bpp)
on a synthetic input.

Upstream stores either a flat state dict or one wrapped under ``state_dict``
(possibly with a ``module.`` prefix from ``DataParallel``). The parameter /
buffer names already match compressai's ``SAAF`` module tree (the Python
implementation was migrated structurally identical, see
``compressai/models/saaf.py``), so the conversion is a thin prefix strip
plus dropping a handful of upstream-only persistent buffers (the
``aux_enc.{i}.olp.identity_matrix`` / ``aux_dec.{i}.olp.identity_matrix``
identity matrices, which compressai's :class:`OLP` registers as
``persistent=False`` and recomputes at construction). ``SAAF.from_state_dict``
then auto-infers N / M / feature_dims / block_num / head_dim / num_slices /
dictionary hyperparameters.

Example::

    python examples/convert_saaf_checkpoint.py \\
        --src candidate/SAAF/mse_0.0018.pth \\
        --dst /tmp/saaf_compressai.pth \\
        --smoke
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from compressai.models import SAAF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the upstream SAAF checkpoint (e.g. mse_0.0018.pth).",
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


_DROP_BUFFER_SUFFIXES = (".olp.identity_matrix",)


def _normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key[len("module.") :] if key.startswith("module.") else key
        if any(new_key.endswith(suffix) for suffix in _DROP_BUFFER_SUFFIXES):
            continue
        cleaned[new_key] = value
    return cleaned


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
    state_dict = _normalize_state_dict(raw_state_dict)
    print(
        f"loaded {len(raw_state_dict)} state-dict keys "
        f"({len(raw_state_dict) - len(state_dict)} dropped)"
    )

    net = SAAF.from_state_dict(state_dict)
    net.eval()
    print(
        "variant: "
        f"N={net.N}, M={net.M}, num_slices={net.num_slices}, "
        f"max_support_slices={net.max_support_slices}, "
        f"feature_dims={tuple(net.feature_dims)}, "
        f"block_num={tuple(net.block_num)}, head_dim={tuple(net.head_dim)}, "
        f"window_size={net.window_size}, "
        f"hyper_channels={net.hyper_channels}, "
        f"hyper_window_size={net.hyper_window_size}, "
        f"hyper_head_dim={net.hyper_head_dim}, "
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
