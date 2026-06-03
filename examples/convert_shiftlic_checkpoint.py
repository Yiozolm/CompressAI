"""Convert an upstream ShiftLIC checkpoint to compressai layout.

The upstream ShiftLIC training script
(https://github.com/baoyu2020/ShiftLIC, ``{small,middle,large}/train.py``)
saves the ``encoder`` / ``decoder`` / ``hyperencoder`` / ``hyperdecoder`` /
``entropy_bottleneck`` / ``gaussian_conditional`` modules with the same
parameter names this fork's :class:`compressai.models.shiftlic.ShiftLIC` uses.

``convert_upstream_shiftlic_state_dict`` (this script is a thin CLI around it)
handles two differences:

- The optional ``module.`` prefix from ``torch.nn.DataParallel`` is stripped.
- For the ``large`` variant the staged checkerboard entropy module's
  parameters (``entropy_parameters_*`` / ``cc_transforms.*`` /
  ``sc_transform_*`` / ``gaussian_conditional.*``) are stored at the top level
  upstream; they are reparented under ``latent_codec.*``. (``small`` / ``middle``
  do not use the codec, so for them only the prefix strip applies.)

``ShiftLIC.from_state_dict`` then infers the variant + N + M.

Forward / bitstream verification against an upstream weight is **not yet
wired in**: at vendoring time the upstream pre-trained ShiftLIC weights were
not published (the only candidate ``.pth.tar`` turned out to be a TinyLIC
checkpoint). The smoke test below runs on the converted weights regardless.

Example::

    python examples/convert_shiftlic_checkpoint.py \\
        --src /path/to/shiftlic_large.pth.tar \\
        --dst /tmp/shiftlic_large_compressai.pth \\
        --smoke
"""

from __future__ import annotations

import argparse
import math

from pathlib import Path
from typing import Dict

import torch

from torch import Tensor

from compressai.models.shiftlic import ShiftLIC

# ----------------------------------------------------------------------------
# Upstream -> compressai state-dict conversion.
#
# Lives here (not in compressai/models/shiftlic.py) so the model module stays a
# clean compressai-native definition -- ``ShiftLIC.from_state_dict`` only loads
# already-converted state dicts.
# ----------------------------------------------------------------------------

# Top-level prefixes that the ``large`` staged checkerboard codec owns.
# Upstream stores them at the model root; compressai keeps them under
# ``latent_codec.*``. The ``small`` / ``middle`` variants do not use the codec
# and keep their own top-level ``gaussian_conditional.*``, so the rewrite is
# applied only when the codec-only context prefixes are present.
_LARGE_ONLY_PREFIXES = (
    "entropy_parameters_",
    "cc_transforms.",
    "sc_transform_",
)
_CODEC_PREFIXES = _LARGE_ONLY_PREFIXES + ("gaussian_conditional.",)


def convert_upstream_shiftlic_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Translate an upstream ShiftLIC state dict into compressai layout.

    Strips the optional ``module.`` prefix. For ``large`` checkpoints (detected
    by the presence of codec-only top-level prefixes) the entropy keys are
    reparented under ``latent_codec.*``. Keys already namespaced under
    ``latent_codec.`` pass through unchanged.
    """
    stripped: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        stripped[key] = value

    is_large = any(
        key.startswith("latent_codec.")
        or any(key.startswith(prefix) for prefix in _LARGE_ONLY_PREFIXES)
        for key in stripped
    )
    if not is_large:
        return stripped

    converted: Dict[str, Tensor] = {}
    for key, value in stripped.items():
        if key.startswith("latent_codec."):
            converted[key] = value
        elif any(key.startswith(prefix) for prefix in _CODEC_PREFIXES):
            converted[f"latent_codec.{key}"] = value
        else:
            converted[key] = value
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the upstream ShiftLIC checkpoint (.pth / .pth.tar).",
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
    state_dict = convert_upstream_shiftlic_state_dict(state_dict)
    print(f"loaded {len(state_dict)} state-dict keys")

    net = ShiftLIC.from_state_dict(state_dict)
    net.eval()
    print(f"variant: {net.variant}, N={net.N}, M={net.M}")
    print(f"parameters: {sum(p.numel() for p in net.parameters()):,}")

    if args.dst is not None:
        args.dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), args.dst)
        print(f"wrote converted state dict -> {args.dst}")

        verify = ShiftLIC(variant=net.variant, N=net.N, M=net.M)
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
