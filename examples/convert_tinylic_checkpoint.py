"""Convert an upstream TinyLIC checkpoint to compressai layout.

The upstream checkpoints (NJU Box, Q1-Q8, https://github.com/lumingzzz/TinyLIC)
carry the same parameter shapes as this fork's
:class:`compressai.models.tinylic.TinyLIC`. Two key differences are handled by
``convert_upstream_tinylic_state_dict`` (this script is a thin CLI around it):

- The optional ``module.`` prefix from ``torch.nn.DataParallel`` is stripped.
- The codec's top-level entropy parameters (``entropy_parameters_*`` /
  ``cc_transforms.*`` / ``sc_transform_*`` / ``gaussian_conditional.*``) are
  rewritten to live under ``latent_codec.*`` so that
  :func:`TinyLIC.from_state_dict` can load them directly.

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
from typing import Dict

import torch

from torch import Tensor

from compressai.models.tinylic import TinyLIC

# ----------------------------------------------------------------------------
# Upstream -> compressai state-dict conversion.
#
# Lives here (not in compressai/models/tinylic.py) so the model module stays a
# clean compressai-native definition -- ``TinyLIC.from_state_dict`` only loads
# already-converted state dicts. Run this script once to translate a published
# upstream checkpoint into compressai layout, then load the result via
# ``from_state_dict``.
# ----------------------------------------------------------------------------

# Top-level prefixes that the staged checkerboard codec owns. Upstream stores
# them at the model root; compressai keeps them under ``latent_codec.*``.
_CODEC_PREFIXES = (
    "entropy_parameters_",
    "cc_transforms.",
    "sc_transform_",
    "gaussian_conditional.",
)


def convert_upstream_tinylic_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Translate an upstream TinyLIC state dict into compressai layout.

    Strips the optional ``module.`` prefix and reparents the codec's top-level
    entropy keys under ``latent_codec.*``. Keys that are already namespaced
    (e.g. a state dict produced by this fork) pass through unchanged.
    """
    converted: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
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
    state_dict = convert_upstream_tinylic_state_dict(state_dict)
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
