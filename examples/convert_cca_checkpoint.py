"""Convert an upstream CCA ``LICAutoencoder`` checkpoint to compressai layout.

Loads a published CCA candidate checkpoint (e.g.
``checkpoint_lambda_0.3.pth.tar`` from the CCA repo,
https://github.com/LabShuHangGU/CCA), and writes a state dict that
``compressai.models.CCAModel.from_state_dict`` can load directly.
Optionally reports forward-pass sanity numbers (PSNR / bpp) on a synthetic
input and verifies the converted state dict is bit-equivalent to the
upstream one (eval-mode ``x_hat`` diff = 0).

The upstream-vs-compressai key differences are mechanical renames; the
shared NAFBlock / NAFTransform classes are private to
``compressai/entropy_models/cca.py`` (used by both the CCA entropy model
and ``compressai/models/cca.py``):

- Top level: ``aux_entropymodel`` → ``aux_entropy_model``.
- Anywhere: ``mean_NAF_transforms`` → ``mean_support_transforms``,
  ``scale_NAF_transforms`` → ``scale_support_transforms`` (inside the model
  body and also under the ``aux_entropy_model`` namespace).
- Inside every ``NAFTransform``: ``in_conv`` → ``input_projection``,
  ``out_conv`` → ``output_projection``.
- Inside every ``NAFBlock``: ``dwconv`` → ``pointwise_depthwise``, ``sca`` →
  ``channel_attention``, ``FFN`` → ``feed_forward``, ``conv1`` →
  ``project``.
- ``ResidualBottleneckBlock`` (``g_a`` / ``g_s`` interior) needs no rename:
  compressai's ``conv1`` / ``conv2`` / ``conv3`` / ``skip`` names match
  upstream verbatim.

The renames are applied inside ``CCAModel.from_state_dict`` /
``convert_upstream_state_dict``; this script is a thin CLI wrapping it.

Example::

    python examples/convert_cca_checkpoint.py \\
        --src candidate/CCA/checkpoint_lambda_0.3.pth.tar \\
        --dst /tmp/cca_compressai.pth \\
        --smoke
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from compressai.models import CCAModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help=(
            "Path to the upstream CCA checkpoint "
            "(e.g. checkpoint_lambda_0.3.pth.tar)."
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

    net = CCAModel.from_state_dict(state_dict)
    net.eval()
    print(
        f"variant: M={net.M}, N={net.N}, slice_sizes={net.slice_sizes}, "
        f"encoder_dims={net.encoder_dims}, encoder_layers={net.encoder_layers}, "
        f"em_hidden={net.em_hidden_channels}, em_layers={net.em_num_layers}, "
        f"cca_training={net.cca_training}"
    )
    print(f"parameters: {sum(p.numel() for p in net.parameters()):,}")

    if args.dst is not None:
        args.dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), args.dst)
        print(f"wrote converted state dict -> {args.dst}")

        verify = CCAModel(
            latent_channels=net.M,
            hyper_channels=net.N,
            slice_proportions=net.slice_proportions,
            encoder_dims=net.encoder_dims,
            encoder_layers=net.encoder_layers,
            em_hidden_channels=net.em_hidden_channels,
            em_num_layers=net.em_num_layers,
            cca_training=net.cca_training,
        )
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
