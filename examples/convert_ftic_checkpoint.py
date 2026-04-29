"""Convert an upstream ICLR2024-FTIC checkpoint to compressai layout.

Loads the published FTIC weight file (e.g. ``ckpt_0018.pth`` from the
ICLR2024-FTIC repo,
https://github.com/qingshi9974/ICLR2024-FTIC), and writes a state dict that
``compressai.models.FrequencyAwareTransFormer.from_state_dict`` can load
directly. Optionally reports forward-pass sanity numbers (PSNR / bpp) on a
synthetic input.

The upstream-vs-compressai key differences handled by
``compressai.models.ftic.convert_upstream_state_dict`` are:

- ``g_a`` / ``g_s`` / ``h_a`` / ``h_mean_s`` / ``h_scale_s`` are flat
  ``nn.Sequential`` modules; we group their indices back into named sub-stages
  (``input_block``, ``stage{1,2,3}.blocks.{i}``, ``stage{1,2,3}.tail`` for the
  six-stage transforms; ``input_block``, ``stage.blocks.{i}``, ``stage.tail``
  for the single-stage hyper transforms).
- Inside each FAT_Block: ``conv1_1`` / ``conv1_2`` -> ``conv1`` / ``conv2``;
  ``trans_block`` -> ``frequency_attention`` (with ``attns`` ->
  ``branch_attentions`` and ``fm`` -> ``frequency_modulation``).
- ``tca.TCA.*`` -> ``tca.tca.*``; ``q1`` / ``k1`` / ``v1`` -> ``q_proj`` /
  ``k_proj`` / ``v_proj``; ``cpe.0`` (``ModuleList`` wrapper) ->
  ``positional_encoding``; ``attn.proj`` -> ``attention.proj``.
- Upstream ``tca.TCA.start_token`` is unused by ``TCA.forward`` (the
  start token is always synthesized via ``start_token_from_hyperprior``);
  compressai's port drops the parameter, so it is discarded here too.
- ``tca.hyper_trans`` is an ``nn.Linear`` upstream (forward flattens to tokens)
  but a 1x1 ``nn.Conv2d`` in compressai (mathematically equivalent); the weight
  is reshaped from ``(640, 640)`` to ``(640, 640, 1, 1)``.
- Upstream ``GsnConditionalLocScaleShift`` wraps a ``ContinuousIndexedGaussianConditional``
  under ``_entropy_model.``; compressai's port subclasses ``GaussianConditional``
  directly. We strip the ``_entropy_model.`` prefix and drop upstream-only
  bookkeeping buffers (``_indexes_table``, ``lower_bound_zero.bound``,
  ``upper_bound_mean.bound``, ``upper_bound_scale.bound``) that have no
  equivalent in the compressai shifted Gaussian.
- ``gaussian_conditional.scale_table`` is rebuilt at construction time from
  ``(min_scale, scale_max, num_scales)`` and isn't stored in the upstream
  checkpoint; ``from_state_dict`` injects the freshly-built table before
  loading.

All renames and the per-buffer dropping are handled inside
``FrequencyAwareTransFormer.from_state_dict`` /
``compressai.models.ftic.convert_upstream_state_dict``; this script is a thin
CLI around them.

Example::

    python examples/convert_ftic_checkpoint.py \\
        --src candidate/ICLR2024-FTIC/ckpt_0018.pth \\
        --dst /tmp/ftic_compressai.pth \\
        --smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from compressai.models import FrequencyAwareTransFormer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help=(
            "Path to the upstream FTIC checkpoint "
            "(e.g. ckpt_0018.pth)."
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

    net = FrequencyAwareTransFormer.from_state_dict(state_dict)
    net.eval()
    print(
        "variant: "
        f"M={net.M}, hyper_channels={net.hyper_channels}, "
        f"num_slices={net.num_slices}, "
        f"feature_dims={net.feature_dims}, "
        f"config={net.config}, num_heads={net.num_heads}, "
        f"window_size={net.window_size}, fm_window_size={net.fm_window_size}, "
        f"tca_depth={net.tca_depth}, tca_ratio={net.tca_ratio}"
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
        y_bpp = -torch.log2(out["likelihoods"]["y"]).sum().item() / n_pix
        z_bpp = -torch.log2(out["likelihoods"]["z"]).sum().item() / n_pix
        print(
            f"smoke: PSNR={psnr:.2f}dB  "
            f"y_bpp={y_bpp:.4f}  z_bpp={z_bpp:.4f}  "
            f"total_bpp={y_bpp + z_bpp:.4f}"
        )


if __name__ == "__main__":
    main()
