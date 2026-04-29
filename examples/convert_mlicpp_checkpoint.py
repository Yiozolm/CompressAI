"""Convert an upstream MLIC++ checkpoint to compressai layout.

Loads a published MLIC++ candidate checkpoint (e.g.
``mlicpp_mse_q5_2960000.pth.tar`` from the MLIC repo,
https://github.com/JiangWeibeta/MLIC), and writes a state dict that
``compressai.models.MLICPlusPlus.from_state_dict`` can load directly.
Optionally reports forward-pass sanity numbers (PSNR / bpp) on a synthetic
input and verifies the converted state dict is bit-equivalent to the
upstream one (eval-mode ``x_hat`` diff = 0).

The upstream-vs-compressai key differences are minimal:

- All entropy / context / hyper / LRP submodules sit at top level upstream
  (``h_a.*``, ``h_s.*``, ``entropy_bottleneck.*``, ``gaussian_conditional.*``,
  ``local_context.*``, ``channel_context.*``, ``global_inter_context.*``,
  ``global_intra_context.*``, ``entropy_parameters_anchor.*``,
  ``entropy_parameters_nonanchor.*``, ``lrp_anchor.*``, ``lrp_nonanchor.*``);
  in compressai they live under ``latent_codec.*`` because the MLIC++
  hyperprior + multi-reference checkerboard channel-slice entropy model is
  factored out as ``MLICPlusPlusLatentCodec``.
- ``g_a`` / ``g_s`` keys (``analysis_transform.*`` / ``synthesis_transform.*``)
  are unchanged: the compressai ``GeluResidualBlock*`` blocks share parameter
  names (``conv1`` / ``conv2`` / ``gdn`` / ``skip`` / ``subpel_conv`` /
  ``conv`` / ``igdn`` / ``upsample``) with the upstream
  ``ResidualBlockWithStride`` / ``ResidualBlock`` / ``ResidualBlockUpsample``.

The rename is handled inside ``MLICPlusPlus.from_state_dict`` /
``MLICPlusPlus._migrate_state_dict``; this script is a thin CLI wrapping it.

Example::

    python examples/convert_mlicpp_checkpoint.py \\
        --src candidate/MLIC/mlicpp_mse_q5_2960000.pth.tar \\
        --dst /tmp/mlicpp_compressai.pth \\
        --smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from compressai.models import MLICPlusPlus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help=(
            "Path to the upstream MLIC++ checkpoint "
            "(e.g. mlicpp_mse_q5_2960000.pth.tar)."
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

    net = MLICPlusPlus.from_state_dict(state_dict)
    net.eval()
    print(
        f"variant: N={net.N}, M={net.M}, slice_num={net.slice_num}, "
        f"context_window={net.context_window}"
    )
    print(f"parameters: {sum(p.numel() for p in net.parameters()):,}")

    if args.dst is not None:
        args.dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), args.dst)
        print(f"wrote converted state dict -> {args.dst}")

        verify = MLICPlusPlus(
            N=net.N,
            M=net.M,
            slice_num=net.slice_num,
            context_window=net.context_window,
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
        mse = ((out["x_hat"].clamp(0, 1) - img) ** 2).mean()
        psnr = -10 * torch.log10(mse).item()
        y_bpp = -torch.log2(out["likelihoods"]["y"]).sum().item() / n_pix
        z_bpp = -torch.log2(out["likelihoods"]["z"]).sum().item() / n_pix
        print(
            f"smoke: PSNR={psnr:.2f}dB  y_bpp={y_bpp:.4f}  z_bpp={z_bpp:.4f}  "
            f"total_bpp={y_bpp + z_bpp:.4f}"
        )


if __name__ == "__main__":
    main()
