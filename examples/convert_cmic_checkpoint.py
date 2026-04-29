"""Convert an upstream CMIC checkpoint to compressai layout.

Loads a published CMIC weight file (e.g. ``0.0017checkpoint_best.pth.tar`` from
the candidate ``CMIC_AuxT`` reference implementation), and writes a state dict
that ``compressai.models.CMIC.from_state_dict`` can load directly with
``strict=True``. Optionally reports forward-pass sanity numbers (PSNR / bpp) on
a synthetic input.

The upstream-vs-compressai key differences (``residual_group.layers`` ->
``blocks``; ``wqkv``/``win_mhsa`` -> ``window_attention.qkv``/
``window_attention``; ``assm`` -> ``content_model`` with ``selectiveScan``
flattened, ``in_proj.0``/``CPE.0`` collapsed and ``cal_embedding`` ->
``prompt_proj``; ``convffn{i}`` -> ``feed_forward{i}`` with ``conv`` ->
``depthwise``; ``OLP`` -> ``olp``; spatial-context ``ly{1,2}`` ->
``layer{1,2}`` with ``depth_conv`` -> ``masked_conv``; ``norm2`` -> ``norm``
for ``GatedTransformCNN``/``Param_Gated``/``Param_Agg_Block``;
``mixer.adaptor`` -> ``mixer.skip``; ``latent_codec.hyper.{h_a,h_s}`` hoisted
to ``latent_codec.{h_a,h_s}`` and ``latent_codec.hyper.entropy_bottleneck`` ->
``latent_codec.z.entropy_bottleneck``; the unused candidate-side
``win_mhsa.relative_position_bias_table`` is zeroed because compressai's
``WindowAttention`` actually adds it; candidate-side haar DWT/IDWT buffers are
dropped in favour of compressai's ``pytorch_wavelets`` initialisation) are all
handled inside ``CMIC._convert_upstream_state_dict`` and
``CMIC.from_state_dict``; this script is a thin CLI around them.

Example::

    python examples/convert_cmic_checkpoint.py \\
        --src candidate/CMIC/0.0017checkpoint_best.pth.tar \\
        --dst /tmp/cmic_compressai.pth \\
        --smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from compressai.models import CMIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the upstream CMIC checkpoint (e.g. *.pth.tar).",
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

    net = CMIC.from_state_dict(state_dict)
    net.eval()
    print(
        "variant: "
        f"N={net.N}, M={net.M}, groups={net.groups}, "
        f"stage_dims={net.stage_dims}, stage_depths={net.stage_depths}, "
        f"num_heads={net.num_heads}, d_state={net.d_state}, "
        f"window_size={net.window_size}, cluster_num={net.cluster_num}"
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
            f"smoke: PSNR={psnr:.2f}dB  y_bpp={y_bpp:.4f}  z_bpp={z_bpp:.4f}  "
            f"total_bpp={y_bpp + z_bpp:.4f}"
        )


if __name__ == "__main__":
    main()
