"""Convert an upstream CMIC checkpoint to compressai layout.

Loads a published CMIC weight file (e.g. ``0.0017checkpoint_best.pth.tar`` from
the candidate ``CMIC_AuxT`` reference implementation, Content-Aware Mamba for
Learned Image Compression, ICLR 2026), and writes a state dict that
``compressai.models.cmic.CMIC.from_state_dict`` can load directly. Optionally
reports forward-pass sanity numbers (PSNR / bpp) on a synthetic input.

The upstream-vs-compressai key differences (all handled inside
``convert_upstream_cmic_state_dict``; this script is a thin CLI around it):

- candidate-side DWT/IDWT haar-coefficient buffers are dropped (compressai
  rebuilds equivalent ``pytorch_wavelets`` buffers in :class:`DWT2D` /
  :class:`IDWT2D`);
- ``OLP`` renamed to ``olp`` inside ``WLS`` / ``iWLS``;
- inside CMIC stage blocks (``g_a.g2`` / ``g_a.g3`` / ``g_s.g1`` / ``g_s.g2``):
  ``residual_group.layers.{j}`` -> ``blocks.{j}``, ``wqkv`` / ``win_mhsa`` ->
  ``window_attention.qkv`` / ``window_attention``, ``convffn{i}.conv`` ->
  ``feed_forward{i}.depthwise``, ``convffn{i}`` -> ``feed_forward{i}``,
  ``assm`` -> ``content_model`` with ``selectiveScan`` flattened,
  ``in_proj.0`` / ``CPE.0`` -> ``in_proj`` / ``cpe``, ``cal_embedding`` ->
  ``prompt_proj``;
- candidate-side ``win_mhsa.relative_position_bias_table`` is zeroed because the
  upstream forward never adds it while compressai's ``WindowAttention`` does;
- hyperprior layout: ``latent_codec.hyper.{h_a,h_s}`` hoisted to
  ``latent_codec.{h_a,h_s}`` and ``latent_codec.hyper.entropy_bottleneck`` ->
  ``latent_codec.z.entropy_bottleneck``;
- spatial context: ``ly{1,2}`` -> ``layer{1,2}``, ``mixer.depth_conv`` ->
  ``mixer.masked_conv``;
- ``norm2`` -> ``norm`` for ``GatedTransformCNN`` / aggregation blocks (CMIC
  block ``norm{1..4}`` are preserved);
- ``mixer.adaptor`` -> ``mixer.skip`` for residual depthwise mixers.

Example::

    python examples/convert_cmic_checkpoint.py \\
        --src candidate/CMIC/0.0017checkpoint_best.pth.tar \\
        --dst /tmp/cmic_compressai.pth \\
        --smoke
"""

from __future__ import annotations

import argparse
import re

from pathlib import Path
from typing import Dict

import torch

from torch import Tensor

from compressai.models.cmic import CMIC

# ----------------------------------------------------------------------------
# Upstream → compressai state-dict conversion.
#
# Lives here (not in compressai/models/cmic.py) so the model module stays a
# clean compressai-native definition — ``CMIC.from_state_dict`` only loads
# already-converted state dicts. Run this script once to translate a published
# upstream checkpoint into compressai layout, then load the result via
# ``from_state_dict``.
# ----------------------------------------------------------------------------

_CMIC_STAGE_PREFIXES = ("g_a.g2.", "g_a.g3.", "g_s.g1.", "g_s.g2.")


def _is_upstream_layout(state_dict: Dict[str, Tensor]) -> bool:
    """Detect the published CMIC checkpoint layout (``residual_group.layers.*``
    block naming or the ``latent_codec.hyper.*`` nested hyperprior)."""
    return any(
        ".residual_group.layers." in key or key.startswith("latent_codec.hyper.")
        for key in state_dict
    )


def convert_upstream_cmic_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Translate an upstream ``CMIC_AuxT`` checkpoint to compressai layout.

    See the module docstring for the full list of key differences. Keys already
    in compressai layout pass through unchanged, so calling this on a
    round-tripped ``state_dict`` is a safe no-op.
    """
    new_state_dict: Dict[str, Tensor] = {}

    for key, value in state_dict.items():
        if re.search(r"AuxT_(?:enc|dec)\.\d+\.dwt\.w_(?:ll|lh|hl|hh)$", key):
            continue
        if re.search(r"AuxT_(?:enc|dec)\.\d+\.(?:dwt|idwt)\.filters$", key):
            continue

        new_key = key
        new_value = value

        new_key = re.sub(r"(AuxT_(?:enc|dec)\.\d+)\.OLP\.", r"\1.olp.", new_key)

        is_cmic_stage = any(prefix in key for prefix in _CMIC_STAGE_PREFIXES)
        if is_cmic_stage:
            new_key = re.sub(
                r"\.residual_group\.layers\.(\d+)\.",
                r".blocks.\1.",
                new_key,
            )
            new_key = re.sub(
                r"\.blocks\.(\d+)\.wqkv\.",
                r".blocks.\1.window_attention.qkv.",
                new_key,
            )
            new_key = re.sub(
                r"\.blocks\.(\d+)\.win_mhsa\.",
                r".blocks.\1.window_attention.",
                new_key,
            )
            new_key = re.sub(
                r"\.convffn(\d)\.conv\.",
                r".feed_forward\1.depthwise.",
                new_key,
            )
            new_key = re.sub(r"\.convffn(\d)\.", r".feed_forward\1.", new_key)
            if ".assm." in new_key:
                new_key = new_key.replace(".assm.", ".content_model.")
                new_key = new_key.replace(
                    ".content_model.selectiveScan.", ".content_model."
                )
                new_key = re.sub(
                    r"\.content_model\.in_proj\.0\.",
                    r".content_model.in_proj.",
                    new_key,
                )
                new_key = re.sub(
                    r"\.content_model\.CPE\.0\.",
                    r".content_model.cpe.",
                    new_key,
                )
                new_key = new_key.replace(
                    ".content_model.cal_embedding.",
                    ".content_model.prompt_proj.",
                )

        new_key = new_key.replace(
            "latent_codec.hyper.entropy_bottleneck.",
            "latent_codec.z.entropy_bottleneck.",
        )
        new_key = new_key.replace("latent_codec.hyper.h_a.", "latent_codec.h_a.")
        new_key = new_key.replace("latent_codec.hyper.h_s.", "latent_codec.h_s.")

        new_key = re.sub(
            r"context_prediction\.ly(\d)\.",
            r"context_prediction.layer\1.",
            new_key,
        )
        if "context_prediction.layer" in new_key:
            new_key = new_key.replace("mixer.depth_conv.", "mixer.masked_conv.")

        if not re.match(r"g_[as]\.g\d+\.blocks\.\d+\.norm[1234]\.", new_key):
            new_key = new_key.replace(".norm2.", ".norm.")

        new_key = new_key.replace(".mixer.adaptor.", ".mixer.skip.")

        if new_key.endswith(".window_attention.relative_position_bias_table"):
            new_value = torch.zeros_like(value)

        new_state_dict[new_key] = new_value

    return new_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help=(
            "Path to the upstream CMIC checkpoint "
            "(e.g. 0.0017checkpoint_best.pth.tar)."
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

    if _is_upstream_layout(state_dict):
        state_dict = convert_upstream_cmic_state_dict(state_dict)
        print(f"converted upstream layout -> {len(state_dict)} keys")

    net = CMIC.from_state_dict(state_dict)
    net.eval()
    print(
        "variant: "
        f"N={net.N}, M={net.M}, groups={net.groups}, "
        f"stage_dims={net.stage_dims}, stage_depths={net.stage_depths}, "
        f"num_heads={net.num_heads}, window_size={net.window_size}"
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
        img = (
            torch.stack(
                [
                    0.5 + 0.3 * torch.sin(8 * xs),
                    0.5 + 0.3 * torch.sin(8 * ys),
                    0.5 + 0.3 * torch.cos(8 * (xs + ys)),
                ],
                dim=0,
            )
            .unsqueeze(0)
            .clamp(0, 1)
        )

        with torch.no_grad():
            out = net(img)
        n_pix = height * width
        psnr = -10 * torch.log10(((out["x_hat"].clamp(0, 1) - img) ** 2).mean()).item()
        y_bpp = -torch.log2(out["likelihoods"]["y"]).sum().item() / n_pix
        z_bpp = -torch.log2(out["likelihoods"]["z"]).sum().item() / n_pix
        print(
            f"smoke: PSNR={psnr:.2f}dB  y_bpp={y_bpp:.4f}  z_bpp={z_bpp:.4f}  "
            f"total_bpp={y_bpp + z_bpp:.4f}"
        )


if __name__ == "__main__":
    main()
