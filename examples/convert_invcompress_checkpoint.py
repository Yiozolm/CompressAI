"""Convert an upstream InvCompress checkpoint to compressai layout.

Loads the published candidate weight file (e.g.
``checkpoint_best_loss-c32dfce8.pth.tar`` from the InvCompress repo,
https://github.com/xyq7/InvCompress), translates it to compressai's
FrEIA-backed module layout, and writes a state dict that
``compressai.models.invcompress.InvCompress.from_state_dict`` can load
directly. Optionally reports forward-pass sanity numbers (PSNR / bpp) on a
synthetic input.

The upstream-vs-compressai key differences handled by
``convert_upstream_invcompress_state_dict`` are:

- ``inv.operations.{i}.weight`` (a trainable ``[C, C]`` matrix in upstream's
  custom ``InvertibleConv1x1``) is mapped to FrEIA's ``Fixed1x1Conv`` buffers
  ``inv.operations.{i}.backend.{M, M_inv, logDetM}``. The transform is
  numerically lossless for inference; ``Fixed1x1Conv`` is non-trainable in
  FrEIA 0.2 so re-finetuning these convs is not supported by the conversion.
- The four parallel coupling sub-bottlenecks ``G1`` / ``G2`` / ``H1`` / ``H2``
  are fused into FrEIA's two ``GLOWCouplingBlock`` subnets (output channels
  ordered ``[scale; translation]`` per the SIGMOID clamp activation). The
  fusion stacks ``conv1`` along the output dim and places ``conv2`` /
  ``conv3`` on the block diagonal with zero cross-coupling, preserving the
  upstream forward bit-for-bit.
- ``enh.*`` and ``attention.*`` keys already match between layouts.

All renames live in ``convert_upstream_invcompress_state_dict``; this script
is a thin CLI around it.

Example::

    python examples/convert_invcompress_checkpoint.py \\
        --src candidate/InvCompress/checkpoint_best_loss-c32dfce8.pth.tar \\
        --dst /tmp/invcompress_compressai.pth \\
        --smoke
"""

from __future__ import annotations

import argparse

from pathlib import Path
from typing import Dict

import torch

from torch import Tensor

from compressai.models.invcompress import InvCompress

# ----------------------------------------------------------------------------
# Upstream -> compressai state-dict conversion.
#
# Lives here (not in compressai/models/invcompress.py) so the model module
# stays a clean compressai-native definition — ``InvCompress.from_state_dict``
# only loads already-converted state dicts. Run this script once to translate a
# published upstream checkpoint into compressai layout, then load the result via
# ``from_state_dict``.
# ----------------------------------------------------------------------------


def _is_upstream_invcompress_state_dict(state_dict: Dict[str, Tensor]) -> bool:
    """Detect the upstream InvCompress key layout.

    Upstream stores the invertible 1x1 convs as plain ``inv.operations.{i}.weight``
    matrices and the coupling subnetworks as four parallel bottlenecks
    (``G1`` / ``G2`` / ``H1`` / ``H2``). The compressai-side layout instead
    routes them through the FrEIA ``Fixed1x1Conv`` and ``GLOWCouplingBlock``
    backends (``inv.operations.{i}.backend.{M,M_inv,logDetM}`` and
    ``inv.operations.{i}.backend.subnet{1,2}.conv{1,2,3}.{weight,bias}``).
    """
    for key in state_dict:
        if key.startswith("inv.operations.") and ".G1.conv1.weight" in key:
            return True
    return False


def _block_diagonal_conv(top_left: Tensor, bottom_right: Tensor) -> Tensor:
    """Stack two ``[O, I, k, k]`` conv weights on the diagonal of a single
    ``[2O, 2I, k, k]`` weight, with zero cross-block weights.

    Used to merge upstream's parallel scale/shift bottlenecks into one FrEIA
    subnet without coupling the two halves through the intermediate convs.
    """
    if top_left.shape != bottom_right.shape:
        raise ValueError(
            "block-diagonal merge requires identically shaped weights, got "
            f"{tuple(top_left.shape)} vs {tuple(bottom_right.shape)}"
        )
    out_ch, in_ch, kh, kw = top_left.shape
    merged = top_left.new_zeros(out_ch * 2, in_ch * 2, kh, kw)
    merged[:out_ch, :in_ch] = top_left
    merged[out_ch:, in_ch:] = bottom_right
    return merged


def convert_upstream_invcompress_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Migrate an upstream InvCompress checkpoint to the compressai layout.

    No-op for state dicts that already use the compressai naming.

    The transform fuses the upstream ``G1`` / ``H1`` (resp. ``G2`` / ``H2``)
    bottlenecks into a single FrEIA ``GLOWCouplingBlock`` subnet whose
    output channels are ``[scale; translation]``. ``conv1`` is row-stacked,
    ``conv2`` and ``conv3`` are placed on the block diagonal with zero
    cross-coupling so the merged bottleneck is numerically identical to the
    two parallel bottlenecks (block-diagonal weights + element-wise
    LeakyReLU = no inter-half mixing).

    The trainable upstream ``inv.operations.{i}.weight`` 1x1 conv is mapped
    to the FrEIA ``Fixed1x1Conv`` buffers ``M`` / ``M_inv`` / ``logDetM``
    (non-trainable in FrEIA 0.2; sufficient for inference-time use of the
    published checkpoints).
    """
    if not _is_upstream_invcompress_state_dict(state_dict):
        return state_dict

    converted: Dict[str, Tensor] = {}
    inv_conv_weights: Dict[int, Tensor] = {}
    coupling_buckets: Dict[int, Dict[str, Dict[str, Tensor]]] = {}

    for key, value in state_dict.items():
        if not key.startswith("inv.operations."):
            converted[key] = value
            continue

        # `inv.operations.{idx}.{rest}` — `rest` distinguishes `weight`
        # (1x1 conv) from `{G,H}{1,2}.conv{1,2,3}.{weight,bias}` (coupling).
        _, _, op_idx_str, *rest = key.split(".")
        op_idx = int(op_idx_str)

        if rest == ["weight"]:
            inv_conv_weights[op_idx] = value
            continue

        # Coupling subnet: rest is e.g. ["G1", "conv1", "weight"].
        if len(rest) == 3 and rest[0] in {"G1", "G2", "H1", "H2"}:
            sub_id, conv_name, param_name = rest
            bucket = coupling_buckets.setdefault(op_idx, {})
            sub_bucket = bucket.setdefault(sub_id, {})
            sub_bucket[f"{conv_name}.{param_name}"] = value
            continue

        raise KeyError(f"Unhandled upstream InvCompress key: {key}")

    for op_idx, weight in inv_conv_weights.items():
        if weight.dim() != 2 or weight.shape[0] != weight.shape[1]:
            raise ValueError(
                f"inv.operations.{op_idx}.weight must be a square 2D matrix; "
                f"got shape {tuple(weight.shape)}"
            )
        channels = weight.shape[0]
        view = weight.view(channels, channels, 1, 1)
        # Compute the inverse in float64 for numerical stability and cast back,
        # mirroring upstream's `inverse(weight.double()).float()`.
        weight_inv = (torch.linalg.inv(weight.to(torch.float64)).to(weight.dtype)).view(
            channels, channels, 1, 1
        )
        log_abs_det = torch.slogdet(weight.to(torch.float64))[1].to(weight.dtype)
        prefix = f"inv.operations.{op_idx}.backend"
        converted[f"{prefix}.M"] = view.contiguous()
        converted[f"{prefix}.M_inv"] = weight_inv.contiguous()
        converted[f"{prefix}.logDetM"] = log_abs_det

    for op_idx, sub_buckets in coupling_buckets.items():
        for from_subs, into_subnet in (
            (("G1", "H1"), "subnet1"),
            (("G2", "H2"), "subnet2"),
        ):
            scale_sub = sub_buckets[from_subs[0]]
            shift_sub = sub_buckets[from_subs[1]]

            scale_conv1_w = scale_sub["conv1.weight"]
            shift_conv1_w = shift_sub["conv1.weight"]
            scale_conv1_b = scale_sub["conv1.bias"]
            shift_conv1_b = shift_sub["conv1.bias"]

            scale_conv2_w = scale_sub["conv2.weight"]
            shift_conv2_w = shift_sub["conv2.weight"]
            scale_conv2_b = scale_sub["conv2.bias"]
            shift_conv2_b = shift_sub["conv2.bias"]

            scale_conv3_w = scale_sub["conv3.weight"]
            shift_conv3_w = shift_sub["conv3.weight"]
            scale_conv3_b = scale_sub["conv3.bias"]
            shift_conv3_b = shift_sub["conv3.bias"]

            # conv1: input is shared (split_len{1,2}), output is concat
            # ``[scale_output, shift_output]`` which FrEIA splits into [s, t].
            merged_conv1_w = torch.cat([scale_conv1_w, shift_conv1_w], dim=0)
            merged_conv1_b = torch.cat([scale_conv1_b, shift_conv1_b], dim=0)

            merged_conv2_w = _block_diagonal_conv(scale_conv2_w, shift_conv2_w)
            merged_conv2_b = torch.cat([scale_conv2_b, shift_conv2_b], dim=0)

            merged_conv3_w = _block_diagonal_conv(scale_conv3_w, shift_conv3_w)
            merged_conv3_b = torch.cat([scale_conv3_b, shift_conv3_b], dim=0)

            prefix = f"inv.operations.{op_idx}.backend.{into_subnet}"
            converted[f"{prefix}.conv1.weight"] = merged_conv1_w.contiguous()
            converted[f"{prefix}.conv1.bias"] = merged_conv1_b.contiguous()
            converted[f"{prefix}.conv2.weight"] = merged_conv2_w.contiguous()
            converted[f"{prefix}.conv2.bias"] = merged_conv2_b.contiguous()
            converted[f"{prefix}.conv3.weight"] = merged_conv3_w.contiguous()
            converted[f"{prefix}.conv3.bias"] = merged_conv3_b.contiguous()

    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help=(
            "Path to the upstream InvCompress checkpoint "
            "(e.g. checkpoint_best_loss-c32dfce8.pth.tar)."
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

    if _is_upstream_invcompress_state_dict(state_dict):
        state_dict = convert_upstream_invcompress_state_dict(state_dict)
        print(f"converted to compressai layout: {len(state_dict)} keys")

    net = InvCompress.from_state_dict(state_dict)
    net.eval()
    print(f"variant: N={net.N}, M={net.M}")
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
            f"smoke: PSNR={psnr:.2f}dB  "
            f"y_bpp={y_bpp:.4f}  z_bpp={z_bpp:.4f}  "
            f"total_bpp={y_bpp + z_bpp:.4f}"
        )


if __name__ == "__main__":
    main()
