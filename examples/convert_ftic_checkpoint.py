"""Convert an upstream ICLR2024-FTIC checkpoint to compressai layout.

Loads the published FTIC weight file (e.g. ``ckpt_0018.pth`` from the
ICLR2024-FTIC repo,
https://github.com/qingshi9974/ICLR2024-FTIC), translates it to compressai's
containerized module layout, and writes a state dict that
``compressai.models.ftic.FrequencyAwareTransFormer.from_state_dict`` can load
directly. Optionally reports forward-pass sanity numbers (PSNR / bpp) on a
synthetic input.

The upstream-vs-compressai key differences handled by
``convert_upstream_ftic_state_dict`` are:

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
- Upstream ``GsnConditionalLocScaleShift`` wraps a
  ``ContinuousIndexedGaussianConditional`` under ``_entropy_model.``;
  compressai's port subclasses ``GaussianConditional`` directly. We strip the
  ``_entropy_model.`` prefix and drop upstream-only bookkeeping buffers
  (``_indexes_table``, ``lower_bound_zero.bound``, ``upper_bound_mean.bound``,
  ``upper_bound_scale.bound``) that have no equivalent in the compressai
  shifted Gaussian.
- ``gaussian_conditional.scale_table`` is rebuilt at construction time from
  ``(min_scale, scale_max, num_scales)`` and isn't stored in the upstream
  checkpoint; ``from_state_dict`` injects the freshly-built table before
  loading.

All renames live in ``convert_upstream_ftic_state_dict``; this script is a
thin CLI around it.

Example::

    python examples/convert_ftic_checkpoint.py \\
        --src candidate/FTIC/ckpt_0018.pth \\
        --dst /tmp/ftic_compressai.pth \\
        --smoke
"""

from __future__ import annotations

import argparse

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from torch import Tensor

from compressai.models.ftic import FrequencyAwareTransFormer

# ----------------------------------------------------------------------------
# Upstream -> compressai state-dict conversion.
#
# Lives here (not in compressai/models/ftic.py) so the model module stays a
# clean compressai-native definition — ``FrequencyAwareTransFormer.from_state_dict``
# only loads already-converted state dicts. Run this script once to translate a
# published upstream checkpoint into compressai layout, then load the result via
# ``from_state_dict``.
# ----------------------------------------------------------------------------


_FAT_BLOCK_RENAMES = {
    "conv1_1": "conv1",
    "conv1_2": "conv2",
}

_TRANS_BLOCK_RENAMES = {
    "fm": "frequency_modulation",
    "attns": "branch_attentions",
}

_TCA_LAYER_RENAMES = {
    "q1": "q_proj",
    "k1": "k_proj",
    "v1": "v_proj",
    "attn": "attention",
}

_GAUSSIAN_DROP_SUFFIXES = (
    "_indexes_table",
    "lower_bound_zero.bound",
    "upper_bound_mean.bound",
    "upper_bound_scale.bound",
)


def _is_upstream_layout(state_dict: Dict[str, Tensor]) -> bool:
    """Detect whether ``state_dict`` follows the upstream FLIC layout."""
    return (
        "g_a.0.conv1.weight" in state_dict
        and "g_a.input_block.conv1.weight" not in state_dict
    )


def _rename_fat_block_inner(suffix: str) -> str:
    """Translate a sub-key inside a FAT_Block to compressai naming."""
    head, _, rest = suffix.partition(".")
    if head == "trans_block":
        sub_head, _, sub_rest = rest.partition(".")
        sub_head = _TRANS_BLOCK_RENAMES.get(sub_head, sub_head)
        new_rest = f"{sub_head}.{sub_rest}" if sub_rest else sub_head
        return f"frequency_attention.{new_rest}"
    if head in _FAT_BLOCK_RENAMES:
        new_head = _FAT_BLOCK_RENAMES[head]
        return f"{new_head}.{rest}" if rest else new_head
    return suffix


def _rename_transform_block(
    prefix: str,
    indices: Sequence[int],
    state_dict: Dict[str, Tensor],
) -> Tuple[Dict[str, Tensor], Tuple[int, ...]]:
    """Map a flat ``Sequential`` transform (``g_a``/``g_s``/``h_a``/...) onto
    the named sub-modules used by the compressai port.

    Returns the renamed sub-state-dict plus the discovered config (number of
    FAT_Blocks per stage). The number of stages is inferred from the data: each
    block is detected by the presence of a ``trans_block.qkv.weight`` key, and
    consecutive runs of blocks form a stage; the index immediately following
    each run is the stage tail. The first non-block index is always the
    ``input_block``.
    """
    sorted_indices = sorted(indices)
    if not sorted_indices:
        raise ValueError(f"no indices found under prefix {prefix!r}")

    block_flags = [
        f"{prefix}.{idx}.trans_block.qkv.weight" in state_dict for idx in sorted_indices
    ]
    if block_flags[0]:
        raise ValueError(f"unexpected FAT_Block at {prefix}.0; input_block missing")

    block_indices: List[List[int]] = []
    tail_indices: List[int] = []
    current_blocks: List[int] = []
    for idx, is_block in zip(sorted_indices[1:], block_flags[1:]):
        if is_block:
            current_blocks.append(idx)
        else:
            block_indices.append(current_blocks)
            tail_indices.append(idx)
            current_blocks = []
    if current_blocks:
        raise ValueError(
            f"trailing FAT_Blocks at {prefix} have no following tail; "
            f"unfinished stage indices: {current_blocks}"
        )

    config = tuple(len(stage) for stage in block_indices)

    out: Dict[str, Tensor] = {}

    def emit(old_idx: int, new_subprefix: str, is_block: bool) -> None:
        old_prefix = f"{prefix}.{old_idx}."
        new_prefix = f"{prefix}.{new_subprefix}."
        for old_key, value in state_dict.items():
            if not old_key.startswith(old_prefix):
                continue
            suffix = old_key[len(old_prefix) :]
            if is_block:
                suffix = _rename_fat_block_inner(suffix)
            out[new_prefix + suffix] = value

    emit(sorted_indices[0], "input_block", is_block=False)
    if len(block_indices) == 1:
        # Hyper transforms expose a single stage as ``stage`` rather than ``stage1``.
        for block_idx, source_idx in enumerate(block_indices[0]):
            emit(source_idx, f"stage.blocks.{block_idx}", is_block=True)
        emit(tail_indices[0], "stage.tail", is_block=False)
    else:
        for stage_idx, (stage_blocks, tail_idx) in enumerate(
            zip(block_indices, tail_indices), start=1
        ):
            for block_idx, source_idx in enumerate(stage_blocks):
                emit(source_idx, f"stage{stage_idx}.blocks.{block_idx}", is_block=True)
            emit(tail_idx, f"stage{stage_idx}.tail", is_block=False)

    return out, config


def _rename_tca(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Map ``tca.TCA.*`` / ``tca.{hyper_trans,entropy_parameters_net}.*`` onto
    the compressai naming."""
    out: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith("tca."):
            continue
        rest = key[len("tca.") :]
        if rest == "TCA.start_token":
            # Defined upstream but never used by ``TCA.forward``; compressai drops it.
            continue
        if rest.startswith("TCA."):
            sub = rest[len("TCA.") :]
            if sub.startswith("layers."):
                # tca.TCA.layers.{i}.{name}.{...}
                parts = sub.split(".")
                # parts = ["layers", i, name, ...]
                idx = parts[1]
                name = parts[2]
                tail = ".".join(parts[3:])
                if name == "cpe" and len(parts) >= 5 and parts[3] == "0":
                    # cpe.0.{...} -> positional_encoding.{...}
                    tail = ".".join(parts[4:])
                    new_sub = f"layers.{idx}.positional_encoding"
                else:
                    name = _TCA_LAYER_RENAMES.get(name, name)
                    new_sub = f"layers.{idx}.{name}"
                if tail:
                    new_sub = f"{new_sub}.{tail}"
                out[f"tca.tca.{new_sub}"] = value
            else:
                out[f"tca.tca.{sub}"] = value
        elif rest.startswith("hyper_trans."):
            sub = rest[len("hyper_trans.") :]
            if sub == "weight":
                # nn.Linear weight (out, in) -> nn.Conv2d 1x1 weight (out, in, 1, 1)
                out["tca.hyper_trans.weight"] = value.view(
                    value.size(0), value.size(1), 1, 1
                )
            elif sub == "bias":
                out["tca.hyper_trans.bias"] = value
            else:
                out[f"tca.hyper_trans.{sub}"] = value
        else:
            out[f"tca.{rest}"] = value
    return out


def _rename_gaussian_conditional(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Strip the ``_entropy_model.`` wrapper prefix and drop upstream-only
    bookkeeping buffers."""
    out: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith("gaussian_conditional."):
            continue
        sub = key[len("gaussian_conditional.") :]
        if sub.startswith("_entropy_model."):
            sub = sub[len("_entropy_model.") :]
        if any(sub.endswith(suffix) for suffix in _GAUSSIAN_DROP_SUFFIXES):
            continue
        out[f"gaussian_conditional.{sub}"] = value
    return out


def convert_upstream_ftic_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Convert an upstream FTIC checkpoint to the compressai naming.

    Handles the renames documented inline:
    flat ``Sequential`` transforms -> ``input_block`` / ``stage{,1,2,3}.blocks`` /
    ``stage{,1,2,3}.tail``; ``conv1_1`` / ``conv1_2`` -> ``conv1`` / ``conv2``
    inside each FAT_Block; ``trans_block`` -> ``frequency_attention`` (with
    ``attns`` -> ``branch_attentions`` and ``fm`` -> ``frequency_modulation``);
    ``tca.TCA.*`` -> ``tca.tca.*`` (with ``q1``/``k1``/``v1`` ->
    ``q_proj``/``k_proj``/``v_proj``, ``cpe.0`` -> ``positional_encoding``,
    ``attn.proj`` -> ``attention.proj``, drops the unused ``start_token``
    parameter, and reshapes the ``hyper_trans`` Linear weight into a 1x1 Conv2d
    weight); ``gaussian_conditional._entropy_model.*`` ->
    ``gaussian_conditional.*`` (drops ``_indexes_table`` and the unused
    ``lower_bound_zero`` / ``upper_bound_*`` LowerBound buffers).
    """
    indices: Dict[str, List[int]] = {
        prefix: sorted(
            {
                int(key.split(".")[1])
                for key in state_dict
                if key.startswith(prefix + ".")
            }
        )
        for prefix in ("g_a", "g_s", "h_a", "h_mean_s", "h_scale_s")
    }
    out: Dict[str, Tensor] = {}
    for prefix in ("g_a", "g_s", "h_a", "h_mean_s", "h_scale_s"):
        renamed, _ = _rename_transform_block(prefix, indices[prefix], state_dict)
        out.update(renamed)
    out.update(_rename_tca(state_dict))
    out.update(_rename_gaussian_conditional(state_dict))
    for key, value in state_dict.items():
        if key.startswith("entropy_bottleneck."):
            out[key] = value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the upstream FTIC checkpoint (e.g. ckpt_0018.pth).",
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
        state_dict = convert_upstream_ftic_state_dict(state_dict)
        print(f"converted to compressai layout: {len(state_dict)} keys")

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
