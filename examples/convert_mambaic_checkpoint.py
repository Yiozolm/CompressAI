"""Convert an upstream MambaIC checkpoint to compressai layout.

Loads the published candidate weight file (e.g. ``checkpoint.pth.tar`` from the
MambaIC repo, https://github.com/AlbertZhangHIT/MambaIC), and writes a state
dict that ``compressai.models.mambaic.MambaIC.from_state_dict`` can load
directly. Optionally reports forward-pass sanity numbers (PSNR / bpp) on a
synthetic input.

The upstream-vs-compressai key differences (all handled inside
``convert_upstream_mambaic_state_dict``; this script is a thin CLI around it):

- Top-level latent-codec submodules (``atten_mean``, ``atten_scale``,
  ``anchor_atten_mean``, ``anchor_atten_scale``, ``cc_*``, ``lrp_transforms``,
  ``context_prediction``, ``context_vss``, ``gaussian_conditional``) move under
  the ``latent_codec.`` namespace.
- The ``nn.Sequential(SWAtten(...))`` wrapper around ``atten_*`` and
  ``anchor_atten_*`` adds an extra ``.0.`` segment that is stripped.
- Upstream stores the same context VSS stages twice — once as
  ``context_vss.{i}.*`` (via the ``ModuleList``) and once as
  ``context_vss_{i+1}.*`` (via separate attributes). The duplicates are
  dropped.
- Inside SWAtten's WMSA the upstream ``embedding_layer`` / ``linear`` /
  ``relative_position_params`` are swapped for compressai's ``attn.qkv`` /
  ``output_proj`` / ``attn.relative_position_bias_table`` (the table permuted
  from ``(heads, 2W-1, 2W-1)`` to ``((2W-1)^2, heads)``), and an identity
  ``attn.proj`` is synthesised since upstream folds it into ``linear``.
- ``Block`` LayerNorms and MLP projections (``ln1`` / ``ln2`` / ``mlp.0`` /
  ``mlp.2``) are renamed to compressai's ``norm1`` / ``norm2`` / ``mlp.fc1`` /
  ``mlp.fc2``.

Example::

    python examples/convert_mambaic_checkpoint.py \\
        --src candidate/MambaIC/checkpoint.pth.tar \\
        --dst /tmp/mambaic_compressai.pth \\
        --smoke
"""

from __future__ import annotations

import argparse
import re

from pathlib import Path
from typing import Dict

import torch

from torch import Tensor

from compressai.models.mambaic import MambaIC

# ----------------------------------------------------------------------------
# Upstream → compressai state-dict conversion.
#
# Lives here (not in compressai/models/mambaic.py) so the model module stays a
# clean compressai-native definition — ``MambaIC.from_state_dict`` only loads
# already-converted state dicts. Run this script once to translate a published
# upstream checkpoint into compressai layout, then load the result via
# ``from_state_dict``.
# ----------------------------------------------------------------------------

_LATENT_PREFIX_REWRITES = (
    ("atten_mean.", "latent_codec.mean_support_transforms.", True),
    ("atten_scale.", "latent_codec.scale_support_transforms.", True),
    ("anchor_atten_mean.", "latent_codec.context_mean_transforms.", True),
    ("anchor_atten_scale.", "latent_codec.context_scale_transforms.", True),
    ("cc_mean_transforms.", "latent_codec.cc_mean_transforms.", False),
    ("cc_scale_transforms.", "latent_codec.cc_scale_transforms.", False),
    ("lrp_transforms.", "latent_codec.lrp_transforms.", False),
    ("context_prediction.", "latent_codec.context_prediction.", False),
    ("context_vss.", "latent_codec.context_vss.", False),
    ("gaussian_conditional.", "latent_codec.gaussian_conditional.", False),
)

_UPSTREAM_CONTEXT_VSS_DUP = re.compile(r"^context_vss_\d+\.")
_SEQUENCE_WRAPPER = re.compile(r"^(\d+)\.0\.(.*)$")


def _is_upstream_layout(state_dict: Dict[str, Tensor]) -> bool:
    """Detect the published MambaIC checkpoint layout (top-level ``atten_*`` /
    ``context_vss`` submodules + duplicated ``context_vss_{i}`` attributes +
    upstream WMSA naming)."""
    for key in state_dict:
        if (
            key.startswith(("atten_mean.", "atten_scale.", "context_vss."))
            or _UPSTREAM_CONTEXT_VSS_DUP.match(key)
            or ".msa.relative_position_params" in key
            or ".msa.embedding_layer." in key
        ):
            return True
    return False


def _ensure_identity_attention_projection(
    state_dict: Dict[str, Tensor],
    output_proj_key: str,
    output_proj_value: Tensor,
) -> None:
    """Synthesise an identity ``attn.proj`` for the WMSA block.

    Upstream folds the output projection into ``linear`` (mapped to
    ``output_proj``), while compressai's :class:`WMSA` keeps a separate
    ``attn.proj``. Insert an identity so the two-stage projection composes to
    the upstream single-stage one.
    """
    prefix, suffix = output_proj_key.rsplit(".msa.output_proj.", 1)
    attn_proj_key = f"{prefix}.msa.attn.proj.{suffix}"
    if attn_proj_key in state_dict:
        return
    if suffix == "weight":
        dimension = output_proj_value.size(0)
        state_dict[attn_proj_key] = torch.eye(
            dimension,
            dtype=output_proj_value.dtype,
            device=output_proj_value.device,
        )
        return
    if suffix == "bias":
        state_dict[attn_proj_key] = torch.zeros_like(output_proj_value)


def convert_upstream_mambaic_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Translate a published MambaIC checkpoint into the compressai key layout.

    See the module docstring for the full list of key differences. Keys already
    in compressai layout pass through unchanged, so calling this on a
    round-tripped ``state_dict`` is a safe no-op.
    """
    converted: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if _UPSTREAM_CONTEXT_VSS_DUP.match(key):
            continue
        new_key = key
        for old_prefix, new_prefix, drop_seq_wrapper in _LATENT_PREFIX_REWRITES:
            if new_key.startswith(old_prefix):
                tail = new_key[len(old_prefix) :]
                if drop_seq_wrapper:
                    match = _SEQUENCE_WRAPPER.match(tail)
                    if match:
                        tail = f"{match.group(1)}.{match.group(2)}"
                new_key = new_prefix + tail
                break

        if ".msa.relative_position_params" in new_key:
            new_key = new_key.replace(
                ".msa.relative_position_params",
                ".msa.attn.relative_position_bias_table",
            )
            value = value.permute(1, 2, 0).reshape(-1, value.size(0)).contiguous()
        elif ".msa.embedding_layer." in new_key:
            new_key = new_key.replace(".msa.embedding_layer.", ".msa.attn.qkv.")
        elif ".msa.linear." in new_key:
            new_key = new_key.replace(".msa.linear.", ".msa.output_proj.")
            _ensure_identity_attention_projection(converted, new_key, value)

        new_key = new_key.replace(".ln1.", ".norm1.")
        new_key = new_key.replace(".ln2.", ".norm2.")
        new_key = new_key.replace(".mlp.0.", ".mlp.fc1.")
        new_key = new_key.replace(".mlp.2.", ".mlp.fc2.")

        converted[new_key] = value
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the upstream MambaIC checkpoint (e.g. checkpoint.pth.tar).",
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
        state_dict = convert_upstream_mambaic_state_dict(state_dict)
        print(f"converted upstream layout -> {len(state_dict)} keys")

    net = MambaIC.from_state_dict(state_dict)
    net.eval()
    print(
        "variant: "
        f"N={net.N}, M={net.M}, hyper_channels={net.hyper_channels}, "
        f"num_slices={net.num_slices}, max_support_slices={net.max_support_slices}, "
        f"depths={tuple(net.depths)}, window_size={net.window_size}"
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
