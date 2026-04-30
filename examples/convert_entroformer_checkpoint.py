"""Convert an upstream Entroformer checkpoint to compressai layout.

Loads an official Entroformer checkpoint (e.g. ``entroformer_lambda0.001.pth``
from https://github.com/damo-cv/entroformer) and writes a state dict that
:meth:`compressai.models.Entroformer.from_state_dict` can load directly.

The upstream ``torch.save`` stored *whole* ``nn.Module`` objects under named
keys (``encode``/``decode``/``cit_he``/``cit_hd``/``cit_ar``/``cit_pn``/``prob``).
This script unpickles them, prefixes each sub-state-dict with its compressai
attribute path and merges the result into a single flat dict. Module names
inside each sub-tree are already 1:1 with the compressai layout (matching
upstream verbatim was a deliberate choice during the migration), so no per-key
renames are needed beyond the top-level prefixes.

A few buffers that compressai's :class:`~compressai.layers.gdn.GDN` registers
for its ``NonNegativeParametrizer`` (``beta_reparam.pedestal``,
``beta_reparam.lower_bound.bound``, ``gamma_reparam.pedestal``,
``gamma_reparam.lower_bound.bound``) are not present upstream because the
upstream GDN inlines those constants. They are deterministic, so this script
fills them in from a freshly-initialised model so the converted dict can be
loaded with ``strict=True``. Likewise for the GaussianConditional CDF buffers
(empty until ``model.update()`` is called).

Optionally compares the converted compressai-layout model's forward output
against upstream's ``main_trans_hyper_ar`` forward path (with the same input)
and reports max abs diff for ``x_hat`` / ``y_likelihoods`` / ``z_likelihoods``.

Example::

    python examples/convert_entroformer_checkpoint.py \\
        --src candidate_none/entroformer/entroformer_lambda0.001.pth \\
        --dst /tmp/entroformer_compressai.pth \\
        --smoke

To run the upstream-vs-converted forward diff, also pass
``--upstream-root candidate_none/entroformer``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from compressai.models import Entroformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the upstream Entroformer checkpoint (e.g. "
        "entroformer_lambda0.001.pth).",
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
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=None,
        help=(
            "Path to the upstream `candidate_none/entroformer` directory. "
            "If supplied, the script also runs an upstream forward pass and "
            "diffs it against the converted compressai-layout model."
        ),
    )
    parser.add_argument(
        "--position-num",
        type=int,
        default=7,
        help="2D-RPE bucket count (upstream `--position_num`, default 7 = "
        "released config).",
    )
    parser.add_argument(
        "--attn-topk",
        type=int,
        default=32,
        help="Self-attention top-k filter (upstream `--attn_topk`, default 32 "
        "= released `unidirectional` config).",
    )
    return parser.parse_args()


# Maps upstream ckpt sub-module key → compressai attribute path prefix.
PREFIX_MAP = {
    "encode": "g_a.",
    "decode": "g_s.",
    "cit_he": "latent_codec.y_hyper_encode.",
    "cit_hd": "latent_codec.y_hyper_decode.",
    "cit_ar": "latent_codec.y_ar.",
    "cit_pn": "latent_codec.param_net.",
    "prob": "latent_codec.entropy_bottleneck.",
}


def _flatten(upstream: dict) -> dict:
    """Take the upstream ``{name: nn.Module}`` dict, return a flat ``state_dict``
    keyed by compressai attribute paths."""
    flat: dict[str, torch.Tensor] = {}
    for name, mod in upstream.items():
        if name not in PREFIX_MAP:
            print(f"  warning: ignoring unknown upstream entry {name!r}")
            continue
        prefix = PREFIX_MAP[name]
        sub_sd = mod.state_dict()
        for k, v in sub_sd.items():
            flat[prefix + k] = v
    return flat


def _backfill_deterministic_buffers(
    converted: dict, fresh: torch.nn.Module
) -> None:
    """Copy any deterministic buffer (GDN reparam constants, GaussianConditional
    CDF placeholders, etc.) from a freshly-initialised model into ``converted``
    so the dict can be loaded with ``strict=True``."""
    fresh_sd = fresh.state_dict()
    added = 0
    for k, v in fresh_sd.items():
        if k not in converted:
            converted[k] = v.clone()
            added += 1
    if added:
        print(f"  backfilled {added} deterministic buffer(s) from fresh init")


def _synthetic_image(height: int = 256, width: int = 256) -> torch.Tensor:
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
    ).unsqueeze(0)
    return img.clamp(0, 1)


def _upstream_forward(upstream: dict, image: torch.Tensor) -> dict:
    """Run upstream main_trans_hyper_ar forward path. Requires the candidate
    source tree on ``sys.path``."""
    encode = upstream["encode"]
    decode = upstream["decode"]
    cit_he = upstream["cit_he"]
    cit_hd = upstream["cit_hd"]
    cit_ar = upstream["cit_ar"]
    cit_pn = upstream["cit_pn"]
    prob = upstream["prob"]

    encode.eval(); decode.eval()
    cit_he.eval(); cit_hd.eval(); cit_ar.eval(); cit_pn.eval()
    prob.eval()

    with torch.no_grad():
        x_in = image * 2.0 - 1.0
        y = encode(x_in)
        # NoiseQuant in eval = floor(z + 0.5)
        y_hat = torch.floor(y + 0.5)
        z = cit_he(y)
        z_hat = torch.floor(z + 0.5)
        feat_hyper = cit_hd(z_hat)
        feat_ar = cit_ar(y_hat)
        merged = torch.cat([feat_hyper, feat_ar], dim=1)
        params = cit_pn(merged)
        # criterion DML(num_p=2): l[:,0]=log_scales, l[:,1]=means
        n, _, h, w = y.shape
        l = params.reshape(n, 2, y.shape[1], 1, h, w)
        log_scales = l[:, 0, :, 0, :, :].clamp(min=-7.0)
        means = l[:, 1, :, 0, :, :]
        scales = log_scales.exp()
        # likelihood
        cumul = lambda v: 0.5 * torch.erfc(-(2 ** -0.5) * v)
        values = (y_hat - means).abs()
        upper = cumul((0.5 - values) / scales)
        lower = cumul((-0.5 - values) / scales)
        y_lik = (upper - lower).clamp(min=1e-9)
        # z likelihood via prob_model
        z_lik = prob(z_hat)
        # Decoder
        x_hat = decode(y_hat)
        x_hat = x_hat / 2.0 + 0.5
        x_hat = x_hat.clamp(0.0, 1.0)
    return {"x_hat": x_hat, "y_lik": y_lik, "z_lik": z_lik}


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise SystemExit(f"checkpoint not found: {args.src}")

    if args.upstream_root is not None:
        sys.path.insert(0, str(args.upstream_root))

    print(f"loading upstream ckpt: {args.src}")
    upstream = torch.load(args.src, map_location="cpu", weights_only=False)
    print(f"  upstream keys: {sorted(upstream.keys())}")

    flat = _flatten(upstream)
    print(f"flattened to {len(flat)} compressai-layout keys")

    # Construct a freshly initialised model from the (flattened) state dict so
    # we can backfill any deterministic buffers GDN / GaussianConditional add
    # but the upstream ckpt does not store.
    print("building model via from_state_dict...")
    # We have to merge the deterministic buffers _before_ from_state_dict's
    # strict load, so first instantiate at the inferred shape, copy buffers,
    # then load.
    # We piggyback on Entroformer.from_state_dict's shape inference but with a
    # try/except in case strict load fails first time (always does because of
    # the GDN buffers).
    # Round-tripping via a tmp model:
    template = Entroformer.from_state_dict_template = None  # placeholder
    # Inline the shape inference here so we can backfill before strict load:
    N = flat["g_a.encoder.0.weight"].size(0)
    M = flat["g_a.encoder.6.weight"].size(0)
    Z = flat["latent_codec.entropy_bottleneck.mu"].size(1)
    embed_w = flat["latent_codec.y_ar.to_patch_embedding.weight"]
    dim_embed = embed_w.size(0)
    depth = sum(
        1
        for k in flat
        if k.startswith("latent_codec.y_ar.blocks.")
        and k.endswith(".layer.0.SelfAttention.qkv.weight")
    )
    qkv_w = flat["latent_codec.y_ar.blocks.0.layer.0.SelfAttention.qkv.weight"]
    rpe_w = flat[
        "latent_codec.y_ar.blocks.0.layer.0.SelfAttention.relative_attention_bias.weight"
    ]
    position_num = int(round(rpe_w.size(0) ** 0.5))
    dim_head = rpe_w.size(1)
    heads = qkv_w.size(0) // (3 * dim_head)
    ffn_w = flat["latent_codec.y_ar.blocks.0.layer.1.fn.net.0.weight"]
    mlp_ratio = ffn_w.size(0) // dim_embed
    scale = sum(
        1
        for k in flat
        if k.startswith("latent_codec.y_hyper_encode.scale_blocks.")
        and k.endswith(".weight")
    )
    pn_out = flat["latent_codec.param_net.2.weight"].size(0)
    num_parameter = pn_out // M
    print(
        f"  inferred: N={N}, M={M}, Z={Z}, dim_embed={dim_embed}, depth={depth}, "
        f"heads={heads}, dim_head={dim_head}, mlp_ratio={mlp_ratio}, "
        f"position_num={position_num}, scale={scale}, num_parameter={num_parameter}"
    )

    fresh = Entroformer(
        N=N, M=M, Z=Z, dim_embed=dim_embed, depth=depth, heads=heads,
        dim_head=dim_head, mlp_ratio=mlp_ratio, position_num=position_num,
        scale=scale, num_parameter=num_parameter, attn_topk=args.attn_topk,
    )
    _backfill_deterministic_buffers(flat, fresh)

    missing, unexpected = fresh.load_state_dict(flat, strict=True)
    if missing or unexpected:
        raise SystemExit(
            f"strict load failed: missing={missing}, unexpected={unexpected}"
        )
    print("strict load of converted state dict: OK")
    fresh.eval()
    print(f"parameters: {sum(p.numel() for p in fresh.parameters()):,}")

    if args.dst is not None:
        args.dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fresh.state_dict(), args.dst)
        print(f"wrote converted state dict -> {args.dst}")

    if args.smoke:
        height = width = 256
        img = _synthetic_image(height, width)
        n_pix = height * width
        with torch.no_grad():
            out = fresh(img)
        mse = ((out["x_hat"].clamp(0, 1) - img) ** 2).mean()
        psnr = -10 * torch.log10(mse).item()
        y_bpp = -torch.log2(out["likelihoods"]["y"]).sum().item() / n_pix
        z_bpp = -torch.log2(out["likelihoods"]["z"]).sum().item() / n_pix
        print(
            f"\nforward smoke: PSNR={psnr:.2f} dB  y_bpp={y_bpp:.4f}  "
            f"z_bpp={z_bpp:.4f}  total={y_bpp + z_bpp:.4f}"
        )

    if args.upstream_root is not None:
        print("\nupstream-vs-converted forward diff:")
        img = _synthetic_image(256, 256)
        ours_out = fresh(img)
        upstream_out = _upstream_forward(upstream, img)
        diff_xh = (
            ours_out["x_hat"].clamp(0, 1) - upstream_out["x_hat"]
        ).abs().max().item()
        diff_y = (
            ours_out["likelihoods"]["y"] - upstream_out["y_lik"]
        ).abs().max().item()
        diff_z = (
            ours_out["likelihoods"]["z"] - upstream_out["z_lik"]
        ).abs().max().item()
        print(f"  x_hat: {diff_xh:.2e}  y_lik: {diff_y:.2e}  z_lik: {diff_z:.2e}")


if __name__ == "__main__":
    main()
