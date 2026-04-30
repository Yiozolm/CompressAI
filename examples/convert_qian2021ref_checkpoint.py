"""Convert an upstream Reference-Based AR checkpoint to compressai layout.

Loads an official Reference-Based AR checkpoint (e.g. ``model_mse1.pth`` from
https://github.com/damo-cv/img-comp-reference) and writes a state dict that
:meth:`compressai.models.RefBasedAR.from_state_dict` can load directly.

Upstream stores whole ``nn.Module`` objects under named keys (``encode`` /
``decode`` / ``pencode`` / ``pdecode`` / ``autoregressive`` / ``prob``); this
script unpickles them, prefixes each sub-state-dict with its compressai
attribute path and merges into a single flat dict. Attribute names inside
each sub-tree are kept identical to upstream (``conv_1x1_{1,2,3}`` etc.) so
no per-key renames are needed beyond top-level prefixes.

Adds the deterministic GSDN reparametriser buffers (``beta_reparam.pedestal``
etc.) that compressai registers but upstream does not store, by copying them
from a freshly-initialised model. Same treatment for the GaussianMixtureConditional
CDF placeholders and the ``relative_position_index`` / ``mask_unfold`` buffers.

Optionally compares the converted forward output against upstream and reports
max abs diff for ``x_hat`` / ``y_likelihoods`` / ``z_likelihoods``.

.. note::
   The upstream LowerBound autograd Function is the legacy (non-static) API,
   which raises in PyTorch ≥ 2.0. The ``--upstream-root`` diff is therefore
   only meaningful on PyTorch ≤ 1.13. On newer PyTorch the script just runs
   the forward path on the converted model and reports its rate-distortion.

Example::

    python examples/convert_qian2021ref_checkpoint.py \\
        --src candidate_none/img-comp-reference/model_mse1.pth \\
        --dst /tmp/qian2021ref_compressai.pth \\
        --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from compressai.models import RefBasedAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src", type=Path, required=True,
        help="Path to the upstream Reference-Based AR checkpoint.",
    )
    parser.add_argument(
        "--dst", type=Path, default=None,
        help="Optional output path for the converted state dict.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run a forward smoke test on a synthetic 256x256 image.",
    )
    parser.add_argument(
        "--upstream-root", type=Path, default=None,
        help="Path to the upstream `candidate_none/img-comp-reference` directory; "
        "only useful on PyTorch ≤ 1.13 due to legacy autograd.Function in "
        "upstream LowerBound.",
    )
    return parser.parse_args()


PREFIX_MAP = {
    "encode": "g_a.",
    "decode": "g_s.",
    "pencode": "h_a.",
    "pdecode": "h_s.",
    "prob": "entropy_bottleneck.",
    "autoregressive": "latent_codec.",
}


def _flatten(upstream: dict) -> dict:
    flat: dict[str, torch.Tensor] = {}
    for name, mod in upstream.items():
        if name not in PREFIX_MAP:
            print(f"  warning: ignoring unknown upstream entry {name!r}")
            continue
        prefix = PREFIX_MAP[name]
        for k, v in mod.state_dict().items():
            flat[prefix + k] = v
    return flat


def _backfill(converted: dict, fresh: torch.nn.Module) -> None:
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


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise SystemExit(f"checkpoint not found: {args.src}")

    if args.upstream_root is not None:
        sys.path.insert(0, str(args.upstream_root))
    else:
        # Even when only converting (no upstream forward diff), the upstream
        # `nn.Module` classes must be importable to unpickle the ckpt because
        # `torch.save({'encode': nn.Module, ...})` stores the class names.
        default_root = Path("candidate_none/img-comp-reference")
        if default_root.exists():
            sys.path.insert(0, str(default_root))

    print(f"loading upstream ckpt: {args.src}")
    upstream = torch.load(args.src, map_location="cpu", weights_only=False)
    print(f"  upstream keys: {sorted(upstream.keys())}")

    flat = _flatten(upstream)
    print(f"flattened to {len(flat)} compressai-layout keys")

    # Shape inference (mirrors RefBasedAR.from_state_dict).
    N = flat["g_a.encoder.0.weight"].size(0)
    M = flat["g_a.encoder.6.weight"].size(0)
    Z = flat["entropy_bottleneck.mu"].size(1)
    norm = "GSDN" if "g_a.encoder.1.beta2" in flat else "GDN"
    head_channels = flat["latent_codec.conv_1x1_1.0.weight"].size(0)
    sk = flat["latent_codec.mask_conv_ref.weight"].size(-1)
    print(
        f"  inferred: N={N}, M={M}, Z={Z}, norm={norm}, head_channels={head_channels}, sk={sk}"
    )

    fresh = RefBasedAR(N=N, M=M, Z=Z, norm=norm, head_channels=head_channels, sk=sk)
    _backfill(flat, fresh)

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
        try:
            print("\nrunning upstream forward (only works on PyTorch ≤ 1.13)...")
            img = _synthetic_image(256, 256)
            up_x_hat, up_y_lik, up_z_lik = _upstream_forward(upstream, img)
            ours_out = fresh(img)
            diff_xh = (
                ours_out["x_hat"].clamp(0, 1) - up_x_hat.clamp(0, 1)
            ).abs().max().item()
            diff_y = (ours_out["likelihoods"]["y"] - up_y_lik).abs().max().item()
            diff_z = (ours_out["likelihoods"]["z"] - up_z_lik).abs().max().item()
            print(f"  x_hat: {diff_xh:.2e}  y_lik: {diff_y:.2e}  z_lik: {diff_z:.2e}")
        except Exception as exc:  # noqa: BLE001 — upstream failures expected on PyTorch ≥ 2.0
            print(f"  upstream forward skipped: {exc}")


def _upstream_forward(upstream: dict, image: torch.Tensor):
    """Run upstream forward path. Requires PyTorch ≤ 1.13 because the upstream
    ``LowerBound`` is a legacy autograd Function."""
    enc = upstream["encode"]; dec = upstream["decode"]
    pen = upstream["pencode"]; pde = upstream["pdecode"]
    ar = upstream["autoregressive"]
    prob = upstream["prob"]
    for m in (enc, dec, pen, pde, ar, prob):
        m.eval()
    sys.path.insert(0, "candidate_none/img-comp-reference")
    from criterion import DiscretizedMixGaussLoss  # noqa: E402

    crit = DiscretizedMixGaussLoss(
        rgb_scale=False, x_min=-128, x_max=127, num_p=3, L=256
    )

    with torch.no_grad():
        y = enc(image)
        y_hat = torch.floor(y + 0.5)
        z = pen(y)
        z_hat = torch.floor(z + 0.5)
        z_feat = pde(z_hat)
        para1, para2, para3, _, _, _ = ar(y_hat, z_feat, crit)
        n, c, h, w = y.shape
        para_merge = torch.cat(
            [
                para1.reshape(n, 3, c, 1, h, w),
                para2.reshape(n, 3, c, 1, h, w),
                para3.reshape(n, 3, c, 1, h, w),
            ],
            dim=3,
        ).reshape(n, -1, h, w)
        y_lik = (-crit(y_hat, para_merge)).exp_()
        z_lik = prob(z_hat)
        x_hat = dec(y_hat)
    return x_hat, y_lik, z_lik


if __name__ == "__main__":
    main()
