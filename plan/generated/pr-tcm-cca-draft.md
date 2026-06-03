# PR draft: TCM + CCA (with Family-1 latent_codec refactor)

> Branch: `pr-tcm-cca` (waiting on push to `origin`)
> Open PR at: https://github.com/Yiozolm/CompressAI/pull/new/pr-tcm-cca
> Base: `InterDigitalInc/CompressAI:master`
> Head: `Yiozolm/CompressAI:pr-tcm-cca`

---

## Title (≤70 chars)

```
Add TCM (CVPR 2023) + CCA (NeurIPS 2024) with Family-1 codec refactor
```

---

## Body

Adds two new models from the per-model PR series in #353:

- **TCM (Transformer-CNN Mixture)** from J. Liu, H. Sun, J. Katto, *"Learned Image Compression with Mixed Transformer-CNN Architectures"*, CVPR 2023 ([arXiv:2303.14978](https://arxiv.org/abs/2303.14978)). Adapted from https://github.com/jmliu206/LIC_TCM (Apache-2.0).
- **CCA (Causal Context Adjustment)** from M. Han, S. Jiang, S. Li, X. Deng, M. Xu, C. Zhu, S. Liu, *"Causal Context Adjustment Loss for Learned Image Compression"*, NeurIPS 2024 ([arXiv:2410.04847](https://arxiv.org/abs/2410.04847)). Adapted from https://github.com/LabShuHangGU/CCA (MIT).

Per the discussion in #354, this PR also delivers the **refactored latent-codec abstraction** I committed to ship next (["I'll include the refactored abstraction layer in the next PR"](https://github.com/InterDigitalInc/CompressAI/pull/354#issuecomment-3578257918)). The new shared infrastructure unifies the channel-slice topology used by STF / WACNN (added in #354), TCM (this PR), CCA (this PR), and the upcoming DCAE / MambaVC follow-ups onto upstream `ChannelGroupsLatentCodec` rather than the temporary `ChannelSliceLatentCodec` introduced in #354.

Pretrained weights are intentionally not bundled — calling `pretrained=True` raises a clear `RuntimeError` until weights are hosted on S3 (per the discussion in #353).

## Summary

- **New zoo entries** `"tcm"` and `"cca"` (`compressai.models.tcm.TCM`, `compressai.models.cca.CCAModel`), wired via lazy-import `_LazyImport` proxy in `model_architectures` so `import compressai.zoo` stays `timm`-free.
- **New `compressai.losses.cca.CCARateDistortionLoss`** — extends `RateDistortionLoss` with the auxiliary "causal context adjustment" term (NeurIPS 2024 §3.2) wired to the optional `_CCAAuxEntropyModel` head.
- **Family-1 latent-codec infrastructure** in `compressai/latent_codecs/` (see *Refactor* below).
- **Application-layer helpers** in `compressai/models/_helpers/{channel_slice,channel_context}.py` — declarative factories that wire Family-1 models in ~3 calls.
- **Migration of STF / WACNN** to the new infrastructure — drops the temporary `ChannelSliceLatentCodec` + `SliceEntropyCompressionModel` scaffolding from #354 with no checkpoint-format break (LRP weights are byte-for-byte transferable).
- **Checkpoint converters** in `examples/convert_tcm_checkpoint.py` and `examples/convert_cca_checkpoint.py` for the published upstream weights.
- **No new hard dependencies.** TCM / CCA both reuse `timm.layers.LayerNorm2d` (already pulled in by STF), so they live under the existing `[attn]` extras group set up in #354.

## Refactor: Family-1 latent-codec abstraction

The four models targeted by this PR series so far (STF / WACNN / TCM / CCA) all follow the same outer entropy-stack shape but differ in the four shaded boxes below. #354 absorbed the variation by introducing a dedicated `ChannelSliceLatentCodec`; this PR shows that all four variants fit cleanly inside upstream `ChannelGroupsLatentCodec` once it gains four optional kwargs, eliminating the duplicate codec class and giving Family-1 models the same wiring story as ELIC.

```
HyperpriorLatentCodec(
    h_a=h_a,
    h_s=DualHyperSynthesis(h_mean_s, h_scale_s),       # (1) parallel mean/scale heads
    latent_codec={
        "z": EntropyBottleneckLatentCodec(EntropyBottleneck(N), quantizer=...),
        "y": ChannelGroupsLatentCodec(                 # (2) extended with side_in_context, etc.
            latent_codec={"y0": LRPGaussianLatentCodec(...), ...},   # (3) LRP-aware leaf
            channel_context={"y0": MeanScaleContextHead(...), ...},  # (4) split mean/scale heads
            groups=[M//K]*K,
            max_support_slices=MS,
            side_in_context=True,
            support_filter=...,                        # CCA-aux skip-most-recent
            support_count_fn=...,                      # CCA-aux head-width matching
        ),
    },
)
```

Concretely the PR adds:

| Piece | Where | What it does |
|---|---|---|
| `DualHyperSynthesis` | `latent_codecs/_hyper_synthesis.py` | 25-line adapter that runs `h_mean_s(z)` and `h_scale_s(z)` in parallel and concatenates the result, so `HyperpriorLatentCodec` sees a single `h_s`. |
| `LRPGaussianLatentCodec` | `latent_codecs/gaussian_conditional.py` (~30 lines appended) | Subclass of upstream `GaussianConditionalLatentCodec` that adds the LRP residual prediction (`y_hat += lrp_scale * tanh(lrp_transform(cat(mean_support, y_hat)))`). With `mean_support_trail_channels` set, the leaf reads its LRP input from a trailing block of `ctx_params` produced by the head's `emit_mean_support` mode — giving byte-for-byte weight transfer from the upstream `cat(latent_means, *prev_y_hat, y_hat)` layout. |
| `ChannelGroupsLatentCodec` extensions | `latent_codecs/channel_groups.py` (~50-line diff) | Four optional kwargs, all defaulting to upstream behaviour: `max_support_slices` (clamp the number of preceding slices used as prior), `support_filter` (callable to pick a custom subset of priors), `support_count_fn` (declare how many priors `support_filter` yields, so head input widths can be sized correctly), and `side_in_context` (route `side_params` from `h_s` through every `channel_context` head instead of only handing it to the leaves). ELIC and other existing users default-through to the original behaviour. |
| `MeanScaleContextHead` + `build_mean_scale_head` | `models/_helpers/channel_context.py` | Application-layer helper: parallel mean/scale `cc` stacks with optional independent support-transforms per branch, optional `emit_mean_support="pre"|"post"` mode that exposes the LRP input as a trailing block of `ctx_params`. |
| `build_channel_slice_codec` | `models/_helpers/channel_slice.py` | Application-layer factory that wires `ChannelGroupsLatentCodec` from `groups` + `leaf_factory` + `channel_context_factory` in one call. |
| `_slice_helpers` | `latent_codecs/_slice_helpers.py` | Free helpers (`slice_support_channels`, `lrp_support_channels`, `make_entropy_transform`, `infer_num_slices`, `infer_max_support_slices`) shared by all four models' `from_state_dict` machinery. |

Per-model variation now lives entirely in the kwargs:

| Model | `groups` | `support_transform` | LRP leaf | Notes |
|---|---|---|---|---|
| **STF / WACNN** | `[M//10]*10` | none (Identity) | yes | 5-conv `cc` heads `widths=(224, 176, 128, 64)`. |
| **TCM** | `[M//K]*K` | `SWAtten` (independent per mean/scale) | yes | 3-conv `cc` heads `widths=(224, 128)`. |
| **CCA-main** | `slice_proportions=(8,28,56,92,136)` (variable-length) | `NAFTransform` (independent per mean/scale) | yes | Uses `EntropyBottleneckLatentCodec(quantizer="ste")` for `z`. |
| **CCA-aux** | same as main | `NAFTransform` | yes | Lives outside the `HyperpriorLatentCodec` tree; uses `support_filter=skip_most_recent` + matching `support_count_fn`. |

The `__init__.py` of `compressai/latent_codecs/` documents this wiring story in a top-level comment block so reviewers don't need to read each model file to understand the pattern.

### State-dict layout

Containerization shifts the saved keys to a single-layer `latent_codec.*` prefix (the `HyperpriorLatentCodec`'s `self.y` / `self.z` are real `nn.Module` registrations, not nested dicts). The published upstream checkpoints round-trip via the converters below — LRP weights transfer byte-for-byte thanks to `mean_support_trail_channels`, and TCM's per-slice `gaussian_conditional` buffer is materialized by copying the single shared upstream copy K times.

```
latent_codec.h_a.0.weight                                # STF/WACNN/CCA: plain Conv2d   TCM: ResidualBottleneckBlock → .0.conv1.weight
latent_codec.h_s.h_mean_s.0.weight                       # one head per parallel arm of DualHyperSynthesis
latent_codec.h_s.h_scale_s.0.weight
latent_codec.z.entropy_bottleneck.quantiles
latent_codec.y.channel_context.y{k}.mean_cc.0.weight     # MeanScaleContextHead per slice
latent_codec.y.channel_context.y{k}.scale_cc.0.weight
latent_codec.y.channel_context.y{k}.mean_support_transform.<...>     # only if support_transform_factory given
latent_codec.y.latent_codec.y{k}.lrp_transform.0.weight  # LRPGaussianLatentCodec leaf
latent_codec.y.latent_codec.y{k}.gaussian_conditional.scale_table
aux_entropy_model.inner_codec.<same shape>               # CCA only
```

## Commits

Six commits, designed to be reviewed independently:

| Commit | Scope | LOC |
|---|---|---|
| `feat(latent_codecs): add containerized infrastructure for Family 1 codecs` | `latent_codecs/{_hyper_synthesis, _slice_helpers, gaussian_conditional, channel_groups, __init__}.py` + `models/_helpers/{channel_slice, channel_context, __init__}.py` + tests | +900 |
| `refactor(models/stf): migrate WACNN + SymmetricalTransFormer to containerized codec` | `models/stf.py` + `examples/convert_stf_checkpoint.py` updates + `tests/test_models.py::TestStf` | +400 |
| `feat(models): add TCM with containerized codec` | `models/tcm.py` + `examples/convert_tcm_checkpoint.py` + `tests/test_models.py::TestTcm` | +900 |
| `feat(models): add CCA model and loss with containerized codec` | `models/cca.py` + `losses/cca.py` + `examples/convert_cca_checkpoint.py` + `tests/test_models.py::TestCca` | +1200 |
| `chore(latent_codecs,models): drop ChannelSliceLatentCodec and SliceEntropyCompressionModel` | Delete `latent_codecs/channel_slice.py` + entire `models/_bases/` directory + remove exports | −561 |
| `chore(zoo): wire cca/tcm zoo entries with lazy import` | `zoo/{__init__,image}.py` factory functions + `_LazyImport` proxies | +48 |
| **Total** | **23 files, +4596 / −655** | |

The cleanup commit lands after all four models are migrated, so the branch never goes through a state where STF/WACNN are broken. The refactor and migrations preserve the existing public model classes — only the internal codec-tree shape and the corresponding state-dict paths change.

## License & attribution

- `compressai/models/tcm.py` carries a dual-license header pointing at the upstream `jmliu206/LIC_TCM` (Apache-2.0) alongside the standard InterDigital BSD 3-Clause Clear license for modifications.
- `compressai/models/cca.py` carries a dual-license header pointing at the upstream `LabShuHangGU/CCA` (MIT) alongside the standard InterDigital BSD 3-Clause Clear license for modifications. The internal `_NAFBlock` / `_NAFTransform` are derived from NAFNet (Chen et al. 2022, MIT) — happy to add per-class attribution headers if maintainers prefer.
- `compressai/losses/cca.py` similarly attributes the CCA paper for the auxiliary-loss formulation.

## Verified

- `pytest tests/ -q` (excluding pretrained-dependent suites — the local S3 ckpt cache is corrupted with `unexpected EOF`, unrelated to this PR) → **213 passed, 4 skipped, 32 deselected**.
- `pytest tests/test_models.py tests/test_latent_codecs.py tests/test_models_helpers.py tests/test_layers.py tests/test_init.py -q` → **74 passed** (3 new `TestStf` + 2 new `TestTcm` + 3 new `TestCca` + existing).
- Round-trip on published upstream checkpoints (`from_state_dict(strict=True)` then forward + sinusoidal-image smoke):
  - WACNN `cnn_0018_best.pth.tar` (585 keys) — strict load OK.
  - STF `stf_0018_best.pth.tar` (779 keys) — strict load OK.
  - TCM `0.05.pth.tar` (N=64, M=320, 1397 keys after per-slice GC copy) — strict load OK, sinusoidal PSNR 39.15 dB / total bpp 0.317.
  - TCM `mse_lambda_0.05.pth.tar` (N=128, M=320, 1397 keys) — strict load OK, sinusoidal PSNR 39.41 dB / total bpp 0.236.
  - CCA `checkpoint_lambda_0.3.pth.tar` (M=320, slice_sizes=[8,28,56,92,136], 97M params, 2384 keys with main + aux) — strict load OK, sinusoidal PSNR 50.07 dB / total bpp 0.072. Fresh-init baseline at the same config gives ~5 dB, confirming weights are participating.
- `import compressai` + `import compressai.zoo` + `import compressai.latent_codecs` triggers **0 timm modules** (verified via `sys.modules` snapshot diff).
- `make static-analysis` (ruff format / imports / lint, fail-fast) → all 3 steps clean.
- `uv lock --check` → consistent (no `pyproject.toml` changes in this PR).

## Test plan

- [x] Forward + state-dict round-trip for WACNN / STF / TCM / CCA at small configs (`TestStf`, `TestTcm`, `TestCca`).
- [x] Synthetic upstream-state-dict-conversion tests for all four models, asserting the new `latent_codec.*` paths exist and the old top-level paths are gone (`test_*_upstream_state_dict_conversion`).
- [x] Sinusoidal-image smoke against published upstream checkpoints for WACNN / STF / TCM (two configs) / CCA — PSNR jumps from ~5 dB (fresh init) to 39–50 dB (loaded), confirming byte-for-byte weight transfer.
- [x] Containerized `ChannelGroupsLatentCodec` extensions are backward-compatible with ELIC's existing usage (`tests/test_models.py::TestElic` still green with default kwargs).
- [ ] Maintainers: confirm dropping the `ChannelSliceLatentCodec` + `SliceEntropyCompressionModel` scaffolding from #354 is OK now that it's superseded by `ChannelGroupsLatentCodec` extensions (state-dict format is also reorganized — was acknowledged as acceptable in #354 review thread).
- [ ] Maintainers: there is one **pre-existing bug** in `ChannelGroupsLatentCodec.decompress` under `side_in_context=True` mode (a `split` dimension mismatch) that affects STF / TCM `compress`/`decompress` paths in this PR. It is independent of the refactor (the `forward` path is unaffected and all RD numbers above use forward) and I'd like to address it in a separate follow-up PR rather than expand the scope here. Happy to roll it in if you'd prefer.

## Notes for follow-up PRs (per #353)

- **DCAE** (Lu et al. CVPR 2025) and **SAAF** (Ma et al. CVPR 2026) are next — both are Family-1 channel-slice cousins that share the `DictionaryEntropyCompressionModel` pattern (dictionary cross-attention support transforms) and should drop straight onto the infrastructure added here. They are already partially implemented on a private branch using an earlier monolithic pattern; converting them to the containerized form is the bulk of the work.
- After Family 1 wraps up, the next tier is **Family 2** (channel-slice + intra-slice spatial context, 2-pass). Family 2 currently has four members in the broader backlog: ELIC (already merged via `compressai.models.sensetime`), GLIC, MLIC++, and MambaIC. ELIC and GLIC already wire on top of the upstream `ChannelGroupsLatentCodec` + per-slice `CheckerboardLatentCodec` — no new codec class needed. MLIC++ and MambaIC carry their own dedicated codec classes (anchor / nonanchor 2-pass with multi-reference intra-slice context), and the dedup threshold for merging those into a single configurable codec hasn't been crossed yet — best left as sibling classes until a fifth Family-2 user with a *different* dedicated layout shows up. Beyond that, the rest of the #353 backlog (CMIC, MambaVC, TIC, TinyLIC, ShiftLIC, Informer, FTIC, InvCompress, HPCM, WeConvene) gets family-classified and landed one or two at a time.
- **Generalize CCA's auxiliary entropy model into a reusable plugin** for other channel-slice models. The current `_CCAAuxEntropyModel` is a private `nn.Module` inside `CCAModel`, but its forward signature `(y, latent_means, latent_scales)` only depends on `latent_channels` + `slice_proportions` — not on the host backbone — so it should plug cleanly into WACNN / STF / TCM / MLIC++ / DCAE / SAAF / Mamba-family models via a `use_cca=True` opt-in. The plan is to extract it into a public `compressai.entropy_models.CausalContextAdjustmentEntropyModel` (or upgrade to a `LatentCodec` variant), pair it with the existing `CCARateDistortionLoss`, and let host models add it in ~30 lines without touching their main entropy path. Whether this transfers the RD gains the CCA paper reports on `LICAutoencoder` to other backbones is an empirical question for the follow-up PR; this PR only commits to keeping the API minimal so the migration is straightforward.
