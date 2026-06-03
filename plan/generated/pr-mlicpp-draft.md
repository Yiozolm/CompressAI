# PR draft: MLIC family + MultiContextCheckerboardLatentCodec

> Branch: `pr-mlicpp` (already pushed to `origin`)
> Open PR at: https://github.com/Yiozolm/CompressAI/pull/new/pr-mlicpp
> Base: `InterDigitalInc/CompressAI:master` after `pr-tcm-cca` lands
> Head: `Yiozolm/CompressAI:pr-mlicpp`
> Depends on: `pr-tcm-cca` (Family-1 codec containerization)

---

## Title

```
Add MLIC family with multi-context checkerboard codec
```

---

## Body

Adds the **MLIC family** from the multi-reference learned image compression line:

- **MLIC** and **MLIC+** from W. Jiang, J. Yang, Y. Zhai, P. Ning, F. Gao, R. Wang, *"MLIC: Multi-Reference Entropy Model for Learned Image Compression"*, ACM Multimedia 2023 ([arXiv:2211.07273](https://arxiv.org/abs/2211.07273)).
- **MLIC++** from W. Jiang, J. Yang, Y. Zhai, F. Gao, R. Wang, *"MLIC++: Linear Complexity Multi-Reference Entropy Modeling for Learned Image Compression"*, ICML 2023 Neural Compression Workshop / ACM TOMM 2025 ([arXiv:2307.15421](https://arxiv.org/abs/2307.15421)).
- **MLICv2** from *"MLICv2: Enhanced Multi-Reference Entropy Modeling"*, ACM TOMM 2025 ([arXiv:2504.19119](https://arxiv.org/abs/2504.19119)).
- **MLICv2+** inference-time refinement via Stochastic Gumbel Annealing (SGA), from Yang, Bamler, Mandt, *"Improving Inference for Neural Image Compression"*, NeurIPS 2020 ([arXiv:2006.04240](https://arxiv.org/abs/2006.04240)).

Source status is intentionally mixed:

- **MLIC++** is adapted from the author's released implementation at https://github.com/JiangWeibeta/MLIC (Apache-2.0), and the real released/candidate MLIC++ checkpoint is covered by the converter smoke test below.
- **MLIC**, **MLIC+**, and **MLICv2** are paper-based reproductions. The author repository does not provide separate official implementations/checkpoints for these variants, so these models follow the papers and reuse the common MLIC++ family code only where the architecture is shared.

This is the first **Family 2** PR in the #353 series. It builds on the previous TCM + CCA refactor (`pr-tcm-cca`), specifically `HyperpriorLatentCodec`, `ChannelGroupsLatentCodec`, and `LRPGaussianLatentCodec`. Pretrained weights are intentionally not bundled - calling `pretrained=True` raises a clear `RuntimeError` until weights are hosted on S3.

## Summary

- **New zoo entries** `"mlic"`, `"mlicplus"`, `"mlicpp"`, and `"mlicv2"`, wired through lazy `_LazyImport` proxies so `import compressai` / `import compressai.zoo` does not import `timm` or the MLIC subpackages.
- **New `MultiContextCheckerboardLatentCodec`** in `compressai/latent_codecs/`: a sibling of `CheckerboardLatentCodec` for two-pass checkerboard codecs with separate anchor / non-anchor entropy-parameter heads, optional spatial context hooks, optional intra-channel context, optional per-pass LRP, and optional selective coding.
- **Shared checkerboard helpers** in `_checkerboard_helpers.py`, used by both the existing `CheckerboardLatentCodec` and the new sibling codec. Existing ELIC / Cheng-style users keep their current API and default `anchor_parity="even"` behavior.
- **MLIC application-layer blocks** in `compressai/layers/lic/mlic/`: MLIC / MLIC+ / MLIC++ local, inter-slice, intra-slice, entropy-parameter, transform, LRP, and checkerboard utilities.
- **MLICv2-specific blocks** in `compressai/layers/lic/mlicv2/`: STM transforms, HGCP, Context Reweighting, 2D RoPE, and GSC selective compression predictor.
- **Single model file** `compressai/models/mlic.py` containing `_BaseMLIC`, `MLIC`, `MLICPlus`, `MLICPlusPlus`, and `MLICv2`. All four variants use the same `build_mlic_slice_codec(variant=...)` factory and the same `_BaseMLIC` template.
- **Checkpoint converter** in `examples/convert_mlic_checkpoint.py --variant {mlic,mlic+,mlicpp,mlicv2}`. The real released/candidate MLIC++ checkpoint path is covered; MLIC / MLIC+ / MLICv2 currently remain fresh-init only because no official source checkpoints are available for those paper-reproduction variants.
- **SGA inference utility** in `compressai.ops.SGAQuantizer`, plus `examples/refine_with_sga.py` and `_BaseMLIC.refine_extract` / `refine_forward` / `set_sga_mode` for MLICv2+ style latent re-optimization.
- **No new dependency group.** MLIC uses the existing `[attn]` optional dependency surface (`timm`) already introduced for STF/TCM/CCA; this PR only adds a clarifying comment in `pyproject.toml`.

## Multi-context checkerboard codec

`MultiContextCheckerboardLatentCodec` generalizes the existing two-pass checkerboard pattern without changing existing `CheckerboardLatentCodec` users:

| Hook | Purpose |
|---|---|
| `entropy_parameters_anchor` / `entropy_parameters_nonanchor` | Separate anchor and non-anchor EP heads, matching MLIC-family checkpoints. |
| `spatial_context_anchor` | Optional anchor-side spatial/global context. MLICv2 uses this for HGCP on slice 0. |
| `spatial_context_nonanchor` | Non-anchor local context. MLIC uses stacked conv, MLIC+ / MLIC++ use windowed attention, MLICv2 reuses the MLIC++ local context. |
| `intra_channel_context_nonanchor` | Optional intra-slice global context from prior slices + current anchor. |
| `selective_predictor` | Optional GSC hook used by MLICv2 to skip symbols whose residuals are predicted to be close to their means. Default `None` is identity. |
| `lrp_anchor` / `lrp_nonanchor` | Optional per-pass latent residual prediction. |
| `lrp_input_builder`, `lrp_activation`, `lrp_scale` | Lets application-layer LRP modules either emit raw residuals or already bounded residuals. MLIC sets `lrp_activation=None` to avoid double `tanh`. |
| `anchor_parity` | MLIC-family factories explicitly set `"odd"` to match `JiangWeibeta/MLIC`; existing CompressAI checkerboard models retain the default `"even"`. |

Optional context hooks are omitted from the EP input when `None`; they do not add zero-padded channels. This is important for loading the MLIC++ checkpoint, whose slice-0 anchor head expects exactly `2M` hyperprior channels.

## MLIC-family wiring

All four model variants are assembled through `build_mlic_slice_codec(variant=...)`:

| Variant | Local context | Inter-slice global | Intra-slice global | Extra v2 hooks |
|---|---|---|---|---|
| `mlic` | `StackedCheckerboardConv` | none | `VanillaGlobalIntraContext` | none |
| `mlic+` | `WindowCheckerboardAttn` | `VanillaGlobalInterContext` | `VanillaGlobalIntraContext` | none |
| `mlicpp` | `LocalContext` | `LinearGlobalInterContext` | `LinearGlobalIntraContext` | none |
| `mlicv2` | `LocalContext` | linear + CR + RoPE | linear + CR + RoPE | HGCP on slice 0 + GSC selective predictor |

MLIC++ was originally prototyped as a separate `compressai/models/mlicpp.py` file, but the final branch deliberately folds it into `compressai/models/mlic.py`. That gives the family a single source of truth for `from_state_dict`, `downsampling_factor`, compress/decompress, SGA refinement, and zoo wiring. The old `compressai.models.mlicpp` path was never upstreamed, so this does not break any released CompressAI API.

## State-dict layout

The family uses the same containerized layout as the previous TCM/CCA PR:

```
latent_codec.h_a.*
latent_codec.h_s.*
latent_codec.z.entropy_bottleneck.*
latent_codec.y.channel_context.y{k}.*
latent_codec.y.latent_codec.y{k}.entropy_parameters_anchor.*
latent_codec.y.latent_codec.y{k}.entropy_parameters_nonanchor.*
latent_codec.y.latent_codec.y{k}.spatial_context_anchor.*        # MLICv2 slice 0
latent_codec.y.latent_codec.y{k}.spatial_context_nonanchor.*
latent_codec.y.latent_codec.y{k}.intra_channel_context_nonanchor.*
latent_codec.y.latent_codec.y{k}.selective_predictor.*           # MLICv2
latent_codec.y.latent_codec.y{k}.lrp_anchor.*
latent_codec.y.latent_codec.y{k}.lrp_nonanchor.*
latent_codec.y.latent_codec.y{k}.y.gaussian_conditional.*
```

`convert_upstream_mlicpp_state_dict` handles:

- root-level `JiangWeibeta/MLIC` checkpoint keys such as `h_a.*`, `h_s.*`, `local_context.*`, and `gaussian_conditional.*`;
- the earlier fork-script monolithic `MLICPlusPlusLatentCodec` layout;
- the intermediate `mlicpp-latent-codec-refactor` layout;
- optional `module.` prefixes from `DataParallel`;
- fanout of the upstream singleton `gaussian_conditional` buffers into the K per-slice codec leaves.

## SGA refinement

This PR also adds the MLICv2+ inference path without adding a new model class:

- `SGAQuantizer` implements the relaxed floor/ceil Gumbel-softmax quantizer with the Yang et al. annealing schedule.
- `EntropyBottleneckLatentCodec`, `GaussianConditionalLatentCodec`, `CheckerboardLatentCodec`, and `MultiContextCheckerboardLatentCodec` can opt into `quantizer="sga"` without changing their default behavior.
- `_BaseMLIC.set_sga_mode(sga)` switches the z codec and all y-slice leaves to the same SGA module; `set_sga_mode(None)` restores the training/default quantizers.
- `_BaseMLIC.refine_extract` and `refine_forward` let `examples/refine_with_sga.py` optimize y/z directly and then evaluate the RD objective.

One caveat: fresh-init MLICv2 has an untrained GSC predictor that tends to skip every symbol, which decouples `y_hat` from the optimized `y`. This is expected for an untrained GSC module. The test suite therefore runs the SGA optimization loop on MLIC++ and keeps MLICv2 to interface/shape coverage until a trained MLICv2 checkpoint is available.

## Commits

Two commits, relative to `pr-tcm-cca`:

| Commit | Scope | LOC |
|---|---|---|
| `b0924fc feat(models): add mlic family` | MLIC / MLIC+ / MLIC++ / MLICv2, `MultiContextCheckerboardLatentCodec`, MLIC-family layers, `build_mlic_slice_codec`, zoo entries, converter, SGA refine API, and tests | 32 files, +5734 / -16 |
| `9cdcb05 feat(latent_codecs): generalize sga quantization` | Extends SGA quantizer support to `GaussianConditionalLatentCodec` / `CheckerboardLatentCodec`, with targeted SGA tests | 3 files, +553 / -115 |
| **Total** | **MLIC-family PR payload** | **35 files, +6287 / -131** |

The first commit is intentionally broad because the final family design shares one `_BaseMLIC` template and one slice-codec factory. The body above splits the review surface into the reusable codec, the application-layer blocks, the four model variants, checkpoint conversion, and SGA refinement.

## License & attribution

- MLIC++-derived model/layer code carries dual-license headers pointing at `JiangWeibeta/MLIC` (Apache-2.0) alongside the standard InterDigital BSD 3-Clause Clear license for modifications.
- MLIC / MLIC+ / MLICv2 are paper-based reproductions in the same family; their shared implementation reuses MLIC++-derived building blocks where the papers specify the same modules, but there is no separate upstream source implementation to attribute for those variants.
- `compressai.ops.sga` attributes the original TensorFlow SGA implementation from `mandt-lab/improving-inference-for-neural-image-compression` and the PyTorch reference port used during implementation.
- `MultiContextCheckerboardLatentCodec` and the shared checkerboard helpers are original CompressAI infrastructure in this PR, derived from the existing CompressAI checkerboard codec structure.

## Verified

- `PATH=".venv/bin:$PATH" make static-analysis` -> ruff format / import order / lint all passed.
- `uv lock --check` -> resolved 231 packages; lockfile is consistent.
- Import audit: `import compressai` + `import compressai.zoo` loads **0 `timm` modules**, **0 `compressai.models.mlic` modules**, **0 `compressai.layers.lic.mlic` modules**, and **0 `compressai.layers.lic.mlicv2` modules**. All four zoo architecture entries are `_LazyImport`.
- Four thin-model sanity checks: `MLIC.downsampling_factor == 64`, `MLICPlus.downsampling_factor == 64`, `MLICPlusPlus.downsampling_factor == 64`, `MLICv2.downsampling_factor == 64`.
- Targeted MLIC-family regression:
  - `.venv/bin/pytest tests/test_mlic_layers.py tests/test_mlicv2_layers.py tests/test_models_helpers.py tests/test_multi_context_checkerboard.py tests/test_multi_context_checkerboard_selective.py tests/test_models.py::TestMlicPlusPlus tests/test_models.py::TestMlicFamily tests/test_models.py::TestMlicv2 tests/test_zoo.py::TestMlicZoo tests/test_sga.py -q`
  - **88 passed, 1 warning**.
- Broad local regression, excluding pretrained-dependent suites and the local macOS DDP smoke:
  - `.venv/bin/pytest tests/ -q --deselect tests/test_eval_model_video.py --deselect tests/test_zoo.py --deselect tests/test_train.py::test_train_example_ddp`
  - **286 passed, 4 skipped, 36 deselected, 1 warning**.
- MLIC++ published/candidate checkpoint smoke:
  - Source: `candidate/MLIC/mlicpp_mse_q5_2960000.pth.tar`.
  - Layout: root-level `JiangWeibeta/MLIC` keys; 1023 source keys.
  - `MLICPlusPlus.from_state_dict(sd)` strict-load succeeds after conversion; inferred `N=192`, `M=320`, `slice_num=10`, `context_window=5`.
  - Converted key count: `1023 -> 1086`; parameter count: `83,501,408`.
  - `examples/convert_mlic_checkpoint.py --variant mlicpp --smoke --smoke-size 64` reports `PSNR=38.24dB`, `y_bpp=1.0242`, `z_bpp=0.0099`, `total_bpp=1.0341`.
  - Fresh-init baseline on the same smoke is `PSNR=5.0608`, so the converted weights are clearly active.
- `tests/test_sga.py -q` after the SGA generalization commit: **25 passed, 1 warning**.
- `git diff --check` -> clean.

Local note: the full original broad command without deselecting `test_train_example_ddp` is not reliable on the local macOS/Codex environment because `torch.distributed.run --standalone` rendezvous hangs/timeouts there. This is unrelated to the MLIC changes and should be covered by Linux CI.

## Test plan

- [x] Forward + state-dict round-trip for MLIC / MLIC+ / MLIC++ / MLICv2 at small configs.
- [x] State-dict path checks for the containerized MLIC-family layout.
- [x] Legacy MLIC++ conversion tests covering current layout, fork-script monolith layout, intermediate refactor layout, and optional `module.` prefix.
- [x] `MultiContextCheckerboardLatentCodec` default, all-hooks, LRP, compress/decompress, selective-predictor, and ELIC-equivalence tests.
- [x] MLIC-layer and MLICv2-layer unit tests for forward shapes, state-dict round-trip, mask behavior, RoPE/CR/HGCP/GSC invariants.
- [x] Zoo lazy factory tests for `mlic`, `mlicplus`, `mlicpp`, and `mlicv2`; `pretrained=True` raises until weights are hosted.
- [x] MLIC++ real checkpoint strict-load + smoke through the unified converter.
- [x] SGAQuantizer, SGA codec hooks, and MLIC SGA refinement loop tests.
- [ ] Maintainers: confirm the single-PR scope (MLIC / MLIC+ / MLIC++ / MLICv2 + SGA inference utility) is acceptable. If preferred, the clean split point is after MLIC / MLIC+ / MLIC++ and before the MLICv2 selective/GSC additions.
- [ ] Maintainers: confirm keeping MLIC under the existing `[attn]` extra is acceptable. No new package is introduced; MLIC reuses `timm` through the existing attention/norm layer surface.

## Notes for follow-up PRs

- If this PR is considered too large, split after the `_BaseMLIC` unification: MLIC / MLIC+ / MLIC++ can stand alone, while MLICv2 can follow with `selective_predictor`, HGCP, CR, RoPE, and GSC.
- **GLIC** is the next Family-2 PR and can reuse the broader channel-slice / checkerboard containerization story, but it is independent of the MLIC application-layer blocks.
- **MambaIC** should re-evaluate whether `MultiContextCheckerboardLatentCodec` can host its VSS/Mamba non-anchor spatial context through the existing `spatial_context_nonanchor` hook.
- **SGA Layer B** (a generic attach/refine helper outside MLIC) is intentionally deferred until a second model family needs the same public API. This PR only adds the reusable quantizer and the MLIC-family integration.
- A small performance follow-up could cache/share `LocalContext` attention masks across the K MLIC++ leaves. The current implementation computes equivalent non-persistent masks per leaf on first use, which is numerically correct but has slightly more first-forward setup work than the upstream helper that shares the mask.
