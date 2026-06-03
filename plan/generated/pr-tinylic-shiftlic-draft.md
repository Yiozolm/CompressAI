# PR draft — TinyLIC + ShiftLIC upstream integration

**Branch:** `pr-tinylic-shiftlic` → base `master` (no stacked dependency)
**Repo:** `Yiozolm/CompressAI`

## Summary

Migrates **TinyLIC** (Lu & Ma, arXiv 2204.11448, Apache 2.0) and **ShiftLIC**
(Bao et al., TCSVT 2025) from the fork's `script` trunk into `compressai/`.
The two models are bundled into one PR because **ShiftLIC large reuses
TinyLIC's entropy model** — a new shared `MultistageCheckerboardLatentCodec`
serves both. Four zoo entries land: `tinylic` + `shiftlic-{small,middle,large}`.

## What's new

| Component | Path | Notes |
|---|---|---|
| `MultistageCheckerboardLatentCodec` | `compressai/latent_codecs/multistage_checkerboard.py` | Staged channel + checkerboard codec. 4 gamma-scheduled slices (cosine=TinyLIC, linear=ShiftLIC); caller injects the cross-channel transform via `make_cc_transform`. |
| `MultistageMaskedConv2d` | `compressai/layers/layers.py` | Mask A/B/C for the four-pass spatial-context branch. |
| NSA blocks | `compressai/layers/attn/nsa.py` | `NSABlock` / `BasicViTLayer` / `ResViTBlock` (Neighborhood Attention). |
| natten fallback | `compressai/layers/attn/natten/` | Vendored pure-PyTorch `NeighborhoodAttention` (qkv/proj/rpb layout) + lazy router. |
| multiplex ops | `compressai/ops/multiplex.py` | `space2depth`/`depth2space`/`demultiplex(_v2)`/`multiplex(_v2)`. |
| TinyLIC | `compressai/models/tinylic.py` | NAT backbone + shared codec. |
| ShiftLIC | `compressai/models/shiftlic.py` | 3 variants; shift blocks inlined (ShiftLIC-exclusive). |
| Converters | `examples/convert_{tinylic,shiftlic}_checkpoint.py` | `convert_upstream_*_state_dict` free functions + CLI. |

## Design decisions

- **No `natten` extra.** The vendored pure-PyTorch `NeighborhoodAttention` is
  always used; the optional PyPI `natten` package is only probed
  (`is_natten_available()`), never imported. Declaring a `[tinylic]` natten
  extra would advertise a dependency the code never consumes, so NSA simply
  rides the existing `[attn]` extra's `timm`. A future revision can wire a CUDA
  fast path through `natten` without breaking state-dict layouts.
- **convert-to-examples.** Following every merged Family-2 model, the model
  modules do **no** `load_state_dict` override — `from_state_dict` is pure
  shape inference, and upstream-checkpoint key remapping (`module.` strip +
  top-level `entropy_parameters_*`/`cc_transforms.*`/`sc_transform_*`/
  `gaussian_conditional.*` → `latent_codec.*`) lives in the `examples/`
  converters.
- **Hyperprior left at model root (MambaIC precedent), not containerized.**
  ShiftLIC small/middle feed `abs(y)` to the hyperencoder (a quirk
  `HyperpriorLatentCodec.forward` can't express), and TinyLIC has public
  pretrained weights whose key fidelity matters. Keeping a hand-written
  `forward` + top-level `entropy_bottleneck`/`h_a`/`h_s` keeps all variants
  consistent and matches the merged MambaIC model.
- **ShiftLIC shift blocks inlined** in `shiftlic.py` (ShiftLIC-exclusive;
  matches CMIC's inline-private-blocks precedent).
- **Import paths aligned to master's deep-import-only layout**:
  `ResViTBlock` from `compressai.layers.attn`; `conv`/`deconv` from
  `compressai.models.utils` (the `compressai.layers.ssm.builders` precedent).

## Licensing

Both upstream repos are Apache 2.0. Files carry the InterDigital
BSD-3-Clause-Clear header + an `adapts code from <upstream>` line, matching the
GLIC/STF precedents. No COPYING/AUTHORS changes (the repo has none).

## Verification

- `pytest tests/` → **387 passed**, 4 skipped. (2 pre-existing failures
  unrelated to this PR: `test_train_example_ddp` = DDP socket/environmental;
  `TestCheng2020::test_pretrained[ms-ssim]` = network download of pre-existing
  Cheng2020 S3 weights.)
- New tests: `TestTinyLIC`, `TestShiftLIC` (×3 variants), `TestNSA`,
  `TestMultiplex`, `TestMultistageMaskedConv2d` — forward, `from_state_dict`
  round-trip, large/TinyLIC compress↔decompress bit-exact round-trip, and
  upstream-conversion round-trip.
- `import compressai` / `import compressai.zoo` pull no `timm` / `natten`
  (deep-import-only preserved).
- ruff format / imports / lint clean on all changed files.
- `uv lock --check` consistent (no new dependencies).

## Commits

1. `feat(ops): add space/depth + checkerboard (de)multiplex helpers`
2. `feat(layers): add MultistageMaskedConv2d (mask A/B/C)`
3. `feat(layers): add vendored Neighborhood Attention fallback`
4. `feat(layers): add NSA blocks (ResViTBlock) for TinyLIC`
5. `feat(latent_codecs): add MultistageCheckerboardLatentCodec`
6. `feat(models): add TinyLIC with NAT backbone`
7. `feat(models): add ShiftLIC (small/middle/large)`
8. `feat(zoo,examples): wire TinyLIC/ShiftLIC entries + converters + tests`
