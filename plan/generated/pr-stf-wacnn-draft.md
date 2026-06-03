# PR draft: STF / WACNN

> Branch: `pr-stf-wacnn` (already pushed to `origin`)
> Open PR at: https://github.com/Yiozolm/CompressAI/pull/new/pr-stf-wacnn
> Base: `InterDigitalInc/CompressAI:master`
> Head: `Yiozolm/CompressAI:pr-stf-wacnn`

---

## Title (≤70 chars)

```
Add WACNN and SymmetricalTransFormer (STF, CVPR 2022)
```

---

## Body

Adds **WACNN** and **SymmetricalTransFormer (STF)** from R. Zou, C. Song, Z. Zhang, *"The Devil Is in the Details: Window-based Attention for Image Compression"*, CVPR 2022 ([arXiv:2203.08450](https://arxiv.org/abs/2203.08450)).

Adapted from the official implementation at https://github.com/Googolxx/STF (Apache-2.0).

This is the first installment of the per-model PR series proposed in #353. Pretrained weights are intentionally not bundled — calling `pretrained=True` raises a clear `RuntimeError` until weights are hosted on S3 (per the discussion in #353).

## Summary

- New zoo entries `"stf"` and `"stf-wacnn"` (`compressai.models.SymmetricalTransFormer` and `compressai.models.WACNN`).
- New `compressai.layers.attn` subpackage with the Swin window-based attention building blocks the two models depend on. **Reuses `timm.models.swin_transformer` wherever the implementation is generic** to avoid vendoring a parallel Swin stack — see *Reuse of timm* below.
- New `ChannelSliceLatentCodec` + `SliceEntropyCompressionModel` base — designed to be reused by the channel-conditional models in follow-up PRs (CCA, TCM, …).
- Checkpoint converter in `examples/convert_stf_checkpoint.py` that loads the published `stf_<bpp>_best.pth.tar` / `cnn_<bpp>_best.pth.tar` files from the upstream repo and writes them in compressai layout.
- `timm` added to `dependencies` (the Swin building blocks reuse `DropPath`, `Mlp`, `trunc_normal_`, `WindowAttention`, `SwinTransformerBlock`, `window_partition`, `window_reverse` from it).

## Reuse of timm

Rather than vendor a full Swin stack inside CompressAI, the implementation in this PR **delegates to `timm.models.swin_transformer` everywhere the upstream STF code matches the Swin reference**. This kept the diff focused on the genuinely STF-specific pieces and shaved ~280 lines from an earlier vendored draft.

| Component | What we do |
|---|---|
| `WindowAttention` | Thin subclass of `timm.models.swin_transformer.WindowAttention` that promotes the `relative_position_index` buffer from `persistent=False` to `True` (so released checkpoints load under strict mode) and accepts the historical `qk_scale` kwarg. ~15 lines instead of a ~50-line reimplementation. |
| `SwinTransformerBlock` (used inside `_STFBasicLayer`) | Use `timm.models.swin_transformer.SwinTransformerBlock(always_partition=True, dynamic_mask=True)` directly. After construction we promote each block's `attn.relative_position_index` to persistent so per-block keys round-trip strict-mode. Avoids reimplementing the cyclic-shift / pad / window-attn / unpad / unshift forward path. |
| `window_partition` / `window_reverse` | Square-window adapters around the timm helpers — the only difference is timm uses `Tuple[int, int]` whereas STF passes `int`. |
| `DropPath`, `Mlp`, `trunc_normal_` | `timm.layers` versions used directly. |
| `WMSA` / `WinNoShiftAttention` (the STF-specific dual-branch sigmoid-gated attention block) | Vendored, but **parameterised with `output_proj=True/False`** so a single class serves both the STF / WACNN topology in this PR (no projection) and the projection-bearing variant used by other window-attention CompressAI models. No private `_STF*` duplicate is kept. |
| Other STF-specific blocks (`SwinBlock`, `SWAtten`, `ConvTransBlock`, `_PatchEmbed`, `_WinBasedAttention`, `WinResidualUnit`, `pad_to_window_multiple`, `build_window_attention_mask`) | Vendored. These are the parts where STF deviates from the Swin reference (or where `timm` does not expose an equivalent), so vendoring keeps the API stable across `timm` releases. |

The dependency on `timm.models.swin_transformer.*` is deliberate (the file lives under `timm.models.*` rather than `timm.layers.*`, so it is not part of timm's stability promise). If maintainers prefer to insulate CompressAI from `timm` model-internals, the subclass / wrapper pattern makes it a small, self-contained ~120-line revert. Happy to do that on request.

## Commits

Three commits, designed to be reviewed independently:

| Commit | Scope | LOC |
|---|---|---|
| `feat(layers): add Swin window-based attention building blocks` | `compressai/layers/attn/{swin,inference,__init__}.py` + tiny re-export in `layers/__init__.py` | +668 |
| `feat(latent_codecs): add ChannelSliceLatentCodec + slice-entropy base` | `compressai/latent_codecs/channel_slice.py` + `compressai/models/_bases/{slice_entropy,__init__}.py` + re-export | +543 |
| `feat(models): add WACNN and SymmetricalTransFormer (STF) from Zou et al. 2022` | `compressai/models/stf.py` + zoo / converter / smoke tests + `timm` in `pyproject.toml` | +833 |
| **Total** | **15 files, +2044 lines, no modifications to existing logic** | |

## License & attribution

`compressai/models/stf.py` carries a dual-license header noting the upstream source URL and Apache-2.0 license alongside the standard InterDigital BSD 3-Clause Clear License for the modifications. The Swin building blocks in `compressai/layers/attn/swin.py` are a mix of timm subclasses / wrappers (covered by timm's Apache-2.0) and STF-derived classes (also Apache-2.0); happy to add per-file attribution headers there as well if maintainers prefer.

## Verified

- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py` → **32 passed** (3 new `TestStf` + 29 existing).
- `WACNN.from_state_dict(model.state_dict())` round-trip → `x_hat` diff = 0.0 (405 keys).
- `SymmetricalTransFormer.from_state_dict(model.state_dict())` round-trip → `x_hat` diff = 0.0 (315 keys).
- `convert_upstream_stf_state_dict` correctly re-roots `module.cc_*` / `module.gaussian_conditional` keys under `latent_codec.*` so the published `Googolxx/STF` checkpoints load via `from_state_dict`.

## Test plan

- [x] Forward + state-dict round-trip for both backbones at small config (already in `TestStf`).
- [x] Smoke-test `examples/convert_stf_checkpoint.py` against an upstream `cnn_<bpp>_best.pth.tar` checkpoint locally (`x_hat` diff = 0 between original and converted state dict in eval mode).
- [ ] Maintainers: confirm `timm` being moved into hard `dependencies` is acceptable (alternative: keep `[stf]` extras group).
- [ ] Maintainers: confirm dependence on `timm.models.swin_transformer.*` (model-internal API) is acceptable, vs. vendoring a CompressAI copy. Reverting is a small isolated change if preferred.
- [ ] Maintainers: if you want the Swin layer files to carry their own attribution headers (in addition to `models/stf.py`), I will add them.

## Notes for follow-up PRs (per #353)

The next PR will add **CCA** + **TCM** together — both reuse `ChannelSliceLatentCodec` from this PR, and CCA contributes a `CausalContextAdjustmentEntropyModel` that TCM can opt into. After that, the remaining license-clear models (`InvCompress`, `MLIC++`, `HPCM`, `SAAF`, `DCAE`, `GLIC`, `TIC`, `TinyLIC`, `ShiftLIC`) follow one or two at a time, each PR layering on top of what's already merged.
