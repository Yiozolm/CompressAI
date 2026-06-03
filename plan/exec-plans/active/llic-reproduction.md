# LLIC Reproduction Plan

> Paper: `candidate/2304.09571v9.pdf`  
> Target: reproduce "LLIC: Large Receptive Field Transform Coding with Adaptive Weights for Learned Image Compression" in the current CompressAI codebase.

## 1. Scope

LLIC is primarily a transform-coding paper. It keeps the entropy model aligned with three baselines and replaces the analysis/synthesis transforms with large receptive-field, self-conditioned blocks:

- `LLIC-STF`: LLIC transform + STF entropy model.
- `LLIC-ELIC`: LLIC transform + ELIC entropy model.
- `LLIC-TCM`: LLIC transform + LIC-TCM entropy model.

The first implementation target should be `LLIC-ELIC`, because ELIC's uneven channel groups and checkerboard entropy path already exist in `compressai/models/sensetime.py` and map cleanly to the paper's `N=192, M=320` setting. `LLIC-STF` and `LLIC-TCM` can follow once the shared transform has stable tests.

## 2. Paper Facts to Preserve

- Main transform has four analysis stages and four synthesis stages.
- Paper figure states `N = 192`, `M = 320`.
- Analysis block counts: `[1, 1, 3, 1]`.
- Synthesis inverse block counts: `[1, 3, 1, 1]`.
- Normal basic block order: `STB -> CTB`.
- Inverse basic block order: `CTB -> STB`.
- STB kernel sizes by analysis stage: `{11, 11, 9, 9}`.
- `DepthRB`: `1x1 conv -> 3x3 depth-wise conv -> 1x1 conv`.
- `SCST`: self-conditioned spatial transform, generating depth-wise large-kernel weights from pooled input features.
- `SCCT`: self-conditioned channel transform, generating channel scaling factors from pooled input features.
- Gate block replaces Transformer FFN: expand with `1x1`, split into two halves, multiply, project with `1x1`, add residual.
- Training crop schedule: `256 x 256` for first 1.2M steps, then `512 x 512` for the remaining steps.
- MSE lambdas: `{18, 35, 67, 130, 250, 483} x 1e-4`.
- MS-SSIM lambdas: `{2.4, 4.58, 8.73, 16.64, 31.73, 60.5}`.

## 3. Code Design

### 3.1 Shared LLIC Layers

Add `compressai/layers/lic/llic.py` with narrowly scoped reusable blocks:

- `LLICDepthRB`
- `LLICGate`
- `SelfConditionedSpatialTransform`
- `SelfConditionedChannelTransform`
- `LLICSpatialTransformBlock`
- `LLICChannelTransformBlock`
- `LLICBasicBlock`
- `LLICInverseBasicBlock`
- `LLICAnalysisTransform`
- `LLICSynthesisTransform`

Implementation note for `SCST`: avoid a full `unfold` implementation as the default because it expands memory by `K^2`. Use grouped dynamic depth-wise convolution:

```python
# x: [B, C, H, W], weights: [B, C, 1, K, K]
x_grouped = x.reshape(1, B * C, H, W)
w_grouped = weights.reshape(B * C, 1, K, K)
y = F.conv2d(x_grouped, w_grouped, padding=K // 2, groups=B * C)
y = y.reshape(B, C, H, W)
```

Keep a small reference path in tests that compares against a slow per-sample/per-channel implementation.

### 3.2 Model Entry Points

Add `compressai/models/llic.py` with three model classes:

- `LLICELIC`
- `LLICSTF`
- `LLICTCM`

Recommended registration names:

- `llic-elic`
- `llic-stf`
- `llic-tcm`

Expose them through:

- `compressai/models/__init__.py`
- `compressai/zoo/image.py`
- `compressai/zoo/__init__.py`
- `tests/test_zoo.py`

No pretrained URLs should be added unless a stable checkpoint source is available. Use empty URL maps and make `pretrained=True` raise the existing "not yet available" style error.

### 3.3 Entropy Model Mapping

- `LLIC-ELIC`: reuse `ChannelGroupsLatentCodec` + `CheckerboardLatentCodec`, following `Elic2022Official`.
- `LLIC-STF`: reuse `SliceEntropyCompressionModel` + `ChannelSliceLatentCodec`, following `SymmetricalTransFormer`, but verify `num_slices` for `M=320`.
- `LLIC-TCM`: reuse the existing TCM channel-slice entropy path, including `SWAtten` mean/scale support transforms.

Treat `h_a`, `h_s` / `h_mean_s`, `h_scale_s`, and latent codecs as part of each baseline entropy model. LLIC should replace `g_a` and `g_s` first, not rewrite entropy coding.

## 4. Execution Plan

### Phase 0: Candidate Tracking

- [ ] Add an `LLIC` entry to `candidate/TODO.md` as `planned-paper-only`.
- [ ] Record that local candidate assets currently include the PDF but no LLIC source code or checkpoint directory.
- [ ] Keep `candidate/TODO.md` unchecked until at least one model is implemented and tested.

### Phase 1: Shared Transform Implementation

- [ ] Implement LLIC shared layers in `compressai/layers/lic/llic.py`.
- [ ] Re-export public blocks from `compressai/layers/lic/__init__.py` and `compressai/layers/__init__.py` only if they are expected to be reused.
- [ ] Add unit tests for:
  - shape preservation,
  - dynamic depth-wise convolution equivalence,
  - `SCST` kernel sizes `{11, 9}`,
  - `SCCT` output scaling shape,
  - `LLICAnalysisTransform` output shape for `N=192, M=320`,
  - `LLICSynthesisTransform` reconstructing back to 3 channels.

### Phase 2: LLIC-ELIC

- [ ] Implement `LLICELIC` using LLIC `g_a/g_s` and the existing ELIC entropy path.
- [ ] Add `from_state_dict` config inference for `N`, `M`, and channel groups.
- [ ] Add tests in `tests/test_models.py`:
  - forward smoke on a small tensor,
  - likelihood keys contain `y` and `z`,
  - `from_state_dict(model.state_dict())` roundtrip,
  - `compress/decompress` smoke if entropy coder update passes.
- [ ] Add zoo registration tests.

### Phase 3: LLIC-STF

- [ ] Implement `LLICSTF` with LLIC `g_a/g_s`.
- [ ] Reuse STF-style channel-slice entropy and hyperprior transforms.
- [ ] Resolve open question: exact `num_slices` for `M=320`; start with a divisor-compatible config such as `num_slices=10` if the paper/code does not specify.
- [ ] Add same model/zoo tests as `LLICELIC`.

### Phase 4: LLIC-TCM

- [ ] Implement `LLICTCM` with LLIC `g_a/g_s`.
- [ ] Reuse TCM-style channel-slice entropy, `SWAtten` support transforms, and optional CCA/AuxT gates only if they remain shape-compatible.
- [ ] Start without `use_auxt` and `use_cca`; add those only after the base model passes.
- [ ] Add same model/zoo tests as `LLICELIC`.

### Phase 5: Training Reproduction

- [ ] Build a minimal training config for one MSE point, preferably `LLIC-ELIC` at `lambda=0.0130`.
- [ ] Verify the two-stage crop schedule:
  - `0 -> 1.2M`: `256 x 256`.
  - `1.2M -> 2.0M`: `512 x 512`.
- [ ] Use the paper LR schedule:
  - `1e-4` initial.
  - `3e-5` at 1.7M.
  - `1e-5` at 1.8M.
  - `3e-6` at 1.9M.
  - record final 1.95M decay as unresolved until verified.
- [ ] Save environment metadata, configs, and dataset hashes per project reproducibility rules.
- [ ] Train a short sanity run first:
  - 1 GPU, tiny subset, 1K-5K steps.
  - Assert RD loss decreases and likelihoods stay finite.
- [ ] Scale to full training only after the short run and Kodak subset evaluation are clean.

### Phase 6: Evaluation and Ablations

- [ ] Evaluate RD curves on Kodak first.
- [ ] Add Tecnick and CLIC Pro Valid for high-resolution behavior.
- [ ] Compute BD-Rate against:
  - current baseline ELIC/STF/TCM in this repo,
  - VTM-17.0 Intra only if the local VTM pipeline exists.
- [ ] Reproduce minimal ablations:
  - static weights vs self-conditioned weights,
  - `K={5,5,5,5}`, `K={7,7,7,7}`, `K={9,9,9,9}`, `K={11,11,9,9}`,
  - `256`-only vs two-stage `256 -> 512` training.
- [ ] Add memory and time tests on increasing crop sizes up to the largest feasible local GPU resolution.

## 5. Success Criteria

### Engineering

- `import compressai` succeeds without new mandatory dependencies.
- All three model factories instantiate from `compressai.zoo`.
- `tests/test_models.py -k llic` and `tests/test_zoo.py -k llic` pass.
- `from_state_dict(model.state_dict())` passes for all three variants.
- `compress/decompress` roundtrip passes where the reused entropy path already supports bitstream coding.

### Reproduction

- Minimum acceptable result: one trained `LLIC-ELIC` MSE checkpoint improves over the repo's ELIC baseline on Kodak at the same lambda.
- Strong result: all six MSE rates for `LLIC-ELIC` produce a smooth RD curve and improve BD-Rate over ELIC.
- Full result: `LLIC-STF`, `LLIC-ELIC`, and `LLIC-TCM` reproduce the paper trend on Kodak and show larger gains on Tecnick/CLIC than on Kodak.

## 6. Risks and Open Questions

- The local candidate assets are paper-only; no upstream state dict exists for `diff=0.0` validation.
- The paper text extracted from PDF does not clearly state the final LR value after 1.95M steps.
- `LLIC-STF` channel-slice settings need confirmation because the paper figure uses `M=320`, while the existing STF implementation defaults to a different latent layout.
- Dynamic large-kernel depth-wise convolution is the main memory risk; grouped convolution should be the default, with explicit large-resolution tests.
- Full reproduction is expensive: exhaustive MSE-only training is `3 variants x 6 rates = 18` full runs; adding MS-SSIM doubles it.

## 7. Recommended Order

1. Implement shared LLIC layers and tests.
2. Implement `LLIC-ELIC`.
3. Run smoke tests and a short training sanity run.
4. Add `LLIC-STF`.
5. Add `LLIC-TCM`.
6. Train full `LLIC-ELIC` MSE curve.
7. Decide whether to train all variants and MS-SSIM after the first RD curve is credible.
