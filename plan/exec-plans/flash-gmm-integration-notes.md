# Notes: FlashGMM Integration

## Sources

### CompressAI workspace
- Path: `/Users/boyce/Program/CompressAI`
- Initial git state: `script...origin/script [ahead 1]`

### FlashGMM workspace
- Path: `/Users/boyce/Program/FlashGMM`

## Findings

_To be filled during inspection._

## FlashGMM Findings

### Core implementation
- `Readme.md` states the core is in:
  - `/Users/boyce/Program/FlashGMM/compressai/cpp_exts/rans/rans_interface.cpp`
  - `/Users/boyce/Program/FlashGMM/compressai/entropy_models/entropy_models.py`
- Main claim: fast GMM entropy coding via search-based decoding, numerical Gaussian CDF approximation, and CPU SIMD.
- Runtime controls:
  - `APPROX_MODE`: 0 Polya/Watterson, 1 A&S, 2 Logistic (readme wording and C++ comments have minor ordering inconsistency; code says 1=A&S, 2=Logistic).
  - `USE_SIMD`: 0 disables SIMD, otherwise enabled by default.

### Python entropy model delta
- Current CompressAI already has `GaussianMixtureConditional` in `compressai/entropy_models/entropy_models.py`.
- Current implementation builds a per-symbol CDF tensor in Python via `_build_cdf()` then calls ordinary `encode_with_indexes`; this is likely the slow path.
- FlashGMM removes Python `_build_cdf()` and calls new C++ bindings:
  - `RansEncoder.encode_with_indexes_gmm(symbols, scales, means, weights, max_value, K)`
  - `RansDecoder.decode_with_indexes_gmm(encoded, scales, means, weights, max_value, K)`
  - `RansDecoder.decode_stream_gmm(scales, means, weights, max_value, K)`
- FlashGMM expects reshaped parameters as contiguous-ish CPU tensors shaped `(num_symbols, K)` and supports K dispatch 1..8.

### C++/build delta
- FlashGMM adds `torch::Tensor` parameters to the `compressai.ans` pybind module, so build switches from `Pybind11Extension` to `torch.utils.cpp_extension.CppExtension` / `BuildExtension`.
- FlashGMM includes `<torch/extension.h>`, `<x86intrin.h>`, `avx_mathfun.h`, and uses `-march=native`.
- This creates portability concerns for macOS ARM / non-x86 / wheel builds; current CompressAI build is pure pybind11 C++ and more portable.

### Latent codec/model delta
- FlashGMM adds `compressai/latent_codecs/gaussian_mixture_conditional.py`, but it lacks `@register_module` despite importing it.
- It includes debug `print()` timing statements and a constructor `print(self.gaussian_mixture_conditional)` that should not be migrated as-is.
- FlashGMM adds GMM variants of CKBD and ELIC; current CompressAI already has non-GMM CKBD/ELIC in `compressai/models/sensetime.py` and many newer models already importing `GaussianMixtureConditional` directly.
- FlashGMM's `elic_gmm.py` registers as `elic2022-official`, conflicting with current model registry name. Must rename if migrated.

### Tests available upstream
- `test_gmm_codec.py`: standalone encode/decode equality for `encode_with_indexes_gmm`, `decode_with_indexes_gmm`, and `decode_stream_gmm`.
- `test_gmm_streaming.py`: streaming chunked decode equality.
- Current repo tests only cover GMM forward behavior, not GMM compress/decompress roundtrip.

## Current CompressAI Findings

- `compressai/entropy_models/__init__.py` already exports `GaussianMixtureConditional`.
- Current rANS interface has only generic CDF-based encode/decode bindings; no GMM fast bindings.
- Current setup.py uses `Pybind11Extension`; pyproject build-system does not include `torch`.
- Current latent codec registry exports `GaussianConditionalLatentCodec`, but no `GaussianMixtureConditionalLatentCodec`.
- Current model registry uses `@register_model`, with `compressai/models/sensetime.py` providing `cheng2020-anchor-checkerboard`, `elic2022-official`, `elic2022-chandelier`.
