# NVTC CompressAI Port

## Status

- Completed: 2026-05-12
- Candidate source: `candidate/NVTC/image/nvtc_image/model/nvtc.py`
- Main files:
  - `compressai/models/nvtc.py`
  - `compressai/models/nvtc_support.py`
  - `compressai/layers/lic/nvtc.py`
  - `compressai/zoo/image.py`
  - `tests/test_models.py`
  - `tests/test_zoo.py`

## Scope

- Ported the NVTC image model to a pure `CompressionModel` subclass.
- Preserved upstream module names inside the state dict where practical:
  `quantizer.*.codebook`, `entropy_model.param_table`,
  `prior_fn.nn.*`, `vt_encoder`, `vt_decoder`, and projection modules.
- Added automatic padding/cropping for inputs not divisible by every
  `downscale_factor * block_size`.
- Added `likelihoods["vq"]` so the standard CompressAI
  `RateDistortionLoss` can consume the model output.
- Added `from_state_dict` config inference for custom NVTC shapes.
- Added zoo factory `nvtc()` and registry key `nvtc`.

## Deliberate Non-Scope

- Practical entropy coding is not implemented. The upstream NVTC image README
  marks this as TODO, so `compress` and `decompress` raise
  `NotImplementedError`.
- No pretrained URLs were mirrored.

## Verification

- `.venv/bin/python -m py_compile compressai/models/nvtc.py compressai/layers/lic/nvtc.py`
- `.venv/bin/python -m pytest tests/test_models.py::TestModels::test_nvtc -q`
- `.venv/bin/python -m pytest tests/test_zoo.py::TestCandidateModels::test_nvtc -q`
