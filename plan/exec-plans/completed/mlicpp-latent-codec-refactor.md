# Task Plan: MLIC++ Latent Codec Refactor

## Goal
Move MLIC++ channel/context entropy modeling into a reusable `LatentCodec` while preserving existing forward, compress/decompress, zoo, and smoke-test behavior.

## Phases
- [x] Phase 1: Inspect existing `LatentCodec` contracts and MLIC++ support code
- [x] Phase 2: Add `MLICPlusPlusLatentCodec` and export/register it
- [x] Phase 3: Simplify `MLICPlusPlus` to delegate entropy modeling to the codec
- [x] Phase 4: Update tests/TODO and verify with `.venv`

## Key Questions
1. Should MLIC++ keep `entropy_bottleneck`/`gaussian_conditional` at model root for API compatibility, or move them fully under codec?
2. Can `compress/decompress` keep the current bitstream shape while moving implementation below the model?

## Decisions Made
- New codec will live under `compressai/latent_codecs/` and use existing registry conventions.
- MLIC++ moves the full hyperprior + channel/context entropy path into `MLICPlusPlusLatentCodec`, because `ChannelGroupsLatentCodec` cannot express MLIC++'s global-intra context dependency on the current anchor pass.
- `MLICPlusPlus` keeps read-only compatibility properties for `h_a`, `h_s`, `entropy_bottleneck`, and `gaussian_conditional` while registering those modules only under `latent_codec`.
- `from_state_dict` migrates old root-level latent entropy keys to `latent_codec.*`.

## Errors Encountered
- Initial codec file exceeded 400 lines; RANS single-sample helpers were split into `compressai/latent_codecs/mlicpp_support.py`.

## Status
**Completed** - MLIC++ delegates entropy modeling to a registered latent codec and targeted `.venv` tests pass.
