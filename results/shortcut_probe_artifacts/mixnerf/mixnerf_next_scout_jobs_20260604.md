# MixNeRF / mask-token-free next scouts

Snapshot: 2026-06-04 JST

## Motivation

The first MixNeRF-MAE-lite result did not validate the partner-token mechanism:
e30 noise-fill AP@50 (`0.5894`) matched or exceeded e100 partner-fill AP@50
(`0.5871`).  The more plausible current hypothesis is therefore mask-token-free
or non-zero filler corruption rather than cross-scene semantic mixing.

## Implementation audit

- The previous MixNeRF runs used `PROBE_MODE=baseline`, so their RGB loss followed
  the public occupied-all RGB objective rather than a pure removed/masked RGB loss.
- Existing logs confirm the internal base mask was disabled:
  - `internal_mask_attrs_overridden=['masking_prob']`
  - `base_mask_mean=0.0`
- The next scouts therefore separate two questions:
  - true masked-loss controls: partner / zero / noise / same-scene-shuffle at e30
  - public-loss noise/zero scaling: noise / zero at e100

## Code changes

- Added `MIXNERF_FILL_MODE=shuffle` for same-scene patch-shuffled filler.
- Added masked-loss MixNeRF conditions:
  - `mixnerf_lite_masked`
  - `mixnerf_lite_zeros_masked`
  - `mixnerf_lite_noise_masked`
  - `mixnerf_lite_shuffle_masked`
- Added probe-loss env overrides for MixNeRF gate runs:
  - `MIXNERF_PROBE_MODE`
  - `MIXNERF_PROBE_RGB_LOSS`
  - `MIXNERF_PROBE_ALPHA_LOSS`
  - `MIXNERF_PROBE_RGB_INPUT`
  - `MIXNERF_PROBE_ALPHA_INPUT`
  - `MIXNERF_PROBE_ALPHA_TARGET`

Validation:
- `py_compile` passed for `mixnerf_probe.py` and `run_swin_mixnerf_mae.py`.
- `bash -n` passed for updated MixNeRF/gate scripts.
- `submit_mixnerf_next_scouts.sh` dry-run expanded the expected qsub commands.

## Submitted jobs

Manifest:
- `output/launcher/mixnerf_next_20260604_191552/submitted.tsv`

| condition | epochs | fill | probe mode | RGB loss | alpha loss | pretrain | FCOS |
|---|---:|---|---|---|---|---:|---:|
| `mixnerf_lite_masked` | 30 | partner | custom | removed_occupied | removed | `1826351.pbs1` | `1826352.pbs1` |
| `mixnerf_lite_zeros_masked` | 30 | zeros | custom | removed_occupied | removed | `1826353.pbs1` | `1826354.pbs1` |
| `mixnerf_lite_noise_masked` | 30 | noise | custom | removed_occupied | removed | `1826355.pbs1` | `1826356.pbs1` |
| `mixnerf_lite_shuffle_masked` | 30 | same-scene shuffle | custom | removed_occupied | removed | `1826357.pbs1` | `1826358.pbs1` |
| `mixnerf_lite_noise` | 100 | noise | baseline | occupied | removed | `1826359.pbs1` | `1826360.pbs1` |
| `mixnerf_lite_zeros` | 100 | zeros | baseline | occupied | removed | `1826361.pbs1` | `1826362.pbs1` |

## Decision rule

- If masked-loss partner-fill beats noise/zero/shuffle, MixNeRF-style partner
  semantics remains viable.
- If masked-loss noise or shuffle wins, the stronger method hypothesis is
  Dithered / mask-token-free NeRF-MAE rather than MixNeRF.
- If none is competitive, stop the encoder-fill branch and return to the
  structure-first budget-curve paper path.
