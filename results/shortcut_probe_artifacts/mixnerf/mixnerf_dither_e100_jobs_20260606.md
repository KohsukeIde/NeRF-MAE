# MixNeRF Dither e100 Scout Jobs

Snapshot: 2026-06-06 JST

Purpose:
- Test whether the e30 same-scene shuffle result survives e100.
- Remove the target-leak caveat by using `shuffle_visible`, which samples same-scene
  filler only from visible patches.
- Add a simple non-zero `mean` control to distinguish scene-distribution matching
  from a generic non-zero filler effect.

Validation before launch:
- `python -m py_compile nerf_mae/model/mae/mixnerf_probe.py nerf_mae/run_swin_mixnerf_mae.py`
- `bash -n` for the dither submitter and gate PBS scripts.
- Dry-run checked that each concurrent pretrain receives a distinct
  `PRETRAIN_MASTER_PORT`.
- Local tensor sanity for `shuffle_visible` logged:
  `same_scene_fill_source=visible_only`, `self_replacement_rate=0.0`,
  `masked_source_rate=0.0`.

Submitted jobs:

| condition | epochs | seed | fill | objective | pretrain | FCOS |
|---|---:|---:|---|---|---:|---:|
| `mixnerf_lite_shuffle_visible_masked` | 100 | 1 | `shuffle_visible` | `removed_occupied` RGB / removed alpha | `1830790.pbs1` | `1830791.pbs1` |
| `mixnerf_lite_zeros_masked` | 100 | 1 | `zeros` | `removed_occupied` RGB / removed alpha | `1830792.pbs1` | `1830793.pbs1` |
| `mixnerf_lite_shuffle_visible_masked` | 100 | 2 | `shuffle_visible` | `removed_occupied` RGB / removed alpha | `1830794.pbs1` | `1830795.pbs1` |
| `mixnerf_lite_zeros_masked` | 100 | 2 | `zeros` | `removed_occupied` RGB / removed alpha | `1830796.pbs1` | `1830797.pbs1` |
| `mixnerf_lite_mean_masked` | 100 | 1 | `mean` | `removed_occupied` RGB / removed alpha | `1830798.pbs1` | `1830799.pbs1` |

Manifest:
- `output/launcher/mixnerf_dither_e100_20260606_024752/submitted.tsv`

Decision:
- `shuffle_visible > zeros` and `shuffle_visible > mean`: visible-only
  same-scene dither remains a plausible separate method branch.
- `mean ~= shuffle_visible`: the effect is likely generic non-zero filler rather
  than scene-distribution matching.
- `shuffle_visible` collapses at e100: stop the MixNeRF / dither branch and keep
  it separated from the main efficiency paper.

