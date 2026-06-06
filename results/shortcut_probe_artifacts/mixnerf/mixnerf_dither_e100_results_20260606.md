# MixNeRF Dither e100 Scout Results

Snapshot: 2026-06-06 JST

Status:
- All e100 dither pretrain and dependent FCOS jobs completed.
- `shuffle_visible` logs confirm visible-only source:
  `same_scene_fill_source=visible_only`, `self_replacement_rate=0.0`,
  `masked_source_rate=0.0`, `base_mask_mean=0.0`.

## Results

| condition | seed | fill | AP@25 | AP@50 | AP@75 | R@50 top300 | R@25 top300 |
|---|---:|---|---:|---:|---:|---:|---:|
| `mixnerf_lite_shuffle_visible_masked` | 1 | visible-only same-scene shuffle | 0.8425 | 0.5766 | 0.1328 | 0.7206 | 0.9632 |
| `mixnerf_lite_shuffle_visible_masked` | 2 | visible-only same-scene shuffle | 0.8512 | 0.5873 | 0.1344 | 0.7279 | 0.9485 |
| `mixnerf_lite_zeros_masked` | 1 | zero | 0.8480 | 0.6262 | 0.1012 | 0.7426 | 0.9338 |
| `mixnerf_lite_zeros_masked` | 2 | zero | 0.8212 | 0.5587 | 0.1304 | 0.6912 | 0.9559 |
| `mixnerf_lite_mean_masked` | 1 | channel mean | 0.8615 | 0.5670 | 0.0875 | 0.6912 | 0.9559 |

Summary:

| condition | n | AP@25 mean | AP@50 mean | AP@50 std | AP@75 mean | R@50 mean |
|---|---:|---:|---:|---:|---:|---:|
| `shuffle_visible` | 2 | 0.8469 | 0.5819 | 0.0076 | 0.1336 | 0.7243 |
| `zero` | 2 | 0.8346 | 0.5925 | 0.0477 | 0.1158 | 0.7169 |
| `mean` | 1 | 0.8615 | 0.5670 | - | 0.0875 | 0.6912 |

## Interpretation

- The clean visible-only same-scene dither does **not** satisfy the planned method
  promotion criterion.  It does not beat zero-fill on mean AP@50
  (`0.5819` vs `0.5925`).
- `shuffle_visible` is much more stable than zero-fill on AP@50, and it is better
  on AP@75 and R@50 mean.  However, the main detection metric for this branch was
  AP@50, and zero-fill seed1 is substantially higher.
- The `mean` non-zero control is close to `shuffle_visible` on AP@50
  (`0.5670` vs `0.5766` for seed1), so the result is not strong evidence for a
  special scene-distribution matching mechanism.
- The e30 all-patch shuffle result should not be upgraded into a method claim.
  The visible-only e100 result is competitive but not decisive.

Decision:
- Do not move dither / mask-token-free MixNeRF into the AAAI main path.
- Keep it as a separated appendix / future-method observation:
  visible-only same-scene dither stabilizes AP@50 and improves AP@75/R@50, but
  does not clearly outperform the simpler zero-fill control on AP@50.
- Stop broad MixNeRF/dither exploration unless a new mechanism is specified
  before launching further jobs.

