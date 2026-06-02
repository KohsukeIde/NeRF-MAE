# Official Low-Label Comparison

Date: 2026-06-02 JST

## Purpose

Address the feedback request to compare the current low-label results against
the NeRF-MAE official low-label table before deciding whether to add another
method branch.

## Source Values

Official values are the Front3D low-label AP@50 values from NeRF-MAE
Table 4 / Table S4. External source checked on 2026-06-02:
`https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12306.pdf`.

| labels | scenes | official scratch AP@50 | official NeRF-MAE AP@50 |
|---:|---:|---:|---:|
| 10% | 12 | 0.15 | 0.18 |
| 25% | 30 | 0.29 | 0.36 |
| 50% | 61 | 0.30 | 0.42 |
| 100% | 122 | 0.41 | 0.54 |

These are paper-reported values, not reruns in this repo. The same-run
comparisons in our current artifact remain the primary evidence.

## Direct AP@50 Comparison

| labels | official scratch | official NeRF-MAE | ours scratch | ours baseline_e300 | ours cosine_ramp_e300 | ours surface_cosine_jitter_e300 | ours best |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | 0.15 | 0.18 | 0.1160 | 0.1328 | 0.2751 | 0.1756 | 0.2751 |
| 25% | 0.29 | 0.36 | 0.3044 | 0.2777 | 0.3639 | 0.3460 | 0.3639 |
| 50% | 0.30 | 0.42 | 0.3666 | 0.4191 | 0.5026 | 0.5217 | 0.5217 |
| 100% | 0.41 | 0.54 | 0.4722 | 0.4695 | 0.5539 | 0.5984 | 0.5984 |

## Key Comparisons

| labels | best condition | best - official NeRF-MAE | best - ours scratch | cosine - official NeRF-MAE | cosine - ours scratch |
|---:|---|---:|---:|---:|---:|
| 10% | cosine_ramp_e300 | +0.0951 | +0.1591 | +0.0951 | +0.1591 |
| 25% | cosine_ramp_e300 | +0.0039 | +0.0595 | +0.0039 | +0.0595 |
| 50% | surface_cosine_jitter_e300 | +0.1017 | +0.1551 | +0.0826 | +0.1360 |
| 100% | surface_cosine_jitter_e300 | +0.0584 | +0.1262 | +0.0139 | +0.0817 |

## Scratch100 Check

The feedback highlighted the importance of testing whether 50% proposed beats
100% scratch.

| comparison | AP@50 delta |
|---|---:|
| ours surface_cosine_jitter_e300 50% - ours scratch 100% | +0.0495 |
| ours cosine_ramp_e300 50% - ours scratch 100% | +0.0304 |
| ours surface_cosine_jitter_e300 50% - official scratch 100% | +0.1117 |
| ours cosine_ramp_e300 50% - official scratch 100% | +0.0926 |

Thus, under the current single-seed protocol, the 50% proposed rows exceed the
100% scratch row in our own run and in the official reported scratch table.

## Main-Variant Read

- `cosine_ramp_e300` should be treated as the base method. It is the strongest
  low-label variant at 10% and 25%, and it is above official NeRF-MAE AP@50 at
  every label fraction.
- `surface_cosine_jitter_e300` should be treated as an in-domain / label-richer
  enhancement. It is weaker than cosine at 10%/25%, but strongest at 50%/100%.
- Do not present this as cherry-picking between two equal methods. Use a
  hierarchy:
  - base: structure-first `cosine_ramp_e300`
  - enhanced: surface-anchored/jittered variant for label-richer in-domain
    detection

## Risk / Caveat

Our `scratch` rows are not identical to the official reported scratch rows; for
example, our 100% scratch AP@50 is `0.4722` while the official table reports
`0.41`. Therefore, the official comparison is supportive but should not be the
only proof. The strongest defensible claims are:

1. Same-run label efficiency: proposed rows substantially outperform our
   same-run scratch and baseline rows.
2. Official-context efficiency: the proposed rows are at or above the
   paper-reported NeRF-MAE low-label AP@50 values, but this should be described
   as a comparison to reported numbers rather than an exact rerun.

## Decision

Do not launch a new method branch solely because of method-novelty anxiety at
this point. First use this table to write the low-label result section and to
decide the final rows for multi-seed validation. Priority 3/4 work
(paper skeleton and visible-token feasibility) is being handled separately by
the user.
