# Optimization-Trajectory Gate Summary

Snapshot: 2026-05-30 JST

## Ramp-shape and surface+cosine results

| condition | epoch | AP@50 | AP@75 | R50@300 | AP@25 | reading |
|---|---:|---:|---:|---:|---:|---|
| surface_maturation_cosine_coord_jitter_tau0p7_k30_w0p05 | 300 | 0.6397 | 0.0766 | 0.7647 | 0.8190 | AP50/recall strongest but localization weak |
| linear_ramp_coord_jitter | 100 | 0.5982 | 0.0942 | 0.7279 | 0.8198 | best ramp-shape AP50 and recall |
| constant_mixed_coord_jitter | 100 | 0.5728 | 0.1062 | 0.6985 | 0.8192 | constant control beats step/reverse/cosine on AP50 |
| step_ramp_coord_jitter | 100 | 0.5693 | 0.0756 | 0.7206 | 0.8079 | alpha-to-rgba order not clearly better than constant |
| reverse_ramp_coord_jitter | 100 | 0.5504 | 0.0885 | 0.6838 | 0.8201 | reverse is weak but not decisively separated from cosine |
| cosine_ramp_coord_jitter | 100 | 0.5400 | 0.1181 | 0.6985 | 0.8244 | best AP75 but weak AP50 |

Expected trajectory-supporting pattern was:

```text
cosine ~= linear ~= step > constant > reverse
```

Observed pattern by AP@50:

```text
linear > constant > step > reverse > cosine
```

Reading:
- The e100 ramp-shape gate does not cleanly support "alpha-to-RGBA order is the
  core mechanism".
- Linear ramp is the best ramp-shape condition, but constant mixed is too strong
  for a clean order claim.
- Cosine is the weakest AP@50 ramp-shape condition in this controlled e100
  rerun, while having the best AP@75.
- Surface+cosine+jitter reaches the highest AP@50/recall observed in this scout,
  but AP@75 is weak, consistent with the earlier local-gating localization issue.

Decision:
- Do not launch adaptive-ordering as the next method by default.
- If continuing method exploration, the only justified follow-up is a narrow
  confirmation of `surface+cosine+jitter` localization/proposal diagnostics or
  linear ramp reproducibility.
- The stronger default is to shift toward an analysis-heavy story: cosine/jitter
  and surface routing can improve coarse transfer, but the simple trajectory
  order explanation is not sufficient.

## Eval-time jitter robustness

| condition | normal AP@50 | jitter AP@50 | delta AP@50 | normal AP@75 | jitter AP@75 | delta AP@75 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_e300 | 0.4695 | 0.5068 | +0.0373 | 0.0869 | 0.0813 | -0.0056 |
| cosine_e300 | 0.5539 | 0.5479 | -0.0060 | 0.1135 | 0.0754 | -0.0381 |
| baseline_coord_jitter_e100 | 0.5564 | 0.5489 | -0.0075 | 0.1015 | 0.0745 | -0.0270 |
| cosine_coord_jitter_e100 | 0.6219 | 0.5621 | -0.0598 | 0.1031 | 0.0640 | -0.0391 |
| shuffle_coord_jitter_e300 | 0.4138 | 0.4000 | -0.0138 | 0.0574 | 0.0474 | -0.0100 |

Reading:
- Eval-time jitter does not explain the original `cosine_coord_jitter_e100`
  gain as simple robustness. It drops more than `baseline_coord_jitter_e100`.
- Coord-jitter pretraining does not produce an obvious AP@75 robustness gain
  under this eval perturbation.
- This weakens a simple "coord-jitter works because it learns test-time
  transform robustness" explanation.
