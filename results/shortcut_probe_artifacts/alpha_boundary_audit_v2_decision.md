# Alpha Boundary / SDF Audit v2 Decision

Date: 2026-06-01 JST

## Purpose

The first alpha-boundary audit showed that raw thresholded alpha is too
fragmented for a Boundary-SDF MAE target. This v2 audit tests whether minimal,
defensible denoising makes the target viable before launching GPU pretraining.

## Artifacts

- 20-scene visual audit:
  `results/shortcut_probe_artifacts/alpha_boundary_audit_v2_front3d_train20/`
- 60-scene no-render robustness audit:
  `results/shortcut_probe_artifacts/alpha_boundary_audit_v2_front3d_train60/`
- Script:
  `nerf_mae/probe_scripts/audit_alpha_boundary_targets_v2.py`

## 60-Scene Summary

| variant | scenes | occ ratio mean | shell/occ mean | components median | raw IoU mean | raw recall mean | sdf inside p90 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_thr001 | 60 | 0.308927 | 0.6611 | 562.5 | 1.0000 | 1.0000 | 3.411 |
| smooth075_thr001 | 60 | 0.409878 | 0.3535 | 151.5 | 0.7530 | 0.9980 | 6.781 |
| smooth100_thr001 | 60 | 0.434797 | 0.3000 | 56.5 | 0.7092 | 0.9968 | 7.828 |
| smooth100_thr002 | 60 | 0.397898 | 0.3361 | 48.0 | 0.7607 | 0.9869 | 7.015 |
| smooth100_thr001_close1_min64 | 60 | 0.409860 | 0.2678 | 29.0 | 0.6315 | 0.8968 | 6.330 |

## Read

Raw thresholded alpha remains unsuitable: it has high fragmentation
(`562.5` median components) and most occupied voxels are shell-like
(`shell/occupied = 0.6611`).

Gaussian smoothing is enough to make the target substantially more plausible:

- `smooth100_thr001` reduces median components from `562.5` to `56.5`.
- `smooth100_thr001` reduces shell/occupied from `0.6611` to `0.3000`.
- It retains almost all raw low-threshold support (`raw recall = 0.9968`) but
  inflates occupancy (`raw IoU = 0.7092`).
- `smooth100_thr002` is more conservative: lower occupancy and better raw IoU
  (`0.7607`), with still-low component count (`48.0`) and high recall
  (`0.9869`).

Morphological closing plus filtering is not the first choice. It gives the
lowest component count and shell/occupied ratio, but drops raw support recall to
`0.8968` and visually risks over-filling scene interiors. Treat it as an
aggressive ablation, not the default target.

## Decision

Do not use raw Boundary-SDF targets.

If the low-label results are not strong enough for the efficiency-only paper
framing and a real method mechanism is needed, the launchable Boundary-SDF
candidate is:

```text
alpha_smoothing_sigma = 1.0
alpha_threshold = 0.02
target = signed distance to smoothed-alpha occupancy boundary
distance_clip = 16 voxels
```

Use `smooth100_thr002` as the default because it gives the cleanest balance of
fragmentation reduction and raw-support fidelity. Use `smooth100_thr001` as the
higher-recall / more-inflated ablation.

Boundary-SDF pretraining should only be launched after the low-label 25%/10%
gate is read:

- If low-label is overwhelming, keep Boundary-SDF as future/appendix.
- If low-label is only moderate, use this v2 result to justify a small
  Boundary-SDF scout.
