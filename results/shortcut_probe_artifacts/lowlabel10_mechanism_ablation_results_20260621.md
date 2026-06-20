# 10% Mechanism Ablation Results

Snapshot: 2026-06-21 JST. All rows use Front3D 10% labels, FCOS1000, finetune seeds 1/2/3. New rows were run with `DETERMINISTIC=0`.

| arm | AP@25 | AP@50 | AP@75 |
|---|---:|---:|---:|
| scratch | 0.4226 ± 0.0228 | 0.1053 ± 0.0113 | 0.0016 ± 0.0016 |
| joint / constant-weight | 0.5315 ± 0.0044 | 0.1180 ± 0.0394 | 0.0008 ± 0.0009 |
| occupancy-only | 0.2447 ± 0.2105 | 0.0591 ± 0.0917 | 0.0002 ± 0.0002 |
| target-shuffled | 0.3170 ± 0.0260 | 0.0568 ± 0.0389 | 0.0001 ± 0.0001 |
| linear ramp | 0.5249 ± 0.0796 | 0.1448 ± 0.0851 | 0.0012 ± 0.0017 |
| cosine ramp | 0.5919 ± 0.0366 | 0.2527 ± 0.0211 | 0.0059 ± 0.0081 |

## Per-seed AP@50

| arm | seed1 | seed2 | seed3 |
|---|---:|---:|---:|
| scratch | 0.1160 | 0.1063 | 0.0935 |
| joint / constant-weight | 0.1328 | 0.1478 | 0.0733 |
| occupancy-only | 0.0118 | 0.1648 | 0.0008 |
| target-shuffled | 0.0778 | 0.0806 | 0.0119 |
| linear ramp | 0.1925 | 0.1953 | 0.0466 |
| cosine ramp | 0.2751 | 0.2498 | 0.2331 |

## Reading

- Cosine ramp is the strongest AP@50 row: `0.2527`. It beats joint by `+0.1347`, occupancy-only by `+0.1936`, and target-shuffled by `+0.1959`.

- The headline mechanism condition holds at 10% labels: `cosine > joint`, `occupancy-only < cosine`, and `shuffle < cosine`.

- Linear is below cosine at 10% labels (`0.1448` vs `0.2527`) and has larger seed sensitivity, so use it only as an optional ramp-shape robustness/nuance row, not as the main method.

- Strongest paper-safe phrasing: intact structure is necessary but not sufficient; the best transfer comes from adding appearance through the structure-first ramp.


## Eval paths


### scratch
- seed1: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/front3d_scratch_lowlabel_pt10_seed1_fcos1000_eval/eval.json`
- seed2: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/front3d_scratch_lowlabel_pt10_seed2_fcos1000_eval/eval.json`
- seed3: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/front3d_scratch_lowlabel_pt10_seed3_fcos1000_eval/eval.json`

### joint / constant-weight
- seed1: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/baseline_e300_lowlabel_pt10_seed1_fcos1000_eval/eval.json`
- seed2: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/baseline_e300_lowlabel_pt10_seed2_fcos1000_eval/eval.json`
- seed3: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/baseline_e300_lowlabel_pt10_seed3_fcos1000_eval/eval.json`

### occupancy-only
- seed1: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/alpha_target_only_e300_lowlabel_pt10_seed1_fcos1000_eval/eval.json`
- seed2: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/alpha_target_only_e300_lowlabel_pt10_seed2_fcos1000_eval/eval.json`
- seed3: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/alpha_target_only_e300_lowlabel_pt10_seed3_fcos1000_eval/eval.json`

### target-shuffled
- seed1: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/target_shuffle_e300_lowlabel_pt10_seed1_fcos1000_eval/eval.json`
- seed2: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/target_shuffle_e300_lowlabel_pt10_seed2_fcos1000_eval/eval.json`
- seed3: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/target_shuffle_e300_lowlabel_pt10_seed3_fcos1000_eval/eval.json`

### linear ramp
- seed1: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/linear_ramp_e300_lowlabel_pt10_seed1_fcos1000_eval/eval.json`
- seed2: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/linear_ramp_e300_lowlabel_pt10_seed2_fcos1000_eval/eval.json`
- seed3: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/linear_ramp_e300_lowlabel_pt10_seed3_fcos1000_eval/eval.json`

### cosine ramp
- seed1: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/cosine_ramp_e300_lowlabel_pt10_seed1_fcos1000_eval/eval.json`
- seed2: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/cosine_ramp_e300_lowlabel_pt10_seed2_fcos1000_eval/eval.json`
- seed3: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/cosine_ramp_e300_lowlabel_pt10_seed3_fcos1000_eval/eval.json`
