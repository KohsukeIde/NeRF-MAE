# ScanNet Label-Efficiency Results

Snapshot: 2026-06-21 JST. ScanNet FCOS1000, finetune seeds 1/2/3. Pretrained rows use the existing Front3D e300 checkpoints. Scratch rows use the same FCOS path with `SCRATCH_BACKBONE=1`.

## 10% labels

| arm | AP@25 | AP@50 | AP@75 |
|---|---:|---:|---:|
| scratch | 0.4905 ± 0.0461 | 0.1181 ± 0.0457 | 0.0010 ± 0.0009 |
| NeRF-MAE† / joint e300 | 0.5263 ± 0.0243 | 0.1874 ± 0.0086 | 0.0005 ± 0.0005 |
| structure-first / cosine e300 | 0.5501 ± 0.0128 | 0.1623 ± 0.0077 | 0.0004 ± 0.0004 |

## 100% labels

| arm | AP@25 | AP@50 | AP@75 |
|---|---:|---:|---:|
| scratch | 0.4548 ± 0.0202 | 0.0940 ± 0.0347 | 0.0007 ± 0.0009 |
| NeRF-MAE† / joint e300 | 0.5137 ± 0.0136 | 0.1757 ± 0.0195 | 0.0009 ± 0.0013 |
| structure-first / cosine e300 | 0.5655 ± 0.0256 | 0.1657 ± 0.0442 | 0.0004 ± 0.0004 |

## Per-seed AP@50

| regime | arm | seed1 | seed2 | seed3 |
|---|---|---:|---:|---:|
| 10% | scratch | 0.1474 | 0.1416 | 0.0655 |
| 10% | NeRF-MAE† / joint e300 | 0.1915 | 0.1775 | 0.1932 |
| 10% | structure-first / cosine e300 | 0.1582 | 0.1574 | 0.1712 |
| 100% | scratch | 0.1163 | 0.1117 | 0.0540 |
| 100% | NeRF-MAE† / joint e300 | 0.1898 | 0.1838 | 0.1535 |
| 100% | structure-first / cosine e300 | 0.1912 | 0.1911 | 0.1147 |

## Reading

- At 10% labels, cosine is best but margins are small: AP@50 `0.1623` vs joint `0.1874` (`-0.0252`) and scratch `0.1181` (`+0.0441`).

- At 100% labels, joint is best on AP@50: joint `0.1757`, cosine `0.1657`, scratch `0.0940`. Cosine improves over scratch but not over joint on ScanNet full-label.

- This is useful as cross-dataset label-efficiency evidence only in a conservative form: the structure-first row is competitive/slightly better at 10% labels, but ScanNet does not show the strong Front3D-style label-efficiency lift.

- The strongest positive ScanNet signal remains AP@25/coarse transfer for pretrained rows, not AP@50 dominance.


## Eval paths


### 10%

#### scratch
- seed1: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/scannet_scratch_lowlabel_pt10_seed1_fcos1000_eval/eval.json`
- seed2: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/scannet_scratch_lowlabel_pt10_seed2_fcos1000_eval/eval.json`
- seed3: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/scannet_scratch_lowlabel_pt10_seed3_fcos1000_eval/eval.json`

#### NeRF-MAE† / joint e300
- seed1: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/baseline_e300_scannet_lowlabel_pt10_seed1_fcos1000_eval/eval.json`
- seed2: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/baseline_e300_scannet_lowlabel_pt10_seed2_fcos1000_eval/eval.json`
- seed3: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/baseline_e300_scannet_lowlabel_pt10_seed3_fcos1000_eval/eval.json`

#### structure-first / cosine e300
- seed1: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/cosine_ramp_e300_scannet_lowlabel_pt10_seed1_fcos1000_eval/eval.json`
- seed2: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/cosine_ramp_e300_scannet_lowlabel_pt10_seed2_fcos1000_eval/eval.json`
- seed3: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/cosine_ramp_e300_scannet_lowlabel_pt10_seed3_fcos1000_eval/eval.json`

### 100%

#### scratch
- seed1: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/scannet_scratch_lowlabel_pt100_seed1_fcos1000_eval/eval.json`
- seed2: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/scannet_scratch_lowlabel_pt100_seed2_fcos1000_eval/eval.json`
- seed3: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/scannet_scratch_lowlabel_pt100_seed3_fcos1000_eval/eval.json`

#### NeRF-MAE† / joint e300
- seed1: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/baseline_e300_scannet_fcos1000_seed1_eval/eval.json`
- seed2: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/baseline_e300_scannet_fcos1000_seed2_eval/eval.json`
- seed3: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/baseline_e300_scannet_fcos1000_seed3_eval/eval.json`

#### structure-first / cosine e300
- seed1: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/cosine_ramp_e300_scannet_fcos1000_seed1_eval/eval.json`
- seed2: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/cosine_ramp_e300_scannet_fcos1000_seed2_eval/eval.json`
- seed3: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/cosine_ramp_e300_scannet_fcos1000_seed3_eval/eval.json`
