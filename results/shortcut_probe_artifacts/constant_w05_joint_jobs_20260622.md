# Constant w=0.5 Joint Control Jobs (2026-06-22)

Purpose: test whether the `cosine_ramp` gain is due to appearance timing rather
than simply reducing the integrated RGB loss magnitude.

## Condition

- Pretrain: `constant_rgb_half`, p1.0, e300, pretrain seed 1.
- Loss: `L_alpha + 0.5 * L_rgb` for all epochs.
- RGB mask: occupied voxels, matching the public/effective NeRF-MAE RGB loss
  region.
- Alpha mask: removed patches.
- FCOS: Front3D, fcos1000, AP50-best selection, `DETERMINISTIC=0`.
- Label regimes: 100% labels and 10% labels.
- Finetune seeds: 1, 2, 3.

## Code Changes

- `nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs`
  - Added `curriculum:constant_rgb_half`.
- `nerf_mae/probe_scripts/abci3_e300_gate_fcos.pbs`
  - Added `curriculum:constant_rgb_half`.
  - Added `FCOS_PERCENT_TRAIN`.
  - Added percent-tagged output names for non-100% label regimes.
  - Changed the FCOS default to `DETERMINISTIC=0`.
- `nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh`
  - Added `constant_rgb_half` save-name/job-name resolution.
  - Added `FCOS_PERCENT_TRAIN` propagation.

Syntax check passed:

```bash
bash -n nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh
bash -n nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs
bash -n nerf_mae/probe_scripts/abci3_e300_gate_fcos.pbs
```

## Submitted Jobs

| role | labels | pretrain seed | finetune seed | job id | dependency |
| --- | ---: | ---: | ---: | --- | --- |
| pretrain | n/a | 1 | n/a | `1930716.pbs1` | none |
| FCOS | 100% | 1 | 1 | `1930717.pbs1` | `1930716.pbs1` |
| FCOS | 100% | 1 | 2 | `1930718.pbs1` | `1930716.pbs1` |
| FCOS | 100% | 1 | 3 | `1930719.pbs1` | `1930716.pbs1` |
| FCOS | 10% | 1 | 1 | `1930720.pbs1` | `1930716.pbs1` |
| FCOS | 10% | 1 | 2 | `1930721.pbs1` | `1930716.pbs1` |
| FCOS | 10% | 1 | 3 | `1930722.pbs1` | `1930716.pbs1` |

## Retry Jobs

The pretrain produced the expected checkpoint, but PBS marked `1930716.pbs1`
as `Exit_status=1` after a multi-node status-marker check:

```text
[info] pretrain complete checkpoint=/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_alpha_rgba_curr_constant_rgb_half_p1.0_e300_seed1_abci3w05/epoch_300.pt
pbsdsh: task 0x00000001 exit status 1
[error] at least one node failed; status dir=...
```

Because the original FCOS jobs were submitted with `afterok:1930716.pbs1`,
they were not run. The checkpoint exists and is usable, so the FCOS jobs were
resubmitted without a dependency.

| role | labels | pretrain seed | finetune seed | retry job id | dependency |
| --- | ---: | ---: | ---: | --- | --- |
| FCOS | 100% | 1 | 1 | `1934230.pbs1` | none |
| FCOS | 100% | 1 | 2 | `1934231.pbs1` | none |
| FCOS | 100% | 1 | 3 | `1934232.pbs1` | none |
| FCOS | 10% | 1 | 1 | `1934233.pbs1` | none |
| FCOS | 10% | 1 | 2 | `1934234.pbs1` | none |
| FCOS | 10% | 1 | 3 | `1934235.pbs1` | none |

Launcher root:

```text
/groups/gag51404/ide/vgi/NeRF-MAE/output/launcher/constant_w05_joint_20260622_043013
```

Machine-readable job table:

```text
/groups/gag51404/ide/vgi/NeRF-MAE/results/shortcut_probe_artifacts/constant_w05_joint_jobs_20260622.csv
```

## Expected Result Files

- 100% seed1:
  `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_constant_rgb_half_p1.0_e300_seed1_abci3w05_epoch300_sched_epoch_seed1_fcos1000_eval/eval.json`
- 100% seed2:
  `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_constant_rgb_half_p1.0_e300_seed1_abci3w05_epoch300_sched_epoch_preseed1_ftseed2_fcos1000_eval/eval.json`
- 100% seed3:
  `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_constant_rgb_half_p1.0_e300_seed1_abci3w05_epoch300_sched_epoch_preseed1_ftseed3_fcos1000_eval/eval.json`
- 10% seed1:
  `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_constant_rgb_half_p1.0_e300_seed1_abci3w05_epoch300_sched_epoch_seed1_pt10_fcos1000_eval/eval.json`
- 10% seed2:
  `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_constant_rgb_half_p1.0_e300_seed1_abci3w05_epoch300_sched_epoch_preseed1_ftseed2_pt10_fcos1000_eval/eval.json`
- 10% seed3:
  `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_constant_rgb_half_p1.0_e300_seed1_abci3w05_epoch300_sched_epoch_preseed1_ftseed3_pt10_fcos1000_eval/eval.json`

## Interpretation Gate

- If `cosine_ramp > constant_rgb_half`, the gain is not only from reducing
  total RGB magnitude; appearance timing matters.
- If `constant_rgb_half ~= cosine_ramp`, the claim should move from
  "appearance later" to "structure-prioritized appearance magnitude."
- If `constant_rgb_half > cosine_ramp`, the current curriculum framing should
  be weakened and the loss-balancing finding becomes the main result.
