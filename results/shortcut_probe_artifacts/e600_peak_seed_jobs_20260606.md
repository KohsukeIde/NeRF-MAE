# e600 Peak Seed Check Jobs

Snapshot: 2026-06-06 JST

Purpose:
- Defend the current main budget-curve claim: `cosine_ramp e600` is the strongest
  mid-budget point and exceeds `baseline e1200` in the seed1 curve.
- Avoid turning the single-seed `cosine_coord_jitter_e100` result into a new main
  pillar before the e600 peak is verified.

Policy decision:
- Adopt the Doc 7 direction.
- Do not promote Doc 8's `cosine_coord_jitter_e100 = 0.6219` to a main paper
  pillar.
- Treat `cosine_coord_jitter_e100` as a short-budget ablation/enhanced variant
  only after config audit and seed confirmation.

Submitted FCOS-only jobs:

| row | finetune seed | job |
|---|---:|---:|
| `cosine_e600` | 2 | `1830815.pbs1` |
| `baseline_e600` | 2 | `1830817.pbs1` |
| `baseline_e1200` | 2 | `1830818.pbs1` |
| `cosine_e600` | 3 | `1830819.pbs1` |
| `baseline_e600` | 3 | `1830820.pbs1` |
| `baseline_e1200` | 3 | `1830821.pbs1` |

Checkpoints:

| row | checkpoint |
|---|---|
| `cosine_e600` | `output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e600_seed1_abci3clean/epoch_600.pt` |
| `baseline_e600` | `output/nerf_mae/results/nerfmae_all_p1.0_e600_seed1_abci3budgetB/epoch_600.pt` |
| `baseline_e1200` | `output/nerf_mae/results/nerfmae_all_p1.0_e1200_seed1_abci3budgetcurve50/epoch_1200.pt` |

Run settings:
- FCOS full-label Front3D, `PERCENT_TRAIN=1.0`
- `FCOS_NUM_EPOCHS=1000`
- `FINETUNE_SEED=2/3`
- `DETERMINISTIC=0`, matching the budget-curve FCOS protocol
- `SKIP_EXISTING=1`

Expected output names:

| row | finetune seed | expected eval |
|---|---:|---|
| `cosine_e600` | 2 | `output/nerf_rpn/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e600_seed1_abci3clean_epoch600_sched_epoch_preseed1_ftseed2_fcos1000_eval/eval.json` |
| `cosine_e600` | 3 | `output/nerf_rpn/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e600_seed1_abci3clean_epoch600_sched_epoch_preseed1_ftseed3_fcos1000_eval/eval.json` |
| `baseline_e600` | 2 | `output/nerf_rpn/results/nerfmae_all_p1.0_e600_seed1_abci3budgetB_epoch600_sched_epoch_preseed1_ftseed2_fcos1000_eval/eval.json` |
| `baseline_e600` | 3 | `output/nerf_rpn/results/nerfmae_all_p1.0_e600_seed1_abci3budgetB_epoch600_sched_epoch_preseed1_ftseed3_fcos1000_eval/eval.json` |
| `baseline_e1200` | 2 | `output/nerf_rpn/results/nerfmae_all_p1.0_e1200_seed1_abci3budgetcurve50_epoch1200_sched_epoch_preseed1_ftseed2_fcos1000_eval/eval.json` |
| `baseline_e1200` | 3 | `output/nerf_rpn/results/nerfmae_all_p1.0_e1200_seed1_abci3budgetcurve50_epoch1200_sched_epoch_preseed1_ftseed3_fcos1000_eval/eval.json` |

Decision rule:
- If `cosine_e600` remains clearly above `baseline_e1200` in 3 finetune seeds,
  keep the mid-budget sample-efficiency peak as the main result.
- If the margin shrinks or reverses, downgrade the strong-efficiency claim and
  frame the budget curve as exploratory / single-seed until more evidence exists.

