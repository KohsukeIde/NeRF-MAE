# Visibility-Gated MAE e100 Scout Results

Snapshot:
- 2026-06-08 JST

Jobs:
- `visibility_skip_gate`: pretrain `1832027.pbs1`, FCOS `1832028.pbs1`
- `visibility_feature_reset`: pretrain `1832029.pbs1`, FCOS `1832030.pbs1`

Status:
- Both pretrain jobs completed and wrote `epoch_100.pt`.
- Both dependent FCOS evals completed.

Pretrain checkpoints:
- `output/nerf_mae/results/nerfmae_visibility_skip_gate_p1.0_e100_seed1_abci3vis_e100_20260607_141851/epoch_100.pt`
- `output/nerf_mae/results/nerfmae_visibility_feature_reset_p1.0_e100_seed1_abci3vis_e100_20260607_141851/epoch_100.pt`

Evaluation paths:
- `output/nerf_rpn/results/nerfmae_visibility_skip_gate_p1.0_e100_seed1_abci3vis_e100_20260607_141851_epoch100_sched_epoch_seed1_fcos1000_eval/eval.json`
- `output/nerf_rpn/results/nerfmae_visibility_feature_reset_p1.0_e100_seed1_abci3vis_e100_20260607_141851_epoch100_sched_epoch_seed1_fcos1000_eval/eval.json`

## Results

| condition | seed | AP@25 | AP@50 | AP@75 | R@50 top300 | R@25 top300 | AR top300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `visibility_skip_gate` | 1 | 0.8173 | 0.5869 | 0.1039 | 0.7279 | 0.9632 | 0.4971 |
| `visibility_feature_reset` | 1 | 0.8026 | 0.4992 | 0.0552 | 0.6618 | 0.9632 | 0.4613 |

Reference context:

| condition | n | AP@50 mean | AP@50 std | AP@75 mean | note |
|---|---:|---:|---:|---:|---|
| `baseline_coord_jitter_e100` | 3 | 0.5454 | 0.0103 | 0.1073 | same e100 coord-jitter baseline |
| `cosine_coord_jitter_e100` | 3 | 0.5873 | 0.0395 | 0.0872 | current strong e100 reference |
| `mixnerf_lite_shuffle_visible_e100` | 2 | 0.5819 | 0.0076 | 0.1336 | closed dither branch reference |
| `budgetcurve_cosine_ramp_e100` | 1 | 0.5711 | - | 0.0940 | non-jitter budget-curve point |

## Interpretation

- `visibility_skip_gate` is viable as a scout: AP@50 is above the
  `baseline_coord_jitter_e100` 3-seed mean and essentially tied with the
  `cosine_coord_jitter_e100` 3-seed mean.
- `visibility_skip_gate` does not clearly exceed the strongest existing e100
  references. It is not strong enough to replace the main budget-curve /
  low-label story.
- `visibility_feature_reset` is a clear no-go. Hard resetting masked-token
  features removes too much information and hurts AP@50, AP@75, and R@50.
- The result supports a narrow claim: masked-placeholder participation is real,
  and soft/skip-level gating can be non-destructive. It does not yet support
  launching a more complex attention KV-gating branch.

Decision:
- Stop `visibility_feature_reset`.
- Keep `visibility_skip_gate` as an appendix/future-method scout, or add one
  additional seed only if the visibility branch becomes strategically important.
- Do not promote attention-level visibility gating unless a reviewer-facing
  mechanism and a stronger result than `cosine_coord_jitter` are established.
