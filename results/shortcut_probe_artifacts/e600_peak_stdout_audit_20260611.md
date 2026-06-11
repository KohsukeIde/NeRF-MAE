# e600 Peak Seed Check Stdout Audit

Snapshot: 2026-06-11 JST

Purpose:
- Close the final stdout/provenance check before treating the e600 peak seed
  check numbers as paper-facing evidence.

Checked jobs:

| condition | finetune seed | job | stdout path |
|---|---:|---:|---|
| `cosine_e600` | 2 | `1830815.pbs1` | `output/launcher/e600_peak_seed_check_20260606/cosine_e600_ftseed2.qsub` |
| `baseline_e600` | 2 | `1830817.pbs1` | `output/launcher/e600_peak_seed_check_20260606/baseline_e600_ftseed2.qsub` |
| `baseline_e1200` | 2 | `1830818.pbs1` | `output/launcher/e600_peak_seed_check_20260606/baseline_e1200_ftseed2.qsub` |
| `cosine_e600` | 3 | `1830819.pbs1` | `output/launcher/e600_peak_seed_check_20260606/cosine_e600_ftseed3.qsub` |
| `baseline_e600` | 3 | `1830820.pbs1` | `output/launcher/e600_peak_seed_check_20260606/baseline_e600_ftseed3.qsub` |
| `baseline_e1200` | 3 | `1830821.pbs1` | `output/launcher/e600_peak_seed_check_20260606/baseline_e1200_ftseed3.qsub` |

Queue status:
- `qstat -u $USER` no longer lists these six jobs.
- The only running jobs at this snapshot are unrelated SSL / BYOL jobs
  (`1830397.pbs1`, `1830399.pbs1`).

Stdout checks:
- All six stdout files end with `[info] FCOS complete eval=.../eval.json`.
- All six stdout files include the expected `run_fcos_pretrained.py --mode train`
  and `--mode eval` commands.
- All six train commands use:
  - `--dataset front3d`
  - `--percent_train 1.0`
  - `--num_epochs 1000`
  - `--lr 1e-4`
  - `--batch_size 2`
  - `--normalize_density`
  - `--rotate_prob 0.5 --flip_prob 0.5 --rot_scale_prob 0.5`
  - `--lr_scheduler onecycle_epoch`
  - `deterministic=False`
- All six eval commands disable eval-time augmentation:
  - `--rotate_prob 0.0 --flip_prob 0.0 --rot_scale_prob 0.0`
  - `--coord_shift_prob 0.0 --coord_shift_max_voxels 0`
- No stdout file contains `Traceback`, `Exception`, `ERROR`, or `RuntimeError`.

Checkpoint checks:

| condition | checkpoint |
|---|---|
| `cosine_e600` | `output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e600_seed1_abci3clean/epoch_600.pt` |
| `baseline_e600` | `output/nerf_mae/results/nerfmae_all_p1.0_e600_seed1_abci3budgetB/epoch_600.pt` |
| `baseline_e1200` | `output/nerf_mae/results/nerfmae_all_p1.0_e1200_seed1_abci3budgetcurve50/epoch_1200.pt` |

Eval JSON checks:

| condition | finetune seed | AP@25 | AP@50 | AP@75 | R@50 top300 | R@50 top1000 | stdout AP@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cosine_e600` | 1 | 0.8220 | 0.6196 | 0.0721 | 0.7279 | 0.7353 | n/a |
| `cosine_e600` | 2 | 0.8220 | 0.4971 | 0.0653 | 0.6397 | 0.6471 | 0.4971 |
| `cosine_e600` | 3 | 0.7958 | 0.5065 | 0.1030 | 0.6250 | 0.6324 | 0.5065 |
| `baseline_e600` | 1 | 0.7994 | 0.4994 | 0.0767 | 0.6765 | 0.6765 | n/a |
| `baseline_e600` | 2 | 0.7838 | 0.4984 | 0.0702 | 0.6838 | 0.6985 | 0.4984 |
| `baseline_e600` | 3 | 0.7998 | 0.4955 | 0.0793 | 0.6765 | 0.6985 | 0.4955 |
| `baseline_e1200` | 1 | 0.7934 | 0.5648 | 0.0809 | 0.7059 | 0.7059 | n/a |
| `baseline_e1200` | 2 | 0.7775 | 0.5087 | 0.0696 | 0.6471 | 0.6691 | 0.5087 |
| `baseline_e1200` | 3 | 0.7706 | 0.4807 | 0.0837 | 0.6176 | 0.6324 | 0.4807 |

Summary:

| condition | AP@50 mean±std | AP@25 mean±std | AP@75 mean±std |
|---|---:|---:|---:|
| `cosine_e600` | 0.5410±0.0682 | 0.8133±0.0151 | 0.0801±0.0201 |
| `baseline_e600` | 0.4978±0.0021 | 0.7943±0.0091 | 0.0754±0.0047 |
| `baseline_e1200` | 0.5181±0.0428 | 0.7805±0.0117 | 0.0781±0.0074 |

Conclusion:
- The stdout/eval provenance blocker is closed for these six jobs.
- The recorded metrics are backed by completed stdout and existing `eval.json`
  files.
- The scientific interpretation remains qualified: `cosine_e600` has the
  highest mean AP@50, but the seed1 peak does not replicate strongly and the
  mean is high-variance. Do not state a decisive e600 win over e1200 baseline.
