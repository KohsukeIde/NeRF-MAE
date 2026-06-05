# cosine_coord_jitter_e100 Config Audit

Snapshot: 2026-06-06 JST

Purpose:
- Check whether the historical `cosine_coord_jitter_e100 = 0.6219` row is clean
  `cosine_ramp + coord_jitter`, or whether it contains surface-maturation /
  other local-gating ingredients.

Source:
- `results/shortcut_probe_artifacts/results_table.csv`
- Pretrain checkpoint:
  `output/nerf_mae/results/nerfmae_cosine_coord_jitter_p1.0_e100_seed1_abci3diag_opt1n8g_det0/epoch_100.pt`

Config summary:

| field | value |
|---|---|
| condition | `cosine_coord_jitter` |
| pretrain seed | 1 |
| epoch | 100 |
| scheduler | `onecycle_epoch` |
| global batch | 16 |
| environment | `ABCI3_1n8g_gb16_det0` |
| probe curriculum | `cosine_rgb_ramp` |
| curriculum epochs | 100 |
| RGB start/end weight | `0.0 -> 1.0` |
| RGB / alpha weight | `1.0 / 1.0` |
| order | `alpha_to_rgba` |
| loss family | `probe_occupied_rgb` |
| RGB loss region | `all_occupied` |
| alpha loss region | `removed_patches` |
| RGB denominator | `occupied_count` |
| alpha denominator | `removed_count` |
| surface maturation fields | empty |

Finetune-seed results:

| finetune seed | AP@25 | AP@50 | AP@75 | R@50 top300 |
|---:|---:|---:|---:|---:|
| 1 | 0.8097 | 0.6219 | 0.1031 | 0.7279 |
| 2 | 0.7976 | 0.5958 | 0.0577 | 0.7353 |
| 3 | 0.7903 | 0.5443 | 0.1009 | 0.6912 |

AP@50 summary:

| condition | mean AP@50 | std AP@50 | n |
|---|---:|---:|---:|
| `baseline_coord_jitter_e100` | 0.5454 | 0.0103 | 3 |
| `cosine_coord_jitter_e100` | 0.5873 | 0.0395 | 3 |

Interpretation:
- The row appears to be clean `cosine_ramp + coord_jitter`, not a surface
  maturation run.
- The seed1 AP@50 `0.6219` shrinks to a 3-finetune-seed mean of `0.5873`.
- Therefore this should not become a new main pillar.
- It can be used as an ablation/enhanced short-budget variant if the paper needs
  to show that scene-level perturbation can complement the structure-first
  schedule.

