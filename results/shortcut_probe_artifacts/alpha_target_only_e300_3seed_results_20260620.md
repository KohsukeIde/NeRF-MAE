# Alpha-Target-Only e300 3-Seed Results

Snapshot: 2026-06-20 JST.

Condition: `alpha_target_only`, p1.0/e300, no coord-jitter, pretrain seed1 (`abci3ato_det0`), FCOS finetune seeds 1/2/3, `DETERMINISTIC=0`.

| finetune seed | AP@25 | AP@50 | AP@75 |
|---:|---:|---:|---:|
| 1 | 0.7560 | 0.4370 | 0.0373 |
| 2 | 0.7739 | 0.4706 | 0.0220 |
| 3 | 0.7747 | 0.4402 | 0.0734 |
| mean ± std | 0.7682 ± 0.0106 | 0.4492 ± 0.0186 | 0.0442 ± 0.0264 |

## Reading

- Occupancy/alpha-only is below the structure-first ramp rows: cosine e300 AP@50 `0.5723 ± 0.0195`, linear e300 AP@50 `0.5528 ± 0.0162`.

- It is also below joint e300 AP@50 `0.4938 ± 0.0289` in this clean e300 protocol.

- This is favorable for the ablation: alpha structure alone is not enough; adding appearance later is necessary for the strongest transfer.

## Eval paths

- seed1: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_target_only_p1.0_e300_seed1_abci3ato_det0_epoch300_sched_epoch_seed1_fcos1000_eval/eval.json`
- seed2: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_target_only_p1.0_e300_seed1_abci3ato_det0_epoch300_sched_epoch_preseed1_ftseed2_fcos1000_eval/eval.json`
- seed3: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_target_only_p1.0_e300_seed1_abci3ato_det0_epoch300_sched_epoch_preseed1_ftseed3_fcos1000_eval/eval.json`
