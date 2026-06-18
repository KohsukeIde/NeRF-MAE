# Reverse Ramp e300 3-Seed Results

Snapshot: 2026-06-19 JST.

Condition: `reverse_ramp`, e300, no coord-jitter, pretrain seed1 (`abci3reverse_det0`), FCOS finetune seeds 1/2/3. Added seed2/3 jobs used `DETERMINISTIC=0`.

| finetune seed | AP@25 | AP@50 | AP@75 |
|---:|---:|---:|---:|
| 1 | 0.7718 | 0.5706 | 0.0924 |
| 2 | 0.8033 | 0.5279 | 0.0683 |
| 3 | 0.7573 | 0.3557 | 0.0001 |
| mean ± std | 0.7775 ± 0.0235 | 0.4847 ± 0.1138 | 0.0536 ± 0.0479 |

## Reading

- The original seed1 result was high on AP@50, but the 3-seed mean drops to `0.4847 ± 0.1138` because seed3 collapses.

- This does not support reverse scheduling as a positive method or a stable counterexample to the main structure-first curve.

- Safe framing: reverse is an unstable/negative control. It can produce a strong seed, but does not survive 3-seed validation and AP@75 is weak.

## Eval paths

- seed1: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_reverse_ramp_p1.0_e300_seed1_abci3reverse_det0_epoch300_sched_epoch_seed1_fcos1000_eval/eval.json`
- seed2: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_reverse_ramp_p1.0_e300_seed1_abci3reverse_det0_epoch300_sched_epoch_preseed1_ftseed2_fcos1000_eval/eval.json`
- seed3: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_reverse_ramp_p1.0_e300_seed1_abci3reverse_det0_epoch300_sched_epoch_preseed1_ftseed3_fcos1000_eval/eval.json`
