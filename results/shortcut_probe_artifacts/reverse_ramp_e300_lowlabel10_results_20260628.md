# Reverse ramp e300 low-label 10% results

- Pretrain: `reverse_ramp`, p1.0/e300, pretrain seed 1 (`abci3reverse_det0`)
- Finetune: FCOS, 10% labels, 1000 epochs, deterministic=0
- Selection/eval: wrapper default AP50-best eval JSON

| ft seed | AP@25 | AP@50 | AP@75 | eval |
|---:|---:|---:|---:|---|
| 1 | 0.538346 | 0.147526 | 0.000530 | `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_reverse_ramp_p1.0_e300_seed1_abci3reverse_det0_epoch300_sched_epoch_seed1_pt10_fcos1000_eval/eval.json` |
| 2 | 0.552525 | 0.175389 | 0.000223 | `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_reverse_ramp_p1.0_e300_seed1_abci3reverse_det0_epoch300_sched_epoch_preseed1_ftseed2_pt10_fcos1000_eval/eval.json` |
| 3 | 0.433783 | 0.100540 | 0.000000 | `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_reverse_ramp_p1.0_e300_seed1_abci3reverse_det0_epoch300_sched_epoch_preseed1_ftseed3_pt10_fcos1000_eval/eval.json` |

| metric | mean | std |
|---|---:|---:|
| AP25 | 0.508218 | 0.064851 |
| AP50 | 0.141152 | 0.037829 |
| AP75 | 0.000251 | 0.000266 |

