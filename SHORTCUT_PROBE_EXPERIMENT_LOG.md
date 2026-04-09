# NeRF-MAE Shortcut Probe Experiment Log

Last updated: 2026-04-08 JST

This file is the running log for the NeRF-MAE shortcut probe experiments.
Primary result root: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output`

## Scope

- Task: Front3D downstream 3D object detection via FCOS
- Main question: whether NeRF-MAE transfer gains come from semantic pretraining or from shortcut-heavy occupancy/layout signals
- Main probe variants:
  - `baseline`
  - `alpha_only`
  - `radiance_only`
  - `masked_only_rgb_loss`
  - `fair scratch`

## Important Caveat

For `alpha_only` and `radiance_only`, pretraining `model_best.pt` is selected by validation PSNR on RGB reconstruction, which is misaligned with those probe objectives.
Because of that, the more reliable e30 comparison is the follow-up run that fixes the pretrain checkpoint to `epoch_30.pt`.

## Current Best Reading

- The checkpoint-selection confound was real.
- After fixing that confound with `epoch_30.pt`, `alpha_only_e30` becomes the strongest pretrained condition on `AP@50`.
- `masked_only_rgb_loss` is weak at both e10 and e30, which supports the claim that the original RGB supervision path matters for the learned encoder representation.
- `baseline` does not consistently beat fair scratch in this quick setting.

## Experiment 1: Quick Pretraining Only

Setup:
- pretrain data fraction: `0.1`
- epochs: `10`
- output root: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_mae/results`
- launcher log: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_shortcut_probe_quick.chain.log`

| variant | pretrain ckpt | PSNR | MSE |
|---|---|---:|---:|
| baseline | `nerfmae_all_p0.1_e10/model_best.pt` | 16.8566 | 0.02094 |
| alpha_only | `nerfmae_alpha_only_p0.1_e10/model_best.pt` | 4.2252 | 0.37916 |
| radiance_only | `nerfmae_radiance_only_p0.1_e10/model_best.pt` | 2.0878 | 0.62126 |
| masked_only_rgb_loss | `nerfmae_masked_only_rgb_loss_p0.1_e10/model_best.pt` | 11.2296 | 0.07593 |

Notes:
- These are reconstruction-side metrics only.
- They do not track downstream transfer quality cleanly.

## Experiment 2: Quick Transfer, e10, `model_best.pt`

Setup:
- pretrain data fraction: `0.1`
- pretrain epochs: `10`
- FCOS finetuning: `100 epochs`
- fair scratch uses the same `run_fcos_pretrained.py` codepath

Result files:
- fair scratch: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/front3d_scratch_samepath_fcos100_eval/eval.json`
- baseline: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_all_p0.1_e10_fcos100_eval/eval.json`
- alpha_only: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_only_p0.1_e10_fcos100_eval/eval.json`
- radiance_only: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_radiance_only_p0.1_e10_fcos100_eval/eval.json`
- masked_only_rgb_loss: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_masked_only_rgb_loss_p0.1_e10_fcos100_eval/eval.json`

| condition | AP@75 | AP@50 | AP@25 | Recall@50 | Recall@25 |
|---|---:|---:|---:|---:|---:|
| fair scratch | 0.0020 | 0.2059 | 0.6270 | 0.4118 | 0.9412 |
| baseline_e10 | 0.0084 | 0.2318 | 0.6096 | 0.5147 | 0.9559 |
| alpha_only_e10 | 0.0021 | 0.2422 | 0.6274 | 0.4559 | 0.9485 |
| radiance_only_e10 | 0.0117 | 0.2034 | 0.6188 | 0.4191 | 0.9559 |
| masked_only_e10 | 0.0092 | 0.1476 | 0.5529 | 0.4412 | 0.9485 |

Reading:
- `alpha_only_e10` is competitive with `baseline_e10`.
- `masked_only_e10` is clearly weak.
- This already suggested that the RGB supervision path matters.

## Experiment 3: e30 Transfer, `model_best.pt`

Setup:
- pretrain data fraction: `0.1`
- pretrain epochs: `30`
- FCOS finetuning: `100 epochs`
- auto summary files:
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_shortcut_probe_30ep_fcos3way.csv`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_shortcut_probe_30ep_fcos3way.json`

Result files:
- baseline: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_all_p0.1_e30_fcos100_eval/eval.json`
- alpha_only: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_only_p0.1_e30_fcos100_eval/eval.json`
- radiance_only: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_radiance_only_p0.1_e30_fcos100_eval/eval.json`

| condition | AP@75 | AP@50 | AP@25 |
|---|---:|---:|---:|
| baseline_e30_modelbest | 0.0019 | 0.1899 | 0.6157 |
| alpha_only_e30_modelbest | 0.0089 | 0.2093 | 0.6007 |
| radiance_only_e30_modelbest | 0.0103 | 0.2159 | 0.5644 |

Reading:
- This table is useful for history only.
- It is not the clean comparison because `model_best.pt` is PSNR-selected.

## Experiment 4: Follow-up e30, `epoch_30.pt` Fixed

This is the current main result.

Setup:
- same pretraining runs as Experiment 3
- pretrain checkpoint fixed to `epoch_30.pt`
- chain log: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_epoch30_followup_debug.chain.log`
- per-run logs:
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_epoch30_followup_debug.baseline_epoch30.log`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_epoch30_followup_debug.alpha_epoch30.log`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_epoch30_followup_debug.radiance_epoch30.log`

Result files:
- baseline: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_all_p0.1_e30_epoch30_fcos100_eval/eval.json`
- alpha_only: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_only_p0.1_e30_epoch30_fcos100_eval/eval.json`
- radiance_only: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_radiance_only_p0.1_e30_epoch30_fcos100_eval/eval.json`

| condition | AP@75 | AP@50 | AP@25 | Recall@50 | Recall@25 |
|---|---:|---:|---:|---:|---:|
| fair scratch | 0.0020 | 0.2059 | 0.6270 | 0.4118 | 0.9412 |
| baseline_e30_epoch30 | 0.0074 | 0.1929 | 0.6303 | 0.4265 | 0.9485 |
| alpha_only_e30_epoch30 | 0.0203 | 0.2271 | 0.6278 | 0.4485 | 0.9485 |
| radiance_only_e30_epoch30 | 0.0012 | 0.1932 | 0.6040 | 0.3971 | 0.9559 |

Reading:
- `alpha_only_e30_epoch30` is the strongest pretrained condition on `AP@50`.
- `alpha_only_e30_epoch30` beats fair scratch on `AP@50`.
- `radiance_only_e30_epoch30` drops relative to the earlier `model_best` comparison.
- This means the checkpoint-selection confound was not small; it changed the ordering.

## Experiment 5: `masked_only_rgb_loss_e30`

Setup:
- pretrain data fraction: `0.1`
- pretrain epochs: `30`
- pretrain log: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_epoch30_followup_debug.masked_pretrain_e30.log`
- FCOS log: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_epoch30_followup_debug.masked_epoch30_fcos.log`

Artifacts:
- pretrain checkpoint: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_mae/results/nerfmae_masked_only_rgb_loss_p0.1_e30/epoch_30.pt`
- eval file: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_masked_only_rgb_loss_p0.1_e30_epoch30_fcos100_eval/eval.json`

Pretrain-side metric:
- `PSNR = 3.5824`
- `MSE = 0.44232`

Downstream result:

| condition | AP@75 | AP@50 | AP@25 | Recall@50 | Recall@25 |
|---|---:|---:|---:|---:|---:|
| masked_only_e30_epoch30 | 0.0001 | 0.1812 | 0.6221 | 0.4485 | 0.9485 |

Reading:
- `masked_only_e30_epoch30` is below fair scratch on `AP@50`.
- It is also below `alpha_only_e30_epoch30`.
- This strengthens the claim that cutting the original RGB supervision path hurts transfer.

## Experiment 6: 3-Seed Replication And `alpha_shuffle`

Setup:
- downstream only for seed replication:
  - `fair scratch`
  - `baseline_e30_epoch30`
  - `alpha_only_e30_epoch30`
  - `masked_only_e30_epoch30`
- seeds: `1, 2, 3`
- `alpha_shuffle`:
  - pretrain: `p0.1`, `30 epochs`, `epoch_30.pt`
  - probe config:
    - `probe_mode=custom`
    - `probe_rgb_input=zero`
    - `probe_alpha_input=shuffle`
    - `probe_rgb_loss=none`
    - `probe_alpha_loss=removed`

Result files:
- `fair scratch`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/front3d_scratch_samepath_seed1_fcos100_eval/eval.json`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/front3d_scratch_samepath_seed2_fcos100_eval/eval.json`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/front3d_scratch_samepath_seed3_fcos100_eval/eval.json`
- `baseline_e30_epoch30`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_all_p0.1_e30_epoch30_seed1_fcos100_eval/eval.json`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_all_p0.1_e30_epoch30_seed2_fcos100_eval/eval.json`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_all_p0.1_e30_epoch30_seed3_fcos100_eval/eval.json`
- `alpha_only_e30_epoch30`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_only_p0.1_e30_epoch30_seed1_fcos100_eval/eval.json`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_only_p0.1_e30_epoch30_seed2_fcos100_eval/eval.json`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_only_p0.1_e30_epoch30_seed3_fcos100_eval/eval.json`
- `masked_only_e30_epoch30`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_masked_only_rgb_loss_p0.1_e30_epoch30_seed1_fcos100_eval/eval.json`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_masked_only_rgb_loss_p0.1_e30_epoch30_seed2_fcos100_eval/eval.json`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_masked_only_rgb_loss_p0.1_e30_epoch30_seed3_fcos100_eval/eval.json`
- `alpha_shuffle`
  - pretrain ckpt: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_mae/results/nerfmae_alpha_shuffle_p0.1_e30_seed1/epoch_30.pt`
  - eval: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_shuffle_p0.1_e30_seed1_epoch30_seed1_fcos100_eval/eval.json`

3-seed downstream summary:

| condition | seed1 AP@50 | seed2 AP@50 | seed3 AP@50 | mean AP@50 | mean AP@25 | mean Recall@50 top300 |
|---|---:|---:|---:|---:|---:|---:|
| fair scratch | 0.2013 | 0.1690 | 0.2530 | 0.2078 | 0.6117 | 0.4412 |
| baseline_e30_epoch30 | 0.2276 | 0.1672 | 0.0391 | 0.1446 | 0.6340 | 0.3652 |
| alpha_only_e30_epoch30 | 0.1695 | 0.2092 | 0.1938 | 0.1908 | 0.6103 | 0.3873 |
| masked_only_e30_epoch30 | 0.2029 | 0.1797 | 0.2055 | 0.1960 | 0.6416 | 0.4559 |

`alpha_shuffle` single-seed downstream result:

| condition | AP@50 | AP@25 | Recall@50 top300 |
|---|---:|---:|---:|
| alpha_shuffle | 0.1970 | 0.6281 | 0.3529 |

Reading:
- The earlier single-seed impression that `alpha_only` clearly beats scratch is not robust after 3 seeds.
- `fair scratch` has the highest mean `AP@50` among the four seeded conditions.
- `baseline_e30_epoch30` is unstable. `seed3` collapses strongly.
- `masked_only_e30_epoch30` no longer looks consistently worse than baseline; on the 3-seed mean it is actually above baseline.
- `alpha_shuffle` lands close to `alpha_only`, so current evidence does not yet isolate "correct alpha spatial layout" as the decisive factor.

## Summary Table

This is the current table to cite first.

| condition | mean AP@50 | mean AP@25 | mean Recall@50 top300 | note |
|---|---:|---:|---:|---|
| fair scratch | 0.2078 | 0.6117 | 0.4412 | same FCOS codepath as pretrained runs |
| baseline_e30_epoch30 | 0.1446 | 0.6340 | 0.3652 | unstable across seeds |
| alpha_only_e30_epoch30 | 0.1908 | 0.6103 | 0.3873 | near scratch, but not above it on mean |
| masked_only_e30_epoch30 | 0.1960 | 0.6416 | 0.4559 | near scratch, above baseline on mean |
| alpha_shuffle | 0.1970 | 0.6281 | 0.3529 | single seed only |

## What We Can Safely Say Now

- The checkpoint-selection confound was real. Using `epoch_30.pt` changed the ordering relative to `model_best.pt`.
- In the current quick setting, there is no robust pretraining advantage over fair scratch.
- `baseline_e30_epoch30` is unstable enough that single-seed readings are not trustworthy.
- `alpha_only` and `masked_only_rgb_loss` both retain substantial downstream utility, but neither has shown a robust advantage over fair scratch.
- The earlier single-seed claim that `masked_only_rgb_loss` clearly hurts relative to baseline does not survive 3-seed replication.
- `alpha_shuffle` is too close to `alpha_only` to support a strong "alpha spatial layout is the whole story" claim.
- Reconstruction-side quality and downstream transfer quality are not moving together in a simple way.

## Next Recommended Experiments

1. Diagnose why `baseline_e30_epoch30` collapses on `seed3`.
2. Add seeds for `alpha_shuffle`.
3. Re-read the existing diagnostic dumps with the new 3-seed interpretation in mind.
4. Keep using `epoch_k.pt` fixed checkpoints for probe variants when the pretrain objective is not aligned with RGB PSNR.

## Experiment 7: Downstream Protocol Diagnosis Chain

Status:
- launched on 2026-04-09
- tmux session: `nerfmae_downstream_protocol_diagnosis_chain`
- chain script:
  - `/home/minesawa/ssl/NeRF-MAE/nerf_mae/probe_scripts/run_downstream_protocol_diagnosis_chain.sh`

Planned phases:
1. scheduler-fixed reevaluation
   - `lr_scheduler=onecycle_epoch`
   - `fair scratch / baseline_e30_epoch30 / alpha_only_e30_epoch30 / masked_only_e30_epoch30`
   - `seed=1,2,3`
2. deterministic + no-aug diagnostic
   - same four conditions
   - `seed=3`
   - `rotate_prob=0`, `flip_prob=0`, `rot_scale_prob=0`
3. `alpha_shuffle` multi-seed
   - pretrain seeds `1,2,3`
   - FCOS reevaluation with `lr_scheduler=onecycle_epoch`
4. freeze-backbone diagnostic
   - same four conditions
   - `seed=3`
   - `freeze_backbone_epochs=10`

Implementation notes:
- FCOS now accepts:
  - `--lr_scheduler`
  - `--scheduler_total_steps`
  - `--scheduler_min_lr`
  - `--freeze_backbone_epochs`
  - `--backbone_lr_scale`
- Shell launchers now also forward:
  - `ROTATE_PROB`
  - `FLIP_PROB`
  - `ROT_SCALE_PROB`
- `DETERMINISTIC=1` now exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python launch in both pretrain and FCOS scripts.

## Utility Scripts

- Seeded pretraining / transfer entrypoints:
  - `/home/minesawa/ssl/NeRF-MAE/nerf_mae/train_mae3d.sh`
  - `/home/minesawa/ssl/NeRF-MAE/nerf_rpn/train_fcos_pretrained.sh`
  - `/home/minesawa/ssl/NeRF-MAE/nerf_rpn/test_fcos_pretrained.sh`
- These now accept:
  - `SEED`
  - `DETERMINISTIC=1`
- `train_mae3d.sh` also forwards custom probe controls:
  - `PROBE_RGB_INPUT`
  - `PROBE_ALPHA_INPUT`
  - `PROBE_RGB_LOSS`
  - `PROBE_ALPHA_LOSS`
  - `PROBE_ALPHA_THRESHOLD`
- Diagnostic dump scripts:
  - `/home/minesawa/ssl/NeRF-MAE/nerf_rpn/run_fcos_diagnostic_variant.sh`
  - `/home/minesawa/ssl/NeRF-MAE/nerf_mae/probe_scripts/run_shortcut_diagnostic_dump_chain.sh`
- Downstream protocol diagnosis chain:
  - `/home/minesawa/ssl/NeRF-MAE/nerf_mae/probe_scripts/run_downstream_protocol_diagnosis_chain.sh`

The diagnostic dump chain writes raw analysis artifacts under:
- `.../front3d_scratch_samepath_fcos100_diagnostics`
- `.../nerfmae_all_p0.1_e30_epoch30_fcos100_diagnostics`
- `.../nerfmae_alpha_only_p0.1_e30_epoch30_fcos100_diagnostics`
- `.../nerfmae_masked_only_rgb_loss_p0.1_e30_epoch30_fcos100_diagnostics`

Each diagnostic directory is expected to contain:
- `proposals/*.npz`
- `voxel_scores/*.npz`
- `eval.json`

## Update Rules

When adding a new experiment to this file:
- write the exact launch setting
- record the exact `eval.json` path
- state whether the pretrain checkpoint is `model_best.pt` or `epoch_k.pt`
- note any fairness caveat
- add the result to `Summary Table` only if it is directly comparable to the current main line
