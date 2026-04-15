# NeRF-MAE Shortcut Probe Experiment Log

Last updated: 2026-04-11 JST

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
- completed on 2026-04-09
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

Results:

`scheduler-fixed` (`lr_scheduler=onecycle_epoch`) 3-seed mean:

| condition | AP@50 | AP@25 | AP@75 | Recall@50 top300 | AR top300 |
|---|---:|---:|---:|---:|---:|
| fair scratch | 0.3567 | 0.7520 | 0.0381 | 0.5784 | 0.4113 |
| baseline | 0.3416 | 0.7599 | 0.0123 | 0.5613 | 0.3993 |
| alpha_only | 0.3710 | 0.7537 | 0.0485 | 0.5931 | 0.4273 |
| masked_only | 0.3506 | 0.7523 | 0.0369 | 0.5760 | 0.4181 |
| alpha_shuffle | 0.3883 | 0.7580 | 0.0305 | 0.5931 | 0.4190 |

Reading:
- The scheduler change materially improves every condition relative to the previous quick transfer setup.
- The earlier story that `fair scratch` clearly dominates no longer holds under the scheduler-fixed protocol.
- `alpha_only` is now slightly above fair scratch on mean `AP@50`, but the margin is small.
- `alpha_shuffle` is also in the same band, so this still does not isolate preserved alpha layout as the decisive factor.
- `baseline` remains weaker on `AP@75`, and still has a relatively weak `seed3`, but it is no longer collapsing to the earlier degree.

Seed-3 diagnostics against the scheduler-fixed baseline:

| condition | AP@50 | AP@25 | AP@75 | Recall@50 top300 | AR top300 |
|---|---:|---:|---:|---:|---:|
| fair scratch sched-seed3 | 0.3918 | 0.7616 | 0.0408 | 0.6103 | 0.4167 |
| baseline sched-seed3 | 0.2380 | 0.7387 | 0.0004 | 0.4706 | 0.3475 |
| alpha_only sched-seed3 | 0.3901 | 0.7696 | 0.0530 | 0.6029 | 0.4309 |
| masked_only sched-seed3 | 0.3379 | 0.7389 | 0.0322 | 0.5588 | 0.4196 |
| fair scratch no-aug seed3 | 0.0573 | 0.3567 | 0.0000 | 0.1691 | 0.2059 |
| baseline no-aug seed3 | 0.1181 | 0.5538 | 0.0000 | 0.2574 | 0.2637 |
| alpha_only no-aug seed3 | 0.2270 | 0.6083 | 0.0036 | 0.4338 | 0.3348 |
| masked_only no-aug seed3 | 0.1219 | 0.5376 | 0.0000 | 0.2426 | 0.2475 |
| fair scratch freeze10 seed3 | 0.3605 | 0.7665 | 0.0433 | 0.5662 | 0.4260 |
| baseline freeze10 seed3 | 0.1559 | 0.6947 | 0.0000 | 0.3529 | 0.3250 |
| alpha_only freeze10 seed3 | 0.3638 | 0.7493 | 0.0292 | 0.5956 | 0.4167 |
| masked_only freeze10 seed3 | 0.3253 | 0.7701 | 0.0292 | 0.5515 | 0.4108 |

Reading:
- `no-aug` hurts every condition badly, so the seed-3 behavior is not explained by stochastic augmentation noise alone.
- `freeze_backbone_epochs=10` does not rescue baseline. It slightly lowers most conditions and leaves baseline clearly behind.
- The strongest current protocol-level effect is the scheduler fix, not the freeze/no-aug diagnostics.

Direct takeaway:
- The old quick-transfer instability story was substantially driven by the downstream optimization recipe.
- After fixing the scheduler, the five conditions cluster much more tightly.
- The current evidence supports a modest "reduced objectives remain competitive" story more than a strong "vanilla baseline is broken" story.

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

## Experiment 8: Alpha-Target-Only Follow-Up and Heavier Budget

Date:
- 2026-04-09 to 2026-04-10 JST

Goal:
- Re-read diagnostic dumps under the scheduler-fixed FCOS protocol
- Test `alpha_target_only`
- Recheck the main reduced-objective story under a heavier pretraining budget

Launch family:
- follow-up chain:
  - `/home/minesawa/ssl/NeRF-MAE/nerf_mae/probe_scripts/run_alpha_target_followup_chain.sh`
- helper scripts:
  - `/home/minesawa/ssl/NeRF-MAE/nerf_mae/probe_scripts/train_alpha_target_only.sh`
  - `/home/minesawa/ssl/NeRF-MAE/nerf_rpn/tools/summarize_diagnostic_dumps.py`

Protocol:
- FCOS uses `lr_scheduler=onecycle_epoch`
- `alpha_target_only` probe is:
  - `probe_mode=custom`
  - `probe_rgb_input=zero`
  - `probe_alpha_input=zero`
  - `probe_rgb_loss=none`
  - `probe_alpha_loss=removed`
- Main comparison remains Front3D FCOS transfer
- Pretrain checkpoint is `epoch_30.pt` for the `e30` line and `epoch_100.pt` for the heavy line

Diagnostic summary outputs:
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/analysis/sched_epoch_seed1_diagnostics_summary.md`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/analysis/sched_epoch_seed1_diagnostics_summary.json`

### Alpha-Target-Only, `p0.1`, `e30`, `epoch_30.pt`, 3 seeds

Per-seed results:

| condition | AP@50 | AP@25 | AP@75 | Recall@50 top300 |
|---|---:|---:|---:|---:|
| alpha_target seed1 | 0.3650 | 0.7532 | 0.0226 | 0.5809 |
| alpha_target seed2 | 0.4111 | 0.7538 | 0.0388 | 0.6103 |
| alpha_target seed3 | 0.3887 | 0.7504 | 0.0233 | 0.6324 |

3-seed mean:

| condition | AP@50 | AP@25 | AP@75 | Recall@50 top300 |
|---|---:|---:|---:|---:|
| fair scratch | 0.3567 | 0.7520 | 0.0381 | 0.5784 |
| baseline | 0.3416 | 0.7599 | 0.0123 | 0.5613 |
| alpha_only | 0.3710 | 0.7537 | 0.0485 | 0.5931 |
| masked_only | 0.3506 | 0.7523 | 0.0369 | 0.5760 |
| alpha_shuffle | 0.3883 | 0.7580 | 0.0305 | 0.5931 |
| alpha_target_only | 0.3883 | 0.7524 | 0.0282 | 0.6078 |

Eval files:
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_only_p0.1_e30_seed1_epoch30_sched_epoch_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_only_p0.1_e30_seed2_epoch30_sched_epoch_seed2_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_only_p0.1_e30_seed3_epoch30_sched_epoch_seed3_fcos100_eval/eval.json`

Reading:
- `alpha_target_only` stays in the same band as `alpha_shuffle`, and both are above fair scratch on mean `AP@50`.
- This weakens the story that preserving alpha input content, or preserving the correct alpha spatial layout at the encoder input, is necessary for transfer in the current quick regime.
- The current evidence is more compatible with a "full RGBA reconstruction is not the decisive factor" interpretation than with a tight causal claim about alpha layout.

### Heavier Pretraining Budget, `p0.1`, `e100`, `seed1`, `epoch_100.pt`

| condition | AP@50 | AP@25 | AP@75 | Recall@50 top300 |
|---|---:|---:|---:|---:|
| baseline e100 seed1 | 0.4227 | 0.7779 | 0.0249 | 0.6324 |
| alpha_only e100 seed1 | 0.4012 | 0.7941 | 0.0457 | 0.6471 |
| alpha_shuffle e100 seed1 | 0.3530 | 0.7014 | 0.0608 | 0.5735 |
| alpha_target_only e100 seed1 | 0.3993 | 0.7726 | 0.0296 | 0.6103 |

Eval files:
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_all_p0.1_e100_seed1_epoch100_sched_epoch_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_only_p0.1_e100_seed1_epoch100_sched_epoch_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_shuffle_p0.1_e100_seed1_epoch100_sched_epoch_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_only_p0.1_e100_seed1_epoch100_sched_epoch_seed1_fcos100_eval/eval.json`

Reading:
- With a heavier pretraining budget, the full baseline improves and becomes the best `AP@50` condition in this single-seed comparison.
- `alpha_only` remains competitive and is strongest on `AP@25` and `Recall@50 top300`.
- `alpha_target_only` remains close to `alpha_only`, which still argues against dense RGBA reconstruction being uniquely necessary.
- `alpha_shuffle` falls back at `e100`, so the earlier `e30` competitiveness of shuffled-alpha does not obviously survive a heavier budget.

### ScanNet

Status:
- skipped

Reason:
- `scannet_rpn_data` was not present under:
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/dataset/finetune/scannet_rpn_data`

Current takeaway after Experiment 8:
- Under the scheduler-corrected FCOS recipe, reduced objectives remain highly competitive with the full RGBA baseline at `e30`.
- `alpha_target_only` shows that strong transfer can persist even when both RGB and alpha inputs are zeroed and only target-side alpha prediction remains.
- However, the heavier `e100` result suggests the full baseline may recover under more training, so the cleanest current statement is:
  - full RGBA reconstruction is not the decisive factor in the quick regime
  - but sample efficiency versus asymptotic behavior remains unresolved

## Experiment 9: `e100` Multi-Seed And `alpha_target_shuffle`

Date:
- 2026-04-10 to 2026-04-11 JST

Goal:
- test whether the `e100` single-seed story survives replication
- add a target-side causal control for `alpha_target_only`

Launch family:
- chain:
  - `/home/minesawa/ssl/NeRF-MAE/nerf_mae/probe_scripts/run_e100_alpha_target_mechanism_chain.sh`
- target-side corruption support:
  - `/home/minesawa/ssl/NeRF-MAE/nerf_mae/model/mae/shortcut_probe.py`
- pretrain helper:
  - `/home/minesawa/ssl/NeRF-MAE/nerf_mae/probe_scripts/train_alpha_target_shuffle.sh`

Protocol:
- FCOS uses `lr_scheduler=onecycle_epoch`
- `e100` lines use `epoch_100.pt`
- `alpha_target_shuffle` is:
  - `probe_mode=custom`
  - `probe_rgb_input=zero`
  - `probe_alpha_input=zero`
  - `probe_alpha_target=shuffle`
  - `probe_rgb_loss=none`
  - `probe_alpha_loss=removed`

### `e100`, `p0.1`, 3 seeds

Per-seed results:

| condition | seed1 AP@50 | seed2 AP@50 | seed3 AP@50 | mean AP@50 | AP@50 std | mean AP@25 | mean AP@75 | mean Recall@50 top300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_e100 | 0.4227 | 0.4198 | 0.2707 | 0.3711 | 0.0710 | 0.7621 | 0.0255 | 0.5564 |
| alpha_only_e100 | 0.4012 | 0.3535 | 0.3775 | 0.3774 | 0.0194 | 0.7918 | 0.0394 | 0.6152 |
| alpha_target_only_e100 | 0.3993 | 0.4815 | 0.4296 | 0.4368 | 0.0340 | 0.7692 | 0.0293 | 0.6348 |

Eval files:
- baseline:
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_all_p0.1_e100_seed1_epoch100_sched_epoch_seed1_fcos100_eval/eval.json`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_all_p0.1_e100_seed2_epoch100_sched_epoch_seed2_fcos100_eval/eval.json`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_all_p0.1_e100_seed3_epoch100_sched_epoch_seed3_fcos100_eval/eval.json`
- alpha_only:
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_only_p0.1_e100_seed1_epoch100_sched_epoch_seed1_fcos100_eval/eval.json`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_only_p0.1_e100_seed2_epoch100_sched_epoch_seed2_fcos100_eval/eval.json`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_only_p0.1_e100_seed3_epoch100_sched_epoch_seed3_fcos100_eval/eval.json`
- alpha_target_only:
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_only_p0.1_e100_seed1_epoch100_sched_epoch_seed1_fcos100_eval/eval.json`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_only_p0.1_e100_seed2_epoch100_sched_epoch_seed2_fcos100_eval/eval.json`
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_only_p0.1_e100_seed3_epoch100_sched_epoch_seed3_fcos100_eval/eval.json`

Reading:
- The earlier `e100` single-seed impression that the full baseline recovers is not robust.
- On the 3-seed mean, `alpha_target_only_e100` is the best `AP@50` condition.
- `alpha_only_e100` is still strongest on `AP@25`.
- This strengthens the claim that dense full-RGBA reconstruction is not the decisive factor even beyond the quick `e30` regime.

### `alpha_target_shuffle`, `p0.1`, `e30`, 3 seeds

| condition | mean AP@50 | AP@50 std | mean AP@25 | mean AP@75 | mean Recall@50 top300 |
|---|---:|---:|---:|---:|---:|
| alpha_target_shuffle | 0.2913 | 0.0917 | 0.7526 | 0.0193 | 0.4828 |

Per-seed AP@50:
- seed1: `0.2888`
- seed2: `0.4049`
- seed3: `0.1802`

Eval files:
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_shuffle_p0.1_e30_seed1_epoch30_sched_epoch_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_shuffle_p0.1_e30_seed2_epoch30_sched_epoch_seed2_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_shuffle_p0.1_e30_seed3_epoch30_sched_epoch_seed3_fcos100_eval/eval.json`

Reading:
- Shuffling target-side alpha hurts relative to `alpha_target_only`.
- The variance is high, but this is the first result that points back toward target-side alpha structure mattering.
- The safest current phrasing is:
  - target-side alpha structure appears more important than encoder-side alpha layout preservation
  - but this still needs a heavier-budget confirmation

## Experiment 10: Low-Label Downstream

Date:
- 2026-04-10 to 2026-04-11 JST

Goal:
- test whether pretraining differences open up when downstream labeled data is reduced

Protocol:
- downstream FCOS only
- scheduler fixed to `onecycle_epoch`
- seed `1`
- pretrained conditions use `epoch_100.pt`
- compared:
  - `scratch`
  - `baseline_e100`
  - `alpha_target_only_e100`

### `percent_train = 0.1`

| condition | AP@50 | AP@25 | AP@75 | Recall@50 top300 | AR top300 |
|---|---:|---:|---:|---:|---:|
| scratch_pt01 | 0.0941 | 0.4287 | 0.0009 | 0.2353 | 0.2490 |
| baseline_pt01 | 0.1238 | 0.4868 | 0.0009 | 0.3309 | 0.2877 |
| alpha_target_only_pt01 | 0.0747 | 0.4646 | 0.0000 | 0.2868 | 0.2681 |

### `percent_train = 0.2`

| condition | AP@50 | AP@25 | AP@75 | Recall@50 top300 | AR top300 |
|---|---:|---:|---:|---:|---:|
| scratch_pt02 | 0.1106 | 0.5782 | 0.0018 | 0.2794 | 0.2853 |
| baseline_pt02 | 0.1828 | 0.5729 | 0.0007 | 0.4118 | 0.3196 |
| alpha_target_only_pt02 | 0.1910 | 0.5728 | 0.0000 | 0.4044 | 0.3113 |

Eval files:
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/front3d_scratch_samepath_sched_epoch_pt01_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/front3d_scratch_samepath_sched_epoch_pt02_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_all_p0.1_e100_seed1_epoch100_sched_epoch_pt01_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_all_p0.1_e100_seed1_epoch100_sched_epoch_pt02_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_only_p0.1_e100_seed1_epoch100_sched_epoch_pt01_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_only_p0.1_e100_seed1_epoch100_sched_epoch_pt02_seed1_fcos100_eval/eval.json`

Reading:
- At `10%` labels, `baseline_e100` is strongest.
- At `20%` labels, `baseline_e100` and `alpha_target_only_e100` are nearly tied, with a slight `AP@50` edge for `alpha_target_only`.
- Both pretrained conditions are clearly above scratch in low-label transfer.
- This is the strongest evidence so far that the reduced objectives are not just matching full supervision in full-data transfer, but remain useful when downstream data is scarce.

## Current Best Reading

- The main story has shifted from "`baseline` is broken" to "full RGBA reconstruction is not the decisive factor."
- Under the scheduler-corrected FCOS protocol, reduced objectives remain highly competitive with the full baseline at `e30`, and `alpha_target_only` is strongest at `e100` on the 3-seed mean.
- `alpha_target_shuffle` weakens performance relative to `alpha_target_only`, so target-side alpha structure now looks more important than encoder-side alpha layout preservation.
- Low-label transfer also supports the usefulness of reduced objectives: both `baseline_e100` and `alpha_target_only_e100` beat scratch, and `alpha_target_only` remains competitive at `20%` labels.
- What is still unresolved is asymptotic behavior at much larger pretraining budgets and whether the target-side alpha effect survives heavier replication.

## Experiment 11: Target-Side Alpha Structure Follow-Up

Date:
- 2026-04-11 to 2026-04-12 JST

Goal:
- test whether `alpha_target_shuffle` still drops under `e100`
- add `alpha_target_zero` as a target-side causal control
- regenerate proposal/voxel-score diagnostics for the target-side comparison

Launch family:
- chain:
  - `/home/minesawa/ssl/NeRF-MAE/nerf_mae/probe_scripts/run_target_alpha_structure_chain.sh`
- helper:
  - `/home/minesawa/ssl/NeRF-MAE/nerf_mae/probe_scripts/train_alpha_target_zero.sh`

Protocol:
- FCOS uses `lr_scheduler=onecycle_epoch`
- `alpha_target_shuffle_e100` uses `epoch_100.pt`
- `alpha_target_zero_e30` uses `epoch_30.pt`
- all results below are Front3D FCOS transfer with seed-matched pretrain and downstream seeds

### `alpha_target_shuffle`, `p0.1`, `e100`, 3 seeds

| seed | AP@50 | AP@25 | AP@75 | Recall@50 top300 |
|---|---:|---:|---:|---:|
| 1 | 0.3904 | 0.7647 | 0.0278 | 0.6029 |
| 2 | 0.3907 | 0.7627 | 0.0118 | 0.6250 |
| 3 | 0.1665 | 0.6794 | 0.0000 | 0.3971 |
| mean | 0.3159 | 0.7356 | 0.0132 | 0.5417 |

Eval files:
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_shuffle_p0.1_e100_seed1_epoch100_sched_epoch_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_shuffle_p0.1_e100_seed2_epoch100_sched_epoch_seed2_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_shuffle_p0.1_e100_seed3_epoch100_sched_epoch_seed3_fcos100_eval/eval.json`

### `alpha_target_zero`, `p0.1`, `e30`, 3 seeds

| seed | AP@50 | AP@25 | AP@75 | Recall@50 top300 |
|---|---:|---:|---:|---:|
| 1 | 0.3825 | 0.7576 | 0.0524 | 0.5441 |
| 2 | 0.3343 | 0.7383 | 0.0190 | 0.5221 |
| 3 | 0.3703 | 0.7439 | 0.0434 | 0.5956 |
| mean | 0.3624 | 0.7466 | 0.0383 | 0.5539 |

Eval files:
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_zero_p0.1_e30_seed1_epoch30_sched_epoch_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_zero_p0.1_e30_seed2_epoch30_sched_epoch_seed2_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_zero_p0.1_e30_seed3_epoch30_sched_epoch_seed3_fcos100_eval/eval.json`

Diagnostics summary:
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/analysis/nerfmae_target_alpha_structure_diagnostics_summary.md`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/analysis/nerfmae_target_alpha_structure_diagnostics_summary.json`

Reading:
- `alpha_target_shuffle_e100` remains weaker than `alpha_target_only_e100` on mean AP@50 (`0.3159` vs `0.4368` from Experiment 9).
- `alpha_target_zero_e30` does not collapse; it remains close to the competitive reduced-objective band at `0.3624` AP@50.
- This means target-side alpha structure matters in the `shuffle` control, but the `zero` control complicates a simple `keep > shuffle > zero` causal chain.
- The most conservative current phrasing is:
  - visible RGBA input is not necessary for strong transfer in this protocol
  - target-side alpha corruption by shuffle hurts
  - but target-side zeroing does not fully destroy transfer, so architecture/position bias and simplified target supervision remain plausible contributors

## Experiment 12: Tiny-RGB and `alpha_target_zero_e100` Follow-Up

Date:
- 2026-04-14 to 2026-04-15 JST

Goal:
- test whether a small auxiliary RGB loss improves the `alpha_target_only` style objective
- complete the heavier-budget `alpha_target_zero`, `p0.1`, `e100` causal control

Launch family:
- chain:
  - `/home/minesawa/ssl/NeRF-MAE/nerf_mae/probe_scripts/run_tiny_rgb_and_zero_followup_chain.sh`
- helper:
  - `/home/minesawa/ssl/NeRF-MAE/nerf_mae/probe_scripts/train_alpha_target_tiny_rgb.sh`

Protocol:
- FCOS uses `lr_scheduler=onecycle_epoch`
- tiny-RGB uses `probe_rgb_input=zero`, `probe_alpha_input=zero`, `probe_alpha_target=keep`, `probe_rgb_loss=removed_occupied`, `probe_alpha_loss=removed`
- tiny-RGB weights are `probe_rgb_weight in {0.02, 0.05, 0.1}`, with `probe_alpha_weight=1.0`
- tiny-RGB uses `p0.1`, `e30`, seed 1, checkpoint `epoch_30.pt`
- `alpha_target_zero_e100` uses `probe_rgb_input=zero`, `probe_alpha_input=zero`, `probe_alpha_target=zero`, `probe_rgb_loss=none`, `probe_alpha_loss=removed`
- `alpha_target_zero_e100` uses `p0.1`, `e100`, 3 seeds, checkpoint `epoch_100.pt`

### Tiny-RGB, `p0.1`, `e30`, seed 1

Full-label Front3D FCOS:

| condition | AP@50 | AP@25 | AP@75 | Recall@50 top300 |
|---|---:|---:|---:|---:|
| tiny-RGB, lambda=0.02 | 0.4068 | 0.7736 | 0.0397 | 0.5809 |
| tiny-RGB, lambda=0.05 | 0.3606 | 0.7470 | 0.0558 | 0.5809 |
| tiny-RGB, lambda=0.10 | 0.3749 | 0.7651 | 0.0593 | 0.5956 |

20% label Front3D FCOS:

| condition | AP@50 | AP@25 | AP@75 | Recall@50 top300 |
|---|---:|---:|---:|---:|
| tiny-RGB, lambda=0.02 | 0.2059 | 0.5798 | 0.0012 | 0.3750 |
| tiny-RGB, lambda=0.05 | 0.1612 | 0.5703 | 0.0000 | 0.3897 |
| tiny-RGB, lambda=0.10 | 0.1228 | 0.5404 | 0.0017 | 0.3456 |

Eval files:
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_tiny_rgb_w0p02_p0.1_e30_seed1_epoch30_sched_epoch_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_tiny_rgb_w0p05_p0.1_e30_seed1_epoch30_sched_epoch_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_tiny_rgb_w0p1_p0.1_e30_seed1_epoch30_sched_epoch_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_tiny_rgb_w0p02_p0.1_e30_seed1_epoch30_sched_epoch_pt02_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_tiny_rgb_w0p05_p0.1_e30_seed1_epoch30_sched_epoch_pt02_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_tiny_rgb_w0p1_p0.1_e30_seed1_epoch30_sched_epoch_pt02_seed1_fcos100_eval/eval.json`

### `alpha_target_zero`, `p0.1`, `e100`, 3 seeds

| seed | AP@50 | AP@25 | AP@75 | Recall@50 top300 |
|---|---:|---:|---:|---:|
| 1 | 0.3221 | 0.7863 | 0.0418 | 0.5294 |
| 2 | 0.4119 | 0.7395 | 0.0302 | 0.6397 |
| 3 | 0.1872 | 0.7040 | 0.0000 | 0.3750 |
| mean | 0.3071 | 0.7433 | 0.0240 | 0.5147 |

Eval files:
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_zero_p0.1_e100_seed1_epoch100_sched_epoch_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_zero_p0.1_e100_seed2_epoch100_sched_epoch_seed2_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_target_zero_p0.1_e100_seed3_epoch100_sched_epoch_seed3_fcos100_eval/eval.json`

Reading:
- The `alpha_target_zero_e100` mean AP@50 is `0.3071`, below `alpha_target_only_e100` (`0.4368`), `alpha_only_e100` (`0.3774`), baseline e100 (`0.3711`), and scratch (`0.3567`).
- Unlike the `e30` zero control, the heavier `e100` zero control clearly weakens transfer. This supports the idea that target-side alpha supervision is not just an arbitrary regularizer; preserving a meaningful target matters at heavier budget.
- `alpha_target_shuffle_e100` and `alpha_target_zero_e100` are both weak relative to `alpha_target_only_e100` and are close to each other on AP@50 (`0.3159` vs `0.3071`).
- Tiny-RGB seed-1 sweep suggests small RGB help is not monotonic. Lambda `0.02` is best among the three in both full-label and 20% label settings, while larger RGB weights hurt the 20% label result.
- Tiny-RGB is still single-seed, so it should be treated as a candidate selection result rather than a final method claim.

## Update Rules

When adding a new experiment to this file:
- write the exact launch setting
- record the exact `eval.json` path
- state whether the pretrain checkpoint is `model_best.pt` or `epoch_k.pt`
- note any fairness caveat
- add the result to `Summary Table` only if it is directly comparable to the current main line
