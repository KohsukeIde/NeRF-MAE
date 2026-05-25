# NeRF-MAE Shortcut Probe Experiment Log

Last updated: 2026-05-25 JST

This file is the running log for the NeRF-MAE shortcut probe experiments.
Primary result root: `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output`
Current ABCI3 workspace root: `/groups/gag51404/ide/vgi/NeRF-MAE`

## Scope

- Task: Front3D downstream 3D object detection via FCOS
- Main question: whether the NeRF-MAE pretraining gain can be explained by alpha/occupancy structure, and whether an explicit alpha-to-RGBA curriculum improves sample efficiency.
- Historical probe variants:
  - `baseline`
  - `alpha_only`
  - `radiance_only`
  - `masked_only_rgb_loss`
  - `fair scratch`
- Current method/control pair:
  - `cosine_ramp`: alpha+RGBA input, alpha target kept, RGB loss ramped from 0 to 1 with a cosine schedule
  - `cosine_ramp_alpha_shuffle`: same schedule, but alpha target shuffled as a structure-destroying control

## Important Caveat

For `alpha_only` and `radiance_only`, pretraining `model_best.pt` is selected by validation PSNR on RGB reconstruction, which is misaligned with those probe objectives.
Because of that, the more reliable e30 comparison is the follow-up run that fixes the pretrain checkpoint to `epoch_30.pt`.

## Current Best Reading

- The early shortcut-probe result was useful, but the strongest current paper
  direction is now the alpha-to-RGBA curriculum and structure/appearance
  decomposition rather than the original `alpha_only` claim.
- The clean ABCI3 e300 gate now has `1` pretrain seed and `3` downstream
  finetune seeds. `cosine_ramp e300` beats same-budget `baseline e300` in all
  three paired finetune seeds, with mean AP@50 `0.5723` vs `0.4938`
  (`+0.0785`). It also beats `cosine_ramp_alpha_shuffle e300` in all three
  paired finetune seeds, with mean AP@50 `0.5723` vs `0.4294` (`+0.1430`).
- This is no longer just a single-downstream-seed candidate, but it is still
  only one pretrain seed. Use it as a strong Phase-1 signal, not a final
  paper-scale claim about pretrain-seed robustness.
- Coordinate-only alpha prior is not strong enough to explain the downstream
  gain by itself. D-MAE scouts are active; `hierarchical_concat` is the current
  best D-MAE scout but must still beat or match the strong
  `cosine_coord_jitter` empirical control to become the main method.

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

## Experiment 13: Tiny-RGB w0p02 e100 Gate

Date:
- auto-appended by `run_tiny_rgb_w0p02_e100_gate_chain.sh`

Goal:
- promote the best tiny-RGB candidate from the e30 seed-1 sweep to the `p0.1`, `e100`, 3-seed gate

Protocol:
- pretrain: `probe_rgb_input=zero`, `probe_alpha_input=zero`, `probe_alpha_target=keep`, `probe_rgb_loss=removed_occupied`, `probe_alpha_loss=removed`
- weights: `probe_rgb_weight=0.02`, `probe_alpha_weight=1.0`
- pretrain budget: `percent_train=0.1`, `epochs=100`, checkpoint `epoch_100.pt`
- downstream: Front3D FCOS, `FCOS_NUM_EPOCHS=100`, `LR_SCHEDULER=onecycle_epoch`, seeds `1,2,3`

### Full-label Front3D FCOS

| seed | AP@50 | AP@25 | AP@75 | Recall@50 top300 |
|---|---:|---:|---:|---:|
| 1 | 0.4516 | 0.7668 | 0.0340 | 0.5956 |
| 2 | 0.4029 | 0.7500 | 0.0281 | 0.5956 |
| 3 | 0.3947 | 0.7719 | 0.0371 | 0.5956 |
| mean | 0.4164 | 0.7629 | 0.0330 | 0.5956 |
| std | 0.0308 | 0.0115 | 0.0046 | 0.0000 |

Eval files:
- `/mnt/urashima/users/minesawa/home-offload/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_target_tiny_rgb_w0p02_p0.1_e100_seed1_epoch100_sched_epoch_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/home-offload/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_target_tiny_rgb_w0p02_p0.1_e100_seed2_epoch100_sched_epoch_seed2_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/home-offload/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_target_tiny_rgb_w0p02_p0.1_e100_seed3_epoch100_sched_epoch_seed3_fcos100_eval/eval.json`

### 20% label Front3D FCOS

| seed | AP@50 | AP@25 | AP@75 | Recall@50 top300 |
|---|---:|---:|---:|---:|
| 1 | 0.1956 | 0.5818 | 0.0006 | 0.3824 |
| 2 | 0.1893 | 0.5912 | 0.0000 | 0.3529 |
| 3 | 0.1177 | 0.5692 | 0.0006 | 0.3456 |
| mean | 0.1675 | 0.5808 | 0.0004 | 0.3603 |
| std | 0.0433 | 0.0110 | 0.0003 | 0.0195 |

Eval files:
- `/mnt/urashima/users/minesawa/home-offload/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_target_tiny_rgb_w0p02_p0.1_e100_seed1_epoch100_sched_epoch_pt02_seed1_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/home-offload/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_target_tiny_rgb_w0p02_p0.1_e100_seed2_epoch100_sched_epoch_pt02_seed2_fcos100_eval/eval.json`
- `/mnt/urashima/users/minesawa/home-offload/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_target_tiny_rgb_w0p02_p0.1_e100_seed3_epoch100_sched_epoch_pt02_seed3_fcos100_eval/eval.json`

Reading:
- Full-label mean AP@50 is `0.4164`; compare against the current e100 means: baseline `0.3711`, alpha_target_only `0.4368`.
- 20% label mean AP@50 is `0.1675`; compare against the current seed-1 references: baseline e100 `0.1828`, alpha_target_only e100 `0.1910`, tiny-RGB e30 `0.2059`.
- This is the method-candidate gate; if it is competitive with alpha_target_only and improves low-label stability, it should be promoted to paper-budget scout.

## Experiment 14: Paper-Budget Scout

Date:
- 2026-04-16 to 2026-04-25 JST

Goal:
- test whether the low-budget `alpha_target_only` advantage survives near-paper budget
- compare controlled self-trained `baseline` and `alpha_target_only` under the same fork, data staging, downstream codepath, and checkpoint rule

Launch script:
- `/home/minesawa/ssl/NeRF-MAE/nerf_mae/probe_scripts/run_paper_budget_scout_chain.sh`

Protocol:
- pretrain dataset: NeRF-MAE pretrain split, `percent_train=1.0`
- pretrain budget: `epochs=1200`, `seed=1`, `GPU_IDS=0,1,2,3`, `BATCH_SIZE_PER_GPU=4`
- downstream dataset: Front3D FCOS
- downstream budget: `FCOS_NUM_EPOCHS=1000`, `seed=1`, `LR_SCHEDULER=onecycle_epoch`
- downstream checkpoint input: pretrain `epoch_1200.pt`
- downstream selection: best FCOS checkpoint by validation `AP@50`, then test eval

Pretrain checkpoints:
- baseline:
  - `/mnt/urashima/users/minesawa/home-offload/ssl/NeRF-MAE/output/nerf_mae/results/nerfmae_all_p1.0_e1200_seed1/epoch_1200.pt`
- alpha_target_only:
  - `/mnt/urashima/users/minesawa/home-offload/ssl/NeRF-MAE/output/nerf_mae/results/nerfmae_alpha_target_only_p1.0_e1200_seed1/epoch_1200.pt`

### Front3D FCOS Test Results

| condition | AP@50 | AP@25 | AP@75 | Recall@50 top300 | Recall@25 top300 |
|---|---:|---:|---:|---:|---:|
| baseline, continuous FCOS e1000 | 0.5892 | 0.8494 | 0.1469 | 0.7132 | 0.9485 |
| alpha_target_only, original run best before crash | 0.4251 | 0.7631 | 0.0472 | 0.6250 | 0.9412 |
| alpha_target_only, resume-from785 best | 0.4602 | 0.7198 | 0.0729 | 0.6324 | 0.9265 |

Eval files:
- baseline:
  - `/mnt/urashima/users/minesawa/home-offload/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_all_p1.0_e1200_seed1_epoch1200_sched_epoch_seed1_fcos1000_eval/eval.json`
- alpha_target_only, original run best before crash:
  - `/mnt/urashima/users/minesawa/home-offload/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_target_only_p1.0_e1200_seed1_epoch1200_sched_epoch_seed1_fcos1000_eval/eval.json`
- alpha_target_only, resume-from785 best:
  - `/mnt/urashima/users/minesawa/home-offload/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_target_only_p1.0_e1200_seed1_epoch1200_sched_epoch_seed1_fcos1000_resume_from785_eval/eval.json`

Run logs:
- chain:
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_paper_budget_scout.chain.log`
- baseline FCOS:
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_paper_budget_scout.nerfmae_all_p1.0_e1200_seed1_epoch1200_sched_epoch_seed1_fcos1000.log`
- alpha_target_only FCOS original:
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_paper_budget_scout.nerfmae_alpha_target_only_p1.0_e1200_seed1_epoch1200_sched_epoch_seed1_fcos1000.log`
- alpha_target_only resume:
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_alpha_target_paper_budget_resume.log`

Important caveats:
- The baseline paper-budget run completed normally.
- The original `alpha_target_only` FCOS run failed at epoch `785` with `RuntimeError: Pin memory thread exited unexpectedly`.
- The `alpha_target_only` partial result evaluates the best checkpoint saved before that crash.
- The `alpha_target_only` resume result is not a strict optimizer/scheduler resume. FCOS checkpoints in this repo save backbone and FCOS module weights, but not optimizer or scheduler state. The resume run therefore fine-tunes for an additional `215` epochs from the saved best model in a separate save path.
- This experiment is single-seed. It is suitable as a paper-budget scout, not as final multi-seed evidence.

Reading:
- At paper-budget scale, the full RGBA baseline is clearly stronger than `alpha_target_only` in this Front3D FCOS setting.
- The low-budget story does not survive unchanged: `alpha_target_only` was competitive or best at `p0.1/e30-e100`, but with `p1.0/e1200` pretraining and `e1000` FCOS, baseline reaches `0.5892` AP@50 while the best alpha-target result is `0.4602`.
- The conservative claim should shift from "alpha_target_only replaces the full baseline" to "target-side alpha is a sample-efficient learning signal, but full RGBA reconstruction becomes important at near-paper budget."
- The original method-style claim should not be advanced without additional evidence. A defensible next step is to verify the official checkpoint / paper-number reproduction and, if needed, run a smaller set of controlled multi-seed or external-dataset checks.

## Experiment 15: Alpha-to-RGBA Curriculum Scout (300 epochs, seed 1)

Goal:
- test alpha-target warmup / RGB-ramp curricula after paper-budget scout showed full RGBA wins asymptotically over zero-input alpha-target-only

Protocol:
- pretrain: `percent_train=1.0`, seed `1`, checkpoint `epoch_N.pt`, visible input kept as RGBA
- loss: alpha loss on removed patches is always active; RGB loss on occupied voxels starts at weight 0 and returns or ramps to 1
- pretrain optimizer setting: `LR=1e-3`, `WEIGHT_DECAY=0.0`, global batch 16 on 4 GPUs
- downstream: Front3D FCOS, `FCOS_NUM_EPOCHS=1000`, `LR_SCHEDULER=onecycle_epoch`, AP50-best checkpoint selection

| pretrain epochs | condition | AP@50 | AP@25 | AP@75 | Recall@50 top300 |
|---:|---|---:|---:|---:|---:|
| 300 | warmup10 | 0.5635 | 0.8181 | 0.0858 | 0.6765 |
| 300 | warmup25 | 0.5391 | 0.8160 | 0.0915 | 0.6985 |
| 300 | cosine_ramp | 0.5987 | 0.8443 | 0.1061 | 0.7059 |

Eval files:
- `/home/minesawa/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_warmup10_p1.0_e300_seed1_epoch300_sched_epoch_seed1_fcos1000_eval/eval.json`
- `/home/minesawa/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_warmup25_p1.0_e300_seed1_epoch300_sched_epoch_seed1_fcos1000_eval/eval.json`
- `/home/minesawa/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1_epoch300_sched_epoch_seed1_fcos1000_eval/eval.json`

Reading:
- This is a staged scout. Promote only the best curriculum to e600/e1200 or multi-seed if it improves sample efficiency against the existing paper-budget baseline reference.

## Experiment 16: Cosine Curriculum Method Validation

Date:
- 2026-05-09 to 2026-05-12 JST

Goal:
- validate the best `cosine_ramp` curriculum against a same-budget vanilla baseline
- check whether increasing the curriculum pretrain budget from e300 to e600 improves fidelity/localization
- test whether the cosine curriculum depends on meaningful early target-alpha structure via `cosine_ramp_alpha_shuffle`

Protocol:
- pretrain dataset: NeRF-MAE pretrain split, `percent_train=1.0`
- pretrain checkpoint rule: downstream uses `epoch_N.pt`, not `model_best.pt`
- downstream dataset: Front3D FCOS
- downstream budget: `FCOS_NUM_EPOCHS=1000`, seed `1`, `LR_SCHEDULER=onecycle_epoch`
- downstream selection: best FCOS checkpoint by validation `AP@50`, then test eval

### Front3D FCOS Test Results

| condition | pretrain epochs | AP@50 | AP@25 | AP@75 | Recall@50 top300 | Recall@25 top300 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 300 | 0.4903 | 0.7818 | 0.1010 | 0.6618 | 0.9559 |
| cosine_ramp | 300 | 0.5987 | 0.8443 | 0.1061 | 0.7059 | 0.9632 |
| cosine_ramp | 600 | 0.5895 | 0.8401 | 0.0912 | 0.7132 | 0.9706 |
| cosine_ramp_alpha_shuffle | 300 | 0.4138 | 0.7186 | 0.0597 | 0.6029 | 0.9485 |
| baseline reference | 1200 | 0.5892 | 0.8494 | 0.1469 | 0.7132 | 0.9485 |

Eval files:
- baseline e300:
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_all_p1.0_e300_seed1_epoch300_sched_epoch_seed1_fcos1000_eval/eval.json`
- cosine_ramp e300:
  - `/home/minesawa/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1_epoch300_sched_epoch_seed1_fcos1000_eval/eval.json`
- cosine_ramp e600:
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e600_seed1_epoch600_sched_epoch_seed1_fcos1000_eval/eval.json`
- cosine_ramp_alpha_shuffle e300:
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_cosine_ramp_alpha_shuffle_p1.0_e300_seed1_epoch300_sched_epoch_seed1_fcos1000_eval/eval.json`
- baseline e1200 reference:
  - `/home/minesawa/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_all_p1.0_e1200_seed1_epoch1200_sched_epoch_seed1_fcos1000_eval/eval.json`

Run notes:
- The original validation chain was initially serial over FCOS jobs.
- After both new pretrains finished, the remaining FCOS jobs were parallelized manually:
  - baseline e300 on GPU1 via `nerfmae_local_minimal_gate_chain`
  - cosine_ramp e600 on GPU0 via `nerfmae_fcos_cosine_e600_parallel`
  - cosine_ramp_alpha_shuffle e300 on GPU2 via `nerfmae_fcos_shuffle_e300_parallel`
- The old serial chain was stopped after baseline e300 eval was produced to avoid duplicate FCOS launches.

Reading:
- Same-budget improvement is strong: `cosine_ramp e300` beats `baseline e300` by `+0.1085` AP@50 (`0.5987` vs `0.4903`).
- Sample-efficiency signal is strong: `cosine_ramp e300` is slightly above the single-seed `baseline e1200` AP@50 reference (`0.5987` vs `0.5892`) while using one quarter of the pretraining epochs.
- Increasing to `cosine_ramp e600` does not improve over e300 in this seed. AP@50 stays near the e1200 baseline reference (`0.5895` vs `0.5892`), but AP@75 remains weaker (`0.0912` vs `0.1469`).
- The shuffle control is decisive in this seed: `cosine_ramp_alpha_shuffle e300` falls far below `cosine_ramp e300` (`0.4138` vs `0.5987` AP@50), supporting the claim that meaningful early target-alpha structure is part of the curriculum effect rather than a pure schedule artifact.
- Current strongest claim: cosine alpha-to-RGBA curriculum improves AP@50 sample efficiency on Front3D FCOS, but fine-localization/AP@75 is not yet recovered relative to the e1200 vanilla baseline.

## Experiment 17: Cosine Curriculum Seed1 Diagnostic Dumps and Seed2 FCOS Status

Date:
- 2026-05-13 to 2026-05-15 JST

Goal:
- dump proposal / voxel-score diagnostics for the key single-seed cosine curriculum comparisons
- opportunistically run completed `baseline_e300 seed2` FCOS while waiting for more 4-GPU pretraining slots

### Diagnostic Dump Summary

Protocol:
- checkpoint: best FCOS checkpoint selected by validation AP@50 from each completed seed1 run
- diagnostic mode: `run_fcos_diagnostic_variant.sh`
- outputs: `--output_proposals`, `--save_level_index`, `--output_voxel_scores`
- dataset: Front3D test split
- top-k diagnostic summary: top300 proposals, TP IoU threshold 0.5

| condition | AP@50 | AP@25 | AP@75 | Recall@50 top300 | top300 mean IoU | frac IoU>=0.5 | TP score mean | FP score mean | first TP rank | voxel peakiness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline e300 | 0.4903 | 0.7819 | 0.1009 | 0.6618 | 0.0595 | 0.0178 | 0.5955 | 0.0373 | 1.25 | 14.7951 |
| cosine_ramp e300 | 0.5987 | 0.8443 | 0.1061 | 0.7059 | 0.0591 | 0.0188 | 0.6060 | 0.0475 | 1.18 | 12.0121 |
| cosine_ramp_alpha_shuffle e300 | 0.4137 | 0.7185 | 0.0597 | 0.6029 | 0.0603 | 0.0165 | 0.5246 | 0.0400 | 3.00 | 13.5157 |
| baseline e1200 | 0.5892 | 0.8494 | 0.1469 | 0.7132 | 0.0523 | 0.0190 | 0.6672 | 0.0118 | 1.12 | 19.4645 |

Diagnostic artifact files:
- `/home/minesawa/ssl/NeRF-MAE/results/shortcut_probe_artifacts/cosine_curriculum_seed1_diagnostics_summary.md`
- `/home/minesawa/ssl/NeRF-MAE/results/shortcut_probe_artifacts/cosine_curriculum_seed1_diagnostics_summary.json`
- `/home/minesawa/ssl/NeRF-MAE/results/shortcut_probe_artifacts/cosine_curriculum_seed1_diagnostics_summary.csv`

Diagnostic dump dirs:
- baseline e300:
  - `/home/minesawa/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_all_p1.0_e300_seed1_epoch300_sched_epoch_seed1_fcos1000_diagnostics`
- cosine_ramp e300:
  - `/home/minesawa/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1_epoch300_sched_epoch_seed1_fcos1000_diagnostics`
- cosine_ramp_alpha_shuffle e300:
  - `/home/minesawa/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_alpha_rgba_curr_cosine_ramp_alpha_shuffle_p1.0_e300_seed1_epoch300_sched_epoch_seed1_fcos1000_diagnostics`
- baseline e1200:
  - `/home/minesawa/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_all_p1.0_e1200_seed1_epoch1200_sched_epoch_seed1_fcos1000_diagnostics`

Reading:
- The diagnostic eval values match the previously recorded test results up to rounding.
- `cosine_ramp e300` improves AP@50 and Recall@50 top300 over `baseline e300`, but the top300 proposal IoU distribution is very similar. The difference is therefore not a simple increase in raw proposal IoU coverage.
- `cosine_ramp_alpha_shuffle e300` is clearly weaker in AP@50 and first-TP rank, consistent with the earlier reading that meaningful early target-alpha structure matters.
- `baseline e1200` still has the best AP@75 and the sharpest voxel score maps, so the AP@75 / localization gap remains a real open issue for the curriculum method.

### Baseline e300 Seed2 FCOS Status

Completed before FCOS:
- pretrain checkpoint exists:
  - `/home/minesawa/ssl/NeRF-MAE/output/nerf_mae/results/nerfmae_all_p1.0_e300_seed2/epoch_300.pt`

FCOS attempt:
- save name:
  - `nerfmae_all_p1.0_e300_seed2_epoch300_sched_epoch_seed2_fcos1000`
- launcher log:
  - `/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_fcos_baseline_e300_seed2_parallel.log`
- status:
  - run reached epoch `640`
  - process stopped during the epoch-640 validation pass before producing the final eval directory
  - no original final `*_eval/eval.json` was produced for this condition
- best checkpoints saved before stopping:
  - AP@50-best checkpoint: `model_best_ap50_ap25_0.5291156768798828_0.8227242231369019.pt`
  - later AP@25-best checkpoints are also present, with best observed validation AP@25 `0.829770028591156`

Partial recovery eval:
- evaluated the stopped run's AP@50-best checkpoint as a partial-status diagnostic
- save name:
  - `nerfmae_all_p1.0_e300_seed2_epoch300_sched_epoch_seed2_fcos1000_partial640_eval`
- eval file:
  - `/home/minesawa/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_all_p1.0_e300_seed2_epoch300_sched_epoch_seed2_fcos1000_partial640_eval/eval.json`
- metrics:
  - AP@50: `0.4395`
  - AP@25: `0.8209`
  - AP@75: `0.0701`
  - Recall@50 top300: `0.6397`
  - Recall@25 top300: `0.9632`

Non-strict continuation eval:
- continuation source:
  - started from the stopped run's AP@50-best model checkpoint because the original FCOS checkpoints did not contain optimizer/scheduler state
  - ran an additional 360-epoch continuation with fresh optimizer state and exponential LR decay
  - selected the continuation AP@50-best validation checkpoint `model_best_ap50_ap25_0.5458228588104248_0.8243664503097534.pt`
- save name:
  - `nerfmae_all_p1.0_e300_seed2_epoch300_sched_epoch_seed2_fcos1000_resume_from640_eval`
- eval file:
  - `/home/minesawa/ssl/NeRF-MAE/output/nerf_rpn/results/nerfmae_all_p1.0_e300_seed2_epoch300_sched_epoch_seed2_fcos1000_resume_from640_eval/eval.json`
- metrics:
  - AP@50: `0.4894`
  - AP@25: `0.8024`
  - AP@75: `0.1197`
  - Recall@50 top300: `0.6618`
  - Recall@25 top300: `0.9485`

Important caveat:
- Because the seed2 FCOS run did not complete normally, the partial recovery eval should not be used in the main 3-seed gate table.
- The non-strict continuation eval is useful as a status point, but a clean uninterrupted rerun remains the safest option before reporting `baseline_e300 seed2` as a main-table downstream seed.
- The existing stopped checkpoint only contains model weights, not optimizer/scheduler state, so the current continuation is necessarily a non-strict model-weight continuation rather than an exact optimizer-state resume.

Implementation notes:
- To run FCOS under the current base Python 3.11 / NumPy 1.26 environment, two source compatibility fixes were required:
  - `np.float` -> `float` in `nerf_mae/model/mae/torch_utils.py`
  - `np.int` -> `int` in `nerf_rpn/model/fcos/fcos.py`
- The rotated IoU CUDA wrapper was also adjusted to prefer package-relative import of `sort_vertices`.
- FCOS training resume support was added after this stop:
  - each completed training epoch now writes a resumable `checkpoint_last.pt`
  - FCOS checkpoints now include optimizer state, scheduler state, best validation metrics, and RNG states
  - `--resume_training --checkpoint <checkpoint_last.pt>` restores those states and continues from `epoch + 1`
  - pruning of validation-best checkpoints now only considers `model_best_*.pt`, so `checkpoint_last.pt` is not deleted by AP-based pruning

### Pending e300 3-Seed Gate Status

Snapshot:
- 2026-05-16 JST

Purpose:
- before moving to fixed-ramp / external-dataset experiments, finish the e300 3-seed gate for:
  - `baseline_e300`
  - `cosine_ramp_e300`
  - `cosine_ramp_alpha_shuffle_e300`

Current completed items:
- `baseline e300 seed1`
  - pretrain: complete, `epoch_300.pt`
  - FCOS e1000 eval: complete
- `baseline e300 seed2`
  - pretrain: complete, `epoch_300.pt`
  - original FCOS e1000: stopped at epoch 640 during validation
  - partial AP@50-best checkpoint eval: complete but diagnostic-only
  - non-strict continuation from AP@50-best model checkpoint reached epoch 360 of the continuation run
  - continuation produced validation checkpoints up to `model_best_ap50_ap25_0.5458228588104248_0.8243664503097534.pt`
  - continuation final test eval is complete: AP@50 `0.4894`, AP@25 `0.8024`, AP@75 `0.1197`, Recall@50 top300 `0.6618`
  - caveat: still non-strict because the continuation could not restore optimizer/scheduler state
- `cosine_ramp e300 seed1`
  - pretrain: complete, `epoch_300.pt`
  - FCOS e1000 eval: complete
- `cosine_ramp_alpha_shuffle e300 seed1`
  - pretrain: complete, `epoch_300.pt`
  - FCOS e1000 eval: complete

Current incomplete pretraining:
- `baseline e300 seed3`
  - latest available checkpoint: `epoch_180.pt`
  - remaining: finish pretrain to `epoch_300.pt`, then run FCOS e1000
- `cosine_ramp e300 seed2`
  - pretrain not started / no result directory
  - remaining: pretrain to `epoch_300.pt`, then run FCOS e1000
- `cosine_ramp e300 seed3`
  - pretrain not started / no result directory
  - remaining: pretrain to `epoch_300.pt`, then run FCOS e1000
- `cosine_ramp_alpha_shuffle e300 seed2`
  - pretrain not started / no result directory
  - remaining: pretrain to `epoch_300.pt`, then run FCOS e1000
- `cosine_ramp_alpha_shuffle e300 seed3`
  - pretrain not started / no result directory
  - remaining: pretrain to `epoch_300.pt`, then run FCOS e1000

Immediate next actions:
1. Run missing final test eval for `baseline_e300 seed2` continuation checkpoint.
2. Finish `baseline_e300 seed3` pretrain from the latest available checkpoint or rerun cleanly to `epoch_300.pt`.
3. Launch `cosine_ramp_e300 seed2/3` pretraining.
4. Launch `cosine_ramp_alpha_shuffle_e300 seed2/3` pretraining.
5. Run FCOS e1000 for each completed pretrain checkpoint.

Gate decision rule:
- continue toward fixed-ramp / external-dataset experiments only if the 3-seed gate preserves:
  - `cosine_ramp_e300` > `baseline_e300` in mean AP@50
  - `cosine_ramp_e300` > `cosine_ramp_alpha_shuffle_e300` in mean AP@50
  - at least 2/3 paired seeds support each comparison

## Experiment 18: ABCI3 Clean e300 Gate Preparation

Snapshot:
- 2026-05-19 JST

Purpose:
- run only the submission-critical single-seed comparison in the new ABCI3 environment unless additional robustness is needed
- finish the clean comparison as quickly as possible while preserving the downstream FCOS protocol

ABCI3 implementation updates:
- environment setup script:
  - `nerf_mae/probe_scripts/setup_abci3_env.sh`
  - creates `${ROOT_DIR}/.venv-abci3`
  - installs PyTorch `2.7.0+cu118`, torchvision `0.22.0`, numpy `1.26.4`, and the Python packages required by MAE/FCOS
- preflight script:
  - `nerf_mae/probe_scripts/abci3_e300_gate_preflight.sh`
  - checks `qsub`, `qstat`, the Python env, pretrain data, FCOS data, and required imports
- data symlink helper:
  - `nerf_mae/probe_scripts/setup_abci3_data_links.sh`
  - expects preprocessed NeRF-MAE data, not raw Structured3D
  - required pretrain source: `features/` plus `nerfmae_split.npz`
  - required FCOS source: `features/`, `obb/`, plus `3dfront_split.npz`
- pretrain PBS:
  - `nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs`
  - supports single-node 8-GPU DDP and multi-node DDP via `pbsdsh`
  - for 2 rt_HF nodes, default is 16 GPUs total with `PRETRAIN_BATCH_SIZE_PER_GPU=1`, preserving global batch size 16
- FCOS PBS:
  - `nerf_mae/probe_scripts/abci3_e300_gate_fcos.pbs`
  - keeps the downstream protocol single-GPU by default
- submitter:
  - `nerf_mae/probe_scripts/submit_abci3_e300_gate.sh`
  - submits 9 pretrain jobs and 9 dependent FCOS jobs:
    - `baseline` seeds 1/2/3
    - `cosine_ramp` seeds 1/2/3
    - `cosine_ramp_alpha_shuffle` seeds 1/2/3
- pipeline submitter:
  - `nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh`
  - defaults to the minimal paper-critical jobs:
    - `baseline e300 seed1`
    - `cosine_ramp e300 seed1`
    - `cosine_ramp_alpha_shuffle e300 seed1`
  - with available checkpoints, use `SUBMIT_PRETRAIN=0` and run only three clean FCOS evals
  - if checkpoints are unavailable, use `PRETRAIN_NODES=2 PRETRAIN_SLOTS=3 PRETRAIN_BATCH_SIZE_PER_GPU=1` to regenerate only these three pretrains
  - the old 3-seed/full gate remains available by overriding `GATE_JOBS`, but it is not the default

DDP code update:
- `nerf_mae/run_swin_mae3d.py` now accepts environment-driven multi-node DDP settings:
  - `NERFMAE_DDP_USE_ENV=1`
  - `NERFMAE_DDP_NODE_RANK`
  - `NERFMAE_DDP_NUM_NODES`
  - `NERFMAE_DDP_WORLD_SIZE`
  - `NERFMAE_DDP_MASTER_ADDR`
  - `NERFMAE_DDP_MASTER_PORT`
- global rank is now `node_rank * local_world_size + local_rank`, avoiding rank collisions across nodes
- `nerf_rpn/run_fcos_pretrained.py` received the same DDP rank fix for future multi-node downstream experiments, though the clean gate keeps FCOS single-GPU unless explicitly changed

Data status:
- `/groups/gag51402/datasets` is accessible.
- Top-level candidates include `Structure3D`, `HM3D`, `ARKitScenes`, `ScanNet`, and others.
- No directly usable preprocessed NeRF-MAE/Front3D RPN directory was found at the obvious paths:
  - `/groups/gag51402/datasets/front3d_rpn_data`
  - `/groups/gag51402/datasets/3dfront_rpn_data`
  - `/groups/gag51402/datasets/nerfmae`
  - `/groups/gag51402/datasets/Structure3D/front3d_rpn_data`
  - `/groups/gag51402/datasets/Structure3D/nerfmae`
- `/groups/gag51402/datasets/Structure3D` appears to be raw or converted Structured3D data, not the `features/`, `obb/`, `*_split.npz` format consumed by the current NeRF-MAE/FCOS scripts.
- Therefore, do not symlink `Structure3D` directly to `dataset/pretrain` or `dataset/finetune/front3d_rpn_data`. Locate or create the preprocessed feature/box directories first.

Recommended launch sequence:
1. Create/verify the env:
   - `PROBE_ENV_PREFIX=/groups/gag51404/ide/vgi/NeRF-MAE/.venv-abci3 bash nerf_mae/probe_scripts/setup_abci3_env.sh`
2. Link preprocessed data once the correct source directories are known:
   - `PRETRAIN_DATA_SRC=/path/to/pretrain FCOS_DATA_SRC=/path/to/front3d_rpn_data bash nerf_mae/probe_scripts/setup_abci3_data_links.sh`

## Experiment 19: ABCI3 Checkpoint Bundle Install and e300 3-Seed Gate Readiness

Snapshot:
- 2026-05-19 JST

Input bundle:
- `/groups/gag51404/ide/vgi/NeRF-MAE/nerfmae_abci_pretrain_checkpoints_20260519.zip`

Installed checkpoint payload:
- `baseline e300 seed1`: `epoch_300.pt`
- `baseline e300 seed2`: `epoch_300.pt`
- `baseline e300 seed3`: `epoch_220.pt` only; usable for resume toward `epoch_300.pt`
- `cosine_ramp e300 seed1`: `epoch_300.pt`
- `cosine_ramp_alpha_shuffle e300 seed1`: `epoch_300.pt`
- `cosine_ramp e600 seed1`: `epoch_600.pt`
- `baseline e1200 seed1`: `epoch_1200.pt`

Verification:
- extracted with `nerf_mae/probe_scripts/install_abci_checkpoint_bundle.sh`
- `sha256sum -c checksums.sha256`: all included checkpoints passed
- created `*_abci3clean` symlinks so clean ABCI3 FCOS outputs can use a separate result namespace
- ABCI3 Python import preflight passed with data checks disabled:
  - PyTorch `2.7.0+cu118`
  - torchvision `0.22.0+cu118`
  - NumPy `1.26.4`

Script updates:
- `abci3_e300_gate_pretrain.pbs` now auto-resumes from the latest `epoch_*.pt`
  in the target pretrain directory when final `epoch_N.pt` is absent.
- `submit_abci3_e300_gate_pipeline.sh` now defaults to the full e300 3-seed
  gate and skips pretrain submission for jobs whose final checkpoint already
  exists.
- The current dry-run submits direct FCOS for:
  - `baseline e300 seed1/2`
  - `cosine_ramp e300 seed1`
  - `cosine_ramp_alpha_shuffle e300 seed1`
- The same dry-run submits 2-node pretrain plus dependent FCOS for:
  - `baseline e300 seed3` resumed from `epoch_220.pt`
  - `cosine_ramp e300 seed2/3`
  - `cosine_ramp_alpha_shuffle e300 seed2/3`

Remaining blocker before real qsub:
- preprocessed data links are still missing.
- The required FCOS data layout is:
  - `features/`
  - `obb/`
  - `3dfront_split.npz`
- The required pretrain layout is:
  - `features/`
  - `nerfmae_split.npz`
- Obvious paths under `/groups/gag51402/datasets` still do not contain those layouts:
  - `/groups/gag51402/datasets/front3d_rpn_data`
  - `/groups/gag51402/datasets/3dfront_rpn_data`
  - `/groups/gag51402/datasets/NeRF-MAE/front3d_rpn_data`
  - `/groups/gag51402/datasets/NeRF-MAE/pretrain`
  - `/groups/gag51402/datasets/nerfmae`
  - `/groups/gag51402/datasets/Structure3D/front3d_rpn_data`
  - `/groups/gag51402/datasets/Structure3D/nerfmae`

Next command once data paths are known:

```bash
PRETRAIN_DATA_SRC=/path/to/pretrain \
FCOS_DATA_SRC=/path/to/front3d_rpn_data \
bash nerf_mae/probe_scripts/setup_abci3_data_links.sh

PRETRAIN_NODES=2 PRETRAIN_SLOTS=3 PRETRAIN_BATCH_SIZE_PER_GPU=1 PRETRAIN_EVAL_INTERVAL=300 \
bash nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh
```

Interpretation:
- The checkpoint bundle is sufficient to avoid rerunning several completed
  pretrains, but it is not sufficient to complete Gate 1 by FCOS-only execution.
- Gate 1 still needs new pretraining for `cosine_ramp seed2/3`,
  `cosine_ramp_alpha_shuffle seed2/3`, and completion of `baseline seed3` from
  `epoch_220.pt` to `epoch_300.pt`.

## Experiment 20: ABCI3 Data Preparation and Dependent e300 Gate Submission

Snapshot:
- 2026-05-19 JST

Decision:
- Since the repo was moved from the previous machine to ABCI, the missing
  preprocessed datasets must be prepared on ABCI rather than symlinking raw
  `Structure3D`.
- The correct source for Front3D FCOS data is the NeRF-RPN Hugging Face dataset,
  because the old Google Drive folder now fails through `gdown`.

Data sources:
- NeRF-MAE pretrain:
  - `https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/nerfmae/NeRF-MAE_pretrain.tar.gz`
  - HEAD size: about `60.5G`
- Front3D RPN finetune:
  - `https://huggingface.co/datasets/lyclyc52/NeRF_RPN/resolve/main/front3d_rpn_data.zip`
  - HEAD size: about `4.3G` / HF listed size `4.62GB`

Implementation:
- added `nerf_mae/probe_scripts/prepare_abci3_data.sh`
  - downloads with `wget --continue`
  - extracts into `dataset/_downloads`
  - locates `features/` plus split files
  - links:
    - `dataset/pretrain`
    - `dataset/finetune/front3d_rpn_data`
  - validates split keys and feature/OBB file presence
- added `nerf_mae/probe_scripts/abci3_prepare_data.pbs`
- added `nerf_mae/probe_scripts/submit_abci3_prepare_data.sh`
- updated `submit_abci3_e300_gate_pipeline.sh` with `GLOBAL_DEPENDENCY`, so
  Gate 1 can be submitted immediately and held until data preparation passes.

Submitted jobs:
- data preparation:
  - `1776943.pbs1`
  - queue: `rt_HC`
  - status at submit check: running
- dependent e300 Gate 1 jobs:
  - `baseline e300 seed1` FCOS: `1776946.pbs1`, depends on data
  - `baseline e300 seed2` FCOS: `1776947.pbs1`, depends on data
  - `baseline e300 seed3` pretrain: `1776948.pbs1`, depends on data
  - `baseline e300 seed3` FCOS: `1776949.pbs1`, depends on seed3 pretrain
  - `cosine_ramp e300 seed1` FCOS: `1776950.pbs1`, depends on data
  - `cosine_ramp e300 seed2` pretrain: `1776951.pbs1`, depends on data
  - `cosine_ramp e300 seed2` FCOS: `1776952.pbs1`, depends on seed2 pretrain
  - `cosine_ramp e300 seed3` pretrain: `1776953.pbs1`, depends on data
  - `cosine_ramp e300 seed3` FCOS: `1776954.pbs1`, depends on seed3 pretrain
  - `shuffle e300 seed1` FCOS: `1776955.pbs1`, depends on data
  - `shuffle e300 seed2` pretrain: `1776956.pbs1`, depends on data and the
    baseline seed3 pretrain slot
  - `shuffle e300 seed2` FCOS: `1776957.pbs1`, depends on seed2 shuffle pretrain
  - `shuffle e300 seed3` pretrain: `1776958.pbs1`, depends on data and the
    cosine seed2 pretrain slot
  - `shuffle e300 seed3` FCOS: `1776959.pbs1`, depends on seed3 shuffle pretrain

Status:
- data download started; `dataset/_downloads/archives/NeRF-MAE_pretrain.tar.gz`
  was growing at the first check.
- all Gate 1 jobs were held by dependencies at the first submit check.

Fast-path correction:
- Initial Gate 1 submission used `PRETRAIN_SLOTS=3`, so the first wave after
  data would have been three 2-node pretrains (`6` rt_HF nodes / `48` GPUs).
- Because total nodes are available, this was not the absolute fastest plan.
- `qalter` is unavailable for normal users on ABCI, so the dependent shuffle
  seed2/3 pretrain and FCOS jobs were deleted and resubmitted without the
  slot-tail dependency.
- Active fast-path pretrain jobs after data prep:
  - `baseline seed3`: `1776948.pbs1`
  - `cosine seed2`: `1776951.pbs1`
  - `cosine seed3`: `1776953.pbs1`
  - `shuffle seed2`: `1776981.pbs1`
  - `shuffle seed3`: `1776983.pbs1`
- These five pretrain jobs each request `2` rt_HF nodes / `16` GPUs, so the
  post-data maximum is `10` rt_HF nodes / `80` GPUs.
- FCOS remains single-node/single-GPU by design to preserve the downstream
  protocol, and the FCOS jobs are dependent on either data prep or their own
  pretrain.
3. Run preflight:
   - `bash nerf_mae/probe_scripts/abci3_e300_gate_preflight.sh`
4. Submit the clean minimal comparison:
   - if checkpoints were brought over:
     - `SUBMIT_PRETRAIN=0 SUBMIT_FCOS=1 bash nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh`
   - if checkpoints must be regenerated:
     - `PRETRAIN_NODES=2 PRETRAIN_SLOTS=3 PRETRAIN_BATCH_SIZE_PER_GPU=1 PRETRAIN_EVAL_INTERVAL=300 bash nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh`

Current blocker:
- the clean gate should not be submitted until preflight passes.
- `.venv-abci3` is built and import checks pass.
- the remaining blocker is the expected preprocessed data symlinks:
  - `dataset/pretrain/features`
  - `dataset/pretrain/nerfmae_split.npz`
  - `dataset/finetune/front3d_rpn_data/features`
  - `dataset/finetune/front3d_rpn_data/obb`
  - `dataset/finetune/front3d_rpn_data/3dfront_split.npz`

## Experiment 21: ABCI3 2-seed e300 Gate Restart After Environment Fixes

Snapshot:
- 2026-05-19 JST

Updated decision:
- Do not spend points on full 3-seed unless the early gate survives.
- Current gate is a 2-seed paired check:
  - `baseline_e300` seeds 1/2
  - `cosine_ramp_e300` seeds 1/2
  - `cosine_ramp_alpha_shuffle_e300` seeds 1/2
- Seed3 jobs from the earlier full-gate submission were cancelled or left out.

Data status:
- ABCI3 data preparation completed successfully.
- Active links:
  - `dataset/pretrain -> dataset/_downloads/pretrain_extract/NeRF-MAE/pretrain`
  - `dataset/finetune/front3d_rpn_data -> dataset/_downloads/front3d_rpn_extract/front3d_rpn_data`
- Validation counts:
  - pretrain feature files: `3260`
  - pretrain split train/val/test: `3260/20/18`
  - FCOS feature/OBB files: `159/159`
  - FCOS split train/val/test: `122/20/17`

ABCI3 environment fixes:
- Built the missing `sort_vertices` CUDA extension for Python 3.11 / PyTorch
  2.7.0+cu118.
- Updated `nerf_rpn/build_rotated_iou.sh` to fall back to system `gcc/g++`
  when conda compiler wrappers are absent.
- Added default `TORCH_CUDA_ARCH_LIST=8.0;9.0`, because login nodes do not
  expose GPUs for architecture auto-detection.
- Added FCOS job preflight build of `sort_vertices` in
  `abci3_e300_gate_fcos.pbs`.
- Fixed multi-node pretrain node-entry env propagation in
  `abci3_e300_gate_pretrain.pbs`; the generated per-node script now embeds
  `ABCI3_PRETRAIN_SCRIPT`, hosts/status files, and DDP master/world variables.
- Patched trusted checkpoint loads for the new PyTorch default
  `weights_only=True`:
  - MAE checkpoint load in `nerf_rpn/model/feature_extractor.py`
  - FCOS checkpoint load in `nerf_rpn/run_fcos_pretrained.py`

Current submitted jobs:
- Missing pretrains, each `2` rt_HF nodes / `16` GPUs:
  - `cosine_ramp e300 seed2`: `1777162.pbs1`
  - `shuffle e300 seed2`: `1777165.pbs1`
- Held FCOS jobs that will start after their matching pretrain:
  - `cosine_ramp e300 seed2`: `1777163.pbs1`
  - `shuffle e300 seed2`: `1777166.pbs1`
- Existing-checkpoint FCOS jobs resubmitted after the checkpoint-load fix:
  - `baseline e300 seed1`: `1777173.pbs1`
  - `baseline e300 seed2`: `1777174.pbs1`
  - `cosine_ramp e300 seed1`: `1777175.pbs1`
  - `shuffle e300 seed1`: `1777176.pbs1`
- Summary job:
  - `1777177.pbs1`
  - dependency: `afterany` on all six FCOS jobs
  - output directory:
    `output/launcher/abci3_e300_gate_2seed_retry2_existing_fcos`

Result summary tooling:
- Added `nerf_mae/probe_scripts/summarize_abci3_e300_gate.py`.
- Added `nerf_mae/probe_scripts/abci3_e300_gate_summary.pbs`.
- The summary reports AP@50/AP@25/AP@75/Recall@50 top300 plus paired AP@50
  diffs:
  - `cosine - baseline`
  - `cosine - shuffle`

Interpretation rule for this gate:
- If 2 seeds agree that `cosine > baseline` and `cosine > shuffle`, then spend
  seed3 / e600 / fixed-ramp points.
- If seed2 does not agree, stop before expanding the grid.

## Experiment 22: ABCI3 Global-Batch-16 8GPU Restart and Null-Prior Scouts

Snapshot:
- 2026-05-19 JST

Batch/parallelism correction:
- A 2-node/16GPU pretrain attempt with global batch 16 made each GPU see batch
  size 1 and was communication-heavy.
- A later global-batch-64 attempt was intentionally stopped after discussion,
  because it changes the training protocol.
- Current main pretrain protocol is global batch 16 on two HF nodes:
  - `PRETRAIN_NODES=2`
  - `PRETRAIN_GPU_IDS=0-7` per node
  - `PRETRAIN_BATCH_SIZE_PER_GPU=1`
  - global batch = `2 * 8 * 1 = 16`

Existing-checkpoint FCOS jobs still running:
  - `baseline e300 seed1`: `1777173.pbs1`
  - `baseline e300 seed2`: `1777174.pbs1`
  - `cosine_ramp e300 seed1`: `1777175.pbs1`
  - `shuffle e300 seed1`: `1777176.pbs1`

Startup check:
- A 1-node/8GPU global-batch-16 restart reached epoch 1 step 0 but was too
  slow: step `0 -> 30` took about `112s`.
- A 2-node/16GPU global-batch-16 restart reached epoch 1 and is faster:
  step `0 -> 30` takes about `59s`.
- A 1-node/4GPU global-batch-16 speed scout was also tested and stopped:
  step `0 -> 10` took about `106s`, so it is much slower than 16GPU.
- `rt_HG` is a 1-GPU queue (`resources_max.ngpus=1`), so it is not a
  multi-GPU replacement for HF pretraining.

Current main gate pretrain jobs:
- `cosine_ramp e300 seed2`, 2 HF nodes / 16 GPUs / global batch 16:
  `1777284.pbs1`
- `shuffle e300 seed2`, 2 HF nodes / 16 GPUs / global batch 16:
  `1777286.pbs1`
- Dependent FCOS jobs:
  - `1777285.pbs1`
  - `1777287.pbs1`

Null-prior diagnostic additions:
- Added no-position and coordinate-jitter controls to the MAE pretrain script:
  - disable absolute position embedding
  - disable relative position bias
  - horizontal rotation/flip
  - zero-fill horizontal coordinate shift
- Added a coordinate-only alpha prior probe:
  `nerf_mae/tools/coord_prior_alpha_probe.py`.
- Coordinate-only probe result:
  - val MSE: `0.052610`
  - occupied IoU: `0.3056`
  - precision / recall: `0.3058 / 0.9981`
  - predicted occupied rate: `0.9864`
- This suggests a coordinate-only tiny prior mostly predicts broad occupancy
  and is not yet a strong standalone explanation for the alpha-target signal.

Queued diagnostic scouts:
- These are queued behind both seed2 main pretrains and run sequentially with
  a 1-node/8GPU global-batch-16 scout protocol:
  - `alpha_target_only_no_pos e100 seed1`: `1777288.pbs1`
  - `baseline_no_pos e100 seed1`: `1777290.pbs1`
  - `alpha_target_only_coord_jitter e100 seed1`: `1777292.pbs1`
  - `cosine_coord_jitter e100 seed1`: `1777294.pbs1`
- Dependent FCOS scout jobs use `FCOS_NUM_EPOCHS=300`:
  - `1777289.pbs1`, `1777291.pbs1`, `1777293.pbs1`, `1777295.pbs1`
- These diagnostics are single-seed only. They are not intended to estimate
  seed variance; they are used as cheap mechanism checks for the null-prior
  hypothesis.
- Queue pruning check:
  - no multi-seed diagnostic jobs are queued.
  - all diagnostic scouts are seed1 only.
  - unrelated `3D-NEPA` jobs were not modified.

Current priority:
- Wait for the two seed2 main pretrains, then their FCOS jobs.
- Use the diagnostic scouts only as null-prior sanity checks; do not expand to
  heavy method runs until the 2-seed e300 gate is read.

## Experiment 23: ABCI3 Checkpoint/Resume Correction

Snapshot:
- 2026-05-20 JST

Issue:
- The running seed2 pretrain jobs do not have intermediate `epoch_*.pt`
  checkpoints.
- Resume support exists, and saved checkpoints include model, optimizer,
  scheduler, epoch, train args, and best metric.
- The practical issue was the launch setting:
  `PRETRAIN_EVAL_INTERVAL=300`.
- In the current training loop, epoch checkpoints were saved only inside the
  eval block. For an e300 run with eval interval 300, this means no
  intermediate `epoch_*.pt` is written before epoch 300.

Impact:
- Running jobs `1777284.pbs1` and `1777286.pbs1` cannot be safely interrupted
  and resumed from an intermediate checkpoint, because none has been written.
- Stopping them now would lose the already completed training time.

Fix for future jobs:
- Added an eval-independent `--checkpoint_interval` argument to
  `nerf_mae/run_swin_mae3d.py`.
- Added `CHECKPOINT_INTERVAL` forwarding in `nerf_mae/train_mae3d.sh`.
- Added `PRETRAIN_CHECKPOINT_INTERVAL=10` default forwarding in the ABCI3
  pretrain and submit scripts.
- The intended setting is now:
  - eval interval: 300 epochs
  - checkpoint interval: 10 epochs
  - keep latest epoch checkpoints according to `--keep_checkpoints`

Verification:
- `python -m py_compile nerf_mae/run_swin_mae3d.py`
- `bash -n nerf_mae/train_mae3d.sh`
- `bash -n nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs`
- `bash -n nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh`

## Experiment 24: ABCI3 Speed Sanity Check vs Itachi/A100 Expectations

Snapshot:
- 2026-05-20 JST

Observation:
- The running 2-node/16GPU H200 global-batch-16 jobs are not faster than the
  previous A100 4GPU expectation.
- Parsed from `worker_0.log`:
  - cosine seed2 current 16GPU: `2.066 sec/step`, projected e300 `35.1h`
  - shuffle seed2 current 16GPU: `2.064 sec/step`, projected e300 `35.1h`
  - 8GPU scout: `3.733 sec/step`, projected e300 `63.5h`
  - 4GPU scout: `10.555 sec/step`, projected e300 `179.4h`

Important caveat:
- The 4GPU scout only logged steps `0 -> 10` and was terminated after a few
  minutes. This includes cold-start, filesystem cache, and early DataLoader
  behavior, so it is not a reliable steady-state benchmark by itself.
- The 16GPU jobs have many epochs of steady-state logs and are reliable:
  e300 is about `35h`.

Known differences from the old/itachi-style execution:
- Current ABCI3 jobs set `DETERMINISTIC=1`, which disables cuDNN benchmark and
  enables deterministic CUDA behavior.
- Current ABCI3 environment is Python `3.11.13`, PyTorch `2.7.0+cu118`,
  CUDA `11.8`, cuDNN `9.1`.
- ABCI3 HF jobs run on H200 nodes but the software stack is still CUDA 11.8.
- Current data path is the group filesystem symlink:
  `dataset/pretrain -> dataset/_downloads/pretrain_extract/NeRF-MAE/pretrain`.
  Each sample is loaded from compressed `.npz` files via `np.load`; the
  pretrain feature directory is about `61G`.
- Current 16GPU global-batch-16 run uses per-GPU microbatch `1`, so it is
  communication-heavy and does not exploit H200 compute well.
- Current 16GPU run is multi-node DDP; the old 4GPU run was likely single-node.

Current interpretation:
- The current speed is consistent with a workload bottlenecked by I/O,
  deterministic kernels, small per-GPU microbatch, and DDP communication, not
  by raw H200 tensor throughput.
- It is still suspicious that 16 H200 GPUs only match the old A100 4GPU
  expectation. Before changing the main jobs, a fair steady-state speed
  benchmark should compare:
  - 4GPU / 8GPU / 16GPU
  - deterministic on/off
  - current shared `.npz` data path vs node-local staged data
  - CUDA 11.8 stack vs an ABCI3 CUDA 12.x/PyTorch cu12x stack if available

## Experiment 25: ABCI3 e1200 Scaling Hardening

Snapshot:
- 2026-05-20 JST

Current running jobs:
- `1777284.pbs1`: cosine e300 seed2, `2` HF nodes / `16` H200 GPUs,
  global batch 16, suffix `abci3gb16_16g`.
- `1777286.pbs1`: shuffle e300 seed2, same resource shape.
- As of the latest log parse, both are around epoch `159/300` with
  `~2.08 sec/step`, ETA around `2026-05-21 07:26 JST`.
- These running jobs were launched before the speed fixes below, so they
  should not be used as evidence for the fixed code path.

Root-cause candidates closed in code:
- Removed the unconditional per-iteration DDP `dist.barrier()` plus scalar
  `all_reduce()` for losses in `Trainer.train_epoch`.
  - DDP already synchronizes gradients during backward.
  - Loss all-reduce is now log-only, at `log_interval`, unless W&B is enabled.
- Added `--profile_step_time` to log DataLoader wait and train step time.
- Added configurable `--train_num_workers`, `--eval_num_workers`, and
  `--persistent_workers`.
- Kept checkpoint/resume hardening from Experiment 23.
- Added runtime logging in the PBS job for host, GPU list, NCCL interface/HCA,
  torch version, CUDA version, and cuDNN version.

Data staging hardening:
- Added optional `STAGE_PRETRAIN_DATA=1` in
  `abci3_e300_gate_pretrain.pbs`.
- Staging copies `PRETRAIN_DATA_SRC` to node-local `PBS_LOCALDIR` or an
  explicit `LOCAL_STAGE_ROOT`, then points `PRETRAIN_DATA_ROOT` there.
- The stage marker now records source path, split checksum, source size, and
  feature count, so stale staged copies are invalidated.
- Free-space check now uses the larger of `STAGE_MIN_FREE_GB` and source size
  plus headroom.

Benchmark harness:
- Added `nerf_mae/probe_scripts/submit_abci3_pretrain_speed_benchmark.sh`.
- Added parser `nerf_mae/tools/parse_pretrain_speed_log.py`.
- Added usage notes in
  `nerf_mae/probe_scripts/README_ABCI3_PRETRAIN_SPEED_BENCH.md`.
- Dry-run is the default. Example:
  `SKIP_PREFLIGHT=1 bash nerf_mae/probe_scripts/submit_abci3_pretrain_speed_benchmark.sh`
- Default benchmark grid:
  - `1n4g`, `1n8g`, `2n16g`
  - global batch `16` and throughput-oriented global batch `64`
  - deterministic `1/0`
  - node-local staging `0/1`
  - profile step timing enabled

Submitted benchmark:
- Run id: `20260520_speed01`
- Manifest:
  `output/launcher/abci3_pretrain_speed_bench/20260520_speed01/manifest.tsv`
- Submitted jobs: `1778738.pbs1` through `1778761.pbs1`.
- Grid: `1n4g / 1n8g / 2n16g` × global batch `16/64` ×
  deterministic `1/0` × staging `0/1`.
- Dependency policy: `BENCH_SLOTS=1`, `afterany` chain, so only one benchmark
  pretrain should run at a time and later rows can still run if an earlier row
  fails.
- First row `1778738.pbs1` started on `hnode158`.
  - PBS allocation is `Resource_List.ngpus=8` even for `1n4g`; therefore
    `1n4g` measures four used GPUs on a full HF node, not an efficient
    four-GPU allocation.
  - First logged step: `data_wait=12.542s`, `step_time=10.571s`; warmup should
    be excluded in the parser.

Benchmark results so far:
- Parsed with `warmup_steps=10`.
- Valid rows:
  - `1n4g`, global batch 16, deterministic 1: `6.64-6.67 sec/step`,
    projected e300 `~113h`.
  - `1n4g`, global batch 16, deterministic 0: `0.83-0.85 sec/step`,
    projected e300 `~14h`.
  - `1n8g`, global batch 16, deterministic 1: `3.73 sec/step`,
    projected e300 `~63h`.
  - `1n8g`, global batch 16, deterministic 0: `0.61 sec/step`,
    projected e300 `~10.4h`.
  - `1n8g`, global batch 64, deterministic 1: `14.53-14.58 sec/step`,
    projected e300 `~62h`.
  - `1n8g`, global batch 64, deterministic 0: `2.22-2.23 sec/step`,
    projected e300 `~9.4-9.5h`.
- Staging `0/1` made no meaningful difference in valid 1-node rows.
  Data wait after warmup was about `0.002s`, so shared filesystem I/O is not
  the bottleneck for these runs.
- The largest observed factor is deterministic kernels:
  `DETERMINISTIC=1` is roughly `6x-8x` slower than `DETERMINISTIC=0`.
- `1n4g` global batch 64 failed with CUDA OOM on H200 (`~136GB` in use), so
  per-GPU batch 16 is too large for this model.
- Original `2n16g` rows failed because empty optional env vars were serialized
  as the literal string `''` in the multi-node node-entry script:
  `--train_num_workers "''"`. Staging rows also attempted to use
  `''/nerfmae_data/pretrain` as the local stage path.
- Fixed the multi-node empty-string normalization for
  `PRETRAIN_TRAIN_NUM_WORKERS`, `PRETRAIN_EVAL_NUM_WORKERS`, and
  `LOCAL_STAGE_ROOT`.
- Resubmitted only the `2n16g` benchmark rows:
  - Run id: `20260520_speed02_2n16g`
  - Jobs: `1779180.pbs1` through `1779187.pbs1`
  - Manifest:
    `output/launcher/abci3_pretrain_speed_bench/20260520_speed02_2n16g/manifest.tsv`

Recommended decision rule before e1200:
- Do not submit e1200 until the short benchmark identifies the fastest valid
  ABCI3 configuration.
- If `1n8g` is close to `2n16g` at global batch 16, prefer single-node for
  production efficiency unless the throughput batch grid clearly favors
  multi-node.
- Treat larger global batches as a protocol change unless downstream parity is
  checked; use them for speed diagnosis first, not for paper comparisons.
- Benchmark CUDA 12.x/PyTorch cu12x only after the current cu118 grid is
  measured, using a separate environment prefix.

Verification:
- `python -m py_compile nerf_mae/run_swin_mae3d.py nerf_mae/tools/parse_pretrain_speed_log.py`
- `bash -n nerf_mae/train_mae3d.sh`
- `bash -n nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs`
- `bash -n nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh`
- `bash -n nerf_mae/probe_scripts/submit_abci3_e300_gate.sh`
- `bash -n nerf_mae/probe_scripts/submit_abci3_pretrain_speed_benchmark.sh`
- Dry-run benchmark command generation.

## Experiment 26: Current ABCI3 Decision After Speed Bench

Snapshot:
- 2026-05-20 JST

Current queue cleanup:
- Canceled obsolete held null-prior diagnostic jobs that were tied to the old
  deterministic/slow submission path:
  `1777288.pbs1` through `1777295.pbs1`.
- Canceled remaining `2n16g` global-batch-64 speed benchmark jobs because this
  is a protocol change and is not needed for the current paper gate.
- Kept running:
  - `1777284.pbs1`: `cosine_ramp`, seed 2, e300 pretrain.
  - `1777286.pbs1`: `cosine_ramp_alpha_shuffle`, seed 2, e300 pretrain.
  - `1777285.pbs1` / `1777287.pbs1`: dependent FCOS jobs for the above.

Current e300 seed-2 pretrains:
- Runtime configuration:
  - `PRETRAIN_NODES=2`
  - `PRETRAIN_GPU_IDS=0-7` per node
  - world size `16`
  - global batch `16`
  - per-GPU microbatch `1`
  - deterministic mode on by script default
- This preserves the old global batch and deterministic flag, but it is not the
  exact itachi/A100 execution geometry. The itachi-era scripts used 4 GPUs with
  `BATCH_SIZE_PER_GPU=4`; these ABCI3 jobs use 16 GPUs with
  `BATCH_SIZE_PER_GPU=1`.
- Therefore, these runs should be treated as a useful gate/diagnostic for the
  e300 seed-2 question, not as the final clean multi-seed protocol if a paper
  table is needed.

Multi-seed protocol decision:
- Final multi-seed evidence should not mix execution protocols.
- Two valid choices:
  1. **Itachi-compatible protocol**:
     `1n4g`, global batch `16`, `BATCH_SIZE_PER_GPU=4`, deterministic on.
     This is closest to the old scripts but is slow on ABCI3 in the measured
     cu118 environment.
  2. **ABCI3-optimized protocol**:
     global batch `16`, deterministic off, using either:
     - `1n8g`: about `0.61 sec/step`, projected e300 `~10.4h`; better point
       efficiency.
     - `2n16g`: about `0.39-0.40 sec/step`, projected e300 `~6.6-6.8h`;
       faster wall-clock but uses two HF nodes.
- For paper-quality multi-seed tables, rerun all compared conditions under one
  of these two protocols. Do not combine itachi-era results, current
  deterministic 16GPU jobs, and future deterministic-off jobs as if they were
  the same controlled multi-seed experiment.

Recommended paper/time plan:
- Let the already-running e300 seed-2 pretrains finish because they are past the
  halfway point and answer the immediate gate:
  whether `cosine_ramp` still beats baseline and alpha-shuffle on another seed.
- Use the resulting FCOS numbers as a go/no-go diagnostic.
- If the signal survives, choose the ABCI3-optimized protocol for any new
  multi-seed or e600/e1200 work. The itachi-compatible path is too expensive on
  the measured ABCI3 setup unless exact historical comparability becomes the
  main requirement.
- Re-submit null-prior diagnostics only as single-seed ABCI3-optimized scouts;
  do not use the canceled old held jobs.

Multi-node PBS note:
- A multi-node PBS job is not expected to appear as `jobid[].pbs1`.
- The `[]` notation indicates a PBS job array, e.g. `1779238[].pbs1`.
- Multi-node pretrain jobs appear as a single job ID with `NDS=2` in `qstat`
  and `Resource_List.select=2:ncpus=192:mem=1920gb:ngpus=8...`.
- Current e300 pretrain jobs are therefore multi-node jobs:
  `1777284.pbs1` and `1777286.pbs1` each show `NDS=2` and two execution hosts.

## Experiment 27: Parallel Null-Prior Diagnostics, ABCI3 Optimized

Snapshot:
- 2026-05-20 JST

Rationale:
- These diagnostics are single-seed mechanism checks, not seed-variance claims.
- They can run in parallel with the ongoing e300 seed-2 gate because each PBS
  job receives its own HF node.
- Use the ABCI3-optimized protocol for new diagnostic work rather than the old
  deterministic multi-node path.

Submitted configuration:
- `1n8g`, single HF node per pretrain.
- `PRETRAIN_GPU_IDS=0-7`.
- `PRETRAIN_BATCH_SIZE_PER_GPU=2`, global batch `16`.
- `DETERMINISTIC=0`.
- `EPOCHS=100`, seed `1`.
- `PRETRAIN_EVAL_INTERVAL=100`, `PRETRAIN_CHECKPOINT_INTERVAL=100`.
- `RUN_SUFFIX=abci3diag_opt1n8g_det0`.
- Submit log dir:
  `output/launcher/abci3_null_prior_diag_opt1n8g_det0_e100`.

Submitted jobs:
- `1779500.pbs1`: `alpha_target_only_no_pos`, e100 seed1 pretrain.
- `1779501.pbs1`: dependent FCOS for `alpha_target_only_no_pos`.
- `1779502.pbs1`: `baseline_no_pos`, e100 seed1 pretrain.
- `1779503.pbs1`: dependent FCOS for `baseline_no_pos`.
- `1779504.pbs1`: `alpha_target_only_coord_jitter`, e100 seed1 pretrain.
- `1779505.pbs1`: dependent FCOS for `alpha_target_only_coord_jitter`.
- `1779506.pbs1`: `cosine_coord_jitter`, e100 seed1 pretrain.
- `1779507.pbs1`: dependent FCOS for `cosine_coord_jitter`.

Intended readout:
- no-pos pair:
  compare `alpha_target_only_no_pos` against `baseline_no_pos` to test whether
  target-alpha shortcut strength depends on positional embeddings.
- coord-jitter pair:
  compare `alpha_target_only_coord_jitter` and `cosine_coord_jitter` against
  their non-jitter references to test whether canonical layout memorization is
  driving the effect.

Results:
- All four optimized diagnostic pretrains completed successfully:
  - `1779500.pbs1`, `alpha_target_only_no_pos`, walltime `03:51:07`.
  - `1779502.pbs1`, `baseline_no_pos`, walltime `03:50:56`.
  - `1779504.pbs1`, `alpha_target_only_coord_jitter`, walltime `03:52:31`.
  - `1779506.pbs1`, `cosine_coord_jitter`, walltime `03:52:26`.
- All four dependent FCOS jobs completed successfully:
  - `1779501.pbs1`, `alpha_target_only_no_pos`, walltime `10:17:45`.
  - `1779503.pbs1`, `baseline_no_pos`, walltime `09:05:08`.
  - `1779505.pbs1`, `alpha_target_only_coord_jitter`, walltime `09:13:45`.
  - `1779507.pbs1`, `cosine_coord_jitter`, walltime `10:08:33`.

| condition | AP@50 | AP@25 | AP@75 | Recall@50 top300 | Recall@25 top300 |
|---|---:|---:|---:|---:|---:|
| `baseline_no_pos` | 0.5371 | 0.7832 | 0.0899 | 0.6912 | 0.9559 |
| `alpha_target_only_no_pos` | 0.4015 | 0.7394 | 0.0725 | 0.5956 | 0.9338 |
| `alpha_target_only_coord_jitter` | 0.4954 | 0.7888 | 0.0622 | 0.6324 | 0.9412 |
| `cosine_coord_jitter` | 0.6219 | 0.8097 | 0.1031 | 0.7279 | 0.9485 |

Reading:
- The no-pos pair does not show a generic collapse: `baseline_no_pos` remains
  strong at `0.5371` AP@50, while `alpha_target_only_no_pos` drops to `0.4015`.
  This supports the hypothesis that the target-alpha-only prior relies on
  position-like structure more than the full visible-input baseline does.
- Coord-jitter does not kill the signal. `alpha_target_only_coord_jitter`
  remains competitive at `0.4954` AP@50, and `cosine_coord_jitter` is very strong
  at `0.6219` AP@50. This argues against a simple "canonical room layout
  memorization only" explanation.
- The combined read is that null/target-alpha behavior is not a pure bad
  shortcut. It looks like a useful structural prior that is partly
  position-dependent, while the cosine curriculum remains robust under
  coordinate jitter in this single-seed scout.

## Experiment 28: e300 Seed-2 Gate Status

Snapshot:
- 2026-05-21 JST

Pretrain status:
- `cosine_ramp e300 seed2` produced:
  `output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed2_abci3gb16_16g/epoch_300.pt`.
- `cosine_ramp_alpha_shuffle e300 seed2` produced:
  `output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_alpha_shuffle_p1.0_e300_seed2_abci3gb16_16g/epoch_300.pt`.
- Both jobs nevertheless exited with status `1` after training/eval because the
  then-running `train_mae3d.sh` hit a shell syntax error in the optional
  augmentation argument block during the multi-node tail. The checkpoints were
  written before that error.
- Because the pretrain PBS jobs exited nonzero, the original `afterok` FCOS jobs
  did not run and no e300 seed-2 FCOS eval was produced automatically.

Action taken:
- Verified current shell scripts with:
  - `bash -n nerf_mae/train_mae3d.sh`
  - `bash -n nerf_mae/probe_scripts/abci3_e300_gate_fcos.pbs`
  - `bash -n nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh`
- Submitted FCOS-only retry jobs using the existing checkpoints:
  - `1783994.pbs1`: `cosine_ramp e300 seed2` FCOS.
  - `1783995.pbs1`: `cosine_ramp_alpha_shuffle e300 seed2` FCOS.
- Submit log dir:
  `output/launcher/abci3_e300_gate_gb16_16g_s2_fcos_retry`.

Pending:
- The e300 seed-2 gate cannot be judged until `1783994.pbs1` and
  `1783995.pbs1` finish and produce their FCOS `eval.json` files.

Results:
- FCOS retry jobs completed successfully:
  - `1783994.pbs1`, `cosine_ramp e300 seed2`, walltime `08:29:54`.
  - `1783995.pbs1`, `cosine_ramp_alpha_shuffle e300 seed2`, walltime `08:12:32`.

Current ABCI eval table:

| condition | seed | AP@50 | AP@25 | AP@75 | Recall@50 top300 | Recall@25 top300 |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_e300` | 1 | 0.4695 | 0.7956 | 0.0869 | 0.6618 | 0.9559 |
| `cosine_ramp_e300` | 1 | 0.5539 | 0.8249 | 0.1135 | 0.7059 | 0.9632 |
| `shuffle_e300` | 1 | 0.4162 | 0.7613 | 0.0326 | 0.5956 | 0.9559 |
| `baseline_e300` | 2 | 0.5038 | 0.8126 | 0.0780 | 0.6838 | 0.9559 |
| `cosine_ramp_e300` | 2 | 0.5366 | 0.8291 | 0.0881 | 0.6912 | 0.9485 |
| `shuffle_e300` | 2 | 0.3964 | 0.7111 | 0.0517 | 0.6176 | 0.9632 |

Paired AP@50 differences:
- Seed 1:
  - `cosine - baseline`: `+0.0843`.
  - `cosine - shuffle`: `+0.1377`.
  - `shuffle - baseline`: `-0.0534`.
- Seed 2:
  - `cosine - baseline`: `+0.0328`.
  - `cosine - shuffle`: `+0.1402`.
  - `shuffle - baseline`: `-0.1074`.

Two-seed AP@50 means:
- `baseline_e300`: `0.4867`.
- `cosine_ramp_e300`: `0.5452`.
- `shuffle_e300`: `0.4063`.
- Mean `cosine - baseline`: `+0.0586`.
- Mean `cosine - shuffle`: `+0.1390`.

Caveat:
- The historical seed-1 artifact used in earlier notes has
  `cosine_ramp_e300 AP@50=0.5987`, `baseline_e300 AP@50=0.4903`, and
  `shuffle_e300 AP@50=0.4138`. The current ABCI clean seed-1 eval is lower for
  cosine (`0.5539`) and baseline (`0.4695`) but preserves the same ordering.
- For paper-quality multi-seed evidence, avoid mixing the old historical
  artifact and current ABCI reruns as if they were one protocol.

Reading:
- The seed-2 result keeps the direction of the cosine curriculum signal:
  `cosine_ramp_e300` beats same-seed `baseline_e300`, but the gain is modest
  (`+0.0328` AP@50) compared with seed 1.
- The mechanism control is more stable: `shuffle_e300` is far below cosine in
  both seeds (`~+0.14` AP@50 gap for cosine over shuffle).
- The current status is stronger than a single-seed candidate, but still not a
  final paper claim. It supports continuing the cosine/prior path, with future
  multi-seed or longer-budget runs done under one unified ABCI protocol.

## Experiment 29: Finetune-Seed Protocol and Phase-1 Relaunch

Snapshot:
- 2026-05-24 JST

Strategy update:
- Scout / diagnosis runs should remain `1 pretrain seed x 1 finetune seed`.
- Paper-scale comparisons should use `1 pretrain seed x 3 finetune seeds` as
  the default unit.
- Pretrain-seed replication is reserved for final flagship comparisons only,
  e.g. `baseline` versus the final selected method.
- `cosine_coord_jitter` is treated as a strong empirical upper bound /
  competitor, not as the main architectural method.
- Coordinate-only alpha prior remains the first architecture decision
  diagnostic. D-MAE scouts should wait for this readout and should be
  asymmetric/hierarchical rather than a full target-alpha bypass.

Code changes:
- Added `FINETUNE_SEED` support to
  `nerf_mae/probe_scripts/abci3_e300_gate_fcos.pbs`.
  - `SEED` continues to identify the pretrain checkpoint.
  - `FINETUNE_SEED` controls the FCOS seed.
  - When the seeds differ, FCOS outputs use
    `_preseed{p}_ftseed{f}_fcos...` naming.
- Updated `nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh` so
  `GATE_JOBS` accepts either:
  - `kind:condition:epochs:pretrain_seed`
  - `kind:condition:epochs:pretrain_seed:finetune_seed`
- Added `baseline_coord_jitter` as a diagnostic pretrain/FCOS condition.
- Added `nerf_mae/tools/build_results_table.py`, which writes:
  `results/shortcut_probe_artifacts/results_table.csv`.
- Extended `nerf_mae/tools/coord_prior_alpha_probe.py` to report MAE, binary
  BCE, occupied AP, threshold sweep, and best IoU threshold in addition to MSE
  and fixed-threshold occupancy metrics.

Validation:
- Passed:
  - `bash -n nerf_mae/probe_scripts/abci3_e300_gate_fcos.pbs`
  - `bash -n nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs`
  - `bash -n nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh`
  - `python -m py_compile nerf_mae/tools/build_results_table.py nerf_mae/tools/coord_prior_alpha_probe.py`
- Generated `results_table.csv` with 93 existing eval rows.

Submitted jobs:
- FCOS-only finetune seed grid using existing e300 pretrain seed-1 ABCI3 clean
  checkpoints:

| job | condition | pretrain seed | finetune seed | notes |
|---|---|---:|---:|---|
| `1794884.pbs1` | `baseline_e300` | 1 | 2 | FCOS only |
| `1794885.pbs1` | `baseline_e300` | 1 | 3 | FCOS only |
| `1794886.pbs1` | `cosine_ramp_e300` | 1 | 2 | FCOS only |
| `1794887.pbs1` | `cosine_ramp_e300` | 1 | 3 | FCOS only |
| `1794888.pbs1` | `shuffle_e300` | 1 | 2 | FCOS only |
| `1794889.pbs1` | `shuffle_e300` | 1 | 3 | FCOS only |

- Baseline coordinate-jitter control:

| job | condition | setting |
|---|---|---|
| `1794890.pbs1` | `baseline_coord_jitter e100 seed1` | 1n8g, global batch 16, deterministic off |
| `1794891.pbs1` | `baseline_coord_jitter` FCOS | depends on `1794890.pbs1` |

- Coordinate-only tiny prior rerun with expanded metrics:

| job | output |
|---|---|
| `1794892.pbs1` | `output/coord_prior_alpha_probe_metrics_20260524` |

Immediate readout once jobs finish:
- e300 paper-scale gate should use the unified pretrain-seed-1 / finetune
  seeds 1,2,3 table:
  - `cosine_ramp_e300 - baseline_e300`
  - `cosine_ramp_e300 - shuffle_e300`
- `baseline_coord_jitter` should be compared against `cosine_coord_jitter`
  and non-jitter `baseline_no_pos` / `baseline_e300` style controls.
- Coordinate-only prior should be judged by occupied AP, best-IoU threshold,
  and whether it is doing more than an almost-all-occupied high-recall prior.

Pending:
- D-MAE v1/v2/v3 implementation should begin after the updated
  coordinate-only prior metrics are available, unless the metric rerun simply
  confirms the earlier weak/trivial coordinate-only result.

Coordinate-only rerun result:
- Job `1794892.pbs1` completed and wrote:
  `output/coord_prior_alpha_probe_metrics_20260524/coord_prior_alpha_probe.md`.

| metric | value |
|---|---:|
| Val MSE | 0.052610 |
| Val MAE | 0.141836 |
| Binary BCE | 0.707545 |
| Occupied AP | 0.4892 |
| Fixed-threshold occupied IoU | 0.3056 |
| Best threshold / IoU | 0.050 / 0.3184 |
| Precision / recall at threshold 0.01 | 0.3058 / 0.9981 |
| Target / predicted occupied rate | 0.3022 / 0.9864 |

Reading:
- The coordinate-only prior is not a strong enough standalone explanation of
  the downstream signal. It largely behaves like a high-recall, broadly
  occupied predictor at the default alpha threshold.
- This supports moving to D-MAE scouts rather than reducing the story to a
  coordinate-only Front3D prior.

## Experiment 30: D-MAE Scout Implementation and Launch

Snapshot:
- 2026-05-24 JST

Implementation:
- Added D-MAE scout controls to `SwinTransformer_MAE3D_Probe`:
  - `probe_decomp_mode=target_alpha_gated_rgb`
  - `probe_decomp_mode=hierarchical_concat`
  - `probe_decomp_mode=hierarchical_film`
- `target_alpha_gated_rgb` is a loss-only scout that uses continuous
  target-alpha weighting for RGB loss.
- `hierarchical_concat` adds a low-level alpha structure head from stage-0 Swin
  features and concatenates the predicted alpha structure into the appearance
  head.
- `hierarchical_film` uses the same predicted alpha structure but conditions
  appearance features with FiLM.
- Existing baseline/curriculum modes remain unchanged unless
  `probe_decomp_mode` is explicitly set.
- FCOS checkpoint loading for pretrained Swin is now non-strict so D-MAE-only
  decoder/head keys do not block transfer to the downstream backbone.

Validation:
- Passed:
  - `python -m py_compile nerf_mae/model/mae/shortcut_probe.py nerf_mae/run_swin_mae3d.py`
  - `python -m py_compile nerf_rpn/model/feature_extractor.py`
  - `bash -n nerf_mae/train_mae3d.sh`
  - `bash -n nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs`
  - `bash -n nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh`
- Synthetic smoke tests passed for:
  - `target_alpha_gated_rgb`
  - `hierarchical_concat`
  - `hierarchical_film`

Initial launch:
- Submitted first D-MAE scout wave:
  - `1795027.pbs1` / `1795028.pbs1`: `dmae_target_alpha_gated_rgb`
  - `1795029.pbs1` / `1795030.pbs1`: `dmae_hier_concat`
  - `1795031.pbs1` / `1795032.pbs1`: `dmae_hier_film`

Early failures and fixes:
- `dmae_target_alpha_gated_rgb` failed immediately because the CLI choices for
  `--probe_rgb_loss` had not been extended to include `target_alpha`.
  Fixed in `nerf_mae/run_swin_mae3d.py`.
- `dmae_hier_concat` and `dmae_hier_film` failed under DDP because the base
  decoder `self.out` was no longer used by the overridden decomposed output
  path. Fixed by replacing `self.out` with `nn.Identity()` for hierarchical
  D-MAE modes, so no unused parameters remain registered.

Rerun jobs:

| job | condition | notes |
|---|---|---|
| `1795034.pbs1` | `dmae_target_alpha_gated_rgb e100 seed1` | pretrain rerun |
| `1795035.pbs1` | `dmae_target_alpha_gated_rgb` FCOS | depends on `1795034.pbs1` |
| `1795036.pbs1` | `dmae_hier_concat e100 seed1` | pretrain rerun |
| `1795037.pbs1` | `dmae_hier_concat` FCOS | depends on `1795036.pbs1` |
| `1795038.pbs1` | `dmae_hier_film e100 seed1` | pretrain rerun |
| `1795039.pbs1` | `dmae_hier_film` FCOS | depends on `1795038.pbs1` |

Run setting:
- `1n8g`, global batch `16`, deterministic off, e100 scout, single seed.

Reading target:
- D-MAE should only be promoted if it is competitive with the strong
  `cosine_coord_jitter`/cosine controls while offering a cleaner
  structure-appearance decomposition story.

## Experiment 31: ABCI3 e300 Finetune-Seed Gate and D-MAE Scout Readout

Snapshot:
- 2026-05-25 JST

Generated summaries:
- `results/shortcut_probe_artifacts/results_table.csv`
  - regenerated with `102` eval rows.
- `output/launcher/abci3_e300_preseed1_ftseed_grid_20260524/summary_final.md`
- `output/launcher/abci3_e300_preseed1_ftseed_grid_20260524/summary_final.csv`
- `output/launcher/abci3_e300_preseed1_ftseed_grid_20260524/summary_final.json`

Clean ABCI3 e300 gate:

| condition | pretrain seed | finetune seed | AP@50 | AP@25 | AP@75 | Recall@50 top300 |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_e300` | 1 | 1 | 0.4695 | 0.7956 | 0.0869 | 0.6618 |
| `baseline_e300` | 1 | 2 | 0.4862 | 0.7903 | 0.0916 | 0.6250 |
| `baseline_e300` | 1 | 3 | 0.5258 | 0.8121 | 0.0947 | 0.6985 |
| `cosine_ramp_e300` | 1 | 1 | 0.5539 | 0.8249 | 0.1135 | 0.7059 |
| `cosine_ramp_e300` | 1 | 2 | 0.5704 | 0.8379 | 0.1176 | 0.6838 |
| `cosine_ramp_e300` | 1 | 3 | 0.5928 | 0.8255 | 0.0891 | 0.7132 |
| `shuffle_e300` | 1 | 1 | 0.4162 | 0.7613 | 0.0326 | 0.5956 |
| `shuffle_e300` | 1 | 2 | 0.4187 | 0.7396 | 0.0750 | 0.5956 |
| `shuffle_e300` | 1 | 3 | 0.4532 | 0.7306 | 0.0282 | 0.6397 |

Mean AP@50 over finetune seeds:

| condition | mean AP@50 | sample std |
|---|---:|---:|
| `baseline_e300` | 0.4938 | 0.0289 |
| `cosine_ramp_e300` | 0.5723 | 0.0195 |
| `shuffle_e300` | 0.4294 | 0.0207 |

Paired AP@50 differences:

| pretrain seed | finetune seed | comparison | AP@50 diff |
|---:|---:|---|---:|
| 1 | 1 | `cosine - baseline` | +0.0843 |
| 1 | 2 | `cosine - baseline` | +0.0842 |
| 1 | 3 | `cosine - baseline` | +0.0670 |
| 1 | 1 | `cosine - shuffle` | +0.1377 |
| 1 | 2 | `cosine - shuffle` | +0.1517 |
| 1 | 3 | `cosine - shuffle` | +0.1396 |

Reading:
- The e300 gate passes under the current finetune-seed protocol:
  `cosine_ramp_e300` beats `baseline_e300` in `3/3` paired finetune seeds and
  beats `shuffle_e300` in `3/3` paired finetune seeds.
- The mean same-budget gain is substantial for this protocol:
  `+0.0785` AP@50 over baseline.
- The mechanism/control gap is even larger:
  `+0.1430` AP@50 over shuffled target-alpha.
- This supports continuing the alpha-to-RGBA curriculum / structural prior
  path. It should still be described as `1` pretrain seed with `3` finetune
  seeds, not as pretrain-seed-robust evidence.

Coordinate-only prior:

| metric | value |
|---|---:|
| Val MSE | 0.052610 |
| Val MAE | 0.141836 |
| Binary BCE | 0.707545 |
| Occupied AP | 0.4892 |
| Fixed-threshold occupied IoU | 0.3056 |
| Best threshold / IoU | 0.050 / 0.3184 |
| Precision / recall at threshold 0.01 | 0.3058 / 0.9981 |
| Target / predicted occupied rate | 0.3022 / 0.9864 |

Reading:
- Coordinate-only is a weak, high-recall prior rather than a strong standalone
  solution. This supports D-MAE / hierarchical structure modeling over a pure
  coordinate-only explanation.

D-MAE e100 scout results:

| condition | pretrain seed | finetune seed | AP@50 | AP@25 | AP@75 | Recall@50 top300 |
|---|---:|---:|---:|---:|---:|---:|
| `dmae_target_alpha_gated_rgb` | 1 | 1 | 0.5045 | 0.8074 | 0.0832 | 0.6544 |
| `dmae_hier_concat` | 1 | 1 | 0.5778 | 0.8312 | 0.1055 | 0.6912 |
| `dmae_hier_film` | 1 | 1 | 0.5443 | 0.8062 | 0.0709 | 0.7059 |
| `cosine_coord_jitter` | 1 | 1 | 0.6219 | 0.8097 | 0.1031 | 0.7279 |
| `alpha_target_only_coord_jitter` | 1 | 1 | 0.4954 | 0.7888 | 0.0622 | 0.6324 |
| `baseline_no_pos` | 1 | 1 | 0.5371 | 0.7832 | 0.0899 | 0.6912 |
| `alpha_target_only_no_pos` | 1 | 1 | 0.4015 | 0.7394 | 0.0725 | 0.5956 |

Reading:
- `dmae_hier_concat` is the best D-MAE scout so far and is meaningfully better
  than the loss-only `target_alpha_gated_rgb` scout.
- `dmae_hier_concat` is also better than `baseline_no_pos`, but it is still
  below `cosine_coord_jitter` on AP@50 (`0.5778` vs `0.6219`).
- Therefore D-MAE is promising but not yet promoted to the main method.
  The next method step should either improve the hierarchical concat design or
  treat `cosine_coord_jitter` as the empirical upper-bound competitor.

Missing / retried control:
- `baseline_coord_jitter e100 seed1` produced:
  `output/nerf_mae/results/nerfmae_baseline_coord_jitter_p1.0_e100_seed1_abci3diag_opt1n8g_det0/epoch_100.pt`
- The original pretrain job wrote the checkpoint and reconstruction eval, but
  then exited nonzero due to an old shell quoting error after the eval block.
  Because the FCOS job depended on `afterok`, it did not run.
- Current `nerf_mae/train_mae3d.sh` passes `bash -n`.
- Submitted an FCOS-only retry from the existing checkpoint:
  `1796104.pbs1`
  - log dir:
    `output/launcher/abci3_baseline_coord_jitter_fcos_retry_20260525`
  - setting: `FCOS_NUM_EPOCHS=1000`, pretrain disabled, existing checkpoint
    reused.

Current decision:
- Gate 1 is positive under the intended finetune-seed protocol.
- Coordinate-only prior does not explain the result away.
- D-MAE hierarchical concat deserves another focused iteration, but the current
  strongest empirical method/control pair remains `cosine_ramp` versus
  `shuffle`, with `cosine_coord_jitter` as a strong scout/upper-bound control.

## Experiment 32: D-MAE Code Integrity and Coord-Jitter Scout Launch

Snapshot:
- 2026-05-25 JST

Git / compile integrity:
- Current HEAD:
  `94f8486a005c3bae31539547b48f6776ab7fcf75`
- Passed:
  - `python -m py_compile nerf_mae/model/mae/shortcut_probe.py nerf_mae/run_swin_mae3d.py nerf_rpn/model/feature_extractor.py`
  - `python -m py_compile nerf_mae/tools/check_fcos_checkpoint_load.py nerf_mae/tools/build_results_table.py`
  - `bash -n nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs`
  - `bash -n nerf_mae/probe_scripts/abci3_e300_gate_fcos.pbs`
  - `bash -n nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh`

Aggregation update:
- `results/shortcut_probe_artifacts/results_table.csv` now includes
  `git_hash`.
- Regenerated table has `102` rows and git hash
  `94f8486a005c3bae31539547b48f6776ab7fcf75`.

FCOS checkpoint-load sanity:
- Added `nerf_mae/tools/check_fcos_checkpoint_load.py`.
- Ran it on:
  `output/nerf_mae/results/nerfmae_dmae_hier_concat_p1.0_e100_seed1_abci3dmae_e100_det0_1n8g/epoch_100.pt`
- Outputs:
  - `results/shortcut_probe_artifacts/load_sanity/dmae_hier_concat_e100_seed1_fcos_load_sanity.json`
  - `results/shortcut_probe_artifacts/load_sanity/dmae_hier_concat_e100_seed1_fcos_load_sanity.md`

Load sanity result:

| check | value |
|---|---:|
| FCOS instantiated | true |
| pass | true |
| encoder missing keys | 0 |
| encoder unexpected keys | 0 |
| encoder exact tensor ratio | 1.000000 |
| encoder exact numel ratio | 1.000000 |
| `pos_embed` exact | 1 / 1 |
| `patch_partition` exact | 4 / 4 |
| `stages.*` exact | 345 / 345 |

Expected non-strict load keys:
- Missing:
  - `out.conv.weight`
  - `out.conv.bias`
- Unexpected:
  - D-MAE-specific `decomp_structure_head.*`
  - D-MAE-specific `decomp_rgb_head.*`

Reading:
- FCOS is using the D-MAE pretrained encoder exactly for
  `pos_embed`, `patch_partition`, and all `stages.*` tensors.
- D-MAE-only heads are discarded as expected. The existing D-MAE downstream AP
  is therefore a valid pretrained-backbone transfer result rather than an
  artifact of failed checkpoint loading.

Code changes for next scout:
- Added `dmae_hier_concat_coord_jitter` as a diagnostic condition in:
  - `nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs`
  - `nerf_mae/probe_scripts/abci3_e300_gate_fcos.pbs`
  - `nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh`
- This condition uses `probe_decomp_mode=hierarchical_concat` plus the same
  coordinate-jitter defaults as the existing coord-jitter scouts:
  - `ROTATE_PROB=1.0`
  - `FLIP_PROB=0.5`
  - `ROT_SCALE_PROB=0.0`
  - `COORD_SHIFT_PROB=1.0`
  - `COORD_SHIFT_MAX_VOXELS=8`

Submitted critical scout:

| job | condition | setting |
|---|---|---|
| `1797098.pbs1` | `dmae_hier_concat_coord_jitter e100 seed1` | pretrain, 1n8g, global batch 16, deterministic off |
| `1797099.pbs1` | `dmae_hier_concat_coord_jitter` FCOS | depends on `1797098.pbs1` |

Current queue status:
- `1796104.pbs1`: `baseline_coord_jitter` FCOS retry is running.
- `1797098.pbs1`: `dmae_hier_concat_coord_jitter` pretrain is running.
- `1797099.pbs1`: dependent FCOS is held until pretrain succeeds.

Next readout:
- Compare `baseline_coord_jitter`, `cosine_coord_jitter`, and
  `dmae_hier_concat_coord_jitter` on AP@50, AP@75, AP@75/AP@50, and proposal
  IoU diagnostics before deciding whether D-MAE remains the main method path.

Proposal-quality diagnostics:
- Added `nerf_rpn/tools/summarize_proposal_quality.py`.
- Added GPU runner:
  `nerf_rpn/tools/abci3_proposal_quality_summary.pbs`.
- Login-node execution is not sufficient because the existing rotated-IoU
  implementation forces CUDA internally. The summary was therefore run as
  `1797124.pbs1` on `rt_HG`.
- Outputs:
  - `results/shortcut_probe_artifacts/proposal_quality/e100_dmae_coord_controls.json`
  - `results/shortcut_probe_artifacts/proposal_quality/e100_dmae_coord_controls.md`
  - `results/shortcut_probe_artifacts/proposal_quality/e300_gate_pre1_ft123.json`
  - `results/shortcut_probe_artifacts/proposal_quality/e300_gate_pre1_ft123.md`

Existing e100 proposal-quality readout:

| condition | AP@50 | AP@75 | AP75/AP50 | mean IoU | frac IoU>=0.5 | center err >=0.5 | size err >=0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_no_pos` | 0.5371 | 0.0899 | 0.1674 | 0.0670 | 0.0186 | 0.0512 | 0.3113 |
| `alpha_coord_jitter` | 0.4954 | 0.0622 | 0.1255 | 0.0634 | 0.0169 | 0.0578 | 0.1716 |
| `cosine_coord_jitter` | 0.6219 | 0.1031 | 0.1657 | 0.0635 | 0.0196 | 0.0498 | 0.1818 |
| `dmae_gate` | 0.5045 | 0.0832 | 0.1650 | 0.0613 | 0.0175 | 0.0491 | 0.2988 |
| `dmae_concat` | 0.5778 | 0.1055 | 0.1826 | 0.0649 | 0.0184 | 0.0501 | 0.2607 |
| `dmae_film` | 0.5443 | 0.0709 | 0.1302 | 0.0686 | 0.0188 | 0.0525 | 0.2563 |

Reading:
- AP@50 alone under-rates `dmae_concat`. It is below
  `cosine_coord_jitter` on AP@50, but slightly above it on AP@75 and has the
  best AP75/AP50 ratio among the listed e100 scouts.
- Proposal mean IoU and frac-IoU>=0.5 are very close across these runs, so the
  D-MAE/cosine difference is not just a large shift in raw proposal IoU
  coverage.
- `dmae_film` has the highest mean proposal IoU but weak AP@75/AP@50. This
  supports keeping `hierarchical_concat` as the D-MAE variant to develop.

Clean e300 proposal-quality readout:
- `cosine_e300` improves AP@50 in all three finetune seeds.
- AP75/AP50 is higher than baseline in finetune seeds 1/2, but seed 3 is lower
  (`0.1502`), so AP@75 should be treated as a localization diagnostic rather
  than the primary e300 sample-efficiency claim.
- Proposal IoU coverage is again similar across baseline/cosine/shuffle,
  consistent with the earlier reading that the main e300 gain is not simply
  more top300 geometric coverage.
