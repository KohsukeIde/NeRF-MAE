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
- Regenerated table has `102` rows. It was first generated at
  `94f8486a005c3bae31539547b48f6776ab7fcf75`, then regenerated after the
  integrity commit at `6dff2390a647c6f1762ba8288466951b5beb1b9a`.

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

Integrity report and public commit status:
- Added:
  `results/shortcut_probe_artifacts/integrity_report.md`
- Pushed commit `6dff2390a647c6f1762ba8288466951b5beb1b9a` to `origin/main`.
- At that check, `git ls-remote origin main` reported:
  `6dff2390a647c6f1762ba8288466951b5beb1b9a refs/heads/main`.
- A later report/figure commit advances `origin/main`; use `git ls-remote
  origin main` for the current public head.
- Regenerated:
  - `results/shortcut_probe_artifacts/results_table.csv`
  - `results/shortcut_probe_artifacts/load_sanity/dmae_hier_concat_e100_seed1_fcos_load_sanity.json`
  - `results/shortcut_probe_artifacts/load_sanity/dmae_hier_concat_e100_seed1_fcos_load_sanity.md`
  with git hash `6dff2390a647c6f1762ba8288466951b5beb1b9a`.

Decision criterion update:
- Do not use a flat "2 of 3" rule for D-MAE continuation.
- Use the tiered criterion recorded in
  `results/shortcut_probe_artifacts/integrity_report.md`:
  - Tier 1: D-MAE coord-jitter is AP@50-equal-or-better than
    `cosine_coord_jitter` and improves AP@75 or AP75/AP50.
  - Tier 2: D-MAE coord-jitter is within `0.02` AP@50 of
    `cosine_coord_jitter` and clearly improves localization/AP75 ratio.
  - Tier 3: D-MAE trails by at least `0.03` AP@50 and does not win on AP@75 or
    AP75/AP50; in this case D-MAE drops to appendix/ablation.

Proposal figure outputs:
- Added `nerf_rpn/tools/plot_proposal_quality.py`.
- Generated:
  - `results/shortcut_probe_artifacts/proposal_quality/e100_dmae_coord_controls.png`
  - `results/shortcut_probe_artifacts/proposal_quality/e300_gate_pre1_ft123.png`
- These are compact summary figures for AP@50/AP@75, AP75/AP50, proposal IoU
  coverage, and center/size error. Full histogram/scatter figures should be
  generated after `dmae_hier_concat_coord_jitter` finishes so the final
  candidate is included.

## Experiment 33: Coord-Jitter Decision Readout

Snapshot:
- 2026-05-26 JST

Completed jobs:

| job | condition | status |
|---|---|---|
| `1796104.pbs1` | `baseline_coord_jitter` FCOS retry | complete |
| `1797098.pbs1` | `dmae_hier_concat_coord_jitter` pretrain | complete |
| `1797099.pbs1` | `dmae_hier_concat_coord_jitter` FCOS | complete |

`qstat` no longer shows these NeRF-MAE decision jobs. Remaining running jobs at
this check are unrelated SSL/BSO jobs.

Updated aggregation:
- Regenerated `results/shortcut_probe_artifacts/results_table.csv`.
- The table now has `104` rows and includes both:
  - `baseline_coord_jitter`
  - `dmae_hier_concat_coord_jitter`

Decision metrics:

| condition | pretrain seed | finetune seed | AP@50 | AP@75 | AP75/AP50 | R50@300 |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_coord_jitter` | 1 | 1 | 0.5564 | 0.1015 | 0.1824 | 0.6765 |
| `cosine_coord_jitter` | 1 | 1 | 0.6219 | 0.1031 | 0.1657 | 0.7279 |
| `dmae_hier_concat` | 1 | 1 | 0.5778 | 0.1055 | 0.1826 | 0.6912 |
| `dmae_hier_concat_coord_jitter` | 1 | 1 | 0.5212 | 0.0858 | 0.1646 | 0.6838 |

Key deltas:

| comparison | AP@50 delta | AP@75 delta | AP75/AP50 delta | R50@300 delta |
|---|---:|---:|---:|---:|
| `cosine_coord_jitter - baseline_coord_jitter` | +0.0655 | +0.0016 | -0.0167 | +0.0515 |
| `dmae_hier_concat_coord_jitter - cosine_coord_jitter` | -0.1007 | -0.0173 | -0.0011 | -0.0441 |
| `dmae_hier_concat_coord_jitter - baseline_coord_jitter` | -0.0352 | -0.0157 | -0.0178 | +0.0074 |

Reading:
- `baseline_coord_jitter` is below `cosine_coord_jitter` by AP@50 `0.0655`.
  Therefore coord-jitter alone does not explain the strong
  `cosine_coord_jitter` scout. The target-alpha-to-RGBA cosine curriculum still
  carries a meaningful signal under coordinate augmentation.
- `dmae_hier_concat_coord_jitter` is worse than `cosine_coord_jitter` on AP@50,
  AP@75, AP75/AP50, and R50@300.
- It is also worse than `baseline_coord_jitter` on AP@50, AP@75, and
  AP75/AP50, with only a small R50@300 improvement.
- Under the pre-registered tiered criterion, this is Tier 3:
  `dmae_hier_concat_coord_jitter` trails `cosine_coord_jitter` by more than
  `0.03` AP@50 and does not compensate with AP@75 or AP75/AP50.

Decision:
- Do not promote current D-MAE as the main method path.
- Keep `dmae_hier_concat` as a useful ablation/localization diagnostic because
  the no-jitter version still has the best AP75/AP50 ratio among the current
  D-MAE scouts.
- Main experimental direction should shift to the cosine/coord-jitter
  curriculum analysis path unless a substantially different D-MAE design is
  proposed.
- The immediate next paper-scale question is not "does coord-jitter alone win?"
  but "which cosine/coord-jitter/curriculum row should be validated with the
  agreed 1 pretrain seed x 3 finetune seeds protocol?"

Additional proposal-quality diagnostic:
- Added `nerf_rpn/tools/abci3_proposal_quality_coord_jitter_decision.pbs`.
- Ran short HG job `1800305.pbs1`.
- Outputs:
  - `results/shortcut_probe_artifacts/proposal_quality/e100_coord_jitter_decision.json`
  - `results/shortcut_probe_artifacts/proposal_quality/e100_coord_jitter_decision.md`
  - `results/shortcut_probe_artifacts/proposal_quality/e100_coord_jitter_decision.png`

Proposal-quality readout:

| condition | AP@50 | AP@75 | AP75/AP50 | R50@300 | mean IoU | frac IoU>=0.5 | first TP rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_coord_jitter` | 0.5564 | 0.1015 | 0.1824 | 0.6765 | 0.0632 | 0.0180 | 1.2941 |
| `cosine_coord_jitter` | 0.6219 | 0.1031 | 0.1657 | 0.7279 | 0.0635 | 0.0196 | 1.0588 |
| `dmae_hier_concat` | 0.5778 | 0.1055 | 0.1826 | 0.6912 | 0.0649 | 0.0184 | 1.3529 |
| `dmae_hier_concat_coord_jitter` | 0.5212 | 0.0858 | 0.1646 | 0.6838 | 0.0702 | 0.0182 | 1.2941 |

Proposal-quality reading:
- `dmae_hier_concat_coord_jitter` has the highest mean proposal IoU, but this
  does not translate into AP@50/AP@75 or better AP75/AP50.
- `cosine_coord_jitter` has the best AP@50, R50@300, frac-IoU>=0.5, and first
  TP rank among this comparison set.
- Therefore the D-MAE coord-jitter failure is not rescued by the proposal
  diagnostic. The current D-MAE design should remain an ablation rather than
  the main method.

## Experiment 34: Surface-Maturation MVP Implementation and Launch Plan

Snapshot:
- 2026-05-26 JST

Motivation:
- Current D-MAE coord-jitter is Tier 3 and should not be the main path.
- `cosine_coord_jitter` remains the strongest scout, but it is still an
  empirical curriculum rather than a mechanistic method.
- The next low-risk method scout is Surface-Maturation: use predicted alpha
  confidence as a stop-gradient gate for RGB reconstruction, with a nonzero
  floor to avoid eliminating RGB loss early.

Implementation source:
- Applied the provided overlay from:
  `surface_maturation_mvp.zip`

Added entrypoints and scripts:
- `nerf_mae/model/mae/surface_maturation_probe.py`
- `nerf_mae/run_swin_surface_maturation.py`
- `nerf_mae/run_swin_grad_monitor.py`
- `nerf_mae/probe_scripts/abci3_surface_maturation_pretrain.pbs`
- `nerf_mae/probe_scripts/abci3_input_alpha_curriculum_pretrain.pbs`
- `nerf_mae/probe_scripts/abci3_grad_conflict_monitor.pbs`
- `nerf_mae/probe_scripts/submit_surface_maturation_sweep.sh`

Integration changes:
- `nerf_mae/train_mae3d.sh` now supports `TRAIN_ENTRYPOINT`, defaulting to
  `run_swin_mae3d.py`. This lets the new wrappers reuse the existing optimized
  ABCI3 train script.
- `abci3_e300_gate_pretrain.pbs` and `abci3_e300_gate_fcos.pbs` now support
  `KIND=surface_maturation` conditions and dependent FCOS evaluation.
- `build_results_table.py` now records Surface-Maturation env columns:
  `SM_MODE`, `SM_CONFIDENCE`, `SM_W_MIN`, `SM_TAU`, `SM_K`,
  `SM_STOP_GATE_GRAD`, `SM_RGB_MASK`, and `SM_INPUT_RGB_CURRICULUM`.

Important implementation correction:
- The overlay originally used `_masked_mean(loss, mask * gate)`, which divides
  by the gated mass and makes the gate mostly a spatial reweighting.
- This was changed to normalize by the ungated base mask support:
  `(loss * mask * gate).sum() / mask.sum()`.
- This matches the intended Surface-Maturation behavior: the predicted-alpha
  gate controls RGB-loss strength while `SM_W_MIN` preserves a floor.

Default scout protocol:
- Pretrain: ABCI3 `1n8g`, global batch `16`, deterministic off.
- Coord-jitter is enabled for all Surface-Maturation scouts so the comparison
  is against the current relevant controls:
  - `baseline_coord_jitter`
  - `cosine_coord_jitter`
  - `dmae_hier_concat_coord_jitter`
- Dependent FCOS is submitted automatically after each pretrain.

Planned jobs:

| condition | setting |
|---|---|
| `surface_maturation_tau0p3_k10_w0p05` | `tau=0.3`, `k=10`, `w_min=0.05` |
| `surface_maturation_tau0p5_k20_w0p05` | `tau=0.5`, `k=20`, `w_min=0.05` |
| `surface_maturation_tau0p7_k30_w0p05` | `tau=0.7`, `k=30`, `w_min=0.05` |
| `input_alpha_curriculum` | `tau=0.5`, `k=20`, `w_min=0.05`, `SM_INPUT_RGB_CURRICULUM=cosine_release` |

Verification before launch:
- Passed:
  - `python -m py_compile nerf_mae/model/mae/surface_maturation_probe.py`
  - `python -m py_compile nerf_mae/run_swin_surface_maturation.py`
  - `python -m py_compile nerf_mae/run_swin_grad_monitor.py`
  - `python -m py_compile nerf_mae/tools/build_results_table.py`
  - `bash -n` for the modified train/PBS/submit scripts.
- Import smoke passed:
  `run_swin_surface_maturation` installs
  `SwinTransformer_MAE3D_SurfaceMaturation` as the training model.

Decision criteria:
- Promote Surface-Maturation only if it gets close to `cosine_coord_jitter` on
  AP@50 or improves AP@75/AP75-over-AP50 while retaining reasonable AP@50.
- If all Surface-Maturation settings and the input-alpha scout are clearly
  below `cosine_coord_jitter`, stop new method scouts and treat
  `cosine_coord_jitter` as the empirical best direction for the next paper
  framing pass.

Launch status:
- Implementation pushed in commit:
  `d73c259` (`Add surface maturation scout wrappers`).
- Runtime-path fix for the gradient monitor pushed in commit:
  `c788718` (`Fix grad monitor ABCI runtime paths`).

Gradient monitor:
- Initial monitor jobs `1800726.pbs1` and `1800727.pbs1` failed immediately
  because `_probe_common.sh` fell back to an old `/mnt/urashima/...` runtime
  path. The PBS was fixed to set repo-local runtime/cache/wandb paths.
- Resubmitted:

| job | condition | run name |
|---|---|---|
| `1800742.pbs1` | baseline | `grad_conflict_baseline_seed1_retry2` |
| `1800743.pbs1` | cosine coord-jitter | `grad_conflict_cosine_coord_jitter_seed1_retry2` |

Early gradient-conflict readout:

| condition | samples | mean cosine | negative samples | min | max |
|---|---:|---:|---:|---:|---:|
| baseline | 10 | -0.0199 | 6 / 10 | -0.7618 | +0.2863 |
| cosine coord-jitter | 10 | -0.0558 | 3 / 10 | -0.9177 | +0.5467 |

Representative samples:

| condition | epoch | iter | grad cosine RGB/alpha |
|---|---:|---:|---:|
| baseline | 1 | 0 | -0.0320 |
| baseline | 1 | 20 | -0.7618 |
| baseline | 5 | 0 | -0.1475 |
| baseline | 5 | 20 | -0.0176 |
| cosine coord-jitter | 1 | 0 | +0.0058 |
| cosine coord-jitter | 1 | 20 | -0.9177 |
| cosine coord-jitter | 5 | 0 | +0.0044 |
| cosine coord-jitter | 5 | 20 | -0.1698 |

Reading:
- There is clear early negative RGB/alpha gradient alignment in the encoder
  stages. This supports monitoring the Surface-Maturation branch rather than
  jumping directly to PCGrad.
- The cosine monitor is also negative at iter 20, so the conflict is not
  trivially removed by the short 5-epoch cosine-ramp setting.
- The sign is not uniformly negative across epochs, so this is evidence for an
  intermittent conflict regime rather than a simple "always conflicting"
  objective.

Submitted Surface-Maturation / input-alpha jobs:

| pretrain job | dependent FCOS job | condition |
|---|---|---|
| `1800728.pbs1` | `1800729.pbs1` | `surface_maturation_tau0p3_k10_w0p05` |
| `1800730.pbs1` | `1800731.pbs1` | `surface_maturation_tau0p5_k20_w0p05` |
| `1800732.pbs1` | `1800733.pbs1` | `surface_maturation_tau0p7_k30_w0p05` |
| `1800734.pbs1` | `1800735.pbs1` | `input_alpha_curriculum` |

Current startup check:
- All four pretrain jobs reached dataset loading with 8 workers/logs.
- Surface-Maturation logs show the expected settings:
  - `tau=0.3`, `k=10`, `rgb_scale=1.0`
  - `tau=0.5`, `k=20`, `rgb_scale=1.0`
  - `tau=0.7`, `k=30`, `rgb_scale=1.0`
  - input-alpha curriculum starts with `rgb_scale=0.0`
- Dependent FCOS jobs are held with `afterok` and will run only if the matching
  pretrain finishes successfully.
- At the latest check, all four pretrains were around epoch 8. The observed
  wall-clock rate is roughly `2.3-2.5 min/epoch`, implying about `12-13 h` for
  e300 pretrain before the dependent FCOS jobs start.

## Experiment 35: Surface-Maturation E300 Scout Results

Snapshot:
- 2026-05-27 JST

Completed jobs:

| pretrain job | dependent FCOS job | condition | status |
|---|---|---|---|
| `1800728.pbs1` | `1800729.pbs1` | `surface_maturation_tau0p3_k10_w0p05` | complete |
| `1800730.pbs1` | `1800731.pbs1` | `surface_maturation_tau0p5_k20_w0p05` | complete |
| `1800732.pbs1` | `1800733.pbs1` | `surface_maturation_tau0p7_k30_w0p05` | complete |
| `1800734.pbs1` | `1800735.pbs1` | `input_alpha_curriculum` | complete |

Aggregation:
- Regenerated `results/shortcut_probe_artifacts/results_table.csv`.
- The table now has `108` rows and records the `SM_*` environment columns.

Surface-Maturation scout metrics:

| condition | AP@50 | AP@75 | AP75/AP50 | R50@300 | gate mean at e300 |
|---|---:|---:|---:|---:|---:|
| `surface_maturation_tau0p3_k10_w0p05` | 0.5246 | 0.0766 | 0.1460 | 0.6618 | 0.3185 |
| `surface_maturation_tau0p5_k20_w0p05` | 0.6079 | 0.0519 | 0.0854 | 0.7132 | 0.2586 |
| `surface_maturation_tau0p7_k30_w0p05` | 0.5973 | 0.0919 | 0.1539 | 0.7279 | 0.2524 |
| `input_alpha_curriculum` | 0.5176 | 0.0727 | 0.1404 | 0.7059 | 0.2749 |

Reference controls:

| condition | AP@50 | AP@75 | AP75/AP50 | R50@300 | note |
|---|---:|---:|---:|---:|---|
| `baseline_e300` ABCI clean, ft seed 1 | 0.4695 | 0.0869 | 0.1851 | 0.6618 | no coord-jitter |
| `cosine_e300` ABCI clean, ft seed 1 | 0.5539 | 0.1135 | 0.2049 | 0.7059 | no coord-jitter |
| `shuffle_e300` ABCI clean, ft seed 1 | 0.4162 | 0.0326 | 0.0783 | 0.5956 | no coord-jitter |
| `baseline_e300` ABCI clean, ft seeds 1-3 mean | 0.4938 | 0.0911 | - | 0.6618 | no coord-jitter |
| `cosine_e300` ABCI clean, ft seeds 1-3 mean | 0.5723 | 0.1067 | - | 0.7010 | no coord-jitter |
| `shuffle_e300` ABCI clean, ft seeds 1-3 mean | 0.4294 | 0.0453 | - | 0.6103 | no coord-jitter |
| `baseline_coord_jitter_e100` | 0.5564 | 0.1015 | 0.1824 | 0.6765 | coord-jitter |
| `cosine_coord_jitter_e100` | 0.6219 | 0.1031 | 0.1657 | 0.7279 | coord-jitter |
| `dmae_hier_concat_coord_jitter_e100` | 0.5212 | 0.0858 | 0.1646 | 0.6838 | coord-jitter |

Reading:
- `tau0p5_k20` gives the best AP@50 among the Surface-Maturation variants:
  `0.6079`. It is close to `cosine_coord_jitter_e100` AP@50 (`0.6219`,
  delta `-0.0140`) and above the clean `cosine_e300` finetune-seed-1 AP@50
  (`0.5539`, delta `+0.0541`).
- However, `tau0p5_k20` collapses AP@75 to `0.0519`. This is a coarse-detection
  gain, not a localization-quality gain.
- `tau0p7_k30` is the best balanced Surface-Maturation variant: AP@50 `0.5973`,
  AP@75 `0.0919`, R50@300 `0.7279`. It matches `cosine_coord_jitter_e100` on
  R50@300 but remains below it on AP@50 and AP@75.
- `tau0p3_k10` and `input_alpha_curriculum` are weak. The input-side alpha
  curriculum branch should be dropped unless a separate reason emerges.

Decision:
- Surface-Maturation is not a clear method win yet. It is better than D-MAE
  coord-jitter on AP@50 and competitive with cosine coord-jitter on coarse AP
  for `tau0p5`, but it does not improve AP@75 or AP75/AP50.
- Do not promote Surface-Maturation to the main method as-is.
- Keep `tau0p7_k30_w0p05` as the only Surface-Maturation variant worth a small
  follow-up if we want a more localization-preserving gate.
- If time is constrained, the paper path should still prioritize
  `cosine_coord_jitter` / cosine curriculum as the empirical best, with
  Surface-Maturation as a negative/partial method scout.

Possible follow-up only if useful:
- Run proposal-quality diagnostics for `tau0p5` and `tau0p7` to see whether
  the AP@75 drop comes from score ranking, center/size error, or proposal IoU.
- If trying one more Surface-Maturation setting, use a less aggressive gate:
  for example `tau=0.65, k=15, w_min=0.10`, based on the fact that AP@50
  survives at high tau but AP@75 remains below cosine.

## Experiment 36: Surface Tau0p7 Proposal Diagnostic and Gradient Conflict

Snapshot:
- 2026-05-27 JST

Decision update:
- Keep only `surface_maturation_tau0p7_k30_w0p05` as the Surface-Maturation
  diagnostic target.
- Drop `tau0p5_k20` from further diagnostics despite its high AP@50 because
  AP@75 is too weak (`0.0519`).

Generated artifacts:
- `results/shortcut_probe_artifacts/proposal_quality/surface_tau0p7_decision.json`
- `results/shortcut_probe_artifacts/proposal_quality/surface_tau0p7_decision.md`
- `results/shortcut_probe_artifacts/proposal_quality/surface_tau0p7_decision.png`

Implementation note:
- Extended `nerf_rpn/tools/summarize_proposal_quality.py` to report:
  proposal IoU histogram, score-IoU calibration, AP@50-TP failure at AP@75,
  and object-size AP@50/AP@75.
- `baseline_e1200` has `eval.json` only in this checkout. No proposal dump or
  FCOS checkpoint was found, so proposal-level diagnostics for `baseline_e1200`
  cannot be regenerated without restoring or rerunning that FCOS checkpoint.

Proposal diagnostic:

| condition | AP@50 | AP@75 | AP75/AP50 | R50@300 | mean IoU | frac IoU>=0.5 | TP50 fail75 | center err >=0.5 | size err >=0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_coord_jitter` | 0.5564 | 0.1015 | 0.1824 | 0.6765 | 0.0632 | 0.0180 | 0.6848 | 0.0474 | 0.1816 |
| `cosine_coord_jitter` | 0.6219 | 0.1031 | 0.1657 | 0.7279 | 0.0635 | 0.0196 | 0.7500 | 0.0498 | 0.1818 |
| `surface_tau0p7_k30` | 0.5973 | 0.0919 | 0.1539 | 0.7279 | 0.0609 | 0.0194 | 0.6869 | 0.0519 | 0.1775 |

Object-size diagnostic:

| condition | AP50 small | AP50 medium | AP50 large | AP75 small | AP75 medium | AP75 large |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_coord_jitter` | 0.2066 | 0.2033 | 0.2453 | 0.0609 | 0.0437 | 0.0323 |
| `cosine_coord_jitter` | 0.1846 | 0.2943 | 0.2639 | 0.0455 | 0.0795 | 0.0272 |
| `surface_tau0p7_k30` | 0.1833 | 0.2232 | 0.2656 | 0.0496 | 0.0173 | 0.0527 |

Reading:
- `surface_tau0p7_k30` matches `cosine_coord_jitter` on R50@300 (`0.7279`)
  and has nearly the same fraction of top300 proposals with IoU >= 0.5
  (`0.0194` vs `0.0196`).
- The AP gap is therefore not explained by missing coarse proposal coverage.
  It is more likely score ranking / calibration and localization quality.
- The clearest weakness is medium-object localization: `surface_tau0p7_k30`
  has AP75-medium `0.0173`, far below `cosine_coord_jitter` `0.0795`.
- Surface also has slightly worse center error at IoU>=0.5 (`0.0519` vs
  `0.0498`) but slightly better size error (`0.1775` vs `0.1818`), so the
  AP@75 drop is not a simple size-regression failure.

Gradient-conflict monitor:

| condition | n | mean cos | median cos | min | max | negative fraction |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 10 | -0.0199 | -0.0198 | -0.7618 | 0.2863 | 0.6000 |
| `cosine_coord_jitter` | 10 | -0.0558 | 0.0334 | -0.9177 | 0.5467 | 0.3000 |

Per-epoch means:

| condition | e1 | e2 | e3 | e4 | e5 |
|---|---:|---:|---:|---:|---:|
| `baseline` | -0.3969 | 0.1542 | 0.1111 | 0.1146 | -0.0826 |
| `cosine_coord_jitter` | -0.4559 | -0.0191 | 0.1610 | 0.1177 | -0.0827 |

Reading:
- The monitor does show strong early alpha/RGB gradient conflict in both
  settings, especially at epoch 1 iter 20.
- `cosine_coord_jitter` does not remove the initial conflict in this compressed
  5-epoch monitor, but it has a lower negative fraction overall (`0.30` vs
  `0.60`) and a positive median gradient cosine.
- This supports a cautious motivation: cosine/alpha curriculum appears to
  reduce or defer alpha/RGB conflict after the earliest phase, but this is not
  strong enough to claim that conflict mitigation is the only mechanism.

Decision:
- Use the gradient-conflict result as motivation/diagnostic support for
  cosine curriculum, not as a standalone proof.
- Keep `cosine_coord_jitter` as the empirical best path.
- Do not continue Surface-Maturation unless a very small targeted follow-up is
  needed for the localization-gate story.

## Experiment 37: Scene-Level Pyramid Alpha/RGB Target Curriculum Setup

Snapshot:
- 2026-05-27 JST

Motivation:
- The strongest current intervention is still scene-level `cosine_coord_jitter`.
- Local/channel interventions did not become main-method candidates:
  D-MAE trails `cosine_coord_jitter`, Surface-Maturation does not improve
  AP@75/localization, and input-side alpha curriculum is weak.
- Working hypothesis: preserving scene-level coherence while training from
  coarse structure to full RGBA fidelity may be more effective than local
  decoder/gate interventions.

Implemented MVP:
- Applied `pyramid_mvp_bundle.zip`.
- Added `nerf_mae/model/mae/pyramid_probe.py`.
- Added `nerf_mae/run_swin_pyramid_mae.py`.
- Added `nerf_mae/probe_scripts/abci3_pyramid_pretrain.pbs`.
- Added `nerf_mae/probe_scripts/submit_pyramid_sweep.sh`.
- Extended the shared ABCI3 pretrain/FCOS scripts with `KIND=pyramid`.
- Extended `results_table.csv` builder with `PYR_*` metadata fields.

Important implementation correction:
- The bundle originally used the pyramid alpha target to build the RGB occupied
  mask. That would make `P_A` change RGB supervision locations, weakening the
  intended orthogonal target-only test.
- The local implementation now builds RGB supervision masks from the original
  full-resolution alpha target and uses the pyramid target only as the
  reconstruction target. This keeps:
  - `pyramid_alpha`: alpha target low-res -> full-res, RGB target/mask full-res
  - `pyramid_rgb`: RGB target low-res -> full-res, alpha target full-res
  - `pyramid_both`: alpha and RGB targets low-res -> full-res

Pyramid scout conditions:

| condition | PYR_MODE | meaning |
|---|---|---|
| `pyramid_alpha` | `alpha` | alpha target pyramid only |
| `pyramid_rgb` | `rgb` | RGB target pyramid only |
| `pyramid_both` | `both` | alpha and RGB target pyramid |

Default protocol:
- Pretrain: e300, seed1, ABCI3 HF 1 node x 8 GPUs, global batch 16,
  deterministic off.
- Augmentation: coord-jitter on, matching `cosine_coord_jitter`
  (`rotate=1.0`, `flip=0.5`, `coord_shift=1.0`, `coord_shift_max=8`).
- Pyramid: `scale=2`, cosine transition over 300 epochs, alpha max-pooling,
  RGB average-pooling, alpha nearest upsampling, RGB trilinear upsampling.
- FCOS: dependent e1000, finetune seed1, same downstream protocol as the other
  scouts.

Validation before submit:
- `python -m py_compile` passed for:
  - `nerf_mae/model/mae/pyramid_probe.py`
  - `nerf_mae/run_swin_pyramid_mae.py`
  - `nerf_mae/tools/build_results_table.py`
- `bash -n` passed for:
  - `nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs`
  - `nerf_mae/probe_scripts/abci3_e300_gate_fcos.pbs`
  - `nerf_mae/probe_scripts/abci3_pyramid_pretrain.pbs`
  - `nerf_mae/probe_scripts/submit_pyramid_sweep.sh`
- ABCI3 preflight passed for pretrain/FCOS data and Python imports.
- `run_swin_pyramid_mae.py` import check passed in `.venv-abci3`.

Original Experiment 39 decision rule:
- Tier 1: AP@50 >= `cosine_coord_jitter` and AP@75 improves.
- Tier 2: AP@50 improves over `cosine_coord_jitter` by >= 0.02 with AP@75
  roughly unchanged.
- Tier 3: no AP@50/AP@75 improvement; keep `cosine_coord_jitter` as empirical
  best and use Pyramid as an ablation.

Submitted jobs:

| condition | PYR_MODE | pretrain job | dependent FCOS job | status at submit |
|---|---|---|---|---|
| `pyramid_alpha` | `alpha` | `1805486.pbs1` | `1805487.pbs1` | pretrain running, FCOS hold |
| `pyramid_rgb` | `rgb` | `1805488.pbs1` | `1805489.pbs1` | pretrain running, FCOS hold |
| `pyramid_both` | `both` | `1805490.pbs1` | `1805491.pbs1` | pretrain running, FCOS hold |

Submission log:
- `output/launcher/pyramid_sweep_20260527_220600/submitted.tsv`

Additional Track-0 submissions:

| purpose | condition | pretrain seed | finetune seed | pretrain job | FCOS job | status at submit |
|---|---|---:|---:|---|---|---|
| coord-jitter shuffle scout | `shuffle_coord_jitter` | 1 | 1 | `1805494.pbs1` | `1805495.pbs1` | pretrain running, FCOS hold |
| coord-jitter FT seed | `baseline_coord_jitter` | 1 | 2 | existing | `1805496.pbs1` | FCOS running |
| coord-jitter FT seed | `baseline_coord_jitter` | 1 | 3 | existing | `1805497.pbs1` | FCOS running |
| coord-jitter FT seed | `cosine_coord_jitter` | 1 | 2 | existing | `1805498.pbs1` | FCOS running |
| coord-jitter FT seed | `cosine_coord_jitter` | 1 | 3 | existing | `1805499.pbs1` | FCOS running |

Submission logs:
- `output/launcher/shuffle_coord_jitter_20260527_2210/submitted.tsv`
- `output/launcher/coord_jitter_finetune_seeds_20260527_2210/submitted.tsv`

## Experiment 38: Pyramid Scout Results and Coord-Jitter Seed Check

Snapshot:
- 2026-05-28 JST

Completed jobs:
- Pyramid scout pretrain/FCOS:
  - `pyramid_alpha`: `1805486.pbs1` -> `1805487.pbs1`
  - `pyramid_rgb`: `1805488.pbs1` -> `1805489.pbs1`
  - `pyramid_both`: `1805490.pbs1` -> `1805491.pbs1`
- Coord-jitter controls:
  - `shuffle_coord_jitter`: `1805494.pbs1` -> `1805495.pbs1`
  - `baseline_coord_jitter` finetune seeds 2/3: `1805496.pbs1`, `1805497.pbs1`
  - `cosine_coord_jitter` finetune seeds 2/3: `1805498.pbs1`, `1805499.pbs1`
- Proposal diagnostic:
  - `pyramid_decision`: `1807156.pbs1`

Aggregation:
- Regenerated `results/shortcut_probe_artifacts/results_table.csv`.
- Current table size: 116 rows.
- Proposal diagnostic artifacts:
  - `results/shortcut_probe_artifacts/proposal_quality/pyramid_decision.json`
  - `results/shortcut_probe_artifacts/proposal_quality/pyramid_decision.md`
  - `results/shortcut_probe_artifacts/proposal_quality/pyramid_decision.png`

Single-finetune-seed scout metrics:

| condition | AP@50 | AP@75 | AP75/AP50 | R50@300 |
|---|---:|---:|---:|---:|
| `baseline_coord_jitter` | 0.5564 | 0.1015 | 0.1824 | 0.6765 |
| `cosine_coord_jitter` | 0.6219 | 0.1031 | 0.1657 | 0.7279 |
| `shuffle_coord_jitter` | 0.4138 | 0.0574 | 0.1388 | 0.6103 |
| `pyramid_alpha` | 0.5694 | 0.0978 | 0.1718 | 0.7206 |
| `pyramid_rgb` | 0.5447 | 0.1163 | 0.2136 | 0.6985 |
| `pyramid_both` | 0.5677 | 0.0521 | 0.0918 | 0.6985 |
| `surface_tau0p7` | 0.5973 | 0.0919 | 0.1539 | 0.7279 |
| `dmae_hier_concat_coord_jitter` | 0.5212 | 0.0858 | 0.1646 | 0.6838 |

Coord-jitter 3-finetune-seed summary:

| condition | AP@50 mean | AP@50 std | AP@75 mean | AP@75 std | R50@300 mean | R50@300 std |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_coord_jitter` | 0.5454 | 0.0103 | 0.1073 | 0.0264 | 0.6912 | 0.0255 |
| `cosine_coord_jitter` | 0.5873 | 0.0395 | 0.0872 | 0.0256 | 0.7181 | 0.0236 |

Paired `cosine_coord_jitter - baseline_coord_jitter`:

| finetune seed | dAP@50 | dAP@75 | dR50@300 |
|---:|---:|---:|---:|
| 1 | +0.0655 | +0.0016 | +0.0515 |
| 2 | +0.0597 | -0.0784 | +0.0147 |
| 3 | +0.0006 | +0.0166 | +0.0147 |

Proposal diagnostic summary:

| condition | AP@50 | AP@75 | R50@300 | mean proposal IoU | frac IoU>=0.5 | TP50 fail75 | first TP rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_coord_jitter` | 0.5564 | 0.1015 | 0.6765 | 0.0632 | 0.0180 | 0.6848 | 1.2941 |
| `cosine_coord_jitter` | 0.6219 | 0.1031 | 0.7279 | 0.0635 | 0.0196 | 0.7500 | 1.0588 |
| `surface_tau0p7` | 0.5973 | 0.0919 | 0.7279 | 0.0609 | 0.0194 | 0.6869 | 1.2941 |
| `shuffle_coord_jitter` | 0.4138 | 0.0574 | 0.6103 | 0.0605 | 0.0163 | 0.7590 | 4.2941 |
| `pyramid_alpha` | 0.5694 | 0.0978 | 0.7206 | 0.0660 | 0.0192 | 0.7347 | 1.2941 |
| `pyramid_rgb` | 0.5447 | 0.1163 | 0.6985 | 0.0655 | 0.0186 | 0.6947 | 3.0000 |
| `pyramid_both` | 0.5677 | 0.0521 | 0.6985 | 0.0641 | 0.0186 | 0.7895 | 1.0588 |

Reading:
- Pyramid does not meet the pre-registered Tier 1 or Tier 2 rule.
  `pyramid_alpha` is the best Pyramid AP@50 scout, but it is below
  `cosine_coord_jitter` (`0.5694` vs `0.6219`) and also below the
  `cosine_coord_jitter` 3-finetune-seed mean (`0.5873`).
- `pyramid_rgb` has the best Pyramid AP@75 (`0.1163`) and AP75/AP50 ratio
  (`0.2136`), but its AP@50 is weak (`0.5447`) and first TP rank is worse.
  This is a localization-oriented ablation signal, not a main-method win.
- `pyramid_both` fails on AP@75 (`0.0521`) despite a reasonable AP@50. Joint
  low-res alpha/RGB target curriculum is not viable in this MVP.
- `shuffle_coord_jitter` is clearly weak (`AP@50=0.4138`, `R50@300=0.6103`,
  first TP rank `4.2941`), supporting that meaningful target-alpha structure
  matters under coord-jitter.
- `cosine_coord_jitter` remains the empirical AP@50/recall best path, but the
  3-finetune-seed result is not a clean localization claim: AP@50 improves over
  baseline on average (`+0.0419`), while AP@75 is lower on average due to seed 2.

Decision:
- Do not promote Pyramid P_A/P_R/P_AR to the main method.
- Do not spend 3-finetune-seed compute on Pyramid winner selection.
- Keep Pyramid as an ablation showing that simple scene-level target pyramid
  decomposition does not recover the `cosine_coord_jitter` gain.
- Keep `cosine_coord_jitter` as the current empirical best for AP@50/recall,
  with cautious wording: it improves sample-efficient coarse detection, but does
  not yet solve tight localization/AP@75.
- The next paper direction should not open another broad method sweep unless it
  directly targets the AP@75/localization gap with a very small, pre-specified
  diagnostic.

## Experiment 39: Mechanism Gate Before SECS-MAE

Snapshot:
- 2026-05-29 JST

Motivation:
- `coord_jitter` is empirically useful, but that does not yet prove that the
  model learned transform equivariance.
- Before implementing an explicit SECS/equivariance objective, first run a
  lighter existing-checkpoint feature equivariance probe.
- Also test one local-control loophole: whether Surface-Maturation becomes
  complementary when combined with the successful cosine + coord-jitter setup.

Implemented:
- Added `nerf_mae/tools/feature_equivariance_probe.py`.
  - Loads existing MAE checkpoints.
  - Applies two independent coord-jitter transforms to the same scene.
  - Extracts encoder feature maps without MAE masking.
  - Inverse-aligns each feature map to canonical scene coordinates.
  - Reports token cosine, normalized L2, and linear CKA by stage and by
    all/surface/empty regions.
- Added `nerf_mae/probe_scripts/abci3_feature_equivariance_probe.pbs`.
- Added `surface_maturation_cosine_coord_jitter_tau0p7_k30_w0p05` to the shared
  pretrain/FCOS scripts.
- Updated `build_results_table.py` so the new Surface+cosine condition is
  indexed with its `SM_*` metadata.

Validation:
- `python -m py_compile` passed for:
  - `nerf_mae/tools/feature_equivariance_probe.py`
  - `nerf_mae/tools/build_results_table.py`
- `bash -n` passed for:
  - `nerf_mae/probe_scripts/abci3_feature_equivariance_probe.pbs`
  - `nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs`
  - `nerf_mae/probe_scripts/abci3_e300_gate_fcos.pbs`

Feature equivariance probe defaults:
- No new pretraining.
- Split: `val_scenes`
- `max_scenes=8`, `num_pairs=2`, `seed=17`
- Coord-jitter transform: `rotate_prob=1.0`, `flip_prob=0.5`,
  `coord_shift_prob=1.0`, `coord_shift_max_voxels=8`
- Stages: `0,1,2,3`
- Checkpoints:
  - `baseline_e300`
  - `cosine_e300`
  - `baseline_coord_jitter_e100`
  - `cosine_coord_jitter_e100`
  - `shuffle_coord_jitter_e300`

Submitted jobs:

| purpose | job | dependency | expected output |
|---|---|---|---|
| feature equivariance probe | `1807419.pbs1` | none | `results/shortcut_probe_artifacts/feature_equivariance/coord_jitter_feature_equivariance.{json,md}` |
| Surface+cosine+jitter pretrain | `1807420.pbs1` | none | `output/nerf_mae/results/nerfmae_surface_maturation_cosine_coord_jitter_tau0p7_k30_w0p05_p1.0_e300_seed1_abci3smcos_cj_det0_1n8g/epoch_300.pt` |
| Surface+cosine+jitter FCOS | `1807421.pbs1` | `afterok:1807420.pbs1` | corresponding FCOS eval JSON |

Submission log:
- `output/launcher/mechanism_gate_20260529_014503/submitted.tsv`

Retry note:
- Initial feature equivariance job `1807419.pbs1` failed because raw scenes are
  not always cubic 160^3 before NeRF-MAE padding, while the first implementation
  extracted features before applying the model's padding path.
- Fixed the probe to apply the same `model.transform()` padding used by
  training before feature extraction.
- Resubmitted feature equivariance probe as `1807435.pbs1`.
- `1807435.pbs1` then hit a missing split feature file. Fixed the probe to
  mirror dataset loading behavior by skipping split scenes whose `.npz` feature
  file is absent, and resubmitted as `1807436.pbs1`.
- `1807436.pbs1` then hit the PyTorch 2.6 `weights_only=True` checkpoint
  default on old local checkpoints. The probe now explicitly uses
  `weights_only=False` for trusted local training checkpoints and was
  resubmitted as `1807437.pbs1`.

Feature equivariance result:
- `1807437.pbs1` completed.
- Artifacts:
  - `results/shortcut_probe_artifacts/feature_equivariance/coord_jitter_feature_equivariance.json`
  - `results/shortcut_probe_artifacts/feature_equivariance/coord_jitter_feature_equivariance.md`
- All checkpoint loads succeeded with `missing=0`, `unexpected=0`.

Key feature equivariance summary:

| label | stage | region | cosine | l2 | cka |
|---|---:|---|---:|---:|---:|
| `baseline_coord_jitter_e100` | 0 | all | 0.6704 | 0.7483 | 0.5296 |
| `cosine_coord_jitter_e100` | 0 | all | 0.7000 | 0.7152 | 0.5368 |
| `baseline_coord_jitter_e100` | 0 | surface | 0.6127 | 0.8305 | 0.4481 |
| `cosine_coord_jitter_e100` | 0 | surface | 0.6645 | 0.7628 | 0.4765 |
| `baseline_coord_jitter_e100` | 1 | surface | 0.6836 | 0.7407 | 0.4514 |
| `cosine_coord_jitter_e100` | 1 | surface | 0.7211 | 0.6837 | 0.5243 |
| `baseline_coord_jitter_e100` | 2 | surface | 0.7308 | 0.6664 | 0.4128 |
| `cosine_coord_jitter_e100` | 2 | surface | 0.7249 | 0.6778 | 0.4107 |
| `baseline_coord_jitter_e100` | 3 | surface | 0.6998 | 0.7393 | 0.5998 |
| `cosine_coord_jitter_e100` | 3 | surface | 0.7007 | 0.7467 | 0.6502 |

Reading:
- The feature probe is positive but not decisive for an equivariance mechanism.
- `cosine_coord_jitter_e100` improves transform-aligned feature similarity over
  `baseline_coord_jitter_e100` in early/mid surface features:
  - stage 0 surface cosine: `0.6645` vs `0.6127`
  - stage 1 surface cosine: `0.7211` vs `0.6836`
  - stage 1 surface CKA: `0.5243` vs `0.4514`
- The advantage does not persist cleanly through all later stages; stage 2
  surface is essentially tied/slightly worse, and stage 3 surface cosine is
  nearly identical.
- Therefore, `coord_jitter + cosine` likely improves some transform-aligned
  early representation dynamics, but this is not strong enough yet to justify
  jumping directly to a full SECS-MAE method.
- Treat explicit equivariance loss as a conditional next step, not the default
  next method. Wait for the Surface+cosine+jitter FCOS result before opening
  another heavy branch.

## Experiment 40: Feature Equivariance Sanity and Robustness Jobs

Snapshot:
- 2026-05-29 JST

Motivation:
- The feature equivariance probe is correlational and could still be an
  alignment/implementation artifact.
- Before any explicit equivariance regularizer scout, validate the probe itself:
  identity-transform sanity, same-random-transform sanity, stage resolution
  metadata, and larger scene/pair robustness.

Implementation update:
- Extended `feature_equivariance_probe.py` with `--pair-mode`:
  - `random`: two independently sampled coord-jitter transforms.
  - `identity`: `T1 == T2 == identity`.
  - `shared_random`: `T1 == T2 == sampled coord-jitter transform`.
- Added stage resolution metadata to the JSON/Markdown output:
  feature shape and input-to-feature stride for each stage.
- Added explicit scene count and transform-pair count to the report.
- Added `PAIR_MODE` support to `abci3_feature_equivariance_probe.pbs`.

Validation:
- `python -m py_compile nerf_mae/tools/feature_equivariance_probe.py`
- `bash -n nerf_mae/probe_scripts/abci3_feature_equivariance_probe.pbs`

Submitted jobs:

| purpose | job | dependency | output prefix |
|---|---|---|---|
| identity sanity | `1807592.pbs1` | none | `identity_feature_equivariance_sanity` |
| shared-random sanity | `1807593.pbs1` | none | `shared_random_feature_equivariance_sanity` |
| robustness probe | `1807594.pbs1` | `afterok:1807592.pbs1:1807593.pbs1` | `coord_jitter_feature_equivariance_robust` |

Sanity expectations:
- `identity` should give near-perfect feature cosine at every stage/region.
- `shared_random` should also be near-perfect; if not, transform/padding/path
  handling is suspect.
- Only if both sanity checks pass should the random-transform robustness result
  be used to motivate or reject an equivariance regularizer.

Surface+cosine status:
- `surface_maturation_cosine_coord_jitter_tau0p7_k30_w0p05` pretrain
  `1807420.pbs1` is still running.
- At 2026-05-29 early JST, it had produced `epoch_40.pt`; dependent FCOS
  `1807421.pbs1` remains on hold.
- Do not submit explicit equivariance regularizer scouts until this FCOS result
  is known, unless sanity/robustness results create a very strong reason.

## Experiment 41: Feature Equivariance Sanity and Robustness Results

Snapshot:
- 2026-05-29 JST

Completed jobs:
- `1807592.pbs1`: identity sanity
- `1807593.pbs1`: shared-random sanity
- `1807594.pbs1`: robustness probe

Artifacts:
- `results/shortcut_probe_artifacts/feature_equivariance/identity_feature_equivariance_sanity.{json,md}`
- `results/shortcut_probe_artifacts/feature_equivariance/shared_random_feature_equivariance_sanity.{json,md}`
- `results/shortcut_probe_artifacts/feature_equivariance/coord_jitter_feature_equivariance_robust.{json,md}`

Sanity results:
- `identity`: all checkpoints/stages/regions have cosine `1.0000`, L2 `0.0000`,
  and CKA `1.0000`.
- `shared_random`: all checkpoints/stages/regions also have cosine `1.0000`,
  L2 `0.0000`, and CKA `1.0000`.
- Stage resolution metadata is correct:

| stage | feature shape | stride |
|---:|---|---|
| 0 | `[40, 40, 40]` | `[4.0, 4.0, 4.0]` |
| 1 | `[20, 20, 20]` | `[8.0, 8.0, 8.0]` |
| 2 | `[10, 10, 10]` | `[16.0, 16.0, 16.0]` |
| 3 | `[5, 5, 5]` | `[32.0, 32.0, 32.0]` |

This passes the core implementation sanity checks: same-scene pairing,
model.eval path, model padding path, stage downsample factors, and same-transform
alignment are not obviously broken.

Robustness probe:
- `pair_mode=random`
- `scene_count=16`
- `num_pairs_per_scene=3`
- `transform_pair_count_per_checkpoint=48`

Key robust surface-region metrics:

| label | stage0 cos | stage0 CKA | stage1 cos | stage1 CKA | stage2 cos | stage3 cos |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_coord_jitter_e100` | 0.5378 | 0.3174 | 0.6038 | 0.3410 | 0.7010 | 0.6454 |
| `cosine_coord_jitter_e100` | 0.5682 | 0.3608 | 0.5759 | 0.3792 | 0.6966 | 0.6401 |
| `baseline_e300` | 0.5122 | 0.2998 | 0.6486 | 0.3526 | 0.7840 | 0.7295 |
| `cosine_e300` | 0.5922 | 0.4027 | 0.6443 | 0.4324 | 0.8380 | 0.8099 |
| `shuffle_coord_jitter_e300` | 0.5618 | 0.2414 | 0.6383 | 0.3791 | 0.9090 | 0.8168 |

Reading:
- The original weak-positive equivariance signal does not become a clean
  mechanism after increasing scene/pair count.
- `cosine_coord_jitter_e100` improves stage0 surface cosine/CKA over
  `baseline_coord_jitter_e100`, but stage1 surface cosine is lower and later
  stages are essentially tied or slightly worse.
- `cosine_e300` without coord-jitter is stronger than `cosine_coord_jitter_e100`
  on several stage/region metrics, so the probe is not isolating coord-jitter
  equivariance as the cause of transfer.
- `shuffle_coord_jitter_e300` has very high stage2/3 cosine despite weak
  downstream AP, which further argues that feature alignment alone is not a
  sufficient causal explanation.

Decision:
- The feature equivariance probe is now implementation-sane, but the mechanism
  evidence is too weak for SECS-MAE to be the next default method.
- Do not launch explicit equivariance regularizer scouts yet.
- Wait for `surface_maturation_cosine_coord_jitter_tau0p7_k30_w0p05` FCOS.
- If Surface+cosine+jitter does not beat `cosine_coord_jitter`, then an early
  surface equivariance regularizer may still be tried as the final method scout,
  but it should be framed as a causal test with strict no-go rules, not as a
  broad new sweep.

Decision rule:
- If `cosine_coord_jitter_e100` is clearly more transform-aligned than
  `baseline_coord_jitter_e100`, the SECS/equivariance branch becomes plausible.
- If `baseline_coord_jitter_e100` and `cosine_coord_jitter_e100` are similar,
  coord-jitter is more likely acting as augmentation/regularization, and SECS
  should not be the next main-method bet.
- If Surface+cosine+jitter beats `cosine_coord_jitter`, local interventions are
  not exhausted; otherwise, the local-control loophole is weaker.

## Experiment 42: Optimization-Trajectory Gate

Snapshot:
- 2026-05-29 JST
- Commit before this update: `e9aabce`

Motivation:
- The SECS/equivariance branch is now implementation-sane but not mechanistically
  clean enough to be the next default method.
- Current working hypothesis is an optimization-trajectory mechanism:
  early alpha/structure learning plus scene-level coord-jitter may guide the
  encoder toward transferable low-frequency structure before appearance/RGB
  fidelity dominates.
- The next gate is therefore not another local method variant, but a
  ramp-shape/order test:
  `cosine ~= linear ~= step > constant > reverse` would support
  structure-to-appearance order as the key factor; `cosine` alone would look
  more like a schedule-specific hyperparameter.

Implementation updates:
- `run_swin_mae3d.py` text logs now include `loss_rgb` and `loss_alpha` next to
  total loss. This is needed for later trajectory/gradient-conflict summaries.
- `abci3_e300_gate_pretrain.pbs` now supports trajectory scout conditions:
  - `cosine_ramp_coord_jitter`
  - `linear_ramp_coord_jitter`
  - `step_ramp_coord_jitter`
  - `reverse_ramp_coord_jitter`
  - `constant_mixed_coord_jitter`
- `abci3_e300_gate_fcos.pbs` recognizes the same curriculum conditions for
  dependent FCOS evaluation.
- Added `submit_ramp_shape_sweep.sh` for the order/schedule scout.
- `build_results_table.py` now records ramp protocol metadata:
  `PROBE_CURRICULUM`, `PROBE_CURRICULUM_EPOCHS`, RGB start/end weights,
  RGB/alpha weights, and `PROBE_ORDER`.
- `run_fcos_pretrained.py` and `test_fcos_pretrained.sh` now support explicit
  eval-time coord jitter (`coord_shift_prob`, `coord_shift_max_voxels`) while
  keeping normal eval default transforms at zero in `test_fcos_pretrained.sh`.
- Added `abci3_eval_time_jitter_robustness.pbs` for checkpoint-only robustness
  evaluation under rotate/flip/coord-shift transforms.

Validation:
- `bash -n nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs`
- `bash -n nerf_mae/probe_scripts/abci3_e300_gate_fcos.pbs`
- `bash -n nerf_mae/probe_scripts/submit_ramp_shape_sweep.sh`
- `bash -n nerf_rpn/test_fcos_pretrained.sh`
- `bash -n nerf_rpn/tools/abci3_eval_time_jitter_robustness.pbs`
- `python -m py_compile nerf_mae/run_swin_mae3d.py nerf_mae/tools/build_results_table.py nerf_rpn/run_fcos_pretrained.py`
- Dry-run submission of all five ramp-shape conditions succeeded.
- `build_results_table.py` generated `116` rows and populated new curriculum
  metadata for existing cosine rows.

Submitted jobs:

| purpose | condition | pretrain job | FCOS/eval job | notes |
|---|---|---:|---:|---|
| surface+cosine+jitter | `surface_maturation_cosine_coord_jitter_tau0p7_k30_w0p05` | `1807420.pbs1` | `1807421.pbs1` | pretrain complete; FCOS running |
| ramp-shape scout | `cosine_ramp_coord_jitter` | `1808709.pbs1` | `1808710.pbs1` | e100, 1n8g, det0 |
| ramp-shape scout | `linear_ramp_coord_jitter` | `1808711.pbs1` | `1808712.pbs1` | e100, 1n8g, det0 |
| ramp-shape scout | `step_ramp_coord_jitter` | `1808713.pbs1` | `1808714.pbs1` | e100, 1n8g, det0; step at half budget |
| ramp-shape scout | `reverse_ramp_coord_jitter` | `1808715.pbs1` | `1808716.pbs1` | e100, 1n8g, det0 |
| ramp-shape scout | `constant_mixed_coord_jitter` | `1808717.pbs1` | `1808718.pbs1` | e100, 1n8g, det0 |
| eval-time jitter robustness | existing checkpoints | n/a | `1808719.pbs1` | eval-only under rotate/flip/shift |

Why e100 first:
- The strongest coord-jitter evidence currently referenced by the feedback is
  `cosine_coord_jitter_e100` (`AP@50 ~= 0.6219`) versus
  `baseline_coord_jitter_e100` (`AP@50 ~= 0.5564`).
- Running the ramp-shape gate at e100 first is the fastest way to test whether
  the ordering effect exists. If the expected order pattern appears, promote
  the winning/critical rows to e300; if not, do not spend e300 compute on a
  weak trajectory-method branch.

Decision rule:
- Support trajectory/order hypothesis if alpha-to-RGBA variants
  (`cosine`, `linear`, `step`) are consistently above `constant_mixed` and
  `reverse`, with no large AP@75/recall collapse.
- If `reverse` or `constant_mixed` ties the alpha-to-RGBA variants, the order
  mechanism is weak and method exploration should stop or move to analysis-only.
- If order is supported, try at most one adaptive-ordering scout derived from
  the trajectory hypothesis. If that does not beat `cosine_coord_jitter`, switch
  to an analysis-heavy paper rather than adding more method knobs.

## Experiment 43: Optimization-Trajectory Gate Results

Snapshot:
- 2026-05-30 JST

Completed jobs:
- Surface+cosine+jitter FCOS: `1807421.pbs1`
- Ramp-shape scout pretrain/FCOS: `1808709-1808718.pbs1`
- Eval-time jitter robustness: `1808719.pbs1`

Artifacts:
- `results/shortcut_probe_artifacts/trajectory_gate_summary.csv`
- `results/shortcut_probe_artifacts/trajectory_gate_summary.md`
- `results/shortcut_probe_artifacts/eval_time_jitter_summary.csv`
- `results/shortcut_probe_artifacts/eval_time_jitter/*_jitter_eval/eval.json`

Ramp-shape / surface+cosine results:

| condition | epoch | AP@50 | AP@75 | R50@300 | AP@25 | reading |
|---|---:|---:|---:|---:|---:|---|
| `surface_maturation_cosine_coord_jitter_tau0p7_k30_w0p05` | 300 | 0.6397 | 0.0766 | 0.7647 | 0.8190 | AP50/recall strongest but localization weak |
| `linear_ramp_coord_jitter` | 100 | 0.5982 | 0.0942 | 0.7279 | 0.8198 | best ramp-shape AP50 and recall |
| `constant_mixed_coord_jitter` | 100 | 0.5728 | 0.1062 | 0.6985 | 0.8192 | constant control beats step/reverse/cosine on AP50 |
| `step_ramp_coord_jitter` | 100 | 0.5693 | 0.0756 | 0.7206 | 0.8079 | alpha-to-rgba order not clearly better than constant |
| `reverse_ramp_coord_jitter` | 100 | 0.5504 | 0.0885 | 0.6838 | 0.8201 | reverse is weak but not decisively separated from cosine |
| `cosine_ramp_coord_jitter` | 100 | 0.5400 | 0.1181 | 0.6985 | 0.8244 | best AP75 but weak AP50 |

Expected trajectory-supporting pattern:

```text
cosine ~= linear ~= step > constant > reverse
```

Observed AP@50 pattern:

```text
linear > constant > step > reverse > cosine
```

Decision:
- This e100 ramp-shape gate does not cleanly support the strong
  optimization-order hypothesis.
- In particular, `constant_mixed_coord_jitter` is too strong relative to step and
  cosine, so "alpha-to-RGBA order is the key mechanism" is not a safe claim.
- Do not launch adaptive-ordering as the default next method.
- `surface+cosine+jitter` is interesting because it gives the best AP@50/recall,
  but AP@75 remains weak, so it should be diagnosed as coarse-transfer/localization
  trade-off rather than promoted immediately as a method.

Eval-time jitter robustness:

| condition | normal AP@50 | jitter AP@50 | delta AP@50 | normal AP@75 | jitter AP@75 | delta AP@75 |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_e300` | 0.4695 | 0.5068 | +0.0373 | 0.0869 | 0.0813 | -0.0056 |
| `cosine_e300` | 0.5539 | 0.5479 | -0.0060 | 0.1135 | 0.0754 | -0.0381 |
| `baseline_coord_jitter_e100` | 0.5564 | 0.5489 | -0.0075 | 0.1015 | 0.0745 | -0.0270 |
| `cosine_coord_jitter_e100` | 0.6219 | 0.5621 | -0.0598 | 0.1031 | 0.0640 | -0.0391 |
| `shuffle_coord_jitter_e300` | 0.4138 | 0.4000 | -0.0138 | 0.0574 | 0.0474 | -0.0100 |

Reading:
- Eval-time jitter does not explain `cosine_coord_jitter_e100` as simple
  transform robustness. It drops more than `baseline_coord_jitter_e100`.
- Coord-jitter pretraining does not produce a clean AP@75 robustness signature
  under this eval perturbation.
- Together with Experiment 41, this weakens both the simple equivariance story
  and the simple eval-time robustness story.

Current paper-direction implication:
- The most defensible reading is now analysis-heavy:
  several interventions can improve coarse transfer, but neither simple
  feature equivariance nor simple alpha-to-RGBA order is a sufficient mechanism.
- If a method branch is continued, the narrowest candidate is
  `surface+cosine+jitter` as a coarse-transfer method with explicit localization
  diagnostics. New broad method searches should stop unless a reviewer-facing
  mechanism can be stated before running the job.

## Experiment 44: Paper-Code Parity and `paper_loss_e300` Kill Experiment

Snapshot:
- 2026-05-31 JST

Why this exists:
- The official released NeRF-MAE implementation appears to optimize an effective
  objective different from the nominal reading of the paper equation:
  RGB/radiance loss is applied over all occupied voxels, while alpha/opacity
  loss is applied over removed patch voxels.
- The missing decisive comparison is not another method variant. It is a
  p1.0/e300 counterfactual where RGB loss is restricted to removed occupied
  voxels.

What was already known:
- `public_code_loss` is the current baseline and has been measured many times
  (`baseline_e300`, `baseline_e1200`, etc.).
- Historical `masked_only_rgb_loss` was measured only in low-budget / older
  protocols, not as a p1.0/e300 clean kill experiment.

Implementation:
- Added `diagnostic:paper_loss` to the ABCI3 pretrain/FCOS scripts.
- `paper_loss` uses:
  - `PROBE_MODE=custom`
  - `PROBE_RGB_INPUT=keep`
  - `PROBE_ALPHA_INPUT=keep`
  - `PROBE_ALPHA_TARGET=keep`
  - `PROBE_RGB_LOSS=removed_occupied`
  - `PROBE_ALPHA_LOSS=removed`
  - `PROBE_RGB_WEIGHT=1.0`
  - `PROBE_ALPHA_WEIGHT=1.0`
- Extended `results_table.csv` metadata with:
  - `LOSS_FAMILY`
  - `RGB_LOSS_REGION`
  - `ALPHA_LOSS_REGION`
  - `RGB_LOSS_DENOM`
  - `ALPHA_LOSS_DENOM`
- Added `results/shortcut_probe_artifacts/paper_code_parity_report.md`.

Validation:
- `bash -n nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs`
- `bash -n nerf_mae/probe_scripts/abci3_e300_gate_fcos.pbs`
- `python -m py_compile nerf_mae/tools/build_results_table.py`
- `python nerf_mae/tools/build_results_table.py --root . --out-csv results/shortcut_probe_artifacts/results_table.csv`

Submitted kill experiment:

| condition | pretrain job | FCOS job | log dir |
|---|---:|---:|---|
| `paper_loss` | `1811826.pbs1` | `1811827.pbs1` | `output/launcher/paper_loss_e300_20260531_002753` |

Current status:
- `1811826.pbs1` is running on `rt_HF`.
- `1811827.pbs1` is held with `afterok:1811826.pbs1`.

Other-task availability check:
- Semantic/SR scripts exist in this repo.
- The currently linked Front3D finetuning data exposes `features/`, `obb/`, and
  `aabb/`, but no ready semantic voxel directory or high-resolution SR feature
  directory was found under the current `dataset/` tree.
- Therefore semantic/SR triage is not immediately runnable without preparing or
  restoring the corresponding data.

Decision rule:
- If `paper_loss_e300 ~= public_code_loss_e300`, the paper/code loss mismatch is
  not the main downstream driver and the objective-fidelity route should stop.
- If `public_code_loss_e300 >> paper_loss_e300`, decompose visible vs masked
  occupied RGB paths next.
- If `paper_loss_e300 > public_code_loss_e300`, the paper-like objective becomes
  a simple-fix route worth validating with finetune seeds.

## Experiment 45: Sample-Efficiency Framing and Runnable Next Steps

Snapshot:
- 2026-05-31 JST

Why this note exists:
- The previous "AP@50 is near the NeRF-MAE paper ceiling" wording was too
  pessimistic. Reaching a multi-source / long-budget AP@50 neighborhood with a
  single-source e300 run should instead be treated as a sample-efficiency
  anomaly, not as a reason to give up.
- However, "e300 gives AP@50 ~= 0.62" is not by itself a strong paper claim.
  The claim must specify what compute/data was reduced, what metric was
  preserved, and whether the signal generalizes beyond AP@50 detection.

Current strongest working claim:

```text
NeRF-MAE AP@50 transfer can become highly sample-efficient under target-side
surface/structure-biased training: a single-source e300 regime approaches the
reported multi-source e1200 AP@50 neighborhood. Whether this reflects a general
representation improvement or a detection/objectness-specific acceleration is
still unresolved.
```

What is already actionable from the feedback:
- Keep `paper_loss_e300` as the first kill experiment. Do not launch another
  broad method variant before this result is available.
- Treat the paper/code objective mismatch as an objective-fidelity analysis, not
  as "the paper is false".
- Use AP@25/AP@50/Recall as paper-protocol metrics. Keep AP@75 as a
  fine-localization diagnostic.
- Reframe `cosine_coord_jitter` around sample efficiency rather than asymptotic
  SOTA.
- Build a compute-normalized comparison table using scene count, epochs, and
  GPU time where available.
- Since public semantic/SR data is not available, replace immediate semantic/SR
  triage with feasible generality checks: ScanNet/cross-dataset detection if
  data is ready, and low-label Front3D detection.
- Run a targeted 1-2 day survey only after the immediate kill experiment is
  interpreted. The survey target is positioning, not discovering another knob.

What is not immediately actionable:
- Semantic voxel labeling and voxel SR are not runnable from the current public
  release or this workspace's linked data. The official README lists those
  finetuning datasets as "Coming Soon", and the local Front3D detection archive
  contains only `features/`, `obb/`, `aabb/`, and `3dfront_split.npz`.
- Therefore semantic/SR should not block the current paper-loss kill
  experiment. If private processed targets appear later, use
  `nerf_mae/probe_scripts/prepare_abci3_other_task_data.sh` to symlink and
  validate them.
- Geometry-derived targets such as alpha-distance / SDF / normal should not be
  implemented yet. They become justified only if `paper_loss_e300` and
  feasible generality checks show that AP@50 improves while fine/dense geometry
  remains weak.

Current data availability:

| data source | available in workspace | ready tasks |
|---|---:|---|
| `dataset/pretrain` | 3260 train / 20 val / 18 test scenes, 3260 feature files | NeRF-MAE pretrain |
| `dataset/finetune/front3d_rpn_data` | 122 train / 20 val / 17 test scenes, 159 feature files | Front3D detection |
| semantic voxel targets | not found | not runnable |
| voxel SR `features_384` targets | not found | not runnable |

Compute-normalized table skeleton:

| setting | pretrain data | epochs | approx scene-epochs | AP@50 | source/status |
|---|---|---:|---:|---:|---|
| NeRF-MAE F3D no aug | F3D | 1200 | TBD | 0.543 | paper-reported; verify table before citation |
| NeRF-MAE F3D aug | F3D | 1200 | TBD | 0.591 | paper-reported; verify table before citation |
| NeRF-MAE multi-source aug | F3D+HM3D+Hypersim | 1200 | TBD | 0.630 | paper-reported; verify table before citation |
| `baseline_e300` | current F3D/pretrain split | 300 | 978k using 3260 train scenes | 0.4695/0.4862/0.5258 across ft seeds | measured |
| `cosine_coord_jitter` | current F3D/pretrain split | 300 or e100-labelled historical row | TBD; protocol must be reconciled | ~=0.62 best row | measured, needs protocol cleanup |
| `paper_loss_e300` | current F3D/pretrain split | 300 | 978k using 3260 train scenes | pending | job `1811826` -> `1811827` |

Concrete artifact:
- The actual compute-normalized table has now been written to:
  - `results/shortcut_probe_artifacts/compute_normalized_sample_efficiency.md`
  - `results/shortcut_probe_artifacts/compute_normalized_sample_efficiency.csv`
- Important correction from that table: the current local pretraining split is
  mixed (`1839` Front3D-like, `1171` HM3D-like, `250` Hypersim-like train
  scenes), not strictly Front3D-only. Any "single-source e300" wording must be
  revised unless a true Front3D-only pretraining row is produced.

Important caution:
- The `cosine_coord_jitter ~=0.62` row must be protocol-audited before it is
  used as the headline comparison. Some logs label this evidence as e100 while
  the broader discussion calls it e300. Before writing, reconcile checkpoint
  epoch, pretrain condition, FCOS finetune seed, and result path in
  `results_table.csv`.

Immediate execution plan:
1. Wait for `paper_loss_e300` (`1811826.pbs1` -> `1811827.pbs1`).
2. After completion, compare against `public_code_loss_e300` on AP@25/AP@50,
   Recall@50, and AP@75 diagnostic.
3. If `paper_loss` differs materially, run at most the visible/masked occupied
   RGB path decomposition:
   - `visible_occupied_rgb_only`
   - `masked_occupied_rgb_only`
4. If `paper_loss` is similar to public code, stop the objective-fidelity
   branch and move to sample-efficiency/generalization checks.
5. For generalization without semantic/SR data, prioritize:
   - ScanNet detection triage if ready data/checkpoint loading exists.
   - Low-label Front3D detection using existing detection data.
6. In parallel, prepare the paper infrastructure:
   - finalize `paper_code_parity_report.md`
   - generate a compute-normalized sample-efficiency table
   - run targeted NeRF-MAE follow-up / neural-field SSL survey

Decision branches after `paper_loss_e300`:
- `paper_loss ~= public_code_loss`: paper/code mismatch is not the main driver;
  write it as an audit result and focus on sample efficiency/generalization.
- `public_code_loss >> paper_loss`: dense occupied RGB supervision may be the
  effective transfer path; decompose visible vs masked occupied RGB.
- `paper_loss > public_code_loss`: paper-like masked occupied RGB becomes a
  simple-fix method route worth validating with finetune seeds.
- Generality positive on cross-dataset/low-label: sample-efficient method paper
  remains viable.
- Generality negative: analysis-heavy route, centered on AP@50 detection
  acceleration versus representation/fine-localization fidelity.

## Experiment 46: ScanNet Detection Transfer Data Prep and Single-Seed Triage

Snapshot:
- 2026-05-31 01:24 JST

Correction:
- The previous "other-task data is not runnable" statement applies to semantic
  voxel labeling and voxel super-resolution targets in the public NeRF-MAE
  release, not to ScanNet OBB detection.
- ScanNet OBB detection is available through the public NeRF-RPN HuggingFace
  dataset. The RPN archive is sufficient for FCOS train/eval because it contains
  extracted `features/`, `obb/`, and `scannet_split.npz`.

Data prepared:
- Downloaded archive:
  `dataset/_downloads/archives/scannet_rpn_data.zip`
- Extracted nested archive and linked:
  `dataset/finetune/scannet_rpn_data -> dataset/_downloads/scannet_rpn_extract/scannet_rpn_data`
- Validated layout:
  - `features`: 90 `.npz`
  - `obb`: 90 `.npy`
  - split: 60 train / 15 val / 15 test scenes

Added helper scripts:
- `nerf_mae/probe_scripts/prepare_abci3_scannet_data.sh`
- `nerf_rpn/tools/abci3_scannet_transfer_fcos.pbs`

Single-seed ScanNet FCOS transfer jobs submitted:

| job | condition | MAE checkpoint |
|---|---|---|
| `1811879.pbs1` | `baseline_e300_scannet_fcos1000_seed1` | `output/nerf_mae/results/nerfmae_all_p1.0_e300_seed1/epoch_300.pt` |
| `1811887.pbs1` | `cosine_ramp_e300_scannet_fcos1000_seed1` | `output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1/epoch_300.pt` |
| `1811881.pbs1` | `cosine_coord_jitter_e100_scannet_fcos1000_seed1` | `output/nerf_mae/results/nerfmae_cosine_coord_jitter_p1.0_e100_seed1_abci3diag_opt1n8g_det0/epoch_100.pt` |
| `1811882.pbs1` | `surface_cosine_jitter_e300_scannet_fcos1000_seed1` | `output/nerf_mae/results/nerfmae_surface_maturation_cosine_coord_jitter_tau0p7_k30_w0p05_p1.0_e300_seed1_abci3smcos_cj_det0_1n8g/epoch_300.pt` |

Protocol notes:
- These are intentionally single-seed cross-dataset triage jobs. Do not expand
  to multi-seed until ScanNet shows a meaningful ranking difference.
- The jobs use `DATASET_NAME=scannet`, `SPLIT_NAME=scannet`, and the prepared
  ScanNet RPN data root.
- Batch size is set conservatively to 1 per GPU for first-pass ScanNet FCOS
  stability.
- Semantic voxel labeling and voxel SR remain separate; the public NeRF-MAE
  README still marks those data releases as unavailable/coming soon.
- First `cosine_ramp_e300` submission `1811880.pbs1` failed before training
  because concurrent jobs raced while building the `sort_vertices` CUDA op.
  The PBS script was updated to import `torch` before `sort_vertices` and to
  guard any future build with a lock; the job was resubmitted as
  `1811887.pbs1`.

Decision use:
- If the e300/curriculum variants transfer to ScanNet, the sample-efficiency
  story is no longer Front3D-only.
- If the ranking collapses on ScanNet, the current effect should be treated as
  Front3D/OBB-objectness-specific unless low-label Front3D provides a separate
  generality axis.

## Experiment 47: ScanNet Transfer Triage Results

Snapshot:
- 2026-05-31 JST

Artifacts:
- `results/shortcut_probe_artifacts/scannet_transfer_triage.csv`
- `results/shortcut_probe_artifacts/scannet_transfer_triage.md`

Protocol:
- Public NeRF-RPN ScanNet OBB detection archive.
- 60 train / 15 val / 15 test scenes.
- FCOS transfer for 1000 epochs, single finetune seed.

Results:

| condition | AP@25 | AP@50 | AP@75 | R@50 top300 | R@50 top1000 |
|---|---:|---:|---:|---:|---:|
| `baseline_e300` | 0.5013 | 0.1898 | 0.0024 | 0.3596 | 0.3596 |
| `cosine_ramp_e300` | 0.5883 | 0.1912 | 0.0006 | 0.4039 | 0.4187 |
| `cosine_coord_jitter_e100` | 0.5540 | 0.1864 | 0.0022 | 0.3695 | 0.3744 |
| `surface_cosine_jitter_e300` | 0.5759 | 0.1782 | 0.0014 | 0.3596 | 0.3645 |

Interpretation:
- `cosine_ramp_e300` is the best AP@50 row, but the gain over
  `baseline_e300` is only `+0.0014`. Treat this as a single-seed tie, not as
  evidence of robust cross-dataset AP@50 improvement.
- `cosine_ramp_e300` does improve AP@25 and R@50, so the strongest ScanNet
  signal is coarse proposal/objectness transfer, not tight AP@50/AP@75
  localization.
- `cosine_coord_jitter_e100` and `surface_cosine_jitter_e300` do not improve
  ScanNet AP@50 over the baseline row.
- Current conclusion: the Front3D AP@50 gain does not cleanly transfer to
  ScanNet AP@50 in this first single-seed triage. The sample-efficiency story
  now needs either `paper_loss_e300` support, low-label support, or a clearer
  coarse-recall/AP@25 framing.

Decision:
- Do not expand these ScanNet rows to multi-seed yet.
- Wait for `paper_loss_e300`. If objective-fidelity shows a material effect,
  rerun the relevant objective on ScanNet rather than spending seeds on older
  variants.

## Experiment 48: Front3D Low-Label 50% Gate Submission

Snapshot:
- 2026-05-31 JST

Purpose:
- Test whether the current sample-efficiency signal also improves label
  efficiency on Front3D, following the feedback that low-label detection is the
  next most important evidence axis after full-label Front3D and ScanNet.

Protocol:
- Dataset: Front3D OBB detection, public/local `front3d_rpn_data`.
- Split: `3dfront_split.npz`, 122 train / 20 val / 17 test scenes.
- Label fraction: `PERCENT_TRAIN=0.5`.
- FCOS transfer: 1000 epochs, single finetune seed `1`.
- Queue: `rt_HG`, one GPU per job.
- This is a gate. Do not expand to all label fractions or multi-seed until the
  50% result shows a meaningful ranking difference.

Added helper:
- `nerf_rpn/tools/abci3_front3d_low_label_fcos.pbs`

Submitted jobs:

| job | condition | checkpoint |
|---|---|---|
| `1812644.pbs1` | `front3d_scratch_lowlabel_pt05_seed1_fcos1000` | scratch backbone |
| `1812645.pbs1` | `baseline_e300_lowlabel_pt05_seed1_fcos1000` | `output/nerf_mae/results/nerfmae_all_p1.0_e300_seed1/epoch_300.pt` |
| `1812646.pbs1` | `cosine_ramp_e300_lowlabel_pt05_seed1_fcos1000` | `output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1/epoch_300.pt` |
| `1812647.pbs1` | `surface_cosine_jitter_e300_lowlabel_pt05_seed1_fcos1000` | `output/nerf_mae/results/nerfmae_surface_maturation_cosine_coord_jitter_tau0p7_k30_w0p05_p1.0_e300_seed1_abci3smcos_cj_det0_1n8g/epoch_300.pt` |

Decision use:
- If `cosine_ramp_e300` or `surface_cosine_jitter_e300` clearly beats
  `baseline_e300` at 50% labels, expand low-label to 25%/50% and finetune
  seeds for the likely paper rows.
- If all pretrained rows are close and only scratch is lower, the paper can
  still claim generic pretraining label efficiency, but not a method-specific
  low-label gain.
- If scratch is competitive, the sample-efficiency story needs to retreat to
  full-label compute efficiency and objective/mechanism analysis.

## Experiment 49: Alpha Boundary / SDF Target-Quality Audit

Snapshot:
- 2026-05-31 JST

Purpose:
- Check whether alpha-derived boundary/SDF/normal targets are visually and
  topologically usable before launching any Boundary-SDF MAE pretraining.
- This follows the feedback that SDF/normal target quality should be audited
  before adding another method branch.

Added helper:
- `nerf_mae/probe_scripts/audit_alpha_boundary_targets.py`

Protocol:
- Dataset: Front3D OBB features from
  `dataset/finetune/front3d_rpn_data/features`.
- Split: first 20 train scenes from
  `dataset/finetune/front3d_rpn_data/3dfront_split.npz`.
- Density-to-alpha conversion: same FCOS-loader style conversion,
  `alpha = 1 - exp(-exp(density) / 100)`.
- Thresholds: `0.01`, `0.05`, `0.1`.
- Per scene/threshold outputs: alpha slice, thresholded occupancy, shell,
  distance-to-shell, alpha-gradient magnitude.

Artifacts:
- `results/shortcut_probe_artifacts/alpha_boundary_audit_front3d_train20/README.md`
- `results/shortcut_probe_artifacts/alpha_boundary_audit_front3d_train20/alpha_boundary_metrics.csv`
- 60 visualization PNGs under
  `results/shortcut_probe_artifacts/alpha_boundary_audit_front3d_train20/`

Summary:

| threshold | scenes | occ ratio mean | shell/occ mean | components median | dist p90 mean | grad p95 mean |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 20 | 0.314632 | 0.6869 | 440.5 | 13.575 | 0.353083 |
| 0.05 | 20 | 0.283159 | 0.7476 | 502.0 | 13.865 | 0.353083 |
| 0.1 | 20 | 0.220480 | 0.8227 | 736.5 | 14.427 | 0.353083 |

Qualitative read:
- Visualization is feasible and useful: the alpha slices expose coherent room
  surfaces, but the hard-thresholded occupancy/shell maps are noisy and
  threshold-sensitive.
- `thr=0.01` is the least fragmented of the tested thresholds, but even there
  the median connected-component count is high and shell/occupied ratio is
  about `0.69`.
- Higher thresholds (`0.05`, `0.1`) increase fragmentation and make most
  occupied voxels shell-like. For example, `3dfront_1013_00` at `thr=0.1`
  has `3054` components and shell/occupied ratio `0.923`.

Decision:
- Do not launch raw thresholded Boundary-SDF MAE yet.
- If the geometry-target route is reopened, first try one of:
  1. low threshold (`0.01`) plus largest-component / small-component filtering,
  2. smoothed alpha before boundary extraction,
  3. shell/gradient target as an auxiliary diagnostic instead of a hard SDF
     objective.
- Keep this as target-quality evidence for the geometry-derived target branch.

## Experiment 50: Compute-Normalized Sample-Efficiency Main Table

Snapshot:
- 2026-05-31 JST

Purpose:
- Turn the compute-efficiency discussion into a paper-facing main table with
  scene count, epochs, scene-epochs, approximate ABCI3/H200 GPU-days, AP@25,
  AP@50, and Recall@50.

Artifacts:
- `results/shortcut_probe_artifacts/compute_normalized_sample_efficiency.md`
- `results/shortcut_probe_artifacts/compute_normalized_sample_efficiency.csv`

Compute convention:
- Paper multi-source pretraining scene count:
  `1998 Front3D + 1330 HM3D + 250 Hypersim = 3578`.
- Paper multi-source e1200 scene-epochs:
  `3578 * 1200 = 4,293,600`.
- Local pretraining split: `3260` train scenes.
- ABCI3/H200 estimate uses measured local 1-node/8-H200 pretraining speed:
  e300 walltime is about `11.63h`, so e300 costs
  `8 * 11.63 / 24 = 3.88` H200 GPU-days. e100 costs `1.29` H200 GPU-days.

Main rows:

| setting | scenes | epochs | scene-epochs | rel. vs paper multi | approx H200 GPU-days | AP@25 | AP@50 | R@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NeRF-MAE (F3D), aug | 1998 | 1200 | 2,397,600 | 0.558 | 9.52 | 0.830 | 0.591 | 0.743 |
| NeRF-MAE (Ours), aug | 3578 | 1200 | 4,293,600 | 1.000 | 17.05 | 0.853 | 0.630 | 0.745 |
| `baseline_e300` | 3260 | 300 | 978,000 | 0.228 | 3.88 | 0.799 | 0.494 | 0.662 |
| `cosine_ramp_e300` | 3260 | 300 | 978,000 | 0.228 | 3.88 | 0.829 | 0.572 | 0.701 |
| `baseline_e1200` | 3260 | 1200 | 3,912,000 | 0.911 | 15.52 | 0.849 | 0.589 | 0.713 |
| `cosine_coord_jitter_e100` | 3260 | 100 | 326,000 | 0.076 | 1.29 | 0.799 | 0.587 | 0.718 |
| `surface+cosine+jitter_e300` | 3260 | 300 | 978,000 | 0.228 | 3.88 | 0.819 | 0.640 | 0.765 |
| `paper_loss_e300` | 3260 | 300 | 978,000 | 0.228 | 3.88 | pending | pending | pending |

Paper-facing read:
- e300 local rows are `22.8%` of the paper multi-source scene-epochs, i.e.
  about `4.39x` less compute under the ABCI/H200-normalized estimate.
- The `~1/12` phrasing should be reserved for `cosine_coord_jitter_e100`,
  which is `7.6%` of paper multi-source scene-epochs, i.e. `13.17x` less
  compute, with mean AP@50 `0.587` and Recall@50 `0.718`.
- The best AP@50 row (`surface+cosine+jitter_e300`, `0.640`) is still
  single-seed and has weak AP@75, so it should be treated as a compute
  efficiency candidate rather than a settled final main-table method row.
- Current local rows are mixed-source. Do not call them "single-source" unless
  a true Front3D-only pretraining row is isolated.

## Experiment 51: `paper_loss_e300` Slow-Run Correction

Snapshot:
- 2026-05-31 JST

Issue:
- Original `paper_loss_e300` pretrain job `1811826.pbs1` was submitted with
  `DETERMINISTIC=1`.
- It was not a 1200-epoch run. PBS variables showed `EPOCHS=300`.
- The large `Time Use` value in `qstat` was cumulative CPU time over the
  allocated CPU cores, not walltime. `qstat -f` showed walltime around `20h`
  when inspected.
- Live worker log showed progress only to about epoch `93/300`, with about
  `13.1 min/epoch`.
- Recent optimized 1-node/8-H200 det0 e300 runs take about `2.32 min/epoch`.
  The slow run was therefore about `5.6x` slower, consistent with
  deterministic CUDA/cuDNN behavior on the 3D convolution/transpose-convolution
  heavy MAE decoder.

Action:
- Stopped the misconfigured jobs:
  - old pretrain: `1811826.pbs1`
  - old dependent FCOS: `1811827.pbs1`
- Resubmitted the diagnostic using the current ABCI-optimized scout protocol:
  `DETERMINISTIC=0`, 1 HF node, 8 H200 GPUs, global batch 16.

Replacement jobs:

| condition | pretrain | dependent FCOS | log dir |
|---|---|---|---|
| `paper_loss_det0` | `1812733.pbs1` | `1812734.pbs1` | `output/launcher/paper_loss_e300_det0_20260531_203813` |

Expected runtime:
- Pretrain should be close to previous det0 e300 runs, about `11.6-12h`.
- Dependent FCOS will start after `1812733.pbs1` completes.

## Experiment 52: `paper_loss_e300` and 50% Low-Label Gate Results

Snapshot:
- 2026-06-01 JST

Artifacts:
- `results/shortcut_probe_artifacts/paper_loss_lowlabel_gate_20260601.md`
- `results/shortcut_probe_artifacts/paper_loss_lowlabel_gate_20260601.csv`
- Updated `results/shortcut_probe_artifacts/compute_normalized_sample_efficiency.md`
- Updated `results/shortcut_probe_artifacts/compute_normalized_sample_efficiency.csv`

Completed jobs:

| condition | pretrain job | FCOS/eval job | status |
|---|---:|---:|---|
| `paper_loss_e300` det0 rerun | `1812733.pbs1` | `1812734.pbs1` | complete |
| `scratch_lowlabel_50` | n/a | `1812644.pbs1` | complete |
| `baseline_e300_lowlabel_50` | existing ckpt | `1812645.pbs1` | complete |
| `cosine_ramp_e300_lowlabel_50` | existing ckpt | `1812646.pbs1` | complete |
| `surface_cosine_jitter_e300_lowlabel_50` | existing ckpt | `1812647.pbs1` | complete |

Results:

| condition | protocol | AP@25 | AP@50 | AP@75 | Recall@25 top300 | Recall@50 top300 |
|---|---|---:|---:|---:|---:|---:|
| `paper_loss_e300` | full-label | 0.7949 | 0.5613 | 0.0742 | 0.9632 | 0.6912 |
| `scratch_lowlabel_50` | 50% labels | 0.7065 | 0.3666 | 0.0513 | 0.9338 | 0.5956 |
| `baseline_e300_lowlabel_50` | 50% labels | 0.7671 | 0.4191 | 0.0241 | 0.9559 | 0.6471 |
| `cosine_ramp_e300_lowlabel_50` | 50% labels | 0.7690 | 0.5026 | 0.0516 | 0.9412 | 0.6691 |
| `surface_cosine_jitter_e300_lowlabel_50` | 50% labels | 0.7811 | 0.5217 | 0.0627 | 0.9559 | 0.6765 |

Read:
- `paper_loss_e300` is viable, not collapsed. It is above the ABCI-clean
  `baseline_e300` mean AP@50 (`0.4938`) and close to `cosine_ramp_e300` mean
  AP@50 (`0.5723`), but it is not better than the strongest surface/curriculum
  rows.
- Therefore the cleanest interpretation is not
  `public_code_loss >> paper_loss` and not `paper_loss > public_code_loss`.
  The paper/code loss mismatch is real, but this single run does not make it
  the dominant transfer mechanism.
- The 50% low-label gate supports the sample-efficiency story. Relative to
  low-label scratch, `surface_cosine_jitter_e300` gives AP@50 `+0.1551`; relative
  to `baseline_e300`, it gives `+0.1026`; relative to `cosine_ramp_e300`, it gives
  `+0.0191`.
- These low-label rows remain single finetune seed and should be treated as a
  gate rather than final statistics.

Decision:
- Do not spend the next jobs on a broad visible-only / masked-only RGB objective
  decomposition unless a paper reviewer-facing reason is sharpened further.
- If compute is spent on validation, prioritize low-label expansion for the
  compact set of rows: scratch/NeRF-RPN anchor, `baseline_e300`,
  `cosine_ramp_e300`, and `surface_cosine_jitter_e300`.
- Keep `paper_loss_e300` in the compute-normalized table as a negative/neutral
  objective-fidelity gate: it rules out a collapse but does not produce a simple
  objective-fix story.

## Experiment 53: Low-Label 25%/10%/100% Expansion Submitted

Snapshot:
- 2026-06-01 JST

Artifacts:
- `results/shortcut_probe_artifacts/lowlabel_expansion_jobs_20260601.md`
- `results/shortcut_probe_artifacts/lowlabel_expansion_jobs_20260601.csv`

Purpose:
- Follow up the positive 50% low-label gate with the compact paper-facing row
  set at 25%, 10%, and 100% labels.
- Avoid launching new method variants. This is validation for the
  compute+label-efficiency paper direction.

Protocol:
- Dataset: Front3D OBB detection.
- FCOS: 1000 epochs, finetune seed `1`.
- PBS: `nerf_rpn/tools/abci3_front3d_low_label_fcos.pbs`.
- Existing 50% rows are complete and were not resubmitted.

Submitted jobs:

| job | labels | condition |
|---|---:|---|
| `1815787.pbs1` | 25% | `scratch` |
| `1815788.pbs1` | 25% | `baseline_e300` |
| `1815789.pbs1` | 25% | `cosine_ramp_e300` |
| `1815790.pbs1` | 25% | `surface_cosine_jitter_e300` |
| `1815791.pbs1` | 10% | `scratch` |
| `1815792.pbs1` | 10% | `baseline_e300` |
| `1815793.pbs1` | 10% | `cosine_ramp_e300` |
| `1815794.pbs1` | 10% | `surface_cosine_jitter_e300` |
| `1815795.pbs1` | 100% | `scratch` |
| `1815796.pbs1` | 100% | `baseline_e300` |
| `1815797.pbs1` | 100% | `cosine_ramp_e300` |
| `1815798.pbs1` | 100% | `surface_cosine_jitter_e300` |

Variant decision rule:
- If `surface_cosine_jitter_e300` is consistently at or above
  `cosine_ramp_e300` on full-label, 50%, and 25% labels, and does not introduce
  a severe ScanNet regression, promote `surface_cosine_jitter_e300` to the main
  method.
- If it is mainly stronger in in-domain low-label but weaker on ScanNet or
  full-label, keep `cosine_ramp_e300` as the main method and use surface
  anchoring as a low-label/in-domain ablation component.
- If 25%/10% collapse or reverse, narrow the label-efficiency claim to the
  moderate-label regime supported by the data, currently 50%.

Checkpoint note:
- Key existing e300/e1200 pretraining runs do not have intermediate
  `epoch_100.pt` / `epoch_200.pt` checkpoints on disk:
  - `baseline_e300`: `epoch_300.pt`
  - `cosine_ramp_e300`: `epoch_300.pt`
  - `surface_cosine_jitter_e300`: `epoch_300.pt`, `model_best.pt`
  - `baseline_e1200`: `epoch_1200.pt`
- For future paper runs requiring epoch curves, use the final checkpoint policy
  recorded in Experiment 54.

## Experiment 54: Final-Run Checkpoint Retention Policy

Snapshot:
- 2026-06-01 JST

Reason:
- The key existing e300/e1200 runs only retain their final checkpoint because
  `run_swin_mae3d.py` defaults to `--keep_checkpoints 1`; setting only
  `CHECKPOINT_INTERVAL` is insufficient because older `epoch_*.pt` files are
  deleted.
- For final paper runs, we want retained checkpoints every 50 epochs for
  learning-curve and epoch-budget tables.

Code changes:
- `nerf_mae/train_mae3d.sh` now accepts `KEEP_CHECKPOINTS` and passes it to
  `run_swin_mae3d.py --keep_checkpoints`.
- ABCI3 final/gate pretrain path now defaults to:
  - `PRETRAIN_CHECKPOINT_INTERVAL=50`
  - `PRETRAIN_KEEP_CHECKPOINTS=0`
- Surface/pyramid/ramp wrapper submit paths that may be reused for final
  variants now forward the same retention controls.

Expected behavior for future final pretraining:
- e300 final runs should retain:
  `epoch_50.pt`, `epoch_100.pt`, `epoch_150.pt`, `epoch_200.pt`,
  `epoch_250.pt`, `epoch_300.pt`, plus `model_best.pt` if evaluation runs.
- e1200 final runs should retain the same 50-epoch cadence through
  `epoch_1200.pt`.

Important:
- Existing runs cannot recover deleted intermediate checkpoints. If the paper
  needs exact e100/e200 points for `baseline_e300`, `cosine_ramp_e300`, or
  `surface_cosine_jitter_e300`, rerun the final selected protocol with this
  retention policy or run separate e100/e200 jobs.

## Experiment 55: Alpha Boundary / SDF Target-Quality Audit v2

Snapshot:
- 2026-06-01 JST

Purpose:
- Follow up Experiment 49, where raw thresholded alpha was too fragmented for
  Boundary-SDF MAE.
- Test whether simple denoising produces a launchable geometry target before
  spending GPU time on Boundary-SDF pretraining.

Added helper:
- `nerf_mae/probe_scripts/audit_alpha_boundary_targets_v2.py`

Protocol:
- Dataset: Front3D OBB features from
  `dataset/finetune/front3d_rpn_data/features`.
- Split: first 20 train scenes for visual audit, first 60 train scenes for
  no-render robustness audit.
- Density-to-alpha conversion: FCOS-loader style conversion,
  `alpha = 1 - exp(-exp(density) / 100)`.
- Variants:
  - raw `alpha > 0.01`
  - Gaussian-smoothed alpha with `sigma=0.75` or `1.0`
  - thresholds `0.01` and `0.02`
  - optional small-component filtering and binary closing.

Artifacts:
- `results/shortcut_probe_artifacts/alpha_boundary_audit_v2_front3d_train20/`
- `results/shortcut_probe_artifacts/alpha_boundary_audit_v2_front3d_train60/`
- `results/shortcut_probe_artifacts/alpha_boundary_audit_v2_decision.md`

60-scene summary:

| variant | scenes | occ ratio mean | shell/occ mean | components median | raw IoU mean | raw recall mean | sdf inside p90 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `raw_thr001` | 60 | 0.308927 | 0.6611 | 562.5 | 1.0000 | 1.0000 | 3.411 |
| `smooth075_thr001` | 60 | 0.409878 | 0.3535 | 151.5 | 0.7530 | 0.9980 | 6.781 |
| `smooth100_thr001` | 60 | 0.434797 | 0.3000 | 56.5 | 0.7092 | 0.9968 | 7.828 |
| `smooth100_thr002` | 60 | 0.397898 | 0.3361 | 48.0 | 0.7607 | 0.9869 | 7.015 |
| `smooth100_thr001_close1_min64` | 60 | 0.409860 | 0.2678 | 29.0 | 0.6315 | 0.8968 | 6.330 |

Read:
- Raw alpha remains unsuitable for Boundary-SDF: high fragmentation and
  shell-heavy occupancy.
- Gaussian smoothing makes the target substantially more plausible.
  `smooth100_thr001` reduces median components from `562.5` to `56.5` and
  shell/occupied from `0.6611` to `0.3000`, while retaining almost all raw
  support.
- `smooth100_thr002` is the better default candidate because it is more
  conservative: lower occupancy, higher raw IoU (`0.7607`), still-low component
  count (`48.0`), and high raw recall (`0.9869`).
- Morphological closing/filtering is too aggressive for the default target:
  it improves component/shell metrics but drops raw-support recall to `0.8968`
  and visually risks over-filling scene interiors.

Decision:
- Do not use raw Boundary-SDF targets.
- If the low-label 25%/10% gate is only moderate and a real method mechanism is
  needed for strong-accept positioning, the launchable Boundary-SDF scout should
  use:
  - `alpha_smoothing_sigma=1.0`
  - `alpha_threshold=0.02`
  - signed distance to the smoothed-alpha occupancy boundary
  - distance clip `16` voxels.
- Keep `smooth100_thr001` as a higher-recall / more-inflated ablation.

## Experiment 56: Boundary-SDF B1 e100 Scout Submission

Snapshot:
- 2026-06-01 JST

Purpose:
- Add one low-cost Boundary-SDF scout in parallel with the low-label gate.
- This follows the route-decision feedback: if low-label 25%/10% is only
  moderate, a real boundary-aware method mechanism may be needed for a
  strong-accept framing.

Code changes:
- Added `nerf_mae/model/mae/boundary_sdf_probe.py`.
- Added `nerf_mae/run_swin_boundary_sdf.py`.
- Extended ABCI3 pretrain/FCOS scripts for
  `KIND=boundary_sdf, CONDITION=boundary_sdf_aux`.

Design:
- Keep the public/effective NeRF-MAE RGB/alpha objective.
- Add a separate decoder-side `sdf_out` head. The existing 4-channel `out`
  head is not resized, so downstream FCOS loading should only see `sdf_out.*`
  as unexpected keys rather than shape mismatches.
- Use audit-v2 target settings:
  - `BOUNDARY_ALPHA_SMOOTH_SIGMA=1.0`
  - `BOUNDARY_ALPHA_THRESHOLD=0.02`
  - `BOUNDARY_DISTANCE_CLIP=16`
  - `BOUNDARY_SDF_WEIGHT=0.2`
  - `BOUNDARY_SDF_MASK=removed`
- The scout uses a GPU-friendly max-pool signed-distance approximation to the
  smoothed-alpha boundary. If this scout is promising, a cached exact-distance
  implementation can be considered for final runs.

Submitted jobs:

| job | role | status |
|---|---|---|
| `1816233.pbs1` | pretrain `boundary_sdf_aux`, e100 | running |
| `1816234.pbs1` | dependent Front3D FCOS e1000 | hold afterok |

Artifacts:
- `results/shortcut_probe_artifacts/boundary_sdf_b1_jobs_20260601.md`
- `results/shortcut_probe_artifacts/boundary_sdf_b1_jobs_20260601.csv`

Initial runtime check:
- The pretrain job started successfully and completed early iterations.
- Example worker-0 log:
  - `epoch 1 [0/204] loss=1.1396 loss_rgb=0.9020 loss_alpha=0.1840`
  - `epoch 1 [180/204] loss=0.2389 loss_rgb=0.1546 loss_alpha=0.0670`
- Since total loss is larger than `loss_rgb + loss_alpha`, the SDF auxiliary
  term is active.

Decision rule:
- Do not promote this to e300 unless e100 is stable and its FCOS result shows
  either AP@50/R50 improvement over the relevant e100 baseline, or comparable
  AP@50 with a clear AP@75/proposal-quality gain.

## Experiment 57: Low-Label Expansion and Boundary-SDF B1 Results

Snapshot:
- 2026-06-02 JST

Artifacts:
- `results/shortcut_probe_artifacts/lowlabel_boundary_sdf_results_20260602.md`
- `results/shortcut_probe_artifacts/lowlabel_boundary_sdf_results_20260602.csv`

Status:
- All low-label expansion jobs completed.
- Boundary-SDF B1 e100 pretrain and dependent FCOS completed.

Low-label results:

| condition | labels | AP@25 | AP@50 | AP@75 | R@50 top300 | R@50 top1000 |
|---|---:|---:|---:|---:|---:|---:|
| scratch | 10% | 0.4453 | 0.1160 | 0.0000 | 0.2794 | 0.3088 |
| baseline_e300 | 10% | 0.5344 | 0.1328 | 0.0001 | 0.3824 | 0.4265 |
| cosine_ramp_e300 | 10% | 0.5996 | 0.2751 | 0.0152 | 0.4485 | 0.4706 |
| surface_cosine_jitter_e300 | 10% | 0.5918 | 0.1756 | 0.0066 | 0.4118 | 0.4412 |
| scratch | 25% | 0.6087 | 0.3044 | 0.0122 | 0.4779 | 0.4779 |
| baseline_e300 | 25% | 0.6550 | 0.2777 | 0.0057 | 0.5147 | 0.5221 |
| cosine_ramp_e300 | 25% | 0.7008 | 0.3639 | 0.0095 | 0.5882 | 0.6029 |
| surface_cosine_jitter_e300 | 25% | 0.7123 | 0.3460 | 0.0200 | 0.5368 | 0.5441 |
| scratch | 50% | 0.7065 | 0.3666 | 0.0513 | 0.5956 | 0.5956 |
| baseline_e300 | 50% | 0.7671 | 0.4191 | 0.0241 | 0.6471 | 0.6544 |
| cosine_ramp_e300 | 50% | 0.7690 | 0.5026 | 0.0516 | 0.6691 | 0.6765 |
| surface_cosine_jitter_e300 | 50% | 0.7811 | 0.5217 | 0.0627 | 0.6765 | 0.6838 |
| scratch | 100% | 0.7952 | 0.4722 | 0.0703 | 0.6176 | 0.6324 |
| baseline_e300 | 100% | 0.7956 | 0.4695 | 0.0869 | 0.6618 | 0.6691 |
| cosine_ramp_e300 | 100% | 0.8249 | 0.5539 | 0.1135 | 0.7059 | 0.7059 |
| surface_cosine_jitter_e300 | 100% | 0.8178 | 0.5984 | 0.1004 | 0.7059 | 0.7279 |

Read:
- Low-label support is positive and stronger than a 50%-only artifact.
- At 10% labels, `cosine_ramp_e300` is the clear AP@50 winner:
  `+0.1591` over scratch and `+0.1423` over baseline_e300.
- At 25% labels, `cosine_ramp_e300` remains best by AP@50:
  `+0.0595` over scratch and `+0.0862` over baseline_e300.
- At 50% and 100%, `surface_cosine_jitter_e300` is best by AP@50.
- Paper framing should use a hierarchy rather than two equal methods:
  `cosine_ramp` as the base label-efficient recipe, with surface/jitter as an
  in-domain or label-richer anchoring component.

Boundary-SDF B1 results:

| condition | epoch | AP@25 | AP@50 | AP@75 | R@50 top300 | R@50 top1000 |
|---|---:|---:|---:|---:|---:|---:|
| boundary_sdf_aux | 100 | 0.8110 | 0.5142 | 0.1031 | 0.6618 | 0.6691 |
| baseline_coord_jitter | 100 | 0.8197 | 0.5564 | 0.1015 | 0.6765 | 0.6912 |
| cosine_coord_jitter | 100 | 0.8097 | 0.6219 | 0.1031 | 0.7279 | 0.7279 |

Decision:
- Boundary-SDF B1 does not clear the e100 promotion gate.
- It is lower than `baseline_coord_jitter_e100` on AP@50 and recall, and far
  below `cosine_coord_jitter_e100` on AP@50.
- Do not promote Boundary-SDF to e300 now.
- Keep it as an audited but non-winning branch unless the paper strategy later
  requires a boundary mechanism with a cleaner exact-distance target.

## Experiment 58: Official Low-Label Table Comparison

Snapshot:
- 2026-06-02 JST

Artifacts:
- `results/shortcut_probe_artifacts/official_lowlabel_comparison_20260602.md`
- `results/shortcut_probe_artifacts/official_lowlabel_comparison_20260602.csv`

Purpose:
- Address the feedback request to compare the current low-label rows directly
  against the official NeRF-MAE low-label table before launching any new method
  branch.
- Priority 3/4 items from the feedback, paper skeleton and visible-token
  feasibility, are intentionally left for the user-side workflow.
- Official values were checked against the ECCV paper PDF on 2026-06-02:
  `https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12306.pdf`.

Official AP@50 values used:

| labels | official scratch | official NeRF-MAE |
|---:|---:|---:|
| 10% | 0.15 | 0.18 |
| 25% | 0.29 | 0.36 |
| 50% | 0.30 | 0.42 |
| 100% | 0.41 | 0.54 |

Direct AP@50 comparison:

| labels | ours scratch | ours baseline_e300 | ours cosine_ramp_e300 | ours surface_cosine_jitter_e300 | ours best |
|---:|---:|---:|---:|---:|---:|
| 10% | 0.1160 | 0.1328 | 0.2751 | 0.1756 | 0.2751 |
| 25% | 0.3044 | 0.2777 | 0.3639 | 0.3460 | 0.3639 |
| 50% | 0.3666 | 0.4191 | 0.5026 | 0.5217 | 0.5217 |
| 100% | 0.4722 | 0.4695 | 0.5539 | 0.5984 | 0.5984 |

Read:
- `cosine_ramp_e300` exceeds the official NeRF-MAE AP@50 at all four label
  fractions:
  - 10%: `0.2751` vs `0.18`
  - 25%: `0.3639` vs `0.36`
  - 50%: `0.5026` vs `0.42`
  - 100%: `0.5539` vs `0.54`
- The best current variant exceeds the official NeRF-MAE table by `+0.0951`,
  `+0.0039`, `+0.1017`, and `+0.0584` AP@50 at 10/25/50/100% labels.
- The important scratch100 check is positive:
  - `surface_cosine_jitter_e300` at 50% labels beats our scratch 100% by
    `+0.0495` AP@50.
  - `cosine_ramp_e300` at 50% labels beats our scratch 100% by `+0.0304`
    AP@50.

Caveat:
- Our same-run scratch rows are not identical to the official reported scratch
  rows; for example, our 100% scratch AP@50 is `0.4722` versus official `0.41`.
  Therefore the official comparison is supportive but should not be the only
  proof. The main proof should be same-run proposed-vs-scratch/baseline.

Main-variant decision:
- Use `cosine_ramp_e300` as the base method because it is strongest at 10% and
  25% labels and already exceeds official NeRF-MAE AP@50 at every label
  fraction.
- Use `surface_cosine_jitter_e300` as an in-domain / label-richer enhancement,
  not as a separate equal method, because it is strongest at 50% and 100% but
  weaker at 10% and 25%.
- Do not launch a new broad method branch solely because of method-novelty
  anxiety before the current low-label result section is written.

## Experiment 59: MixNeRF-MAE-lite MVP Integration

Snapshot:
- 2026-06-03 JST

Source bundle:
- `/groups/gag51404/ide/vgi/NeRF-MAE/mixnerf_mae_mvp.zip`

Purpose:
- Add a minimal encoder-side MixNeRF-MAE scout to test whether NeRF-MAE's
  full-grid Swin encoder is helped or hurt by explicit masked-placeholders.
- Unlike the previous loss/target weighting variants, this branch replaces
  masked target-scene patches with plausible donor-scene patches while keeping
  the dense 3D Swin topology and computing loss against the original target
  scene.

Implemented files:
- `nerf_mae/model/mae/mixnerf_probe.py`
- `nerf_mae/run_swin_mixnerf_mae.py`
- `nerf_mae/probe_scripts/abci3_mixnerf_pretrain.pbs`
- `nerf_mae/probe_scripts/submit_mixnerf_smoke.sh`
- `nerf_mae/probe_scripts/submit_mixnerf_controls.sh`
- `nerf_mae/probe_scripts/encoder_mask_path_report.py`
- `nerf_mae/probe_scripts/mask_predictability_probe.py`
- `nerf_mae/probe_scripts/mix_input_sanity.py`

Gate integration:
- `abci3_e300_gate_pretrain.pbs` now supports:
  - `KIND=mixnerf,CONDITION=mixnerf_lite`
  - `KIND=mixnerf,CONDITION=mixnerf_lite_zeros`
  - `KIND=mixnerf,CONDITION=mixnerf_lite_noise`
- `abci3_e300_gate_fcos.pbs` resolves matching checkpoint names for downstream
  FCOS evaluation.
- The bundled submit helpers were adjusted to run from the repository root,
  matching the existing ABCI gate-script `PBS_O_WORKDIR` assumption.

Validation:
- `py_compile` passed for the MixNeRF model, runner, and probe scripts.
- `bash -n` passed for the MixNeRF PBS/submit scripts and the updated e300 gate
  pretrain/FCOS scripts.
- ABCI virtualenv import passed for `SwinTransformer_MAE3D_MixNeRF`.
- Lightweight forward smoke passed on a 64^3 synthetic pair:
  - `patch_mask_mean=0.75`
  - `voxel_mask_mean=0.75`
  - `partner_mean_self_match=0.0`
  - `base_mask_mean=0.0`
- Real-scene mixed-input sanity passed after adding patch-aligned cropping for
  unequal scene extents:
  - original shapes: `(4, 106, 100, 160)` and `(4, 160, 94, 130)`
  - cropped shape: `(4, 104, 92, 128)`
  - `mask_mean=0.75`

Artifacts:
- `results/shortcut_probe_artifacts/mixnerf/encoder_mask_path_report.md`
- `results/shortcut_probe_artifacts/mixnerf/mask_predictability_protocol.md`

First launch plan:
- Partner-fill smoke/scout:
  - e10 pretrain sanity
  - e30 pretrain + FCOS
  - e100 pretrain + FCOS
- Controls after the partner path is stable:
  - e30 partner / zeros / noise

Submitted jobs:
- e10 partner pretrain: `1819871.pbs1`
- e30 partner pretrain: `1819872.pbs1`
- e100 partner pretrain: `1819873.pbs1`
- e30 partner FCOS, dependent on `1819872.pbs1`: `1819874.pbs1`
- e100 partner FCOS, dependent on `1819873.pbs1`: `1819875.pbs1`
- e30 zero-fill control pretrain, dependent on e10 partner pretrain
  `1819871.pbs1`: `1819878.pbs1`
- e30 noise-fill control pretrain, dependent on e10 partner pretrain
  `1819871.pbs1`: `1819879.pbs1`
- e30 zero-fill control FCOS, dependent on `1819878.pbs1`: `1819880.pbs1`
- e30 noise-fill control FCOS, dependent on `1819879.pbs1`: `1819881.pbs1`

Decision gate:
- Continue only if e10/e30 losses are stable, `base_mask_mean` remains near
  zero, partner-fill is better than zero/noise controls, and e100 is competitive
  with the matching baseline/cosine e100 rows.

## Experiment 60: Budget Curve and Reversal Seed Defense

Snapshot:
- 2026-06-03 JST

Artifact:
- `results/shortcut_probe_artifacts/budget_curve_reversal_jobs_20260603.md`

Purpose:
- Follow the final feedback direction for the AAAI paper: use a budget curve as
  the primary efficiency evidence and defend the thin low-label reversal with
  a small finetune-seed expansion.
- Prioritize the budget curve as the current AAAI critical path while keeping
  visible-token / MixNeRF style masking mechanisms conditional, not permanently
  separated. If MixNeRF is clearly stronger, or if the budget curve is not
  strong enough by itself, revisit whether the masking-mechanism branch should
  be folded into the paper.

Budget-curve launch:
- Corrected after noticing that both the `cosine_ramp` schedule and the one-cycle
  learning-rate schedule depend on total `EPOCHS`.
- The main budget curve should therefore use dedicated total-budget runs, not
  intermediate checkpoints from an e1200 run for e100/e300/e600.
- The e1200 pretrains remain valid for the e1200 point. Incorrect dependent
  FCOS jobs for e1200-intermediate e100/e300/e600 points were cancelled.

Submitted budget-curve jobs:

| row | job | status |
|---|---:|---|
| baseline e1200 pretrain | `1821253.pbs1` | keep for e1200 only |
| cosine_ramp e1200 pretrain | `1821254.pbs1` | keep for e1200 only |
| baseline e100 pretrain | `1821358.pbs1` | dedicated budget run |
| baseline e100 FCOS | `1821359.pbs1` | dependent on e100 |
| baseline e600 pretrain | `1821360.pbs1` | dedicated budget run |
| baseline e600 FCOS | `1821361.pbs1` | dependent on e600 |
| baseline e1200 FCOS | `1821258.pbs1` | dependent on e1200 |
| cosine_ramp e100 pretrain | `1821362.pbs1` | dedicated budget run |
| cosine_ramp e100 FCOS | `1821363.pbs1` | dependent on e100 |
| cosine_ramp e600 FCOS | `1821364.pbs1` | FCOS-only on existing dedicated e600 |
| cosine_ramp e1200 FCOS | `1821262.pbs1` | dependent on e1200 |

Cancelled:
- `1821255.pbs1`, `1821256.pbs1`, `1821257.pbs1`
- `1821259.pbs1`, `1821260.pbs1`, `1821261.pbs1`

Submitted reversal-defense jobs:

| row | seed | job |
|---|---:|---:|
| scratch 100% | 2 | `1821264.pbs1` |
| cosine_ramp e300 50% | 2 | `1821265.pbs1` |
| surface_cosine_jitter e300 50% | 2 | `1821266.pbs1` |
| scratch 100% | 3 | `1821267.pbs1` |
| cosine_ramp e300 50% | 3 | `1821268.pbs1` |
| surface_cosine_jitter e300 50% | 3 | `1821269.pbs1` |

Decision:
- Figure 1 should be the AP@50/R@50 budget curve if the curve shows early
  structure-first saturation.
- The 50%-label reversal remains headline-worthy only if seed2/3 preserve the
  mean margin against scratch100.

## Experiment 61: MixNeRF-MAE-lite Scout Results

Snapshot:
- 2026-06-04 JST

Artifact:
- `results/shortcut_probe_artifacts/mixnerf/mixnerf_results_20260604.md`

Completed FCOS evals:

| condition | pretrain | AP@25 | AP@50 | AP@75 | R@50 top300 |
|---|---:|---:|---:|---:|---:|
| MixNeRF partner-fill | e30 | 0.8486 | 0.5433 | 0.1112 | 0.6765 |
| MixNeRF partner-fill | e100 | 0.8125 | 0.5871 | 0.0670 | 0.7206 |
| zero-fill control | e30 | 0.8271 | 0.5292 | 0.1675 | 0.6544 |
| noise-fill control | e30 | 0.8259 | 0.5894 | 0.1000 | 0.7206 |

Interpretation:
- MixNeRF partner-fill improves AP@50 from e30 to e100, but AP@75 collapses at
  e100.
- The e30 noise-fill control reaches AP@50 `0.5894`, slightly above MixNeRF e100
  `0.5871`, with the same R@50 top300.
- This does not isolate a useful partner-token mechanism. MixNeRF / visible-token
  masking should remain conditional rather than replacing the current budget-curve
  and structure-first paper path.

## Experiment 62: MixNeRF Masked-Loss and Dithered-Filler Follow-up

Snapshot:
- 2026-06-04 JST

Artifact:
- `results/shortcut_probe_artifacts/mixnerf/mixnerf_next_scout_jobs_20260604.md`

Motivation:
- The first MixNeRF result showed that e30 noise-fill AP@50 (`0.5894`) matched or
  exceeded e100 partner-fill AP@50 (`0.5871`).
- The previous MixNeRF runs used `PROBE_MODE=baseline`, so they tested filler
  corruption under the public occupied-all RGB objective rather than a pure
  removed/masked RGB objective.
- Existing logs confirmed the internal base mask was disabled
  (`base_mask_mean=0.0`, `internal_mask_attrs_overridden=['masking_prob']`).

Implementation updates:
- Added same-scene patch-shuffle filler via `MIXNERF_FILL_MODE=shuffle`.
- Added masked-loss MixNeRF conditions for partner / zero / noise / shuffle.
- Added MixNeRF probe-loss env overrides so gate runs can use
  `PROBE_MODE=custom`, `PROBE_RGB_LOSS=removed_occupied`,
  `PROBE_ALPHA_LOSS=removed`.
- Validation passed:
  - `py_compile` for `mixnerf_probe.py` and `run_swin_mixnerf_mae.py`
  - `bash -n` for updated MixNeRF/gate scripts
  - `submit_mixnerf_next_scouts.sh` dry-run

Submitted jobs:

| condition | epochs | fill | probe mode | RGB loss | pretrain | FCOS |
|---|---:|---|---|---|---:|---:|
| `mixnerf_lite_masked` | 30 | partner | custom | removed_occupied | `1826351.pbs1` | `1826352.pbs1` |
| `mixnerf_lite_zeros_masked` | 30 | zeros | custom | removed_occupied | `1826353.pbs1` | `1826354.pbs1` |
| `mixnerf_lite_noise_masked` | 30 | noise | custom | removed_occupied | `1826355.pbs1` | `1826356.pbs1` |
| `mixnerf_lite_shuffle_masked` | 30 | same-scene shuffle | custom | removed_occupied | `1826357.pbs1` | `1826358.pbs1` |
| `mixnerf_lite_noise` | 100 | noise | baseline | occupied | `1826359.pbs1` | `1826360.pbs1` |
| `mixnerf_lite_zeros` | 100 | zeros | baseline | occupied | `1826361.pbs1` | `1826362.pbs1` |

Decision rule:
- `partner > noise/zero/shuffle` under masked loss: MixNeRF partner semantics is
  revived.
- `noise` or `shuffle` wins: pivot method hypothesis toward Dithered /
  mask-token-free NeRF-MAE.
- no competitive result: stop the encoder-fill branch.

## Experiment 63: Budget Curve, Reversal Seeds, and MixNeRF Follow-up Results

Snapshot:
- 2026-06-06 JST

Artifacts:
- `results/shortcut_probe_artifacts/budget_curve_reversal_results_20260606.md`
- `results/shortcut_probe_artifacts/mixnerf/mixnerf_followup_results_20260606.md`

Queue status:
- Budget-curve, reversal-defense, and MixNeRF follow-up jobs are complete.
- Remaining running jobs are unrelated SSL jobs (`simclrv1`, `simclrv2`,
  `byol`, `ibot`).

Budget curve:

| method | epochs | AP@25 | AP@50 | AP@75 | R@50 top300 |
|---|---:|---:|---:|---:|---:|
| baseline | 100 | 0.7940 | 0.5422 | 0.0830 | 0.6912 |
| baseline | 300 | 0.7956 | 0.4695 | 0.0869 | 0.6618 |
| baseline | 600 | 0.7994 | 0.4994 | 0.0767 | 0.6765 |
| baseline | 1200 | 0.7934 | 0.5648 | 0.0809 | 0.7059 |
| cosine_ramp | 100 | 0.8095 | 0.5711 | 0.0940 | 0.7132 |
| cosine_ramp | 300 | 0.8249 | 0.5539 | 0.1135 | 0.7059 |
| cosine_ramp | 600 | 0.8220 | 0.6196 | 0.0721 | 0.7279 |
| cosine_ramp | 1200 | 0.8338 | 0.5490 | 0.0640 | 0.6838 |

Budget-curve interpretation:
- The dedicated-budget curve is not monotonic.
- `cosine_ramp` is stronger than baseline at e100/e300/e600, with the strongest
  AP@50 at e600 (`0.6196`).
- The e1200 `cosine_ramp` point drops below baseline e1200 on AP@50
  (`0.5490` vs `0.5648`), so the paper should not claim monotonic budget
  saturation or long-budget dominance.
- The defensible read is a mid-budget sample-efficiency peak.

Low-label 50% reversal, 3 finetune seeds:

| condition | mean AP@50 | std AP@50 | n |
|---|---:|---:|---:|
| scratch 100% | 0.4828 | 0.0468 | 3 |
| cosine_ramp 50% | 0.4970 | 0.0062 | 3 |
| surface_cosine_jitter 50% | 0.5150 | 0.0201 | 3 |

Reversal interpretation:
- The reversal is moderate rather than decisive.
- `cosine_ramp 50%` has a thin AP@50 mean gain over scratch 100% (`+0.0142`).
- `surface_cosine_jitter 50%` is stronger (`+0.0322`) and lower variance than
  scratch, but this should be framed as matching/modestly exceeding full-label
  scratch rather than as an overwhelming reversal.

Low-label seed1 grid:
- `cosine_ramp` is best at 10% and 25% labels.
- `surface_cosine_jitter` is best at 50% and 100% labels.
- This supports using `cosine_ramp` as the safer base method and surface
  anchoring as an in-domain / label-richer enhancement unless later cross-axis
  results favor surface anchoring consistently.

MixNeRF follow-up:

| condition | objective | epochs | AP@25 | AP@50 | AP@75 | R@50 top300 |
|---|---|---:|---:|---:|---:|---:|
| MixNeRF partner-fill | masked RGB | 30 | 0.8408 | 0.5567 | 0.1361 | 0.6765 |
| zero-fill control | masked RGB | 30 | 0.8127 | 0.4881 | 0.0499 | 0.6765 |
| noise-fill control | masked RGB | 30 | 0.8136 | 0.5276 | 0.1808 | 0.6765 |
| same-scene shuffle-fill | masked RGB | 30 | 0.8337 | 0.5805 | 0.1239 | 0.6838 |
| noise-fill control | public occupied RGB | 100 | 0.8197 | 0.4909 | 0.0642 | 0.6544 |
| zero-fill control | public occupied RGB | 100 | 0.8398 | 0.5459 | 0.0772 | 0.6838 |

MixNeRF interpretation:
- Under masked RGB, same-scene shuffle beats partner-fill on AP@50
  (`0.5805` vs `0.5567`).
- Cross-scene partner semantics is therefore not isolated.
- The previous public-objective e30 noise-fill result does not scale to e100.
- Keep MixNeRF / visible-token filling separated from the main paper path unless
  a more targeted future mechanism beats the budget-curve results.

## Experiment 64: Visible-only Dithered MixNeRF e100 Scout Launch

Snapshot:
- 2026-06-06 JST

Artifacts:
- `results/shortcut_probe_artifacts/mixnerf/implementation_audit_20260606.md`
- `results/shortcut_probe_artifacts/mixnerf/mixnerf_dither_e100_jobs_20260606.md`

Reason:
- The previous same-scene `shuffle` control used all patches, so it was not a
  clean mask-token-free / visible-only dither test.
- e30 rankings are not reliable enough for a method branch; previous noise-fill
  looked good at e30 but collapsed at e100.
- Need a simple non-zero control before treating same-scene dither as
  distribution matching rather than merely "anything non-zero is better than
  zero".

Implementation updates:
- Added `MIXNERF_FILL_MODE=shuffle_visible`, which samples replacement patches
  only from same-scene visible patches (`patch_mask == 0`).
- Added `MIXNERF_FILL_MODE=mean` and `constant` for simple non-zero controls.
- Added PBS condition support for:
  - `mixnerf_lite_shuffle_visible_masked`
  - `mixnerf_lite_mean_masked`
- Added `submit_mixnerf_dither_e100_scouts.sh`.

Validation:
- `py_compile` passed for `mixnerf_probe.py` and `run_swin_mixnerf_mae.py`.
- `bash -n` passed for the dither submitter and gate PBS scripts.
- Dry-run confirmed distinct `PRETRAIN_MASTER_PORT` values for concurrent
  pretrains.
- Local tensor sanity logged:
  - `same_scene_fill_source=visible_only`
  - `self_replacement_rate=0.0`
  - `masked_source_rate=0.0`

Submitted jobs:

| condition | epochs | seed | fill | objective | pretrain | FCOS |
|---|---:|---:|---|---|---:|---:|
| `mixnerf_lite_shuffle_visible_masked` | 100 | 1 | `shuffle_visible` | `removed_occupied` RGB / removed alpha | `1830790.pbs1` | `1830791.pbs1` |
| `mixnerf_lite_zeros_masked` | 100 | 1 | `zeros` | `removed_occupied` RGB / removed alpha | `1830792.pbs1` | `1830793.pbs1` |
| `mixnerf_lite_shuffle_visible_masked` | 100 | 2 | `shuffle_visible` | `removed_occupied` RGB / removed alpha | `1830794.pbs1` | `1830795.pbs1` |
| `mixnerf_lite_zeros_masked` | 100 | 2 | `zeros` | `removed_occupied` RGB / removed alpha | `1830796.pbs1` | `1830797.pbs1` |
| `mixnerf_lite_mean_masked` | 100 | 1 | `mean` | `removed_occupied` RGB / removed alpha | `1830798.pbs1` | `1830799.pbs1` |

Queue status at launch:
- The five e100 pretrains are running.
- The five dependent FCOS eval jobs are held on their corresponding pretrains.
- Existing unrelated SSL jobs remain running; this dither branch should not block
  the main budget-curve / efficiency paper path.

Decision:
- `shuffle_visible > zeros` and `shuffle_visible > mean`: keep dither as a
  separate method candidate.
- `mean ~= shuffle_visible`: likely simple non-zero filler, weak novelty.
- `shuffle_visible` collapses at e100: stop this branch.

## Experiment 65: e600 Peak Seed Check Launch and e100 Coord-jitter Audit

Snapshot:
- 2026-06-06 JST

Artifacts:
- `results/shortcut_probe_artifacts/e600_peak_seed_jobs_20260606.md`
- `results/shortcut_probe_artifacts/e600_peak_seed_jobs_20260606.csv`
- `results/shortcut_probe_artifacts/cosine_coord_jitter_e100_config_audit_20260606.md`

Decision:
- Adopt the Doc 7 direction.
- Do not promote the Doc 8 `cosine_coord_jitter_e100 = 0.6219` row to a main
  paper pillar.
- Keep `cosine_coord_jitter_e100` as an ablation/enhanced short-budget result
  only after config audit and seed confirmation.

Submitted FCOS-only seed checks:

| row | finetune seed | job |
|---|---:|---:|
| `cosine_e600` | 2 | `1830815.pbs1` |
| `baseline_e600` | 2 | `1830817.pbs1` |
| `baseline_e1200` | 2 | `1830818.pbs1` |
| `cosine_e600` | 3 | `1830819.pbs1` |
| `baseline_e600` | 3 | `1830820.pbs1` |
| `baseline_e1200` | 3 | `1830821.pbs1` |

Run settings:
- Full-label Front3D FCOS, `PERCENT_TRAIN=1.0`
- `FCOS_NUM_EPOCHS=1000`
- Existing MAE checkpoints only; no new pretraining
- `DETERMINISTIC=0`, matching the budget-curve FCOS protocol

`cosine_coord_jitter_e100` audit:
- Config appears to be clean `cosine_ramp + coord_jitter`, not surface maturation.
- `PROBE_CURRICULUM=cosine_rgb_ramp`
- `PROBE_CURRICULUM_EPOCHS=100`
- `RGB_LOSS_REGION=all_occupied`
- `ALPHA_LOSS_REGION=removed_patches`
- Surface maturation fields are empty in `results_table.csv`.

Existing finetune-seed results for `cosine_coord_jitter_e100`:

| finetune seed | AP@50 |
|---:|---:|
| 1 | 0.6219 |
| 2 | 0.5958 |
| 3 | 0.5443 |

Summary:
- `cosine_coord_jitter_e100`: AP@50 `0.5873 +/- 0.0395` over 3 finetune seeds.
- `baseline_coord_jitter_e100`: AP@50 `0.5454 +/- 0.0103` over 3 finetune seeds.
- The single-seed `0.6219` row shrinks under finetune-seed replication, so it
  should remain an ablation/enhanced variant rather than a main paper pillar.

## Experiment 66: e600 Peak Seed Check Results

Snapshot:
- 2026-06-06 JST

Artifact:
- `results/shortcut_probe_artifacts/e600_peak_seed_results_20260606.md`

Per-seed AP@50:

| condition | seed1 | seed2 | seed3 | mean±std |
|---|---:|---:|---:|---:|
| `cosine_e600` | 0.6196 | 0.4971 | 0.5065 | 0.5410±0.0682 |
| `baseline_e600` | 0.4994 | 0.4984 | 0.4955 | 0.4978±0.0021 |
| `baseline_e1200` | 0.5648 | 0.5087 | 0.4807 | 0.5181±0.0428 |

Paired AP@50 differences:

| finetune seed | `cosine_e600 - baseline_e600` | `cosine_e600 - baseline_e1200` |
|---:|---:|---:|
| 1 | +0.1202 | +0.0548 |
| 2 | -0.0014 | -0.0117 |
| 3 | +0.0110 | +0.0257 |
| mean | +0.0432 | +0.0229 |

Interpretation:
- The seed1 e600 peak does not replicate strongly.
- `cosine_e600` remains above both baselines in mean AP@50, but the margin is
  high-variance and driven heavily by seed1.
- This is not strong enough to claim a robust/decisive e600 peak over e1200
  baseline.
- Safer framing: use the budget curve as an efficiency observation with
  seed-band caveats, and combine it with low-label stability rather than making
  e600 peak the sole strong-accept pillar.

## Experiment 67: Visible-only Dithered MixNeRF e100 Scout Results

Snapshot:
- 2026-06-06 JST

Artifact:
- `results/shortcut_probe_artifacts/mixnerf/mixnerf_dither_e100_results_20260606.md`

Status:
- All e100 dither pretrain and dependent FCOS jobs completed.
- `shuffle_visible` logs confirm visible-only source:
  - `same_scene_fill_source=visible_only`
  - `self_replacement_rate=0.0`
  - `masked_source_rate=0.0`
  - `base_mask_mean=0.0`

Results:

| condition | seed | fill | AP@25 | AP@50 | AP@75 | R@50 top300 |
|---|---:|---|---:|---:|---:|---:|
| `mixnerf_lite_shuffle_visible_masked` | 1 | visible-only same-scene shuffle | 0.8425 | 0.5766 | 0.1328 | 0.7206 |
| `mixnerf_lite_shuffle_visible_masked` | 2 | visible-only same-scene shuffle | 0.8512 | 0.5873 | 0.1344 | 0.7279 |
| `mixnerf_lite_zeros_masked` | 1 | zero | 0.8480 | 0.6262 | 0.1012 | 0.7426 |
| `mixnerf_lite_zeros_masked` | 2 | zero | 0.8212 | 0.5587 | 0.1304 | 0.6912 |
| `mixnerf_lite_mean_masked` | 1 | channel mean | 0.8615 | 0.5670 | 0.0875 | 0.6912 |

AP@50 summary:

| condition | n | mean AP@50 | std AP@50 | mean AP@75 | mean R@50 |
|---|---:|---:|---:|---:|---:|
| `shuffle_visible` | 2 | 0.5819 | 0.0076 | 0.1336 | 0.7243 |
| `zero` | 2 | 0.5925 | 0.0477 | 0.1158 | 0.7169 |
| `mean` | 1 | 0.5670 | - | 0.0875 | 0.6912 |

Interpretation:
- Clean visible-only same-scene dither does not beat zero-fill on mean AP@50
  (`0.5819` vs `0.5925`).
- `shuffle_visible` is more stable and has better AP@75/R@50 mean, but the
  planned promotion criterion was AP@50 over zero and mean.
- The `mean` control is close enough to seed1 `shuffle_visible` that this is not
  strong evidence for a special scene-distribution matching mechanism.

Decision:
- Do not move dither / mask-token-free MixNeRF into the AAAI main path.
- Keep it separated as an appendix/future-method observation.
- Stop broad MixNeRF/dither exploration unless a reviewer-facing mechanism is
  specified before launching more jobs.

## Experiment 68: Visibility-Gated Feasibility Gate Setup

Snapshot:
- 2026-06-07 JST

Artifacts:
- `results/shortcut_probe_artifacts/mixnerf/dither_branch_decision_20260607.md`
- `results/shortcut_probe_artifacts/visibility/visibility_gating_feasibility_report_20260607.md`
- `nerf_mae/probe_scripts/encoder_mask_participation_report.py`
- `nerf_mae/probe_scripts/abci3_encoder_mask_participation.pbs`

Decision:
- MixNeRF/dither remains closed as a main-paper method path.
- Visibility-Gated MAE is not implemented yet.
- The next required action is a masked-token participation measurement on real
  checkpoints/data.

Measurement outputs:
- `feature_norm_by_stage.csv`
- `patch_merge_mask_stats.csv`
- `skip_feature_mask_stats.csv`
- `attention_mass_by_block.csv`
- `encoder_mask_participation_report.md/json`

Gate:
- Proceed to V0/V1 only if stage0/1 masked-visible feature norm ratio is >=
  0.25, or if patch-merge/skip stats show persistent masked-placeholder
  participation.
- Do not implement attention KV-gating unless simpler gates are positive.

Guardrail:
- Closed MixNeRF launchers now require `ALLOW_CLOSED_MIXNERF=1` before
  submission to avoid accidental jobs on the closed branch.

## Experiment 69: Encoder Mask Participation Gate Result

Snapshot:
- 2026-06-07 JST

Artifact:
- `results/shortcut_probe_artifacts/visibility/encoder_mask_participation_20260607_141908_thr001/encoder_mask_participation_report.md`
- `results/shortcut_probe_artifacts/visibility/visibility_gate_decision_20260607.md`

Result:
- Gate is positive.
- Stage0/1 masked-visible feature norm ratios exceed the 0.25 threshold for all
  measured checkpoints.

Key AP-agnostic mechanism numbers:

| checkpoint | stage0 ratio | stage1 ratio | stage2 ratio |
|---|---:|---:|---:|
| `baseline_e300` | 0.7578 | 0.6989 | 0.6995 |
| `cosine_ramp_e300` | 0.7303 | 0.7463 | 0.7872 |
| `cosine_coord_jitter_e100` | 0.6071 | 0.7066 | 0.5134 |
| `dither_shuffle_visible_e100` | 1.2778 | 1.3940 | 0.7603 |

Interpretation:
- Masked placeholders materially participate in encoder/skip features.
- Visibility-Gated V0/V1 scouts are justified.
- Attention KV-gating remains out of scope until V0/V1 results justify higher
  implementation risk.

Submitted scouts:

| condition | seed | pretrain job | dependent FCOS job |
|---|---:|---|---|
| `visibility_skip_gate` | 1 | `1832027.pbs1` | `1832028.pbs1` |
| `visibility_feature_reset` | 1 | `1832029.pbs1` | `1832030.pbs1` |

## Experiment 70: Visibility-Gated MAE e100 Scout Results

Snapshot:
- 2026-06-08 JST

Artifact:
- `results/shortcut_probe_artifacts/visibility/visibility_gated_e100_results_20260608.md`

Status:
- Both visibility-gated e100 pretrain jobs completed.
- Both dependent FCOS eval jobs completed.

Results:

| condition | seed | AP@25 | AP@50 | AP@75 | R@50 top300 | R@25 top300 | AR top300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `visibility_skip_gate` | 1 | 0.8173 | 0.5869 | 0.1039 | 0.7279 | 0.9632 | 0.4971 |
| `visibility_feature_reset` | 1 | 0.8026 | 0.4992 | 0.0552 | 0.6618 | 0.9632 | 0.4613 |

Reference context:

| condition | n | AP@50 mean | AP@50 std | AP@75 mean |
|---|---:|---:|---:|---:|
| `baseline_coord_jitter_e100` | 3 | 0.5454 | 0.0103 | 0.1073 |
| `cosine_coord_jitter_e100` | 3 | 0.5873 | 0.0395 | 0.0872 |
| `mixnerf_lite_shuffle_visible_e100` | 2 | 0.5819 | 0.0076 | 0.1336 |
| `budgetcurve_cosine_ramp_e100` | 1 | 0.5711 | - | 0.0940 |

Interpretation:
- `visibility_skip_gate` survives as a scout. It beats the
  `baseline_coord_jitter_e100` 3-seed mean and essentially ties the
  `cosine_coord_jitter_e100` 3-seed mean on AP@50.
- `visibility_skip_gate` does not clearly beat the strongest existing e100
  references, so it should not replace the main efficiency / low-label branch.
- `visibility_feature_reset` is a no-go. Hard feature reset damages transfer.

Decision:
- Stop `visibility_feature_reset`.
- Keep `visibility_skip_gate` as appendix/future-method evidence unless a
  stronger reviewer-facing mechanism is specified.
- Do not escalate to attention KV-gating from these results alone.

## Experiment 71: Visibility Audit and Masked Skip Shortcut Diagnostic

Snapshot:
- 2026-06-08 JST

Artifacts:
- `results/shortcut_probe_artifacts/visibility/visibility_implementation_audit_20260608.md`
- pending diagnostic output:
  `results/shortcut_probe_artifacts/visibility/masked_skip_shortcut_20260608_gate/`

Code:
- `nerf_mae/probe_scripts/masked_skip_shortcut_diagnostic.py`
- `nerf_mae/probe_scripts/abci3_masked_skip_shortcut_diagnostic.pbs`

Audit correction:
- The completed `visibility_skip_gate` e100 job was launched with
  `PROBE_CURRICULUM=cosine_rgb_ramp`.
- Therefore the existing AP@50 `0.5869` result is more precisely
  `cosine-ramp + decoder skip gate`, not pure skip-gate.
- Launchers now separate:
  - `visibility_skip_gate`: pure decoder skip gate, default curriculum `none`
  - `visibility_cosine_skip_gate`: decoder skip gate plus cosine RGB-ramp

Implementation audit:
- `skip_gate` leaves encoder stage propagation unchanged.
- It gates stage0/1/2 features before appending them to the decoder skip list.
- Gate is applied before decoder concat, not after concat.
- `nerf_rpn` has no visibility/skip-gate integration, so downstream FCOS is
  not changed by the pretraining wrapper.

Submitted diagnostic:

| diagnostic | job |
|---|---|
| `masked_skip_shortcut_20260608_gate` | `1833050.pbs1` |

Diagnostic protocol:
- Existing checkpoints only; no new pretraining.
- Checkpoints:
  - `baseline_coord_jitter_e100`
  - `cosine_coord_jitter_e100`
  - existing `visibility_cosine_skip_gate_e100`
- Reconstruction forward modes:
  - normal skip
  - masked-position skip zeroed
  - visible-position skip zeroed
  - all skip zeroed
- Metrics:
  - public reconstruction loss
  - RGB occupied loss
  - RGB removed-occupied loss
  - RGB visible-occupied loss
  - alpha removed loss
  - masked/visible skip gradient ratio on one scene

Decision rule:
- If masked-position skip zeroing materially changes reconstruction loss or
  gradient attribution, the decoder skip shortcut mechanism remains plausible.
- If it barely changes either, visibility should stay appendix/future work and
  no cosine+skip or attention-gating branch should be promoted.

Diagnostic result:
- Retry job `1833070.pbs1` completed.
- Output:
  `results/shortcut_probe_artifacts/visibility/masked_skip_shortcut_20260608_gate_retry/masked_skip_shortcut_diagnostic.md`

Key loss deltas vs normal:

| checkpoint | masked skip zero delta | visible skip zero delta | all skip zero delta |
|---|---:|---:|---:|
| `baseline_coord_jitter_e100` | +0.0194 | +0.0185 | +0.1211 |
| `cosine_coord_jitter_e100` | +0.0191 | +0.0202 | +0.1253 |
| `visibility_cosine_skip_gate_e100` | -0.0492 | +0.3168 | +0.0911 |

Key gradient attribution:

| checkpoint | stage0 masked/visible grad | stage1 masked/visible grad | stage2 masked/visible grad |
|---|---:|---:|---:|
| `baseline_coord_jitter_e100` | 2.6337 | 1.4215 | 1.1098 |
| `cosine_coord_jitter_e100` | 2.4144 | 1.5561 | 1.1133 |
| `visibility_cosine_skip_gate_e100` | 2.5677 | 1.7738 | 1.3907 |

Interpretation:
- Decoder skip shortcut mechanism is plausible.
- In baseline/cosine checkpoints, masked-position skip zeroing worsens
  reconstruction loss, and masked skip locations receive larger gradients than
  visible skip locations in early stages.
- In the existing visibility checkpoint, `normal` evaluation means restoring
  masked skip features that were gated during training; this hurts
  reconstruction, while `masked_zero` recovers the trained behavior.

Next action:
- Isolate pure `visibility_skip_gate` because the previous visibility result
  was actually cosine-ramp + skip gate.
- Add a second seed for `visibility_cosine_skip_gate` only after this positive
  diagnostic.

Submitted follow-up jobs:

| condition | seed | pretrain job | dependent FCOS job | note |
|---|---:|---|---|---|
| `visibility_skip_gate` | 1 | `1833071.pbs1` | `1833072.pbs1` | pure skip gate; no cosine curriculum |
| `visibility_cosine_skip_gate` | 2 | `1833073.pbs1` | `1833074.pbs1` | cosine-ramp + skip gate stability check |

Manifests:
- `output/launcher/visibility_gated_abci3vis_pure_e100_20260608/submitted.tsv`
- `output/launcher/visibility_gated_abci3vis_cosineskip_e100_20260608/submitted.tsv`

## Experiment 72: Visibility Follow-up Results

Snapshot:
- 2026-06-08 JST

Artifact:
- `results/shortcut_probe_artifacts/visibility/visibility_followup_results_20260608.md`

Status:
- `visibility_skip_gate` pure seed1 completed.
- `visibility_cosine_skip_gate` seed2 completed.

Important naming correction:
- The earlier 2026-06-07 `visibility_skip_gate` result used
  `PROBE_CURRICULUM=cosine_rgb_ramp`; it is treated here as
  `visibility_cosine_skip_gate` seed1.
- The 2026-06-08 `visibility_skip_gate` result is the pure skip-gate isolation
  with `PROBE_CURRICULUM=none`.

Results:

| condition | seed | AP@25 | AP@50 | AP@75 | R@50 top300 | R@25 top300 | AR top300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `visibility_skip_gate` pure | 1 | 0.7844 | 0.5480 | 0.0687 | 0.6912 | 0.9559 | 0.4775 |
| `visibility_cosine_skip_gate` | 1 | 0.8173 | 0.5869 | 0.1039 | 0.7279 | 0.9632 | 0.4971 |
| `visibility_cosine_skip_gate` | 2 | 0.7931 | 0.5492 | 0.0787 | 0.7353 | 0.9485 | 0.4966 |

Reference context:

| condition | n | AP@50 mean | AP@50 std | AP@75 mean | R@50 mean | AR mean |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_coord_jitter_e100` | 3 | 0.5454 | 0.0103 | 0.1073 | 0.6912 | 0.4953 |
| `cosine_coord_jitter_e100` | 3 | 0.5873 | 0.0395 | 0.0872 | 0.7181 | 0.4897 |
| `visibility_cosine_skip_gate_e100` | 2 | 0.5681 | 0.0267 | 0.0913 | 0.7316 | 0.4968 |

Interpretation:
- The pure skip-gate isolation is essentially baseline-level, not a method win.
- The cosine+skip combination is not stable: seed2 drops from seed1 AP@50
  0.5869 to 0.5492.
- The mechanism diagnostic remains positive, but the downstream intervention is
  not strong enough to promote.

Decision:
- Do not promote visibility gating to the main AAAI path.
- Do not launch attention KV-gating from these results.
- Stop visibility method exploration unless a new reviewer-facing mechanism is
  specified before running jobs.

## Experiment 73: ScanNet normalize_density Audit and Non-normalized Rerun

Snapshot:
- 2026-06-11 JST

Artifacts:
- `results/shortcut_probe_artifacts/scannet_normalize_density_audit_20260611.md`
- `results/shortcut_probe_artifacts/scannet_nonorm_jobs_20260611.csv`
- `results/shortcut_probe_artifacts/scannet_nonorm_retry_jobs_20260611.csv`

Finding:
- Existing ScanNet transfer triage runs used `--normalize_density` in both train
  and eval.
- Evidence appears in logs such as `scn_basee300.o1811879`,
  `scn_cose300.o1811887`, `scn_cj_e100.o1811881`, and
  `scn_smcj_e300.o1811882`.
- The wrapper path was `abci3_scannet_transfer_fcos.pbs ->
  run_fcos_probe_variant.sh -> train_fcos_pretrained.sh /
  test_fcos_pretrained.sh`; the train/test wrappers previously appended
  `--normalize_density` unconditionally.

Patch:
- Added a `NORMALIZE_DENSITY` env switch to `train_fcos_pretrained.sh` and
  `test_fcos_pretrained.sh`.
- Set `NORMALIZE_DENSITY=0` by default in `abci3_scannet_transfer_fcos.pbs`.
- Front3D/default behavior remains normalized unless explicitly overridden.

Rationale:
- This is not eval-only because existing ScanNet FCOS checkpoints were trained
  with normalized density. Eval-only off would create a train/eval mismatch.
- Re-run train+eval for ScanNet with `NORMALIZE_DENSITY=0`.

Submitted jobs:

| condition | job | normalize_density |
|---|---:|---:|
| `baseline_e300` | `1897280.pbs1` | 0 |
| `cosine_ramp_e300` | `1897281.pbs1` | 0 |
| `cosine_coord_jitter_e100` | `1897282.pbs1` | 0 |
| `surface_cosine_jitter_e300` | `1897283.pbs1` | 0 |

Path-fix retry:
- Initial jobs `1897280.pbs1`-`1897283.pbs1` failed immediately before
  training with `AssertionError: The checkpoint does not exist`.
- Cause: `MAE_CHECKPOINT=output/...` was validated from repo root in the PBS
  wrapper, but then passed unchanged after `cd nerf_rpn`, where it resolved as
  `nerf_rpn/output/...`.
- Fix: `run_fcos_probe_variant.sh` now canonicalizes `PRETRAIN_CHECKPOINT`
  before changing into `nerf_rpn`, accepting repo-root relative paths,
  `nerf_rpn`-relative paths, and absolute paths.

Retry jobs:

| condition | failed job | retry job | normalize_density |
|---|---:|---:|---:|
| `baseline_e300` | `1897280.pbs1` | `1897293.pbs1` | 0 |
| `cosine_ramp_e300` | `1897281.pbs1` | `1897294.pbs1` | 0 |
| `cosine_coord_jitter_e100` | `1897282.pbs1` | `1897295.pbs1` | 0 |
| `surface_cosine_jitter_e300` | `1897283.pbs1` | `1897296.pbs1` | 0 |

Status:
- Retry jobs are running on `rt_HG` as of 2026-06-11 16:39 JST.
- Worker logs show training has reached epoch 1-2, so checkpoint loading is past
  the previous failure point.

Decision:
- Treat the existing ScanNet table as legacy normalized-density triage until the
  non-normalized reruns finish.
- Use the non-normalized table for paper-facing ScanNet claims if the official
  ScanNet protocol requires density normalization off.
