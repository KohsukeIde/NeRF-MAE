# Boundary-SDF B1 Scout Jobs

Date: 2026-06-01 JST

## Purpose

Run one low-cost scout for the denoised Boundary-SDF route while the low-label
gate is still running. This is a hedge for the case where the current
structure-first recipe is label-efficient but not strong enough for a
strong-accept method framing.

## Scout

`B1: denoised SDF auxiliary`

```text
base objective = public/effective NeRF-MAE objective
auxiliary head = decoder-side 1-channel SDF head
alpha_smoothing_sigma = 1.0
alpha_threshold = 0.02
distance_clip = 16
loss = RGB + alpha + 0.2 * SDF
SDF loss mask = removed patches
pretrain = e100, seed1, p1.0
downstream = Front3D FCOS e1000, finetune seed1
```

The SDF target is a GPU-friendly max-pool distance approximation to the
smoothed-alpha boundary. The target settings come from the audit v2 decision:
`smooth100_thr002`.

## Jobs

| job | state at submit | role |
|---|---|---|
| `1816233.pbs1` | running | pretrain `boundary_sdf_aux`, e100 |
| `1816234.pbs1` | hold afterok | dependent Front3D FCOS e1000 |

## Paths

Pretrain output:

```text
output/nerf_mae/results/nerfmae_boundary_sdf_aux_p1.0_e100_seed1_abci3b1sdf/
```

Expected checkpoint:

```text
output/nerf_mae/results/nerfmae_boundary_sdf_aux_p1.0_e100_seed1_abci3b1sdf/epoch_100.pt
```

Expected FCOS eval:

```text
output/nerf_rpn/results/nerfmae_boundary_sdf_aux_p1.0_e100_seed1_abci3b1sdf_epoch100_sched_epoch_seed1_fcos1000_eval/eval.json
```

## Initial Runtime Check

The pretrain job started successfully and completed early iterations:

```text
epoch 1 [0/204]    loss=1.1396  loss_rgb=0.9020  loss_alpha=0.1840
epoch 1 [180/204]  loss=0.2389  loss_rgb=0.1546  loss_alpha=0.0670
```

The difference between total loss and `loss_rgb + loss_alpha` confirms that
the SDF auxiliary term is active.
