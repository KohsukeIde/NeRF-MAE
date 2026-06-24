# Anchor-RPN Breadth Loader/Protocol Audit (2026-06-24)

## Status

The previous Anchor-RPN 10/25/50 label results should not be used as detector-head breadth evidence or as a scientific negative result.

## What was checked

The initial feedback hypothesized that FCOS and Anchor use different MAE checkpoint loading paths. In this working tree, that hypothesis is not supported:

- FCOS uses `SwinTransformer_FPN_Pretrained_Skip as SwinTransformer_FPN_Pretrained`.
- Anchor also uses `SwinTransformer_FPN_Pretrained_Skip as SwinTransformer_FPN_Pretrained`.
- Both construct the same MAE-compatible downstream backbone wrapper for MAE checkpoints.

An explicit constructor-vs-direct initialization audit was run for the cosine e300 checkpoint:

```text
results/shortcut_probe_artifacts/anchor_mae_init_modes_cosine_e300_20260624.json
```

Result:

```text
common_base_keys = 350
max_abs_diff_common = 0.0
mismatch_count = 0
direct_missing_count = 0
direct_unexpected_encoder_count = 0
```

Therefore, the old Anchor underperformance is more likely a protocol/training issue than an FCOS-vs-Anchor MAE loader mismatch.

## Code changes made

`nerf_rpn/run_rpn.py` now supports:

- `--mae_init_mode {constructor,direct}` for loader-audit runs.
- `--backbone_lr_scale` to match FCOS-style head/backbone optimizer grouping.
- `--lr_scheduler {onecycle_epoch,onecycle_legacy,constant}` and `--scheduler_total_steps`.
- `--freeze_backbone_epochs`.

`nerf_rpn/run_anchor_variant.sh` and `nerf_rpn/tools/abci3_front3d_low_label_anchor.pbs` pass these controls through.

## Current sanity gate

Submitted 3 Anchor-RPN 50% label jobs:

```text
results/shortcut_probe_artifacts/anchor_head_fcosp_proto_p50_seed1_sanity_jobs_retry1_20260624.csv
```

Conditions:

- scratch / joint e300 / cosine e300
- Front3D 50% labels
- seed1
- Anchor-RPN 1000 epochs
- LR 1e-4
- batch size 8
- `ANCHOR_MAE_BACKBONE_ARCH=1`
- `ANCHOR_MAE_INIT_MODE=direct`
- `DETERMINISTIC=0`
- AP@50-best checkpoint

Decision rule:

- If pretrained arms still fail to beat scratch, do not proceed to 3-seed Anchor breadth.
- If pretrained > scratch, expand to 25%/50% 3-seed with the same corrected protocol.
