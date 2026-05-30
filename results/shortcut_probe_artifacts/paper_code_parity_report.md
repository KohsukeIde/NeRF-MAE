# Paper-Code Parity Report

Snapshot: 2026-05-31 JST

## Status

The official released NeRF-MAE implementation and this fork use the same
effective baseline reconstruction objective in the visible source checked so
far: RGB/radiance loss is normalized over all occupied voxels
(`target_alpha > 0.01`), while opacity/alpha loss is normalized over removed
patch voxels. The RGB line that would restrict the loss to `mask_remove` is
present but commented out in both visible implementations.

This means that most existing baseline/cosine/coord-jitter results in this
project are already measurements of the released effective objective, not a
fresh fork-only variant.

## Loss Parity

| source | RGB loss region | alpha loss region | denominator | status |
|---|---|---|---|---|
| paper equation / nominal reading | removed & occupied | removed patches | masked occupied / removed | needs counterfactual |
| official released code | all occupied | removed patches | occupied count / removed count | verified from visible source |
| Kohsuke fork baseline | all occupied | removed patches | occupied count / removed count | verified locally |
| `paper_loss` scout | removed & occupied | removed patches | removed occupied / removed | launched as kill experiment |
| historical `masked_only_rgb_loss` | removed & occupied-like | removed patches | historical probe implementation | only low-budget/older protocol |

Local fork evidence:
- `nerf_mae/model/mae/swin_mae3d.py::forward_loss`
- RGB:
  `loss_rgb = (loss_rgb * mask).sum() / mask.sum()`
- Commented masked-RGB alternative:
  `# loss_rgb = (loss_rgb * mask_remove).sum() / mask_remove.sum()`
- Alpha:
  `loss_alpha = (loss_alpha * mask_remove).sum() / mask_remove.sum()`

Interpretation:
- `public_code_loss` is the current baseline.
- The missing decisive comparison is `paper_loss_e300`, not another broad
  method variant.
- If `paper_loss_e300 ~= public_code_loss_e300`, the paper/code mismatch is not
  the main downstream driver.
- If `paper_loss_e300` differs materially, objective-fidelity becomes a real
  scientific route.

## Augmentation Parity

| source | flip | rotate | scale | grid shift | phase |
|---|---:|---:|---:|---:|---|
| paper description | 0.5 | 0.5 | 0.5 | no | pretrain + transfer |
| code default args | args-driven, default 0.5 | args-driven, default 0.5 | args-driven, default 0.5 | no by default | script dependent |
| `coord_jitter` scouts | yes | yes | usually disabled in our scouts | yes, zero-fill horizontal shift | probe-specific |

Interpretation:
- `coord_jitter` overlaps with the paper's augmentation story but is not
  identical. The grid-space zero-fill shift is a probe-specific extension.
- Experiment 43 showed eval-time jitter robustness does not explain
  `cosine_coord_jitter_e100`.

## Metric Parity

| metric | paper role | project role |
|---|---|---|
| AP@25/AP@50 | main detection metrics | main detection metrics |
| Recall@25/@50 | main detection metrics | main detection metrics |
| AP@75 | not a main reported metric | fine-localization diagnostic |
| semantic voxel labeling | reported downstream task | triage candidate if data available |
| voxel super-resolution | reported downstream task | triage candidate if data available |

Interpretation:
- AP@75 should be used as a diagnostic, not as the primary claim against the
  NeRF-MAE paper protocol.
- Any strong method claim should eventually be checked beyond AP@50 detection,
  but the public release may not include semantic/SR data in a ready-to-run form.

## Existing Results Relevant to This Audit

| condition | protocol | status |
|---|---|---|
| `baseline_e300/e1200` | public effective objective | already measured |
| `cosine_e300/e600`, `cosine_coord_jitter` | public/probe occupied-RGB objective | already measured |
| `masked_only_rgb_loss` | paper-like low-budget probe | measured only in low-budget/older protocols |
| `paper_loss_e300` | p1.0/e300 clean counterfactual | submitted as `1811826.pbs1` -> `1811827.pbs1` |

## Immediate Decision

Run exactly one kill experiment first:

```text
public_code_loss_e300 vs paper_loss_e300
```

Use the same dataset split, p1.0, e300, FCOS e1000, seed 1, and AP@25/AP@50
main metrics. AP@75 remains diagnostic. Only if this comparison differs
materially should we run `visible_occupied_rgb_only` or
`masked_occupied_rgb_only`.

Submitted job:

| condition | pretrain job | FCOS job | log dir |
|---|---:|---:|---|
| `paper_loss` | `1811826.pbs1` | `1811827.pbs1` | `output/launcher/paper_loss_e300_20260531_002753` |

## Other-Task Availability Check

The repo contains scripts for semantic voxel labeling and voxel SR:

- `nerf_rpn/run_voxel_semantics.py`
- `nerf_rpn/train_voxel_semantics.sh`
- `nerf_rpn/run_voxelSR.py`
- `nerf_rpn/train_voxelSR.sh`
- `nerf_rpn/test_voxelSR.sh`

However, the currently linked Front3D finetuning data only exposes:

```text
features/
obb/
aabb/
```

No ready semantic voxel directory or high-resolution SR feature directory was
found under the current `dataset/` tree. Therefore semantic/SR triage is not an
immediate run-until-data-is-prepared task in this ABCI workspace.
