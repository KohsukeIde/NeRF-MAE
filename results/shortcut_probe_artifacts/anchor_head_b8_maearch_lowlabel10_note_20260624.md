# Anchor-head breadth rerun note (2026-06-24)

## Why the previous anchor-head low-label table is invalid

The earlier anchor-head low-label results in
`anchor_head_lowlabel_results_20260624.*` are not used for paper claims. They
were produced before the detector-head pipeline sanity checks and showed a
collapsed detector pattern: near-zero AP@50 despite high proposal recall, and
pretrained arms below scratch. Those numbers confound representation transfer
with detector/head setup failures.

## Sanity gate that now passes

We first checked full-label Anchor-RPN scratch with AP-best checkpoint selection
and batch size 8 on one GPU.

| setup | backbone | AP@25 | AP@50 | AP@75 | R@25 top1000 | R@50 top1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| official scratch | official Swin-FPN | 0.5346 | 0.2845 | 0.0024 | 0.9706 | 0.6103 |
| scratch, MAE-compatible arch | `--mae_backbone_arch` | 0.5699 | 0.3210 | 0.0023 | 0.9706 | 0.6765 |

The MAE-compatible scratch backbone does not collapse and is at least as strong
as the official scratch sanity, so the breadth rerun uses this architecture for
all arms.

## Submitted low-label breadth rerun

Protocol:

- Front3D OBB detection, 10% labels.
- Anchor-RPN head, 200 epochs.
- Batch size 8, one GPU.
- `model_best_ap50.pt` selection.
- `DETERMINISTIC=0`.
- `--mae_backbone_arch` for scratch and pretrained arms.

Jobs are recorded in:

`results/shortcut_probe_artifacts/anchor_head_b8_maearch_lowlabel10_jobs_20260624.csv`

Arms:

- scratch, seeds 1/2/3.
- joint e300 checkpoint, seeds 1/2/3.
- cosine/structure-first e300 checkpoint, seeds 1/2/3.

These are the only anchor-head results that should be considered for detector
breadth unless a later note supersedes them.
