# Compute-Normalized Sample-Efficiency Table

Snapshot: 2026-05-31 JST

Purpose:
- Compare NeRF-MAE paper settings and this project's measured settings by
  pretraining scene-epochs (`pretrain train scenes * pretrain epochs`).
- Keep AP@50 as the paper-protocol detection metric. AP@75 remains a diagnostic
  and is intentionally not used in this compute table.

Sources:
- NeRF-MAE main paper Table 2 and Fig. 3:
  https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12306.pdf
- NeRF-MAE supplement implementation/compute details:
  https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12306-supp.pdf
- Local project results:
  `results/shortcut_probe_artifacts/results_table.csv`

## Scene Count Conventions

Paper data mix:
- Fig. 3 reports `1998` Front3D scenes, `1330` HM3D scenes, `250` Hypersim
  scenes, and `90` ScanNet scenes, `3668` total.
- The paper states that `NeRF-MAE (Ours)` uses F3D, HM3D, and Hypersim for
  self-supervised pretraining, while ScanNet is held out for cross-dataset
  transfer. Therefore this table uses `1998 + 1330 + 250 = 3578` as the paper
  multi-source pretraining scene count.

Local data mix:
- Current `dataset/pretrain/nerfmae_split.npz` train split has `3260` scenes:
  `1839` Front3D-like, `1171` HM3D-like, and `250` Hypersim-like scenes.
- Current detection finetune split has `122` train / `20` val / `17` test scenes.

Reference denominators:
- Paper F3D e1200 scene-epochs: `1998 * 1200 = 2,397,600`.
- Paper multi-source e1200 pretrain scene-epochs:
  `3578 * 1200 = 4,293,600`.
- Local e1200 scene-epochs: `3260 * 1200 = 3,912,000`.

## Main Table

| setting | pretrain data | train scenes | epochs | scene-epochs | rel. to paper F3D e1200 | rel. to paper multi e1200 | AP@50 | status / caveat |
|---|---|---:|---:|---:|---:|---:|---:|---|
| NeRF-MAE (F3D), no aug | paper Front3D | 1998 | 1200 | 2,397,600 | 1.000 | 0.558 | 0.543 | paper Table 2 |
| NeRF-MAE (F3D), aug | paper Front3D | 1998 | 1200 | 2,397,600 | 1.000 | 0.558 | 0.591 | paper Table 2 |
| NeRF-MAE (Ours), aug | paper F3D+HM3D+Hypersim | 3578 | 1200 | 4,293,600 | 1.791 | 1.000 | 0.630 | paper Table 2; ScanNet excluded from pretrain count |
| `baseline_e300` | local mixed split | 3260 | 300 | 978,000 | 0.408 | 0.228 | 0.4938 mean | 3 finetune seeds, ABCI clean |
| `cosine_ramp_e300` | local mixed split | 3260 | 300 | 978,000 | 0.408 | 0.228 | 0.5723 mean | 3 finetune seeds, ABCI clean; seed-1 historical diagnostic also reached 0.5987 |
| `baseline_e1200` | local mixed split | 3260 | 1200 | 3,912,000 | 1.632 | 0.911 | 0.5892 | single finetune seed |
| `cosine_coord_jitter_e100` | local mixed split | 3260 | 100 | 326,000 | 0.136 | 0.076 | 0.5873 mean | 3 finetune seeds; ft seed 1 reached 0.6219 |
| `surface+cosine+jitter_e300` | local mixed split | 3260 | 300 | 978,000 | 0.408 | 0.228 | 0.6397 | single finetune seed; AP@75 weak, localization diagnostic needed |
| `paper_loss_e300` | local mixed split | 3260 | 300 | 978,000 | 0.408 | 0.228 | pending | kill experiment `1811826.pbs1` -> `1811827.pbs1` |

## Immediate Reading

- The clean `cosine_ramp_e300` 3-finetune-seed mean (`0.5723`) uses about
  `40.8%` of paper F3D e1200 scene-epochs and `22.8%` of paper multi-source
  e1200 pretraining scene-epochs. It does not exceed the paper F3D+aug AP@50
  (`0.591`) on mean, but it is close.
- The `cosine_coord_jitter_e100` mean (`0.5873`) is especially compute-efficient:
  it uses only `13.6%` of paper F3D e1200 and `7.6%` of paper multi-source
  e1200 scene-epochs. However, it is an e100 condition and the single best
  `0.6219` row is a single finetune seed; it must not be presented as a stable
  3-seed claim.
- The strongest AP@50 row in the current table is
  `surface+cosine+jitter_e300` (`0.6397`) at `22.8%` of paper multi-source e1200
  scene-epochs, but it is single-seed and has weak AP@75. It is a
  sample-efficiency candidate, not yet a localization/fidelity claim.
- The current local pretraining split is not strictly "single-source"; it is a
  local mixed split. Any paper text that says "single-source e300" must be
  corrected unless a truly Front3D-only pretraining run is isolated.

## Missing Before Paper Use

1. Reconcile `cosine_coord_jitter` protocol labels (`e100` vs `e300`) before
   using the `0.6219` number as a headline.
2. Add `paper_loss_e300` after jobs `1811826` and `1811827` finish.
3. If claiming compute efficiency, include both scene-epochs and measured
   GPU-hours/GPU-days where logs allow it.
4. Keep AP@75/proposal diagnostics separate from this AP@50 compute table.
