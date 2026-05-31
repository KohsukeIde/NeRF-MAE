# Compute-Normalized Sample-Efficiency Main Table

Snapshot: 2026-05-31 JST

Purpose:
- Provide the paper-facing compute table requested in the feedback.
- Compare NeRF-MAE paper rows and current local rows by scene count, epochs,
  scene-epochs, approximate ABCI3/H200-equivalent GPU-days, and paper-protocol
  OBB detection metrics.
- AP@50 is the primary paper metric. AP@75 remains a localization diagnostic
  and is intentionally excluded from this main compute table.

Sources and conventions:
- NeRF-MAE Table 2 reports Front3D Recall@25 / Recall@50 / AP@25 / AP@50 for
  `NeRF-MAE (F3D)` as `0.963 / 0.743 / 0.830 / 0.591`, and `NeRF-MAE (Ours)`
  as `0.972 / 0.745 / 0.853 / 0.630`.
- NeRF-MAE Fig. 3 reports `1998` Front3D scenes, `1330` HM3D scenes, `250`
  Hypersim scenes, and `90` ScanNet scenes. The paper's self-supervised
  pretraining mix for `NeRF-MAE (Ours)` is treated as Front3D+HM3D+Hypersim,
  i.e. `1998 + 1330 + 250 = 3578` scenes. ScanNet is excluded from the
  pretraining scene count because it is used for cross-dataset transfer.
- The released NeRF-MAE README says the default 1200-epoch pretraining uses
  `batch_size=32`, `8` A100 GPUs, and takes around `2` days. That is about
  `16` A100 GPU-days for the paper default.
- Local ABCI3/H200-equivalent GPU-days are estimated from measured local
  1-node/8-H200 pretraining speed: e300 walltime is `~11.63h`, so e300 costs
  `8 * 11.63 / 24 = 3.88` H200 GPU-days. e100 costs `1.29` H200 GPU-days.
  Paper rows are scaled by scene-epochs using the same throughput. These are
  approximate compute-normalized values, not paper-reported H200 timings.
- Current local pretraining split has `3260` train scenes in
  `dataset/pretrain/nerfmae_split.npz`. This is a local mixed split, not a
  strictly Front3D-only run.

## Main Table

| setting | pretrain data | train scenes | epochs | scene-epochs | rel. scene-epochs vs paper multi | reduction vs paper multi | approx ABCI H200 GPU-days | AP@25 | AP@50 | Recall@50 | status / caveat |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NeRF-MAE (F3D), aug | paper Front3D | 1998 | 1200 | 2,397,600 | 0.558 | 1.79x less | 9.52 | 0.830 | 0.591 | 0.743 | paper Table 2 |
| NeRF-MAE (Ours), aug | paper F3D+HM3D+Hypersim | 3578 | 1200 | 4,293,600 | 1.000 | 1.00x | 17.05 | 0.853 | 0.630 | 0.745 | paper Table 2; README default is also about 16 A100 GPU-days |
| `baseline_e300` | local mixed split | 3260 | 300 | 978,000 | 0.228 | 4.39x less | 3.88 | 0.799 | 0.494 | 0.662 | 1 pretrain seed x 3 finetune seeds, ABCI clean |
| `cosine_ramp_e300` | local mixed split | 3260 | 300 | 978,000 | 0.228 | 4.39x less | 3.88 | 0.829 | 0.572 | 0.701 | 1 pretrain seed x 3 finetune seeds, ABCI clean |
| `baseline_e1200` | local mixed split | 3260 | 1200 | 3,912,000 | 0.911 | 1.10x less | 15.52 | 0.849 | 0.589 | 0.713 | single finetune seed |
| `cosine_coord_jitter_e100` | local mixed split | 3260 | 100 | 326,000 | 0.076 | 13.17x less | 1.29 | 0.799 | 0.587 | 0.718 | 1 pretrain seed x 3 finetune seeds; seed-1 AP@50 was 0.6219 |
| `surface+cosine+jitter_e300` | local mixed split | 3260 | 300 | 978,000 | 0.228 | 4.39x less | 3.88 | 0.819 | 0.640 | 0.765 | single finetune seed; strong AP@50/recall but weak AP@75 |
| `paper_loss_e300` | local mixed split | 3260 | 300 | 978,000 | 0.228 | 4.39x less | 3.88 | pending | pending | pending | kill experiment `1811826.pbs1` -> `1811827.pbs1` |

## Paper-Facing Reading

- The clean e300 comparison is a `4.39x` scene-epoch / ABCI-H200-GPU-day
  reduction relative to paper multi-source e1200. It is not a `1/12` claim.
- The `~1/12` phrasing applies to `cosine_coord_jitter_e100`: it uses `7.6%`
  of paper multi-source scene-epochs, i.e. about `13.17x` less compute by this
  estimate, while reaching `0.587` mean AP@50 and `0.718` Recall@50.
- The strongest current AP@50 row is `surface+cosine+jitter_e300` at `0.640`,
  slightly above the paper multi-source AP@50 `0.630`, using `22.8%` of the
  paper multi-source scene-epochs. This is still a single-finetune-seed row and
  should not be promoted to the main claim before the running low-label and
  `paper_loss_e300` gates are interpreted.
- The current local rows are not strictly "single-source"; they use the local
  mixed pretraining split. A paper claim should phrase this as
  "fewer-epoch / fewer-scene-epoch pretraining" unless a Front3D-only pretrain
  row is isolated.

## Safe Claim Wording

Use:

> Under an ABCI3/H200-normalized compute estimate, our e300 local pretraining
> rows use about 22.8% of the paper multi-source 1200-epoch scene-epochs, while
> the e100 coord-jitter curriculum uses about 7.6%. Several rows approach the
> paper Front3D-augmented AP@50, and the best single-seed e300 row reaches the
> paper multi-source AP@50 band.

Avoid:

> single-source 300 epochs matches multi-source 1200 epochs

unless a true Front3D-only pretraining row is added.

## Remaining Before Camera-Ready Use

1. Add `paper_loss_e300` after jobs `1811826` and `1811827` finish.
2. Add the Front3D low-label 50% gate after jobs `1812644`-`1812647` finish.
3. Decide whether `surface+cosine+jitter_e300` gets finetune-seed expansion; do
   not treat the single-seed `0.640` as a stable final number yet.
4. If a "single-source" claim is desired, run or isolate Front3D-only
   pretraining. Current local pretraining rows are mixed-source.
