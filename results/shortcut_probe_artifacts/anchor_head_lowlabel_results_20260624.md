# Anchor-head Front3D 10% label results (2026-06-24)

**Validity status: invalid for paper claims.** These results are kept only as a
failed pipeline diagnostic. The detector operates near the noise floor
(AP@50 ~ 0 for all arms), and the high-recall/near-zero-AP pattern indicates a
scoring/ranking failure rather than a meaningful representation comparison.
Do not use this table as evidence that structure-first transfer fails under an
anchor head. See `anchor_head_lowlabel_invalid_note_20260624.md`.

- Detector head: anchor-based NeRF-RPN (`run_rpn.py`).
- Data: Front3D, `PERCENT_TRAIN=0.1`, 3 finetune seeds.
- Backbone comparison is architecture-matched: scratch uses MAE-compatible Swin-FPN architecture with random initialization.
- Metrics are final test eval JSONs from `model_best.pt`, not train-loop validation prints.

| arm | seed | AP@25 | AP@50 | AP@75 | R@25 top2500 | R@50 top2500 |
|---|---:|---:|---:|---:|---:|---:|
| scratch_maearch | 1 | 0.0648 | 0.0020 | 0.0000 | 0.9118 | 0.2353 |
| scratch_maearch | 2 | 0.0870 | 0.0030 | 0.0000 | 0.9118 | 0.2279 |
| scratch_maearch | 3 | 0.0604 | 0.0022 | 0.0000 | 0.9044 | 0.1838 |
| joint_e300 | 1 | 0.0009 | 0.0000 | 0.0000 | 0.4706 | 0.0221 |
| joint_e300 | 2 | 0.0037 | 0.0001 | 0.0000 | 0.6618 | 0.0735 |
| joint_e300 | 3 | 0.0557 | 0.0005 | 0.0000 | 0.9118 | 0.1471 |
| structure_first_cosine_e300 | 1 | 0.0557 | 0.0005 | 0.0000 | 0.8824 | 0.1250 |
| structure_first_cosine_e300 | 2 | 0.0019 | 0.0000 | 0.0000 | 0.6691 | 0.0515 |
| structure_first_cosine_e300 | 3 | 0.1039 | 0.0024 | 0.0000 | 0.9118 | 0.1985 |

| arm | AP@25 mean±sd | AP@50 mean±sd | AP@75 mean±sd | R@25 mean±sd | R@50 mean±sd |
|---|---:|---:|---:|---:|---:|
| scratch_maearch | 0.0707±0.0143 | 0.0024±0.0005 | 0.0000±0.0000 | 0.9093±0.0042 | 0.2157±0.0278 |
| joint_e300 | 0.0201±0.0309 | 0.0002±0.0002 | 0.0000±0.0000 | 0.6814±0.2212 | 0.0809±0.0628 |
| structure_first_cosine_e300 | 0.0538±0.0510 | 0.0010±0.0013 | 0.0000±0.0000 | 0.8211±0.1324 | 0.1250±0.0735 |

- CSV: `/groups/gag51404/ide/vgi/NeRF-MAE/results/shortcut_probe_artifacts/anchor_head_lowlabel_results_20260624.csv`
