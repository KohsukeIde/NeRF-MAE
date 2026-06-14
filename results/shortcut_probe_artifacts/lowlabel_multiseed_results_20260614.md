# Low-Label Multi-Seed Results

Snapshot: 2026-06-14 JST

Rows:
- Front3D OBB detection.
- FCOS 1000 epochs.
- Pretrain checkpoint fixed to e300 seed1 for `baseline_e300` and `cosine_ramp_e300`.
- Values are mean +- sample std over finetune seeds 1/2/3.

## 10% Labels

| condition | AP@25 | AP@50 | AP@75 | Recall@50 top300 |
|---|---:|---:|---:|---:|
| scratch | 0.4226 +- 0.0228 | 0.1053 +- 0.0113 | 0.0016 +- 0.0016 | 0.2574 +- 0.0195 |
| baseline_e300 | 0.5315 +- 0.0044 | 0.1180 +- 0.0394 | 0.0008 +- 0.0009 | 0.3211 +- 0.0877 |
| cosine_ramp_e300 | 0.5919 +- 0.0366 | 0.2527 +- 0.0211 | 0.0059 +- 0.0081 | 0.4534 +- 0.0085 |

AP@50 deltas:
- cosine - scratch: +0.1474
- cosine - baseline_e300: +0.1347

Reading:
- This is the strongest label-efficiency result.
- The gain is not a seed1 artifact; it remains large over 3 finetune seeds.

## 25% Labels

| condition | AP@25 | AP@50 | AP@75 | Recall@50 top300 |
|---|---:|---:|---:|---:|
| scratch | 0.6094 +- 0.0170 | 0.2946 +- 0.0094 | 0.0074 +- 0.0042 | 0.4828 +- 0.0085 |
| baseline_e300 | 0.6438 +- 0.0226 | 0.2964 +- 0.0446 | 0.0047 +- 0.0012 | 0.5196 +- 0.0516 |
| cosine_ramp_e300 | 0.6812 +- 0.0190 | 0.3187 +- 0.0405 | 0.0049 +- 0.0043 | 0.5392 +- 0.0449 |

AP@50 deltas:
- cosine - scratch: +0.0241
- cosine - baseline_e300: +0.0223

Reading:
- The 25% AP@50 mean is higher for cosine, but the margin is small relative to seed variance.
- AP@25 and Recall@50 are more consistently favorable than AP@50.
- Use as supporting evidence, not as the primary headline.

## Per-Seed AP@50

| labels | condition | seed1 | seed2 | seed3 |
|---:|---|---:|---:|---:|
| 10% | scratch | 0.1160 | 0.1063 | 0.0935 |
| 10% | baseline_e300 | 0.1328 | 0.1478 | 0.0733 |
| 10% | cosine_ramp_e300 | 0.2751 | 0.2498 | 0.2331 |
| 25% | scratch | 0.3044 | 0.2857 | 0.2937 |
| 25% | baseline_e300 | 0.2777 | 0.3474 | 0.2642 |
| 25% | cosine_ramp_e300 | 0.3639 | 0.3065 | 0.2857 |

Paper implication:
- Strongest defensible headline: `cosine_ramp_e300` substantially improves the extremely low-label 10% regime.
- Safer broader claim: it improves budget-aware transfer on average, with the largest and most stable gain at 10%; 25% is positive but variance-limited.
