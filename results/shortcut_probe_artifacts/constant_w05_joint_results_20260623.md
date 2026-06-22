# Constant w=0.5 Joint Control Results (2026-06-23)

Condition: `constant_rgb_half`, p1.0/e300, pretrain seed1, FCOS fcos1000
finetune seeds 1/2/3, `DETERMINISTIC=0`.

Purpose: separate **appearance timing** from **integrated RGB-loss magnitude**.
This control gives RGB supervision from epoch 1, but with a constant
`PROBE_RGB_WEIGHT=0.5`, roughly matching the integrated RGB weight of the
0-to-1 ramp.

## Per-Seed Results

| labels | finetune seed | AP@25 | AP@50 | AP@75 |
| --- | ---: | ---: | ---: | ---: |
| 100% | 1 | 0.7893 | 0.5066 | 0.0689 |
| 100% | 2 | 0.8027 | 0.5019 | 0.0587 |
| 100% | 3 | 0.7976 | 0.5585 | 0.0476 |
| 10% | 1 | 0.5937 | 0.1844 | 0.0008 |
| 10% | 2 | 0.5152 | 0.1311 | 0.0003 |
| 10% | 3 | 0.4970 | 0.0961 | 0.0017 |

## Mean +/- Std

| labels | AP@25 | AP@50 | AP@75 |
| --- | ---: | ---: | ---: |
| 100% | 0.7965 +/- 0.0068 | 0.5223 +/- 0.0314 | 0.0584 +/- 0.0106 |
| 10% | 0.5353 +/- 0.0514 | 0.1372 +/- 0.0444 | 0.0009 +/- 0.0007 |

## Comparison To Existing Controls

### Full Labels

| condition | AP@50 mean +/- std |
| --- | ---: |
| target-shuffled | 0.4294 +/- 0.0207 |
| occupancy-only | 0.4492 +/- 0.0186 |
| constant joint, w=1.0 | 0.4938 +/- 0.0289 |
| constant joint, w=0.5 | 0.5223 +/- 0.0314 |
| linear ramp | 0.5528 +/- 0.0162 |
| cosine ramp | 0.5724 +/- 0.0195 |

### 10% Labels

| condition | AP@50 mean +/- std |
| --- | ---: |
| target-shuffled | 0.0568 +/- 0.0389 |
| occupancy-only | 0.0591 +/- 0.0917 |
| constant joint, w=1.0 | 0.1180 +/- 0.0394 |
| constant joint, w=0.5 | 0.1372 +/- 0.0444 |
| linear ramp | 0.1448 +/- 0.0851 |
| cosine ramp | 0.2527 +/- 0.0211 |

## Reading

- `constant w=0.5` is better than `constant w=1.0`, so reducing early RGB
  pressure helps.
- `cosine ramp` remains clearly better than `constant w=0.5`:
  - full labels: +0.0500 AP@50
  - 10% labels: +0.1155 AP@50
- Therefore the gain is not explained only by reducing total RGB-loss
  magnitude. The timing/order of appearance supervision matters, especially in
  the headline 10% label regime.

Recommended paper wording:

> A constant half-weight RGB objective improves over joint reconstruction,
> indicating that excessive appearance pressure is harmful. However, it remains
> below the ramp, especially at 10% labels, showing that delaying appearance
> supervision provides an additional benefit beyond loss-magnitude reduction.
