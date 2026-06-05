# Budget Curve and Low-label Reversal Results

Snapshot: 2026-06-06 JST

Current queue status:
- Budget-curve and reversal-defense jobs are complete.
- Remaining running jobs are unrelated SSL jobs (`simclrv1`, `simclrv2`,
  `byol`, `ibot`).

## Dedicated-budget curve

Each budget point below uses a run whose total pretraining epoch count matches
the plotted budget. The e100/e300/e600/e1200 points are therefore not intermediate
checkpoints from a single e1200 schedule.

| method | epochs | AP@25 | AP@50 | AP@75 | R@50 top300 | R@25 top300 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 100 | 0.7940 | 0.5422 | 0.0830 | 0.6912 | 0.9632 |
| baseline | 300 | 0.7956 | 0.4695 | 0.0869 | 0.6618 | 0.9559 |
| baseline | 600 | 0.7994 | 0.4994 | 0.0767 | 0.6765 | 0.9485 |
| baseline | 1200 | 0.7934 | 0.5648 | 0.0809 | 0.7059 | 0.9485 |
| cosine_ramp | 100 | 0.8095 | 0.5711 | 0.0940 | 0.7132 | 0.9559 |
| cosine_ramp | 300 | 0.8249 | 0.5539 | 0.1135 | 0.7059 | 0.9632 |
| cosine_ramp | 600 | 0.8220 | 0.6196 | 0.0721 | 0.7279 | 0.9412 |
| cosine_ramp | 1200 | 0.8338 | 0.5490 | 0.0640 | 0.6838 | 0.9632 |

Interpretation:
- The curve is not a clean monotonic saturation curve.
- `cosine_ramp` is stronger than baseline at e100/e300/e600, with the strongest
  AP@50 at e600 (`0.6196`).
- The e1200 `cosine_ramp` point drops below baseline e1200 on AP@50
  (`0.5490` vs `0.5648`), so the paper should not claim monotonic improvement
  with budget.
- The defensible read is a mid-budget sample-efficiency peak, not a guaranteed
  long-budget dominance story.

## Low-label grid, seed1

| condition | label fraction | AP@25 | AP@50 | AP@75 | R@50 top300 |
|---|---:|---:|---:|---:|---:|
| scratch | 10% | 0.4453 | 0.1160 | 0.0000 | 0.2794 |
| scratch | 25% | 0.6087 | 0.3044 | 0.0122 | 0.4779 |
| scratch | 50% | 0.7065 | 0.3666 | 0.0513 | 0.5956 |
| scratch | 100% | 0.7952 | 0.4722 | 0.0703 | 0.6176 |
| baseline e300 | 10% | 0.5344 | 0.1328 | 0.0001 | 0.3824 |
| baseline e300 | 25% | 0.6550 | 0.2777 | 0.0057 | 0.5147 |
| baseline e300 | 50% | 0.7671 | 0.4191 | 0.0241 | 0.6471 |
| baseline e300 | 100% | 0.7956 | 0.4695 | 0.0869 | 0.6618 |
| cosine_ramp e300 | 10% | 0.5996 | 0.2751 | 0.0152 | 0.4485 |
| cosine_ramp e300 | 25% | 0.7008 | 0.3639 | 0.0095 | 0.5882 |
| cosine_ramp e300 | 50% | 0.7690 | 0.5026 | 0.0516 | 0.6691 |
| cosine_ramp e300 | 100% | 0.8249 | 0.5539 | 0.1135 | 0.7059 |
| surface_cosine_jitter e300 | 10% | 0.5918 | 0.1756 | 0.0066 | 0.4118 |
| surface_cosine_jitter e300 | 25% | 0.7123 | 0.3460 | 0.0200 | 0.5368 |
| surface_cosine_jitter e300 | 50% | 0.7811 | 0.5217 | 0.0627 | 0.6765 |
| surface_cosine_jitter e300 | 100% | 0.8178 | 0.5984 | 0.1004 | 0.7059 |

Interpretation:
- `cosine_ramp` is the better low-label method at 10% and 25%.
- `surface_cosine_jitter` is best at 50% and 100% in-domain.
- This argues for treating `cosine_ramp` as the safer base method and
  surface anchoring as an in-domain / label-richer enhancement unless cross-axis
  results make surface anchoring consistently superior.

## 50%-label reversal, 3 finetune seeds

| condition | seed | AP@25 | AP@50 | AP@75 | R@50 top300 |
|---|---:|---:|---:|---:|---:|
| scratch 100% | 1 | 0.7952 | 0.4722 | 0.0703 | 0.6176 |
| scratch 100% | 2 | 0.7766 | 0.5340 | 0.0789 | 0.6397 |
| scratch 100% | 3 | 0.7916 | 0.4422 | 0.0827 | 0.6029 |
| cosine_ramp 50% | 1 | 0.7690 | 0.5026 | 0.0516 | 0.6691 |
| cosine_ramp 50% | 2 | 0.7802 | 0.4904 | 0.0434 | 0.6471 |
| cosine_ramp 50% | 3 | 0.7523 | 0.4980 | 0.0521 | 0.6471 |
| surface_cosine_jitter 50% | 1 | 0.7811 | 0.5217 | 0.0627 | 0.6765 |
| surface_cosine_jitter 50% | 2 | 0.8170 | 0.5309 | 0.0695 | 0.6691 |
| surface_cosine_jitter 50% | 3 | 0.7859 | 0.4924 | 0.0466 | 0.6691 |

AP@50 summary:

| condition | mean AP@50 | std AP@50 | n |
|---|---:|---:|---:|
| scratch 100% | 0.4828 | 0.0468 | 3 |
| cosine_ramp 50% | 0.4970 | 0.0062 | 3 |
| surface_cosine_jitter 50% | 0.5150 | 0.0201 | 3 |

Interpretation:
- The reversal is moderate, not overwhelming.
- `cosine_ramp 50%` beats scratch 100% in mean AP@50 by `+0.0142`, but not in
  all paired seeds.
- `surface_cosine_jitter 50%` is stronger, beating scratch 100% in mean AP@50 by
  `+0.0322` with lower variance than scratch.
- A safe paper claim is that 50%-label structure-first pretraining can match or
  modestly exceed full-label scratch with lower variance. Avoid calling this a
  large or decisive reversal.

