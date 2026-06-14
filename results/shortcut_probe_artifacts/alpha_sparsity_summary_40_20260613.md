# Alpha Sparsity / Entropy Stats

Scenes: 40

Density-to-alpha conversion: `alpha = clip(1 - exp(-exp(density) / 100), 0, 1)`.

## Summary

| metric | mean | std | median | min | max |
|---|---:|---:|---:|---:|---:|
| alpha_mean | 0.0852289 | 0.0152618 | 0.0852691 | 0.0495516 | 0.111845 |
| alpha_p95 | 0.539008 | 0.111287 | 0.543664 | 0.346608 | 0.753 |
| alpha_p99 | 0.948533 | 0.0474302 | 0.966036 | 0.828522 | 0.999997 |
| occ_frac_gt_0p01 | 0.31466 | 0.0559794 | 0.307608 | 0.178243 | 0.451708 |
| occ_entropy_gt_0p01 | 0.887894 | 0.064818 | 0.890347 | 0.676218 | 0.99326 |
| alpha_hist_entropy_64 | 2.52455 | 0.335461 | 2.53453 | 1.66253 | 3.23811 |

Interpretation:
- `occ_frac_gt_0p01` estimates how much of the grid carries non-empty alpha evidence.
- Low occupancy fraction and low binary occupancy entropy support treating alpha as a sparse structural signal.
- RGB appearance statistics are intentionally omitted from this quick table; this is an alpha-structure diagnostic, not a method claim.
