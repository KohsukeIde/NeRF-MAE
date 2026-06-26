# Gradient Conflict Quick Probe Summary (2026-06-25/26)

Short diagnostic only: Front3D p10, 2 epochs, seed1, 1 GPU, `GM_INTERVAL=1`, `GM_MAX_BATCHES=3`, parameter filter `stages`. Values are cosine similarity between RGB-loss and alpha-loss gradients and gradient norms for the monitored parameter subset.

| condition | n | cos mean±std | cos min/max | RGB grad norm mean | alpha grad norm mean |
|---|---:|---:|---:|---:|---:|
| baseline | 6 | 0.1651±0.4601 | -0.3420/0.8819 | 0.2076 | 0.0155 |
| cosine_ramp | 6 | 0.0408±0.4568 | -0.8086/0.4991 | 0.2829 | 0.0130 |
| constant_rgb_half | 6 | 0.0210±0.2374 | -0.2717/0.3507 | 0.1944 | 0.0173 |

## baseline
| epoch | iter | cos | norm_rgb | norm_alpha |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0.350693 | 0.467732 | 0.017289 |
| 1 | 1 | -0.006216 | 0.366439 | 0.025012 |
| 1 | 2 | -0.262127 | 0.189450 | 0.016968 |
| 2 | 0 | 0.881860 | 0.143487 | 0.009364 |
| 2 | 1 | -0.341973 | 0.026665 | 0.018704 |
| 2 | 2 | 0.368258 | 0.051900 | 0.005377 |

## cosine_ramp
| epoch | iter | cos | norm_rgb | norm_alpha |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0.350693 | 0.467732 | 0.017289 |
| 1 | 1 | 0.135032 | 0.548764 | 0.020141 |
| 1 | 2 | 0.091803 | 0.316716 | 0.010183 |
| 2 | 0 | 0.499112 | 0.171597 | 0.010543 |
| 2 | 1 | -0.808644 | 0.070963 | 0.017481 |
| 2 | 2 | -0.023165 | 0.121590 | 0.002157 |

## constant_rgb_half
| epoch | iter | cos | norm_rgb | norm_alpha |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0.350693 | 0.467732 | 0.017289 |
| 1 | 1 | -0.003544 | 0.365916 | 0.024118 |
| 1 | 2 | -0.271698 | 0.194009 | 0.016073 |
| 2 | 0 | 0.240145 | 0.037357 | 0.015878 |
| 2 | 1 | -0.172452 | 0.072678 | 0.021404 |
| 2 | 2 | -0.017180 | 0.028956 | 0.009321 |

## Interpretation

- All three conditions show noisy RGB/alpha gradient alignment with both positive and negative cosine values. This supports only a weak diagnostic claim that early RGB/alpha gradients are not consistently aligned.
- The quick probe does not show a clean separation where cosine ramp removes gradient conflict relative to baseline or w=0.5. Means are close to zero to mildly positive, and variance is large.
- RGB gradient norms are roughly one order of magnitude larger than alpha gradient norms in this monitored subset. This is useful as optimization context, but should not be elevated to a main mechanism without a longer/multi-seed monitor.
- Recommended paper use: appendix-only sanity/diagnostic. Do not claim that gradient conflict causally explains the detector gains from this quick probe alone.
