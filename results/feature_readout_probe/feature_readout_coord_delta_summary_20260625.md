# Feature Readout Coordinate-Only Control Summary (2026-06-25)

Metric is balanced AP from the frozen-feature linear readout probe, not detector AP. `delta_coord` subtracts the coordinate-only `[x,y,z]` readout at the same split/stage/target.

## 10% train scenes
### objectness
| arm | stage0 AP / Δ | stage1 AP / Δ | stage2 AP / Δ |
|---|---:|---:|---:|
| coord_only | 0.5683 / -- | 0.2666 / -- | 0.1677 / -- |
| scratch | 0.7814 / +0.2132 | 0.5849 / +0.3183 | 0.5677 / +0.3999 |
| joint_e300 | 0.8017 / +0.2334 | 0.5890 / +0.3224 | 0.4932 / +0.3255 |
| cosine_e300 | 0.8029 / +0.2347 | 0.6097 / +0.3431 | 0.5254 / +0.3577 |
| linear_e300 | 0.8074 / +0.2391 | 0.5819 / +0.3153 | 0.5202 / +0.3525 |
| w05_e300 | 0.8080 / +0.2398 | 0.5779 / +0.3112 | 0.4975 / +0.3298 |
| occupancy_only_e300 | 0.6770 / +0.1087 | 0.2994 / +0.0327 | 0.1181 / -0.0496 |
| shuffle_e300 | 0.7553 / +0.1870 | 0.5169 / +0.2503 | 0.3459 / +0.1782 |

### occupancy
| arm | stage0 AP / Δ | stage1 AP / Δ | stage2 AP / Δ |
|---|---:|---:|---:|
| coord_only | 0.5789 / -- | 0.6563 / -- | 0.8657 / -- |
| scratch | 0.9272 / +0.3483 | 0.9421 / +0.2858 | 0.9832 / +0.1175 |
| joint_e300 | 0.9905 / +0.4116 | 0.9881 / +0.3317 | 0.9953 / +0.1295 |
| cosine_e300 | 0.9939 / +0.4150 | 0.9930 / +0.3367 | 0.9952 / +0.1295 |
| linear_e300 | 0.9920 / +0.4131 | 0.9911 / +0.3348 | 0.9938 / +0.1280 |
| w05_e300 | 0.9884 / +0.4095 | 0.9826 / +0.3263 | 0.9953 / +0.1295 |
| occupancy_only_e300 | 0.8706 / +0.2917 | 0.7202 / +0.0639 | 0.7868 / -0.0789 |
| shuffle_e300 | 0.9729 / +0.3940 | 0.9734 / +0.3171 | 0.9756 / +0.1099 |

### shell
| arm | stage0 AP / Δ | stage1 AP / Δ | stage2 AP / Δ |
|---|---:|---:|---:|
| coord_only | 0.5539 / -- | 0.6177 / -- | 0.8306 / -- |
| scratch | 0.6598 / +0.1059 | 0.8273 / +0.2096 | 0.9720 / +0.1414 |
| joint_e300 | 0.9425 / +0.3886 | 0.9810 / +0.3633 | 0.9887 / +0.1581 |
| cosine_e300 | 0.9620 / +0.4081 | 0.9866 / +0.3689 | 0.9915 / +0.1609 |
| linear_e300 | 0.9554 / +0.4015 | 0.9842 / +0.3665 | 0.9943 / +0.1636 |
| w05_e300 | 0.9426 / +0.3887 | 0.9729 / +0.3552 | 0.9909 / +0.1603 |
| occupancy_only_e300 | 0.5769 / +0.0230 | 0.6621 / +0.0444 | 0.8782 / +0.0476 |
| shuffle_e300 | 0.7866 / +0.2327 | 0.9239 / +0.3062 | 0.9628 / +0.1322 |

## 100% train scenes
### objectness
| arm | stage0 AP / Δ | stage1 AP / Δ | stage2 AP / Δ |
|---|---:|---:|---:|
| coord_only | 0.6426 / -- | 0.3678 / -- | 0.2615 / -- |
| scratch | 0.7996 / +0.1569 | 0.6174 / +0.2496 | 0.6024 / +0.3409 |
| joint_e300 | 0.8087 / +0.1661 | 0.6391 / +0.2713 | 0.5660 / +0.3045 |
| cosine_e300 | 0.8142 / +0.1716 | 0.6514 / +0.2836 | 0.5180 / +0.2565 |
| linear_e300 | 0.8170 / +0.1743 | 0.6399 / +0.2721 | 0.5496 / +0.2881 |
| w05_e300 | 0.8163 / +0.1737 | 0.5996 / +0.2318 | 0.4041 / +0.1426 |
| occupancy_only_e300 | 0.7802 / +0.1376 | 0.4434 / +0.0756 | 0.1855 / -0.0760 |
| shuffle_e300 | 0.7776 / +0.1350 | 0.5796 / +0.2118 | 0.3874 / +0.1259 |

### occupancy
| arm | stage0 AP / Δ | stage1 AP / Δ | stage2 AP / Δ |
|---|---:|---:|---:|
| coord_only | 0.5775 / -- | 0.6532 / -- | 0.8745 / -- |
| scratch | 0.9498 / +0.3724 | 0.9568 / +0.3036 | 0.9888 / +0.1143 |
| joint_e300 | 0.9924 / +0.4149 | 0.9921 / +0.3388 | 0.9962 / +0.1217 |
| cosine_e300 | 0.9952 / +0.4178 | 0.9945 / +0.3413 | 0.9971 / +0.1226 |
| linear_e300 | 0.9931 / +0.4157 | 0.9935 / +0.3402 | 0.9966 / +0.1221 |
| w05_e300 | 0.9906 / +0.4131 | 0.9902 / +0.3370 | 0.9960 / +0.1215 |
| occupancy_only_e300 | 0.9320 / +0.3545 | 0.8072 / +0.1540 | 0.8857 / +0.0112 |
| shuffle_e300 | 0.9777 / +0.4002 | 0.9785 / +0.3253 | 0.9831 / +0.1086 |

### shell
| arm | stage0 AP / Δ | stage1 AP / Δ | stage2 AP / Δ |
|---|---:|---:|---:|
| coord_only | 0.5598 / -- | 0.6212 / -- | 0.8320 / -- |
| scratch | 0.6608 / +0.1009 | 0.8294 / +0.2082 | 0.9799 / +0.1479 |
| joint_e300 | 0.9492 / +0.3893 | 0.9848 / +0.3636 | 0.9956 / +0.1635 |
| cosine_e300 | 0.9680 / +0.4081 | 0.9893 / +0.3680 | 0.9959 / +0.1639 |
| linear_e300 | 0.9589 / +0.3990 | 0.9869 / +0.3657 | 0.9962 / +0.1641 |
| w05_e300 | 0.9487 / +0.3889 | 0.9830 / +0.3618 | 0.9936 / +0.1616 |
| occupancy_only_e300 | 0.6591 / +0.0992 | 0.7204 / +0.0992 | 0.7591 / -0.0729 |
| shuffle_e300 | 0.8062 / +0.2463 | 0.9424 / +0.3212 | 0.9699 / +0.1379 |

## Key objectness deltas (10%)
- stage0: coord_only 0.5683; top residuals: w05_e300 0.8080 (Δ+0.2398), linear_e300 0.8074 (Δ+0.2391), cosine_e300 0.8029 (Δ+0.2347), joint_e300 0.8017 (Δ+0.2334)
- stage1: coord_only 0.2666; top residuals: cosine_e300 0.6097 (Δ+0.3431), joint_e300 0.5890 (Δ+0.3224), scratch 0.5849 (Δ+0.3183), linear_e300 0.5819 (Δ+0.3153)
- stage2: coord_only 0.1677; top residuals: scratch 0.5677 (Δ+0.3999), cosine_e300 0.5254 (Δ+0.3577), linear_e300 0.5202 (Δ+0.3525), w05_e300 0.4975 (Δ+0.3298)

## Key objectness deltas (100%)
- stage0: coord_only 0.6426; top residuals: linear_e300 0.8170 (Δ+0.1743), w05_e300 0.8163 (Δ+0.1737), cosine_e300 0.8142 (Δ+0.1716), joint_e300 0.8087 (Δ+0.1661)
- stage1: coord_only 0.3678; top residuals: cosine_e300 0.6514 (Δ+0.2836), linear_e300 0.6399 (Δ+0.2721), joint_e300 0.6391 (Δ+0.2713), scratch 0.6174 (Δ+0.2496)
- stage2: coord_only 0.2615; top residuals: scratch 0.6024 (Δ+0.3409), joint_e300 0.5660 (Δ+0.3045), linear_e300 0.5496 (Δ+0.2881), cosine_e300 0.5180 (Δ+0.2565)

## Interpretation

- Coordinate-only carries nonzero objectness signal, especially with 100% train scenes, so coordinate/canonical-room priors are a real confound for absolute readout scores.
- However, coordinate-only does not explain the frozen-feature readout: scratch and pretrained arms remain substantially above coordinate-only for objectness at every stage in both 10% and 100% settings.
- The most defensible positive statement is the stage1 residual: cosine has the largest objectness delta over coordinate-only in both 10% and 100% probes.
- Occupancy/shell readouts are very high for joint/cosine/linear/w0.5 and clearly above coordinate-only, while occupancy-only and shuffle are weaker, especially at deeper stages.
- Do not use `acc_at_logit0` as the primary evidence; stage2 has strong class imbalance and coordinate-only can obtain high accuracy by near-all-negative prediction. Use balanced AP/AUROC.
- This remains a single-run diagnostic. It supports the claim that the probe is not merely an xyz readout, but it should stay in appendix/Discussion rather than becoming a main result.
