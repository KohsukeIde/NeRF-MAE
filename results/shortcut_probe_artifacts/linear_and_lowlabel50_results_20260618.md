# Linear Ramp and 50% Low-Label Results

Snapshot: 2026-06-18 JST.

## Main e300 Ramp Shape

| condition | AP@25 | AP@50 | AP@75 |
|---|---:|---:|---:|
| joint_e300 | 0.7993 ± 0.0114 | 0.4938 ± 0.0289 | 0.0911 ± 0.0040 |
| shuffle_e300 | 0.7438 ± 0.0158 | 0.4294 ± 0.0207 | 0.0453 ± 0.0259 |
| linear_e300_det0 | 0.8039 ± 0.0048 | 0.5528 ± 0.0162 | 0.0984 ± 0.0154 |
| cosine_e300 | 0.8294 ± 0.0074 | 0.5723 ± 0.0195 | 0.1067 ± 0.0154 |

AP@50 deltas:
- linear - joint: +0.0589
- linear - shuffle: +0.1234
- linear - cosine: -0.0196

Reading: linear is clearly above joint/shuffle and close to cosine, supporting ramp-shape robustness. Cosine remains the default because it is slightly higher on AP@50/AP@75 in this run set.

## 50% Label Rows

| condition | AP@25 | AP@50 | AP@75 |
|---|---:|---:|---:|
| scratch_50 | 0.6946 ± 0.0586 | 0.3444 ± 0.0333 | 0.0297 ± 0.0207 |
| nerfmae_joint_e300_50 | 0.7639 ± 0.0069 | 0.4248 ± 0.0510 | 0.0374 ± 0.0115 |
| cosine_e300_50 | 0.7672 ± 0.0141 | 0.4970 ± 0.0062 | 0.0490 ± 0.0049 |
| surface_cosine_jitter_e300_50 | 0.7947 ± 0.0194 | 0.5150 ± 0.0201 | 0.0596 ± 0.0117 |
| scratch_100 | 0.7878 ± 0.0099 | 0.4828 ± 0.0468 | 0.0773 ± 0.0064 |

AP@50 deltas:
- cosine 50% - scratch 50%: +0.1526
- cosine 50% - NeRF-MAE e300 50%: +0.0722
- cosine 50% - scratch 100%: +0.0142
- surface 50% - scratch 100%: +0.0322
