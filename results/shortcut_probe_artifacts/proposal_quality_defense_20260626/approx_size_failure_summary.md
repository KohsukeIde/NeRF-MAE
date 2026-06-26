# Approximate Size/Fails Diagnostics (2026-06-26)

CPU diagnostic using approximate oriented 3D IoU. This is not a replacement for official AP; use only as reviewer-defense / failure analysis. Size bins are GT-volume tertiles over evaluated scenes.

## front3d_p10_seed1
size volume edges: 10273.262, 27947.673
| arm | small AP50 | med AP50 | large AP50 | small AP75 | med AP75 | large AP75 | small R50@300 | med R50@300 | large R50@300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scratch | 0.015 | 0.023 | 0.156 | 0.000 | 0.000 | 0.000 | 0.323 | 0.167 | 0.447 |
| joint_e300 | 0.021 | 0.052 | 0.117 | 0.000 | 0.000 | 0.000 | 0.447 | 0.531 | 0.348 |
| structure_first | 0.016 | 0.168 | 0.237 | 0.000 | 0.002 | 0.039 | 0.342 | 0.557 | 0.550 |

- scratch score-IoU bins: Q0: score=0.145, IoU=0.103, frac50=0.035; Q1: score=0.046, IoU=0.034, frac50=0.000; Q2: score=0.034, IoU=0.030, frac50=0.002; Q3: score=0.028, IoU=0.023, frac50=0.000; Q4: score=0.022, IoU=0.021, frac50=0.000
- joint_e300 score-IoU bins: Q0: score=0.273, IoU=0.146, frac50=0.046; Q1: score=0.128, IoU=0.057, frac50=0.002; Q2: score=0.095, IoU=0.040, frac50=0.001; Q3: score=0.079, IoU=0.031, frac50=0.000; Q4: score=0.066, IoU=0.025, frac50=0.002
- structure_first score-IoU bins: Q0: score=0.185, IoU=0.137, frac50=0.057; Q1: score=0.054, IoU=0.047, frac50=0.002; Q2: score=0.037, IoU=0.030, frac50=0.001; Q3: score=0.029, IoU=0.024, frac50=0.000; Q4: score=0.022, IoU=0.024, frac50=0.000

## scannet_p10_seed1
size volume edges: 2873.644, 7581.638
| arm | small AP50 | med AP50 | large AP50 | small AP75 | med AP75 | large AP75 | small R50@300 | med R50@300 | large R50@300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scratch | 0.020 | 0.094 | 0.075 | 0.000 | 0.005 | 0.000 | 0.287 | 0.383 | 0.401 |
| joint_e300 | 0.056 | 0.089 | 0.097 | 0.000 | 0.000 | 0.000 | 0.352 | 0.296 | 0.337 |
| structure_first | 0.041 | 0.074 | 0.092 | 0.000 | 0.001 | 0.001 | 0.349 | 0.356 | 0.417 |

- scratch score-IoU bins: Q0: score=0.235, IoU=0.162, frac50=0.069; Q1: score=0.077, IoU=0.076, frac50=0.006; Q2: score=0.056, IoU=0.060, frac50=0.002; Q3: score=0.046, IoU=0.053, frac50=0.003; Q4: score=0.037, IoU=0.039, frac50=0.001
- joint_e300 score-IoU bins: Q0: score=0.267, IoU=0.168, frac50=0.074; Q1: score=0.111, IoU=0.077, frac50=0.002; Q2: score=0.081, IoU=0.059, frac50=0.000; Q3: score=0.064, IoU=0.057, frac50=0.001; Q4: score=0.051, IoU=0.035, frac50=0.000
- structure_first score-IoU bins: Q0: score=0.259, IoU=0.173, frac50=0.074; Q1: score=0.095, IoU=0.084, frac50=0.006; Q2: score=0.068, IoU=0.062, frac50=0.003; Q3: score=0.054, IoU=0.048, frac50=0.000; Q4: score=0.041, IoU=0.037, frac50=0.002
