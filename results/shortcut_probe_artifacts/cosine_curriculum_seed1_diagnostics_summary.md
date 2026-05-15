# Diagnostic Summary

| label | AP@50 | AP@75 | top300 mean IoU | frac IoU>=0.5 | TP score mean | FP score mean | first TP rank | top50 TP rate | voxel peakiness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_e300 | 0.4903 | 0.1009 | 0.0595 | 0.0178 | 0.5955 | 0.0373 | 1.25 | 0.1000 | 14.7951 |
| cosine_e300 | 0.5987 | 0.1061 | 0.0591 | 0.0188 | 0.6060 | 0.0475 | 1.18 | 0.1106 | 12.0121 |
| shuffle_e300 | 0.4137 | 0.0597 | 0.0603 | 0.0165 | 0.5246 | 0.0400 | 3.00 | 0.0965 | 13.5157 |
| baseline_e1200 | 0.5892 | 0.1469 | 0.0523 | 0.0190 | 0.6672 | 0.0118 | 1.12 | 0.1129 | 19.4645 |

## baseline_e300

- Eval: AP@50=0.4903, AP@25=0.7819, AP@75=0.1009, Recall@50 top300=0.6618, AR top300=0.4799
- Proposal: mean IoU=0.0595, median IoU=0.0000, p90 IoU=0.1986, frac IoU>=0.25/0.5/0.75=0.0635/0.0178/0.0057
- Ranking: TP score mean=0.5955, FP score mean=0.0373, first TP rank mean=1.25, top50/top100/top300 TP rate=0.1000/0.0524/0.0178
- Level share: L0=0.5767, L1=0.2753, L2=0.1069, L3=0.0412
- TP level share: L0=0.2747, L1=0.4615, L2=0.2637, L3=0.0000
- Voxel sharpness: peakiness=14.7951, max/mean=88.6766, std=0.0521

## cosine_e300

- Eval: AP@50=0.5987, AP@25=0.8443, AP@75=0.1061, Recall@50 top300=0.7059, AR top300=0.4892
- Proposal: mean IoU=0.0591, median IoU=0.0017, p90 IoU=0.1909, frac IoU>=0.25/0.5/0.75=0.0606/0.0188/0.0063
- Ranking: TP score mean=0.6060, FP score mean=0.0475, first TP rank mean=1.18, top50/top100/top300 TP rate=0.1106/0.0559/0.0188
- Level share: L0=0.5825, L1=0.2920, L2=0.0886, L3=0.0369
- TP level share: L0=0.1875, L1=0.5417, L2=0.2708, L3=0.0000
- Voxel sharpness: peakiness=12.0121, max/mean=51.1770, std=0.0502

## shuffle_e300

- Eval: AP@50=0.4137, AP@25=0.7185, AP@75=0.0597, Recall@50 top300=0.6029, AR top300=0.4431
- Proposal: mean IoU=0.0603, median IoU=0.0000, p90 IoU=0.1985, frac IoU>=0.25/0.5/0.75=0.0669/0.0165/0.0039
- Ranking: TP score mean=0.5246, FP score mean=0.0400, first TP rank mean=3.00, top50/top100/top300 TP rate=0.0965/0.0488/0.0165
- Level share: L0=0.5282, L1=0.3051, L2=0.1257, L3=0.0410
- TP level share: L0=0.2381, L1=0.4762, L2=0.2857, L3=0.0000
- Voxel sharpness: peakiness=13.5157, max/mean=83.6087, std=0.0495

## baseline_e1200

- Eval: AP@50=0.5892, AP@25=0.8494, AP@75=0.1469, Recall@50 top300=0.7132, AR top300=0.5093
- Proposal: mean IoU=0.0523, median IoU=0.0000, p90 IoU=0.1659, frac IoU>=0.25/0.5/0.75=0.0537/0.0190/0.0076
- Ranking: TP score mean=0.6672, FP score mean=0.0118, first TP rank mean=1.12, top50/top100/top300 TP rate=0.1129/0.0565/0.0190
- Level share: L0=0.5239, L1=0.3382, L2=0.0973, L3=0.0406
- TP level share: L0=0.2062, L1=0.5361, L2=0.2577, L3=0.0000
- Voxel sharpness: peakiness=19.4645, max/mean=233.6821, std=0.0491
