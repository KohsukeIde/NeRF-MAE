# Anchor-head 25%/50% label rerun results (2026-06-24)
## Protocol
- Detector: Anchor-RPN head.
- Dataset: Front3D OBB, 25% and 50% training labels.
- Finetune: 200 epochs, batch size 8 on one GPU, AP@50-best checkpoint.
- Backbone: MAE-compatible Swin FPN architecture for all arms.
- Conditions: scratch, joint e300, cosine e300.
- Seeds: 1, 2, 3.
- `DETERMINISTIC=0`.

## Seed-wise Results: 25% labels

| condition | seed | AP@25 | AP@50 | AP@75 | R@25 | R@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scratch | 1 | 0.2078 | 0.0183 | 0.0026 | 0.9338 | 0.2794 |
| scratch | 2 | 0.2140 | 0.0252 | 0.0000 | 0.9412 | 0.3015 |
| scratch | 3 | 0.2110 | 0.0150 | 0.0001 | 0.9412 | 0.3309 |
| joint e300 | 1 | 0.1121 | 0.0079 | 0.0000 | 0.9632 | 0.2721 |
| joint e300 | 2 | 0.0756 | 0.0020 | 0.0000 | 0.9412 | 0.2059 |
| joint e300 | 3 | 0.0760 | 0.0058 | 0.0000 | 0.9118 | 0.2794 |
| cosine e300 | 1 | 0.1984 | 0.0152 | 0.0000 | 0.9412 | 0.3162 |
| cosine e300 | 2 | 0.1415 | 0.0038 | 0.0000 | 0.9485 | 0.2574 |
| cosine e300 | 3 | 0.1391 | 0.0167 | 0.0000 | 0.9412 | 0.2500 |

## Seed-wise Results: 50% labels

| condition | seed | AP@25 | AP@50 | AP@75 | R@25 | R@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scratch | 1 | 0.4276 | 0.1107 | 0.0025 | 0.9559 | 0.4485 |
| scratch | 2 | 0.4344 | 0.1495 | 0.0001 | 0.9632 | 0.3971 |
| scratch | 3 | 0.4093 | 0.1764 | 0.0000 | 0.9485 | 0.4706 |
| joint e300 | 1 | 0.2816 | 0.0949 | 0.0000 | 0.9559 | 0.3897 |
| joint e300 | 2 | 0.2871 | 0.0486 | 0.0002 | 0.9338 | 0.3309 |
| joint e300 | 3 | 0.1796 | 0.0320 | 0.0000 | 0.9412 | 0.3088 |
| cosine e300 | 1 | 0.4021 | 0.0897 | 0.0001 | 0.9632 | 0.4265 |
| cosine e300 | 2 | 0.3081 | 0.0842 | 0.0006 | 0.9559 | 0.4191 |
| cosine e300 | 3 | 0.3866 | 0.1221 | 0.0001 | 0.9559 | 0.4853 |

## Summary

| labels | condition | AP@25 | AP@50 | AP@75 | R@25 | R@50 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 25% | scratch | 0.2109 ± 0.0031 | 0.0195 ± 0.0052 | 0.0009 ± 0.0015 | 0.9387 ± 0.0042 | 0.3039 ± 0.0258 |
| 25% | joint e300 | 0.0879 ± 0.0209 | 0.0052 ± 0.0030 | 0.0000 ± 0.0000 | 0.9387 ± 0.0258 | 0.2525 ± 0.0405 |
| 25% | cosine e300 | 0.1597 ± 0.0336 | 0.0119 ± 0.0071 | 0.0000 ± 0.0000 | 0.9436 ± 0.0042 | 0.2745 ± 0.0363 |
| 50% | scratch | 0.4238 ± 0.0130 | 0.1456 ± 0.0330 | 0.0009 ± 0.0014 | 0.9559 ± 0.0074 | 0.4387 ± 0.0377 |
| 50% | joint e300 | 0.2495 ± 0.0605 | 0.0585 ± 0.0326 | 0.0001 ± 0.0001 | 0.9436 ± 0.0112 | 0.3431 ± 0.0418 |
| 50% | cosine e300 | 0.3656 ± 0.0504 | 0.0987 ± 0.0205 | 0.0002 ± 0.0003 | 0.9583 ± 0.0042 | 0.4436 ± 0.0363 |

## Interpretation

Unlike the 10% Anchor-RPN setting, 25% and 50% labels are above the AP@50 floor. However, the detector-head breadth trend is not positive for structure-first pretraining: scratch remains the strongest arm at both 25% and 50% labels under this corrected Anchor-RPN protocol. Cosine e300 improves over joint e300 at 25% AP@50 but does not beat scratch; at 50%, cosine e300 is below both scratch and joint e300 on AP@50.

This should not be used as supporting detector-head breadth evidence. It is a valid negative/limitation result for Anchor-RPN breadth under the public Anchor-RPN head, after full-label sanity and MAE-compatible backbone checks.
