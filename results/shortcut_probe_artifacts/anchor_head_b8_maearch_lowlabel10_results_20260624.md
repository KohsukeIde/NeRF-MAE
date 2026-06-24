# Anchor-head 10% label rerun results (2026-06-24)

## Protocol

This rerun supersedes the invalid `anchor_head_lowlabel_results_20260624.*`
table.

- Detector: Anchor-RPN head.
- Dataset: Front3D OBB, 10% training labels.
- Finetune: 200 epochs, batch size 8 on one GPU, AP@50-best checkpoint.
- Backbone: MAE-compatible Swin FPN architecture for all arms.
- Seeds: 1, 2, 3.
- `DETERMINISTIC=0`.

Full-label sanity passed before this rerun:

| setup | AP@25 | AP@50 | AP@75 |
| --- | ---: | ---: | ---: |
| official scratch, batch8 | 0.5346 | 0.2845 | 0.0024 |
| MAE-compatible scratch, batch8 | 0.5699 | 0.3210 | 0.0023 |

## Seed-wise Results

| condition | seed | AP@25 | AP@50 | AP@75 | R@25 | R@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scratch | 1 | 0.0821 | 0.0096 | 0.0000 | 0.7647 | 0.1691 |
| scratch | 2 | 0.0728 | 0.0010 | 0.0000 | 0.7132 | 0.1324 |
| scratch | 3 | 0.1525 | 0.0111 | 0.0000 | 0.7721 | 0.1397 |
| joint e300 | 1 | 0.0800 | 0.0058 | 0.0000 | 0.8529 | 0.1544 |
| joint e300 | 2 | 0.0315 | 0.0012 | 0.0000 | 0.6985 | 0.1618 |
| joint e300 | 3 | 0.1209 | 0.0056 | 0.0000 | 0.8456 | 0.1471 |
| cosine e300 | 1 | 0.0558 | 0.0008 | 0.0000 | 0.8088 | 0.1103 |
| cosine e300 | 2 | 0.0922 | 0.0068 | 0.0000 | 0.7574 | 0.1103 |
| cosine e300 | 3 | 0.0789 | 0.0036 | 0.0000 | 0.8309 | 0.1471 |

## Summary

| condition | AP@25 | AP@50 | AP@75 | R@25 | R@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| scratch | 0.1025 ± 0.0436 | 0.0072 ± 0.0055 | 0.0000 ± 0.0000 | 0.7500 ± 0.0321 | 0.1471 ± 0.0195 |
| joint e300 | 0.0775 ± 0.0448 | 0.0042 ± 0.0026 | 0.0000 ± 0.0000 | 0.7990 ± 0.0871 | 0.1544 ± 0.0074 |
| cosine e300 | 0.0756 ± 0.0184 | 0.0038 ± 0.0030 | 0.0000 ± 0.0000 | 0.7990 ± 0.0377 | 0.1225 ± 0.0212 |

## Interpretation

The corrected Anchor-RPN pipeline is not globally broken: full-label scratch
sanity reaches AP@50 around 0.28-0.32. However, under the 10% label setting, the
anchor head remains near the AP@50 noise floor for all arms. Cosine/structure
first does not outperform scratch or joint under this detector-head/label
budget.

This result should **not** be used as positive detector-head breadth evidence.
The defensible paper treatment is:

- Do not include Anchor-RPN 10% as a supporting breadth table.
- If mentioned, place it in limitations/appendix as an attempted detector-head
  breadth check where the public Anchor-RPN head is not label-efficient at 10%.
- Keep the main claim scoped to the FCOS/NeRF-RPN detector used by the released
  NeRF-MAE transfer pipeline.
