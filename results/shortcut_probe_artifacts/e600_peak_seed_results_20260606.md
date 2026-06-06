# e600 Peak Seed Check Results

Snapshot: 2026-06-06 JST

Purpose:
- Verify whether the seed1 budget-curve peak
  (`cosine_e600 AP@50=0.6196 > baseline_e1200 AP@50=0.5648`) survives
  finetune-seed replication.

Queue status:
- The e600 peak seed-check jobs are complete.
- Remaining running jobs are unrelated SSL / dither scout jobs.

## Per-seed results

| condition | finetune seed | AP@25 | AP@50 | AP@75 | R@50 top300 | R@25 top300 |
|---|---:|---:|---:|---:|---:|---:|
| cosine_e600 | 1 | 0.8220 | 0.6196 | 0.0721 | 0.7279 | 0.9412 |
| cosine_e600 | 2 | 0.8220 | 0.4971 | 0.0653 | 0.6397 | 0.9559 |
| cosine_e600 | 3 | 0.7958 | 0.5065 | 0.1030 | 0.6250 | 0.9485 |
| baseline_e600 | 1 | 0.7994 | 0.4994 | 0.0767 | 0.6765 | 0.9485 |
| baseline_e600 | 2 | 0.7838 | 0.4984 | 0.0702 | 0.6838 | 0.9559 |
| baseline_e600 | 3 | 0.7998 | 0.4955 | 0.0793 | 0.6765 | 0.9706 |
| baseline_e1200 | 1 | 0.7934 | 0.5648 | 0.0809 | 0.7059 | 0.9485 |
| baseline_e1200 | 2 | 0.7775 | 0.5087 | 0.0696 | 0.6471 | 0.9412 |
| baseline_e1200 | 3 | 0.7706 | 0.4807 | 0.0837 | 0.6176 | 0.9632 |

## Summary

| condition | AP@25 mean±std | AP@50 mean±std | AP@75 mean±std | R@50 mean±std | R@25 mean±std |
|---|---:|---:|---:|---:|---:|
| cosine_e600 | 0.8133±0.0151 | 0.5410±0.0682 | 0.0801±0.0201 | 0.6642±0.0557 | 0.9485±0.0074 |
| baseline_e600 | 0.7943±0.0091 | 0.4978±0.0021 | 0.0754±0.0047 | 0.6789±0.0042 | 0.9583±0.0112 |
| baseline_e1200 | 0.7805±0.0117 | 0.5181±0.0428 | 0.0781±0.0074 | 0.6569±0.0449 | 0.9510±0.0112 |

## Paired AP@50 differences

| finetune seed | cosine_e600 - baseline_e600 | cosine_e600 - baseline_e1200 |
|---:|---:|---:|
| 1 | +0.1202 | +0.0548 |
| 2 | -0.0014 | -0.0117 |
| 3 | +0.0110 | +0.0257 |
| mean | +0.0432 | +0.0229 |

## Interpretation

- The seed1 e600 peak does not replicate strongly.
- `cosine_e600` remains above `baseline_e600` and `baseline_e1200` in mean AP@50,
  but the margin is driven heavily by seed1 and has high variance.
- `cosine_e600` improves AP@25 mean over both baselines, but R@50 is not clearly
  better than `baseline_e600`.
- This is not strong enough to support a headline claim that e600 robustly beats
  e1200 baseline.
- Safer paper framing:
  - keep the single-seed budget curve as an efficiency observation;
  - report critical-point seed bands;
  - frame e600 as a promising but variable mid-budget optimum;
  - avoid "strong/decisive e600 peak" language.

Decision:
- Downgrade the e600 peak from a strong main pillar to a qualified efficiency
  result.
- The paper should lean more on the combined package: AP@25/AP@50 early transfer,
  low-label stability, compute-normalized protocol, and ablations.

