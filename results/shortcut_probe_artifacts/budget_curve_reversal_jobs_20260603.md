# Budget Curve and Reversal Seed Jobs

Snapshot:
- 2026-06-03 JST

Purpose:
- Follow the final AAAI paper direction: use the budget curve as the primary
  efficiency figure and defend the thin 50%-label reversal with extra finetune
  seeds.
- Prioritize the budget curve as the current AAAI critical path, but do not
  permanently exclude MixNeRF / visible-token ideas. If the MixNeRF scout is
  clearly stronger, or if the budget curve is not strong enough by itself,
  revisit whether the masking-mechanism branch should be folded into the paper.

## Budget Curve

Corrected curve plan:
- Each budget point must use a training run whose total epoch count matches the
  plotted budget. This matters because both the cosine RGB curriculum and the
  one-cycle learning-rate schedule depend on total `EPOCHS`.
- The initially submitted e1200 intermediate-checkpoint FCOS jobs for e100/e300/e600
  were therefore cancelled. The e1200 pretrains remain valid for the e1200 point.
- Dedicated e100/e600 pretrains were added; existing dedicated e300/e600
  checkpoints are reused where available.

Pretrain jobs:

| condition | save suffix | job | status |
|---|---|---:|
| baseline e1200 | `abci3budgetcurve50` | `1821253.pbs1` | keep for e1200 only |
| cosine_ramp e1200 | `abci3budgetcurve50` | `1821254.pbs1` | keep for e1200 only |
| baseline e100 | `abci3budgetB` | `1821358.pbs1` | added after schedule correction |
| baseline e600 | `abci3budgetB` | `1821360.pbs1` | added after schedule correction |
| cosine_ramp e100 | `abci3budgetB` | `1821362.pbs1` | added after schedule correction |

Dependent FCOS jobs:

| condition | budget epoch | job | status |
|---|---:|---:|
| baseline | 100 | `1821359.pbs1` | dependent on dedicated e100 |
| baseline | 300 | existing eval | dedicated e300 already available |
| baseline | 600 | `1821361.pbs1` | dependent on dedicated e600 |
| baseline | 1200 | `1821258.pbs1` | dependent on e1200 |
| cosine_ramp | 100 | `1821363.pbs1` | dependent on dedicated e100 |
| cosine_ramp | 300 | existing eval | dedicated e300 already available |
| cosine_ramp | 600 | `1821364.pbs1` | FCOS-only on existing dedicated e600 |
| cosine_ramp | 1200 | `1821262.pbs1` | dependent on e1200 |

Cancelled after schedule correction:
- `1821255.pbs1`, `1821256.pbs1`, `1821257.pbs1`
- `1821259.pbs1`, `1821260.pbs1`, `1821261.pbs1`

## Reversal Seed Defense

Headline to defend:

```text
50% labels with structure-first pretraining > 100% labels scratch
```

Existing seed1 rows:
- `front3d_scratch_lowlabel_pt100_seed1_fcos1000`
- `cosine_ramp_e300_lowlabel_pt05_seed1_fcos1000`
- `surface_cosine_jitter_e300_lowlabel_pt05_seed1_fcos1000`

Added FCOS-only seed2/seed3 jobs:

| row | seed | job |
|---|---:|---:|
| scratch 100% | 2 | `1821264.pbs1` |
| cosine_ramp e300 50% | 2 | `1821265.pbs1` |
| surface_cosine_jitter e300 50% | 2 | `1821266.pbs1` |
| scratch 100% | 3 | `1821267.pbs1` |
| cosine_ramp e300 50% | 3 | `1821268.pbs1` |
| surface_cosine_jitter e300 50% | 3 | `1821269.pbs1` |

Decision rules:
- If cosine 50% beats scratch 100% in mean AP@50 with at least 2/3 paired
  finetune seeds, keep the reversal as a headline.
- If cosine 50% is thin but surface 50% is robust, use cosine as the base method
  and surface anchoring as the label-richer enhancement.
- If both collapse against scratch 100%, demote the reversal and use the budget
  curve as the main claim.
