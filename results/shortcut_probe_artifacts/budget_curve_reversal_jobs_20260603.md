# Budget Curve and Reversal Seed Jobs

Snapshot:
- 2026-06-03 JST

Purpose:
- Follow the final AAAI paper direction: use the budget curve as the primary
  efficiency figure and defend the thin 50%-label reversal with extra finetune
  seeds.
- Keep MixNeRF / visible-token ideas out of the AAAI critical path unless they
  independently clear their own scout gate.

## Budget Curve

Clean curve plan:
- Train one baseline e1200 run and one structure-first/cosine e1200 run with
  `PRETRAIN_CHECKPOINT_INTERVAL=50`.
- Evaluate checkpoints at epochs 100, 300, 600, and 1200.
- This avoids separate e100/e300/e600 pretrains and gives a defensible
  single-trajectory budget curve.

Pretrain jobs:

| condition | save suffix | job |
|---|---|---:|
| baseline | `abci3budgetcurve50` | `1821253.pbs1` |
| cosine_ramp | `abci3budgetcurve50` | `1821254.pbs1` |

Dependent FCOS jobs:

| condition | checkpoint epoch | job |
|---|---:|---:|
| baseline | 100 | `1821255.pbs1` |
| baseline | 300 | `1821256.pbs1` |
| baseline | 600 | `1821257.pbs1` |
| baseline | 1200 | `1821258.pbs1` |
| cosine_ramp | 100 | `1821259.pbs1` |
| cosine_ramp | 300 | `1821260.pbs1` |
| cosine_ramp | 600 | `1821261.pbs1` |
| cosine_ramp | 1200 | `1821262.pbs1` |

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
