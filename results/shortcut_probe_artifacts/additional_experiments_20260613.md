# Additional Experiments Submitted on 2026-06-13

Purpose:
- Support the label-efficiency-first paper framing with multi-seed low-label rows.
- Add a clean appearance-first reverse control for the `order matters` claim.

## Low-Label Multi-Seed Expansion

Submitted FCOS-only jobs for labels `{10%, 25%}`, conditions `{scratch, baseline_e300, cosine_ramp_e300}`, and finetune seeds `{2, 3}`.

Existing seed-1 rows are from `lowlabel_expansion_jobs_20260601.md`.

CSV log:
- `results/shortcut_probe_artifacts/lowlabel_multiseed_jobs_20260613.csv`

Design:
- Pretrain checkpoints are fixed to seed-1 e300 checkpoints for `baseline_e300` and `cosine_ramp_e300`.
- Only FCOS finetune seed varies, matching the current paper-scale seed protocol for low-label tables.

## Appearance-First Reverse Control

Submitted a clean e300 reverse ramp control without coord-jitter:

| job | stage | condition | notes |
|---|---|---|---|
| `1909302.pbs1` | pretrain | `reverse_ramp` | `RUN_SUFFIX=abci3reverse`, e300, seed1 |
| `1909303.pbs1` | FCOS | `reverse_ramp` | depends on `afterok:1909302.pbs1` |

Implementation note:
- `reverse_ramp` was added to the ABCI3 e300 gate PBS wrappers as a clean, no coord-jitter counterpart to the existing `reverse_ramp_coord_jitter`.
- It uses the same probe objective as `cosine_ramp`, but with `PROBE_CURRICULUM_RGB_START_WEIGHT=1.0` and `PROBE_CURRICULUM_RGB_END_WEIGHT=0.0`.

Decision rule:
- If `reverse_ramp_e300` is clearly below `cosine_ramp_e300`, the paper can state that structure-to-appearance order matters.
- If it is comparable, the order claim must be softened to a budget-aware empirical property rather than a causal mechanism.
