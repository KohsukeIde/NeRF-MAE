# Reverse Ramp e300 Results

Snapshot: 2026-06-15 JST

Purpose:
- Test whether the structure-to-appearance ordering is necessary by reversing the RGB schedule.
- Clean control: `reverse_ramp`, e300, no coord-jitter, seed1.

## Jobs

| job | stage | status |
|---|---|---|
| `1909302.pbs1` | pretrain | canceled; accidentally used `DETERMINISTIC=1` and was too slow |
| `1909303.pbs1` | FCOS | canceled with `1909302.pbs1` |
| `1912406.pbs1` | pretrain | complete |
| `1912407.pbs1` | FCOS | complete |

Active pretrain:
- `output/nerf_mae/results/nerfmae_alpha_rgba_curr_reverse_ramp_p1.0_e300_seed1_abci3reverse_det0/epoch_300.pt`

Eval:
- `output/nerf_rpn/results/nerfmae_alpha_rgba_curr_reverse_ramp_p1.0_e300_seed1_abci3reverse_det0_epoch300_sched_epoch_seed1_fcos1000_eval/eval.json`

## Metrics

| condition | AP@25 | AP@50 | AP@75 | Recall@25 top300 | Recall@50 top300 |
|---|---:|---:|---:|---:|---:|
| baseline_e300 | 0.7956 | 0.4695 | 0.0869 | 0.9559 | 0.6618 |
| cosine_ramp_e300 | 0.8249 | 0.5539 | 0.1135 | 0.9632 | 0.7059 |
| reverse_ramp_e300 | 0.7718 | 0.5706 | 0.0924 | 0.9412 | 0.7132 |

Notes:
- `baseline_e300` and `cosine_ramp_e300` use the current B-generation e300 rows.
- `reverse_ramp_e300` is the clean no-jitter appearance-first control.

## Reading

This does **not** support a strong causal claim that alpha-to-RGBA ordering is necessary for AP@50.

Observed:
- Reverse is **higher** than cosine on AP@50: `0.5706` vs `0.5539`.
- Reverse is lower on AP@25 and AP@75: `0.7718` vs `0.8249`, and `0.0924` vs `0.1135`.
- Recall@50 is slightly higher for reverse: `0.7132` vs `0.7059`.

Safe paper implication:
- Do not claim "order matters" as a primary mechanism from this control.
- The stronger supported claim is budget-aware/surface-structured supervision, with ordering treated as an empirical design choice rather than a proven causal factor.
- If this control is included, frame it as a limitation/nuance: AP@50 can be improved by reverse scheduling, but fine/localization metrics do not follow the same direction.
