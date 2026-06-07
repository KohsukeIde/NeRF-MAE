# Dither / MixNeRF Branch Decision

Snapshot: 2026-06-07 JST

## Decision

Dither / MixNeRF visible-token filling is closed as a main-paper path.

It remains useful as a diagnostic appendix/future-method note, but it should not
displace the structure-first / budget-curve paper path unless a new mechanism is
specified before running new jobs.

## Evidence

Clean e100 visible-only same-scene dither did not beat the zero-fill control on
mean AP@50:

| condition | n | mean AP@50 | std AP@50 | mean AP@75 | mean R@50 |
|---|---:|---:|---:|---:|---:|
| `shuffle_visible` | 2 | 0.5819 | 0.0076 | 0.1336 | 0.7243 |
| `zero` | 2 | 0.5925 | 0.0477 | 0.1158 | 0.7169 |
| `mean` | 1 | 0.5670 | - | 0.0875 | 0.6912 |

The positive part is narrow:
- `shuffle_visible` is more stable than zero on AP@50.
- `shuffle_visible` has higher mean AP@75 and R@50.

The negative part is decisive for main-method promotion:
- The planned promotion criterion was AP@50 over zero and mean.
- Zero-fill has high variance but a higher two-seed mean.
- Mean-fill is close enough that the result does not isolate a special
  scene-distribution matching mechanism.

## Guardrail Added

The old MixNeRF launchers now require explicit opt-in:

```bash
ALLOW_CLOSED_MIXNERF=1 bash nerf_mae/probe_scripts/submit_mixnerf_dither_e100_scouts.sh
ALLOW_CLOSED_MIXNERF=1 bash nerf_mae/probe_scripts/submit_mixnerf_next_scouts.sh
```

Without `ALLOW_CLOSED_MIXNERF=1`, they exit before submitting jobs.  This avoids
accidental compute spent on a closed branch.

## Remaining Loophole

The dither result does not prove that visible-token/asymmetric encoder-side
methods are hopeless.  It only says that this filler-based MixNeRF variant is
not strong enough.  A cleaner encoder-side branch must first measure whether
masked placeholders materially participate in encoder/skip features.

