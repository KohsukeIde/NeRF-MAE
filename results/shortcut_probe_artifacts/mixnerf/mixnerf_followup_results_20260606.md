# MixNeRF Follow-up Results

Snapshot: 2026-06-06 JST

Current queue status:
- No MixNeRF follow-up jobs are still queued/running.
- Remaining running jobs are unrelated SSL jobs (`simclrv1`, `simclrv2`,
  `byol`, `ibot`).

## FCOS eval results

| condition | objective | epochs | AP@25 | AP@50 | AP@75 | R@50 top300 | R@25 top300 |
|---|---|---:|---:|---:|---:|---:|---:|
| MixNeRF partner-fill | masked RGB (`removed_occupied`) | 30 | 0.8408 | 0.5567 | 0.1361 | 0.6765 | 0.9632 |
| zero-fill control | masked RGB (`removed_occupied`) | 30 | 0.8127 | 0.4881 | 0.0499 | 0.6765 | 0.9412 |
| noise-fill control | masked RGB (`removed_occupied`) | 30 | 0.8136 | 0.5276 | 0.1808 | 0.6765 | 0.9485 |
| same-scene shuffle-fill control | masked RGB (`removed_occupied`) | 30 | 0.8337 | 0.5805 | 0.1239 | 0.6838 | 0.9412 |
| MixNeRF partner-fill | public occupied RGB | 30 | 0.8486 | 0.5433 | 0.1112 | 0.6765 | 0.9706 |
| MixNeRF partner-fill | public occupied RGB | 100 | 0.8125 | 0.5871 | 0.0670 | 0.7206 | 0.9632 |
| zero-fill control | public occupied RGB | 30 | 0.8271 | 0.5292 | 0.1675 | 0.6544 | 0.9559 |
| zero-fill control | public occupied RGB | 100 | 0.8398 | 0.5459 | 0.0772 | 0.6838 | 0.9559 |
| noise-fill control | public occupied RGB | 30 | 0.8259 | 0.5894 | 0.1000 | 0.7206 | 0.9632 |
| noise-fill control | public occupied RGB | 100 | 0.8197 | 0.4909 | 0.0642 | 0.6544 | 0.9485 |

## Interpretation

- Under the stricter masked-RGB objective, same-scene shuffle-fill has the best
  AP@50 (`0.5805`), ahead of partner-fill (`0.5567`).
- This does not validate a cross-scene MixNeRF partner-token semantic mechanism.
  The stronger signal is closer to a non-zero filler / same-scene dithered
  masking control.
- The earlier public-objective noise-fill e30 result (`0.5894`) does not scale
  to e100 (`0.4909`), so that result should not be treated as robust.
- MixNeRF / visible-token filling should stay separated from the main paper
  path unless a future, more targeted mechanism beats the budget-curve results.

