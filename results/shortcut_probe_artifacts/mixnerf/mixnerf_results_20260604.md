# MixNeRF-MAE-lite scout results

Snapshot: 2026-06-04 JST

## FCOS eval results

| condition | pretrain | AP@25 | AP@50 | AP@75 | R@50 top300 | note |
|---|---:|---:|---:|---:|---:|---|
| MixNeRF partner-fill | e30 | 0.8486 | 0.5433 | 0.1112 | 0.6765 | first partner scout |
| MixNeRF partner-fill | e100 | 0.8125 | 0.5871 | 0.0670 | 0.7206 | completed e100 scout |
| zero-fill control | e30 | 0.8271 | 0.5292 | 0.1675 | 0.6544 | control |
| noise-fill control | e30 | 0.8259 | 0.5894 | 0.1000 | 0.7206 | control; matches/exceeds MixNeRF e100 AP@50 |

## Interpretation

- MixNeRF partner-fill does improve from e30 to e100 on AP@50
  (`0.5433 -> 0.5871`), but the e100 AP@75 drops to `0.0670`.
- The e30 noise-fill control reaches AP@50 `0.5894`, slightly above MixNeRF e100
  `0.5871`, with the same R@50 top300 `0.7206`.
- Therefore the current MixNeRF-lite result does not isolate a useful partner-token
  mechanism. It is not strong enough to displace the current budget-curve /
  structure-first paper path.
- Keep MixNeRF / visible-token mechanisms conditional only if later budget-curve
  results are weak or if a stricter control/mask-predictability analysis motivates
  a redesigned version.
