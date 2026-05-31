# Alpha Boundary Target Audit

Rows: 60

| threshold | scenes | occ ratio mean | shell/occ mean | components median | dist p90 mean | grad p95 mean |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 20 | 0.314632 | 0.6869 | 440.5 | 13.575 | 0.353083 |
| 0.05 | 20 | 0.283159 | 0.7476 | 502.0 | 13.865 | 0.353083 |
| 0.1 | 20 | 0.220480 | 0.8227 | 736.5 | 14.427 | 0.353083 |

Interpretation guide:
- Very high component counts indicate noisy alpha topology.
- Very high shell/occupied ratio means occupancy is thin/sparse; SDF may mostly encode shell proximity.
- If distance maps are smooth around coherent surfaces in the PNGs, SDF targets are plausible.
- If alpha slices are fragmented or threshold-sensitive, prefer shell/normal diagnostics before SDF pretraining.
