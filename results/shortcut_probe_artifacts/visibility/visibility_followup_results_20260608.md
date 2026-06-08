# Visibility Follow-up Results

Snapshot:
- 2026-06-08 JST

## Jobs

| condition | seed | pretrain job | FCOS job | status |
|---|---:|---|---|---|
| `visibility_skip_gate` | 1 | `1833071.pbs1` | `1833072.pbs1` | complete |
| `visibility_cosine_skip_gate` | 2 | `1833073.pbs1` | `1833074.pbs1` | complete |

Important naming correction:
- `visibility_skip_gate` in this artifact means pure skip-gate with
  `PROBE_CURRICULUM=none`.
- The earlier 2026-06-07 result named `visibility_skip_gate` used
  `PROBE_CURRICULUM=cosine_rgb_ramp`, so it is treated here as
  `visibility_cosine_skip_gate` seed1.

## Completed Results

| condition | seed | AP@25 | AP@50 | AP@75 | R@50 top300 | R@25 top300 | AR top300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `visibility_skip_gate` pure | 1 | 0.7844 | 0.5480 | 0.0687 | 0.6912 | 0.9559 | 0.4775 |
| `visibility_cosine_skip_gate` | 1 | 0.8173 | 0.5869 | 0.1039 | 0.7279 | 0.9632 | 0.4971 |
| `visibility_cosine_skip_gate` | 2 | 0.7931 | 0.5492 | 0.0787 | 0.7353 | 0.9485 | 0.4966 |

Reference context:

| condition | n | AP@50 mean | AP@50 std | AP@75 mean | R@50 mean | AR mean |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_coord_jitter_e100` | 3 | 0.5454 | 0.0103 | 0.1073 | 0.6912 | 0.4953 |
| `cosine_coord_jitter_e100` | 3 | 0.5873 | 0.0395 | 0.0872 | 0.7181 | 0.4897 |
| `visibility_cosine_skip_gate_e100` | 2 | 0.5681 | 0.0267 | 0.0913 | 0.7316 | 0.4968 |

## Interpretation

- Pure `visibility_skip_gate` does not improve over `baseline_coord_jitter_e100`
  in a meaningful way. AP@50 is 0.5480 vs baseline mean 0.5454, while AP@75
  and AR are lower.
- `visibility_cosine_skip_gate` is not stable enough to promote. Seed1 looked
  promising at AP@50 0.5869, but seed2 dropped to 0.5492. The 2-seed AP@50 mean
  is 0.5681, below `cosine_coord_jitter_e100` mean 0.5873.
- The masked-skip diagnostic remains useful mechanism evidence, but the
  intervention itself does not currently produce a stronger downstream method.

Decision:
- Do not promote visibility gating to the AAAI main method path.
- Do not launch attention KV-gating from these results.
- Stop visibility method exploration unless a new reviewer-facing mechanism is
  specified before running jobs.
- Keep the diagnostic and negative result as appendix/future-method evidence.
