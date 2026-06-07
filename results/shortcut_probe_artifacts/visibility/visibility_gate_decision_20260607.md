# Visibility-Gated Gate Decision

Snapshot: 2026-06-07 JST

Measurement artifact:
- `results/shortcut_probe_artifacts/visibility/encoder_mask_participation_20260607_141908_thr001/encoder_mask_participation_report.md`

## Result

The non-invasive participation gate is positive.

Masked-placeholder features retain substantial norm in the current dense-grid
encoder and decoder skip path.  This satisfies the pre-defined go criterion for
trying low-risk Visibility-Gated V0/V1 scouts.

## Key Numbers

Masked / visible feature-norm ratio:

| checkpoint | stage0 | stage1 | stage2 |
|---|---:|---:|---:|
| `baseline_e300` | 0.7578 | 0.6989 | 0.6995 |
| `cosine_ramp_e300` | 0.7303 | 0.7463 | 0.7872 |
| `cosine_coord_jitter_e100` | 0.6071 | 0.7066 | 0.5134 |
| `dither_shuffle_visible_e100` | 1.2778 | 1.3940 | 0.7603 |

The go threshold was stage0/1 ratio >= 0.25.  All measured checkpoints exceed
that threshold by a wide margin.

Patch-merge mask stats:

| merge | mixed ratio | note |
|---|---:|---|
| `merge0_to_1` | 0.0000 | mask is block-structured at this scale |
| `merge1_to_2` | 0.0000 | mask is still block-structured |
| `merge2_to_3` | 0.9120 | high mixing at deep stage |

Interpretation:
- Early patch merging does not mix visible/masked groups under this masking
  pattern, so the most direct target is not early patch-merge leakage.
- The stronger signal is persistent masked-placeholder norm in stage features
  that feed decoder skips.

## Decision

Proceed to V0/V1 only:

- V0 `visibility_feature_reset`
- V1 `visibility_skip_gate`

Do not implement attention KV-gating yet.  Attention mass was intentionally not
measured by the non-invasive probe, and adding attention masks would introduce
shifted-window and all-masked-window failure modes.

## Launch Criteria

The first scout should be:

```text
EPOCHS=100
SEEDS=1
CONDITIONS="visibility_skip_gate visibility_feature_reset"
```

Decision after e100:
- If either V0/V1 is competitive with `cosine_ramp` / structure-first budget
  curve on AP@50 without harming AP@75/R@50, keep the branch.
- If both are weak, stop Visibility-Gated and do not try attention gating.

## Submitted Scouts

Submitted 2026-06-07:

| condition | seed | pretrain job | dependent FCOS job |
|---|---:|---|---|
| `visibility_skip_gate` | 1 | `1832027.pbs1` | `1832028.pbs1` |
| `visibility_feature_reset` | 1 | `1832029.pbs1` | `1832030.pbs1` |
