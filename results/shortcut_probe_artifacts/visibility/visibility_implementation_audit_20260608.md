# Visibility Implementation Audit

Snapshot:
- 2026-06-08 JST

Scope:
- `nerf_mae/model/mae/visibility_gated_probe.py`
- `nerf_mae/run_swin_visibility_gated.py`
- `nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs`
- `nerf_mae/probe_scripts/abci3_e300_gate_fcos.pbs`
- `nerf_mae/probe_scripts/submit_visibility_gated_scouts.sh`
- `nerf_rpn`

## Findings

### Gate Location

`VISGATE_MODE=skip_gate` keeps encoder propagation unchanged.
The wrapper runs:

```text
x = stage_i(x)
feature = permute(x)
feature = feature * visible_mask
features.append(feature)
```

The gate is applied before the feature is appended to the decoder skip list.
It is not applied after decoder concatenation.

Decoder use:

```text
decoder4(features[3], features[2])
decoder3(dec3, features[1])
decoder2(dec2, features[0])
decoder1(dec1)
```

Thus stage2, stage1, and stage0 are decoder skip sources. Stage3 is the main
decoder input, not a skip.

### Stage Mask Shape

The model patch grid is 40^3 for 160^3 inputs with patch size 4.
The visibility wrapper converts the patch mask to each stage shape:

| stage | expected shape | source |
|---|---:|---|
| 0 | 40^3 | original patch mask |
| 1 | 20^3 | pooled patch mask |
| 2 | 10^3 | pooled patch mask |
| 3 | 5^3 | pooled patch mask |

Current downsample rule in `visibility_gated_probe.py` is max pooling. This is
safe for preventing masked child tokens from contributing to coarser skip
locations, but it is stricter than mean/soft visibility.

### Downstream Leakage

`rg -n "visibility|skip_gate|feature_reset" nerf_rpn` finds no FCOS feature
extractor integration of the visibility gate. The downstream model loads the
pretrained encoder weights normally. Therefore the visibility gate is a
pretraining-side architecture/objective intervention, not a downstream FCOS
architecture change.

### Important Correction: Existing Result Used Cosine Curriculum

The completed jobs:

```text
visibility_skip_gate: pretrain 1832027.pbs1, FCOS 1832028.pbs1
visibility_feature_reset: pretrain 1832029.pbs1, FCOS 1832030.pbs1
```

were launched with:

```text
PROBE_CURRICULUM=cosine_rgb_ramp
PROBE_CURRICULUM_EPOCHS=100
PROBE_CURRICULUM_RGB_START_WEIGHT=0.0
PROBE_CURRICULUM_RGB_END_WEIGHT=1.0
PROBE_CURRICULUM_ALPHA_WEIGHT=1.0
```

Therefore the reported `visibility_skip_gate` result is more precisely:

```text
cosine-ramp + decoder skip gate, e100, seed1
```

It should not be interpreted as a pure skip-gate result.

### Launcher Fix

The launchers now distinguish:

| condition | default curriculum | meaning |
|---|---|---|
| `visibility_skip_gate` | `none` | pure decoder skip gate |
| `visibility_cosine_skip_gate` | `cosine_rgb_ramp` | cosine-ramp + decoder skip gate |
| `visibility_feature_reset` | `none` | pure feature reset |
| `visibility_reset_skip` | `none` | feature reset + skip gate |

This prevents future result-name ambiguity.

## Risk Assessment

- `feature_reset` is a no-go because it gates encoder propagation and damages
  transfer.
- `cosine-ramp + skip_gate` is a viable scout but not a winning method yet.
- The next gate is not another combination run. It is a decoder skip shortcut
  diagnostic:
  - normal skip
  - masked-position skip zeroed
  - visible-position skip zeroed
  - all skip zeroed

Only if masked-position skip zeroing materially changes reconstruction loss or
gradient attribution should the visibility route be promoted.
