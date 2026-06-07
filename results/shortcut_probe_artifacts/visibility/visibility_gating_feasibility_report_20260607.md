# Visibility-Gated Feasibility Report

Snapshot: 2026-06-07 JST

## Current Strategy Confidence

Not 100% yet.  The strategy is only defensible if a measurement gate shows that
masked placeholders still participate materially in encoder/skip features.

Therefore the current strategy is:

1. Do not implement Visibility-Gated MAE yet.
2. Measure masked-token participation in the current dense-grid encoder.
3. Implement only V0/V1 if the measurement is positive.
4. Do not implement attention KV-gating unless V0/V1 or measurement evidence
   justifies the additional complexity.

## Code-Level Facts

- `forward_encoder_ecoder` patch-partitions the full dense volume, adds
  `pos_embed`, then applies `window_masking_3d`.
- `window_masking_3d` replaces masked patch-grid tokens with `mask_token`.
- All Swin stages process the resulting dense grid.
- Decoder skip connections use stage features:
  - `decoder4(features[3], features[2])`
  - `decoder3(dec3, features[1])`
  - `decoder2(dec2, features[0])`
- Patch merging concatenates 2x2x2 neighboring tokens, so visible/masked states
  can mix through hierarchy unless a mask propagation rule is introduced.

These facts make Visibility-Gated feasible, but not yet justified.

## Measurement Gate

Added non-invasive probe:

```bash
python nerf_mae/probe_scripts/encoder_mask_participation_report.py
```

The ABCI runner is:

```bash
qsub nerf_mae/probe_scripts/abci3_encoder_mask_participation.pbs
```

Expected artifacts:

```text
results/shortcut_probe_artifacts/visibility/<run_name>/encoder_mask_participation_report.md
results/shortcut_probe_artifacts/visibility/<run_name>/encoder_mask_participation_report.json
results/shortcut_probe_artifacts/visibility/<run_name>/feature_norm_by_stage.csv
results/shortcut_probe_artifacts/visibility/<run_name>/patch_merge_mask_stats.csv
results/shortcut_probe_artifacts/visibility/<run_name>/skip_feature_mask_stats.csv
results/shortcut_probe_artifacts/visibility/<run_name>/attention_mass_by_block.csv
```

The attention CSV is intentionally marked as not measured in the non-invasive
probe.  The current attention implementation does not expose attention tensors
or accept a visibility mask.  Attention-mass logging should only be added if the
feature/skip/merge gate is positive.

## Go / No-Go Rule

Go only if at least one condition is satisfied:

- Stage0/1 masked-visible feature-norm ratio is >= 0.25.
- Patch-merge mixed-group ratio is high and masked skip norms persist into the
  decoder-facing stage features.
- A later intrusive attention audit shows visible-to-masked attention mass >=
  0.15.

No-go if:

- Stage0 masked-visible feature-norm ratio is tiny and no early mixed merge path
  exists.
- Masked skip norms vanish before decoder-facing stages.
- The only positive signal would require attention-gating instrumentation before
  any simpler V0/V1 evidence exists.

## If Go

First implement only low-risk variants:

- V0: feature reset / masked-feature replacement after stage outputs.
- V1: skip-gated features before decoder skip connections.

Do not start with attention KV-gating.  V2 has higher risk because it must handle
shifted-window masks, all-masked windows, Q/K/V masking semantics, and residual
updates to masked queries.

## If No-Go

Do not implement Visibility-Gated MAE.  Return to the structure-first / low-label
efficiency paper, and keep encoder-side methods as future work.

