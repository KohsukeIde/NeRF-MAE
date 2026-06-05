# MixNeRF / Dithered Masking Implementation Audit

Snapshot: 2026-06-06 JST

## Current finding

The previous `MIXNERF_FILL_MODE=shuffle` implementation shuffles all same-scene
patches, not only visible patches.  It therefore does not strictly prove a
visible-only, mask-token-free mechanism, because a masked target patch can be
sampled as a filler for another masked location.

This does not necessarily invalidate the e30 result as a diagnostic control, but
it is not clean enough for a method claim.

## Code changes for the next scout

Added clean fill modes in `nerf_mae/model/mae/mixnerf_probe.py`:

| fill mode | purpose |
|---|---|
| `shuffle_visible` | same-scene filler sampled only from visible patches (`patch_mask == 0`) |
| `mean` | simple non-zero channel-mean filler control |
| `constant` | scalar non-zero filler control, available but not in the default scout |

The `shuffle_visible` path logs:

```text
same_scene_fill_source = visible_only
self_replacement_rate = 0.0
masked_source_rate = 0.0
visible_patch_count_*
masked_patch_count_*
```

Local tensor sanity check:

```text
mask_mean = 0.75
same_scene_fill_source = visible_only
self_replacement_rate = 0.0
masked_source_rate = 0.0
visible_patch_count_min = 16
masked_patch_count_min = 48
```

## Previous log evidence

Existing MixNeRF pretrain logs show:

```text
patch_mask_mean ~= 0.75
voxel_mask_mean ~= 0.75
internal_mask_attrs_overridden = ['masking_prob']
base_mask_mean = 0.0
```

So the wrapper disables the base internal mask and uses its own mix mask for the
loss path.  The next scout keeps this setup and uses:

```text
PROBE_MODE = custom
PROBE_RGB_LOSS = removed_occupied
PROBE_ALPHA_LOSS = removed
MIXNERF_DISABLE_INTERNAL_MASK = 1
MIXNERF_LOG_STATS = 1
```

## Remaining interpretation caveats

- `zero-fill` under the custom masked objective is not identical to the original
  public NeRF-MAE baseline.  It is a controlled placeholder-fill condition under
  the same MixNeRF wrapper and masked-loss protocol.
- The e30 ranking is not sufficient for a method claim because the earlier noise
  control was not stable at e100.  The e100 masked-objective scout is required
  before upgrading this branch.
- If `mean` is close to `shuffle_visible`, the result is likely a simple
  non-zero filler effect rather than a scene-distribution matching mechanism.

## Submitted next-scout design

Minimum e100 masked-objective scout:

| condition | fill | objective | seeds |
|---|---|---|---|
| `mixnerf_lite_shuffle_visible_masked` | `shuffle_visible` | `removed_occupied` RGB / removed alpha | 1, 2 |
| `mixnerf_lite_zeros_masked` | `zeros` | `removed_occupied` RGB / removed alpha | 1, 2 |
| `mixnerf_lite_mean_masked` | `mean` | `removed_occupied` RGB / removed alpha | 1 |

Decision:

- `shuffle_visible > zeros` and `shuffle_visible > mean`: distribution-preserving
  dither remains a plausible method branch.
- `mean ~= shuffle_visible`: likely non-zero filler, weak novelty.
- `shuffle_visible` collapses at e100: stop this branch and keep it separated
  from the main efficiency paper.

