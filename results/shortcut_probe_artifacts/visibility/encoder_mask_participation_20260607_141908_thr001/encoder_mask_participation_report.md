# Encoder Mask Participation Report

Generated from `nerf_mae/probe_scripts/encoder_mask_participation_report.py`.

## Protocol

- `features_dir`: `/groups/gag51404/ide/vgi/NeRF-MAE/dataset/_downloads/front3d_rpn_extract/front3d_rpn_data/features`
- `split_file`: `/groups/gag51404/ide/vgi/NeRF-MAE/dataset/_downloads/front3d_rpn_extract/front3d_rpn_data/3dfront_split.npz`
- `split_key`: `val_scenes`
- `scenes`: `['3dfront_0083_01', '3dfront_0042_01']`
- `resolution`: `160`
- `masking_prob`: `0.75`
- `normalize_density`: `True`

Gate rule from the strategy feedback:
- Go if stage0/1 masked-visible feature norm ratio is >= 0.25, or patch-merge mixed groups are high with persistent masked skip norms.
- Attention-mass measurement is not included here because the current Swin attention function does not expose masks without an intrusive model patch; this report covers the non-invasive gates first.

## baseline_e300

- checkpoint: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_all_p1.0_e300_seed1/epoch_300.pt`
- loaded keys: `383/383`
- missing keys: `0`; unexpected keys: `0`
- mask mean: `0.7475`
- occupied patch fraction: `0.1477`

| stage | all mask/visible | occupied mask/visible | empty mask/visible | masked tokens | visible tokens |
|---|---:|---:|---:|---:|---:|
| stage0 | 0.7578 | 0.6278 | 0.7847 | 47840 | 16160 |
| stage1 | 0.6989 | 0.5489 | 0.7771 | 5980 | 2020 |
| stage2 | 0.6995 | 0.8304 | 0.6467 | 747 | 252 |
| stage3 | nan | nan | nan | 125 | 0 |

| merge | mixed ratio | all masked | all visible | groups |
|---|---:|---:|---:|---:|
| merge0_to_1 | 0.0000 | 0.7475 | 0.2525 | 8000 |
| merge1_to_2 | 0.0000 | 0.7475 | 0.2525 | 1000 |
| merge2_to_3 | 0.9120 | 0.0880 | 0.0000 | 125 |

## cosine_ramp_e300

- checkpoint: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1/epoch_300.pt`
- loaded keys: `383/383`
- missing keys: `0`; unexpected keys: `0`
- mask mean: `0.7475`
- occupied patch fraction: `0.1477`

| stage | all mask/visible | occupied mask/visible | empty mask/visible | masked tokens | visible tokens |
|---|---:|---:|---:|---:|---:|
| stage0 | 0.7303 | 0.5987 | 0.7600 | 47840 | 16160 |
| stage1 | 0.7463 | 0.6003 | 0.8139 | 5980 | 2020 |
| stage2 | 0.7872 | 0.9659 | 0.7172 | 747 | 252 |
| stage3 | nan | nan | nan | 125 | 0 |

| merge | mixed ratio | all masked | all visible | groups |
|---|---:|---:|---:|---:|
| merge0_to_1 | 0.0000 | 0.7475 | 0.2525 | 8000 |
| merge1_to_2 | 0.0000 | 0.7475 | 0.2525 | 1000 |
| merge2_to_3 | 0.9120 | 0.0880 | 0.0000 | 125 |

## cosine_coord_jitter_e100

- checkpoint: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_cosine_coord_jitter_p1.0_e100_seed1_abci3diag_opt1n8g_det0/epoch_100.pt`
- loaded keys: `383/383`
- missing keys: `0`; unexpected keys: `0`
- mask mean: `0.7475`
- occupied patch fraction: `0.1477`

| stage | all mask/visible | occupied mask/visible | empty mask/visible | masked tokens | visible tokens |
|---|---:|---:|---:|---:|---:|
| stage0 | 0.6071 | 0.4830 | 0.6367 | 47840 | 16160 |
| stage1 | 0.7066 | 0.4606 | 0.8736 | 5980 | 2020 |
| stage2 | 0.5134 | 0.6483 | 0.4579 | 747 | 252 |
| stage3 | nan | nan | nan | 125 | 0 |

| merge | mixed ratio | all masked | all visible | groups |
|---|---:|---:|---:|---:|
| merge0_to_1 | 0.0000 | 0.7475 | 0.2525 | 8000 |
| merge1_to_2 | 0.0000 | 0.7475 | 0.2525 | 1000 |
| merge2_to_3 | 0.9120 | 0.0880 | 0.0000 | 125 |

## dither_shuffle_visible_e100

- checkpoint: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_mixnerf_lite_shuffle_visible_masked_p1.0_e100_seed1_abci3dither_e100/epoch_100.pt`
- loaded keys: `383/383`
- missing keys: `0`; unexpected keys: `0`
- mask mean: `0.7475`
- occupied patch fraction: `0.1477`

| stage | all mask/visible | occupied mask/visible | empty mask/visible | masked tokens | visible tokens |
|---|---:|---:|---:|---:|---:|
| stage0 | 1.2778 | 0.6959 | 1.5301 | 47840 | 16160 |
| stage1 | 1.3940 | 0.8825 | 1.6675 | 5980 | 2020 |
| stage2 | 0.7603 | 0.6790 | 0.8040 | 747 | 252 |
| stage3 | nan | nan | nan | 125 | 0 |

| merge | mixed ratio | all masked | all visible | groups |
|---|---:|---:|---:|---:|
| merge0_to_1 | 0.0000 | 0.7475 | 0.2525 | 8000 |
| merge1_to_2 | 0.0000 | 0.7475 | 0.2525 | 1000 |
| merge2_to_3 | 0.9120 | 0.0880 | 0.0000 | 125 |

