# Masked Skip Shortcut Diagnostic

## Protocol

- `features_dir`: `/groups/gag51404/ide/vgi/NeRF-MAE/dataset/_downloads/front3d_rpn_extract/front3d_rpn_data/features`
- `split_file`: `/groups/gag51404/ide/vgi/NeRF-MAE/dataset/_downloads/front3d_rpn_extract/front3d_rpn_data/3dfront_split.npz`
- `split_key`: `val_scenes`
- `scenes`: `['3dfront_0083_01', '3dfront_0042_01']`
- `resolution`: `160`
- `masking_prob`: `0.75`
- `normalize_density`: `True`

Modes:
- `normal`: original decoder skips
- `masked_zero`: zero masked-position decoder skip features only
- `visible_zero`: zero visible-position decoder skip features only
- `all_zero`: zero all decoder skip features

## baseline_coord_jitter_e100

- checkpoint: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_baseline_coord_jitter_p1.0_e100_seed1_abci3diag_opt1n8g_det0/epoch_100.pt`
- loaded keys: `383/383`
- missing keys: `0`; unexpected keys: `0`

| mode | n | loss | RGB occupied | RGB removed occupied | RGB visible occupied | alpha removed |
|---|---:|---:|---:|---:|---:|---:|
| all_zero | 2 | 0.154890 | 0.079871 | 0.080062 | 0.078630 | 0.075019 |
| masked_zero | 2 | 0.053242 | 0.027827 | 0.036057 | 0.004105 | 0.025415 |
| normal | 2 | 0.033802 | 0.012720 | 0.015788 | 0.003387 | 0.021082 |
| visible_zero | 2 | 0.052310 | 0.023828 | 0.022288 | 0.028060 | 0.028482 |

| mode | loss delta vs normal | RGB removed occupied delta | alpha removed delta |
|---|---:|---:|---:|
| all_zero | 0.121088 | 0.064274 | 0.053937 |
| masked_zero | 0.019440 | 0.020269 | 0.004333 |
| normal | 0.000000 | 0.000000 | 0.000000 |
| visible_zero | 0.018508 | 0.006501 | 0.007400 |

| stage | n | masked grad mean | visible grad mean | masked/visible grad |
|---|---:|---:|---:|---:|
| stage0 | 1 | 9.171321e-07 | 3.482253e-07 | 2.6337 |
| stage1 | 1 | 3.674055e-07 | 2.584568e-07 | 1.4215 |
| stage2 | 1 | 8.013751e-09 | 7.220930e-09 | 1.1098 |

## cosine_coord_jitter_e100

- checkpoint: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_cosine_coord_jitter_p1.0_e100_seed1_abci3diag_opt1n8g_det0/epoch_100.pt`
- loaded keys: `383/383`
- missing keys: `0`; unexpected keys: `0`

| mode | n | loss | RGB occupied | RGB removed occupied | RGB visible occupied | alpha removed |
|---|---:|---:|---:|---:|---:|---:|
| all_zero | 2 | 0.158844 | 0.086746 | 0.084416 | 0.093287 | 0.072097 |
| masked_zero | 2 | 0.052668 | 0.027848 | 0.036068 | 0.004261 | 0.024820 |
| normal | 2 | 0.033542 | 0.013353 | 0.016295 | 0.004505 | 0.020188 |
| visible_zero | 2 | 0.053728 | 0.029000 | 0.026279 | 0.036615 | 0.024728 |

| mode | loss delta vs normal | RGB removed occupied delta | alpha removed delta |
|---|---:|---:|---:|
| all_zero | 0.125302 | 0.068121 | 0.051909 |
| masked_zero | 0.019126 | 0.019774 | 0.004632 |
| normal | 0.000000 | 0.000000 | 0.000000 |
| visible_zero | 0.020186 | 0.009985 | 0.004540 |

| stage | n | masked grad mean | visible grad mean | masked/visible grad |
|---|---:|---:|---:|---:|
| stage0 | 1 | 9.825034e-07 | 4.069412e-07 | 2.4144 |
| stage1 | 1 | 5.774477e-07 | 3.710867e-07 | 1.5561 |
| stage2 | 1 | 9.881226e-09 | 8.876015e-09 | 1.1133 |

## visibility_cosine_skip_gate_e100

- checkpoint: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_visibility_skip_gate_p1.0_e100_seed1_abci3vis_e100_20260607_141851/epoch_100.pt`
- loaded keys: `383/383`
- missing keys: `0`; unexpected keys: `0`

| mode | n | loss | RGB occupied | RGB removed occupied | RGB visible occupied | alpha removed |
|---|---:|---:|---:|---:|---:|---:|
| all_zero | 2 | 0.178976 | 0.114357 | 0.117497 | 0.103326 | 0.064619 |
| masked_zero | 2 | 0.038587 | 0.017057 | 0.021718 | 0.003740 | 0.021530 |
| normal | 2 | 0.087836 | 0.052662 | 0.063638 | 0.020350 | 0.035174 |
| visible_zero | 2 | 0.404602 | 0.269852 | 0.271401 | 0.266829 | 0.134750 |

| mode | loss delta vs normal | RGB removed occupied delta | alpha removed delta |
|---|---:|---:|---:|
| all_zero | 0.091140 | 0.053859 | 0.029445 |
| masked_zero | -0.049249 | -0.041919 | -0.013644 |
| normal | 0.000000 | 0.000000 | 0.000000 |
| visible_zero | 0.316766 | 0.207764 | 0.099576 |

| stage | n | masked grad mean | visible grad mean | masked/visible grad |
|---|---:|---:|---:|---:|
| stage0 | 1 | 3.059614e-06 | 1.191560e-06 | 2.5677 |
| stage1 | 1 | 1.282478e-06 | 7.229941e-07 | 1.7738 |
| stage2 | 1 | 2.287275e-08 | 1.644737e-08 | 1.3907 |

