# Feature Equivariance Probe

Config:

- pair_mode: `random`
- scene_count: `16`
- num_pairs_per_scene: `3`
- transform_pair_count_per_checkpoint: `48`
- stages: `[0, 1, 2, 3]`
- max_tokens: `8192`

Stage resolution metadata:

| stage | feature_shape | stride_xyz |
|---:|---|---|
| 0 | `[40, 40, 40]` | `[4.0, 4.0, 4.0]` |
| 1 | `[20, 20, 20]` | `[8.0, 8.0, 8.0]` |
| 2 | `[10, 10, 10]` | `[16.0, 16.0, 16.0]` |
| 3 | `[5, 5, 5]` | `[32.0, 32.0, 32.0]` |

Checkpoint load reports:

- `baseline_e300`: epoch=300 missing=0 unexpected=0 path=`/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_all_p1.0_e300_seed1/epoch_300.pt`
- `cosine_e300`: epoch=300 missing=0 unexpected=0 path=`/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1/epoch_300.pt`
- `baseline_coord_jitter_e100`: epoch=100 missing=0 unexpected=0 path=`/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_baseline_coord_jitter_p1.0_e100_seed1_abci3diag_opt1n8g_det0/epoch_100.pt`
- `cosine_coord_jitter_e100`: epoch=100 missing=0 unexpected=0 path=`/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_cosine_coord_jitter_p1.0_e100_seed1_abci3diag_opt1n8g_det0/epoch_100.pt`
- `shuffle_coord_jitter_e300`: epoch=300 missing=0 unexpected=0 path=`/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_shuffle_coord_jitter_p1.0_e300_seed1_abci3shuf_cj_det0_1n8g/epoch_300.pt`

| label | stage | region | n | cosine | l2 | cka | tokens |
|---|---:|---|---:|---:|---:|---:|---:|
| baseline_coord_jitter_e100 | 0 | all | 48 | 0.6104 | 0.8188 | 0.3936 | 16649.1 |
| baseline_coord_jitter_e100 | 0 | empty | 48 | 0.6546 | 0.7604 | 0.3664 | 10114.4 |
| baseline_coord_jitter_e100 | 0 | surface | 48 | 0.5378 | 0.9191 | 0.3174 | 6534.7 |
| baseline_coord_jitter_e100 | 1 | all | 48 | 0.6542 | 0.7574 | 0.4121 | 2104.3 |
| baseline_coord_jitter_e100 | 1 | empty | 48 | 0.6966 | 0.6907 | 0.3268 | 1089.7 |
| baseline_coord_jitter_e100 | 1 | surface | 48 | 0.6038 | 0.8314 | 0.3410 | 1014.7 |
| baseline_coord_jitter_e100 | 2 | all | 48 | 0.7160 | 0.6655 | 0.4453 | 267.2 |
| baseline_coord_jitter_e100 | 2 | empty | 48 | 0.7341 | 0.6179 | 0.3838 | 109.8 |
| baseline_coord_jitter_e100 | 2 | surface | 48 | 0.7010 | 0.6958 | 0.3965 | 157.4 |
| baseline_coord_jitter_e100 | 3 | all | 48 | 0.6665 | 0.7714 | 0.5814 | 34.6 |
| baseline_coord_jitter_e100 | 3 | empty | 48 | 0.7657 | 0.6250 | 0.7091 | 8.7 |
| baseline_coord_jitter_e100 | 3 | surface | 48 | 0.6454 | 0.8002 | 0.5604 | 26.0 |
| baseline_e300 | 0 | all | 48 | 0.5746 | 0.8404 | 0.3526 | 15110.8 |
| baseline_e300 | 0 | empty | 48 | 0.6189 | 0.7798 | 0.3298 | 9468.0 |
| baseline_e300 | 0 | surface | 48 | 0.5122 | 0.9302 | 0.2998 | 5642.7 |
| baseline_e300 | 1 | all | 48 | 0.6739 | 0.7237 | 0.3829 | 1890.0 |
| baseline_e300 | 1 | empty | 48 | 0.6921 | 0.6809 | 0.3584 | 1048.9 |
| baseline_e300 | 1 | surface | 48 | 0.6486 | 0.7709 | 0.3526 | 841.1 |
| baseline_e300 | 2 | all | 48 | 0.8039 | 0.5420 | 0.4632 | 253.8 |
| baseline_e300 | 2 | empty | 48 | 0.8078 | 0.5106 | 0.3815 | 106.6 |
| baseline_e300 | 2 | surface | 48 | 0.7840 | 0.5872 | 0.3973 | 147.2 |
| baseline_e300 | 3 | all | 48 | 0.7510 | 0.6473 | 0.6447 | 33.7 |
| baseline_e300 | 3 | empty | 48 | 0.8149 | 0.5409 | 0.8212 | 8.7 |
| baseline_e300 | 3 | surface | 48 | 0.7295 | 0.6831 | 0.6262 | 25.0 |
| cosine_coord_jitter_e100 | 0 | all | 48 | 0.6653 | 0.7421 | 0.4642 | 17890.2 |
| cosine_coord_jitter_e100 | 0 | empty | 48 | 0.7157 | 0.6757 | 0.4458 | 11110.6 |
| cosine_coord_jitter_e100 | 0 | surface | 48 | 0.5682 | 0.8765 | 0.3608 | 6779.6 |
| cosine_coord_jitter_e100 | 1 | all | 48 | 0.6474 | 0.7647 | 0.4631 | 2220.5 |
| cosine_coord_jitter_e100 | 1 | empty | 48 | 0.6973 | 0.6920 | 0.3978 | 1197.1 |
| cosine_coord_jitter_e100 | 1 | surface | 48 | 0.5759 | 0.8678 | 0.3792 | 1023.4 |
| cosine_coord_jitter_e100 | 2 | all | 48 | 0.7340 | 0.6498 | 0.4660 | 298.2 |
| cosine_coord_jitter_e100 | 2 | empty | 48 | 0.7911 | 0.5482 | 0.4348 | 119.6 |
| cosine_coord_jitter_e100 | 2 | surface | 48 | 0.6966 | 0.7126 | 0.4282 | 178.6 |
| cosine_coord_jitter_e100 | 3 | all | 48 | 0.6507 | 0.8031 | 0.6383 | 38.7 |
| cosine_coord_jitter_e100 | 3 | empty | 48 | 0.6757 | 0.7655 | 0.7472 | 9.7 |
| cosine_coord_jitter_e100 | 3 | surface | 48 | 0.6401 | 0.8185 | 0.6165 | 29.0 |
| cosine_e300 | 0 | all | 48 | 0.6616 | 0.7352 | 0.4745 | 18556.5 |
| cosine_e300 | 0 | empty | 48 | 0.7085 | 0.6675 | 0.4589 | 12204.8 |
| cosine_e300 | 0 | surface | 48 | 0.5922 | 0.8428 | 0.4027 | 6351.7 |
| cosine_e300 | 1 | all | 48 | 0.6943 | 0.6922 | 0.4945 | 2326.9 |
| cosine_e300 | 1 | empty | 48 | 0.7428 | 0.6103 | 0.4803 | 1324.9 |
| cosine_e300 | 1 | surface | 48 | 0.6443 | 0.7782 | 0.4324 | 1002.0 |
| cosine_e300 | 2 | all | 48 | 0.8506 | 0.4672 | 0.5528 | 300.3 |
| cosine_e300 | 2 | empty | 48 | 0.8818 | 0.4003 | 0.5656 | 133.7 |
| cosine_e300 | 2 | surface | 48 | 0.8380 | 0.4987 | 0.4776 | 166.6 |
| cosine_e300 | 3 | all | 48 | 0.8121 | 0.5500 | 0.7279 | 40.3 |
| cosine_e300 | 3 | empty | 48 | 0.8308 | 0.5142 | 0.8202 | 12.4 |
| cosine_e300 | 3 | surface | 48 | 0.8099 | 0.5585 | 0.7037 | 27.9 |
| shuffle_coord_jitter_e300 | 0 | all | 48 | 0.6228 | 0.7909 | 0.3584 | 16484.2 |
| shuffle_coord_jitter_e300 | 0 | empty | 48 | 0.6510 | 0.7457 | 0.3398 | 10144.5 |
| shuffle_coord_jitter_e300 | 0 | surface | 48 | 0.5618 | 0.8886 | 0.2414 | 6339.7 |
| shuffle_coord_jitter_e300 | 1 | all | 48 | 0.6920 | 0.7153 | 0.4467 | 2032.9 |
| shuffle_coord_jitter_e300 | 1 | empty | 48 | 0.7274 | 0.6586 | 0.4053 | 1078.1 |
| shuffle_coord_jitter_e300 | 1 | surface | 48 | 0.6383 | 0.8007 | 0.3791 | 954.8 |
| shuffle_coord_jitter_e300 | 2 | all | 48 | 0.9106 | 0.3487 | 0.4504 | 265.6 |
| shuffle_coord_jitter_e300 | 2 | empty | 48 | 0.9150 | 0.3287 | 0.4558 | 109.9 |
| shuffle_coord_jitter_e300 | 2 | surface | 48 | 0.9090 | 0.3572 | 0.4033 | 155.8 |
| shuffle_coord_jitter_e300 | 3 | all | 48 | 0.8234 | 0.5380 | 0.5795 | 35.9 |
| shuffle_coord_jitter_e300 | 3 | empty | 48 | 0.8601 | 0.4750 | 0.7570 | 8.9 |
| shuffle_coord_jitter_e300 | 3 | surface | 48 | 0.8168 | 0.5467 | 0.5408 | 27.0 |
