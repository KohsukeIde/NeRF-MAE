# Feature Equivariance Probe

Checkpoint load reports:

- `baseline_e300`: epoch=300 missing=0 unexpected=0 path=`/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_all_p1.0_e300_seed1/epoch_300.pt`
- `cosine_e300`: epoch=300 missing=0 unexpected=0 path=`/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1/epoch_300.pt`
- `baseline_coord_jitter_e100`: epoch=100 missing=0 unexpected=0 path=`/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_baseline_coord_jitter_p1.0_e100_seed1_abci3diag_opt1n8g_det0/epoch_100.pt`
- `cosine_coord_jitter_e100`: epoch=100 missing=0 unexpected=0 path=`/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_cosine_coord_jitter_p1.0_e100_seed1_abci3diag_opt1n8g_det0/epoch_100.pt`
- `shuffle_coord_jitter_e300`: epoch=300 missing=0 unexpected=0 path=`/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_shuffle_coord_jitter_p1.0_e300_seed1_abci3shuf_cj_det0_1n8g/epoch_300.pt`

| label | stage | region | n | cosine | l2 | cka | tokens |
|---|---:|---|---:|---:|---:|---:|---:|
| baseline_coord_jitter_e100 | 0 | all | 16 | 0.6704 | 0.7483 | 0.5296 | 13332.1 |
| baseline_coord_jitter_e100 | 0 | empty | 16 | 0.6990 | 0.7085 | 0.4804 | 8724.5 |
| baseline_coord_jitter_e100 | 0 | surface | 16 | 0.6127 | 0.8305 | 0.4481 | 4607.6 |
| baseline_coord_jitter_e100 | 1 | all | 16 | 0.7239 | 0.6761 | 0.5245 | 1685.9 |
| baseline_coord_jitter_e100 | 1 | empty | 16 | 0.7501 | 0.6299 | 0.4398 | 964.0 |
| baseline_coord_jitter_e100 | 1 | surface | 16 | 0.6836 | 0.7407 | 0.4514 | 721.9 |
| baseline_coord_jitter_e100 | 2 | all | 16 | 0.7460 | 0.6334 | 0.4030 | 206.0 |
| baseline_coord_jitter_e100 | 2 | empty | 16 | 0.7567 | 0.5961 | 0.4544 | 90.8 |
| baseline_coord_jitter_e100 | 2 | surface | 16 | 0.7308 | 0.6664 | 0.4128 | 115.2 |
| baseline_coord_jitter_e100 | 3 | all | 16 | 0.6951 | 0.7463 | 0.5950 | 25.3 |
| baseline_coord_jitter_e100 | 3 | empty | 16 | 0.7484 | 0.6625 | 0.6772 | 8.2 |
| baseline_coord_jitter_e100 | 3 | surface | 16 | 0.6998 | 0.7393 | 0.5998 | 17.1 |
| baseline_e300 | 0 | all | 16 | 0.6484 | 0.7590 | 0.4582 | 11864.3 |
| baseline_e300 | 0 | empty | 16 | 0.6930 | 0.6953 | 0.4385 | 7484.7 |
| baseline_e300 | 0 | surface | 16 | 0.5899 | 0.8528 | 0.3646 | 4379.6 |
| baseline_e300 | 1 | all | 16 | 0.7355 | 0.6485 | 0.4659 | 1432.6 |
| baseline_e300 | 1 | empty | 16 | 0.7622 | 0.5912 | 0.4285 | 791.7 |
| baseline_e300 | 1 | surface | 16 | 0.7327 | 0.6712 | 0.4262 | 640.9 |
| baseline_e300 | 2 | all | 16 | 0.8220 | 0.5129 | 0.4859 | 192.1 |
| baseline_e300 | 2 | empty | 16 | 0.8291 | 0.4771 | 0.4524 | 81.1 |
| baseline_e300 | 2 | surface | 16 | 0.8544 | 0.4784 | 0.4949 | 111.1 |
| baseline_e300 | 3 | all | 16 | 0.7889 | 0.5937 | 0.6840 | 24.5 |
| baseline_e300 | 3 | empty | 16 | 0.8156 | 0.5430 | 0.7328 | 7.1 |
| baseline_e300 | 3 | surface | 16 | 0.8077 | 0.5710 | 0.7207 | 17.4 |
| cosine_coord_jitter_e100 | 0 | all | 16 | 0.7000 | 0.7152 | 0.5368 | 12398.6 |
| cosine_coord_jitter_e100 | 0 | empty | 16 | 0.7254 | 0.6836 | 0.5360 | 7452.6 |
| cosine_coord_jitter_e100 | 0 | surface | 16 | 0.6645 | 0.7628 | 0.4765 | 4946.0 |
| cosine_coord_jitter_e100 | 1 | all | 16 | 0.7293 | 0.6731 | 0.5719 | 1543.5 |
| cosine_coord_jitter_e100 | 1 | empty | 16 | 0.7537 | 0.6340 | 0.5146 | 823.8 |
| cosine_coord_jitter_e100 | 1 | surface | 16 | 0.7211 | 0.6837 | 0.5243 | 719.8 |
| cosine_coord_jitter_e100 | 2 | all | 16 | 0.7547 | 0.6320 | 0.4626 | 190.2 |
| cosine_coord_jitter_e100 | 2 | empty | 16 | 0.7971 | 0.5432 | 0.4519 | 75.9 |
| cosine_coord_jitter_e100 | 2 | surface | 16 | 0.7249 | 0.6778 | 0.4107 | 114.2 |
| cosine_coord_jitter_e100 | 3 | all | 16 | 0.6988 | 0.7496 | 0.6705 | 24.4 |
| cosine_coord_jitter_e100 | 3 | empty | 16 | 0.7836 | 0.6285 | 0.7801 | 6.5 |
| cosine_coord_jitter_e100 | 3 | surface | 16 | 0.7007 | 0.7467 | 0.6502 | 17.9 |
| cosine_e300 | 0 | all | 16 | 0.6987 | 0.7128 | 0.5672 | 13434.5 |
| cosine_e300 | 0 | empty | 16 | 0.7318 | 0.6699 | 0.5462 | 9422.8 |
| cosine_e300 | 0 | surface | 16 | 0.5913 | 0.8630 | 0.4464 | 4011.8 |
| cosine_e300 | 1 | all | 16 | 0.7442 | 0.6485 | 0.5829 | 1668.1 |
| cosine_e300 | 1 | empty | 16 | 0.7720 | 0.6047 | 0.5720 | 1079.3 |
| cosine_e300 | 1 | surface | 16 | 0.6667 | 0.7599 | 0.4631 | 588.8 |
| cosine_e300 | 2 | all | 16 | 0.8250 | 0.5370 | 0.4438 | 223.6 |
| cosine_e300 | 2 | empty | 16 | 0.8304 | 0.5153 | 0.4392 | 120.9 |
| cosine_e300 | 2 | surface | 16 | 0.8259 | 0.5428 | 0.4056 | 102.7 |
| cosine_e300 | 3 | all | 16 | 0.8248 | 0.5597 | 0.6865 | 28.8 |
| cosine_e300 | 3 | empty | 16 | 0.8439 | 0.5186 | 0.8004 | 12.6 |
| cosine_e300 | 3 | surface | 16 | 0.8399 | 0.5355 | 0.6501 | 16.2 |
| shuffle_coord_jitter_e300 | 0 | all | 16 | 0.6020 | 0.8194 | 0.2952 | 9903.0 |
| shuffle_coord_jitter_e300 | 0 | empty | 16 | 0.6103 | 0.8006 | 0.3009 | 5723.4 |
| shuffle_coord_jitter_e300 | 0 | surface | 16 | 0.5510 | 0.9135 | 0.1996 | 4179.6 |
| shuffle_coord_jitter_e300 | 1 | all | 16 | 0.6742 | 0.7443 | 0.4017 | 1200.9 |
| shuffle_coord_jitter_e300 | 1 | empty | 16 | 0.6715 | 0.7347 | 0.3874 | 611.9 |
| shuffle_coord_jitter_e300 | 1 | surface | 16 | 0.6277 | 0.8311 | 0.3617 | 588.9 |
| shuffle_coord_jitter_e300 | 2 | all | 16 | 0.9155 | 0.3527 | 0.3581 | 137.8 |
| shuffle_coord_jitter_e300 | 2 | empty | 16 | 0.9199 | 0.3389 | 0.3396 | 57.0 |
| shuffle_coord_jitter_e300 | 2 | surface | 16 | 0.9067 | 0.3742 | 0.3519 | 80.8 |
| shuffle_coord_jitter_e300 | 3 | all | 16 | 0.8693 | 0.4813 | 0.6707 | 20.0 |
| shuffle_coord_jitter_e300 | 3 | empty | 16 | 0.9135 | 0.3951 | 0.7386 | 4.1 |
| shuffle_coord_jitter_e300 | 3 | surface | 16 | 0.8621 | 0.4947 | 0.5762 | 15.9 |
