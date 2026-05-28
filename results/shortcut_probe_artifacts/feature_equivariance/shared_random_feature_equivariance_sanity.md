# Feature Equivariance Probe

Config:

- pair_mode: `shared_random`
- scene_count: `4`
- num_pairs_per_scene: `1`
- transform_pair_count_per_checkpoint: `4`
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
| baseline_coord_jitter_e100 | 0 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 21983.5 |
| baseline_coord_jitter_e100 | 0 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 15724.5 |
| baseline_coord_jitter_e100 | 0 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 6259.0 |
| baseline_coord_jitter_e100 | 1 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 2771.5 |
| baseline_coord_jitter_e100 | 1 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 1779.8 |
| baseline_coord_jitter_e100 | 1 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 991.8 |
| baseline_coord_jitter_e100 | 2 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 368.2 |
| baseline_coord_jitter_e100 | 2 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 192.2 |
| baseline_coord_jitter_e100 | 2 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 176.0 |
| baseline_coord_jitter_e100 | 3 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 43.8 |
| baseline_coord_jitter_e100 | 3 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 15.2 |
| baseline_coord_jitter_e100 | 3 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 28.5 |
| baseline_e300 | 0 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 21820.0 |
| baseline_e300 | 0 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 16261.5 |
| baseline_e300 | 0 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 5558.5 |
| baseline_e300 | 1 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 2742.0 |
| baseline_e300 | 1 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 1844.8 |
| baseline_e300 | 1 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 897.2 |
| baseline_e300 | 2 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 368.5 |
| baseline_e300 | 2 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 192.2 |
| baseline_e300 | 2 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 176.2 |
| baseline_e300 | 3 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 40.0 |
| baseline_e300 | 3 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 15.2 |
| baseline_e300 | 3 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 24.8 |
| cosine_coord_jitter_e100 | 0 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 22204.0 |
| cosine_coord_jitter_e100 | 0 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 14534.2 |
| cosine_coord_jitter_e100 | 0 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 7669.8 |
| cosine_coord_jitter_e100 | 1 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 2778.0 |
| cosine_coord_jitter_e100 | 1 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 1635.0 |
| cosine_coord_jitter_e100 | 1 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 1143.0 |
| cosine_coord_jitter_e100 | 2 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 370.0 |
| cosine_coord_jitter_e100 | 2 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 166.2 |
| cosine_coord_jitter_e100 | 2 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 203.8 |
| cosine_coord_jitter_e100 | 3 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 40.0 |
| cosine_coord_jitter_e100 | 3 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 13.0 |
| cosine_coord_jitter_e100 | 3 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 27.0 |
| cosine_e300 | 0 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 21893.0 |
| cosine_e300 | 0 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 15479.5 |
| cosine_e300 | 0 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 6413.5 |
| cosine_e300 | 1 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 2791.5 |
| cosine_e300 | 1 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 1812.5 |
| cosine_e300 | 1 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 979.0 |
| cosine_e300 | 2 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 367.0 |
| cosine_e300 | 2 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 177.2 |
| cosine_e300 | 2 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 189.8 |
| cosine_e300 | 3 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 43.0 |
| cosine_e300 | 3 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 14.8 |
| cosine_e300 | 3 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 28.2 |
| shuffle_coord_jitter_e300 | 0 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 21269.0 |
| shuffle_coord_jitter_e300 | 0 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 14742.8 |
| shuffle_coord_jitter_e300 | 0 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 6526.2 |
| shuffle_coord_jitter_e300 | 1 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 2663.5 |
| shuffle_coord_jitter_e300 | 1 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 1673.0 |
| shuffle_coord_jitter_e300 | 1 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 990.5 |
| shuffle_coord_jitter_e300 | 2 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 332.5 |
| shuffle_coord_jitter_e300 | 2 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 188.0 |
| shuffle_coord_jitter_e300 | 2 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 144.5 |
| shuffle_coord_jitter_e300 | 3 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 40.0 |
| shuffle_coord_jitter_e300 | 3 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 17.5 |
| shuffle_coord_jitter_e300 | 3 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 22.5 |
