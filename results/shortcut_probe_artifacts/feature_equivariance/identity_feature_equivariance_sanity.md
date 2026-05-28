# Feature Equivariance Probe

Config:

- pair_mode: `identity`
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
| baseline_coord_jitter_e100 | 0 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 23530.0 |
| baseline_coord_jitter_e100 | 0 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 12775.2 |
| baseline_coord_jitter_e100 | 0 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 10754.8 |
| baseline_coord_jitter_e100 | 1 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 2900.0 |
| baseline_coord_jitter_e100 | 1 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 1318.0 |
| baseline_coord_jitter_e100 | 1 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 1582.0 |
| baseline_coord_jitter_e100 | 2 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 370.0 |
| baseline_coord_jitter_e100 | 2 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 111.8 |
| baseline_coord_jitter_e100 | 2 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 258.2 |
| baseline_coord_jitter_e100 | 3 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 43.8 |
| baseline_coord_jitter_e100 | 3 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 5.5 |
| baseline_coord_jitter_e100 | 3 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 38.2 |
| baseline_e300 | 0 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 23530.0 |
| baseline_e300 | 0 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 12775.2 |
| baseline_e300 | 0 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 10754.8 |
| baseline_e300 | 1 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 2900.0 |
| baseline_e300 | 1 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 1318.0 |
| baseline_e300 | 1 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 1582.0 |
| baseline_e300 | 2 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 370.0 |
| baseline_e300 | 2 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 111.8 |
| baseline_e300 | 2 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 258.2 |
| baseline_e300 | 3 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 43.8 |
| baseline_e300 | 3 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 5.5 |
| baseline_e300 | 3 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 38.2 |
| cosine_coord_jitter_e100 | 0 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 23530.0 |
| cosine_coord_jitter_e100 | 0 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 12775.2 |
| cosine_coord_jitter_e100 | 0 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 10754.8 |
| cosine_coord_jitter_e100 | 1 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 2900.0 |
| cosine_coord_jitter_e100 | 1 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 1318.0 |
| cosine_coord_jitter_e100 | 1 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 1582.0 |
| cosine_coord_jitter_e100 | 2 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 370.0 |
| cosine_coord_jitter_e100 | 2 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 111.8 |
| cosine_coord_jitter_e100 | 2 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 258.2 |
| cosine_coord_jitter_e100 | 3 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 43.8 |
| cosine_coord_jitter_e100 | 3 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 5.5 |
| cosine_coord_jitter_e100 | 3 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 38.2 |
| cosine_e300 | 0 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 23530.0 |
| cosine_e300 | 0 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 12775.2 |
| cosine_e300 | 0 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 10754.8 |
| cosine_e300 | 1 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 2900.0 |
| cosine_e300 | 1 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 1318.0 |
| cosine_e300 | 1 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 1582.0 |
| cosine_e300 | 2 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 370.0 |
| cosine_e300 | 2 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 111.8 |
| cosine_e300 | 2 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 258.2 |
| cosine_e300 | 3 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 43.8 |
| cosine_e300 | 3 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 5.5 |
| cosine_e300 | 3 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 38.2 |
| shuffle_coord_jitter_e300 | 0 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 23530.0 |
| shuffle_coord_jitter_e300 | 0 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 12775.2 |
| shuffle_coord_jitter_e300 | 0 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 10754.8 |
| shuffle_coord_jitter_e300 | 1 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 2900.0 |
| shuffle_coord_jitter_e300 | 1 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 1318.0 |
| shuffle_coord_jitter_e300 | 1 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 1582.0 |
| shuffle_coord_jitter_e300 | 2 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 370.0 |
| shuffle_coord_jitter_e300 | 2 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 111.8 |
| shuffle_coord_jitter_e300 | 2 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 258.2 |
| shuffle_coord_jitter_e300 | 3 | all | 4 | 1.0000 | 0.0000 | 1.0000 | 43.8 |
| shuffle_coord_jitter_e300 | 3 | empty | 4 | 1.0000 | 0.0000 | 1.0000 | 5.5 |
| shuffle_coord_jitter_e300 | 3 | surface | 4 | 1.0000 | 0.0000 | 1.0000 | 38.2 |
