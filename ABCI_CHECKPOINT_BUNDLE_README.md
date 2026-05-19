# NeRF-MAE ABCI Pretrain Checkpoint Bundle

Created: 2026-05-19 JST
Repo root on source: /home/minesawa/ssl/NeRF-MAE

Unpack from the NeRF-MAE repo root on ABCI:
```bash
unzip nerfmae_abci_pretrain_checkpoints_20260519.zip
```

Included checkpoints:

| path | purpose |
|---|---|
| output/nerf_mae/results/nerfmae_all_p1.0_e300_seed1/epoch_300.pt | baseline e300 seed1 reference / FCOS reuse |
| output/nerf_mae/results/nerfmae_all_p1.0_e300_seed2/epoch_300.pt | baseline e300 seed2 pretrain complete |
| output/nerf_mae/results/nerfmae_all_p1.0_e300_seed3/epoch_220.pt | resume baseline e300 seed3 toward epoch300 |
| output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1/epoch_300.pt | cosine e300 seed1 reference / FCOS reuse |
| output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_alpha_shuffle_p1.0_e300_seed1/epoch_300.pt | shuffle-control e300 seed1 reference / FCOS reuse |
| output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e600_seed1/epoch_600.pt | cosine e600 seed1 scout reference |
| output/nerf_mae/results/nerfmae_all_p1.0_e1200_seed1/epoch_1200.pt | vanilla e1200 paper-budget reference |

Not included: new seed2/3 cosine and shuffle checkpoints because those pretrains have not started locally.
Not included by default: FCOS best checkpoints; this bundle is for pretrain resume/downstream transfer.
