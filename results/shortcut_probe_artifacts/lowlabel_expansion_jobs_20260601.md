# Low-Label Expansion Jobs

Snapshot: 2026-06-01 JST

Purpose:
- Expand the completed 50% low-label gate to `25%`, `10%`, and `100%`
  using the compact paper-facing row set.
- Keep this as a single-finetune-seed direction gate. Promote only the final
  paper rows to 3 finetune seeds after the main variant is chosen.

Protocol:
- Dataset: Front3D OBB detection.
- FCOS: 1000 epochs, finetune seed `1`.
- PBS: `nerf_rpn/tools/abci3_front3d_low_label_fcos.pbs`.
- Deterministic setting: `DETERMINISTIC=1`, matching the existing 50% low-label
  gate.
- Existing 50% rows were not resubmitted.

## Submitted Jobs

| job | labels | condition | checkpoint |
|---|---:|---|---|
| `1815787.pbs1` | 25% | `scratch` | scratch backbone |
| `1815788.pbs1` | 25% | `baseline_e300` | `output/nerf_mae/results/nerfmae_all_p1.0_e300_seed1/epoch_300.pt` |
| `1815789.pbs1` | 25% | `cosine_ramp_e300` | `output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1/epoch_300.pt` |
| `1815790.pbs1` | 25% | `surface_cosine_jitter_e300` | `output/nerf_mae/results/nerfmae_surface_maturation_cosine_coord_jitter_tau0p7_k30_w0p05_p1.0_e300_seed1_abci3smcos_cj_det0_1n8g/epoch_300.pt` |
| `1815791.pbs1` | 10% | `scratch` | scratch backbone |
| `1815792.pbs1` | 10% | `baseline_e300` | `output/nerf_mae/results/nerfmae_all_p1.0_e300_seed1/epoch_300.pt` |
| `1815793.pbs1` | 10% | `cosine_ramp_e300` | `output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1/epoch_300.pt` |
| `1815794.pbs1` | 10% | `surface_cosine_jitter_e300` | `output/nerf_mae/results/nerfmae_surface_maturation_cosine_coord_jitter_tau0p7_k30_w0p05_p1.0_e300_seed1_abci3smcos_cj_det0_1n8g/epoch_300.pt` |
| `1815795.pbs1` | 100% | `scratch` | scratch backbone |
| `1815796.pbs1` | 100% | `baseline_e300` | `output/nerf_mae/results/nerfmae_all_p1.0_e300_seed1/epoch_300.pt` |
| `1815797.pbs1` | 100% | `cosine_ramp_e300` | `output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1/epoch_300.pt` |
| `1815798.pbs1` | 100% | `surface_cosine_jitter_e300` | `output/nerf_mae/results/nerfmae_surface_maturation_cosine_coord_jitter_tau0p7_k30_w0p05_p1.0_e300_seed1_abci3smcos_cj_det0_1n8g/epoch_300.pt` |

## Main-Variant Decision Rule

- If `surface_cosine_jitter_e300` is consistently at or above
  `cosine_ramp_e300` on full-label, 50%, and 25% labels, and does not introduce
  a severe ScanNet regression, promote `surface_cosine_jitter_e300` to the main
  method.
- If `surface_cosine_jitter_e300` is mainly stronger in in-domain low-label
  but weaker on ScanNet or full-label, keep `cosine_ramp_e300` as the main
  method and use surface anchoring as an in-domain/low-label ablation component.
- If 25%/10% collapse or reverse, narrow the label-efficiency claim to the
  moderate-label regime supported by the data, currently 50%.

## Checkpoint Availability Note

Current key pretraining checkpoints available on disk:

| condition | available checkpoints |
|---|---|
| `baseline_e300` | `epoch_300.pt` |
| `cosine_ramp_e300` | `epoch_300.pt` |
| `surface_cosine_jitter_e300` | `epoch_300.pt`, `model_best.pt` |
| `baseline_e1200` | `epoch_1200.pt` |

Intermediate `epoch_100.pt` / `epoch_200.pt` checkpoints are not present for
the key e300/e1200 runs above. If epoch-wise curves are needed for the paper,
future pretraining jobs should explicitly set `PRETRAIN_CHECKPOINT_INTERVAL=100`
or run separate e100/e200 pretraining jobs under the final protocol.
