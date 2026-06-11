# ScanNet normalize_density Audit and Non-normalized Rerun

Snapshot: 2026-06-11 JST

Question:
- Were the existing ScanNet transfer triage results run with
  `--normalize_density`?

Answer:
- Yes. Existing ScanNet FCOS train/eval commands included `--normalize_density`.
- Therefore the previous ScanNet triage should not be treated as the
  official-recommended ScanNet protocol if ScanNet should be run without density
  normalization.

Evidence:
- Existing ScanNet logs such as `scn_basee300.o1811879`, `scn_cose300.o1811887`,
  `scn_cj_e100.o1811881`, and `scn_smcj_e300.o1811882` show
  `--dataset scannet ... --normalize_density` in both train and eval commands.
- The wrapper path was:
  `abci3_scannet_transfer_fcos.pbs -> run_fcos_probe_variant.sh ->
  train_fcos_pretrained.sh / test_fcos_pretrained.sh`.
- `train_fcos_pretrained.sh` and `test_fcos_pretrained.sh` previously appended
  `--normalize_density` unconditionally.

Patch:
- Added `NORMALIZE_DENSITY` env control to:
  - `nerf_rpn/train_fcos_pretrained.sh`
  - `nerf_rpn/test_fcos_pretrained.sh`
- Set `NORMALIZE_DENSITY=0` by default for:
  - `nerf_rpn/tools/abci3_scannet_transfer_fcos.pbs`
- Front3D/default wrappers keep `NORMALIZE_DENSITY=1` unless explicitly
  overridden.

Why train+eval rerun, not eval-only:
- Existing ScanNet FCOS checkpoints were trained with normalized density.
- Evaluating them with non-normalized density would create a train/eval
  distribution mismatch.
- The protocol-clean rerun therefore retrains and evaluates FCOS with
  `NORMALIZE_DENSITY=0`.

Submitted non-normalized ScanNet reruns:

| condition | job | normalize_density | expected eval |
|---|---:|---:|---|
| `baseline_e300` | `1897280.pbs1` | 0 | `output/nerf_rpn/results/baseline_e300_scannet_nonorm_fcos1000_seed1_eval/eval.json` |
| `cosine_ramp_e300` | `1897281.pbs1` | 0 | `output/nerf_rpn/results/cosine_ramp_e300_scannet_nonorm_fcos1000_seed1_eval/eval.json` |
| `cosine_coord_jitter_e100` | `1897282.pbs1` | 0 | `output/nerf_rpn/results/cosine_coord_jitter_e100_scannet_nonorm_fcos1000_seed1_eval/eval.json` |
| `surface_cosine_jitter_e300` | `1897283.pbs1` | 0 | `output/nerf_rpn/results/surface_cosine_jitter_e300_scannet_nonorm_fcos1000_seed1_eval/eval.json` |

Decision:
- Until these non-normalized reruns finish, the existing ScanNet table should be
  treated as a legacy / normalized-density triage table.
- The paper-facing ScanNet table should use the non-normalized rerun if the
  official ScanNet protocol requires `normalize_density=off`.

