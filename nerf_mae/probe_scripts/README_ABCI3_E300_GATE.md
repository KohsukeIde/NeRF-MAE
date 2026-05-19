# ABCI3 e300 Gate

This note documents the clean e300 gate for the cosine alpha-to-RGBA curriculum.

## Goal

Run the paper-critical e300 seed gate for:

- `baseline`
- `cosine_ramp`
- `cosine_ramp_alpha_shuffle`

Each condition uses `e300`, followed by the existing FCOS e1000 downstream protocol. The current priority is the 3-seed gate; single-seed results are candidate signals, not paper claims.

## Setup

Create the ABCI3 Python environment:

```bash
PROBE_ENV_PREFIX=/groups/gag51404/ide/vgi/NeRF-MAE/.venv-abci3 \
bash nerf_mae/probe_scripts/setup_abci3_env.sh
```

Link preprocessed data:

```bash
PRETRAIN_DATA_SRC=/path/to/pretrain \
FCOS_DATA_SRC=/path/to/front3d_rpn_data \
bash nerf_mae/probe_scripts/setup_abci3_data_links.sh
```

For checkpoint-only FCOS eval, only the FCOS data link is required:

```bash
REQUIRE_PRETRAIN_DATA=0 \
FCOS_DATA_SRC=/path/to/front3d_rpn_data \
bash nerf_mae/probe_scripts/setup_abci3_data_links.sh
```

The expected data format is:

- pretrain: `features/` and `nerfmae_split.npz`
- FCOS: `features/`, `obb/`, and `3dfront_split.npz`

Do not link raw `Structure3D` directly unless it already contains this preprocessed layout.

If pretrain checkpoints are brought from the old environment, link them into the expected ABCI3 result layout:

```bash
BUNDLE=/groups/gag51404/ide/vgi/NeRF-MAE/nerfmae_abci_pretrain_checkpoints_20260519.zip \
RUN_SUFFIX=abci3clean \
bash nerf_mae/probe_scripts/install_abci_checkpoint_bundle.sh
```

The bundle installer extracts without overwriting existing files, verifies
`checksums.sha256`, and creates `*_abci3clean` symlinks so the clean ABCI3
FCOS/pretrain jobs can keep a separate result namespace. If a partial checkpoint
such as `baseline_e300 seed3 epoch_220.pt` is present, the pretrain PBS
auto-resumes it by default.

## Preflight

```bash
bash nerf_mae/probe_scripts/abci3_e300_gate_preflight.sh
```

This must pass before submitting jobs.

## Submit

Fastest path for already included checkpoints: run only clean FCOS evals on ABCI3.

```bash
GATE_JOBS="baseline:baseline:300:1 baseline:baseline:300:2 curriculum:cosine_ramp:300:1 curriculum:cosine_ramp_alpha_shuffle:300:1" \
SUBMIT_PRETRAIN=0 SUBMIT_FCOS=1 \
bash nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh
```

This submits FCOS jobs only for checkpoints already installed from the bundle:

```text
baseline seed1/2 -> FCOS
cosine_ramp seed1 -> FCOS
alpha_shuffle seed1 -> FCOS
```

For the current Gate 1, submit the full 3-seed e300 grid. Existing final
checkpoints are skipped, `baseline seed3` resumes from `epoch_220.pt` when that
checkpoint was installed, and missing cosine/shuffle seed2/3 checkpoints are
trained from scratch:

```bash
PRETRAIN_NODES=2 PRETRAIN_SLOTS=3 PRETRAIN_BATCH_SIZE_PER_GPU=1 PRETRAIN_EVAL_INTERVAL=300 \
bash nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh
```

`PRETRAIN_EVAL_INTERVAL=300` skips intermediate MAE validation and still saves the final `epoch_300.pt`; it does not change the optimizer updates used for downstream evaluation.

To submit only missing pretrains plus their dependent FCOS jobs:

```bash
GATE_JOBS="baseline:baseline:300:3 curriculum:cosine_ramp:300:2 curriculum:cosine_ramp:300:3 curriculum:cosine_ramp_alpha_shuffle:300:2 curriculum:cosine_ramp_alpha_shuffle:300:3" \
PRETRAIN_NODES=2 PRETRAIN_SLOTS=9 PRETRAIN_BATCH_SIZE_PER_GPU=1 PRETRAIN_EVAL_INTERVAL=300 \
bash nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh
```

The unchained submitter is still available when you intentionally want every pretrain job eligible immediately:

```bash
bash nerf_mae/probe_scripts/submit_abci3_e300_gate.sh
```

Both submitters create 9 pretrain jobs and 9 dependent FCOS jobs. FCOS remains single-GPU by default to preserve the downstream protocol.

Useful controls:

- `DRY_RUN=1`: print qsub commands without submitting
- `PRETRAIN_SLOTS=3`: maximum number of concurrent pretrain jobs
- `PRETRAIN_EVAL_INTERVAL=300`: skip intermediate MAE validation for fastest clean `epoch_300.pt` runs
- `AUTO_RESUME_PRETRAIN=1`: resume from the latest `epoch_*.pt` in the target pretrain directory
- `SUBMIT_PRETRAIN=0`: submit FCOS only, assuming checkpoints already exist
- `SUBMIT_FCOS=0`: submit pretrain only
- `RUN_SUFFIX=abci3clean`: suffix for checkpoint/result directories
- `ABCI_GROUP=gag51404`: ABCI group
- `PRETRAIN_QUEUE=rt_HF`, `FCOS_QUEUE=rt_HG`: queues

## Outputs

Pretrain checkpoints:

```text
output/nerf_mae/results/<pretrain_save_name>/epoch_300.pt
```

FCOS evals:

```text
output/nerf_rpn/results/<pretrain_save_name>_epoch300_sched_epoch_seed<seed>_fcos1000_eval/eval.json
```
