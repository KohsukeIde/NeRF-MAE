# ABCI3 Pretrain Speed Benchmark

This harness submits short, isolated NeRF-MAE pretrain jobs for comparing ABCI3 scaling:

- `1n4g`: one `rt_HF` node using GPUs `0-3`
- `1n8g`: one `rt_HF` node using GPUs `0-7`
- `2n16g`: two `rt_HF` nodes using GPUs `0-7` on each node

The default grid runs deterministic on/off, staging off/on, global batch `16`, and a throughput-oriented global batch `64`. Jobs use the existing pretrain PBS path, but write under a benchmark-only suffix:

```text
output/nerf_mae/results/*_abci3speed_<run_id>_<topology>_gb<batch>_det<0|1>
```

These names are distinct from `abci3clean`, `abci3gb16`, and the main experiment suffixes.

## Dry Run

Dry-run is the default and does not submit jobs:

```bash
cd /groups/gag51404/ide/vgi/NeRF-MAE
SKIP_PREFLIGHT=1 \
bash nerf_mae/probe_scripts/submit_abci3_pretrain_speed_benchmark.sh
```

Remove `SKIP_PREFLIGHT=1` to validate `qsub`, the Python environment, and pretrain data before printing commands.

## Submit

Submit only when the printed grid is what you want:

```bash
cd /groups/gag51404/ide/vgi/NeRF-MAE
DRY_RUN=0 \
BENCH_RUN_ID=20260520_speed01 \
bash nerf_mae/probe_scripts/submit_abci3_pretrain_speed_benchmark.sh
```

Useful overrides:

- `BENCH_GLOBAL_BATCHES="16 64"`: global batches to test
- `BENCH_DETERMINISTIC_VALUES="1 0"`: deterministic on/off
- `BENCH_STAGE_PRETRAIN_VALUES="0 1"`: shared filesystem vs node-local staged pretrain data
- `BENCH_SLOTS=1`: number of concurrent dependency chains; keep this low to avoid staging/I/O interference
- `BENCH_DEPENDENCY_TYPE=afterany`: continue the chain even if an earlier benchmark row fails
- `BENCH_EPOCHS=1`: short benchmark length
- `PRETRAIN_LOG_INTERVAL=1`: log every step for cleaner timing
- `PRETRAIN_PROFILE_STEP_TIME=1`: include explicit `step_time: ...s` in `worker_0.log`
- `PRETRAIN_WALLTIME=02:00:00`: PBS walltime
- `BENCH_TOPOLOGIES="1n4g:1:0-3 1n8g:1:0-7 2n16g:2:0-7"`: topology grid
- `PRETRAIN_TRAIN_NUM_WORKERS=4 PRETRAIN_PERSISTENT_WORKERS=1`: test a higher DataLoader worker setting
- `ABCI3_CUDA_MODULE=<module>` plus `PROBE_ENV_PREFIX=<venv>`: compare the current cu118 environment with a CUDA 12.x/PyTorch cu12x environment

The submitter writes a manifest at:

```text
output/launcher/abci3_pretrain_speed_bench/<run_id>/manifest.tsv
```

## Parse Speed

After jobs finish, parse `sec/step` from each result's `log/worker_0.log`:

```bash
python nerf_mae/tools/parse_pretrain_speed_log.py \
  --manifest output/launcher/abci3_pretrain_speed_bench/20260520_speed01/manifest.tsv \
  --warmup-steps 10 \
  --md-out output/launcher/abci3_pretrain_speed_bench/20260520_speed01/summary.md \
  --csv-out output/launcher/abci3_pretrain_speed_bench/20260520_speed01/summary.csv \
  --json-out output/launcher/abci3_pretrain_speed_bench/20260520_speed01/summary.json
```

Warmup exclusion omits logged steps at or before `--warmup-steps` when explicit `step_time: ...s` entries are present. For older logs without explicit step timing, it omits timestamp-delta intervals whose starting completed-step count is below `--warmup-steps`. The parser only uses training lines in `worker_0.log`, so it does not include model construction, data loading before the first logged step, final eval, or checkpoint save time.

## Recommended Decision Order

Run the default grid first. If `1n8g` is close to `2n16g` at global batch 16, prefer single-node for production unless the throughput batch grid clearly reverses that. Then add `STAGE_PRETRAIN_DATA=1` to the best one-node and two-node candidates; if staging improves `data_wait`, keep it for long e1200 runs. Finally, repeat the winning one or two candidates in a CUDA 12.x/PyTorch cu12x environment before committing e1200 budget.
