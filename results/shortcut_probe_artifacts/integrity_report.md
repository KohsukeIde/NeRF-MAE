# Shortcut Probe Integrity Report

Generated: 2026-05-25 JST

## Public Git State

Local HEAD:

```text
6dff2390a647c6f1762ba8288466951b5beb1b9a
```

Origin main:

```text
6dff2390a647c6f1762ba8288466951b5beb1b9a	refs/heads/main
```

Commit stat:

```text
6dff239 Add D-MAE coord-jitter scout and integrity diagnostics
 SHORTCUT_PROBE_EXPERIMENT_LOG.md                   | 131 +++++++
 nerf_mae/probe_scripts/abci3_e300_gate_fcos.pbs    |   2 +-
 .../probe_scripts/abci3_e300_gate_pretrain.pbs     |  17 +-
 .../submit_abci3_e300_gate_pipeline.sh             |   3 +-
 nerf_mae/tools/build_results_table.py              |  20 +-
 nerf_mae/tools/check_fcos_checkpoint_load.py       | 243 ++++++++++++
 nerf_rpn/tools/abci3_proposal_quality_summary.pbs  |  55 +++
 nerf_rpn/tools/summarize_proposal_quality.py       | 307 ++++++++++++++++
 ...ae_hier_concat_e100_seed1_fcos_load_sanity.json |  88 +++++
 ...dmae_hier_concat_e100_seed1_fcos_load_sanity.md |  34 ++
 .../proposal_quality/e100_dmae_coord_controls.json | 272 ++++++++++++++
 .../proposal_quality/e100_dmae_coord_controls.md   |  59 +++
 .../proposal_quality/e300_gate_pre1_ft123.json     | 407 +++++++++++++++++++++
 .../proposal_quality/e300_gate_pre1_ft123.md       |  86 +++++
 results/shortcut_probe_artifacts/results_table.csv | 206 +++++------
 15 files changed, 1820 insertions(+), 110 deletions(-)
```

## Code Checks

Passed:

```bash
python -m py_compile \
  nerf_mae/model/mae/shortcut_probe.py \
  nerf_mae/run_swin_mae3d.py \
  nerf_rpn/model/feature_extractor.py \
  nerf_mae/tools/check_fcos_checkpoint_load.py \
  nerf_rpn/tools/summarize_proposal_quality.py
bash -n nerf_rpn/tools/abci3_proposal_quality_summary.pbs
bash -n nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs
bash -n nerf_mae/probe_scripts/abci3_e300_gate_fcos.pbs
bash -n nerf_mae/probe_scripts/submit_abci3_e300_gate_pipeline.sh
```

## D-MAE FCOS Load Sanity

Checkpoint:

```text
/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_dmae_hier_concat_p1.0_e100_seed1_abci3dmae_e100_det0_1n8g/epoch_100.pt
```

Sanity report:

- JSON: `results/shortcut_probe_artifacts/load_sanity/dmae_hier_concat_e100_seed1_fcos_load_sanity.json`
- Markdown: `results/shortcut_probe_artifacts/load_sanity/dmae_hier_concat_e100_seed1_fcos_load_sanity.md`

Result:

| check | value |
|---|---:|
| FCOS instantiated | true |
| pass | true |
| missing keys | 2 |
| unexpected keys | 8 |
| encoder missing keys | 0 |
| encoder unexpected keys | 0 |
| encoder exact tensor ratio | 1.000000 |
| encoder exact numel ratio | 1.000000 |

Encoder exact-load details:

| prefix | exact tensors | total tensors | exact numel ratio |
|---|---:|---:|---:|
| `pos_embed` | 1 | 1 | 1.000000 |
| `patch_partition` | 4 | 4 | 1.000000 |
| `stages.*` | 345 | 345 | 1.000000 |

Expected non-strict keys:

- Missing:
  - `out.conv.weight`
  - `out.conv.bias`
- Unexpected:
  - `decomp_structure_head.0.weight`
  - `decomp_structure_head.0.bias`
  - `decomp_structure_head.3.weight`
  - `decomp_structure_head.3.bias`
  - `decomp_structure_head.6.weight`
  - `decomp_structure_head.6.bias`
  - `decomp_rgb_head.weight`
  - `decomp_rgb_head.bias`

Interpretation:

The downstream FCOS feature extractor loads the D-MAE pretrained encoder exactly for `pos_embed`, `patch_partition`, and all `stages.*` tensors. D-MAE-specific heads are discarded as expected, so the D-MAE downstream AP is not explained by failed backbone loading.

## FCOS Eval Config Anchor

`dmae_hier_concat e100 seed1` row from `results_table.csv`:

| field | value |
|---|---|
| condition | `dmae_hier_concat` |
| pretrain seed | `1` |
| finetune seed | `1` |
| epoch | `100` |
| dataset | `front3d` |
| scheduler | `onecycle_epoch` |
| checkpoint | `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_dmae_hier_concat_p1.0_e100_seed1_abci3dmae_e100_det0_1n8g/epoch_100.pt` |
| eval path | `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_rpn/results/nerfmae_dmae_hier_concat_p1.0_e100_seed1_abci3dmae_e100_det0_1n8g_epoch100_sched_epoch_seed1_fcos1000_eval/eval.json` |
| AP@50 | `0.5777786374` |
| AP@75 | `0.1055305004` |
| Recall@50 top300 | `0.6911764741` |
| git hash column | `6dff2390a647c6f1762ba8288466951b5beb1b9a` |

Note:

The `git_hash` column records the commit used to generate this aggregation table. The underlying D-MAE scout was run before this integrity commit, but the D-MAE implementation itself is already present in the committed history and the current load-sanity check was run at `6dff2390a647c6f1762ba8288466951b5beb1b9a`.

## Decision Criteria

Do not use a flat "2 of 3" rule for D-MAE continuation. Use the following hierarchy once `baseline_coord_jitter` and `dmae_hier_concat_coord_jitter` are complete.

Tier 1, strong method-paper path:

- `dmae_hier_concat_coord_jitter` is at least equal to `cosine_coord_jitter` on AP@50.
- It also improves AP@75 or AP75/AP50.

Tier 2, defensible localization-method path:

- `dmae_hier_concat_coord_jitter` trails `cosine_coord_jitter` on AP@50 by no more than `0.02`.
- It clearly improves AP@75/AP@50 or localization diagnostics.
- Frame as localization fidelity rather than pure AP@50 sample efficiency.

Tier 3, drop D-MAE as main method:

- `dmae_hier_concat_coord_jitter` trails `cosine_coord_jitter` by at least `0.03` AP@50.
- It does not win on AP@75 or AP75/AP50.
- It does not improve over `baseline_coord_jitter`.

Pending jobs:

- `baseline_coord_jitter` FCOS retry: `1796104.pbs1`
- `dmae_hier_concat_coord_jitter` pretrain: `1797098.pbs1`
- dependent `dmae_hier_concat_coord_jitter` FCOS: `1797099.pbs1`
