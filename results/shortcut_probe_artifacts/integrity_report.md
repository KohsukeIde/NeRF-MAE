# Shortcut Probe Integrity Report

Generated: 2026-05-25 JST

## Public Git State

Checked D-MAE integrity commit:

```text
6dff2390a647c6f1762ba8288466951b5beb1b9a
```

Origin main immediately after publishing that commit:

```text
6dff2390a647c6f1762ba8288466951b5beb1b9a	refs/heads/main
```

This report itself is committed after the checked D-MAE integrity commit, so
`origin/main` may be newer than the checked commit. The checked commit remains
the code/artifact commit whose stat is recorded below.

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

## Coord-Jitter Decision Results

Checked on 2026-05-26 JST. The previously pending jobs completed:

- `baseline_coord_jitter` FCOS retry: `1796104.pbs1`
- `dmae_hier_concat_coord_jitter` pretrain: `1797098.pbs1`
- dependent `dmae_hier_concat_coord_jitter` FCOS: `1797099.pbs1`

Results table:

| condition | AP@50 | AP@75 | AP75/AP50 | R50@300 |
|---|---:|---:|---:|---:|
| `baseline_coord_jitter` | 0.5564 | 0.1015 | 0.1824 | 0.6765 |
| `cosine_coord_jitter` | 0.6219 | 0.1031 | 0.1657 | 0.7279 |
| `dmae_hier_concat` | 0.5778 | 0.1055 | 0.1826 | 0.6912 |
| `dmae_hier_concat_coord_jitter` | 0.5212 | 0.0858 | 0.1646 | 0.6838 |

Key deltas:

- `cosine_coord_jitter - baseline_coord_jitter`: AP@50 `+0.0655`, AP@75 `+0.0016`, R50@300 `+0.0515`.
- `dmae_hier_concat_coord_jitter - cosine_coord_jitter`: AP@50 `-0.1007`, AP@75 `-0.0173`, AP75/AP50 `-0.0011`, R50@300 `-0.0441`.
- `dmae_hier_concat_coord_jitter - baseline_coord_jitter`: AP@50 `-0.0352`, AP@75 `-0.0157`, AP75/AP50 `-0.0178`, R50@300 `+0.0074`.

Decision:

- `baseline_coord_jitter` does not explain away `cosine_coord_jitter`. The
  cosine target-alpha-to-RGBA curriculum still provides a large AP@50 gain under
  coord-jitter.
- `dmae_hier_concat_coord_jitter` lands in Tier 3: it trails
  `cosine_coord_jitter` by more than `0.03` AP@50 and does not win on AP@75 or
  AP75/AP50.
- D-MAE should not be promoted as the main method path from this scout. Keep
  `dmae_hier_concat` as an ablation/localization diagnostic, but move the main
  paper direction toward cosine/coord-jitter curriculum analysis unless a
  substantially different D-MAE design is proposed.

Additional proposal-quality summary:

- Ran `1800305.pbs1` with
  `nerf_rpn/tools/abci3_proposal_quality_coord_jitter_decision.pbs`.
- Outputs:
  - `results/shortcut_probe_artifacts/proposal_quality/e100_coord_jitter_decision.json`
  - `results/shortcut_probe_artifacts/proposal_quality/e100_coord_jitter_decision.md`
  - `results/shortcut_probe_artifacts/proposal_quality/e100_coord_jitter_decision.png`

Proposal-quality table:

| condition | AP@50 | AP@75 | AP75/AP50 | R50@300 | mean IoU | frac IoU>=0.5 | first TP rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_coord_jitter` | 0.5564 | 0.1015 | 0.1824 | 0.6765 | 0.0632 | 0.0180 | 1.2941 |
| `cosine_coord_jitter` | 0.6219 | 0.1031 | 0.1657 | 0.7279 | 0.0635 | 0.0196 | 1.0588 |
| `dmae_hier_concat` | 0.5778 | 0.1055 | 0.1826 | 0.6912 | 0.0649 | 0.0184 | 1.3529 |
| `dmae_hier_concat_coord_jitter` | 0.5212 | 0.0858 | 0.1646 | 0.6838 | 0.0702 | 0.0182 | 1.2941 |

Reading:

- `dmae_hier_concat_coord_jitter` has the highest mean proposal IoU, but it
  does not convert that into AP@50, AP@75, or AP75/AP50.
- `cosine_coord_jitter` remains strongest on AP@50, R50@300, frac IoU>=0.5,
  and first-TP rank.
- The proposal diagnostic does not overturn the Tier 3 D-MAE decision.
