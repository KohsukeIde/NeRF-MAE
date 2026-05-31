# ScanNet Transfer Triage

Snapshot: 2026-05-31 JST

Protocol:
- Dataset: public NeRF-RPN ScanNet OBB detection archive.
- Split: `scannet_split.npz`, 60 train / 15 val / 15 test scenes.
- FCOS transfer: 1000 epochs, single finetune seed, batch size 1 per GPU.
- Purpose: cross-dataset single-seed triage only. Do not treat as a final
  multi-seed paper table.

| condition | AP@25 | AP@50 | AP@75 | R@50 top300 | R@50 top1000 |
|---|---:|---:|---:|---:|---:|
| `baseline_e300` | 0.5013 | 0.1898 | 0.0024 | 0.3596 | 0.3596 |
| `cosine_ramp_e300` | 0.5883 | 0.1912 | 0.0006 | 0.4039 | 0.4187 |
| `cosine_coord_jitter_e100` | 0.5540 | 0.1864 | 0.0022 | 0.3695 | 0.3744 |
| `surface_cosine_jitter_e300` | 0.5759 | 0.1782 | 0.0014 | 0.3596 | 0.3645 |

Reading:
- `cosine_ramp_e300` is the best ScanNet row on AP@50, but the margin over
  `baseline_e300` is only `+0.0014`, effectively a single-seed tie.
- `cosine_ramp_e300` does improve AP@25 and recall, suggesting a coarse
  proposal/objectness benefit on ScanNet.
- `cosine_coord_jitter_e100` and `surface_cosine_jitter_e300` do not improve
  AP@50 over `baseline_e300` on ScanNet.
- This weakens the claim that the Front3D AP@50 gain directly transfers across
  datasets. The safer current statement is that the curriculum signal may
  improve coarse ScanNet recall/AP@25, while AP@50 generalization is not yet
  established.

Next decision:
- Do not expand ScanNet to multi-seed unless we decide that AP@25/recall is a
  paper-relevant axis.
- Wait for `paper_loss_e300`; if objective-fidelity shows a material effect,
  rerun the relevant objective on ScanNet instead of expanding these older
  variants.
