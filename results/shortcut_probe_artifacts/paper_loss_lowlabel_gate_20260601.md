# Paper-Loss and Low-Label Gate

Snapshot: 2026-06-01 JST

Purpose:
- Record the completed `paper_loss_e300` kill experiment and the first
  Front3D 50% low-label gate using the same scalar extraction convention as
  the other OBB detection artifacts.
- Treat all rows here as single-finetune-seed gates unless explicitly noted.

## Completed Jobs

| condition | pretrain job | FCOS/eval job | status |
|---|---:|---:|---|
| `paper_loss_e300` det0 rerun | `1812733.pbs1` | `1812734.pbs1` | complete |
| `scratch_lowlabel_50` | n/a | `1812644.pbs1` | complete |
| `baseline_e300_lowlabel_50` | existing ckpt | `1812645.pbs1` | complete |
| `cosine_ramp_e300_lowlabel_50` | existing ckpt | `1812646.pbs1` | complete |
| `surface_cosine_jitter_e300_lowlabel_50` | existing ckpt | `1812647.pbs1` | complete |

## Metrics

| condition | protocol | AP@25 | AP@50 | AP@75 | Recall@25 top300 | Recall@50 top300 |
|---|---|---:|---:|---:|---:|---:|
| `paper_loss_e300` | full-label | 0.7949 | 0.5613 | 0.0742 | 0.9632 | 0.6912 |
| `scratch_lowlabel_50` | 50% labels | 0.7065 | 0.3666 | 0.0513 | 0.9338 | 0.5956 |
| `baseline_e300_lowlabel_50` | 50% labels | 0.7671 | 0.4191 | 0.0241 | 0.9559 | 0.6471 |
| `cosine_ramp_e300_lowlabel_50` | 50% labels | 0.7690 | 0.5026 | 0.0516 | 0.9412 | 0.6691 |
| `surface_cosine_jitter_e300_lowlabel_50` | 50% labels | 0.7811 | 0.5217 | 0.0627 | 0.9559 | 0.6765 |

## Immediate Read

- `paper_loss_e300` is not a collapse. It is clearly above `baseline_e300`
  mean AP@50 from the ABCI clean 3-finetune-seed table (`0.4938`) and close to
  `cosine_ramp_e300` mean AP@50 (`0.5723`), but it does not beat the strongest
  compute-efficiency rows.
- This weakens the strong objective-mismatch route. The released all-occupied
  RGB objective is not necessary for nontrivial transfer, and the paper-like
  removed-occupied RGB objective is not an obvious simple fix.
- The 50% low-label gate supports the sample-efficiency direction:
  `surface_cosine_jitter_e300` improves AP@50 over scratch by `+0.1551`, over
  `baseline_e300` by `+0.1026`, and over `cosine_ramp_e300` by `+0.0191`.
- The low-label rows are still single finetune seed. They are useful as a
  direction gate, not as final paper statistics.

## Decision

- Do not launch visible-only / masked-only occupied RGB decomposition solely
  from the current `paper_loss_e300` result; the difference is not decisive
  enough to justify another objective grid.
- Promote low-label detection as the next validation axis if compute is spent:
  expand only the most relevant rows, not every historical variant.
- If a main-table claim is made, prioritize:
  `baseline_e300`, `cosine_ramp_e300`, `surface_cosine_jitter_e300`, and
  `scratch`/NeRF-RPN anchors at 50% labels, then optionally 25% labels.
