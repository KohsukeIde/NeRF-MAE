# Anchor-head low-label result validity note (2026-06-24)

The Front3D 10% anchor-head comparison in `anchor_head_lowlabel_results_20260624.*` is **invalid for paper claims** and should not be used as negative evidence about structure-first transfer.

Reasons:

1. The detector head is near the noise floor for all arms: AP@50 is ~0 for scratch, joint, and structure-first, far below the working FCOS 10% protocol.
2. The high recall / near-zero AP pattern indicates a detector scoring/ranking failure, not a representation comparison.
3. Pretrained arms underperform architecture-matched scratch, so checkpoint/backbone adaptation and/or training protocol must be gated before arm comparison.
4. The run used a low-label anchor setting without first proving that the anchor pipeline reaches a sane operating point under this local protocol.

Action:

- Exclude these numbers from the main paper and rebuttal-facing claims.
- Run anchor sanity gates before any further detector-head breadth comparison:
  - official anchor scratch, Front3D 100% labels, seed1;
  - MAE-compatible scratch, Front3D 100% labels, seed1.
- Only if the MAE-compatible scratch sanity reaches a non-collapsed AP/Recall regime should joint/structure-first pretrained arms be compared.
