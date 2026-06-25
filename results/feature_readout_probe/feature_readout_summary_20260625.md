# Feature Readout Probe Summary (2026-06-25)

Inputs:
- `readout_p10_main_20260625`
- `readout_p100_main_20260625`

Protocol:
- frozen MAE encoder stages only (`patch_partition + pos_embed + stages`)
- no detector head, no FPN neck
- balanced sampling for linear readout, so `balanced_ap` is not detector AP
- targets: OBB objectness, occupancy, denoised shell

## 10% labels

Objectness balanced AP:

| arm | stage0 | stage1 | stage2 |
|---|---:|---:|---:|
| scratch | 0.7814 | 0.5849 | 0.5677 |
| joint | 0.8017 | 0.5890 | 0.4932 |
| cosine | 0.8029 | 0.6097 | 0.5254 |
| linear | 0.8074 | 0.5819 | 0.5202 |
| w=0.5 | 0.8080 | 0.5779 | 0.4975 |
| occupancy-only | 0.6770 | 0.2994 | 0.1181 |
| shuffle | 0.7553 | 0.5169 | 0.3459 |

Occupancy readout: cosine/linear/joint/w=0.5 are all very high; cosine is best at stage0/1.
Shell readout: cosine is best at stage0/1; linear is slightly best at stage2.

## 100% labels

Objectness balanced AP:

| arm | stage0 | stage1 | stage2 |
|---|---:|---:|---:|
| scratch | 0.7996 | 0.6174 | 0.6024 |
| joint | 0.8087 | 0.6391 | 0.5660 |
| cosine | 0.8142 | 0.6514 | 0.5180 |
| linear | 0.8170 | 0.6399 | 0.5496 |
| w=0.5 | 0.8163 | 0.5996 | 0.4041 |
| occupancy-only | 0.7802 | 0.4434 | 0.1855 |
| shuffle | 0.7776 | 0.5796 | 0.3874 |

Occupancy/shell readout again ranks cosine/linear near top and clearly above scratch/shuffle/occupancy-only.

## Interpretation

Positive:
- The probe is valid mechanically: frozen encoder only, no FPN/head, target generation sanity passed.
- Structure-first features make occupancy and shell almost linearly readable, especially stage0/1.
- Occupancy-only and shuffle are poor on objectness and deeper-stage objectness, supporting that intact structure plus later appearance is not equivalent to occupancy-only or shuffled target.
- Cosine is best for stage1 objectness in both 10% and 100% settings.

Caveat:
- Objectness readout does not exactly reproduce downstream AP ordering. Stage0 objectness has w=0.5/linear/cosine/joint very close, with w=0.5 or linear slightly above cosine.
- Scratch objectness is already strong, especially at stage2, so do not claim that the frozen readout alone recovers most low-label detector gain.

Recommended paper use:
- Use this as mechanism-depth evidence, not as a decisive causal proof.
- Safe claim: "Structure-first pretraining makes structural occupancy/shell and mid-level objectness more linearly readable; this representation-level trend is consistent with, but does not fully explain, the downstream label-efficiency gains."
- Avoid: "feature readout exactly matches downstream ordering" or "frozen linear readout recovers most of the detector gain."
