# Low-Label Expansion and Boundary-SDF B1 Results

Date: 2026-06-02 JST

## Completed Jobs

All low-label expansion jobs and the Boundary-SDF B1 scout have completed.
No matching jobs remain in `qstat`.

## Low-Label Front3D FCOS Results

All rows use finetune seed 1.

| condition | label fraction | AP@25 | AP@50 | AP@75 | R@50 top300 | R@50 top1000 |
|---|---:|---:|---:|---:|---:|---:|
| scratch | 10% | 0.4453 | 0.1160 | 0.0000 | 0.2794 | 0.3088 |
| baseline_e300 | 10% | 0.5344 | 0.1328 | 0.0001 | 0.3824 | 0.4265 |
| cosine_ramp_e300 | 10% | 0.5996 | 0.2751 | 0.0152 | 0.4485 | 0.4706 |
| surface_cosine_jitter_e300 | 10% | 0.5918 | 0.1756 | 0.0066 | 0.4118 | 0.4412 |
| scratch | 25% | 0.6087 | 0.3044 | 0.0122 | 0.4779 | 0.4779 |
| baseline_e300 | 25% | 0.6550 | 0.2777 | 0.0057 | 0.5147 | 0.5221 |
| cosine_ramp_e300 | 25% | 0.7008 | 0.3639 | 0.0095 | 0.5882 | 0.6029 |
| surface_cosine_jitter_e300 | 25% | 0.7123 | 0.3460 | 0.0200 | 0.5368 | 0.5441 |
| scratch | 50% | 0.7065 | 0.3666 | 0.0513 | 0.5956 | 0.5956 |
| baseline_e300 | 50% | 0.7671 | 0.4191 | 0.0241 | 0.6471 | 0.6544 |
| cosine_ramp_e300 | 50% | 0.7690 | 0.5026 | 0.0516 | 0.6691 | 0.6765 |
| surface_cosine_jitter_e300 | 50% | 0.7811 | 0.5217 | 0.0627 | 0.6765 | 0.6838 |
| scratch | 100% | 0.7952 | 0.4722 | 0.0703 | 0.6176 | 0.6324 |
| baseline_e300 | 100% | 0.7956 | 0.4695 | 0.0869 | 0.6618 | 0.6691 |
| cosine_ramp_e300 | 100% | 0.8249 | 0.5539 | 0.1135 | 0.7059 | 0.7059 |
| surface_cosine_jitter_e300 | 100% | 0.8178 | 0.5984 | 0.1004 | 0.7059 | 0.7279 |

## Low-Label Read

- The label-efficiency signal survives beyond the original 50% gate.
- At 10% labels, `cosine_ramp_e300` is the clear winner:
  AP@50 is `0.2751`, improving over scratch by `+0.1591` and over
  baseline_e300 by `+0.1423`.
- At 25% labels, `cosine_ramp_e300` is also best by AP@50:
  `0.3639`, improving over scratch by `+0.0595` and over baseline_e300 by
  `+0.0862`.
- At 50% and 100% labels, `surface_cosine_jitter_e300` is best by AP@50:
  `0.5217` at 50% and `0.5984` at 100%.
- The clean paper framing should not present `cosine_ramp` and
  `surface_cosine_jitter` as two equal methods. The safer hierarchy is:
  `cosine_ramp` as the base label-efficient recipe, with surface/jitter as an
  in-domain or label-richer anchoring component.

## Boundary-SDF B1 e100 Scout

| condition | epoch | AP@25 | AP@50 | AP@75 | R@50 top300 | R@50 top1000 |
|---|---:|---:|---:|---:|---:|---:|
| boundary_sdf_aux | 100 | 0.8110 | 0.5142 | 0.1031 | 0.6618 | 0.6691 |
| baseline_coord_jitter | 100 | 0.8197 | 0.5564 | 0.1015 | 0.6765 | 0.6912 |
| cosine_coord_jitter | 100 | 0.8097 | 0.6219 | 0.1031 | 0.7279 | 0.7279 |

## Boundary-SDF Decision

- B1 completed successfully, but it does not clear the e100 promotion gate.
- Compared with `baseline_coord_jitter_e100`, Boundary-SDF is lower on AP@50
  and R@50, with only a tiny AP@75 difference.
- Compared with `cosine_coord_jitter_e100`, Boundary-SDF is much lower on
  AP@50 and recall.
- Do not promote Boundary-SDF to e300 now.
- Keep Boundary-SDF as an audited, plausible-but-not-yet-winning method branch
  for appendix/future or for a later exact-distance implementation only if
  paper strategy requires a new mechanism.
