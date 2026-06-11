# ScanNet normalize_density Audit

Snapshot: 2026-06-11 JST

## Corrected conclusion

Existing ScanNet transfer triage runs are acceptable with respect to density
normalization. Although their shell commands include `--normalize_density`, the
ScanNet code path does not use this CLI flag to choose whether to transform the
input density channel.

For `dataset=scannet`, the FCOS/RPN trainers instantiate `ScanNetRPNDataset`
without passing `args.normalize_density`. `ScanNetRPNDataset` always converts the
raw ScanNet density channel to alpha using its ScanNet-specific ReLU-based
formula:

```text
activation = clip(density, min=0)
alpha = 1 - exp(-activation / 100)
```

Therefore:

- The previous claim that the paper-facing ScanNet table requires a
  non-normalized rerun was too strong.
- The existing ScanNet table should not be discarded as a wrong
  `normalize_density=on` protocol.
- The attempted `NORMALIZE_DENSITY=0` rerun does not create a true
  non-normalized ScanNet condition unless the ScanNet dataset class itself is
  changed.

## Evidence

Local implementation:

- `nerf_rpn/run_fcos_pretrained.py`:
  - Front3D/Hypersim pass `normalize_density=self.args.normalize_density`.
  - ScanNet calls `ScanNetRPNDataset(self.train_scenes, features_path,
    boxes_path, ...)` without passing the flag.
- `nerf_rpn/run_fcos.py` and `nerf_rpn/run_rpn.py` follow the same pattern.
- `nerf_rpn/datasets.py`:
  - `ScanNetRPNDataset.__init__()` calls `super(..., normalize_density=False,
    ...)`, loads data, then always overwrites the density channel with
    `self.density_to_alpha(density)`.

Official release parity:

- The public NeRF-RPN `ScanNetRPNDataset` has the same behavior: it calls the
  base dataset with `normalize_density=False`, then always applies the
  ScanNet-specific density-to-alpha conversion.
- The public NeRF-RPN sample `train_fcos.sh` / `test_fcos.sh` scripts include
  `--normalize_density` for the released Front3D example.
- The public NeRF-MAE `train_fcos_pretrained.sh` / `test_fcos_pretrained.sh`
  scripts also include `--normalize_density` for the released Front3D example.

Source URLs checked:

- `https://raw.githubusercontent.com/lyclyc52/NeRF_RPN/main/nerf_rpn/datasets.py`
- `https://raw.githubusercontent.com/lyclyc52/NeRF_RPN/main/nerf_rpn/train_fcos.sh`
- `https://raw.githubusercontent.com/lyclyc52/NeRF_RPN/main/nerf_rpn/test_fcos.sh`
- `https://raw.githubusercontent.com/zubair-irshad/NeRF-MAE/main/nerf_rpn/datasets.py`
- `https://raw.githubusercontent.com/zubair-irshad/NeRF-MAE/main/nerf_rpn/train_fcos_pretrained.sh`
- `https://raw.githubusercontent.com/zubair-irshad/NeRF-MAE/main/nerf_rpn/test_fcos_pretrained.sh`

## Existing ScanNet results

The existing ScanNet triage logs include `--dataset scannet` and
`--normalize_density`, but the flag is ignored by the ScanNet dataset branch.
Those results should be treated as the standard local ScanNet FCOS transfer
triage unless another protocol issue is identified.

Existing result artifact:

- `results/shortcut_probe_artifacts/scannet_transfer_triage.md`
- `results/shortcut_probe_artifacts/scannet_transfer_triage.csv`

## Canceled rerun

Four `NORMALIZE_DENSITY=0` ScanNet train+eval reruns were submitted after the
first audit:

| condition | initial job | retry job |
|---|---:|---:|
| `baseline_e300` | `1897280.pbs1` | `1897293.pbs1` |
| `cosine_ramp_e300` | `1897281.pbs1` | `1897294.pbs1` |
| `cosine_coord_jitter_e100` | `1897282.pbs1` | `1897295.pbs1` |
| `surface_cosine_jitter_e300` | `1897283.pbs1` | `1897296.pbs1` |

The initial jobs failed from a relative checkpoint-path bug. The retry jobs
started successfully, but were canceled after this corrected audit because they
would not test a distinct ScanNet normalization protocol.

## Decision

- Use the existing ScanNet triage table for the current paper-facing analysis
  unless a different ScanNet protocol issue is found.
- Keep `NORMALIZE_DENSITY` controls in the Front3D/general wrappers because the
  flag is operational there.
- Do not spend additional compute on ScanNet `normalize_density` on/off reruns
  unless we intentionally implement a new ScanNet raw-density ablation.
