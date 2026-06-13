# Qualitative Detection Assets

This directory stores paper-facing qualitative OBB overlays and the intermediate
images used to select them.

## What To Inspect First

Use these for figure selection:

```text
final/
paper_shortlists/gt_10/
```

`final/` contains the current recommended subset copied from the GT shortlist.
`paper_shortlists/gt_10/` contains 10 readable candidates with panels ordered as:

```text
Front3D: GT / scratch / NeRF-MAE† / structure-first
ScanNet: GT / NeRF-MAE† / structure-first
```

The contact sheet is:

```text
paper_shortlists/gt_10/contact_sheet_gt_10.png
```

The older no-GT shortlist is kept only for reference:

```text
paper_shortlists/no_gt_10/
```

## Directory Roles

```text
final/             selected paper-facing exports only
paper_shortlists/  candidate sets for human selection
rankings/          approximate-IoU scene ranking CSVs used for candidate search
render_cache/      downloaded NeRF-RPN RGB render data and camera transforms
work_archive/      old drafts, rejected candidates, contact sheets, debug views
```

`work_archive/` is not paper-facing. It is kept so previous visual decisions can
be audited without cluttering the selection directories.

## Inputs

Front3D uses released NeRF-RPN RGB renders and camera transforms:

```text
render_cache/front3d_render_data/front3d_nerf_data/<scene>/train/images/*.jpg
render_cache/front3d_render_data/front3d_nerf_data/<scene>/train/transforms.json
```

Front3D GT boxes come from `transforms.json` `bounding_boxes`.

Predicted proposal dumps:

```text
scratch:   output/nerf_rpn/results/front3d_scratch_lowlabel_pt100_seed1_fcos1000_eval/proposals
NeRF-MAE†: output/nerf_rpn/results/budgetcurve_baseline_e1200_seed1_fcos1000_eval/proposals
ours:      output/nerf_rpn/results/budgetcurve_cosine_ramp_e600_seed1_fcos1000_eval/proposals
```

ScanNet uses public `scannet_rpn_data` GT/features/proposals plus released
`scannet_nerf_data` RGB views/camera transforms. Proposals and GT are converted
from RPN grid coordinates back to ScanNet world coordinates with each scene's
`bbox_min`, `bbox_max`, and `resolution`, then projected with the same convention
as `data/scannet/visualize_bbox.py`.

```text
GT:        dataset/finetune/scannet_rpn_data/obb
NeRF-MAE†: output/nerf_rpn/results/baseline_e300_scannet_fcos1000_seed1_eval/proposals
ours:      output/nerf_rpn/results/cosine_ramp_e300_scannet_fcos1000_seed1_eval/proposals
```

No local ScanNet scratch proposal dump was found when these figures were built.
If a ScanNet scratch dump is generated, add it to `SCANNET_METHODS` in
`figures_src/build_qualitative_detection_results.py`.

## Reproduction

Download selected render scenes:

```bash
python figures_src/download_front3d_render_scene.py \
  --scene 3dfront_0143_00 \
  --output-dir figures_src/qualitative_detection_assets/render_cache/front3d_render_data

python figures_src/download_scannet_render_scene.py \
  --scene scene0151_00 \
  --output-dir figures_src/qualitative_detection_assets/render_cache/scannet_render_data
```

Build one qualitative panel:

```bash
python figures_src/build_qualitative_detection_results.py \
  --front3d-scene 3dfront_0143_00 \
  --front3d-frame 0015 \
  --front3d-extra-frame "" \
  --scannet-scene scene0151_00 \
  --scannet-frame 2634 \
  --top-k 3 \
  --score-threshold 0.35
```

Regenerate a ranked/readable candidate pool:

```bash
python figures_src/select_qualitative_detection_wins.py \
  --dataset front3d \
  --gt-dir dataset/finetune/front3d_rpn_data/obb \
  --baseline-dir output/nerf_rpn/results/budgetcurve_baseline_e1200_seed1_fcos1000_eval/proposals \
  --ours-dir output/nerf_rpn/results/budgetcurve_cosine_ramp_e600_seed1_fcos1000_eval/proposals \
  --output figures_src/qualitative_detection_assets/rankings/front3d_ours_wins_ranked.csv

python figures_src/generate_qualitative_win_candidates.py \
  --output-dir figures_src/qualitative_detection_assets/readable_win_candidates_gt
```

The ranking uses approximate CPU 3D OBB IoU only for qualitative candidate
selection. It is not a replacement for the official detection metrics.
