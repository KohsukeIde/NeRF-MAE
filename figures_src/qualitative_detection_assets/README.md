# Qualitative Detection Assets

These assets fill the paper placeholder:

```text
Figure 5: Qualitative detection results.
Predicted oriented boxes on Front3D and ScanNet scenes:
scratch vs. NeRF-MAE† vs. structure-first, same scenes.
```

## Current Draft

```text
final/fig5_qualitative_detection_draft.png
final/front3d_3dfront_0143_00_0015_pred_obb_threeway.png
final/front3d_3dfront_0143_00_0172_pred_obb_threeway.png
final/scannet_scene0151_00_2634_pred_obb_rgb.png
```

## Inputs

Front3D uses released NeRF-RPN RGB renders and camera transforms:

```text
front3d_render_data/front3d_nerf_data/3dfront_0143_00/train/images/0172.jpg
front3d_render_data/front3d_nerf_data/3dfront_0143_00/train/transforms.json
```

Predicted proposal dumps:

```text
scratch:   output/nerf_rpn/results/front3d_scratch_lowlabel_pt100_seed1_fcos1000_eval/proposals
NeRF-MAE†: output/nerf_rpn/results/budgetcurve_baseline_e1200_seed1_fcos1000_eval/proposals
ours:      output/nerf_rpn/results/budgetcurve_cosine_ramp_e600_seed1_fcos1000_eval/proposals
```

ScanNet uses public `scannet_rpn_data` features/proposals and released
`scannet_nerf_data` RGB views/camera transforms. Proposals are converted from
RPN grid coordinates back to ScanNet world coordinates with each scene's
`bbox_min`, `bbox_max`, and `resolution`, then projected with the same
convention as `data/scannet/visualize_bbox.py`.

```text
NeRF-MAE†: output/nerf_rpn/results/baseline_e300_scannet_fcos1000_seed1_eval/proposals
ours:      output/nerf_rpn/results/cosine_ramp_e300_scannet_fcos1000_seed1_eval/proposals
```

No local ScanNet scratch proposal dump was found at the time this draft was
created; add it to `SCANNET_METHODS` in the script when available. The current
ScanNet RGB panel compares NeRF-MAE† and structure-first on the same view.

## Reproduction

Download the selected Front3D render scene:

```bash
python figures_src/download_front3d_render_scene.py \
  --scene 3dfront_0143_00 \
  --output-dir figures_src/qualitative_detection_assets/front3d_render_data
```

Download the selected ScanNet render scene:

```bash
python figures_src/download_scannet_render_scene.py \
  --scene scene0151_00 \
  --output-dir figures_src/qualitative_detection_assets/scannet_render_data
```

Build the current draft:

```bash
python figures_src/build_qualitative_detection_results.py \
  --front3d-scene 3dfront_0143_00 \
  --front3d-frame 0172 \
  --front3d-extra-frame 0015 \
  --scannet-scene scene0151_00 \
  --top-k 3 \
  --score-threshold 0.55
```

Use a lower threshold or larger `--top-k` only for debugging. For paper figures,
the current setting is intentionally sparse so predicted boxes remain readable.
