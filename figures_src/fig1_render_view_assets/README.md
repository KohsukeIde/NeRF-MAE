# Fig. 1 Render-View Assets

This directory supersedes the earlier `rgbsigma` grid scatter/volume visualizations for the main Fig. 1 visual.

## Policy

Fig. 1 should follow the NeRF-MAE / NeRF-RPN visual policy:

- use real rendered RGB views from the released Front3D NeRF data,
- optionally overlay 3D boxes on the same rendered view,
- keep raw radiance-density grid views as auxiliary/debug material, not as the main teaser asset.

The earlier grid-composited RGB assets looked noisy because they visualized the extracted `rgbsigma` grid directly. They are useful for debugging structure/alpha, but they are not comparable to the rendered scene panels in NeRF-MAE Fig. 5/teaser.

## Source

Rendered images and camera transforms are from the NeRF-RPN Hugging Face release:

```text
https://huggingface.co/datasets/lyclyc52/NeRF_RPN
front3d_nerf_data.zip
```

The local detection data under `dataset/finetune/front3d_rpn_data` only contains `features`, `obb`, and `aabb`. It does not contain rendered images, `transforms.json`, or instant-ngp checkpoints.

## Selected Fig. 1 Scene

Current selected scene/view:

```text
scene: 3dfront_0131_00
frame: train/images/0180.jpg
box projection: train/transforms.json bounding_boxes
drawn box indices: 1,2,3,4,5,6
```

Final assets:

```text
final/fig1_3dfront_0131_00_0180_render_rgb.png
final/fig1_3dfront_0131_00_0180_grid_rgba_same_camera.png
final/fig1_3dfront_0131_00_0180_grid_alpha_same_camera.png  # blue, depth-shaded
final/fig1_3dfront_0131_00_0180_render_rgb_bbox_furniture.png
final/fig1_3dfront_0131_00_0180_render_rgb_and_bbox_furniture_pair.png
final/fig1_3dfront_0131_00_0180_render_rgb_grid_rgba_alpha_bbox_quad.png
final/fig1_3dfront_0131_00_0180_render_rgb_crop_sofa_wall.png
final/fig1_3dfront_0131_00_0180_grid_rgba_crop_sofa_wall.png
final/fig1_3dfront_0131_00_0180_grid_alpha_crop_sofa_wall.png
final/fig1_3dfront_0131_00_0180_bbox_crop_sofa_wall.png
final/fig1_3dfront_0131_00_0180_render_rgb_grid_rgba_alpha_bbox_quad_crop_sofa_wall.png
```

Backup assets:

```text
final/fig1_3dfront_0045_00_0252_render_rgb.png
final/fig1_3dfront_0045_00_0252_render_rgb_bbox.png
```

## Reproduction

Install remote zip support once:

```bash
python -m pip install --user remotezip
```

Download rendered RGB views and camera transforms for candidate scenes:

```bash
python figures_src/download_front3d_render_scene.py \
  --scene 3dfront_0131_00 \
  --scene 3dfront_0045_00 \
  --scene 3dfront_0033_01 \
  --scene 3dfront_0135_00
```

Build rendered-view contact sheets:

```bash
python figures_src/select_fig1_render_views.py
```

Build same-camera RGB / RGBA-grid / blue-alpha candidate sheets for several
views of the selected scene:

```bash
python figures_src/build_fig1_camera_view_candidates.py
```

Generate the selected rendered view with projected boxes:

```bash
python figures_src/build_fig1_render_assets.py
```

Generate alpha/RGBA views from the extracted `rgbsigma` grid using the same
camera as the selected rendered RGB view:

```bash
python figures_src/render_camera_aligned_grid_views.py \
  --scene 3dfront_0131_00 \
  --frame 0180 \
  --samples 256 \
  --opacity-scale 1.15 \
  --alpha-threshold 0.0 \
  --crop 370,25,640,385
```

## Notes

- The bbox overlay here uses the ground-truth `bounding_boxes` already stored in `transforms.json`, projected onto the released RGB render view.
- `grid_alpha_same_camera` and `grid_rgba_same_camera` are not separate
  photorealistic NeRF renders. They are camera-aligned ray-marched views of the
  extracted radiance-density grid (`dataset/finetune/front3d_rpn_data/features`),
  using the same camera frame as the released RGB image. They are useful for
  explaining alpha/structure vs RGBA appearance, while the clean rendered RGB
  image should remain the main visual.
- The alpha panel is intentionally blue and depth-shaded: opacity controls how
  strongly structure appears, while expected ray depth changes the blue tone.
  This avoids reading voxel/grid texture as the main signal and makes the
  structure-first concept clearer.
- The paper-facing Fig. 1 asset should use the `crop_sofa_wall` outputs. The
  full camera view is kept for auditability, but the extracted grid contains
  low-density artifacts on the left wall of this frame. Cropping to the sofa and
  back-wall region follows the NeRF-MAE teaser/Fig. 5 style: show a clean
  interpretable part of the actual scene rather than the whole room.
- For predicted proposal visualization exactly like NeRF-RPN, use `nerf_rpn/scripts/proposals2ngp.py` to inject proposal boxes into `transforms.json`, then render through the instant-ngp fork referenced in the NeRF-MAE README. That is heavier and not needed for the Fig. 1 conceptual render.
- Candidate sheets are under `candidates/`; raw grid render assets remain under
  `debug/grid_volume_views/` for debugging and structure/alpha illustrations
  only.
