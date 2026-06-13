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

Inspect these first:

```text
paper_candidates/
final/
```

`paper_candidates/` contains the smaller set intended for human selection. The
`final/` directory keeps only the current exported components needed to rebuild
the candidate sheet. Downloaded render data is under `render_cache/`.

Current selected scene/view:

```text
scene: 3dfront_0131_00
frame: train/images/0180.jpg
box projection: train/transforms.json bounding_boxes
drawn box indices: 1,2,3,4,5,6
```

Final assets:

```text
paper_candidates/fig1_recommended_render_completed_alpha_bbox_quad.png
paper_candidates/fig1_render_completed_grid_rgba_candidate.png
paper_candidates/fig1_alpha_structure_gray_candidate.png
paper_candidates/fig1_alpha_palette_comparison.png
paper_candidates/fig1_render_rgb_candidate.png
paper_candidates/fig1_render_rgb_bbox_candidate.png
```

## Reproduction

Install remote zip support once:

```bash
python -m pip install --user remotezip
```

Download rendered RGB views and camera transforms for candidate scenes:

```bash
python figures_src/download_front3d_render_scene.py \
  --scene 3dfront_0131_00
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
  --opacity-scale 0.75 \
  --alpha-threshold 0.0 \
  --no-connect-voxels \
  --screen-connect-kernel 31 \
  --screen-connect-iterations 1 \
  --screen-fill-opacity-threshold 0.18 \
  --alpha-screen-connect-kernel 71 \
  --alpha-screen-connect-iterations 2 \
  --alpha-screen-fill-opacity-threshold 0.35 \
  --alpha-style depth-normal \
  --alpha-palette gray \
  --alpha-hole-fill-kernel 63 \
  --alpha-hole-fill-near-threshold 0.20 \
  --alpha-hole-fill-opacity-threshold 0.045 \
  --grid-background render
```

## Notes

- The bbox overlay here uses the ground-truth `bounding_boxes` already stored in `transforms.json`, projected onto the released RGB render view.
- `grid_alpha_same_camera` and `grid_rgba_same_camera` are not separate
  photorealistic NeRF renders. They are camera-aligned ray-marched views of the
  extracted radiance-density grid (`dataset/finetune/front3d_rpn_data/features`),
  using the same camera frame as the released RGB image. They are useful for
  explaining alpha/structure vs RGBA appearance, while the clean rendered RGB
  image should remain the main visual.
- The alpha panel is intentionally gray and depth/normal-shaded in the current
  paper candidate. This reads more like geometry/mesh than the earlier blue
  signal-map style. Use `--alpha-palette blue` only for diagnostic figures that
  need to emphasize alpha as a structure signal.
- The current paper-facing alpha panel uses `--alpha-style depth-normal`,
  which shades the same-camera ray-marched alpha/depth as a connected surface.
  This replaced the marching-cubes-only mesh candidate because the latter
  dropped low-alpha side/back walls and made them look like white missing
  geometry.
- The grid RGBA panel preserves the camera-aligned grid color where the grid is
  opaque, but composites transparent/low-opacity regions over the matching
  rendered RGB view instead of a blank white background. This prevents missing
  grid opacity from reading as white geometry.
- For diagnostics, use `--grid-background flat`. That shows the extracted grid
  alone. In the selected 0180 view, the far-left table/door region is mostly
  low-opacity in the extracted grid; if `--grid-background render` is used, that
  region is primarily the released RGB render showing through rather than
  recovered grid geometry.
- The paper-facing `fig1_render_completed_grid_rgba_candidate.png` therefore
  should be described as render-completed/image-composited grid visualization,
  not as a pure mesh or pure grid render.
- We do not dilate all visible voxels by default because that degrades
  foreground furniture quality.
- The alpha panel uses a separate, stronger screen-space opacity fill and the
  gray palette. This is a figure-facing structure visualization, not a separate
  photorealistic render.
- For predicted proposal visualization exactly like NeRF-RPN, use `nerf_rpn/scripts/proposals2ngp.py` to inject proposal boxes into `transforms.json`, then render through the instant-ngp fork referenced in the NeRF-MAE README. That is heavier and not needed for the Fig. 1 conceptual render.
- Intermediate contact sheets and rejected debug renders were removed from this
  directory. Regenerate them with the commands above if needed.
