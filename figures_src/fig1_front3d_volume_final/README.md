# Fig. 1 Front3D Volume Render Asset

Selected scene:
- `3dfront_0135_00`

Source:
- `dataset/finetune/front3d_rpn_data/features/3dfront_0135_00.npz`
- This is a released Front3D NeRF-RPN/NeRF-MAE `rgbsigma` grid, not a schematic.

Why this scene was selected:
- The alpha-only render exposes room-scale structure and furniture layout.
- The RGB render shows a clear appearance signal on the same geometry.
- The scene has enough objects to make the structure/appearance contrast
  visually legible without being too cluttered for a first-page conceptual
  figure.

Rendered assets:
- `3dfront_0135_00_alpha.png`: alpha/structure view.
- `3dfront_0135_00_rgb.png`: RGB/appearance view.
- `fig1_front3d_0135_alpha_rgb_pair_clean.png`: clean side-by-side preview.
- `fig1_front3d_0135_alpha_rgb_pair.png`: labeled side-by-side preview.

Generation command:

```bash
python figures_src/render_front3d_fig1_volume.py \
  --scene 3dfront_0135_00 \
  --out-dir figures_src/fig1_front3d_volume_final \
  --dpi 320 --max-points 260000 --marker-size 0.34 \
  --azimuth -42 --elevation 57
```

Rendering details:
- Density is converted to alpha using the same Front3D conversion used by
  `nerf_rpn/datasets.py`:
  `alpha = clip(1 - exp(-exp(density) / 100), 0, 1)`.
- Points are sampled from occupied voxels with `alpha > 0.06`, weighted toward
  stronger alpha values.
- RGB values are clipped to `[0, 1]`.
- The final pair is intended as a data-derived visual asset for the Fig. 1
  conceptual comparison; quantitative results should remain in Fig. 2 / Table 1.
