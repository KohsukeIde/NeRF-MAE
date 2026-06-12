#!/usr/bin/env python3
"""Build candidate sheets for camera-aligned Fig. 1 render/alpha/RGBA views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from render_camera_aligned_grid_views import (
    composite_grid_over_render_background,
    connect_screen_gaps,
    load_rgba_volume,
    render_camera_aligned,
    save_alpha,
    save_rgb,
)


DEFAULT_FRAMES = ["0180", "0150", "0224", "0134", "0108", "0191"]


def make_sheet(rows: list[tuple[str, Path, Path, Path]], out_path: Path) -> None:
    thumb_w, thumb_h = 240, 180
    margin, label_h, title_h = 12, 22, 30
    cols = 3
    sheet = Image.new(
        "RGB",
        (cols * thumb_w + (cols + 1) * margin, title_h + len(rows) * (thumb_h + label_h + margin) + margin),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 8), "Rendered RGB / same-camera grid RGBA / blue alpha candidates", fill=(20, 20, 20))
    col_labels = ["rendered RGB", "grid RGBA", "alpha / structure"]
    for row_idx, (frame, rgb_path, rgba_path, alpha_path) in enumerate(rows):
        y = title_h + margin + row_idx * (thumb_h + label_h + margin)
        for col_idx, path in enumerate([rgb_path, rgba_path, alpha_path]):
            x = margin + col_idx * (thumb_w + margin)
            img = Image.open(path).convert("RGB")
            img = ImageOps.contain(img, (thumb_w, thumb_h), Image.BILINEAR)
            bg = Image.new("RGB", (thumb_w, thumb_h), (248, 248, 248))
            bg.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
            sheet.paste(bg, (x, y))
            draw.rectangle([x, y, x + thumb_w - 1, y + thumb_h - 1], outline=(210, 210, 210), width=1)
            draw.text((x + 3, y + thumb_h + 4), f"{frame} / {col_labels[col_idx]}", fill=(25, 25, 25))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="3dfront_0131_00")
    parser.add_argument("--frame", action="append", help="Frame id. Can be repeated.")
    parser.add_argument("--feature-dir", default="dataset/finetune/front3d_rpn_data/features")
    parser.add_argument(
        "--render-root",
        default="figures_src/fig1_render_view_assets/nerf_rpn_front3d_nerf_data/front3d_nerf_data",
    )
    parser.add_argument("--output-dir", default="figures_src/fig1_render_view_assets/candidates/camera_aligned_views")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--tile-rows", type=int, default=24)
    parser.add_argument("--opacity-scale", type=float, default=0.75)
    parser.add_argument("--alpha-threshold", type=float, default=0.0)
    parser.add_argument("--screen-connect-kernel", type=int, default=31)
    parser.add_argument("--screen-connect-iterations", type=int, default=1)
    parser.add_argument("--screen-connect-opacity-gain", type=float, default=0.9)
    parser.add_argument("--screen-fill-opacity-threshold", type=float, default=0.18)
    parser.add_argument("--alpha-screen-connect-kernel", type=int, default=71)
    parser.add_argument("--alpha-screen-connect-iterations", type=int, default=2)
    parser.add_argument("--alpha-screen-connect-opacity-gain", type=float, default=0.85)
    parser.add_argument("--alpha-screen-fill-opacity-threshold", type=float, default=0.35)
    args = parser.parse_args()

    frames = args.frame or DEFAULT_FRAMES
    scene_root = Path(args.render_root) / args.scene
    transforms_path = scene_root / "train" / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(
            f"Missing {transforms_path}. Run figures_src/download_front3d_render_scene.py --scene {args.scene} first."
        )
    transforms = json.loads(transforms_path.read_text())
    volume, meta = load_rgba_volume(Path(args.feature_dir) / f"{args.scene}.npz")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for frame in frames:
        render_path = scene_root / "train" / "images" / f"{frame}.jpg"
        rgb_out = out / f"{args.scene}_{frame}_render_rgb.png"
        rgba_out = out / f"{args.scene}_{frame}_grid_rgba_same_camera.png"
        alpha_out = out / f"{args.scene}_{frame}_grid_alpha_blue_depth_same_camera.png"
        Image.open(render_path).convert("RGB").save(rgb_out)
        grid_rgb_raw, opacity_raw, depth_raw = render_camera_aligned(
            volume=volume,
            meta=meta,
            transforms=transforms,
            frame=frame,
            width=args.width,
            height=args.height,
            samples=args.samples,
            tile_rows=args.tile_rows,
            opacity_scale=args.opacity_scale,
            alpha_threshold=args.alpha_threshold,
        )
        grid_rgb, opacity, depth = connect_screen_gaps(
            grid_rgb_raw,
            opacity_raw,
            depth_raw,
            kernel_size=args.screen_connect_kernel,
            iterations=args.screen_connect_iterations,
            opacity_gain=args.screen_connect_opacity_gain,
            fill_opacity_threshold=args.screen_fill_opacity_threshold,
        )
        _, opacity_alpha, depth_alpha = connect_screen_gaps(
            grid_rgb_raw,
            opacity_raw,
            depth_raw,
            kernel_size=args.alpha_screen_connect_kernel,
            iterations=args.alpha_screen_connect_iterations,
            opacity_gain=args.alpha_screen_connect_opacity_gain,
            fill_opacity_threshold=args.alpha_screen_fill_opacity_threshold,
        )
        grid_rgb = composite_grid_over_render_background(grid_rgb, opacity, rgb_out)
        save_rgb(rgba_out, grid_rgb)
        save_alpha(alpha_out, opacity_alpha, depth_alpha)
        rows.append((frame, rgb_out, rgba_out, alpha_out))
    make_sheet(rows, out / f"{args.scene}_camera_aligned_view_candidates.png")


if __name__ == "__main__":
    main()
