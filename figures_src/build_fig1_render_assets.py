#!/usr/bin/env python3
"""Build final Fig. 1 rendered RGB and bbox assets from downloaded Front3D views."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from render_nerfrpn_view_with_boxes import overlay_boxes


SCENE = "3dfront_0131_00"
FRAME = "0180"
BOX_INDICES = [1, 2, 3, 4, 5, 6]
ROOT = Path("figures_src/fig1_render_view_assets/render_cache/nerf_rpn_front3d_nerf_data/front3d_nerf_data")
OUT = Path("figures_src/fig1_render_view_assets/final")


def make_pair(clean_path: Path, boxed_path: Path, out_path: Path) -> None:
    clean = Image.open(clean_path).convert("RGB")
    boxed = Image.open(boxed_path).convert("RGB")
    target_h = 420
    clean = ImageOps.contain(clean, (560, target_h), Image.BILINEAR)
    boxed = ImageOps.contain(boxed, (560, target_h), Image.BILINEAR)
    margin = 18
    sheet = Image.new("RGB", (clean.width + boxed.width + margin * 3, target_h + margin * 2), "white")
    sheet.paste(clean, (margin, margin))
    sheet.paste(boxed, (margin * 2 + clean.width, margin))
    draw = ImageDraw.Draw(sheet)
    for x, img in [(margin, clean), (margin * 2 + clean.width, boxed)]:
        draw.rectangle([x, margin, x + img.width - 1, margin + img.height - 1], outline=(210, 210, 210), width=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scene_root = ROOT / SCENE
    image_path = scene_root / "train" / "images" / f"{FRAME}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(
            f"Missing {image_path}. Run figures_src/download_front3d_render_scene.py --scene {SCENE} first."
        )

    clean_out = OUT / f"fig1_{SCENE}_{FRAME}_render_rgb.png"
    bbox_out = OUT / f"fig1_{SCENE}_{FRAME}_render_rgb_bbox_furniture.png"
    pair_out = OUT / f"fig1_{SCENE}_{FRAME}_render_rgb_and_bbox_furniture_pair.png"
    Image.open(image_path).convert("RGB").save(clean_out)
    overlay_boxes(
        scene_root=scene_root,
        frame_stem=FRAME,
        out_path=bbox_out,
        convention="nerf",
        crop=None,
        max_boxes=None,
        box_indices=BOX_INDICES,
        title=None,
    )
    make_pair(clean_out, bbox_out, pair_out)
    print(f"wrote {clean_out}")
    print(f"wrote {bbox_out}")
    print(f"wrote {pair_out}")


if __name__ == "__main__":
    main()
