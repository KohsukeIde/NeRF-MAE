#!/usr/bin/env python3
"""Select readable rendered Front3D views for Fig. 1.

This works on the NeRF-RPN released `front3d_nerf_data` render images. It does
not use the radiance-density grid, because those grid visualizations are not
comparable to the NeRF-MAE paper teaser/Fig. 5 render panels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def image_score(path: Path) -> tuple[float, dict[str, float]]:
    img = Image.open(path).convert("RGB").resize((160, 120), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    nonwhite = float(np.mean(gray < 0.92))
    nonblack = float(np.mean(gray > 0.08))
    contrast = float(gray.std())
    sat = float((arr.max(axis=2) - arr.min(axis=2)).mean())
    # Simple crop-aware balance: reject views dominated by one empty side.
    h, w = gray.shape
    thirds = [gray[:, : w // 3], gray[:, w // 3 : 2 * w // 3], gray[:, 2 * w // 3 :]]
    occupancy_balance = float(min(np.mean(t < 0.92) for t in thirds))
    score = (
        2.0 * min(nonwhite, 0.85)
        + 1.2 * contrast
        + 0.8 * sat
        + 0.8 * occupancy_balance
        - 1.5 * max(0.0, 0.18 - nonwhite)
        - 0.8 * max(0.0, 0.35 - nonblack)
    )
    return score, {
        "nonwhite": nonwhite,
        "nonblack": nonblack,
        "contrast": contrast,
        "sat": sat,
        "balance": occupancy_balance,
    }


def make_contact_sheet(paths: list[Path], out_path: Path, title: str, thumb_w: int = 192) -> None:
    if not paths:
        raise ValueError("No paths for contact sheet")
    thumb_h = int(thumb_w * 3 / 4)
    label_h = 22
    cols = 6
    rows = int(np.ceil(len(paths) / cols))
    margin = 12
    title_h = 34
    sheet = Image.new(
        "RGB",
        (cols * thumb_w + (cols + 1) * margin, title_h + rows * (thumb_h + label_h) + (rows + 1) * margin),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 8), title, fill=(20, 20, 20))
    for idx, path in enumerate(paths):
        row, col = divmod(idx, cols)
        x = margin + col * (thumb_w + margin)
        y = title_h + margin + row * (thumb_h + label_h + margin)
        img = Image.open(path).convert("RGB")
        img = ImageOps.contain(img, (thumb_w, thumb_h), Image.BILINEAR)
        frame = Image.new("RGB", (thumb_w, thumb_h), (246, 246, 246))
        frame.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
        sheet.paste(frame, (x, y))
        draw.rectangle([x, y, x + thumb_w - 1, y + thumb_h - 1], outline=(210, 210, 210), width=1)
        draw.text((x + 4, y + thumb_h + 3), path.stem, fill=(35, 35, 35))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="figures_src/fig1_render_view_assets/render_cache/nerf_rpn_front3d_nerf_data/front3d_nerf_data")
    parser.add_argument("--output-dir", default="figures_src/fig1_render_view_assets/candidates/render_view_contact_sheets")
    parser.add_argument("--top-k", type=int, default=36)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = ["scene,rank,frame,score,nonwhite,nonblack,contrast,saturation,balance,path"]
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        image_dir = scene_dir / "train" / "images"
        paths = sorted(image_dir.glob("*.jpg"))
        if not paths:
            continue
        scored = []
        for path in paths:
            score, metrics = image_score(path)
            scored.append((score, path, metrics))
        scored.sort(reverse=True, key=lambda x: x[0])
        top_paths = [p for _, p, _ in scored[: args.top_k]]
        make_contact_sheet(
            top_paths,
            out_dir / f"{scene_dir.name}_top{args.top_k}_render_views.png",
            f"{scene_dir.name}: top rendered RGB view candidates",
        )
        uniform_paths = [paths[i] for i in np.linspace(0, len(paths) - 1, min(args.top_k, len(paths)), dtype=int)]
        make_contact_sheet(
            uniform_paths,
            out_dir / f"{scene_dir.name}_uniform_render_views.png",
            f"{scene_dir.name}: uniform rendered RGB view samples",
        )
        for rank, (score, path, metrics) in enumerate(scored[: args.top_k], start=1):
            summary_lines.append(
                ",".join(
                    [
                        scene_dir.name,
                        str(rank),
                        path.stem,
                        f"{score:.6f}",
                        f"{metrics['nonwhite']:.6f}",
                        f"{metrics['nonblack']:.6f}",
                        f"{metrics['contrast']:.6f}",
                        f"{metrics['sat']:.6f}",
                        f"{metrics['balance']:.6f}",
                        str(path),
                    ]
                )
            )
    (out_dir / "render_view_candidate_scores.csv").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
