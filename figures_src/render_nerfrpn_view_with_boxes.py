#!/usr/bin/env python3
"""Overlay NeRF-RPN/NeRF-MAE bounding boxes on released rendered RGB views.

The released Front3D NeRF data contains rendered RGB frames, camera transforms,
and ground-truth boxes in instant-ngp coordinates. This script projects those
3D boxes onto selected RGB frames so Fig. 1 can follow the NeRF-MAE/NeRF-RPN
visualization policy instead of showing raw rgbsigma grid colors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


BOX_EDGES = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def box_corners(box: dict) -> np.ndarray:
    ext = np.asarray(box["extents"], dtype=np.float64) / 2.0
    orient = np.asarray(box["orientation"], dtype=np.float64)
    pos = np.asarray(box["position"], dtype=np.float64)
    local = np.array(
        [
            [-ext[0], -ext[1], -ext[2]],
            [ext[0], -ext[1], -ext[2]],
            [-ext[0], ext[1], -ext[2]],
            [ext[0], ext[1], -ext[2]],
            [-ext[0], -ext[1], ext[2]],
            [ext[0], -ext[1], ext[2]],
            [-ext[0], ext[1], ext[2]],
            [ext[0], ext[1], ext[2]],
        ],
        dtype=np.float64,
    )
    return pos[None, :] + local @ orient.T


def project_points(points: np.ndarray, c2w: np.ndarray, intr: dict, convention: str) -> tuple[np.ndarray, np.ndarray]:
    points_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    cam = (np.linalg.inv(c2w) @ points_h.T).T[:, :3]
    if convention == "opencv":
        depth = cam[:, 2]
        u = intr["fx"] * cam[:, 0] / np.maximum(depth, 1e-8) + intr["cx"]
        v = intr["fy"] * cam[:, 1] / np.maximum(depth, 1e-8) + intr["cy"]
    elif convention == "nerf":
        depth = -cam[:, 2]
        u = intr["fx"] * cam[:, 0] / np.maximum(depth, 1e-8) + intr["cx"]
        v = intr["cy"] - intr["fy"] * cam[:, 1] / np.maximum(depth, 1e-8)
    elif convention == "ngp":
        depth = cam[:, 2]
        u = intr["fx"] * cam[:, 0] / np.maximum(depth, 1e-8) + intr["cx"]
        v = intr["cy"] - intr["fy"] * cam[:, 1] / np.maximum(depth, 1e-8)
    else:
        raise ValueError(f"Unknown convention: {convention}")
    return np.stack([u, v], axis=1), depth


def draw_box(
    draw: ImageDraw.ImageDraw,
    pts2d: np.ndarray,
    depth: np.ndarray,
    image_size: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
) -> int:
    w, h = image_size
    drawn = 0
    for a, b in BOX_EDGES:
        if depth[a] <= 0 or depth[b] <= 0:
            continue
        p0 = pts2d[a]
        p1 = pts2d[b]
        # Keep lines that at least touch a loose canvas margin.
        margin = 80
        if (
            max(p0[0], p1[0]) < -margin
            or min(p0[0], p1[0]) > w + margin
            or max(p0[1], p1[1]) < -margin
            or min(p0[1], p1[1]) > h + margin
        ):
            continue
        line = [tuple(p0), tuple(p1)]
        draw.line(line, fill=(255, 255, 255), width=width + 3)
        draw.line(line, fill=color, width=width)
        drawn += 1
    return drawn


def overlay_boxes(
    scene_root: Path,
    frame_stem: str,
    out_path: Path,
    convention: str,
    crop: str | None,
    max_boxes: int | None,
    box_indices: list[int] | None,
    title: str | None,
) -> tuple[Path, int]:
    transforms_path = scene_root / "train" / "transforms.json"
    data = json.loads(transforms_path.read_text())
    frames = {Path(frame["file_path"]).stem: frame for frame in data["frames"]}
    if frame_stem not in frames:
        raise KeyError(f"{frame_stem} not found in {transforms_path}")
    image_path = scene_root / "train" / "images" / f"{frame_stem}.jpg"
    image = Image.open(image_path).convert("RGB")
    intr = {
        "fx": float(data.get("fl_x", data["w"] / (2.0 * np.tan(float(data["camera_angle_x"]) / 2.0)))),
        "fy": float(data.get("fl_y", data["h"] / (2.0 * np.tan(float(data["camera_angle_y"]) / 2.0)))),
        "cx": float(data.get("cx", data["w"] / 2.0)),
        "cy": float(data.get("cy", data["h"] / 2.0)),
    }
    c2w = np.asarray(frames[frame_stem]["transform_matrix"], dtype=np.float64)
    boxes = data.get("bounding_boxes", [])
    if box_indices is not None:
        boxes = [boxes[i] for i in box_indices if 0 <= i < len(boxes)]
    if max_boxes is not None:
        boxes = boxes[:max_boxes]

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    colors = [
        (255, 138, 26),
        (27, 150, 255),
        (46, 204, 113),
        (191, 85, 236),
        (230, 74, 25),
        (0, 172, 193),
    ]
    drawn_edges = 0
    for idx, box in enumerate(boxes):
        pts2d, depth = project_points(box_corners(box), c2w, intr, convention=convention)
        drawn_edges += draw_box(draw, pts2d, depth, canvas.size, colors[idx % len(colors)], width=3)

    if title:
        label = Image.new("RGB", (canvas.width, 34), "white")
        label_draw = ImageDraw.Draw(label)
        label_draw.text((8, 8), title, fill=(20, 20, 20))
        combined = Image.new("RGB", (canvas.width, canvas.height + label.height), "white")
        combined.paste(label, (0, 0))
        combined.paste(canvas, (0, label.height))
        canvas = combined

    if crop:
        x0, y0, x1, y1 = [int(v) for v in crop.split(",")]
        canvas = canvas.crop((x0, y0, x1, y1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path, drawn_edges


def make_pair(clean_path: Path, boxed_path: Path, out_path: Path) -> None:
    clean = Image.open(clean_path).convert("RGB")
    boxed = Image.open(boxed_path).convert("RGB")
    target_h = 360
    clean = ImageOps.contain(clean, (480, target_h), Image.BILINEAR)
    boxed = ImageOps.contain(boxed, (480, target_h), Image.BILINEAR)
    margin = 16
    title_h = 36
    w = clean.width + boxed.width + margin * 3
    h = max(clean.height, boxed.height) + title_h + margin * 2
    sheet = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 10), "rendered RGB view", fill=(20, 20, 20))
    draw.text((margin * 2 + clean.width, 10), "same view with 3D boxes", fill=(20, 20, 20))
    sheet.paste(clean, (margin, title_h + margin))
    sheet.paste(boxed, (margin * 2 + clean.width, title_h + margin))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--output-dir", default="figures_src/fig1_render_view_assets/candidates/bbox_overlays")
    parser.add_argument("--convention", choices=["nerf", "ngp", "opencv"], default="nerf")
    parser.add_argument("--crop", help="Optional x0,y0,x1,y1 crop after drawing.")
    parser.add_argument("--max-boxes", type=int)
    parser.add_argument(
        "--box-indices",
        help="Optional comma-separated original box indices to draw, e.g. 1,2,4. Useful for paper figures.",
    )
    args = parser.parse_args()

    scene_root = Path(args.scene_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = scene_root.name
    image_path = scene_root / "train" / "images" / f"{args.frame}.jpg"
    clean_out = out_dir / f"{scene}_{args.frame}_rgb_render.png"
    Image.open(image_path).convert("RGB").save(clean_out)
    boxed_out, drawn = overlay_boxes(
        scene_root=scene_root,
        frame_stem=args.frame,
        out_path=out_dir / f"{scene}_{args.frame}_rgb_render_boxes_{args.convention}.png",
        convention=args.convention,
        crop=args.crop,
        max_boxes=args.max_boxes,
        box_indices=([int(v) for v in args.box_indices.split(",")] if args.box_indices else None),
        title=None,
    )
    make_pair(clean_out, boxed_out, out_dir / f"{scene}_{args.frame}_rgb_render_pair_{args.convention}.png")
    print(f"wrote {clean_out}")
    print(f"wrote {boxed_out} (drawn_edges={drawn})")


if __name__ == "__main__":
    main()
