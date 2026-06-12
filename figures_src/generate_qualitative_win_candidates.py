#!/usr/bin/env python3
"""Generate readable qualitative win candidates from ranked detection scenes."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import build_qualitative_detection_results as q


def visible_line_score_front3d(scene_root: Path, feature_dir: Path, scene: str, frame: str, top_k: int) -> float:
    transforms = json.loads((scene_root / "train" / "transforms.json").read_text())
    frames = {Path(item["file_path"]).stem: item for item in transforms["frames"]}
    image_path = scene_root / "train" / "images" / f"{frame}.jpg"
    if frame not in frames or not image_path.exists():
        return -1.0
    with Image.open(image_path) as image:
        image_size = image.size
    intr = {
        "fx": float(transforms.get("fl_x", transforms["w"] / (2.0 * np.tan(float(transforms["camera_angle_x"]) / 2.0)))),
        "fy": float(transforms.get("fl_y", transforms["h"] / (2.0 * np.tan(float(transforms["camera_angle_y"]) / 2.0)))),
        "cx": float(transforms.get("cx", transforms["w"] / 2.0)),
        "cy": float(transforms.get("cy", transforms["h"] / 2.0)),
    }
    c2w = np.asarray(frames[frame]["transform_matrix"], dtype=np.float64)
    feature = q.front3d_feature_dict(feature_dir / f"{scene}.npz")
    ours_boxes, _ = q.load_top_proposals(q.FRONT3D_METHODS[-1].proposal_dir, scene, top_k, 0.0)
    baseline_boxes, _ = q.load_top_proposals(q.FRONT3D_METHODS[1].proposal_dir, scene, top_k, 0.0)
    ours_ngp = q.obb_to_ngp_boxes(ours_boxes, feature, dataset="front3d")
    baseline_ngp = q.obb_to_ngp_boxes(baseline_boxes, feature, dataset="front3d")

    def score_boxes(boxes: list[dict]) -> float:
        score = 0.0
        for box in boxes:
            lines = q.project_box_edges(box, c2w, intr, "nerf", image_size)
            if not lines:
                continue
            xs = [p[0] for line in lines for p in line]
            ys = [p[1] for line in lines for p in line]
            span = min(max(xs) - min(xs), image_size[0]) * min(max(ys) - min(ys), image_size[1])
            center_penalty = abs((min(xs) + max(xs)) / 2.0 - image_size[0] / 2.0) / image_size[0]
            score += len(lines) * 15.0 + min(span / 10000.0, 70.0) - center_penalty * 8.0
        return score

    # Prefer frames where ours is readable and baseline is also visible enough
    # to make the comparison fair, but do not require baseline to be correct.
    return score_boxes(ours_ngp) + 0.35 * score_boxes(baseline_ngp)


def visible_line_score_scannet(scene_root: Path, feature_dir: Path, scene: str, frame_stem: str, top_k: int) -> float:
    transforms = q.load_scannet_transforms(scene_root, "train")
    frames = {Path(item["file_path"]).stem: item for item in transforms["frames"]}
    if frame_stem not in frames:
        return -1.0
    frame = frames[frame_stem]
    image_path = q.scannet_frame_image_path(scene_root, frame)
    if not image_path.exists():
        return -1.0
    with Image.open(image_path) as image:
        image_size = image.size
    feature_path = feature_dir / f"{scene}.npz"
    ours_boxes, _ = q.load_top_proposals(q.SCANNET_METHODS[-1].proposal_dir, scene, top_k, 0.0)
    baseline_boxes, _ = q.load_top_proposals(q.SCANNET_METHODS[0].proposal_dir, scene, top_k, 0.0)
    ours_world = q.scannet_grid_boxes_to_world(ours_boxes, feature_path)
    baseline_world = q.scannet_grid_boxes_to_world(baseline_boxes, feature_path)

    def score_boxes(boxes: np.ndarray) -> float:
        score = 0.0
        for box in boxes:
            lines = q.scannet_project_box_edges(box, frame, image_size)
            if not lines:
                continue
            xs = [p[0] for line in lines for p in line]
            ys = [p[1] for line in lines for p in line]
            span = min(max(xs) - min(xs), image_size[0]) * min(max(ys) - min(ys), image_size[1])
            center_penalty = abs((min(xs) + max(xs)) / 2.0 - image_size[0] / 2.0) / image_size[0]
            score += len(lines) * 15.0 + min(span / 10000.0, 70.0) - center_penalty * 8.0
        return score

    return score_boxes(ours_world) + 0.35 * score_boxes(baseline_world)


def best_front3d_frames(scene: str, render_root: Path, feature_dir: Path, count: int, top_k: int) -> list[str]:
    scene_root = render_root / scene
    image_dir = scene_root / "train" / "images"
    frames = [p.stem for p in sorted(image_dir.glob("*.jpg"))]
    scored = [(visible_line_score_front3d(scene_root, feature_dir, scene, frame, top_k), frame) for frame in frames]
    scored = [item for item in scored if item[0] > 0]
    scored.sort(reverse=True)
    return [frame for _score, frame in scored[:count]]


def best_scannet_frames(scene: str, render_root: Path, feature_dir: Path, count: int, top_k: int) -> list[str]:
    scene_root = render_root / scene
    transforms = q.load_scannet_transforms(scene_root, "train")
    frames = [Path(item["file_path"]).stem for item in transforms["frames"] if q.scannet_frame_image_path(scene_root, item).exists()]
    scored = [(visible_line_score_scannet(scene_root, feature_dir, scene, frame, top_k), frame) for frame in frames]
    scored = [item for item in scored if item[0] > 0]
    scored.sort(reverse=True)
    return [frame for _score, frame in scored[:count]]


def make_contact_sheet(paths: list[Path], out_path: Path, max_width: int = 1200) -> None:
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, (max_width, 280), Image.BILINEAR)
        label = Image.new("RGB", (image.width, 24), "white")
        draw = ImageDraw.Draw(label)
        draw.text((6, 4), path.name, fill=(20, 20, 20))
        cell = Image.new("RGB", (image.width, image.height + 24), "white")
        cell.paste(label, (0, 0))
        cell.paste(image, (0, 24))
        thumbs.append(cell)
    if not thumbs:
        return
    margin = 12
    width = max(t.width for t in thumbs) + margin * 2
    height = sum(t.height for t in thumbs) + margin * (len(thumbs) + 1)
    sheet = Image.new("RGB", (width, height), "white")
    y = margin
    for thumb in thumbs:
        sheet.paste(thumb, (margin, y))
        y += thumb.height + margin
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def ranked_scenes(csv_path: Path, limit: int) -> list[str]:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (float(r["delta_top3_mean_iou"]), float(r["delta_top5_mean_iou"])), reverse=True)
    return [row["scene"] for row in rows[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("figures_src/qualitative_detection_assets/readable_win_candidates"))
    parser.add_argument("--front3d-count", type=int, default=6)
    parser.add_argument("--scannet-count", type=int, default=4)
    parser.add_argument("--frames-per-scene", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    out_dir = args.output_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    front_render = Path("figures_src/qualitative_detection_assets/front3d_render_data/front3d_nerf_data")
    scan_render = Path("figures_src/qualitative_detection_assets/scannet_render_data/scannet_nerf_data")
    front_feature = Path("dataset/finetune/front3d_rpn_data/features")
    scan_feature = Path("dataset/finetune/scannet_rpn_data/features")

    made: list[Path] = []
    index = 1
    for scene in ranked_scenes(Path("figures_src/qualitative_detection_assets/front3d_ours_wins_ranked.csv"), args.front3d_count):
        if not (front_render / scene).exists():
            continue
        for frame in best_front3d_frames(scene, front_render, front_feature, args.frames_per_scene, args.top_k):
            ns = SimpleNamespace(
                front3d_scene=scene,
                front3d_frame=frame,
                front3d_render_root=str(front_render),
                front3d_feature_dir=str(front_feature),
                front3d_convention="nerf",
                top_k=args.top_k,
                score_threshold=0.35,
            )
            path = q.make_front3d_panel(ns, out_dir)
            final = out_dir / f"candidate_{index:02d}_front3d_{scene}_{frame}.png"
            path.rename(final)
            made.append(final)
            index += 1

    for scene in ranked_scenes(Path("figures_src/qualitative_detection_assets/scannet_ours_wins_ranked.csv"), args.scannet_count):
        if not (scan_render / scene).exists():
            continue
        for frame in best_scannet_frames(scene, scan_render, scan_feature, args.frames_per_scene, args.top_k):
            ns = SimpleNamespace(
                scannet_scene=scene,
                scannet_render_root=str(scan_render),
                scannet_split="train",
                scannet_frame=frame,
                scannet_feature_dir=str(scan_feature),
                top_k=args.top_k,
                score_threshold=0.35,
            )
            path = q.make_scannet_rgb_panel(ns, out_dir)
            final = out_dir / f"candidate_{index:02d}_scannet_{scene}_{frame}.png"
            path.rename(final)
            made.append(final)
            index += 1

    make_contact_sheet(made[:20], out_dir / "contact_sheet.png")
    print(f"wrote {len(made)} candidates to {out_dir}")
    for path in made:
        print(path)


if __name__ == "__main__":
    main()
