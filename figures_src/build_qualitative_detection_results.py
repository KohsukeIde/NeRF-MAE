#!/usr/bin/env python3
"""Build qualitative predicted-OBB figures for Front3D and ScanNet.

Both Front3D and ScanNet panels project predicted OBB proposals onto released
NeRF-RPN RGB render views when the corresponding render artifacts are present.
ScanNet BEV alpha projection remains available as a fallback/debug view.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerf_rpn.scripts.proposals2ngp import obb_to_ngp_boxes
from figures_src.render_nerfrpn_view_with_boxes import BOX_EDGES, box_corners, project_points


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    proposal_dir: Path
    color: tuple[int, int, int]


FRONT3D_METHODS = [
    MethodSpec(
        "scratch",
        "scratch",
        Path("output/nerf_rpn/results/front3d_scratch_lowlabel_pt100_seed1_fcos1000_eval/proposals"),
        (229, 57, 53),
    ),
    MethodSpec(
        "nerfmae",
        "NeRF-MAE\u2020",
        Path("output/nerf_rpn/results/budgetcurve_baseline_e1200_seed1_fcos1000_eval/proposals"),
        (247, 148, 29),
    ),
    MethodSpec(
        "ours",
        "structure-first",
        Path("output/nerf_rpn/results/budgetcurve_cosine_ramp_e600_seed1_fcos1000_eval/proposals"),
        (26, 145, 96),
    ),
]

GT_SPEC = MethodSpec(
    "gt",
    "GT",
    Path("__ground_truth__"),
    (62, 101, 214),
)

SCANNET_METHODS = [
    MethodSpec(
        "nerfmae",
        "NeRF-MAE\u2020",
        Path("output/nerf_rpn/results/baseline_e300_scannet_fcos1000_seed1_eval/proposals"),
        (247, 148, 29),
    ),
    MethodSpec(
        "ours",
        "structure-first",
        Path("output/nerf_rpn/results/cosine_ramp_e300_scannet_fcos1000_seed1_eval/proposals"),
        (26, 145, 96),
    ),
]


def density_to_alpha_front3d(density: np.ndarray) -> np.ndarray:
    density = np.clip(density, -20.0, 20.0)
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


def density_to_alpha_scannet(density: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.exp(-np.clip(density, a_min=0.0, a_max=None) / 100.0), 0.0, 1.0)


def load_top_proposals(proposal_dir: Path, scene: str, top_k: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    path = proposal_dir / f"{scene}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as z:
        boxes = z["proposals"].astype(np.float64)
        scores = z["scores"].astype(np.float64)
    order = np.argsort(scores)[::-1]
    keep = [idx for idx in order if scores[idx] >= threshold]
    if len(keep) < min(top_k, len(order)):
        keep = list(order[:top_k])
    keep = keep[:top_k]
    return boxes[keep], scores[keep]


def front3d_feature_dict(feature_path: Path) -> dict:
    with np.load(feature_path) as z:
        return {key: z[key] for key in ["resolution", "bbox_min", "bbox_max", "scale", "offset", "from_mitsuba"]}


def project_box_edges(
    box: dict,
    c2w: np.ndarray,
    intr: dict[str, float],
    convention: str,
    image_size: tuple[int, int],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    pts2d, depth = project_points(box_corners(box), c2w, intr, convention=convention)
    width, height = image_size
    lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
    margin = 80
    for a, b in BOX_EDGES:
        if depth[a] <= 0 or depth[b] <= 0:
            continue
        p0, p1 = pts2d[a], pts2d[b]
        if (
            max(p0[0], p1[0]) < -margin
            or min(p0[0], p1[0]) > width + margin
            or max(p0[1], p1[1]) < -margin
            or min(p0[1], p1[1]) > height + margin
        ):
            continue
        lines.append((tuple(p0), tuple(p1)))
    return lines


def draw_projected_boxes(
    image_path: Path,
    transforms: dict,
    frame: str,
    boxes: list[dict],
    color: tuple[int, int, int],
    title: str,
    convention: str = "nerf",
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    frames = {Path(item["file_path"]).stem: item for item in transforms["frames"]}
    c2w = np.asarray(frames[frame]["transform_matrix"], dtype=np.float64)
    intr = {
        "fx": float(transforms.get("fl_x", transforms["w"] / (2.0 * np.tan(float(transforms["camera_angle_x"]) / 2.0)))),
        "fy": float(transforms.get("fl_y", transforms["h"] / (2.0 * np.tan(float(transforms["camera_angle_y"]) / 2.0)))),
        "cx": float(transforms.get("cx", transforms["w"] / 2.0)),
        "cy": float(transforms.get("cy", transforms["h"] / 2.0)),
    }
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for box in boxes:
        for p0, p1 in project_box_edges(box, c2w, intr, convention, canvas.size):
            draw.line([p0, p1], fill=(255, 255, 255), width=5)
            draw.line([p0, p1], fill=color, width=3)
    return add_title(canvas, title)


def select_front3d_frame(
    scene_root: Path,
    proposal_boxes: list[dict],
    preferred_frame: str | None,
    convention: str,
) -> str:
    if preferred_frame:
        return preferred_frame
    transforms = json.loads((scene_root / "train" / "transforms.json").read_text())
    intr = {
        "fx": float(transforms.get("fl_x", transforms["w"] / (2.0 * np.tan(float(transforms["camera_angle_x"]) / 2.0)))),
        "fy": float(transforms.get("fl_y", transforms["h"] / (2.0 * np.tan(float(transforms["camera_angle_y"]) / 2.0)))),
        "cx": float(transforms.get("cx", transforms["w"] / 2.0)),
        "cy": float(transforms.get("cy", transforms["h"] / 2.0)),
    }
    best_score = -1
    best_frame = Path(transforms["frames"][0]["file_path"]).stem
    for frame_item in transforms["frames"]:
        frame = Path(frame_item["file_path"]).stem
        image_path = scene_root / "train" / "images" / f"{frame}.jpg"
        if not image_path.exists():
            continue
        c2w = np.asarray(frame_item["transform_matrix"], dtype=np.float64)
        visible_edges = 0
        for box in proposal_boxes[:8]:
            visible_edges += len(project_box_edges(box, c2w, intr, convention, (transforms["w"], transforms["h"])))
        if visible_edges > best_score:
            best_score = visible_edges
            best_frame = frame
    return best_frame


def scannet_feature_dict(feature_path: Path) -> dict:
    with np.load(feature_path) as z:
        return {key: z[key] for key in ["resolution", "bbox_min", "bbox_max"]}


def scannet_grid_boxes_to_world(boxes: np.ndarray, feature_path: Path) -> np.ndarray:
    feature = scannet_feature_dict(feature_path)
    resolution = feature["resolution"].astype(np.float64)
    bbox_min = feature["bbox_min"].astype(np.float64)
    bbox_max = feature["bbox_max"].astype(np.float64)
    scale = (bbox_max - bbox_min) / resolution
    world = boxes.astype(np.float64).copy()
    world[:, :3] = world[:, :3] * scale + bbox_min
    world[:, 3:6] = world[:, 3:6] * scale
    return world


def scannet_obb_corners_world(box: np.ndarray) -> np.ndarray:
    x, y, z, w, l, h, theta = box.astype(np.float64)
    half = np.array([w, l, h], dtype=np.float64) / 2.0
    local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [-half[0], half[1], half[2]],
            [half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return local @ rot.T + np.array([x, y, z], dtype=np.float64)


def scannet_world_to_proj(frame: dict, width: int, height: int) -> np.ndarray:
    # Match data/scannet/visualize_bbox.py so qualitative overlays follow the
    # released NeRF-RPN ScanNet visualization convention.
    cam2world = np.asarray(frame["transform_matrix"], dtype=np.float64).copy()
    cam2world[:, [1, 2]] *= -1
    fy = float(frame["fy"])
    focal = fy / height
    zscale = 1.0 / focal
    xyscale = height
    cam2proj = np.array(
        [
            [xyscale, 0.0, width * 0.5 * zscale, 0.0],
            [0.0, xyscale, height * 0.5 * zscale, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, zscale, 0.0],
        ],
        dtype=np.float64,
    )
    return cam2proj @ np.linalg.inv(cam2world)


def scannet_project_points(points: np.ndarray, world2proj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    projected = (world2proj @ points_h.T).T
    depth = projected[:, 3]
    pts2d = projected[:, :2] / np.maximum(depth[:, None], 1e-8)
    return pts2d, depth


def scannet_project_box_edges(
    box: np.ndarray,
    frame: dict,
    image_size: tuple[int, int],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    width, height = image_size
    world2proj = scannet_world_to_proj(frame, width, height)
    pts2d, depth = scannet_project_points(scannet_obb_corners_world(box), world2proj)
    lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
    margin = 80
    for a, b in BOX_EDGES:
        if depth[a] <= 0 or depth[b] <= 0:
            continue
        p0, p1 = pts2d[a], pts2d[b]
        if (
            max(p0[0], p1[0]) < -margin
            or min(p0[0], p1[0]) > width + margin
            or max(p0[1], p1[1]) < -margin
            or min(p0[1], p1[1]) > height + margin
        ):
            continue
        lines.append((tuple(p0), tuple(p1)))
    return lines


def scannet_frame_image_path(scene_root: Path, frame: dict) -> Path:
    return scene_root / frame["file_path"]


def load_scannet_transforms(scene_root: Path, split: str) -> dict:
    transforms_path = scene_root / f"transforms_{split}.json"
    if not transforms_path.exists():
        raise FileNotFoundError(transforms_path)
    return json.loads(transforms_path.read_text())


def select_scannet_frame(
    scene_root: Path,
    transforms: dict,
    boxes_world: np.ndarray,
    preferred_frame: str | None,
) -> tuple[str, dict]:
    frames_by_stem = {Path(item["file_path"]).stem: item for item in transforms["frames"]}
    if preferred_frame:
        if preferred_frame not in frames_by_stem:
            raise KeyError(f"{preferred_frame} not found in ScanNet transforms")
        return preferred_frame, frames_by_stem[preferred_frame]

    best_score = -1.0
    best_stem = Path(transforms["frames"][0]["file_path"]).stem
    best_frame = transforms["frames"][0]
    for frame in transforms["frames"]:
        image_path = scannet_frame_image_path(scene_root, frame)
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            image_size = image.size
        visible_edges = 0
        in_canvas_points = 0
        for box in boxes_world[:8]:
            lines = scannet_project_box_edges(box, frame, image_size)
            visible_edges += len(lines)
            for p0, p1 in lines:
                for p in [p0, p1]:
                    if 0 <= p[0] < image_size[0] and 0 <= p[1] < image_size[1]:
                        in_canvas_points += 1
        score = visible_edges * 10 + in_canvas_points
        if score > best_score:
            best_score = score
            best_stem = Path(frame["file_path"]).stem
            best_frame = frame
    return best_stem, best_frame


def add_title(image: Image.Image, title: str) -> Image.Image:
    title_h = 34
    out = Image.new("RGB", (image.width, image.height + title_h), "white")
    draw = ImageDraw.Draw(out)
    draw.text((10, 9), title, fill=(20, 20, 20))
    out.paste(image, (0, title_h))
    return out


def make_front3d_panel(args: argparse.Namespace, out_dir: Path) -> Path:
    scene_root = Path(args.front3d_render_root) / args.front3d_scene
    transforms_path = scene_root / "train" / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(
            f"Missing {transforms_path}. Run figures_src/download_front3d_render_scene.py "
            f"--scene {args.front3d_scene} --output-dir {Path(args.front3d_render_root).parents[0]}"
        )
    transforms = json.loads(transforms_path.read_text())
    feature_dict = front3d_feature_dict(Path(args.front3d_feature_dir) / f"{args.front3d_scene}.npz")

    converted_by_method: dict[str, list[dict]] = {}
    for method in FRONT3D_METHODS:
        boxes, scores = load_top_proposals(method.proposal_dir, args.front3d_scene, args.top_k, args.score_threshold)
        ngp_boxes = obb_to_ngp_boxes(boxes, feature_dict, dataset="front3d")
        converted_by_method[method.key] = ngp_boxes

    frame = select_front3d_frame(
        scene_root,
        converted_by_method["ours"],
        args.front3d_frame,
        convention=args.front3d_convention,
    )
    image_path = scene_root / "train" / "images" / f"{frame}.jpg"
    panels = []
    if getattr(args, "include_gt", True):
        panels.append(
            draw_projected_boxes(
                image_path,
                transforms,
                frame,
                transforms.get("bounding_boxes", []),
                GT_SPEC.color,
                f"Front3D {args.front3d_scene} / {GT_SPEC.label}",
                convention=args.front3d_convention,
            )
        )
    for method in FRONT3D_METHODS:
        panels.append(
            draw_projected_boxes(
                image_path,
                transforms,
                frame,
                converted_by_method[method.key],
                method.color,
                f"Front3D {args.front3d_scene} / {method.label}",
                convention=args.front3d_convention,
            )
        )
    out = out_dir / f"front3d_{args.front3d_scene}_{frame}_pred_obb_threeway.png"
    make_row(panels, out)
    return out


def obb_bev_polygon(box: np.ndarray) -> np.ndarray:
    x, y, _z, w, l, _h, theta = box
    c, s = np.cos(theta), np.sin(theta)
    dx = np.array([c, s]) * w / 2.0
    dy = np.array([-s, c]) * l / 2.0
    return np.stack(
        [
            np.array([x, y]) - dx - dy,
            np.array([x, y]) + dx - dy,
            np.array([x, y]) + dx + dy,
            np.array([x, y]) - dx + dy,
        ]
    )


def draw_bev_panel(
    feature_path: Path,
    proposal_path: Path,
    gt_path: Path,
    dataset: str,
    color: tuple[int, int, int],
    title: str,
    top_k: int,
    threshold: float,
    image_size: int = 520,
) -> Image.Image:
    with np.load(feature_path) as z:
        rgbsigma = z["rgbsigma"].astype(np.float32)
    alpha = density_to_alpha_scannet(rgbsigma[..., 3]) if dataset == "scannet" else density_to_alpha_front3d(rgbsigma[..., 3])
    bev = np.max(alpha, axis=2)
    lo, hi = np.percentile(bev, [2, 99.5])
    bev = np.clip((bev - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    bg = np.stack([0.96 - 0.36 * bev, 0.98 - 0.30 * bev, 1.0 - 0.18 * bev], axis=-1)
    image = Image.fromarray((bg * 255).astype(np.uint8)).resize((image_size, image_size), Image.BILINEAR)
    draw = ImageDraw.Draw(image)

    h, w = bev.shape[1], bev.shape[0]

    def to_px(points: np.ndarray) -> list[tuple[float, float]]:
        xs = points[:, 0] / max(w, 1) * image_size
        ys = (1.0 - points[:, 1] / max(h, 1)) * image_size
        return list(map(tuple, np.stack([xs, ys], axis=1)))

    gt = np.load(gt_path).astype(np.float64)
    for box in gt:
        pts = to_px(obb_bev_polygon(box))
        draw.line(pts + [pts[0]], fill=(70, 70, 70), width=1)

    boxes, scores = load_top_proposals(proposal_path.parent, proposal_path.stem, top_k, threshold)
    for box in boxes:
        pts = to_px(obb_bev_polygon(box))
        draw.line(pts + [pts[0]], fill=(255, 255, 255), width=5)
        draw.line(pts + [pts[0]], fill=color, width=3)

    return add_title(image, title)


def make_scannet_panel(args: argparse.Namespace, out_dir: Path) -> Path:
    scene_root = Path(args.scannet_render_root) / args.scannet_scene
    if not args.scannet_bev and scene_root.exists():
        return make_scannet_rgb_panel(args, out_dir)
    if not args.scannet_bev:
        print(f"ScanNet render scene missing at {scene_root}; falling back to BEV alpha projection.")
    return make_scannet_bev_panel(args, out_dir)


def draw_scannet_rgb_boxes(
    image_path: Path,
    frame: dict,
    boxes_world: np.ndarray,
    color: tuple[int, int, int],
    title: str,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for box in boxes_world:
        for p0, p1 in scannet_project_box_edges(box, frame, canvas.size):
            draw.line([p0, p1], fill=(255, 255, 255), width=5)
            draw.line([p0, p1], fill=color, width=3)
    return add_title(canvas, title)


def make_scannet_rgb_panel(args: argparse.Namespace, out_dir: Path) -> Path:
    scene_root = Path(args.scannet_render_root) / args.scannet_scene
    transforms = load_scannet_transforms(scene_root, args.scannet_split)
    feature_path = Path(args.scannet_feature_dir) / f"{args.scannet_scene}.npz"

    boxes_by_method: dict[str, np.ndarray] = {}
    for method in SCANNET_METHODS:
        boxes, _scores = load_top_proposals(method.proposal_dir, args.scannet_scene, args.top_k, args.score_threshold)
        boxes_by_method[method.key] = scannet_grid_boxes_to_world(boxes, feature_path)

    frame_stem, frame = select_scannet_frame(
        scene_root,
        transforms,
        boxes_by_method[SCANNET_METHODS[-1].key],
        args.scannet_frame,
    )
    image_path = scannet_frame_image_path(scene_root, frame)
    panels = []
    if getattr(args, "include_gt", True):
        gt_grid = np.load(Path(args.scannet_obb_dir) / f"{args.scannet_scene}.npy").astype(np.float64)
        gt_world = scannet_grid_boxes_to_world(gt_grid, feature_path)
        panels.append(
            draw_scannet_rgb_boxes(
                image_path=image_path,
                frame=frame,
                boxes_world=gt_world,
                color=GT_SPEC.color,
                title=f"ScanNet {args.scannet_scene} / {GT_SPEC.label}",
            )
        )
    for method in SCANNET_METHODS:
        panels.append(
            draw_scannet_rgb_boxes(
                image_path=image_path,
                frame=frame,
                boxes_world=boxes_by_method[method.key],
                color=method.color,
                title=f"ScanNet {args.scannet_scene} / {method.label}",
            )
        )
    out = out_dir / f"scannet_{args.scannet_scene}_{frame_stem}_pred_obb_rgb.png"
    make_row(panels, out)
    return out


def make_scannet_bev_panel(args: argparse.Namespace, out_dir: Path) -> Path:
    panels = []
    for method in SCANNET_METHODS:
        proposal_path = method.proposal_dir / f"{args.scannet_scene}.npz"
        panels.append(
            draw_bev_panel(
                feature_path=Path(args.scannet_feature_dir) / f"{args.scannet_scene}.npz",
                proposal_path=proposal_path,
                gt_path=Path(args.scannet_obb_dir) / f"{args.scannet_scene}.npy",
                dataset="scannet",
                color=method.color,
                title=f"ScanNet {args.scannet_scene} / {method.label}",
                top_k=args.top_k,
                threshold=args.score_threshold,
                image_size=520,
            )
        )
    out = out_dir / f"scannet_{args.scannet_scene}_bev_pred_obb.png"
    make_row(panels, out)
    return out


def make_row(images: list[Image.Image], out_path: Path) -> None:
    target_h = 390
    resized = [ImageOps.contain(img, (520, target_h), Image.BILINEAR) for img in images]
    margin = 16
    width = sum(img.width for img in resized) + margin * (len(resized) + 1)
    height = max(img.height for img in resized) + margin * 2
    sheet = Image.new("RGB", (width, height), "white")
    x = margin
    for img in resized:
        sheet.paste(img, (x, margin))
        x += img.width + margin
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def make_combined(front_paths: list[Path], scan_path: Path, out_path: Path) -> None:
    fronts = [Image.open(path).convert("RGB") for path in front_paths]
    scan = Image.open(scan_path).convert("RGB")
    target_w = max([scan.width] + [front.width for front in fronts])
    fronts = [ImageOps.contain(front, (target_w, 390), Image.BILINEAR) for front in fronts]
    scan = ImageOps.contain(scan, (target_w, 420), Image.BILINEAR)
    margin = 18
    title_h = 34
    height = sum(front.height for front in fronts) + scan.height + title_h * (len(fronts) + 1) + margin * (len(fronts) + 2)
    out = Image.new("RGB", (target_w + margin * 2, height), "white")
    draw = ImageDraw.Draw(out)
    y = margin
    for index, front in enumerate(fronts):
        title = "Front3D qualitative predicted OBBs" if index == 0 else "Front3D qualitative predicted OBBs (top-view candidate)"
        draw.text((margin, y + 8), title, fill=(20, 20, 20))
        y += title_h
        out.paste(front, (margin, y))
        y += front.height + margin
    scan_title = "ScanNet qualitative predicted OBBs"
    if "bev" in scan_path.name:
        scan_title += " (BEV alpha projection fallback)"
    else:
        scan_title += " (released RGB render view)"
    draw.text((margin, y + 8), scan_title, fill=(20, 20, 20))
    y += title_h
    out.paste(scan, (margin, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="figures_src/qualitative_detection_assets/final")
    parser.add_argument("--front3d-scene", default="3dfront_0143_00")
    parser.add_argument("--front3d-frame", default="", help="Empty string auto-selects a frame.")
    parser.add_argument("--front3d-extra-frame", default="0015", help="Optional second Front3D frame; empty disables it.")
    parser.add_argument("--front3d-render-root", default="figures_src/qualitative_detection_assets/render_cache/front3d_render_data/front3d_nerf_data")
    parser.add_argument("--front3d-feature-dir", default="dataset/finetune/front3d_rpn_data/features")
    parser.add_argument("--front3d-convention", choices=["nerf", "ngp", "opencv"], default="nerf")
    parser.add_argument("--scannet-scene", default="scene0151_00")
    parser.add_argument("--scannet-feature-dir", default="dataset/finetune/scannet_rpn_data/features")
    parser.add_argument("--scannet-obb-dir", default="dataset/finetune/scannet_rpn_data/obb")
    parser.add_argument("--scannet-render-root", default="figures_src/qualitative_detection_assets/render_cache/scannet_render_data/scannet_nerf_data")
    parser.add_argument("--scannet-split", choices=["train", "test"], default="train")
    parser.add_argument("--scannet-frame", default="", help="Empty string auto-selects a frame.")
    parser.add_argument("--scannet-bev", action="store_true", help="Force ScanNet BEV alpha projection instead of RGB overlay.")
    parser.add_argument("--no-gt", dest="include_gt", action="store_false", help="Do not include the GT panel.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    args = parser.parse_args()
    if args.front3d_frame == "":
        args.front3d_frame = None
    if args.front3d_extra_frame == "":
        args.front3d_extra_frame = None
    if args.scannet_frame == "":
        args.scannet_frame = None

    out_dir = Path(args.output_dir)
    front_paths = [make_front3d_panel(args, out_dir)]
    if args.front3d_extra_frame:
        extra_args = copy.copy(args)
        extra_args.front3d_frame = args.front3d_extra_frame
        extra_path = make_front3d_panel(extra_args, out_dir)
        if extra_path not in front_paths:
            front_paths.append(extra_path)
    scan_path = make_scannet_panel(args, out_dir)
    combined = out_dir / "fig5_qualitative_detection_draft.png"
    make_combined(front_paths, scan_path, combined)
    for front_path in front_paths:
        print(f"wrote {front_path}")
    print(f"wrote {scan_path}")
    print(f"wrote {combined}")


if __name__ == "__main__":
    main()
