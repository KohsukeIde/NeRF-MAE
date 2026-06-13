#!/usr/bin/env python3
"""Render a mesh/normal view from the Front3D alpha grid for Fig. 1.

This is a figure-facing alternative to direct alpha volume compositing. It
extracts an occupancy surface from the released `rgbsigma` grid, projects it
into the same camera view as the rendered RGB panel, and shades it by
surface normal/depth so the alpha signal reads as geometry rather than a blue
opacity blob.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy.ndimage import binary_closing, gaussian_filter, maximum_filter
from skimage.measure import marching_cubes

from render_camera_aligned_grid_views import density_to_alpha, grid_to_world
from render_nerfrpn_view_with_boxes import project_points


def load_alpha_and_meta(feature_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray | float | bool]]:
    with np.load(feature_path) as z:
        rgbsigma = z["rgbsigma"].astype(np.float32)
        meta = {
            "resolution": z["resolution"].astype(np.float64),
            "bbox_min": z["bbox_min"].astype(np.float64),
            "bbox_max": z["bbox_max"].astype(np.float64),
            "scale": float(z["scale"]),
            "offset": z["offset"].astype(np.float64),
            "from_mitsuba": bool(z["from_mitsuba"]),
        }
    return density_to_alpha(rgbsigma[..., 3]), meta


def intrinsics_from_transforms(transforms: dict) -> dict[str, float]:
    return {
        "fx": float(transforms.get("fl_x", transforms["w"] / (2.0 * np.tan(float(transforms["camera_angle_x"]) / 2.0)))),
        "fy": float(transforms.get("fl_y", transforms["h"] / (2.0 * np.tan(float(transforms["camera_angle_y"]) / 2.0)))),
        "cx": float(transforms.get("cx", transforms["w"] / 2.0)),
        "cy": float(transforms.get("cy", transforms["h"] / 2.0)),
    }


def mask_alpha_by_camera_depth(
    alpha: np.ndarray,
    meta: dict[str, np.ndarray | float | bool],
    transforms: dict,
    frame: str,
    min_depth_percentile: float | None,
    max_depth_percentile: float | None,
    alpha_threshold: float,
) -> np.ndarray:
    if min_depth_percentile is None and max_depth_percentile is None:
        return alpha

    frames = {Path(item["file_path"]).stem: item for item in transforms["frames"]}
    if frame not in frames:
        raise KeyError(f"{frame} not found in transforms")
    c2w = np.asarray(frames[frame]["transform_matrix"], dtype=np.float64)
    w2c = np.linalg.inv(c2w)

    coords = np.indices(alpha.shape, dtype=np.float32).reshape(3, -1).T
    world = grid_to_world(coords.astype(np.float64), meta)
    world_h = np.concatenate([world, np.ones((len(world), 1), dtype=np.float64)], axis=1)
    cam = (w2c @ world_h.T).T[:, :3]
    depth = (-cam[:, 2]).reshape(alpha.shape)

    valid = (alpha > alpha_threshold) & np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return alpha

    keep = valid.copy()
    if min_depth_percentile is not None:
        min_depth = np.percentile(depth[valid], min_depth_percentile)
        keep &= depth >= min_depth
    if max_depth_percentile is not None:
        max_depth = np.percentile(depth[valid], max_depth_percentile)
        keep &= depth <= max_depth

    masked = np.zeros_like(alpha, dtype=np.float32)
    masked[keep] = alpha[keep]
    print(
        f"voxel-depth mask kept {int(keep.sum())}/{int(valid.sum())} occupied voxels "
        f"({float(keep.sum()) / max(float(valid.sum()), 1.0):.3f})"
    )
    return masked


def extract_alpha_mesh(
    alpha: np.ndarray,
    meta: dict[str, np.ndarray | float | bool],
    level: float,
    smooth_sigma: float,
    step_size: int,
    connect_kernel: int,
    connect_iterations: int,
    connect_gain: float,
    close_threshold: float,
    close_kernel: int,
    close_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    field = alpha.astype(np.float32)

    # Optional 3D connection before marching cubes. This is intentionally done
    # in voxel space, not in screen space: adjacent low-density wall/furniture
    # voxels are first connected, then a mesh is extracted and rendered from the
    # same camera as the RGB panel.
    if connect_iterations > 0:
        if connect_kernel % 2 != 1:
            raise ValueError("--connect-kernel must be odd")
        connected = field
        for _ in range(connect_iterations):
            connected = maximum_filter(connected, size=connect_kernel, mode="nearest")
        field = np.maximum(field, connected * connect_gain).astype(np.float32)

    if close_iterations > 0:
        if close_kernel % 2 != 1:
            raise ValueError("--close-kernel must be odd")
        threshold = close_threshold if close_threshold > 0.0 else max(level * 0.5, 1e-4)
        structure = np.ones((close_kernel, close_kernel, close_kernel), dtype=bool)
        closed = binary_closing(field > threshold, structure=structure, iterations=close_iterations)
        field = np.maximum(field, closed.astype(np.float32) * max(level * 1.15, threshold)).astype(np.float32)

    field = gaussian_filter(field, sigma=smooth_sigma) if smooth_sigma > 0.0 else field
    verts_grid, faces, _normals, _values = marching_cubes(
        field,
        level=level,
        spacing=(1.0, 1.0, 1.0),
        step_size=step_size,
        allow_degenerate=False,
    )
    verts_world = grid_to_world(verts_grid.astype(np.float64), meta)
    return verts_world.astype(np.float32), faces.astype(np.int32)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    denom = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(denom, 1e-8)


def render_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    transforms: dict,
    frame: str,
    width: int,
    height: int,
    color_mode: str,
    blur_radius: float,
    min_depth_percentile: float | None,
    max_depth_percentile: float | None,
) -> np.ndarray:
    frames = {Path(item["file_path"]).stem: item for item in transforms["frames"]}
    if frame not in frames:
        raise KeyError(f"{frame} not found in transforms")
    c2w = np.asarray(frames[frame]["transform_matrix"], dtype=np.float64)
    intr = intrinsics_from_transforms(transforms)
    pts2d, depth = project_points(vertices, c2w, intr, convention="nerf")
    normals = face_normals(vertices, faces)

    cam_pos = c2w[:3, 3]
    centers = vertices[faces].mean(axis=1)
    view_dir = cam_pos[None, :] - centers
    view_dir = view_dir / np.maximum(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-8)
    normal_view = np.abs((normals * view_dir).sum(axis=1))

    light_dir = (c2w[:3, :3] @ np.array([0.25, -0.45, -1.0], dtype=np.float64))
    light_dir = light_dir / np.linalg.norm(light_dir)
    normal_light = np.maximum(0.0, (normals * light_dir[None, :]).sum(axis=1))
    shade = 0.28 + 0.50 * normal_view + 0.22 * normal_light
    shade = np.clip(shade, 0.12, 1.0)

    face_depth = depth[faces].mean(axis=1)
    valid = np.all(depth[faces] > 0.0, axis=1)
    valid &= np.all(np.isfinite(pts2d[faces]), axis=(1, 2))
    margin = max(width, height) * 0.15
    tri2d = pts2d[faces]
    valid &= np.max(tri2d[..., 0], axis=1) >= -margin
    valid &= np.min(tri2d[..., 0], axis=1) <= width + margin
    valid &= np.max(tri2d[..., 1], axis=1) >= -margin
    valid &= np.min(tri2d[..., 1], axis=1) <= height + margin

    visible_depth = face_depth[valid]
    if len(visible_depth) > 0:
        if min_depth_percentile is not None:
            min_depth = np.percentile(visible_depth, min_depth_percentile)
            valid &= face_depth >= min_depth
        if max_depth_percentile is not None:
            max_depth = np.percentile(visible_depth, max_depth_percentile)
            valid &= face_depth <= max_depth

    order = np.argsort(face_depth[valid])[::-1]
    valid_faces = np.nonzero(valid)[0][order]

    bg = np.array([0.92, 0.97, 1.00], dtype=np.float32)
    canvas = Image.new("RGB", (width, height), tuple((bg * 255).astype(np.uint8)))
    draw = ImageDraw.Draw(canvas, "RGBA")

    if len(valid_faces) == 0:
        return np.asarray(canvas).astype(np.float32) / 255.0
    depth_valid = face_depth[valid_faces]
    d_lo, d_hi = np.percentile(depth_valid, [2, 98])
    depth_norm = np.clip((face_depth - d_lo) / max(d_hi - d_lo, 1e-8), 0.0, 1.0)

    for idx in valid_faces:
        poly = [tuple(p) for p in tri2d[idx]]
        if color_mode == "gray":
            base_near = np.array([0.22, 0.29, 0.36], dtype=np.float32)
            base_far = np.array([0.74, 0.80, 0.86], dtype=np.float32)
        else:
            base_near = np.array([0.10, 0.34, 0.82], dtype=np.float32)
            base_far = np.array([0.68, 0.88, 1.00], dtype=np.float32)
        base = base_near * (1.0 - depth_norm[idx]) + base_far * depth_norm[idx]
        rgb = np.clip(base * shade[idx] + 0.10, 0.0, 1.0)
        alpha = 235 if color_mode == "gray" else 225
        draw.polygon(poly, fill=tuple((rgb * 255).astype(np.uint8).tolist() + [alpha]))

    if blur_radius > 0.0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.asarray(canvas).astype(np.float32) / 255.0


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def make_compare_sheet(render_path: Path, mesh_paths: list[Path], out_path: Path) -> None:
    paths = [render_path] + mesh_paths
    labels = ["rendered RGB"] + [p.stem.replace("fig1_", "").replace("_", " ") for p in mesh_paths]
    images = [Image.open(p).convert("RGB") for p in paths]
    thumb_w, thumb_h = 320, 240
    images = [ImageOps.contain(img, (thumb_w, thumb_h), Image.BILINEAR) for img in images]
    margin, label_h = 14, 28
    sheet = Image.new("RGB", (thumb_w * len(images) + margin * (len(images) + 1), thumb_h + label_h + margin * 2), "white")
    draw = ImageDraw.Draw(sheet)
    x = margin
    for img, label in zip(images, labels):
        bg = Image.new("RGB", (thumb_w, thumb_h), (248, 248, 248))
        bg.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
        sheet.paste(bg, (x, margin))
        draw.rectangle([x, margin, x + thumb_w - 1, margin + thumb_h - 1], outline=(210, 210, 210), width=1)
        draw.text((x + 4, margin + thumb_h + 7), label[:42], fill=(20, 20, 20))
        x += thumb_w + margin
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="3dfront_0131_00")
    parser.add_argument("--frame", default="0180")
    parser.add_argument("--feature-dir", default="dataset/finetune/front3d_rpn_data/features")
    parser.add_argument(
        "--render-root",
        default="figures_src/fig1_render_view_assets/render_cache/nerf_rpn_front3d_nerf_data/front3d_nerf_data",
    )
    parser.add_argument("--output-dir", default="figures_src/fig1_render_view_assets/paper_candidates")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--level", type=float, default=0.16)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--step-size", type=int, default=2)
    parser.add_argument("--blur-radius", type=float, default=0.0)
    parser.add_argument("--connect-kernel", type=int, default=1)
    parser.add_argument("--connect-iterations", type=int, default=0)
    parser.add_argument("--connect-gain", type=float, default=0.65)
    parser.add_argument("--close-threshold", type=float, default=0.0)
    parser.add_argument("--close-kernel", type=int, default=3)
    parser.add_argument("--close-iterations", type=int, default=0)
    parser.add_argument(
        "--min-depth-percentile",
        type=float,
        default=None,
        help="Render only faces farther than this visible-face depth percentile. Diagnostic for back-only mesh.",
    )
    parser.add_argument(
        "--max-depth-percentile",
        type=float,
        default=None,
        help="Render only faces nearer than this visible-face depth percentile.",
    )
    parser.add_argument(
        "--min-voxel-depth-percentile",
        type=float,
        default=None,
        help="Zero out foreground voxels before meshing; keep occupied voxels farther than this depth percentile.",
    )
    parser.add_argument(
        "--max-voxel-depth-percentile",
        type=float,
        default=None,
        help="Zero out back voxels before meshing; keep occupied voxels nearer than this depth percentile.",
    )
    parser.add_argument("--voxel-depth-alpha-threshold", type=float, default=0.01)
    args = parser.parse_args()

    scene_root = Path(args.render_root) / args.scene / "train"
    transforms = json.loads((scene_root / "transforms.json").read_text())
    alpha, meta = load_alpha_and_meta(Path(args.feature_dir) / f"{args.scene}.npz")
    alpha = mask_alpha_by_camera_depth(
        alpha,
        meta,
        transforms,
        frame=args.frame,
        min_depth_percentile=args.min_voxel_depth_percentile,
        max_depth_percentile=args.max_voxel_depth_percentile,
        alpha_threshold=args.voxel_depth_alpha_threshold,
    )
    vertices, faces = extract_alpha_mesh(
        alpha,
        meta,
        level=args.level,
        smooth_sigma=args.smooth_sigma,
        step_size=args.step_size,
        connect_kernel=args.connect_kernel,
        connect_iterations=args.connect_iterations,
        connect_gain=args.connect_gain,
        close_threshold=args.close_threshold,
        close_kernel=args.close_kernel,
        close_iterations=args.close_iterations,
    )
    print(
        f"mesh vertices={len(vertices)} faces={len(faces)} level={args.level} "
        f"smooth={args.smooth_sigma} step={args.step_size} "
        f"connect={args.connect_kernel}x{args.connect_iterations}@{args.connect_gain} "
        f"close={args.close_kernel}x{args.close_iterations}@{args.close_threshold}"
    )

    out_dir = Path(args.output_dir)
    outputs: list[Path] = []
    for mode in ["blue", "gray"]:
        image = render_mesh(
            vertices=vertices,
            faces=faces,
            transforms=transforms,
            frame=args.frame,
            width=args.width,
            height=args.height,
            color_mode=mode,
            blur_radius=args.blur_radius,
            min_depth_percentile=args.min_depth_percentile,
            max_depth_percentile=args.max_depth_percentile,
        )
        path = out_dir / f"fig1_{args.scene}_{args.frame}_alpha_mesh_{mode}_normal_depth.png"
        save_rgb(path, image)
        outputs.append(path)
        print(f"wrote {path}")

    render_rgb = out_dir / "fig1_render_rgb_bbox_candidate.png"
    if not render_rgb.exists():
        render_rgb = Path("figures_src/fig1_render_view_assets/final") / f"fig1_{args.scene}_{args.frame}_render_rgb_bbox_furniture.png"
    if render_rgb.exists():
        sheet = out_dir / f"fig1_{args.scene}_{args.frame}_alpha_mesh_compare.png"
        make_compare_sheet(render_rgb, outputs, sheet)
        print(f"wrote {sheet}")


if __name__ == "__main__":
    main()
