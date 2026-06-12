#!/usr/bin/env python3
"""Render alpha/RGBA grid views aligned to a released Front3D camera frame.

This complements the real rendered RGB image from `front3d_nerf_data` with
camera-aligned views of the extracted radiance-density grid used by NeRF-MAE.
It is intended for Fig. 1 panels where the same rendered view should be shown
as RGB appearance and alpha/structure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps


def density_to_alpha(density: np.ndarray) -> np.ndarray:
    density = np.clip(density, -20.0, 20.0)
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


def connect_rgba_volume(
    volume: torch.Tensor,
    kernel_size: int,
    iterations: int,
    alpha_gain: float,
    rgb_fill_kernel: int,
) -> torch.Tensor:
    """Connect sparse neighboring voxels for figure-facing grid rendering.

    The released `rgbsigma` grid is an extracted radiance-density grid, not a
    full instant-ngp render. Thin walls/furniture may have small holes at the
    voxel level; direct alpha compositing over a white background makes those
    gaps look like missing depth. We close those small gaps in the 3D volume
    before ray marching, and fill RGB in newly occupied voxels from nearby
    alpha-weighted colors.
    """
    if iterations <= 0:
        return volume
    if kernel_size % 2 != 1 or rgb_fill_kernel % 2 != 1:
        raise ValueError("connection kernels must be odd")

    rgb = volume[:, :3]
    alpha = volume[:, 3:4]

    connected = alpha
    padding = kernel_size // 2
    for _ in range(iterations):
        connected = F.max_pool3d(connected, kernel_size=kernel_size, stride=1, padding=padding)
    connected = torch.maximum(alpha, (connected * alpha_gain).clamp(0.0, 1.0))

    fill_padding = rgb_fill_kernel // 2
    color_num = F.avg_pool3d(rgb * alpha, kernel_size=rgb_fill_kernel, stride=1, padding=fill_padding)
    color_den = F.avg_pool3d(alpha, kernel_size=rgb_fill_kernel, stride=1, padding=fill_padding)
    local_rgb = color_num / color_den.clamp_min(1e-6)
    fill_mask = ((connected > alpha) & (color_den > 1e-5)).expand_as(rgb)
    rgb = torch.where(fill_mask, local_rgb, rgb)
    return torch.cat([rgb.clamp(0.0, 1.0), connected.clamp(0.0, 1.0)], dim=1)


def load_rgba_volume(
    feature_path: Path,
    connect_voxels: bool = False,
    connect_kernel: int = 5,
    connect_iterations: int = 2,
    connect_alpha_gain: float = 0.75,
    rgb_fill_kernel: int = 7,
) -> tuple[torch.Tensor, dict[str, np.ndarray | float | bool]]:
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
    rgb = np.clip(rgbsigma[..., :3], 0.0, 1.0)
    alpha = density_to_alpha(rgbsigma[..., 3])
    rgba = np.concatenate([rgb, alpha[..., None].astype(np.float32)], axis=-1)
    # grid_sample expects [N, C, D, H, W], with normalized grid order x, y, z.
    # The source rgbsigma is [W, L, H, C], so spatial dims become [H, L, W].
    volume = torch.from_numpy(rgba).permute(3, 2, 1, 0).unsqueeze(0).contiguous()
    volume = volume.float()
    if connect_voxels:
        volume = connect_rgba_volume(
            volume,
            kernel_size=connect_kernel,
            iterations=connect_iterations,
            alpha_gain=connect_alpha_gain,
            rgb_fill_kernel=rgb_fill_kernel,
        )
    return volume, meta


def grid_to_world(grid_points: np.ndarray, meta: dict[str, np.ndarray | float | bool]) -> np.ndarray:
    grid_res = meta["resolution"]
    bbox_min = meta["bbox_min"]
    bbox_max = meta["bbox_max"]
    scale = float(meta["scale"])
    offset = meta["offset"]
    from_mitsuba = bool(meta["from_mitsuba"])
    perm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)

    diag = bbox_max - bbox_min
    pos = grid_points / grid_res * diag + bbox_min
    x = (perm @ pos.T).T
    off = perm @ offset
    if from_mitsuba:
        x[:, [0, 2]] *= -1
    else:
        x = x[:, [2, 0, 1]]
    return (x - off) / scale


def world_to_grid(world_points: torch.Tensor, meta: dict[str, np.ndarray | float | bool]) -> torch.Tensor:
    # Inverse of grid_to_world for the Front3D/non-mitsuba path used here.
    if bool(meta["from_mitsuba"]):
        raise NotImplementedError("from_mitsuba=True is not needed for selected Front3D Fig.1 scene.")
    grid_res = torch.as_tensor(meta["resolution"], dtype=world_points.dtype, device=world_points.device)
    bbox_min = torch.as_tensor(meta["bbox_min"], dtype=world_points.dtype, device=world_points.device)
    bbox_max = torch.as_tensor(meta["bbox_max"], dtype=world_points.dtype, device=world_points.device)
    scale = torch.as_tensor(float(meta["scale"]), dtype=world_points.dtype, device=world_points.device)
    offset = torch.as_tensor(meta["offset"], dtype=world_points.dtype, device=world_points.device)
    # Inverse of the point-position part of proposals2ngp.py's non-mitsuba
    # path. The y/z sign flip there applies to orientation columns, not to
    # the translation column used for point locations.
    off = torch.stack([offset[1], offset[2], offset[0]])
    pos = world_points * scale + off
    diag = bbox_max - bbox_min
    return (pos - bbox_min) / diag * grid_res


def ray_box_intersection(orig: torch.Tensor, dirs: torch.Tensor, box_min: torch.Tensor, box_max: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inv_d = 1.0 / torch.where(torch.abs(dirs) < 1e-8, torch.full_like(dirs, 1e-8), dirs)
    t0 = (box_min - orig) * inv_d
    t1 = (box_max - orig) * inv_d
    tmin = torch.minimum(t0, t1).amax(dim=-1)
    tmax = torch.maximum(t0, t1).amin(dim=-1)
    return tmin.clamp_min(0.0), tmax


def render_camera_aligned(
    volume: torch.Tensor,
    meta: dict[str, np.ndarray | float | bool],
    transforms: dict,
    frame: str,
    width: int,
    height: int,
    samples: int,
    tile_rows: int,
    opacity_scale: float,
    alpha_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = {Path(f["file_path"]).stem: f for f in transforms["frames"]}
    c2w = torch.as_tensor(frames[frame]["transform_matrix"], dtype=torch.float32)
    fx = float(transforms.get("fl_x", transforms["w"] / (2.0 * np.tan(float(transforms["camera_angle_x"]) / 2.0))))
    fy = float(transforms.get("fl_y", transforms["h"] / (2.0 * np.tan(float(transforms["camera_angle_y"]) / 2.0))))
    cx = float(transforms.get("cx", transforms["w"] / 2.0))
    cy = float(transforms.get("cy", transforms["h"] / 2.0))

    # Render at native camera geometry, optionally down/up-sampled by width/height.
    sx = float(transforms["w"]) / float(width)
    sy = float(transforms["h"]) / float(height)
    fx_s, fy_s, cx_s, cy_s = fx / sx, fy / sy, cx / sx, cy / sy

    grid_shape = np.array(volume.shape[-1:-4:-1], dtype=np.float64)  # [W, L, H]
    corners = np.array(
        [[x, y, z] for x in [0, grid_shape[0]] for y in [0, grid_shape[1]] for z in [0, grid_shape[2]]],
        dtype=np.float64,
    )
    world_corners = grid_to_world(corners, meta)
    box_min = torch.as_tensor(world_corners.min(axis=0), dtype=torch.float32)
    box_max = torch.as_tensor(world_corners.max(axis=0), dtype=torch.float32)

    rgb_out = torch.ones((height, width, 3), dtype=torch.float32)
    opacity_out = torch.zeros((height, width), dtype=torch.float32)
    depth_out = torch.ones((height, width), dtype=torch.float32)
    R = c2w[:3, :3]
    origin = c2w[:3, 3]
    ts_base = torch.linspace(0.0, 1.0, samples, dtype=torch.float32)

    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, dtype=torch.float32),
            torch.arange(0, width, dtype=torch.float32),
            indexing="ij",
        )
        dirs_cam = torch.stack([(xx - cx_s) / fx_s, -(yy - cy_s) / fy_s, -torch.ones_like(xx)], dim=-1)
        dirs = torch.matmul(dirs_cam.reshape(-1, 3), R.T)
        dirs = F.normalize(dirs, dim=-1)
        orig = origin.expand_as(dirs)
        t_near, t_far = ray_box_intersection(orig, dirs, box_min, box_max)
        valid = t_far > t_near
        ts = t_near[:, None] + (t_far - t_near)[:, None] * ts_base[None, :]
        pts_world = orig[:, None, :] + dirs[:, None, :] * ts[..., None]
        pts_grid = world_to_grid(pts_world.reshape(-1, 3), meta).reshape(-1, samples, 3)
        res = torch.as_tensor(meta["resolution"], dtype=torch.float32)
        norm = torch.empty_like(pts_grid)
        norm[..., 0] = pts_grid[..., 0] / max(float(res[0] - 1), 1.0) * 2.0 - 1.0
        norm[..., 1] = pts_grid[..., 1] / max(float(res[1] - 1), 1.0) * 2.0 - 1.0
        norm[..., 2] = pts_grid[..., 2] / max(float(res[2] - 1), 1.0) * 2.0 - 1.0
        norm = norm.reshape(1, (y1 - y0) * width, samples, 1, 3)
        sampled = F.grid_sample(volume, norm, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled = sampled.reshape(4, (y1 - y0) * width, samples).permute(1, 2, 0)
        alpha = sampled[..., 3].clamp(0.0, 1.0)
        alpha = ((alpha - alpha_threshold) / max(1.0 - alpha_threshold, 1e-6)).clamp(0.0, 1.0)
        step_alpha = (1.0 - torch.exp(-opacity_scale * alpha)).clamp(0.0, 0.95)
        step_alpha = torch.where(valid[:, None], step_alpha, torch.zeros_like(step_alpha))
        trans = torch.cumprod(torch.cat([torch.ones_like(step_alpha[:, :1]), 1.0 - step_alpha + 1e-6], dim=1), dim=1)[:, :-1]
        weights = trans * step_alpha
        opacity = weights.sum(dim=1).clamp(0.0, 1.0)
        color = (weights[..., None] * sampled[..., :3].clamp(0.0, 1.0)).sum(dim=1)
        depth = (weights * ts_base[None, :]).sum(dim=1) / opacity.clamp_min(1e-6)
        surface_rgb = color / opacity[:, None].clamp_min(1e-4)
        # Figure-facing grid view: the extracted radiance-density grid is much
        # sparser than the original NGP render. Compositing directly over white
        # makes far/low-density structure disappear, so use an opacity matte
        # over a very light cool background while preserving sampled RGB.
        matte = (opacity * 1.75).clamp(0.0, 1.0).pow(0.55)
        bg = torch.tensor([0.93, 0.95, 0.97], dtype=torch.float32)
        rgb = surface_rgb.clamp(0.0, 1.0) * matte[:, None] + bg[None, :] * (1.0 - matte[:, None])
        rgb_out[y0:y1] = rgb.reshape(y1 - y0, width, 3)
        opacity_out[y0:y1] = opacity.reshape(y1 - y0, width)
        depth_out[y0:y1] = depth.reshape(y1 - y0, width)
    return rgb_out.numpy(), opacity_out.numpy(), depth_out.numpy()


def connect_screen_gaps(
    rgb: np.ndarray,
    opacity: np.ndarray,
    depth: np.ndarray,
    kernel_size: int,
    iterations: int,
    opacity_gain: float,
    fill_opacity_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill only low-opacity screen-space gaps left by sparse voxel footprints.

    This intentionally preserves existing high-opacity pixels. Strong global
    dilation makes the foreground look worse, so the fill is restricted to
    white/transparent holes that have nearby occupied evidence.
    """
    if iterations <= 0:
        return rgb, opacity, depth
    if kernel_size % 2 != 1:
        raise ValueError("screen connection kernel must be odd")

    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    op_t = torch.from_numpy(opacity).unsqueeze(0).unsqueeze(0).float()
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()

    padding = kernel_size // 2
    connected = op_t
    for _ in range(iterations):
        connected = F.max_pool2d(connected, kernel_size=kernel_size, stride=1, padding=padding)

    color_num = F.avg_pool2d(rgb_t * op_t, kernel_size=kernel_size, stride=1, padding=padding)
    color_den = F.avg_pool2d(op_t, kernel_size=kernel_size, stride=1, padding=padding)
    local_rgb = color_num / color_den.clamp_min(1e-6)
    depth_num = F.avg_pool2d(depth_t * op_t, kernel_size=kernel_size, stride=1, padding=padding)
    local_depth = depth_num / color_den.clamp_min(1e-6)

    fill = ((op_t < fill_opacity_threshold) & (connected > fill_opacity_threshold) & (color_den > 1e-4))
    connected = torch.where(fill, torch.maximum(op_t, (connected * opacity_gain).clamp(0.0, 1.0)), op_t)
    fill = fill.expand_as(rgb_t)
    rgb_t = torch.where(fill, local_rgb, rgb_t)
    depth_t = torch.where(fill[:, :1], local_depth, depth_t)
    return (
        rgb_t.squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0),
        connected.squeeze(0).squeeze(0).numpy().clip(0.0, 1.0),
        depth_t.squeeze(0).squeeze(0).numpy().clip(0.0, 1.0),
    )


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def composite_grid_over_render_background(grid_rgb: np.ndarray, opacity: np.ndarray, render_rgb_path: Path) -> np.ndarray:
    """Use the matching real render as the matte background for grid RGBA.

    The extracted radiance-density grid has low-opacity holes that become
    white if rendered over a blank background. For a figure-facing same-camera
    view, show grid color where the grid has opacity and let the corresponding
    released render show through where the grid is transparent.
    """
    if not render_rgb_path.exists():
        return grid_rgb
    h, w = opacity.shape
    render_bg = Image.open(render_rgb_path).convert("RGB").resize((w, h), Image.BILINEAR)
    render_bg = np.asarray(render_bg).astype(np.float32) / 255.0

    internal_bg = np.array([0.93, 0.95, 0.97], dtype=np.float32)
    matte = np.clip(opacity * 1.75, 0.0, 1.0) ** 0.55
    matte_3 = matte[..., None]
    surface_rgb = (grid_rgb - internal_bg[None, None, :] * (1.0 - matte_3)) / np.maximum(matte_3, 1e-4)
    surface_rgb = np.clip(surface_rgb, 0.0, 1.0)
    return surface_rgb * matte_3 + render_bg * (1.0 - matte_3)


def save_alpha(path: Path, opacity: np.ndarray, depth: np.ndarray) -> None:
    # Figure-friendly blue structure view with depth shading. Keep this lighter
    # than the raw depth map so Fig. 1 reads as structure, not a saturated mask.
    alpha_strength = np.clip(0.76 * (np.clip(opacity, 0.0, 1.0) ** 0.64), 0.0, 0.84)
    depth = np.clip(depth, 0.0, 1.0)
    valid = opacity > 0.015
    if np.any(valid):
        lo, hi = np.percentile(depth[valid], [2, 98])
        depth = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    near = np.array([0.16, 0.43, 0.92], dtype=np.float32)
    far = np.array([0.58, 0.84, 1.00], dtype=np.float32)
    structure_color = near[None, None, :] * (1.0 - depth[..., None]) + far[None, None, :] * depth[..., None]
    bg = np.array([0.90, 0.96, 1.00], dtype=np.float32)
    img = bg[None, None, :] * (1.0 - alpha_strength[..., None]) + alpha_strength[..., None] * structure_color
    save_rgb(path, img)


def make_quad(render_rgb_path: Path, grid_rgba_path: Path, alpha_path: Path, bbox_path: Path, out_path: Path) -> None:
    paths = [render_rgb_path, grid_rgba_path, alpha_path, bbox_path]
    labels = ["rendered RGB", "grid RGBA over render", "grid alpha / structure", "rendered RGB + boxes"]
    imgs = [Image.open(p).convert("RGB") for p in paths]
    thumb_w, thumb_h = 360, 270
    imgs = [ImageOps.contain(img, (thumb_w, thumb_h), Image.BILINEAR) for img in imgs]
    margin, label_h = 16, 30
    sheet = Image.new("RGB", (thumb_w * 2 + margin * 3, (thumb_h + label_h) * 2 + margin * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (img, label) in enumerate(zip(imgs, labels)):
        row, col = divmod(i, 2)
        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + label_h + margin)
        bg = Image.new("RGB", (thumb_w, thumb_h), (248, 248, 248))
        bg.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
        sheet.paste(bg, (x, y))
        draw.rectangle([x, y, x + thumb_w - 1, y + thumb_h - 1], outline=(210, 210, 210), width=1)
        draw.text((x + 4, y + thumb_h + 7), label, fill=(25, 25, 25))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def save_crop_set(
    render_rgb_path: Path,
    grid_rgba_path: Path,
    alpha_path: Path,
    bbox_path: Path,
    scene: str,
    frame: str,
    crop: tuple[int, int, int, int],
    out: Path,
) -> None:
    """Save a paper-facing crop that keeps the render/grid panels aligned."""
    source_paths = {
        "render_rgb": render_rgb_path,
        "grid_rgba": grid_rgba_path,
        "grid_alpha": alpha_path,
        "bbox": bbox_path,
    }
    cropped = {}
    for key, path in source_paths.items():
        crop_path = out / f"fig1_{scene}_{frame}_{key}_crop_sofa_wall.png"
        Image.open(path).convert("RGB").crop(crop).save(crop_path)
        cropped[key] = crop_path

    thumb_w, thumb_h = 260, 320
    margin, label_h = 16, 30
    labels = [
        ("render_rgb", "rendered RGB crop"),
        ("grid_rgba", "grid RGBA crop"),
        ("grid_alpha", "grid alpha / structure crop"),
        ("bbox", "rendered RGB + boxes crop"),
    ]
    sheet = Image.new("RGB", (thumb_w * 2 + margin * 3, (thumb_h + label_h) * 2 + margin * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (key, label) in enumerate(labels):
        img = Image.open(cropped[key]).convert("RGB")
        img = ImageOps.contain(img, (thumb_w, thumb_h), Image.BILINEAR)
        row, col = divmod(i, 2)
        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + label_h + margin)
        bg = Image.new("RGB", (thumb_w, thumb_h), (248, 248, 248))
        bg.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
        sheet.paste(bg, (x, y))
        draw.rectangle([x, y, x + thumb_w - 1, y + thumb_h - 1], outline=(210, 210, 210), width=1)
        draw.text((x + 4, y + thumb_h + 7), label, fill=(25, 25, 25))
    sheet.save(out / f"fig1_{scene}_{frame}_render_rgb_grid_rgba_alpha_bbox_quad_crop_sofa_wall.png")


def parse_crop(raw: str) -> tuple[int, int, int, int]:
    vals = [int(v.strip()) for v in raw.split(",")]
    if len(vals) != 4:
        raise ValueError("--crop must be formatted as left,top,right,bottom")
    return vals[0], vals[1], vals[2], vals[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="3dfront_0131_00")
    parser.add_argument("--frame", default="0180")
    parser.add_argument("--feature-dir", default="dataset/finetune/front3d_rpn_data/features")
    parser.add_argument(
        "--render-root",
        default="figures_src/fig1_render_view_assets/nerf_rpn_front3d_nerf_data/front3d_nerf_data",
    )
    parser.add_argument("--output-dir", default="figures_src/fig1_render_view_assets/final")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--tile-rows", type=int, default=24)
    parser.add_argument("--opacity-scale", type=float, default=0.75)
    parser.add_argument("--alpha-threshold", type=float, default=0.0)
    parser.add_argument("--connect-voxels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--connect-kernel", type=int, default=5)
    parser.add_argument("--connect-iterations", type=int, default=2)
    parser.add_argument("--connect-alpha-gain", type=float, default=0.75)
    parser.add_argument("--rgb-fill-kernel", type=int, default=7)
    parser.add_argument("--screen-connect-kernel", type=int, default=31)
    parser.add_argument("--screen-connect-iterations", type=int, default=1)
    parser.add_argument("--screen-connect-opacity-gain", type=float, default=0.9)
    parser.add_argument("--screen-fill-opacity-threshold", type=float, default=0.18)
    parser.add_argument("--alpha-screen-connect-kernel", type=int, default=71)
    parser.add_argument("--alpha-screen-connect-iterations", type=int, default=2)
    parser.add_argument("--alpha-screen-connect-opacity-gain", type=float, default=0.85)
    parser.add_argument("--alpha-screen-fill-opacity-threshold", type=float, default=0.35)
    parser.add_argument(
        "--crop",
        default="",
        help="Aligned crop for paper-facing panels, formatted left,top,right,bottom. Empty string disables crops.",
    )
    args = parser.parse_args()

    scene_root = Path(args.render_root) / args.scene
    transforms_path = scene_root / "train" / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(
            f"Missing {transforms_path}. Run figures_src/download_front3d_render_scene.py --scene {args.scene} first."
        )
    transforms = json.loads(transforms_path.read_text())
    volume, meta = load_rgba_volume(
        Path(args.feature_dir) / f"{args.scene}.npz",
        connect_voxels=args.connect_voxels,
        connect_kernel=args.connect_kernel,
        connect_iterations=args.connect_iterations,
        connect_alpha_gain=args.connect_alpha_gain,
        rgb_fill_kernel=args.rgb_fill_kernel,
    )
    grid_rgb_raw, opacity_raw, depth_raw = render_camera_aligned(
        volume=volume,
        meta=meta,
        transforms=transforms,
        frame=args.frame,
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

    out = Path(args.output_dir)
    rgba_path = out / f"fig1_{args.scene}_{args.frame}_grid_rgba_same_camera.png"
    alpha_path = out / f"fig1_{args.scene}_{args.frame}_grid_alpha_same_camera.png"
    render_rgb_path = out / f"fig1_{args.scene}_{args.frame}_render_rgb.png"
    bbox_path = out / f"fig1_{args.scene}_{args.frame}_render_rgb_bbox_furniture.png"
    grid_rgb = composite_grid_over_render_background(grid_rgb, opacity, render_rgb_path)
    save_rgb(rgba_path, grid_rgb)
    save_alpha(alpha_path, opacity_alpha, depth_alpha)

    if render_rgb_path.exists() and bbox_path.exists():
        make_quad(
            render_rgb_path,
            rgba_path,
            alpha_path,
            bbox_path,
            out / f"fig1_{args.scene}_{args.frame}_render_rgb_grid_rgba_alpha_bbox_quad.png",
        )
        if args.crop:
            save_crop_set(
                render_rgb_path=render_rgb_path,
                grid_rgba_path=rgba_path,
                alpha_path=alpha_path,
                bbox_path=bbox_path,
                scene=args.scene,
                frame=args.frame,
                crop=parse_crop(args.crop),
                out=out,
            )
    print(f"wrote {rgba_path}")
    print(f"wrote {alpha_path}")


if __name__ == "__main__":
    main()
