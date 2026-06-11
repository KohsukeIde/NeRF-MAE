#!/usr/bin/env python3
"""Render real Front3D rgbsigma grids as Fig. 1 volume-render assets.

This script intentionally avoids point/scatter visualization. It renders the
released Front3D radiance-density grid with orthographic ray marching and
front-to-back alpha compositing so the output reads as a volume-rendered scene
rather than as raw voxel samples.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ViewSpec:
    name: str
    azimuth: float
    elevation: float


DEFAULT_VIEW = ViewSpec("main", -42.0, 57.0)
VIEW_CANDIDATES = (
    DEFAULT_VIEW,
    ViewSpec("view_b", -28.0, 62.0),
    ViewSpec("view_c", -58.0, 52.0),
)


def density_to_alpha(density: np.ndarray) -> np.ndarray:
    """Front3D density-to-alpha conversion used by `nerf_rpn/datasets.py`."""
    density = np.clip(density, -20.0, 20.0)
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


def scene_score(feature_path: Path, boxes_path: Path | None) -> dict[str, float | str | int]:
    with np.load(feature_path) as z:
        rgbsigma = z["rgbsigma"].astype(np.float32)
    alpha = density_to_alpha(rgbsigma[..., 3])
    mask = alpha > 0.03
    rgb = np.clip(rgbsigma[..., :3], 0.0, 1.0)
    rgb_valid = rgb[mask]
    if mask.any():
        coords = np.argwhere(mask)
        ext = coords.max(axis=0) - coords.min(axis=0) + 1
        occ = float(mask.mean())
        alpha_mass = float(alpha[mask].mean())
        color_std = float(rgb_valid.std(axis=0).mean()) if rgb_valid.size else 0.0
        color_range = float(np.quantile(rgb_valid, 0.95) - np.quantile(rgb_valid, 0.05)) if rgb_valid.size else 0.0
        proj_area = float(mask.max(axis=2).mean())
        compact = float(np.prod(ext / np.array(mask.shape)))
    else:
        occ = alpha_mass = color_std = color_range = proj_area = compact = 0.0
    nboxes = 0
    if boxes_path is not None and boxes_path.exists():
        nboxes = int(np.load(boxes_path, allow_pickle=True).shape[0])
    box_term = math.exp(-((nboxes - 7.0) ** 2) / 18.0)
    occ_term = math.exp(-((occ - 0.28) ** 2) / 0.035)
    score = (
        2.0 * box_term
        + 1.5 * occ_term
        + 1.2 * color_std
        + 0.8 * color_range
        + 0.8 * proj_area
        + 0.4 * compact
        + 0.4 * alpha_mass
    )
    return {
        "scene": feature_path.stem,
        "feature_path": str(feature_path),
        "boxes": nboxes,
        "occupancy": occ,
        "alpha_mean": alpha_mass,
        "color_std": color_std,
        "color_range": color_range,
        "proj_area": proj_area,
        "compact": compact,
        "score": float(score),
    }


def load_rgba_volume(feature_path: Path, gamma: float, auto_exposure: bool) -> tuple[torch.Tensor, np.ndarray]:
    """Return volume as torch tensor `[1, 4, Z, Y, X]` and native XYZ shape."""
    with np.load(feature_path) as z:
        rgbsigma = z["rgbsigma"].astype(np.float32)
    rgb = np.clip(rgbsigma[..., :3], 0.0, 1.0)
    alpha = density_to_alpha(rgbsigma[..., 3])
    mask = alpha > 0.03
    if auto_exposure and mask.any():
        valid = rgb[mask]
        lo = float(np.quantile(valid, 0.01))
        hi = float(np.quantile(valid, 0.995))
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = np.clip(rgb, 0.0, 1.0) ** (1.0 / gamma)
    rgba = np.concatenate([rgb, alpha[..., None].astype(np.float32)], axis=-1)
    # grid_sample expects `[N, C, D, H, W]`, where grid coordinates are x, y, z.
    volume = torch.from_numpy(np.transpose(rgba, (3, 2, 1, 0))).unsqueeze(0).contiguous()
    return volume.float(), np.array(alpha.shape, dtype=np.float32)


def rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    rz = np.array(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az), np.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(el), -np.sin(el)],
            [0.0, np.sin(el), np.cos(el)],
        ],
        dtype=np.float32,
    )
    # Row-vector convention: camera = object @ R.T, object = camera @ R.
    return rx @ rz


def camera_bounds(native_shape_xyz: np.ndarray, view: ViewSpec, padding: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aspect = native_shape_xyz / float(native_shape_xyz.max())
    corners = np.array(
        [
            [sx * aspect[0] / 2.0, sy * aspect[1] / 2.0, sz * aspect[2] / 2.0]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float32,
    )
    rot = rotation_matrix(view.azimuth, view.elevation)
    cam = corners @ rot.T
    lo = cam.min(axis=0)
    hi = cam.max(axis=0)
    span = hi - lo
    lo[:2] -= padding * span[:2]
    hi[:2] += padding * span[:2]
    lo[2] -= 0.02 * span[2]
    hi[2] += 0.02 * span[2]
    return lo, hi, aspect, rot


def composite_tile(
    samples: torch.Tensor,
    depth_values: torch.Tensor,
    alpha_threshold: float,
    opacity_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # samples: `[4, S, H, W]`, sampled front-to-back.
    rgb = samples[:3].permute(1, 2, 3, 0).clamp(0.0, 1.0)
    alpha = samples[3].clamp(0.0, 1.0)
    alpha = ((alpha - alpha_threshold) / max(1.0 - alpha_threshold, 1e-6)).clamp(0.0, 1.0)
    # Per-step opacity. The exponent form keeps the knob stable for different
    # ray sample counts while avoiding immediate saturation.
    a = (1.0 - torch.exp(-opacity_scale * alpha)).clamp(0.0, 0.98)
    one_minus = (1.0 - a).clamp_min(1e-6)
    trans_inclusive = torch.cumprod(one_minus, dim=0)
    trans_exclusive = torch.cat([torch.ones_like(trans_inclusive[:1]), trans_inclusive[:-1]], dim=0)
    weights = trans_exclusive * a
    opacity = weights.sum(dim=0).clamp(0.0, 1.0)
    color = (weights[..., None] * rgb).sum(dim=0)
    depth = (weights * depth_values[:, None, None]).sum(dim=0) / opacity.clamp_min(1e-6)
    return color, opacity, depth


def boost_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    """Increase chroma while preserving luminance for print readability."""
    if factor <= 1.0:
        return img
    luma = (
        0.299 * img[..., 0]
        + 0.587 * img[..., 1]
        + 0.114 * img[..., 2]
    )
    return np.clip(luma[..., None] + factor * (img - luma[..., None]), 0.0, 1.0)


def render_volume(
    volume: torch.Tensor,
    native_shape_xyz: np.ndarray,
    view: ViewSpec,
    width: int,
    height: int,
    samples: int,
    tile_rows: int,
    alpha_threshold: float,
    opacity_scale: float,
    padding: float,
    saturation: float,
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi, aspect, rot = camera_bounds(native_shape_xyz, view, padding)
    x = torch.linspace(float(lo[0]), float(hi[0]), width)
    y = torch.linspace(float(hi[1]), float(lo[1]), height)
    z = torch.linspace(float(hi[2]), float(lo[2]), samples)  # front-to-back
    depth_norm = torch.linspace(0.0, 1.0, samples)
    rot_t = torch.from_numpy(rot.astype(np.float32))
    aspect_t = torch.from_numpy(aspect.astype(np.float32))

    rgb_out = torch.empty((height, width, 3), dtype=torch.float32)
    opacity_out = torch.empty((height, width), dtype=torch.float32)
    depth_out = torch.empty((height, width), dtype=torch.float32)

    for y0 in range(0, height, tile_rows):
        y1 = min(y0 + tile_rows, height)
        yy, xx, zz = torch.meshgrid(y[y0:y1], x, z, indexing="ij")
        cam = torch.stack([xx, yy, zz], dim=-1)
        obj = cam @ rot_t
        norm = obj / (aspect_t / 2.0).clamp_min(1e-6)
        grid = norm.permute(2, 0, 1, 3).unsqueeze(0).contiguous()
        with torch.no_grad():
            sampled = F.grid_sample(
                volume,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(0)
        color, opacity, depth = composite_tile(sampled, depth_norm, alpha_threshold, opacity_scale)
        rgb_out[y0:y1] = color + (1.0 - opacity[..., None])
        opacity_out[y0:y1] = opacity
        depth_out[y0:y1] = depth

    rgb_img = boost_saturation(rgb_out.clamp(0.0, 1.0).numpy(), saturation)
    opacity_np = opacity_out.clamp(0.0, 1.0).numpy()
    depth_np = depth_out.clamp(0.0, 1.0).numpy()
    alpha_strength = np.clip(opacity_np**0.75, 0.0, 1.0)
    front_shade = 1.0 - depth_np
    gray_near = np.array([0.18, 0.18, 0.18], dtype=np.float32)
    gray_far = np.array([0.70, 0.70, 0.70], dtype=np.float32)
    structure_color = gray_far[None, None, :] * (1.0 - front_shade[..., None]) + gray_near[None, None, :] * front_shade[..., None]
    alpha_img = (1.0 - alpha_strength[..., None]) + alpha_strength[..., None] * structure_color
    return rgb_img, np.clip(alpha_img, 0.0, 1.0)


def save_image(img: np.ndarray, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="nearest")
    ax.axis("off")
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def make_pair(alpha_path: Path, rgb_path: Path, out_path: Path, dpi: int, labeled: bool) -> None:
    imgs = [plt.imread(alpha_path), plt.imread(rgb_path)]
    labels = ["alpha / structure view", "opacity-composited RGB view"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), dpi=dpi)
    for ax, img, label in zip(axes, imgs, labels):
        ax.imshow(img)
        ax.axis("off")
        if labeled:
            ax.text(0.03, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=10, color="#1f2933")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def pair_slug(scene: str) -> str:
    if scene.startswith("3dfront_"):
        scene = scene.removeprefix("3dfront_")
    if scene.endswith("_00"):
        scene = scene.removesuffix("_00")
    return scene


def make_contact_sheet(rows: list[tuple[str, Path, Path]], out_path: Path, dpi: int) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(7.2, max(2.4 * n, 2.4)), dpi=dpi)
    if n == 1:
        axes = np.array([axes])
    for row, (label, alpha_path, rgb_path) in enumerate(rows):
        for col, path in enumerate([alpha_path, rgb_path]):
            axes[row, col].imshow(plt.imread(path))
            axes[row, col].axis("off")
            axes[row, col].set_title(f"{label} / {'alpha' if col == 0 else 'RGB composite'}", fontsize=8)
    fig.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def choose_feature_paths(args: argparse.Namespace) -> list[Path]:
    feature_paths = sorted(args.features_dir.glob("*.npz"))
    if args.scene:
        wanted = set(args.scene)
        selected = [p for p in feature_paths if p.stem in wanted]
        missing = wanted - {p.stem for p in selected}
        if missing:
            raise FileNotFoundError(f"missing scenes: {sorted(missing)}")
        return selected
    rows = [
        scene_score(p, args.boxes_dir / f"{p.stem}.npy")
        for p in feature_paths[: args.scan_limit]
    ]
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "candidate_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return [Path(str(r["feature_path"])) for r in rows[: args.top_k]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, default=Path("dataset/finetune/front3d_rpn_data/features"))
    parser.add_argument("--boxes-dir", type=Path, default=Path("dataset/finetune/front3d_rpn_data/obb"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures_src/fig1_front3d_volume"))
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--scan-limit", type=int, default=160)
    parser.add_argument("--width", type=int, default=1120)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--samples", type=int, default=224)
    parser.add_argument("--tile-rows", type=int, default=36)
    parser.add_argument("--alpha-threshold", type=float, default=0.05)
    parser.add_argument("--opacity-scale", type=float, default=0.11)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--azimuth", type=float, default=DEFAULT_VIEW.azimuth)
    parser.add_argument("--elevation", type=float, default=DEFAULT_VIEW.elevation)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--gamma", type=float, default=1.25)
    parser.add_argument("--saturation", type=float, default=2.2)
    parser.add_argument("--no-auto-exposure", action="store_true")
    parser.add_argument("--render-view-candidates", action="store_true")
    args = parser.parse_args()

    rendered: list[tuple[str, Path, Path]] = []
    for feature_path in choose_feature_paths(args):
        volume, native_shape = load_rgba_volume(feature_path, args.gamma, not args.no_auto_exposure)
        main_view = ViewSpec("main", args.azimuth, args.elevation)
        rgb, alpha = render_volume(
            volume,
            native_shape,
            main_view,
            args.width,
            args.height,
            args.samples,
            args.tile_rows,
            args.alpha_threshold,
            args.opacity_scale,
            args.padding,
            args.saturation,
        )
        alpha_path = args.out_dir / f"{feature_path.stem}_alpha.png"
        rgb_path = args.out_dir / f"{feature_path.stem}_rgb.png"
        save_image(alpha, alpha_path, args.dpi)
        save_image(rgb, rgb_path, args.dpi)
        rendered.append((feature_path.stem, alpha_path, rgb_path))
        print(f"rendered {feature_path.stem}: {alpha_path} {rgb_path}")

        if args.render_view_candidates:
            view_rows: list[tuple[str, Path, Path]] = []
            view_dir = args.out_dir / "view_candidates"
            for view in VIEW_CANDIDATES:
                rgb_v, alpha_v = render_volume(
                    volume,
                    native_shape,
                    view,
                    args.width,
                    args.height,
                    args.samples,
                    args.tile_rows,
                    args.alpha_threshold,
                    args.opacity_scale,
                    args.padding,
                    args.saturation,
                )
                alpha_v_path = view_dir / f"{feature_path.stem}_{view.name}_alpha.png"
                rgb_v_path = view_dir / f"{feature_path.stem}_{view.name}_rgb.png"
                save_image(alpha_v, alpha_v_path, args.dpi)
                save_image(rgb_v, rgb_v_path, args.dpi)
                view_rows.append((view.name, alpha_v_path, rgb_v_path))
            make_contact_sheet(view_rows, args.out_dir / f"{feature_path.stem}_view_contact_sheet.png", args.dpi)

    make_contact_sheet(rendered, args.out_dir / "contact_sheet.png", args.dpi)
    if len(rendered) == 1:
        scene, alpha_path, rgb_path = rendered[0]
        slug = pair_slug(scene)
        make_pair(alpha_path, rgb_path, args.out_dir / f"fig1_front3d_{slug}_alpha_rgb_pair_clean.png", args.dpi, labeled=False)
        make_pair(alpha_path, rgb_path, args.out_dir / f"fig1_front3d_{slug}_alpha_rgb_pair.png", args.dpi, labeled=True)
    print(f"contact sheet: {args.out_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
