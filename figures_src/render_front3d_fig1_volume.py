#!/usr/bin/env python3
"""Render real Front3D NeRF-MAE volumes for Fig. 1 artwork.

The output is intentionally data-derived but presentation-oriented: it creates
orthographic point/surface renders of the released Front3D `rgbsigma` grids,
including an alpha-only structural view and an RGB appearance view.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def density_to_alpha(density: np.ndarray) -> np.ndarray:
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
        proj_area = float((mask.max(axis=2)).mean())
        compact = float(np.prod(ext / np.array(mask.shape)))
    else:
        occ = alpha_mass = color_std = color_range = proj_area = compact = 0.0
    nboxes = 0
    if boxes_path is not None and boxes_path.exists():
        nboxes = int(np.load(boxes_path, allow_pickle=True).shape[0])
    # Prefer moderately rich rooms: enough objects and silhouette, but not a
    # completely filled volume. The weights are for visual selection only.
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


def load_points(
    feature_path: Path,
    alpha_threshold: float,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(feature_path) as z:
        rgbsigma = z["rgbsigma"].astype(np.float32)
    alpha = density_to_alpha(rgbsigma[..., 3])
    mask = alpha > alpha_threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError(f"No occupied voxels in {feature_path}")
    weights = alpha[mask]
    if coords.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        # Weighted sample keeps strong surfaces while preserving room context.
        p = weights / weights.sum()
        idx = rng.choice(coords.shape[0], size=max_points, replace=False, p=p)
        coords = coords[idx]
        weights = weights[idx]
    shape = np.array(alpha.shape, dtype=np.float32)
    points = (coords.astype(np.float32) / np.maximum(shape - 1.0, 1.0)) - 0.5
    # Keep the native aspect ratio instead of forcing all rooms to a cube.
    points *= shape / float(shape.max())
    rgb = np.clip(rgbsigma[..., :3], 0.0, 1.0)[tuple(coords.T)]
    return points, rgb, weights


def project_points(points: np.ndarray, azimuth_deg: float, elevation_deg: float) -> tuple[np.ndarray, np.ndarray]:
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
    p = points @ rz.T @ rx.T
    xy = p[:, :2]
    depth = p[:, 2]
    return xy, depth


def render_panel(
    points: np.ndarray,
    rgb: np.ndarray,
    alpha: np.ndarray,
    mode: str,
    out_path: Path,
    title: str | None,
    azimuth: float,
    elevation: float,
    dpi: int,
    marker_size: float,
    dark: bool,
) -> None:
    xy, depth = project_points(points, azimuth, elevation)
    order = np.argsort(depth)
    xy = xy[order]
    rgb = rgb[order]
    alpha = alpha[order]
    alpha_vis = np.clip(0.05 + 0.75 * np.sqrt(alpha), 0.05, 0.85)
    if mode == "alpha":
        strength = np.clip((alpha - np.quantile(alpha, 0.05)) / (np.quantile(alpha, 0.98) - np.quantile(alpha, 0.05) + 1e-6), 0, 1)
        colors = np.stack(
            [
                0.10 + 0.55 * strength,
                0.35 + 0.45 * strength,
                0.65 + 0.25 * strength,
                alpha_vis,
            ],
            axis=1,
        )
    elif mode == "rgb":
        gamma_rgb = np.clip(rgb, 0.0, 1.0) ** (1.0 / 1.25)
        colors = np.concatenate([gamma_rgb, alpha_vis[:, None]], axis=1)
    else:
        raise ValueError(f"unknown mode: {mode}")

    fig = plt.figure(figsize=(4.2, 4.2), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    if dark:
        fig.patch.set_facecolor("#101418")
        ax.set_facecolor("#101418")
    else:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=marker_size, linewidths=0, rasterized=True)
    pad = 0.06
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    center = (lo + hi) / 2.0
    span = max(float((hi - lo).max()), 1e-6) * (1.0 + pad)
    ax.set_xlim(center[0] - span / 2.0, center[0] + span / 2.0)
    ax.set_ylim(center[1] - span / 2.0, center[1] + span / 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.text(
            0.03,
            0.96,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="white" if dark else "#1f2933",
            family="DejaVu Sans",
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_contact_sheet(image_paths: list[tuple[str, Path, Path]], out_path: Path, dpi: int) -> None:
    n = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(7.0, max(2.2 * n, 2.2)), dpi=dpi)
    if n == 1:
        axes = np.array([axes])
    for row, (scene, alpha_path, rgb_path) in enumerate(image_paths):
        for col, path in enumerate([alpha_path, rgb_path]):
            img = plt.imread(path)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            axes[row, col].set_title(f"{scene} / {'alpha' if col == 0 else 'rgb'}", fontsize=8)
    fig.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, default=Path("dataset/finetune/front3d_rpn_data/features"))
    parser.add_argument("--boxes-dir", type=Path, default=Path("dataset/finetune/front3d_rpn_data/obb"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures_src/fig1_front3d_volume"))
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--scan-limit", type=int, default=160)
    parser.add_argument("--alpha-threshold", type=float, default=0.06)
    parser.add_argument("--max-points", type=int, default=180_000)
    parser.add_argument("--azimuth", type=float, default=-42.0)
    parser.add_argument("--elevation", type=float, default=57.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--marker-size", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dark", action="store_true")
    args = parser.parse_args()

    feature_paths = sorted(args.features_dir.glob("*.npz"))
    if args.scene:
        wanted = set(args.scene)
        feature_paths = [p for p in feature_paths if p.stem in wanted]
        missing = wanted - {p.stem for p in feature_paths}
        if missing:
            raise FileNotFoundError(f"missing scenes: {sorted(missing)}")
    else:
        feature_paths = feature_paths[: args.scan_limit]
        rows = [
            scene_score(p, args.boxes_dir / f"{p.stem}.npy")
            for p in feature_paths
        ]
        rows.sort(key=lambda r: float(r["score"]), reverse=True)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "candidate_scores.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        feature_paths = [Path(str(r["feature_path"])) for r in rows[: args.top_k]]

    rendered: list[tuple[str, Path, Path]] = []
    for p in feature_paths:
        points, rgb, alpha = load_points(p, args.alpha_threshold, args.max_points, args.seed)
        alpha_path = args.out_dir / f"{p.stem}_alpha.png"
        rgb_path = args.out_dir / f"{p.stem}_rgb.png"
        render_panel(
            points,
            rgb,
            alpha,
            "alpha",
            alpha_path,
            None,
            args.azimuth,
            args.elevation,
            args.dpi,
            args.marker_size,
            args.dark,
        )
        render_panel(
            points,
            rgb,
            alpha,
            "rgb",
            rgb_path,
            None,
            args.azimuth,
            args.elevation,
            args.dpi,
            args.marker_size,
            args.dark,
        )
        rendered.append((p.stem, alpha_path, rgb_path))
        print(f"rendered {p.stem}: {alpha_path} {rgb_path}")

    make_contact_sheet(rendered, args.out_dir / "contact_sheet.png", args.dpi)
    print(f"contact sheet: {args.out_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
