#!/usr/bin/env python3
"""Audit alpha-derived boundary/SDF targets before launching geometry pretraining.

This script is intentionally dataset-agnostic for NeRF-RPN style feature
archives. It reads `rgbsigma[..., -1]`, converts density to alpha with the same
normalization used by the FCOS loaders, then reports threshold-dependent
occupancy, shell, connected components, distance-to-boundary statistics, and
slice visualizations.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def density_to_alpha(density: np.ndarray) -> np.ndarray:
    density = density.astype(np.float32, copy=False)
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


def load_alpha(path: Path, normalize_density: bool) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        rgbsigma = data["rgbsigma"].astype(np.float32, copy=False)
    alpha_or_density = rgbsigma[..., -1]
    if normalize_density:
        return density_to_alpha(alpha_or_density)
    return np.clip(alpha_or_density, 0.0, 1.0)


def select_scenes(features_dir: Path, split_path: Path | None, split_key: str, max_scenes: int) -> list[str]:
    if split_path is not None and split_path.exists():
        with np.load(split_path, allow_pickle=True) as split:
            if split_key not in split:
                raise KeyError(f"{split_key!r} not found in {split_path}; keys={list(split.keys())}")
            scenes = [str(x) for x in split[split_key]]
    else:
        scenes = sorted(p.stem for p in features_dir.glob("*.npz"))
    return scenes[:max_scenes]


def choose_slice(mask: np.ndarray, axis: int) -> int:
    reduce_axes = tuple(i for i in range(mask.ndim) if i != axis)
    counts = mask.sum(axis=reduce_axes)
    if counts.max() <= 0:
        return mask.shape[axis] // 2
    return int(np.argmax(counts))


def take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(volume, index, axis=axis)


def robust_imshow(ax, image: np.ndarray, title: str, cmap: str = "viridis", vmin=None, vmax=None):
    ax.imshow(np.asarray(image).T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def summarize_distance(distance: np.ndarray, truncate: float) -> dict[str, float]:
    clipped = np.minimum(distance.astype(np.float32, copy=False), truncate)
    return {
        "dist_mean": float(clipped.mean()),
        "dist_p50": float(np.percentile(clipped, 50)),
        "dist_p90": float(np.percentile(clipped, 90)),
        "dist_p99": float(np.percentile(clipped, 99)),
    }


def component_count(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    structure = np.ones((3, 3, 3), dtype=bool)
    _, count = ndimage.label(mask, structure=structure)
    return int(count)


def audit_one(
    scene: str,
    alpha: np.ndarray,
    thresholds: list[float],
    output_dir: Path,
    slice_axis: int,
    distance_truncate: float,
    render: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    grad = np.gradient(alpha.astype(np.float32, copy=False))
    grad_mag = np.sqrt(sum(g * g for g in grad))
    alpha_max = float(alpha.max())
    alpha_mean = float(alpha.mean())

    for threshold in thresholds:
        occupied = alpha > threshold
        occupied_count = int(occupied.sum())
        total_count = int(occupied.size)
        if occupied_count > 0:
            eroded = ndimage.binary_erosion(occupied, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
            shell = occupied & ~eroded
        else:
            shell = np.zeros_like(occupied, dtype=bool)

        shell_count = int(shell.sum())
        # Distance to alpha-derived shell. If shell is empty, mark with NaNs.
        if shell_count > 0:
            distance = ndimage.distance_transform_edt(~shell).astype(np.float32)
            dist_stats = summarize_distance(distance, distance_truncate)
        else:
            distance = np.full(alpha.shape, np.nan, dtype=np.float32)
            dist_stats = {"dist_mean": math.nan, "dist_p50": math.nan, "dist_p90": math.nan, "dist_p99": math.nan}

        index = choose_slice(occupied, slice_axis)
        row: dict[str, object] = {
            "scene": scene,
            "threshold": threshold,
            "shape": "x".join(str(x) for x in alpha.shape),
            "slice_axis": slice_axis,
            "slice_index": index,
            "alpha_max": alpha_max,
            "alpha_mean": alpha_mean,
            "occupied_voxels": occupied_count,
            "occupied_ratio": occupied_count / total_count,
            "component_count": component_count(occupied),
            "shell_voxels": shell_count,
            "shell_ratio_total": shell_count / total_count,
            "shell_ratio_occupied": shell_count / occupied_count if occupied_count else math.nan,
            "grad_mag_mean": float(grad_mag.mean()),
            "grad_mag_p95": float(np.percentile(grad_mag, 95)),
            **dist_stats,
        }
        rows.append(row)

        if render:
            fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), constrained_layout=True)
            alpha_slice = take_slice(alpha, slice_axis, index)
            occupied_slice = take_slice(occupied.astype(np.float32), slice_axis, index)
            shell_slice = take_slice(shell.astype(np.float32), slice_axis, index)
            distance_slice = take_slice(np.minimum(distance, distance_truncate), slice_axis, index)
            grad_slice = take_slice(grad_mag, slice_axis, index)
            robust_imshow(axes[0], alpha_slice, f"{scene}\nalpha", cmap="magma", vmin=0.0, vmax=max(0.1, alpha_max))
            robust_imshow(axes[1], occupied_slice, f"alpha>{threshold:g}", cmap="gray", vmin=0.0, vmax=1.0)
            robust_imshow(axes[2], shell_slice, "shell", cmap="gray", vmin=0.0, vmax=1.0)
            robust_imshow(axes[3], distance_slice, f"dist to shell\nclip {distance_truncate:g}", cmap="viridis", vmin=0.0, vmax=distance_truncate)
            robust_imshow(axes[4], grad_slice, "alpha grad |.|", cmap="plasma")
            fig.suptitle(
                f"thr={threshold:g} occ={row['occupied_ratio']:.4f} shell/occ={row['shell_ratio_occupied']:.3f} comps={row['component_count']}",
                fontsize=10,
            )
            fig.savefig(output_dir / f"{scene}_thr{str(threshold).replace('.', 'p')}.png", dpi=140)
            plt.close(fig)

    return rows


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    thresholds = sorted({float(r["threshold"]) for r in rows})
    lines = ["# Alpha Boundary Target Audit", ""]
    lines.append(f"Rows: {len(rows)}")
    lines.append("")
    lines.append("| threshold | scenes | occ ratio mean | shell/occ mean | components median | dist p90 mean | grad p95 mean |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for threshold in thresholds:
        subset = [r for r in rows if float(r["threshold"]) == threshold]
        def mean(key: str) -> float:
            vals = np.array([float(r[key]) for r in subset], dtype=np.float64)
            return float(np.nanmean(vals))
        comps = np.array([float(r["component_count"]) for r in subset], dtype=np.float64)
        lines.append(
            f"| {threshold:g} | {len(subset)} | {mean('occupied_ratio'):.6f} | "
            f"{mean('shell_ratio_occupied'):.4f} | {np.nanmedian(comps):.1f} | "
            f"{mean('dist_p90'):.3f} | {mean('grad_mag_p95'):.6f} |"
        )
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("- Very high component counts indicate noisy alpha topology.")
    lines.append("- Very high shell/occupied ratio means occupancy is thin/sparse; SDF may mostly encode shell proximity.")
    lines.append("- If distance maps are smooth around coherent surfaces in the PNGs, SDF targets are plausible.")
    lines.append("- If alpha slices are fragmented or threshold-sensitive, prefer shell/normal diagnostics before SDF pretraining.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, default=Path("dataset/finetune/front3d_rpn_data/features"))
    parser.add_argument("--split", type=Path, default=Path("dataset/finetune/front3d_rpn_data/3dfront_split.npz"))
    parser.add_argument("--split-key", default="train_scenes")
    parser.add_argument("--output-dir", type=Path, default=Path("results/shortcut_probe_artifacts/alpha_boundary_audit"))
    parser.add_argument("--thresholds", default="0.01,0.05,0.1")
    parser.add_argument("--max-scenes", type=int, default=20)
    parser.add_argument("--slice-axis", type=int, default=2)
    parser.add_argument("--distance-truncate", type=float, default=16.0)
    parser.add_argument("--no-normalize-density", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    scenes = select_scenes(args.features_dir, args.split, args.split_key, args.max_scenes)
    if not scenes:
        raise SystemExit(f"no scenes found under {args.features_dir}")

    all_rows: list[dict[str, object]] = []
    for i, scene in enumerate(scenes, start=1):
        feature_path = args.features_dir / f"{scene}.npz"
        if not feature_path.exists():
            print(f"[warn] missing feature for split scene {scene}: {feature_path}")
            continue
        print(f"[{i}/{len(scenes)}] {scene}")
        alpha = load_alpha(feature_path, normalize_density=not args.no_normalize_density)
        rows = audit_one(
            scene=scene,
            alpha=alpha,
            thresholds=thresholds,
            output_dir=output_dir,
            slice_axis=args.slice_axis,
            distance_truncate=args.distance_truncate,
            render=not args.no_render,
        )
        all_rows.extend(rows)

    csv_path = output_dir / "alpha_boundary_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    write_summary(all_rows, output_dir / "README.md")
    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {output_dir / 'README.md'}")
    print(f"[ok] png count={len(list(output_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
