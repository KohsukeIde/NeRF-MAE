#!/usr/bin/env python3
"""Compute alpha sparsity and entropy statistics for released rgbsigma grids."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


THRESHOLDS = (0.001, 0.01, 0.05, 0.1, 0.5)


def density_to_alpha(density: np.ndarray) -> np.ndarray:
    safe_density = np.clip(density.astype(np.float32), -50.0, 20.0)
    alpha = 1.0 - np.exp(-np.exp(safe_density) / 100.0)
    return np.clip(alpha, 0.0, 1.0)


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def histogram_entropy(values: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    total = int(hist.sum())
    if total == 0:
        return 0.0
    probs = hist.astype(np.float64) / float(total)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def summarize_scene(path: Path, include_rgb: bool = False) -> dict[str, float | int | str]:
    data = np.load(path)
    rgbsigma = data["rgbsigma"]
    alpha = density_to_alpha(rgbsigma[..., 3])
    flat = alpha.reshape(-1)
    row: dict[str, float | int | str] = {
        "scene": path.stem,
        "path": str(path),
        "n_voxels": int(flat.size),
        "alpha_mean": float(flat.mean()),
        "alpha_std": float(flat.std()),
        "alpha_p50": float(np.quantile(flat, 0.50)),
        "alpha_p90": float(np.quantile(flat, 0.90)),
        "alpha_p95": float(np.quantile(flat, 0.95)),
        "alpha_p99": float(np.quantile(flat, 0.99)),
        "alpha_hist_entropy_64": histogram_entropy(flat, bins=64),
    }
    for threshold in THRESHOLDS:
        key = f"{threshold:g}".replace(".", "p")
        frac = float((flat > threshold).mean())
        row[f"occ_frac_gt_{key}"] = frac
        row[f"occ_entropy_gt_{key}"] = binary_entropy(frac)

    if include_rgb:
        occ = alpha > 0.01
        rgb = np.clip(rgbsigma[..., :3], 0.0, 1.0)
        if bool(occ.any()):
            occ_rgb = rgb[occ]
            row["occupied_rgb_mean"] = float(occ_rgb.mean())
            row["occupied_rgb_std"] = float(occ_rgb.std())
        else:
            row["occupied_rgb_mean"] = float("nan")
            row["occupied_rgb_std"] = float("nan")
    return row


def aggregate(rows: list[dict[str, float | int | str]]) -> list[dict[str, str | float]]:
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key not in {"scene", "path"} and isinstance(value, (float, int))
    ]
    summary = []
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        summary.append(
            {
                "metric": key,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, scene_rows: list[dict], summary_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [
        "alpha_mean",
        "alpha_p95",
        "alpha_p99",
        "occ_frac_gt_0p01",
        "occ_entropy_gt_0p01",
        "alpha_hist_entropy_64",
    ]
    by_metric = {row["metric"]: row for row in summary_rows}
    lines = [
        "# Alpha Sparsity / Entropy Stats",
        "",
        f"Scenes: {len(scene_rows)}",
        "",
        "Density-to-alpha conversion: `alpha = clip(1 - exp(-exp(density) / 100), 0, 1)`.",
        "",
        "## Summary",
        "",
        "| metric | mean | std | median | min | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in selected:
        row = by_metric.get(metric)
        if row is None:
            continue
        lines.append(
            "| {metric} | {mean:.6g} | {std:.6g} | {median:.6g} | {min:.6g} | {max:.6g} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- `occ_frac_gt_0p01` estimates how much of the grid carries non-empty alpha evidence.",
            "- Low occupancy fraction and low binary occupancy entropy support treating alpha as a sparse structural signal.",
            "- `occupied_rgb_std` is reported only over occupied voxels and is a descriptive appearance statistic, not a method claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("dataset/finetune/front3d_rpn_data/features"),
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means all scenes.")
    parser.add_argument("--include-rgb", action="store_true")
    parser.add_argument("--scene-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.features_dir.glob("*.npz"))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"no .npz files found under {args.features_dir}")

    rows = [summarize_scene(path, include_rgb=args.include_rgb) for path in paths]
    summary = aggregate(rows)
    write_csv(args.scene_csv, rows)
    write_csv(args.summary_csv, summary)
    write_markdown(args.summary_md, rows, summary)
    print(f"wrote {args.scene_csv}")
    print(f"wrote {args.summary_csv}")
    print(f"wrote {args.summary_md}")


if __name__ == "__main__":
    main()
