#!/usr/bin/env python3
"""Denoised alpha-boundary/SDF target-quality audit.

The first audit showed that raw thresholded alpha produces fragmented
occupancy and shell-heavy targets. This v2 audit tests whether simple,
paper-defensible denoising steps make alpha-derived boundary/SDF targets
usable before spending GPU time on Boundary-SDF MAE pretraining.
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
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


@dataclass(frozen=True)
class Variant:
    name: str
    threshold: float
    smooth_sigma: float = 0.0
    close_iters: int = 0
    open_iters: int = 0
    min_component_voxels: int = 0


def default_variants() -> list[Variant]:
    return [
        Variant("raw_thr001", threshold=0.01),
        Variant("smooth075_thr001", threshold=0.01, smooth_sigma=0.75),
        Variant("smooth100_thr001", threshold=0.01, smooth_sigma=1.0),
        Variant("smooth100_thr002", threshold=0.02, smooth_sigma=1.0),
        Variant("smooth100_thr001_min64", threshold=0.01, smooth_sigma=1.0, min_component_voxels=64),
        Variant("smooth100_thr001_min256", threshold=0.01, smooth_sigma=1.0, min_component_voxels=256),
        Variant("smooth100_thr001_close1_min64", threshold=0.01, smooth_sigma=1.0, close_iters=1, min_component_voxels=64),
    ]


def parse_variants(text: str | None) -> list[Variant]:
    if not text:
        return default_variants()
    variants: list[Variant] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        # Format: name:thr:sigma:mincc:close:open
        parts = item.split(":")
        if len(parts) < 3:
            raise ValueError(f"variant {item!r} must be name:threshold:sigma[:mincc[:close[:open]]]")
        name = parts[0]
        threshold = float(parts[1])
        sigma = float(parts[2])
        mincc = int(parts[3]) if len(parts) > 3 and parts[3] else 0
        close = int(parts[4]) if len(parts) > 4 and parts[4] else 0
        open_ = int(parts[5]) if len(parts) > 5 and parts[5] else 0
        variants.append(Variant(name, threshold, sigma, close, open_, mincc))
    return variants


def component_stats(mask: np.ndarray, min_component_voxels: int) -> tuple[np.ndarray, dict[str, float]]:
    if not mask.any():
        empty_stats = {
            "component_count": 0,
            "largest_component_fraction": math.nan,
            "top5_component_fraction": math.nan,
            "small_component_fraction": math.nan,
            "removed_component_voxels": 0,
        }
        return mask, empty_stats

    structure = np.ones((3, 3, 3), dtype=bool)
    labels, count = ndimage.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())
    comp_sizes = sizes[1:]
    occupied_count = int(mask.sum())
    largest = int(comp_sizes.max()) if comp_sizes.size else 0
    top5 = int(np.sort(comp_sizes)[-5:].sum()) if comp_sizes.size else 0
    small_voxels = int(comp_sizes[comp_sizes < max(1, min_component_voxels)].sum()) if comp_sizes.size else 0

    filtered = mask
    removed_voxels = 0
    if min_component_voxels > 0:
        keep_labels = np.flatnonzero(sizes >= min_component_voxels)
        keep_labels = keep_labels[keep_labels != 0]
        filtered = np.isin(labels, keep_labels)
        removed_voxels = int(occupied_count - filtered.sum())

    stats = {
        "component_count": int(count),
        "largest_component_fraction": largest / occupied_count if occupied_count else math.nan,
        "top5_component_fraction": top5 / occupied_count if occupied_count else math.nan,
        "small_component_fraction": small_voxels / occupied_count if occupied_count else math.nan,
        "removed_component_voxels": removed_voxels,
    }
    return filtered, stats


def make_mask(alpha: np.ndarray, variant: Variant) -> tuple[np.ndarray, np.ndarray]:
    source = alpha
    if variant.smooth_sigma > 0:
        source = ndimage.gaussian_filter(alpha, sigma=variant.smooth_sigma).astype(np.float32, copy=False)
    mask = source > variant.threshold
    structure = np.ones((3, 3, 3), dtype=bool)
    if variant.close_iters > 0:
        mask = ndimage.binary_closing(mask, structure=structure, iterations=variant.close_iters)
    if variant.open_iters > 0:
        mask = ndimage.binary_opening(mask, structure=structure, iterations=variant.open_iters)
    return source, mask


def signed_distance(mask: np.ndarray, truncate: float) -> np.ndarray:
    if not mask.any():
        return np.full(mask.shape, np.nan, dtype=np.float32)
    if mask.all():
        return np.full(mask.shape, np.nan, dtype=np.float32)
    outside = ndimage.distance_transform_edt(~mask).astype(np.float32)
    inside = ndimage.distance_transform_edt(mask).astype(np.float32)
    sdf = outside - inside
    return np.clip(sdf, -truncate, truncate)


def shell_from_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return mask & ~eroded


def safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return math.nan
    return float(np.percentile(values, q))


def summarize_mask(
    scene: str,
    alpha: np.ndarray,
    raw_reference: np.ndarray,
    variant: Variant,
    output_mask: np.ndarray,
    source_alpha: np.ndarray,
    comp_stats: dict[str, float],
    truncate: float,
) -> dict[str, object]:
    total = int(output_mask.size)
    occ = int(output_mask.sum())
    shell = shell_from_mask(output_mask)
    shell_count = int(shell.sum())
    sdf = signed_distance(output_mask, truncate=truncate)
    inside_abs = np.abs(sdf[output_mask]) if occ else np.array([], dtype=np.float32)
    outside_near = np.abs(sdf[(~output_mask) & np.isfinite(sdf) & (np.abs(sdf) <= truncate)])

    intersection = int((output_mask & raw_reference).sum())
    union = int((output_mask | raw_reference).sum())
    raw_count = int(raw_reference.sum())

    grad = np.gradient(source_alpha.astype(np.float32, copy=False))
    grad_mag = np.sqrt(sum(g * g for g in grad))
    shell_grad = grad_mag[shell]

    return {
        "scene": scene,
        "variant": variant.name,
        "threshold": variant.threshold,
        "smooth_sigma": variant.smooth_sigma,
        "close_iters": variant.close_iters,
        "open_iters": variant.open_iters,
        "min_component_voxels": variant.min_component_voxels,
        "shape": "x".join(str(x) for x in alpha.shape),
        "alpha_max": float(alpha.max()),
        "alpha_mean": float(alpha.mean()),
        "source_alpha_mean": float(source_alpha.mean()),
        "source_alpha_p95": float(np.percentile(source_alpha, 95)),
        "occupied_voxels": occ,
        "occupied_ratio": occ / total,
        "raw_reference_recall": intersection / raw_count if raw_count else math.nan,
        "raw_reference_precision": intersection / occ if occ else math.nan,
        "raw_reference_iou": intersection / union if union else math.nan,
        "shell_voxels": shell_count,
        "shell_ratio_total": shell_count / total,
        "shell_ratio_occupied": shell_count / occ if occ else math.nan,
        "interior_ratio_occupied": 1.0 - (shell_count / occ) if occ else math.nan,
        "sdf_inside_abs_p50": safe_percentile(inside_abs, 50),
        "sdf_inside_abs_p90": safe_percentile(inside_abs, 90),
        "sdf_outside_near_abs_p50": safe_percentile(outside_near, 50),
        "sdf_outside_near_abs_p90": safe_percentile(outside_near, 90),
        "shell_grad_mag_mean": float(shell_grad.mean()) if shell_grad.size else math.nan,
        "shell_grad_mag_p95": safe_percentile(shell_grad, 95),
        **comp_stats,
    }


def audit_one(scene: str, alpha: np.ndarray, variants: list[Variant], truncate: float) -> tuple[list[dict[str, object]], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    raw_source, raw_mask = make_mask(alpha, Variant("raw_ref_thr001", threshold=0.01))
    _ = raw_source
    rows: list[dict[str, object]] = []
    visual_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for variant in variants:
        source_alpha, mask = make_mask(alpha, variant)
        mask, comp = component_stats(mask, variant.min_component_voxels)
        row = summarize_mask(
            scene=scene,
            alpha=alpha,
            raw_reference=raw_mask,
            variant=variant,
            output_mask=mask,
            source_alpha=source_alpha,
            comp_stats=comp,
            truncate=truncate,
        )
        rows.append(row)
        visual_data[variant.name] = (source_alpha, mask, signed_distance(mask, truncate))
    return rows, visual_data


def render_scene(
    scene: str,
    alpha: np.ndarray,
    variants: list[Variant],
    visual_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
    slice_axis: int,
    truncate: float,
) -> None:
    _, raw_mask = make_mask(alpha, Variant("raw_ref_thr001", threshold=0.01))
    index = choose_slice(raw_mask, slice_axis)
    nrows = len(variants)
    fig, axes = plt.subplots(nrows, 4, figsize=(12, max(2.2 * nrows, 4)), constrained_layout=True)
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, variant in enumerate(variants):
        source_alpha, mask, sdf = visual_data[variant.name]
        shell = shell_from_mask(mask)
        robust_imshow(
            axes[r, 0],
            take_slice(source_alpha, slice_axis, index),
            f"{variant.name}\nsource alpha",
            cmap="magma",
            vmin=0.0,
            vmax=max(0.1, float(np.percentile(source_alpha, 99))),
        )
        robust_imshow(
            axes[r, 1],
            take_slice(mask.astype(np.float32), slice_axis, index),
            "occupancy",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        robust_imshow(
            axes[r, 2],
            take_slice(shell.astype(np.float32), slice_axis, index),
            "shell",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        robust_imshow(
            axes[r, 3],
            take_slice(sdf, slice_axis, index),
            f"signed dist\nclip +/-{truncate:g}",
            cmap="coolwarm",
            vmin=-truncate,
            vmax=truncate,
        )

    fig.suptitle(f"{scene} alpha-boundary denoising audit v2, axis={slice_axis}, slice={index}", fontsize=11)
    fig.savefig(output_dir / f"{scene}_boundary_audit_v2.png", dpi=140)
    plt.close(fig)


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    variants = sorted({str(r["variant"]) for r in rows})
    lines = ["# Alpha Boundary Target Audit v2", ""]
    lines.append(f"Rows: {len(rows)}")
    lines.append("")
    lines.append(
        "| variant | scenes | occ ratio mean | shell/occ mean | components median | "
        "largest comp mean | top5 comp mean | small comp mean | raw IoU mean | raw recall mean | sdf inside p90 mean |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for variant in variants:
        subset = [r for r in rows if str(r["variant"]) == variant]

        def mean(key: str) -> float:
            vals = np.array([float(r[key]) for r in subset], dtype=np.float64)
            return float(np.nanmean(vals))

        comps = np.array([float(r["component_count"]) for r in subset], dtype=np.float64)
        lines.append(
            f"| {variant} | {len(subset)} | {mean('occupied_ratio'):.6f} | "
            f"{mean('shell_ratio_occupied'):.4f} | {np.nanmedian(comps):.1f} | "
            f"{mean('largest_component_fraction'):.4f} | {mean('top5_component_fraction'):.4f} | "
            f"{mean('small_component_fraction'):.4f} | {mean('raw_reference_iou'):.4f} | "
            f"{mean('raw_reference_recall'):.4f} | {mean('sdf_inside_abs_p90'):.3f} |"
        )
    lines.append("")
    lines.append("Decision guide:")
    lines.append("- A usable Boundary-SDF target should reduce fragmentation without deleting most raw low-threshold support.")
    lines.append("- Prefer variants with lower component counts, lower shell/occupied ratio, and reasonable raw-reference recall.")
    lines.append("- If only aggressive filtering looks clean but raw recall collapses, the target is likely too biased for a main method.")
    lines.append("- Signed-distance maps should show coherent bands around surfaces in the PNG comparisons.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, default=Path("dataset/finetune/front3d_rpn_data/features"))
    parser.add_argument("--split", type=Path, default=Path("dataset/finetune/front3d_rpn_data/3dfront_split.npz"))
    parser.add_argument("--split-key", default="train_scenes")
    parser.add_argument("--output-dir", type=Path, default=Path("results/shortcut_probe_artifacts/alpha_boundary_audit_v2"))
    parser.add_argument("--max-scenes", type=int, default=20)
    parser.add_argument("--render-scenes", type=int, default=8)
    parser.add_argument("--slice-axis", type=int, default=2)
    parser.add_argument("--distance-truncate", type=float, default=16.0)
    parser.add_argument("--variants", default=None, help="Comma-separated name:thr:sigma[:mincc[:close[:open]]] specs.")
    parser.add_argument("--no-normalize-density", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = parse_variants(args.variants)
    scenes = select_scenes(args.features_dir, args.split, args.split_key, args.max_scenes)
    if not scenes:
        raise SystemExit(f"no scenes found under {args.features_dir}")

    all_rows: list[dict[str, object]] = []
    for i, scene in enumerate(scenes, start=1):
        feature_path = args.features_dir / f"{scene}.npz"
        if not feature_path.exists():
            print(f"[warn] missing feature for split scene {scene}: {feature_path}", flush=True)
            continue
        print(f"[{i}/{len(scenes)}] {scene}", flush=True)
        alpha = load_alpha(feature_path, normalize_density=not args.no_normalize_density)
        rows, visual_data = audit_one(scene, alpha, variants, truncate=args.distance_truncate)
        all_rows.extend(rows)
        if not args.no_render and i <= args.render_scenes:
            render_scene(
                scene=scene,
                alpha=alpha,
                variants=variants,
                visual_data=visual_data,
                output_dir=output_dir,
                slice_axis=args.slice_axis,
                truncate=args.distance_truncate,
            )

    if not all_rows:
        raise SystemExit("no rows produced")

    csv_path = output_dir / "alpha_boundary_metrics_v2.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    write_summary(all_rows, output_dir / "README.md")
    print(f"[ok] wrote {csv_path}", flush=True)
    print(f"[ok] wrote {output_dir / 'README.md'}", flush=True)
    print(f"[ok] png count={len(list(output_dir.glob('*.png')))}", flush=True)


if __name__ == "__main__":
    main()
