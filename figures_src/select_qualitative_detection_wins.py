#!/usr/bin/env python3
"""Rank qualitative scenes where structure-first beats the NeRF-MAE baseline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon


def obb_polygon(box: np.ndarray) -> Polygon:
    x, y, _z, w, l, _h, theta = box.astype(float)
    c, s = np.cos(theta), np.sin(theta)
    dx = np.array([c, s]) * w / 2.0
    dy = np.array([-s, c]) * l / 2.0
    pts = [
        np.array([x, y]) - dx - dy,
        np.array([x, y]) + dx - dy,
        np.array([x, y]) + dx + dy,
        np.array([x, y]) - dx + dy,
    ]
    return Polygon(pts)


def approx_iou_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    poly_a = obb_polygon(box_a)
    poly_b = obb_polygon(box_b)
    if not poly_a.is_valid or not poly_b.is_valid or poly_a.area <= 0.0 or poly_b.area <= 0.0:
        return 0.0
    inter_area = poly_a.intersection(poly_b).area
    if inter_area <= 0.0:
        return 0.0
    za0, za1 = box_a[2] - box_a[5] / 2.0, box_a[2] + box_a[5] / 2.0
    zb0, zb1 = box_b[2] - box_b[5] / 2.0, box_b[2] + box_b[5] / 2.0
    inter_h = max(0.0, min(za1, zb1) - max(za0, zb0))
    inter = inter_area * inter_h
    vol_a = poly_a.area * box_a[5]
    vol_b = poly_b.area * box_b[5]
    return float(inter / max(vol_a + vol_b - inter, 1e-9))


def pairwise_iou(gt: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    return np.array([[approx_iou_3d(g, b) for b in boxes] for g in gt], dtype=np.float64)


def scene_metrics(gt_path: Path, proposal_path: Path, top_ks: tuple[int, ...]) -> dict[str, float]:
    gt = np.load(gt_path).astype(np.float64)
    with np.load(proposal_path) as z:
        boxes = z["proposals"].astype(np.float64)
        scores = z["scores"].astype(np.float64)
    if len(gt) == 0 or len(boxes) == 0:
        return {f"top{k}_{key}": 0.0 for k in top_ks for key in ["mean_iou", "r25", "r50"]}
    order = np.argsort(scores)[::-1]
    iou = pairwise_iou(gt, boxes)
    out: dict[str, float] = {}
    for k in top_ks:
        sel = order[:k]
        best = iou[:, sel].max(axis=1) if len(sel) else np.zeros(len(gt))
        out[f"top{k}_mean_iou"] = float(best.mean())
        out[f"top{k}_r25"] = float((best >= 0.25).mean())
        out[f"top{k}_r50"] = float((best >= 0.50).mean())
    out["top1_score"] = float(scores[order[0]]) if len(order) else 0.0
    out["gt_count"] = float(len(gt))
    return out


def collect_scenes(gt_dir: Path, baseline_dir: Path, ours_dir: Path, top_ks: tuple[int, ...]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for baseline_file in sorted(baseline_dir.glob("*.npz")):
        scene = baseline_file.stem
        ours_file = ours_dir / baseline_file.name
        gt_file = gt_dir / f"{scene}.npy"
        if not ours_file.exists() or not gt_file.exists():
            continue
        base = scene_metrics(gt_file, baseline_file, top_ks)
        ours = scene_metrics(gt_file, ours_file, top_ks)
        row: dict[str, float | str] = {"scene": scene}
        for key, value in base.items():
            row[f"baseline_{key}"] = value
        for key, value in ours.items():
            row[f"ours_{key}"] = value
        for k in top_ks:
            row[f"delta_top{k}_mean_iou"] = ours[f"top{k}_mean_iou"] - base[f"top{k}_mean_iou"]
            row[f"delta_top{k}_r50"] = ours[f"top{k}_r50"] - base[f"top{k}_r50"]
            row[f"delta_top{k}_r25"] = ours[f"top{k}_r25"] - base[f"top{k}_r25"]
        rows.append(row)
    rows.sort(key=lambda item: (item["delta_top3_mean_iou"], item["delta_top5_mean_iou"]), reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["front3d", "scannet"], required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--ours-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-ks", default="3,5,10")
    parser.add_argument("--print-top", type=int, default=10)
    args = parser.parse_args()

    top_ks = tuple(int(x) for x in args.top_ks.split(",") if x)
    rows = collect_scenes(args.gt_dir, args.baseline_dir, args.ours_dir, top_ks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["scene"]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")
    for row in rows[: args.print_top]:
        print(
            f"{row['scene']}: "
            f"dTop3IoU={row['delta_top3_mean_iou']:.3f}, "
            f"baseTop3={row['baseline_top3_mean_iou']:.3f}, "
            f"oursTop3={row['ours_top3_mean_iou']:.3f}, "
            f"baseR50={row['baseline_top3_r50']:.2f}, "
            f"oursR50={row['ours_top3_r50']:.2f}"
        )


if __name__ == "__main__":
    main()
