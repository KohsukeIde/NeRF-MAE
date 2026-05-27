#!/usr/bin/env python3
"""Summarize FCOS proposal quality from saved proposal dumps."""

from __future__ import annotations

import argparse
import json
import math
import sys
import sysconfig
from pathlib import Path
from typing import Any

import numpy as np
import torch

NERF_RPN_DIR = Path(__file__).resolve().parents[1]
if str(NERF_RPN_DIR) not in sys.path:
    sys.path.insert(0, str(NERF_RPN_DIR))
CUDA_OP_DIR = NERF_RPN_DIR / "model" / "rotated_iou" / "cuda_op"
soabi = sysconfig.get_config_var("SOABI") or f"cpython-{sys.version_info.major}{sys.version_info.minor}"
build_dirs = sorted(
    (CUDA_OP_DIR / "build").glob("lib.*"),
    key=lambda path: (soabi not in path.name, path.name),
) if (CUDA_OP_DIR / "build").exists() else []
for build_dir in reversed(build_dirs):
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))

from model.utils import box_iou_3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostic",
        action="append",
        required=True,
        help="Label and eval directory in the form label=/path/to/eval_dir",
    )
    parser.add_argument("--gt-boxes-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def parse_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid --diagnostic spec: {spec}")
        label, path = spec.split("=", 1)
        parsed.append((label, Path(path)))
    return parsed


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def proposal_iou_histogram(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"bins": [], "counts": [], "fractions": []}
    edges = np.asarray([0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.000001], dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    counts, _ = np.histogram(arr, bins=edges)
    total = max(int(counts.sum()), 1)
    labels = [
        "[0,0.05)",
        "[0.05,0.10)",
        "[0.10,0.25)",
        "[0.25,0.50)",
        "[0.50,0.75)",
        "[0.75,1.00]",
    ]
    return {
        "bins": labels,
        "counts": [int(x) for x in counts.tolist()],
        "fractions": [float(x / total) for x in counts.tolist()],
    }


def score_iou_calibration(scores: list[float], ious: list[float], n_bins: int = 5) -> list[dict[str, float]]:
    if not scores or not ious:
        return []
    score_arr = np.asarray(scores, dtype=np.float64)
    iou_arr = np.asarray(ious, dtype=np.float64)
    order = np.argsort(-score_arr)
    splits = np.array_split(order, n_bins)
    bins = []
    for idx, split in enumerate(splits):
        if split.size == 0:
            continue
        bin_scores = score_arr[split]
        bin_ious = iou_arr[split]
        bins.append(
            {
                "rank_bin": idx,
                "count": int(split.size),
                "score_min": float(bin_scores.min()),
                "score_max": float(bin_scores.max()),
                "score_mean": float(bin_scores.mean()),
                "iou_mean": float(bin_ious.mean()),
                "frac_iou_ge_050": float((bin_ious >= 0.50).mean()),
                "frac_iou_ge_075": float((bin_ious >= 0.75).mean()),
            }
        )
    return bins


def voc_ap_from_detections(tp_flags: list[bool], fp_flags: list[bool], num_gt: int) -> float:
    if num_gt <= 0 or not tp_flags:
        return 0.0
    tp = np.cumsum(np.asarray(tp_flags, dtype=np.float64))
    fp = np.cumsum(np.asarray(fp_flags, dtype=np.float64))
    recalls = tp / float(num_gt)
    precisions = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))


def object_size_ap(
    scene_records: list[dict[str, np.ndarray]],
    size_edges: tuple[float, float],
    iou_thresh: float,
) -> dict[str, float]:
    labels = ("small", "medium", "large")
    out: dict[str, float] = {}
    lo_hi = {
        "small": (-np.inf, size_edges[0]),
        "medium": (size_edges[0], size_edges[1]),
        "large": (size_edges[1], np.inf),
    }
    for label in labels:
        lo, hi = lo_hi[label]
        group_gt: list[np.ndarray] = []
        num_gt = 0
        detections: list[tuple[float, int, int]] = []
        for scene_idx, record in enumerate(scene_records):
            gt_boxes = record["gt_boxes"]
            gt_volumes = np.prod(np.maximum(gt_boxes[:, 3:6], 1e-6), axis=1)
            mask = (gt_volumes > lo) & (gt_volumes <= hi)
            cur_gt = gt_boxes[mask]
            group_gt.append(cur_gt)
            num_gt += int(cur_gt.shape[0])
            for proposal_idx, score in enumerate(record["scores"]):
                detections.append((float(score), scene_idx, proposal_idx))
        detections.sort(key=lambda item: item[0], reverse=True)
        used = [np.zeros(len(gt), dtype=bool) for gt in group_gt]
        tp_flags: list[bool] = []
        fp_flags: list[bool] = []
        for _, scene_idx, proposal_idx in detections:
            gt = group_gt[scene_idx]
            if gt.shape[0] == 0:
                tp_flags.append(False)
                fp_flags.append(True)
                continue
            gt_boxes = scene_records[scene_idx]["gt_boxes"]
            gt_volumes = np.prod(np.maximum(gt_boxes[:, 3:6], 1e-6), axis=1)
            mask = (gt_volumes > lo) & (gt_volumes <= hi)
            overlaps = scene_records[scene_idx]["ious"][proposal_idx, mask]
            gt_idx = int(overlaps.argmax())
            if float(overlaps[gt_idx]) >= iou_thresh and not used[scene_idx][gt_idx]:
                tp_flags.append(True)
                fp_flags.append(False)
                used[scene_idx][gt_idx] = True
            else:
                tp_flags.append(False)
                fp_flags.append(True)
        out[f"{label}_ap"] = voc_ap_from_detections(tp_flags, fp_flags, num_gt)
        out[f"{label}_num_gt"] = float(num_gt)
    return out


def load_eval(eval_dir: Path) -> dict[str, Any]:
    with (eval_dir / "eval.json").open() as f:
        return json.load(f)


def metric(eval_json: dict[str, Any], key: str, subkey: str = "ap") -> float:
    value = eval_json.get(key, {})
    if isinstance(value, dict):
        return float(value[subkey])
    return float(value)


def summarize_one(label: str, eval_dir: Path, gt_boxes_dir: Path, top_k: int, device: torch.device) -> dict[str, Any]:
    proposal_dir = eval_dir / "proposals"
    if not proposal_dir.exists():
        raise FileNotFoundError(f"missing proposal dir: {proposal_dir}")
    eval_json = load_eval(eval_dir)

    max_ious: list[float] = []
    scores_all: list[float] = []
    tp_scores_50: list[float] = []
    fp_scores_50: list[float] = []
    first_tp_ranks_50: list[float] = []
    top50_tp_rates_50: list[float] = []
    top100_tp_rates_50: list[float] = []
    top300_tp_rates_50: list[float] = []
    center_err_norm_25: list[float] = []
    center_err_norm_50: list[float] = []
    size_rel_err_25: list[float] = []
    size_rel_err_50: list[float] = []
    scene_records: list[dict[str, np.ndarray]] = []
    gt_volumes_all: list[float] = []
    level_counts = np.zeros(4, dtype=np.float64)
    tp50_level_counts = np.zeros(4, dtype=np.float64)
    scene_count = 0

    for npz_path in sorted(proposal_dir.glob("*.npz")):
        scene = npz_path.stem
        gt_path = gt_boxes_dir / f"{scene}.npy"
        if not gt_path.exists():
            continue
        proposal_npz = np.load(npz_path)
        proposals = proposal_npz["proposals"].astype(np.float32)
        scores = proposal_npz["scores"].astype(np.float32)
        level_indices = proposal_npz["level_indices"].astype(np.int64)
        gt_boxes = np.load(gt_path).astype(np.float32)
        if proposals.size == 0 or gt_boxes.size == 0:
            continue

        order = np.argsort(-scores)
        proposals = proposals[order][:top_k]
        scores = scores[order][:top_k]
        level_indices = level_indices[order][:top_k]
        if proposals.size == 0:
            continue

        proposals_t = torch.from_numpy(proposals).to(device)
        gt_t = torch.from_numpy(gt_boxes).to(device)
        with torch.no_grad():
            ious = box_iou_3d(proposals_t, gt_t).detach().cpu().numpy()
        scene_records.append(
            {
                "proposals": proposals.copy(),
                "scores": scores.copy(),
                "gt_boxes": gt_boxes.copy(),
                "ious": ious.copy(),
            }
        )
        gt_volumes_all.extend(np.prod(np.maximum(gt_boxes[:, 3:6], 1e-6), axis=1).tolist())
        best_gt = ious.argmax(axis=1)
        best_iou = ious.max(axis=1)

        matched_gt = gt_boxes[best_gt]
        center_err = np.linalg.norm(proposals[:, :3] - matched_gt[:, :3], axis=1)
        gt_diag = np.linalg.norm(np.maximum(matched_gt[:, 3:6], 1e-6), axis=1)
        center_norm = center_err / np.maximum(gt_diag, 1e-6)
        size_rel = np.mean(
            np.abs(proposals[:, 3:6] - matched_gt[:, 3:6])
            / np.maximum(matched_gt[:, 3:6], 1e-6),
            axis=1,
        )

        tp50 = best_iou >= 0.5
        tp25 = best_iou >= 0.25
        max_ious.extend(best_iou.tolist())
        scores_all.extend(scores.tolist())
        tp_scores_50.extend(scores[tp50].tolist())
        fp_scores_50.extend(scores[~tp50].tolist())
        center_err_norm_25.extend(center_norm[tp25].tolist())
        center_err_norm_50.extend(center_norm[tp50].tolist())
        size_rel_err_25.extend(size_rel[tp25].tolist())
        size_rel_err_50.extend(size_rel[tp50].tolist())

        if tp50.any():
            first_tp_ranks_50.append(float(np.argmax(tp50) + 1))
        for limit, bucket in (
            (50, top50_tp_rates_50),
            (100, top100_tp_rates_50),
            (300, top300_tp_rates_50),
        ):
            actual = min(limit, tp50.shape[0])
            if actual > 0:
                bucket.append(float(tp50[:actual].mean()))

        clipped_levels = np.clip(level_indices, 0, 3)
        level_counts += np.bincount(clipped_levels, minlength=4)[:4]
        tp50_level_counts += np.bincount(clipped_levels[tp50], minlength=4)[:4]
        scene_count += 1

    ap50 = metric(eval_json, "ap_50")
    ap75 = metric(eval_json, "ap_75")
    tp50_count = sum(float(x >= 0.50) for x in max_ious)
    tp75_count = sum(float(x >= 0.75) for x in max_ious)
    if gt_volumes_all:
        size_q33, size_q66 = np.quantile(np.asarray(gt_volumes_all, dtype=np.float64), [1 / 3, 2 / 3])
        size_ap50 = object_size_ap(scene_records, (float(size_q33), float(size_q66)), 0.50)
        size_ap75 = object_size_ap(scene_records, (float(size_q33), float(size_q66)), 0.75)
    else:
        size_q33 = size_q66 = 0.0
        size_ap50 = {}
        size_ap75 = {}
    level_share = level_counts / level_counts.sum() if level_counts.sum() else np.zeros(4)
    tp_level_share = (
        tp50_level_counts / tp50_level_counts.sum()
        if tp50_level_counts.sum()
        else np.zeros(4)
    )
    return {
        "label": label,
        "eval_dir": str(eval_dir),
        "scene_count": scene_count,
        "eval": {
            "ap25": metric(eval_json, "ap_25"),
            "ap50": ap50,
            "ap75": ap75,
            "ap75_over_ap50": ap75 / ap50 if ap50 else 0.0,
            "recall50_top300": metric(eval_json, "recall_50_top_300", "ar"),
            "recall25_top300": metric(eval_json, "recall_25_top_300", "ar"),
        },
        "proposal": {
            "top_k": top_k,
            "mean_iou": mean(max_ious),
            "median_iou": quantile(max_ious, 0.5),
            "p90_iou": quantile(max_ious, 0.9),
            "frac_iou_ge_025": mean([float(x >= 0.25) for x in max_ious]),
            "frac_iou_ge_050": mean([float(x >= 0.50) for x in max_ious]),
            "frac_iou_ge_075": mean([float(x >= 0.75) for x in max_ious]),
            "tp50_fail75_fraction": float((tp50_count - tp75_count) / tp50_count) if tp50_count else 0.0,
            "score_mean": mean(scores_all),
            "tp50_score_mean": mean(tp_scores_50),
            "fp50_score_mean": mean(fp_scores_50),
            "first_tp50_rank_mean": mean(first_tp_ranks_50),
            "top50_tp50_rate": mean(top50_tp_rates_50),
            "top100_tp50_rate": mean(top100_tp_rates_50),
            "top300_tp50_rate": mean(top300_tp_rates_50),
            "center_error_norm_iou_ge_025": mean(center_err_norm_25),
            "center_error_norm_iou_ge_050": mean(center_err_norm_50),
            "size_rel_error_iou_ge_025": mean(size_rel_err_25),
            "size_rel_error_iou_ge_050": mean(size_rel_err_50),
            "level_share": {f"level_{idx}": float(value) for idx, value in enumerate(level_share)},
            "tp50_level_share": {
                f"level_{idx}": float(value) for idx, value in enumerate(tp_level_share)
            },
            "iou_histogram": proposal_iou_histogram(max_ious),
            "score_iou_calibration": score_iou_calibration(scores_all, max_ious),
            "object_size_edges": {
                "small_medium_volume": float(size_q33),
                "medium_large_volume": float(size_q66),
            },
            "object_size_ap50": size_ap50,
            "object_size_ap75": size_ap75,
        },
    }


def fmt(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return "0.0000"
    return f"{value:.4f}"


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Proposal Quality Summary",
        "",
        "| label | AP@50 | AP@75 | AP75/AP50 | R50@300 | mean IoU | frac IoU>=0.5 | TP50 fail75 | center err >=0.5 | size err >=0.5 | first TP rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, payload in summary.items():
        ev = payload["eval"]
        pr = payload["proposal"]
        lines.append(
            "| {label} | {ap50} | {ap75} | {ratio} | {recall} | {mean_iou} | {frac50} | {fail75} | {center50} | {size50} | {first_tp} |".format(
                label=label,
                ap50=fmt(ev["ap50"]),
                ap75=fmt(ev["ap75"]),
                ratio=fmt(ev["ap75_over_ap50"]),
                recall=fmt(ev["recall50_top300"]),
                mean_iou=fmt(pr["mean_iou"]),
                frac50=fmt(pr["frac_iou_ge_050"]),
                fail75=fmt(pr["tp50_fail75_fraction"]),
                center50=fmt(pr["center_error_norm_iou_ge_050"]),
                size50=fmt(pr["size_rel_error_iou_ge_050"]),
                first_tp=fmt(pr["first_tp50_rank_mean"]),
            )
        )
    lines.append("")
    for label, payload in summary.items():
        pr = payload["proposal"]
        lines.extend(
            [
                f"## {label}",
                "",
                "- Proposal IoU: mean={mean_iou}, median={median_iou}, p90={p90_iou}, frac>=0.25/0.5/0.75={f25}/{f50}/{f75}".format(
                    mean_iou=fmt(pr["mean_iou"]),
                    median_iou=fmt(pr["median_iou"]),
                    p90_iou=fmt(pr["p90_iou"]),
                    f25=fmt(pr["frac_iou_ge_025"]),
                    f50=fmt(pr["frac_iou_ge_050"]),
                    f75=fmt(pr["frac_iou_ge_075"]),
                ),
                "- IoU histogram: "
                + ", ".join(
                    f"{bin_label}={fraction:.4f}"
                    for bin_label, fraction in zip(
                        pr["iou_histogram"]["bins"],
                        pr["iou_histogram"]["fractions"],
                    )
                ),
                "- Ranking: tp50 score={tp_score}, fp50 score={fp_score}, first TP rank={first_tp}, top50/top100/top300 TP={top50}/{top100}/{top300}".format(
                    tp_score=fmt(pr["tp50_score_mean"]),
                    fp_score=fmt(pr["fp50_score_mean"]),
                    first_tp=fmt(pr["first_tp50_rank_mean"]),
                    top50=fmt(pr["top50_tp50_rate"]),
                    top100=fmt(pr["top100_tp50_rate"]),
                    top300=fmt(pr["top300_tp50_rate"]),
                ),
                "- Localization among matched proposals: center norm err IoU>=0.25/0.5={c25}/{c50}, size rel err IoU>=0.25/0.5={s25}/{s50}".format(
                    c25=fmt(pr["center_error_norm_iou_ge_025"]),
                    c50=fmt(pr["center_error_norm_iou_ge_050"]),
                    s25=fmt(pr["size_rel_error_iou_ge_025"]),
                    s50=fmt(pr["size_rel_error_iou_ge_050"]),
                ),
                "- Level share: "
                + ", ".join(f"L{i}={pr['level_share'][f'level_{i}']:.4f}" for i in range(4)),
                "- TP50 level share: "
                + ", ".join(f"L{i}={pr['tp50_level_share'][f'level_{i}']:.4f}" for i in range(4)),
                "- Score-IoU calibration: "
                + "; ".join(
                    "Q{rank}: score={score:.4f}, IoU={iou:.4f}, frac50={frac50:.4f}, frac75={frac75:.4f}".format(
                        rank=item["rank_bin"],
                        score=item["score_mean"],
                        iou=item["iou_mean"],
                        frac50=item["frac_iou_ge_050"],
                        frac75=item["frac_iou_ge_075"],
                    )
                    for item in pr["score_iou_calibration"]
                ),
                "- Object size AP@50: small={s50}, medium={m50}, large={l50}; AP@75: small={s75}, medium={m75}, large={l75}".format(
                    s50=fmt(pr["object_size_ap50"].get("small_ap", 0.0)),
                    m50=fmt(pr["object_size_ap50"].get("medium_ap", 0.0)),
                    l50=fmt(pr["object_size_ap50"].get("large_ap", 0.0)),
                    s75=fmt(pr["object_size_ap75"].get("small_ap", 0.0)),
                    m75=fmt(pr["object_size_ap75"].get("medium_ap", 0.0)),
                    l75=fmt(pr["object_size_ap75"].get("large_ap", 0.0)),
                ),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    summary = {}
    for label, eval_dir in parse_specs(args.diagnostic):
        summary[label] = summarize_one(
            label=label,
            eval_dir=eval_dir,
            gt_boxes_dir=args.gt_boxes_dir,
            top_k=args.top_k,
            device=device,
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(summary, f, indent=2)
    args.output_md.write_text(build_markdown(summary) + "\n")
    print(f"[info] wrote {args.output_json}")
    print(f"[info] wrote {args.output_md}")


if __name__ == "__main__":
    main()
