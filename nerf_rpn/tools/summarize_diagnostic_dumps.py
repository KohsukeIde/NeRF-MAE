import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model.utils import box_iou_3d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize proposal/voxel diagnostic dumps into JSON and Markdown."
    )
    parser.add_argument(
        "--diagnostic",
        action="append",
        required=True,
        help="Label and directory in the form label=/abs/path/to/diagnostic_dir",
    )
    parser.add_argument(
        "--gt_boxes_dir",
        required=True,
        help="Directory containing Front3D OBB .npy files.",
    )
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--tp_iou", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    return parser.parse_args()


def parse_diag_specs(specs):
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --diagnostic spec: {spec}")
        label, path = spec.split("=", 1)
        parsed.append((label, Path(path)))
    return parsed


def quantile(values, q):
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def mean_or_zero(values):
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def summarize_voxel_scores(voxel_dir):
    per_level = {}
    for npz_path in sorted(voxel_dir.glob("*.npz")):
        arr = np.load(npz_path)
        for level_key in arr.files:
            values = arr[level_key].astype(np.float32).reshape(-1)
            if values.size == 0:
                continue
            bucket = per_level.setdefault(
                level_key,
                {
                    "mean": [],
                    "std": [],
                    "max": [],
                    "p99": [],
                    "peakiness": [],
                    "max_over_mean": [],
                },
            )
            mean_v = float(values.mean())
            std_v = float(values.std())
            max_v = float(values.max())
            p99_v = float(np.quantile(values, 0.99))
            bucket["mean"].append(mean_v)
            bucket["std"].append(std_v)
            bucket["max"].append(max_v)
            bucket["p99"].append(p99_v)
            if mean_v > 0:
                bucket["peakiness"].append(p99_v / mean_v)
                bucket["max_over_mean"].append(max_v / mean_v)

    summary = {}
    peakiness_all = []
    max_over_mean_all = []
    std_all = []
    for level_key, stats in sorted(per_level.items(), key=lambda item: int(item[0])):
        level_summary = {
            "mean": mean_or_zero(stats["mean"]),
            "std": mean_or_zero(stats["std"]),
            "max": mean_or_zero(stats["max"]),
            "p99": mean_or_zero(stats["p99"]),
            "peakiness": mean_or_zero(stats["peakiness"]),
            "max_over_mean": mean_or_zero(stats["max_over_mean"]),
        }
        summary[f"level_{level_key}"] = level_summary
        if stats["peakiness"]:
            peakiness_all.extend(stats["peakiness"])
        if stats["max_over_mean"]:
            max_over_mean_all.extend(stats["max_over_mean"])
        if stats["std"]:
            std_all.extend(stats["std"])

    return {
        "levels": summary,
        "global_peakiness": mean_or_zero(peakiness_all),
        "global_max_over_mean": mean_or_zero(max_over_mean_all),
        "global_std": mean_or_zero(std_all),
    }


def summarize_proposals(diag_dir, gt_boxes_dir, top_k, tp_iou, device):
    proposals_dir = diag_dir / "proposals"
    eval_path = diag_dir / "eval.json"
    if not proposals_dir.exists():
        raise FileNotFoundError(f"Missing proposals dir: {proposals_dir}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval.json: {eval_path}")

    with eval_path.open() as f:
        eval_json = json.load(f)

    max_ious = []
    tp_scores = []
    fp_scores = []
    first_tp_ranks = []
    top50_tp_rates = []
    top100_tp_rates = []
    top300_tp_rates = []
    level_counts = np.zeros(4, dtype=np.float64)
    tp_level_counts = np.zeros(4, dtype=np.float64)
    scene_stats = []

    for npz_path in sorted(proposals_dir.glob("*.npz")):
        scene = npz_path.stem
        gt_path = gt_boxes_dir / f"{scene}.npy"
        if not gt_path.exists():
            continue

        proposal_npz = np.load(npz_path)
        proposals = proposal_npz["proposals"].astype(np.float32)
        scores = proposal_npz["scores"].astype(np.float32)
        level_indices = proposal_npz["level_indices"].astype(np.int64)
        gt_boxes = np.load(gt_path).astype(np.float32)

        if proposals.shape[0] == 0 or gt_boxes.shape[0] == 0:
            continue

        order = np.argsort(-scores)
        proposals = proposals[order][:top_k]
        scores = scores[order][:top_k]
        level_indices = level_indices[order][:top_k]

        proposals_t = torch.from_numpy(proposals).to(device)
        gt_t = torch.from_numpy(gt_boxes).to(device)
        ious = box_iou_3d(proposals_t, gt_t).detach().cpu()
        max_iou_per_prop = ious.max(dim=1).values.numpy()
        tp_mask = max_iou_per_prop >= tp_iou

        max_ious.extend(max_iou_per_prop.tolist())
        tp_scores.extend(scores[tp_mask].tolist())
        fp_scores.extend(scores[~tp_mask].tolist())
        if tp_mask.any():
            first_tp_ranks.append(float(np.argmax(tp_mask) + 1))
        level_hist = np.bincount(level_indices.clip(0, 3), minlength=4)[:4]
        level_counts += level_hist
        tp_level_hist = np.bincount(level_indices[tp_mask].clip(0, 3), minlength=4)[:4]
        tp_level_counts += tp_level_hist

        for limit, bucket in ((50, top50_tp_rates), (100, top100_tp_rates), (300, top300_tp_rates)):
            actual = min(limit, tp_mask.shape[0])
            if actual > 0:
                bucket.append(float(tp_mask[:actual].mean()))

        scene_stats.append(
            {
                "scene": scene,
                "proposal_count": int(proposals.shape[0]),
                "mean_iou": float(max_iou_per_prop.mean()) if max_iou_per_prop.size else 0.0,
                "top50_tp_rate": float(tp_mask[: min(50, tp_mask.shape[0])].mean())
                if tp_mask.size
                else 0.0,
            }
        )

    level_share = (
        (level_counts / level_counts.sum()).tolist() if level_counts.sum() > 0 else [0.0] * 4
    )
    tp_level_share = (
        (tp_level_counts / tp_level_counts.sum()).tolist()
        if tp_level_counts.sum() > 0
        else [0.0] * 4
    )

    return {
        "eval": {
            "ap50": float(eval_json["ap_50"]["ap"]),
            "ap25": float(eval_json["ap_25"]["ap"]),
            "ap75": float(eval_json["ap_75"]["ap"]),
            "rec50_top300": float(eval_json["recall_50_top_300"]["ar"]),
            "rec25_top300": float(eval_json["recall_25_top_300"]["ar"]),
            "ar_top300": float(eval_json["recall_ar_top_300"]["ar"]),
        },
        "proposal": {
            "scene_count": len(scene_stats),
            "mean_iou": mean_or_zero(max_ious),
            "median_iou": quantile(max_ious, 0.5),
            "p90_iou": quantile(max_ious, 0.9),
            "frac_iou_ge_025": mean_or_zero([x >= 0.25 for x in max_ious]),
            "frac_iou_ge_050": mean_or_zero([x >= 0.50 for x in max_ious]),
            "frac_iou_ge_075": mean_or_zero([x >= 0.75 for x in max_ious]),
            "tp_score_mean": mean_or_zero(tp_scores),
            "fp_score_mean": mean_or_zero(fp_scores),
            "first_tp_rank_mean": mean_or_zero(first_tp_ranks),
            "top50_tp_rate": mean_or_zero(top50_tp_rates),
            "top100_tp_rate": mean_or_zero(top100_tp_rates),
            "top300_tp_rate": mean_or_zero(top300_tp_rates),
            "level_share": {f"level_{i}": float(level_share[i]) for i in range(4)},
            "tp_level_share": {f"level_{i}": float(tp_level_share[i]) for i in range(4)},
        },
        "scene_stats": scene_stats,
    }


def build_markdown(summary):
    lines = []
    lines.append("# Diagnostic Summary")
    lines.append("")
    lines.append(
        "| label | AP@50 | AP@75 | top300 mean IoU | frac IoU>=0.5 | TP score mean | FP score mean | first TP rank | top50 TP rate | voxel peakiness |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, payload in summary.items():
        eval_s = payload["eval"]
        prop_s = payload["proposal"]
        voxel_s = payload["voxel"]
        lines.append(
            "| {label} | {ap50:.4f} | {ap75:.4f} | {mean_iou:.4f} | {frac50:.4f} | {tp_score:.4f} | {fp_score:.4f} | {first_tp:.2f} | {top50:.4f} | {peakiness:.4f} |".format(
                label=label,
                ap50=eval_s["ap50"],
                ap75=eval_s["ap75"],
                mean_iou=prop_s["mean_iou"],
                frac50=prop_s["frac_iou_ge_050"],
                tp_score=prop_s["tp_score_mean"],
                fp_score=prop_s["fp_score_mean"],
                first_tp=prop_s["first_tp_rank_mean"],
                top50=prop_s["top50_tp_rate"],
                peakiness=voxel_s["global_peakiness"],
            )
        )
    lines.append("")
    for label, payload in summary.items():
        lines.append(f"## {label}")
        lines.append("")
        lines.append(
            "- Eval: AP@50={ap50:.4f}, AP@25={ap25:.4f}, AP@75={ap75:.4f}, Recall@50 top300={rec50_top300:.4f}, AR top300={ar_top300:.4f}".format(
                **payload["eval"]
            )
        )
        lines.append(
            "- Proposal: mean IoU={mean_iou:.4f}, median IoU={median_iou:.4f}, p90 IoU={p90_iou:.4f}, frac IoU>=0.25/0.5/0.75={frac_iou_ge_025:.4f}/{frac_iou_ge_050:.4f}/{frac_iou_ge_075:.4f}".format(
                **payload["proposal"]
            )
        )
        lines.append(
            "- Ranking: TP score mean={tp_score_mean:.4f}, FP score mean={fp_score_mean:.4f}, first TP rank mean={first_tp_rank_mean:.2f}, top50/top100/top300 TP rate={top50_tp_rate:.4f}/{top100_tp_rate:.4f}/{top300_tp_rate:.4f}".format(
                **payload["proposal"]
            )
        )
        lines.append(
            "- Level share: "
            + ", ".join(
                f"L{i}={payload['proposal']['level_share'][f'level_{i}']:.4f}"
                for i in range(4)
            )
        )
        lines.append(
            "- TP level share: "
            + ", ".join(
                f"L{i}={payload['proposal']['tp_level_share'][f'level_{i}']:.4f}"
                for i in range(4)
            )
        )
        lines.append(
            "- Voxel sharpness: peakiness={global_peakiness:.4f}, max/mean={global_max_over_mean:.4f}, std={global_std:.4f}".format(
                **payload["voxel"]
            )
        )
        lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gt_boxes_dir = Path(args.gt_boxes_dir)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    summary = {}
    for label, diag_dir in parse_diag_specs(args.diagnostic):
        proposal_summary = summarize_proposals(
            diag_dir=diag_dir,
            gt_boxes_dir=gt_boxes_dir,
            top_k=args.top_k,
            tp_iou=args.tp_iou,
            device=device,
        )
        voxel_summary = summarize_voxel_scores(diag_dir / "voxel_scores")
        proposal_summary["voxel"] = voxel_summary
        summary[label] = proposal_summary

    with output_json.open("w") as f:
        json.dump(summary, f, indent=2)
    output_md.write_text(build_markdown(summary))


if __name__ == "__main__":
    main()
