#!/usr/bin/env python3
"""Summarize the ABCI3 e300 FCOS gate results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


CONDITIONS = ("baseline", "cosine_ramp", "cosine_ramp_alpha_shuffle")
LABELS = {
    "baseline": "baseline",
    "cosine_ramp": "cosine",
    "cosine_ramp_alpha_shuffle": "shuffle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--fcos-epochs", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--run-suffix", default="abci3clean")
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def pretrain_name(condition: str, epochs: int, seed: int, suffix: str) -> str:
    suffix_part = f"_{suffix}" if suffix else ""
    if condition == "baseline":
        return f"nerfmae_all_p1.0_e{epochs}_seed{seed}{suffix_part}"
    return f"nerfmae_alpha_rgba_curr_{condition}_p1.0_e{epochs}_seed{seed}{suffix_part}"


def eval_path(root: Path, condition: str, epochs: int, seed: int, suffix: str, fcos_epochs: int) -> Path:
    pre = pretrain_name(condition, epochs, seed, suffix)
    save = f"{pre}_epoch{epochs}_sched_epoch_seed{seed}_fcos{fcos_epochs}_eval"
    return root / "output" / "nerf_rpn" / "results" / save / "eval.json"


def metric(data: dict[str, Any] | None, key: str, nested: str = "ap") -> float | None:
    if data is None:
        return None
    value = data.get(key)
    if isinstance(value, dict):
        value = value.get(nested)
    if value is None:
        return None
    return float(value)


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def load_eval(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int], dict[str, Any]] = {}

    for seed in args.seeds:
        for condition in CONDITIONS:
            path = eval_path(args.root, condition, args.epochs, seed, args.run_suffix, args.fcos_epochs)
            data = load_eval(path)
            row = {
                "condition": LABELS[condition],
                "condition_key": condition,
                "seed": seed,
                "status": "done" if data is not None else "missing",
                "ap25": metric(data, "ap_25"),
                "ap50": metric(data, "ap_50"),
                "ap75": metric(data, "ap_75"),
                "recall50_top300": metric(data, "recall_50_top_300", "ar"),
                "recall50_top1000": metric(data, "recall_50_top_1000", "ar"),
                "eval_json": str(path),
            }
            rows.append(row)
            by_key[(condition, seed)] = row

    diffs: list[dict[str, Any]] = []
    for seed in args.seeds:
        base = by_key[("baseline", seed)]["ap50"]
        cos = by_key[("cosine_ramp", seed)]["ap50"]
        shuf = by_key[("cosine_ramp_alpha_shuffle", seed)]["ap50"]
        if cos is not None and base is not None:
            diffs.append({"seed": seed, "comparison": "cosine-baseline", "ap50_diff": cos - base})
        if cos is not None and shuf is not None:
            diffs.append({"seed": seed, "comparison": "cosine-shuffle", "ap50_diff": cos - shuf})

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "seed",
                "status",
                "ap25",
                "ap50",
                "ap75",
                "recall50_top300",
                "recall50_top1000",
                "eval_json",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    complete_ap50 = [row for row in rows if row["ap50"] is not None]
    by_condition = {
        LABELS[condition]: [row["ap50"] for row in rows if row["condition_key"] == condition and row["ap50"] is not None]
        for condition in CONDITIONS
    }
    diff_groups = {
        name: [row["ap50_diff"] for row in diffs if row["comparison"] == name]
        for name in ("cosine-baseline", "cosine-shuffle")
    }

    payload = {
        "rows": rows,
        "diffs": diffs,
        "means": {key: (mean(vals) if vals else None) for key, vals in by_condition.items()},
        "diff_means": {key: (mean(vals) if vals else None) for key, vals in diff_groups.items()},
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# ABCI3 e300 Gate Summary",
        "",
        "| condition | seed | status | AP@50 | AP@25 | AP@75 | Recall@50 top300 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {seed} | {status} | {ap50} | {ap25} | {ap75} | {recall} |".format(
                condition=row["condition"],
                seed=row["seed"],
                status=row["status"],
                ap50=fmt(row["ap50"]),
                ap25=fmt(row["ap25"]),
                ap75=fmt(row["ap75"]),
                recall=fmt(row["recall50_top300"]),
            )
        )

    lines.extend(["", "## Paired AP@50 Diffs", "", "| seed | comparison | diff |", "|---:|---|---:|"])
    for row in diffs:
        lines.append(f"| {row['seed']} | {row['comparison']} | {row['ap50_diff']:.4f} |")

    lines.extend(["", "## Gate Readout", ""])
    lines.append(f"- Completed metric rows: {len(complete_ap50)}/{len(rows)}")
    for key, vals in by_condition.items():
        lines.append(f"- Mean AP@50 {key}: {fmt(mean(vals) if vals else None)}")
    for key, vals in diff_groups.items():
        positives = sum(value > 0 for value in vals)
        lines.append(f"- {key}: {positives}/{len(vals)} positive, mean diff {fmt(mean(vals) if vals else None)}")

    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
