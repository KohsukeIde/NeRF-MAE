#!/usr/bin/env python3
"""Plot compact proposal-quality figures from summary JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--max-label-chars", type=int, default=22)
    return parser.parse_args()


def short_label(label: str, max_chars: int) -> str:
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 1] + "."


def values(payload: dict, key: str) -> list[float]:
    vals = []
    for item in payload.values():
        if key.startswith("eval."):
            vals.append(float(item["eval"][key.split(".", 1)[1]]))
        elif key.startswith("proposal."):
            vals.append(float(item["proposal"][key.split(".", 1)[1]]))
        else:
            raise ValueError(key)
    return vals


def grouped_bar(ax, labels: list[str], series: list[tuple[str, list[float]]], ylabel: str) -> None:
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    for idx, (name, vals) in enumerate(series):
        offset = (idx - (len(series) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def main() -> None:
    args = parse_args()
    with args.summary_json.open() as f:
        payload = json.load(f)
    labels = [short_label(label, args.max_label_chars) for label in payload]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    if args.title:
        fig.suptitle(args.title, fontsize=14)

    grouped_bar(
        axes[0, 0],
        labels,
        [
            ("AP@50", values(payload, "eval.ap50")),
            ("AP@75", values(payload, "eval.ap75")),
        ],
        "AP",
    )
    grouped_bar(
        axes[0, 1],
        labels,
        [("AP75/AP50", values(payload, "eval.ap75_over_ap50"))],
        "ratio",
    )
    grouped_bar(
        axes[1, 0],
        labels,
        [
            ("mean IoU", values(payload, "proposal.mean_iou")),
            ("frac IoU>=0.5", values(payload, "proposal.frac_iou_ge_050")),
        ],
        "proposal quality",
    )
    grouped_bar(
        axes[1, 1],
        labels,
        [
            ("center err IoU>=0.5", values(payload, "proposal.center_error_norm_iou_ge_050")),
            ("size err IoU>=0.5", values(payload, "proposal.size_rel_error_iou_ge_050")),
        ],
        "error",
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=180)
    print(f"[info] wrote {args.output_png}")


if __name__ == "__main__":
    main()
