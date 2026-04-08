"""Write quick-transfer summary JSON/CSV from TSV rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "variant",
    "pretrain_ckpt",
    "fcos_ckpt",
    "status",
    "ap25",
    "ap50",
    "ap75",
    "recall_50_top_300",
    "recall_50_top_1000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-tsv", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out", required=True)
    return parser.parse_args()


def load_metric(eval_path: str | None, key: str, nested: str | None = None) -> Any:
    if not eval_path:
      return None
    path = Path(eval_path)
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    value = data.get(key)
    if isinstance(value, dict) and nested is not None:
        return value.get(nested)
    return value


def main() -> None:
    args = parse_args()
    rows_path = Path(args.rows_tsv)
    out_json = Path(args.json_out)
    out_csv = Path(args.csv_out)

    records = []
    if rows_path.exists():
        with rows_path.open() as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                fieldnames=["variant", "pretrain_ckpt", "fcos_ckpt", "status", "eval_json"],
            )
            for row in reader:
                eval_json = row["eval_json"] or None
                records.append(
                    {
                        "variant": row["variant"],
                        "pretrain_ckpt": row["pretrain_ckpt"],
                        "fcos_ckpt": row["fcos_ckpt"],
                        "status": row["status"],
                        "ap25": load_metric(eval_json, "ap_25", "ap"),
                        "ap50": load_metric(eval_json, "ap_50", "ap"),
                        "ap75": load_metric(eval_json, "ap_75", "ap"),
                        "recall_50_top_300": load_metric(eval_json, "recall_50_top_300", "ar"),
                        "recall_50_top_1000": load_metric(eval_json, "recall_50_top_1000", "ar"),
                    }
                )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(records, f, indent=2)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


if __name__ == "__main__":
    main()
