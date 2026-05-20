#!/usr/bin/env python3
"""Parse NeRF-MAE pretrain speed from worker_0.log files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


STEP_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO\] "
    r"epoch (?P<epoch>\d+) \[(?P<iter>\d+)/(?P<steps>\d+)\]"
)
STEP_TIME_RE = re.compile(r"step_time: (?P<step_time>[0-9.]+)s")


@dataclass(frozen=True)
class StepRecord:
    timestamp: datetime
    epoch: int
    iteration: int
    steps_per_epoch: int
    completed_step: int
    step_time: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, help="TSV manifest from submit_abci3_pretrain_speed_benchmark.sh")
    parser.add_argument("--log", type=Path, action="append", default=[], help="worker_0.log path; repeatable")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Exclude intervals that start before this completed-step count")
    parser.add_argument("--min-intervals", type=int, default=1)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter="\t")]


def rows_from_logs(logs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for log in logs:
        save_name = log.parent.parent.name if log.parent.name == "log" else log.stem
        rows.append(
            {
                "topology": "",
                "nodes": "",
                "local_gpus": "",
                "world_size": "",
                "global_batch": "",
                "batch_size_per_gpu": "",
                "deterministic": "",
                "stage_pretrain_data": "",
                "epochs": "",
                "seed": "",
                "save_name": save_name,
                "worker_log": str(log),
            }
        )
    return rows


def parse_step_records(path: Path) -> list[StepRecord]:
    raw: list[tuple[datetime, int, int, int, float | None]] = []
    with path.open(errors="replace") as f:
        for line in f:
            match = STEP_RE.match(line)
            if not match:
                continue
            step_time_match = STEP_TIME_RE.search(line)
            raw.append(
                (
                    datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S,%f"),
                    int(match.group("epoch")),
                    int(match.group("iter")),
                    int(match.group("steps")),
                    float(step_time_match.group("step_time")) if step_time_match else None,
                )
            )
    if not raw:
        return []
    first_epoch = min(epoch for _, epoch, _, _, _ in raw)
    records = []
    for timestamp, epoch, iteration, steps_per_epoch, step_time in raw:
        completed_step = (epoch - first_epoch) * steps_per_epoch + iteration + 1
        records.append(StepRecord(timestamp, epoch, iteration, steps_per_epoch, completed_step, step_time))
    records.sort(key=lambda item: item.timestamp)
    return records


def summarize_log(path: Path, warmup_steps: int, min_intervals: int) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing_log", "worker_log": str(path)}

    records = parse_step_records(path)
    if len(records) < 2:
        return {
            "status": "no_step_intervals",
            "worker_log": str(path),
            "logged_steps": len(records),
        }

    has_explicit_step_times = any(record.step_time is not None for record in records)
    explicit_step_times = [
        record.step_time
        for record in records
        if record.step_time is not None and record.completed_step > warmup_steps
    ]
    if has_explicit_step_times and len(explicit_step_times) < min_intervals:
        return {
            "status": "insufficient_intervals",
            "timing_source": "explicit_step_time",
            "worker_log": str(path),
            "logged_steps": len(records),
            "intervals": len(explicit_step_times),
            "measured_steps": len(explicit_step_times),
        }
    if explicit_step_times:
        return {
            "status": "ok",
            "timing_source": "explicit_step_time",
            "worker_log": str(path),
            "logged_steps": len(records),
            "intervals": len(explicit_step_times),
            "measured_steps": len(explicit_step_times),
            "first_logged_step": records[0].completed_step,
            "last_logged_step": records[-1].completed_step,
            "mean_sec_per_step": mean(explicit_step_times),
            "median_sec_per_step": median(explicit_step_times),
            "p10_sec_per_step": percentile(explicit_step_times, 0.10),
            "p90_sec_per_step": percentile(explicit_step_times, 0.90),
        }

    intervals: list[float] = []
    measured_steps = 0
    for prev, cur in zip(records, records[1:]):
        step_delta = cur.completed_step - prev.completed_step
        sec_delta = (cur.timestamp - prev.timestamp).total_seconds()
        if step_delta <= 0 or sec_delta <= 0:
            continue
        if prev.completed_step < warmup_steps:
            continue
        intervals.append(sec_delta / step_delta)
        measured_steps += step_delta

    if len(intervals) < min_intervals:
        return {
            "status": "insufficient_intervals",
            "timing_source": "timestamp_delta",
            "worker_log": str(path),
            "logged_steps": len(records),
            "intervals": len(intervals),
            "measured_steps": measured_steps,
        }

    return {
        "status": "ok",
        "timing_source": "timestamp_delta",
        "worker_log": str(path),
        "logged_steps": len(records),
        "intervals": len(intervals),
        "measured_steps": measured_steps,
        "first_logged_step": records[0].completed_step,
        "last_logged_step": records[-1].completed_step,
        "mean_sec_per_step": mean(intervals),
        "median_sec_per_step": median(intervals),
        "p10_sec_per_step": percentile(intervals, 0.10),
        "p90_sec_per_step": percentile(intervals, 0.90),
    }


def maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: Any, digits: int = 3) -> str:
    number = maybe_float(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def main() -> None:
    args = parse_args()
    if not args.manifest and not args.log:
        raise SystemExit("provide --manifest or at least one --log")

    rows = read_manifest(args.manifest) if args.manifest else rows_from_logs(args.log)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        summary = summarize_log(Path(row["worker_log"]), args.warmup_steps, args.min_intervals)
        merged = {**row, **summary}
        global_batch = maybe_float(merged.get("global_batch"))
        mean_sec = maybe_float(merged.get("mean_sec_per_step"))
        median_sec = maybe_float(merged.get("median_sec_per_step"))
        merged["mean_samples_per_sec"] = global_batch / mean_sec if global_batch and mean_sec else None
        merged["median_samples_per_sec"] = global_batch / median_sec if global_batch and median_sec else None
        out_rows.append(merged)

    fieldnames = [
        "status",
        "topology",
        "nodes",
        "local_gpus",
        "world_size",
        "global_batch",
        "batch_size_per_gpu",
        "deterministic",
        "stage_pretrain_data",
        "epochs",
        "seed",
        "save_name",
        "logged_steps",
        "intervals",
        "measured_steps",
        "timing_source",
        "mean_sec_per_step",
        "median_sec_per_step",
        "p10_sec_per_step",
        "p90_sec_per_step",
        "mean_samples_per_sec",
        "median_samples_per_sec",
        "worker_log",
    ]

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(out_rows)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w") as f:
            json.dump(out_rows, f, indent=2)

    lines = [
        "# ABCI3 Pretrain Speed Benchmark",
        "",
        f"Warmup exclusion: intervals starting before completed step {args.warmup_steps} are omitted.",
        "",
        "| status | topology | global batch | deterministic | staging | source | mean sec/step | median sec/step | mean samples/s | measured steps |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in out_rows:
        lines.append(
            "| {status} | {topology} | {gb} | {det} | {stage} | {source} | {mean_s} | {median_s} | {samples} | {steps} |".format(
                status=row.get("status", ""),
                topology=row.get("topology", ""),
                gb=row.get("global_batch", ""),
                det=row.get("deterministic", ""),
                stage=row.get("stage_pretrain_data", ""),
                source=row.get("timing_source", ""),
                mean_s=fmt(row.get("mean_sec_per_step")),
                median_s=fmt(row.get("median_sec_per_step")),
                samples=fmt(row.get("mean_samples_per_sec"), 2),
                steps=row.get("measured_steps", "NA"),
            )
        )
    md = "\n".join(lines) + "\n"
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(md)
    print(md, end="")


if __name__ == "__main__":
    main()
