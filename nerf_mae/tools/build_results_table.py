#!/usr/bin/env python3
"""Build a single CSV index for shortcut-probe FCOS evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--eval-roots",
        type=Path,
        nargs="+",
        default=[
            Path("output/nerf_rpn/results"),
            Path("results/shortcut_probe_artifacts/eval"),
        ],
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/shortcut_probe_artifacts/results_table.csv"),
    )
    return parser.parse_args()


def git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def strip_eval_suffix(name: str) -> str:
    for suffix in ("_eval", "_diagnostics"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def infer_condition(pretrain_save_name: str, run_name: str) -> str:
    if run_name.startswith("front3d_scratch"):
        return "scratch"
    core = re.sub(r"_abci3.*$", "", pretrain_save_name)
    core = re.sub(r"_seed\d+$", "", core)
    core = re.sub(r"_p[0-9.]+_e\d+$", "", core)
    if core == "nerfmae_all":
        return "baseline"
    if core.startswith("nerfmae_alpha_rgba_curr_"):
        return core.removeprefix("nerfmae_alpha_rgba_curr_")
    if core.startswith("nerfmae_"):
        return core.removeprefix("nerfmae_")
    return core or "unknown"


def infer_protocol(pretrain_save_name: str, source: str) -> tuple[str, str]:
    if "abci3gb16_16g" in pretrain_save_name:
        return "16", "ABCI3_2n16g_gb16"
    if "abci3diag_opt1n8g_det0" in pretrain_save_name:
        return "16", "ABCI3_1n8g_gb16_det0"
    if "abci3sm_cj_det0_1n8g" in pretrain_save_name:
        return "16", "ABCI3_1n8g_gb16_det0_surface_maturation_coord_jitter"
    if "abci3input_cj_det0_1n8g" in pretrain_save_name:
        return "16", "ABCI3_1n8g_gb16_det0_input_alpha_coord_jitter"
    if "abci3pyr_cj_det0_1n8g" in pretrain_save_name:
        return "16", "ABCI3_1n8g_gb16_det0_pyramid_coord_jitter"
    if "abci3shuf_cj_det0_1n8g" in pretrain_save_name:
        return "16", "ABCI3_1n8g_gb16_det0_shuffle_coord_jitter"
    if "abci3clean" in pretrain_save_name:
        return "", "ABCI3_clean"
    if source == "shortcut_probe_artifacts":
        return "", "historical"
    return "", "unknown"


def infer_surface_env(condition: str) -> dict[str, str]:
    fields = {
        "SM_MODE": "",
        "SM_CONFIDENCE": "",
        "SM_W_MIN": "",
        "SM_TAU": "",
        "SM_K": "",
        "SM_STOP_GATE_GRAD": "",
        "SM_RGB_MASK": "",
        "SM_INPUT_RGB_CURRICULUM": "",
    }
    if condition.startswith("surface_maturation_tau") or condition == "input_alpha_curriculum":
        fields.update(
            {
                "SM_MODE": "surface_maturation",
                "SM_CONFIDENCE": "raw_alpha",
                "SM_W_MIN": "0.05",
                "SM_STOP_GATE_GRAD": "1",
                "SM_RGB_MASK": "removed_occupied",
                "SM_INPUT_RGB_CURRICULUM": "none",
            }
        )
    if condition == "surface_maturation_tau0p3_k10_w0p05":
        fields.update({"SM_TAU": "0.3", "SM_K": "10", "SM_W_MIN": "0.05"})
    elif condition == "surface_maturation_tau0p5_k20_w0p05":
        fields.update({"SM_TAU": "0.5", "SM_K": "20", "SM_W_MIN": "0.05"})
    elif condition == "surface_maturation_tau0p7_k30_w0p05":
        fields.update({"SM_TAU": "0.7", "SM_K": "30", "SM_W_MIN": "0.05"})
    elif condition == "input_alpha_curriculum":
        fields.update(
            {
                "SM_TAU": "0.5",
                "SM_K": "20",
                "SM_W_MIN": "0.05",
                "SM_INPUT_RGB_CURRICULUM": "cosine_release",
            }
        )
    return fields


def infer_pyramid_env(condition: str) -> dict[str, str]:
    fields = {
        "PYR_MODE": "",
        "PYR_SCALE": "",
        "PYR_SCHEDULE": "",
        "PYR_EPOCHS": "",
        "PYR_ALPHA_POOL": "",
        "PYR_RGB_POOL": "",
        "PYR_UPSAMPLE": "",
        "PYR_ALPHA_UPSAMPLE": "",
    }
    if condition in {"pyramid_alpha", "pyramid_rgb", "pyramid_both"}:
        mode = {
            "pyramid_alpha": "alpha",
            "pyramid_rgb": "rgb",
            "pyramid_both": "both",
        }[condition]
        fields.update(
            {
                "PYR_MODE": mode,
                "PYR_SCALE": "2",
                "PYR_SCHEDULE": "cosine",
                "PYR_EPOCHS": "epoch",
                "PYR_ALPHA_POOL": "max",
                "PYR_RGB_POOL": "avg",
                "PYR_UPSAMPLE": "trilinear",
                "PYR_ALPHA_UPSAMPLE": "nearest",
            }
        )
    return fields


def extract_metric(data: dict[str, Any], key: str, subkey: str) -> str:
    value = data.get(key, {})
    if isinstance(value, dict) and subkey in value:
        return f"{float(value[subkey]):.10g}"
    return ""


def find_checkpoint(root: Path, pretrain_save_name: str, epoch: str) -> str:
    if not pretrain_save_name or not epoch:
        return ""
    path = root / "output" / "nerf_mae" / "results" / pretrain_save_name / f"epoch_{epoch}.pt"
    return str(path.resolve()) if path.exists() else ""


def parse_run(root: Path, eval_json: Path, git_hash: str) -> dict[str, str]:
    run_name = eval_json.parent.name
    stem = strip_eval_suffix(run_name)
    source = (
        "shortcut_probe_artifacts"
        if "results/shortcut_probe_artifacts/eval" in eval_json.as_posix()
        else "output"
    )

    fcos_epochs = ""
    fcos_match = re.search(r"_fcos(\d+)", stem)
    if fcos_match:
        fcos_epochs = fcos_match.group(1)

    pretrain_save_name = ""
    epoch = ""
    scheduler = ""
    pretrain_seed = ""
    finetune_seed = ""

    epoch_match = re.search(r"_epoch(\d+)", stem)
    if epoch_match:
        pretrain_save_name = stem[: epoch_match.start()]
        epoch = epoch_match.group(1)
        rest = stem[epoch_match.end() :]
        sched_match = re.search(r"_sched_([^_]+)", rest)
        if sched_match:
            scheduler = "onecycle_epoch" if sched_match.group(1) == "epoch" else sched_match.group(1)
    else:
        fcos_part = re.search(r"_fcos\d+", stem)
        pretrain_save_name = stem[: fcos_part.start()] if fcos_part else stem
        epoch_match = re.search(r"_e(\d+)(?:_|$)", pretrain_save_name)
        epoch = epoch_match.group(1) if epoch_match else ""

    pre_seed_match = re.search(r"_seed(\d+)(?:_|$)", pretrain_save_name)
    if pre_seed_match:
        pretrain_seed = pre_seed_match.group(1)
    if run_name.startswith("front3d_scratch"):
        pretrain_seed = ""

    ft_pair = re.search(r"_preseed(\d+)_ftseed(\d+)_fcos", stem)
    if ft_pair:
        pretrain_seed = ft_pair.group(1)
        finetune_seed = ft_pair.group(2)
    else:
        ft_seed_match = re.search(r"(?:_pt\d+)?_seed(\d+)_fcos", stem)
        if ft_seed_match:
            finetune_seed = ft_seed_match.group(1)
        elif run_name.startswith("front3d_scratch"):
            scratch_seed = re.search(r"_seed(\d+)", stem)
            finetune_seed = scratch_seed.group(1) if scratch_seed else ""
    if not finetune_seed:
        finetune_seed = pretrain_seed

    condition = infer_condition(pretrain_save_name, run_name)
    global_batch, gpu_env = infer_protocol(pretrain_save_name, source)
    surface_env = infer_surface_env(condition)
    pyramid_env = infer_pyramid_env(condition)
    with eval_json.open() as f:
        data = json.load(f)

    row = {
        "condition": condition,
        "pretrain_seed": pretrain_seed,
        "finetune_seed": finetune_seed,
        "epoch": epoch,
        "dataset": "front3d",
        "scheduler": scheduler,
        "global_batch": global_batch,
        "GPU_env": gpu_env,
        "checkpoint_path": find_checkpoint(root, pretrain_save_name, epoch),
        "eval_path": str(eval_json.resolve()),
        "AP@50": extract_metric(data, "ap_50", "ap"),
        "AP@25": extract_metric(data, "ap_25", "ap"),
        "AP@75": extract_metric(data, "ap_75", "ap"),
        "R50@300": extract_metric(data, "recall_50_top_300", "ar"),
        "R25@300": extract_metric(data, "recall_25_top_300", "ar"),
        "fcos_epochs": fcos_epochs,
        "pretrain_save_name": pretrain_save_name,
        "run_name": run_name,
        "source": source,
        "git_hash": git_hash,
    }
    if pyramid_env.get("PYR_EPOCHS") == "epoch":
        pyramid_env["PYR_EPOCHS"] = epoch
    row.update(surface_env)
    row.update(pyramid_env)
    return row


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    eval_jsons: list[Path] = []
    for eval_root in args.eval_roots:
        base = eval_root if eval_root.is_absolute() else root / eval_root
        if base.exists():
            eval_jsons.extend(sorted(base.glob("**/eval.json")))

    current_git_hash = git_head(root)
    rows = [parse_run(root, path, current_git_hash) for path in eval_jsons]
    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["condition"],
            int(row["epoch"] or 0),
            int(row["pretrain_seed"] or 0),
            int(row["finetune_seed"] or 0),
            row["run_name"],
        )
    )

    fieldnames = [
        "condition",
        "pretrain_seed",
        "finetune_seed",
        "epoch",
        "dataset",
        "scheduler",
        "global_batch",
        "GPU_env",
        "checkpoint_path",
        "eval_path",
        "AP@50",
        "AP@25",
        "AP@75",
        "R50@300",
        "R25@300",
        "fcos_epochs",
        "pretrain_save_name",
        "run_name",
        "source",
        "git_hash",
        "SM_MODE",
        "SM_CONFIDENCE",
        "SM_W_MIN",
        "SM_TAU",
        "SM_K",
        "SM_STOP_GATE_GRAD",
        "SM_RGB_MASK",
        "SM_INPUT_RGB_CURRICULUM",
        "PYR_MODE",
        "PYR_SCALE",
        "PYR_SCHEDULE",
        "PYR_EPOCHS",
        "PYR_ALPHA_POOL",
        "PYR_RGB_POOL",
        "PYR_UPSAMPLE",
        "PYR_ALPHA_UPSAMPLE",
    ]
    out_csv = args.out_csv if args.out_csv.is_absolute() else root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[info] wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
