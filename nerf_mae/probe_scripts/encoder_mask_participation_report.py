#!/usr/bin/env python3
"""Measure masked-token participation in the NeRF-MAE encoder.

This is a feasibility gate for visibility-gated / visible-token methods.  It does
not change training.  Given one or more checkpoints and a small set of real
volumes, it runs the MAE encoder path with normal masking and reports:

- masked/visible feature-norm ratio at each Swin stage;
- occupied and empty-region variants of the same ratio;
- patch-merging group statistics, especially mixed visible+masked groups;
- decoder skip feature norm ratios, using the same stage features.

The go/no-go question is whether masked placeholders still participate strongly
enough in early encoder/skip features to justify a Visibility-Gated method.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def add_repo_to_path(repo: Path) -> None:
    repo = repo.resolve()
    for path in (repo, repo / "nerf_mae"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def density_to_alpha(density: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


def load_volume(features_dir: Path, scene: str, normalize_density: bool) -> torch.Tensor:
    path = features_dir / f"{scene}.npz"
    with np.load(path) as data:
        rgbsigma = data["rgbsigma"].astype(np.float32, copy=True)
    if normalize_density:
        rgbsigma[..., -1] = density_to_alpha(rgbsigma[..., -1])
    rgbsigma = np.transpose(rgbsigma, (3, 0, 1, 2))
    tensor = torch.from_numpy(rgbsigma)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    return tensor.float().contiguous()


def parse_checkpoint_specs(values: Iterable[str]) -> List[Tuple[str, Optional[Path]]]:
    specs: List[Tuple[str, Optional[Path]]] = []
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
            specs.append((label.strip(), Path(path).expanduser().resolve()))
        elif value == "random_init":
            specs.append(("random_init", None))
        else:
            path = Path(value).expanduser().resolve()
            specs.append((path.parent.name, path))
    return specs


def select_scenes(features_dir: Path, split_file: Optional[Path], split_key: str, max_scenes: int) -> List[str]:
    scenes: List[str] = []
    if split_file is not None and split_file.exists():
        with np.load(split_file, allow_pickle=True) as split:
            if split_key not in split:
                raise KeyError(f"{split_key!r} not found in {split_file}; keys={list(split.keys())}")
            scenes = [str(item) for item in split[split_key].tolist()]
    else:
        scenes = [path.stem for path in sorted(features_dir.glob("*.npz"))]

    existing = [scene for scene in scenes if (features_dir / f"{scene}.npz").exists()]
    if max_scenes > 0:
        existing = existing[:max_scenes]
    if not existing:
        raise RuntimeError(f"no feature files found under {features_dir}")
    return existing


def build_model(repo: Path, resolution: int, masking_prob: float, device: torch.device):
    add_repo_to_path(repo)
    from model.mae.shortcut_probe import SwinTransformer_MAE3D_Probe

    model = SwinTransformer_MAE3D_Probe(
        patch_size=[4, 4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[4, 4, 4],
        stochastic_depth_prob=0.1,
        expand_dim=True,
        masking_prob=masking_prob,
        resolution=resolution,
        masking_strategy=None,
        probe_mode="baseline",
    )
    return model.to(device).eval()


def load_checkpoint(model: torch.nn.Module, path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {
            "checkpoint": "random_init",
            "missing": [],
            "unexpected": [],
            "loaded_keys": 0,
            "total_model_keys": len(model.state_dict()),
        }
    # Local experiment checkpoints include train args / numpy scalar metadata.
    # PyTorch 2.6 defaults to weights_only=True, which rejects that metadata.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    clean = {}
    for key, value in state_dict.items():
        clean[key[7:] if key.startswith("module.") else key] = value
    result = model.load_state_dict(clean, strict=False)
    model_keys = set(model.state_dict().keys())
    loaded = sum(1 for key in clean if key in model_keys)
    return {
        "checkpoint": str(path),
        "epoch": int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1,
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "loaded_keys": loaded,
        "total_model_keys": len(model_keys),
    }


def pad_to_resolution(x: torch.Tensor, resolution: int) -> torch.Tensor:
    c, h, w, d = x.shape
    out = torch.zeros((1, c, resolution, resolution, resolution), dtype=x.dtype)
    hh, ww, dd = min(h, resolution), min(w, resolution), min(d, resolution)
    out[:, :, :hh, :ww, :dd] = x[None, :, :hh, :ww, :dd]
    return out


def downsample_mask(
    mask_b1hwd: torch.Tensor,
    target_shape: Tuple[int, int, int],
    threshold: float = 0.5,
) -> torch.Tensor:
    # mask_b1hwd: [B, 1, H, W, D]; output bool [B, Ht, Wt, Dt].
    h, w, d = mask_b1hwd.shape[-3:]
    th, tw, td = target_shape
    if (h, w, d) == (th, tw, td):
        return mask_b1hwd[:, 0] > threshold
    kh, kw, kd = max(h // th, 1), max(w // tw, 1), max(d // td, 1)
    pooled = F.max_pool3d(mask_b1hwd.float(), kernel_size=(kh, kw, kd), stride=(kh, kw, kd))
    return pooled[:, 0, :th, :tw, :td] > threshold


def pool_stage_masks(mask_stage0: torch.Tensor, num_stages: int = 4) -> List[torch.Tensor]:
    masks = [mask_stage0]
    current = mask_stage0[:, None].float()
    for _ in range(1, num_stages):
        current = F.max_pool3d(current, kernel_size=2, stride=2)
        masks.append(current[:, 0] > 0.5)
    return masks


def merge_stats(mask_stage: torch.Tensor) -> Dict[str, float]:
    # mask_stage: bool [B,H,W,D]. Groups are 2x2x2 children before PatchMerging.
    b, h, w, d = mask_stage.shape
    hh, ww, dd = h - h % 2, w - w % 2, d - d % 2
    if min(hh, ww, dd) <= 0:
        return {"groups": 0, "mixed_ratio": float("nan"), "all_masked_ratio": float("nan"), "all_visible_ratio": float("nan")}
    x = mask_stage[:, :hh, :ww, :dd].reshape(b, hh // 2, 2, ww // 2, 2, dd // 2, 2)
    count = x.float().sum(dim=(2, 4, 6))
    groups = count.numel()
    mixed = ((count > 0) & (count < 8)).float().mean().item()
    all_masked = (count == 8).float().mean().item()
    all_visible = (count == 0).float().mean().item()
    return {
        "groups": int(groups),
        "mixed_ratio": mixed,
        "all_masked_ratio": all_masked,
        "all_visible_ratio": all_visible,
    }


def masked_visible_norm_ratio(
    feature: torch.Tensor,
    mask: torch.Tensor,
    region: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    # feature: [B,C,H,W,D], mask/region: [B,H,W,D].
    norm = feature.float().pow(2).sum(dim=1).sqrt()
    if region is None:
        region = torch.ones_like(mask, dtype=torch.bool)
    masked = mask & region
    visible = (~mask) & region
    if masked.sum() == 0 or visible.sum() == 0:
        return {
            "masked_mean": float("nan"),
            "visible_mean": float("nan"),
            "masked_visible_ratio": float("nan"),
            "masked_count": int(masked.sum().item()),
            "visible_count": int(visible.sum().item()),
        }
    masked_mean = norm[masked].mean().item()
    visible_mean = norm[visible].mean().item()
    return {
        "masked_mean": masked_mean,
        "visible_mean": visible_mean,
        "masked_visible_ratio": masked_mean / max(visible_mean, 1e-12),
        "masked_count": int(masked.sum().item()),
        "visible_count": int(visible.sum().item()),
    }


@dataclass
class SceneResult:
    scene: str
    stage_metrics: Dict[str, Dict[str, float]]
    merge_metrics: Dict[str, Dict[str, float]]
    mask_mean: float
    occupied_patch_frac: float


@torch.no_grad()
def run_one_scene(
    model: torch.nn.Module,
    volume: torch.Tensor,
    scene: str,
    resolution: int,
    device: torch.device,
    seed: int,
) -> SceneResult:
    random.seed(seed)
    torch.manual_seed(seed)
    x = pad_to_resolution(volume, resolution).to(device)
    alpha = x[:, -1:, :, :, :]

    # Mirror forward_encoder_ecoder but stop before decoder.
    tokens = model.patch_partition(x)
    tokens = tokens + model.pos_embed.type_as(tokens).to(tokens.device).clone().detach()
    masked_tokens, mask_patches = model.window_masking_3d(
        tokens, p_remove=model.masking_prob, mask_token=model.mask_token
    )
    mask_stage0 = mask_patches[..., 0].bool()
    stage_masks = pool_stage_masks(mask_stage0, num_stages=len(model.stages))
    occupied_stage0 = downsample_mask(alpha, tuple(mask_stage0.shape[-3:]), threshold=0.01)
    occupied_masks = pool_stage_masks(occupied_stage0, num_stages=len(model.stages))

    features: List[torch.Tensor] = []
    z = masked_tokens
    for stage in model.stages:
        z = stage(z)
        features.append(torch.permute(z, [0, 4, 1, 2, 3]).contiguous())

    stage_metrics: Dict[str, Dict[str, float]] = {}
    for idx, feature in enumerate(features):
        shape = tuple(feature.shape[-3:])
        mask = downsample_mask(stage_masks[idx][:, None].float(), shape)
        occ = downsample_mask(occupied_masks[idx][:, None].float(), shape)
        empty = ~occ
        all_metrics = masked_visible_norm_ratio(feature, mask)
        occ_metrics = masked_visible_norm_ratio(feature, mask, occ)
        empty_metrics = masked_visible_norm_ratio(feature, mask, empty)
        stage_metrics[f"stage{idx}"] = {
            **{f"all_{k}": v for k, v in all_metrics.items()},
            **{f"occupied_{k}": v for k, v in occ_metrics.items()},
            **{f"empty_{k}": v for k, v in empty_metrics.items()},
        }

    merge_metrics: Dict[str, Dict[str, float]] = {}
    for idx in range(len(stage_masks) - 1):
        merge_metrics[f"merge{idx}_to_{idx + 1}"] = merge_stats(stage_masks[idx])

    return SceneResult(
        scene=scene,
        stage_metrics=stage_metrics,
        merge_metrics=merge_metrics,
        mask_mean=float(mask_stage0.float().mean().item()),
        occupied_patch_frac=float(occupied_stage0.float().mean().item()),
    )


def mean_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({key for item in dicts for key in item})
    out: Dict[str, float] = {}
    for key in keys:
        vals = [item[key] for item in dicts if key in item and np.isfinite(item[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def aggregate_scene_results(results: List[SceneResult]) -> Dict[str, object]:
    stages = sorted(results[0].stage_metrics)
    merges = sorted(results[0].merge_metrics)
    return {
        "mask_mean": float(np.mean([item.mask_mean for item in results])),
        "occupied_patch_frac": float(np.mean([item.occupied_patch_frac for item in results])),
        "stages": {
            stage: mean_dicts([item.stage_metrics[stage] for item in results])
            for stage in stages
        },
        "merges": {
            merge: mean_dicts([item.merge_metrics[merge] for item in results])
            for merge in merges
        },
    }


def write_markdown(path: Path, payload: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Encoder Mask Participation Report")
    lines.append("")
    lines.append(f"Generated from `{payload['script']}`.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    for key in ("features_dir", "split_file", "split_key", "scenes", "resolution", "masking_prob", "normalize_density"):
        lines.append(f"- `{key}`: `{payload.get(key)}`")
    lines.append("")
    lines.append("Gate rule from the strategy feedback:")
    lines.append("- Go if stage0/1 masked-visible feature norm ratio is >= 0.25, or patch-merge mixed groups are high with persistent masked skip norms.")
    lines.append("- Attention-mass measurement is not included here because the current Swin attention function does not expose masks without an intrusive model patch; this report covers the non-invasive gates first.")
    lines.append("")

    for label, result in payload["results"].items():
        lines.append(f"## {label}")
        load = result["load"]
        lines.append("")
        lines.append(f"- checkpoint: `{load.get('checkpoint')}`")
        lines.append(f"- loaded keys: `{load.get('loaded_keys')}/{load.get('total_model_keys')}`")
        lines.append(f"- missing keys: `{len(load.get('missing', []))}`; unexpected keys: `{len(load.get('unexpected', []))}`")
        agg = result["aggregate"]
        lines.append(f"- mask mean: `{agg['mask_mean']:.4f}`")
        lines.append(f"- occupied patch fraction: `{agg['occupied_patch_frac']:.4f}`")
        lines.append("")
        lines.append("| stage | all mask/visible | occupied mask/visible | empty mask/visible | masked tokens | visible tokens |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for stage, metrics in agg["stages"].items():
            lines.append(
                "| {stage} | {all_ratio:.4f} | {occ_ratio:.4f} | {empty_ratio:.4f} | {masked} | {visible} |".format(
                    stage=stage,
                    all_ratio=metrics.get("all_masked_visible_ratio", float("nan")),
                    occ_ratio=metrics.get("occupied_masked_visible_ratio", float("nan")),
                    empty_ratio=metrics.get("empty_masked_visible_ratio", float("nan")),
                    masked=int(metrics.get("all_masked_count", 0)),
                    visible=int(metrics.get("all_visible_count", 0)),
                )
            )
        lines.append("")
        lines.append("| merge | mixed ratio | all masked | all visible | groups |")
        lines.append("|---|---:|---:|---:|---:|")
        for merge, metrics in agg["merges"].items():
            lines.append(
                "| {merge} | {mixed:.4f} | {masked:.4f} | {visible:.4f} | {groups} |".format(
                    merge=merge,
                    mixed=metrics.get("mixed_ratio", float("nan")),
                    masked=metrics.get("all_masked_ratio", float("nan")),
                    visible=metrics.get("all_visible_ratio", float("nan")),
                    groups=int(metrics.get("groups", 0)),
                )
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_csvs(
    payload: Dict[str, object],
    feature_csv: Path,
    merge_csv: Path,
    skip_csv: Path,
    attention_csv: Path,
) -> None:
    feature_csv.parent.mkdir(parents=True, exist_ok=True)

    with feature_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "stage",
                "region",
                "masked_visible_ratio",
                "masked_mean",
                "visible_mean",
                "masked_count",
                "visible_count",
            ],
        )
        writer.writeheader()
        for label, result in payload["results"].items():
            for stage, metrics in result["aggregate"]["stages"].items():
                for region in ("all", "occupied", "empty"):
                    writer.writerow(
                        {
                            "checkpoint": label,
                            "stage": stage,
                            "region": region,
                            "masked_visible_ratio": metrics.get(f"{region}_masked_visible_ratio"),
                            "masked_mean": metrics.get(f"{region}_masked_mean"),
                            "visible_mean": metrics.get(f"{region}_visible_mean"),
                            "masked_count": metrics.get(f"{region}_masked_count"),
                            "visible_count": metrics.get(f"{region}_visible_count"),
                        }
                    )

    with merge_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["checkpoint", "merge", "mixed_ratio", "all_masked_ratio", "all_visible_ratio", "groups"],
        )
        writer.writeheader()
        for label, result in payload["results"].items():
            for merge, metrics in result["aggregate"]["merges"].items():
                writer.writerow(
                    {
                        "checkpoint": label,
                        "merge": merge,
                        "mixed_ratio": metrics.get("mixed_ratio"),
                        "all_masked_ratio": metrics.get("all_masked_ratio"),
                        "all_visible_ratio": metrics.get("all_visible_ratio"),
                        "groups": metrics.get("groups"),
                    }
                )

    # Decoder skip features are stage0/1/2 in the current architecture:
    # decoder4(features[3], features[2]), decoder3(dec3, features[1]),
    # decoder2(dec2, features[0]).  Stage3 is the main decoder4 input, not a skip.
    with skip_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "skip",
                "source_stage",
                "all_masked_visible_ratio",
                "occupied_masked_visible_ratio",
                "empty_masked_visible_ratio",
            ],
        )
        writer.writeheader()
        skip_sources = {"stage0": "decoder2_skip", "stage1": "decoder3_skip", "stage2": "decoder4_skip"}
        for label, result in payload["results"].items():
            for stage, skip_name in skip_sources.items():
                metrics = result["aggregate"]["stages"][stage]
                writer.writerow(
                    {
                        "checkpoint": label,
                        "skip": skip_name,
                        "source_stage": stage,
                        "all_masked_visible_ratio": metrics.get("all_masked_visible_ratio"),
                        "occupied_masked_visible_ratio": metrics.get("occupied_masked_visible_ratio"),
                        "empty_masked_visible_ratio": metrics.get("empty_masked_visible_ratio"),
                    }
                )

    with attention_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["status", "reason", "next_step"])
        writer.writeheader()
        writer.writerow(
            {
                "status": "not_measured_non_intrusive_probe",
                "reason": "shifted_window_attention does not accept or expose a visibility mask/attention tensor in the current implementation",
                "next_step": "only implement attention-mass logging if feature/skip/merge gates are positive",
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--features-dir", type=Path, default=Path("dataset/_downloads/front3d_rpn_extract/front3d_rpn_data/features"))
    parser.add_argument("--split-file", type=Path, default=Path("dataset/_downloads/front3d_rpn_extract/front3d_rpn_data/3dfront_split.npz"))
    parser.add_argument("--split-key", default="val_scenes")
    parser.add_argument("--checkpoint", action="append", default=[], help="label=/path/to/epoch.pt, or random_init")
    parser.add_argument("--max-scenes", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--masking-prob", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--normalize-density", action="store_true", default=True)
    parser.add_argument("--no-normalize-density", dest="normalize_density", action="store_false")
    parser.add_argument("--out-json", type=Path, default=Path("results/shortcut_probe_artifacts/visibility/encoder_mask_participation_report.json"))
    parser.add_argument("--out-md", type=Path, default=Path("results/shortcut_probe_artifacts/visibility/encoder_mask_participation_report.md"))
    parser.add_argument("--feature-csv", type=Path, default=Path("results/shortcut_probe_artifacts/visibility/feature_norm_by_stage.csv"))
    parser.add_argument("--merge-csv", type=Path, default=Path("results/shortcut_probe_artifacts/visibility/patch_merge_mask_stats.csv"))
    parser.add_argument("--skip-csv", type=Path, default=Path("results/shortcut_probe_artifacts/visibility/skip_feature_mask_stats.csv"))
    parser.add_argument("--attention-csv", type=Path, default=Path("results/shortcut_probe_artifacts/visibility/attention_mass_by_block.csv"))
    args = parser.parse_args()

    if not args.checkpoint:
        args.checkpoint = ["random_init"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    scenes = select_scenes(args.features_dir, args.split_file if args.split_file else None, args.split_key, args.max_scenes)
    specs = parse_checkpoint_specs(args.checkpoint)

    payload: Dict[str, object] = {
        "script": "nerf_mae/probe_scripts/encoder_mask_participation_report.py",
        "features_dir": str(args.features_dir),
        "split_file": str(args.split_file),
        "split_key": args.split_key,
        "scenes": scenes,
        "resolution": args.resolution,
        "masking_prob": args.masking_prob,
        "normalize_density": bool(args.normalize_density),
        "device": str(device),
        "results": {},
    }

    for label, checkpoint_path in specs:
        model = build_model(args.repo, args.resolution, args.masking_prob, device)
        load_info = load_checkpoint(model, checkpoint_path)
        scene_results = []
        for scene_idx, scene in enumerate(scenes):
            volume = load_volume(args.features_dir, scene, args.normalize_density)
            scene_results.append(
                run_one_scene(
                    model=model,
                    volume=volume,
                    scene=scene,
                    resolution=args.resolution,
                    device=device,
                    seed=args.seed + scene_idx,
                )
            )
        payload["results"][label] = {
            "load": load_info,
            "scene_results": [
                {
                    "scene": item.scene,
                    "mask_mean": item.mask_mean,
                    "occupied_patch_frac": item.occupied_patch_frac,
                    "stage_metrics": item.stage_metrics,
                    "merge_metrics": item.merge_metrics,
                }
                for item in scene_results
            ],
            "aggregate": aggregate_scene_results(scene_results),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_markdown(args.out_md, payload)
    write_csvs(payload, args.feature_csv, args.merge_csv, args.skip_csv, args.attention_csv)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.feature_csv}")
    print(f"Wrote {args.merge_csv}")
    print(f"Wrote {args.skip_csv}")
    print(f"Wrote {args.attention_csv}")


if __name__ == "__main__":
    main()
