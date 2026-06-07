#!/usr/bin/env python3
"""Decoder skip shortcut diagnostic for visibility-gated NeRF-MAE.

This is a no-training diagnostic.  It loads one or more MAE checkpoints and
reruns the reconstruction decoder while perturbing only decoder skip features:

- normal
- masked skip locations zeroed
- visible skip locations zeroed
- all skip locations zeroed

It also optionally records gradient norm ratios on the normal reconstruction
loss for masked vs visible skip locations.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
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


def pool_mask_to_shape(mask_patches: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
    target_h, target_w, target_d = [int(x) for x in shape]
    mask = mask_patches[..., 0].to(dtype=torch.float32)
    if tuple(mask.shape[-3:]) == (target_h, target_w, target_d):
        return mask > 0.5
    x = mask[:, None]
    h, w, d = x.shape[-3:]
    if h % target_h == 0 and w % target_w == 0 and d % target_d == 0:
        kernel = (h // target_h, w // target_w, d // target_d)
        x = F.max_pool3d(x, kernel_size=kernel, stride=kernel)
        return x[:, 0, :target_h, :target_w, :target_d] > 0.5
    x = F.interpolate(x, size=(target_h, target_w, target_d), mode="nearest")
    return x[:, 0] > 0.5


def gate_skip(feature: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "normal":
        return feature
    if mode == "masked_zero":
        return feature * (~mask[:, None]).to(feature.dtype)
    if mode == "visible_zero":
        return feature * mask[:, None].to(feature.dtype)
    if mode == "all_zero":
        return torch.zeros_like(feature)
    raise ValueError(f"unknown skip mode {mode!r}")


def forward_with_skip_mode(model: torch.nn.Module, x: torch.Tensor, mode: str, retain_skip_grad: bool = False):
    tokens = model.patch_partition(x)
    tokens = tokens + model.pos_embed.type_as(tokens).to(tokens.device).clone().detach()
    tokens, mask_patches = model.window_masking_3d(
        tokens, p_remove=model.masking_prob, mask_token=model.mask_token
    )
    features: List[torch.Tensor] = []
    stage_masks: List[torch.Tensor] = []
    z = tokens
    for stage in model.stages:
        z = stage(z)
        feature = torch.permute(z, [0, 4, 1, 2, 3]).contiguous()
        if retain_skip_grad:
            feature.retain_grad()
        features.append(feature)
        stage_masks.append(pool_mask_to_shape(mask_patches, tuple(feature.shape[-3:])).to(feature.device))

    # decoder4 uses stage2 as skip, decoder3 stage1, decoder2 stage0.
    skip2 = gate_skip(features[2], stage_masks[2], mode)
    skip1 = gate_skip(features[1], stage_masks[1], mode)
    skip0 = gate_skip(features[0], stage_masks[0], mode)
    dec3 = model.decoder4(features[3], skip2)
    dec2 = model.decoder3(dec3, skip1)
    dec1 = model.decoder2(dec2, skip0)
    dec0 = model.decoder1(dec1)
    out = model.out(dec0)
    return out, mask_patches, features, stage_masks


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def compute_loss_metrics(model: torch.nn.Module, target: torch.Tensor, pred: torch.Tensor, mask_batch: torch.Tensor, mask_patches: torch.Tensor) -> Dict[str, float]:
    target_p, mask_p = model.patchify_3d(target, mask_batch)
    pred_p = model.patchify_3d(pred)
    removed = (mask_p.squeeze(-1).int() * mask_patches).unsqueeze(-1).to(pred_p.dtype)
    visible = (mask_p.squeeze(-1).int() * (1 - mask_patches.int())).unsqueeze(-1).to(pred_p.dtype)
    target_rgb = target_p[..., :3]
    target_alpha = target_p[..., 3].unsqueeze(-1)
    pred_rgb = pred_p[..., :3]
    pred_alpha = model.alpha_activation(pred_p[..., 3].unsqueeze(-1))
    occupied = (target_alpha > 0.01).to(pred_p.dtype)

    rgb_mse = (pred_rgb - target_rgb) ** 2
    alpha_mse = (pred_alpha - target_alpha) ** 2
    rgb_occ = occupied
    rgb_removed_occ = occupied * removed
    rgb_visible_occ = occupied * visible

    loss_rgb_occupied = masked_mean(rgb_mse, rgb_occ)
    loss_rgb_removed_occupied = masked_mean(rgb_mse, rgb_removed_occ)
    loss_rgb_visible_occupied = masked_mean(rgb_mse, rgb_visible_occ)
    loss_alpha_removed = masked_mean(alpha_mse, removed)
    loss = loss_rgb_occupied + loss_alpha_removed
    return {
        "loss": float(loss.detach().cpu()),
        "loss_rgb_occupied": float(loss_rgb_occupied.detach().cpu()),
        "loss_rgb_removed_occupied": float(loss_rgb_removed_occupied.detach().cpu()),
        "loss_rgb_visible_occupied": float(loss_rgb_visible_occupied.detach().cpu()),
        "loss_alpha_removed": float(loss_alpha_removed.detach().cpu()),
        "removed_count": float(removed.sum().detach().cpu()),
        "visible_count": float(visible.sum().detach().cpu()),
        "occupied_count": float(occupied.sum().detach().cpu()),
    }


def prepare_target(model: torch.nn.Module, volume: torch.Tensor, device: torch.device):
    padded, masks = model.transform([volume])
    target = torch.cat(tuple(padded), dim=0).to(device)
    mask_batch = torch.cat(tuple(masks), dim=0).to(device)
    return target, mask_batch


@torch.no_grad()
def run_perturbation_scene(model: torch.nn.Module, volume: torch.Tensor, scene: str, device: torch.device, seed: int) -> List[Dict[str, object]]:
    random.seed(seed)
    torch.manual_seed(seed)
    target, mask_batch = prepare_target(model, volume, device)
    rows: List[Dict[str, object]] = []
    for mode in ("normal", "masked_zero", "visible_zero", "all_zero"):
        pred, mask_patches, _, _ = forward_with_skip_mode(model, target, mode=mode, retain_skip_grad=False)
        metrics = compute_loss_metrics(model, target, pred, mask_batch, mask_patches)
        rows.append({"scene": scene, "mode": mode, **metrics})
    return rows


def grad_ratios_for_scene(model: torch.nn.Module, volume: torch.Tensor, scene: str, device: torch.device, seed: int) -> List[Dict[str, object]]:
    random.seed(seed)
    torch.manual_seed(seed)
    model.zero_grad(set_to_none=True)
    target, mask_batch = prepare_target(model, volume, device)
    pred, mask_patches, features, stage_masks = forward_with_skip_mode(
        model, target, mode="normal", retain_skip_grad=True
    )
    metrics = compute_loss_metrics(model, target, pred, mask_batch, mask_patches)
    loss = torch.as_tensor(metrics["loss"], device=device)
    # Recompute differentiable public objective for backward.
    target_p, mask_p = model.patchify_3d(target, mask_batch)
    pred_p = model.patchify_3d(pred)
    removed = (mask_p.squeeze(-1).int() * mask_patches).unsqueeze(-1).to(pred_p.dtype)
    target_rgb = target_p[..., :3]
    target_alpha = target_p[..., 3].unsqueeze(-1)
    pred_rgb = pred_p[..., :3]
    pred_alpha = model.alpha_activation(pred_p[..., 3].unsqueeze(-1))
    occupied = (target_alpha > 0.01).to(pred_p.dtype)
    diff_rgb = (pred_rgb - target_rgb) ** 2
    diff_alpha = (pred_alpha - target_alpha) ** 2
    loss = masked_mean(diff_rgb, occupied) + masked_mean(diff_alpha, removed)
    loss.backward()
    rows: List[Dict[str, object]] = []
    for stage_idx in (0, 1, 2):
        grad = features[stage_idx].grad
        if grad is None:
            continue
        mask = stage_masks[stage_idx]
        norm = grad.float().pow(2).sum(dim=1).sqrt()
        masked = mask
        visible = ~mask
        masked_mean = norm[masked].mean() if masked.any() else torch.tensor(float("nan"), device=device)
        visible_mean = norm[visible].mean() if visible.any() else torch.tensor(float("nan"), device=device)
        rows.append(
            {
                "scene": scene,
                "stage": stage_idx,
                "masked_grad_mean": float(masked_mean.detach().cpu()),
                "visible_grad_mean": float(visible_mean.detach().cpu()),
                "masked_visible_grad_ratio": float((masked_mean / visible_mean.clamp_min(1e-12)).detach().cpu()),
                "masked_count": int(masked.sum().detach().cpu()),
                "visible_count": int(visible.sum().detach().cpu()),
                "loss": float(loss.detach().cpu()),
            }
        )
    model.zero_grad(set_to_none=True)
    return rows


def mean_rows(rows: List[Dict[str, object]], keys: List[str], group_keys: List[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    out = []
    for group, items in sorted(groups.items(), key=lambda kv: kv[0]):
        dst = {k: v for k, v in zip(group_keys, group)}
        for key in keys:
            vals = [float(item[key]) for item in items if key in item and np.isfinite(float(item[key]))]
            dst[key] = float(np.mean(vals)) if vals else float("nan")
        dst["n"] = len(items)
        out.append(dst)
    return out


def write_markdown(path: Path, payload: Dict[str, object]) -> None:
    lines = ["# Masked Skip Shortcut Diagnostic", ""]
    lines.append("## Protocol")
    lines.append("")
    for key in ("features_dir", "split_file", "split_key", "scenes", "resolution", "masking_prob", "normalize_density"):
        lines.append(f"- `{key}`: `{payload.get(key)}`")
    lines.append("")
    lines.append("Modes:")
    lines.append("- `normal`: original decoder skips")
    lines.append("- `masked_zero`: zero masked-position decoder skip features only")
    lines.append("- `visible_zero`: zero visible-position decoder skip features only")
    lines.append("- `all_zero`: zero all decoder skip features")
    lines.append("")
    for label, result in payload["results"].items():
        lines.append(f"## {label}")
        load = result["load"]
        lines.append("")
        lines.append(f"- checkpoint: `{load.get('checkpoint')}`")
        lines.append(f"- loaded keys: `{load.get('loaded_keys')}/{load.get('total_model_keys')}`")
        lines.append(f"- missing keys: `{len(load.get('missing', []))}`; unexpected keys: `{len(load.get('unexpected', []))}`")
        lines.append("")
        lines.append("| mode | n | loss | RGB occupied | RGB removed occupied | RGB visible occupied | alpha removed |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        normal = None
        for row in result["loss_summary"]:
            if row["mode"] == "normal":
                normal = row
            lines.append(
                "| {mode} | {n} | {loss:.6f} | {rgb_occ:.6f} | {rgb_rem:.6f} | {rgb_vis:.6f} | {alpha:.6f} |".format(
                    mode=row["mode"],
                    n=int(row["n"]),
                    loss=row["loss"],
                    rgb_occ=row["loss_rgb_occupied"],
                    rgb_rem=row["loss_rgb_removed_occupied"],
                    rgb_vis=row["loss_rgb_visible_occupied"],
                    alpha=row["loss_alpha_removed"],
                )
            )
        if normal:
            lines.append("")
            lines.append("| mode | loss delta vs normal | RGB removed occupied delta | alpha removed delta |")
            lines.append("|---|---:|---:|---:|")
            for row in result["loss_summary"]:
                lines.append(
                    "| {mode} | {dloss:.6f} | {drgb:.6f} | {dalpha:.6f} |".format(
                        mode=row["mode"],
                        dloss=row["loss"] - normal["loss"],
                        drgb=row["loss_rgb_removed_occupied"] - normal["loss_rgb_removed_occupied"],
                        dalpha=row["loss_alpha_removed"] - normal["loss_alpha_removed"],
                    )
                )
        if result.get("grad_summary"):
            lines.append("")
            lines.append("| stage | n | masked grad mean | visible grad mean | masked/visible grad |")
            lines.append("|---|---:|---:|---:|---:|")
            for row in result["grad_summary"]:
                lines.append(
                    "| stage{stage} | {n} | {masked:.6e} | {visible:.6e} | {ratio:.4f} |".format(
                        stage=int(row["stage"]),
                        n=int(row["n"]),
                        masked=row["masked_grad_mean"],
                        visible=row["visible_grad_mean"],
                        ratio=row["masked_visible_grad_ratio"],
                    )
                )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--features-dir", type=Path, default=Path("dataset/_downloads/front3d_rpn_extract/front3d_rpn_data/features"))
    parser.add_argument("--split-file", type=Path, default=Path("dataset/_downloads/front3d_rpn_extract/front3d_rpn_data/3dfront_split.npz"))
    parser.add_argument("--split-key", default="val_scenes")
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--max-scenes", type=int, default=2)
    parser.add_argument("--grad-scenes", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--masking-prob", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--normalize-density", action="store_true", default=True)
    parser.add_argument("--no-normalize-density", dest="normalize_density", action="store_false")
    parser.add_argument("--out-json", type=Path, default=Path("results/shortcut_probe_artifacts/visibility/masked_skip_shortcut_diagnostic.json"))
    parser.add_argument("--out-md", type=Path, default=Path("results/shortcut_probe_artifacts/visibility/masked_skip_shortcut_diagnostic.md"))
    parser.add_argument("--loss-csv", type=Path, default=Path("results/shortcut_probe_artifacts/visibility/masked_skip_shortcut_loss.csv"))
    parser.add_argument("--grad-csv", type=Path, default=Path("results/shortcut_probe_artifacts/visibility/masked_skip_shortcut_grad.csv"))
    args = parser.parse_args()

    if not args.checkpoint:
        args.checkpoint = ["random_init"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    add_repo_to_path(args.repo)
    scenes = select_scenes(args.features_dir, args.split_file if args.split_file else None, args.split_key, args.max_scenes)
    specs = parse_checkpoint_specs(args.checkpoint)

    payload: Dict[str, object] = {
        "script": "nerf_mae/probe_scripts/masked_skip_shortcut_diagnostic.py",
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
    flat_loss_rows: List[Dict[str, object]] = []
    flat_grad_rows: List[Dict[str, object]] = []

    for label, checkpoint_path in specs:
        model = build_model(args.repo, args.resolution, args.masking_prob, device)
        load_info = load_checkpoint(model, checkpoint_path)
        loss_rows: List[Dict[str, object]] = []
        grad_rows: List[Dict[str, object]] = []
        for scene_idx, scene in enumerate(scenes):
            volume = load_volume(args.features_dir, scene, args.normalize_density)
            scene_loss_rows = run_perturbation_scene(model, volume, scene, device, args.seed + scene_idx)
            for row in scene_loss_rows:
                row = {"checkpoint": label, **row}
                loss_rows.append(row)
                flat_loss_rows.append(row)
            if scene_idx < args.grad_scenes:
                scene_grad_rows = grad_ratios_for_scene(model, volume, scene, device, args.seed + scene_idx)
                for row in scene_grad_rows:
                    row = {"checkpoint": label, **row}
                    grad_rows.append(row)
                    flat_grad_rows.append(row)

        payload["results"][label] = {
            "load": load_info,
            "loss_rows": loss_rows,
            "loss_summary": mean_rows(
                loss_rows,
                [
                    "loss",
                    "loss_rgb_occupied",
                    "loss_rgb_removed_occupied",
                    "loss_rgb_visible_occupied",
                    "loss_alpha_removed",
                    "removed_count",
                    "visible_count",
                    "occupied_count",
                ],
                ["mode"],
            ),
            "grad_rows": grad_rows,
            "grad_summary": mean_rows(
                grad_rows,
                ["masked_grad_mean", "visible_grad_mean", "masked_visible_grad_ratio", "masked_count", "visible_count", "loss"],
                ["stage"],
            ),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_markdown(args.out_md, payload)
    write_csv(
        args.loss_csv,
        flat_loss_rows,
        [
            "checkpoint",
            "scene",
            "mode",
            "loss",
            "loss_rgb_occupied",
            "loss_rgb_removed_occupied",
            "loss_rgb_visible_occupied",
            "loss_alpha_removed",
            "removed_count",
            "visible_count",
            "occupied_count",
        ],
    )
    write_csv(
        args.grad_csv,
        flat_grad_rows,
        [
            "checkpoint",
            "scene",
            "stage",
            "masked_grad_mean",
            "visible_grad_mean",
            "masked_visible_grad_ratio",
            "masked_count",
            "visible_count",
            "loss",
        ],
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.loss_csv}")
    print(f"Wrote {args.grad_csv}")


if __name__ == "__main__":
    main()
