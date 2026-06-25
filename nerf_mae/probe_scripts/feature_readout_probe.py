#!/usr/bin/env python3
"""Frozen encoder readout probes for NeRF-MAE representations.

This script intentionally avoids detector heads and FPN necks. The FPN layers
used by NeRF-RPN/FCOS are randomly initialized during downstream training, so a
frozen representation probe should read the pretrained Swin encoder stages
directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerf_rpn.datasets import Front3DRPNDataset
from nerf_rpn.model.feature_extractor import (
    SwinTransformer_FPN_Pretrained_Skip,
    _pos_embed_like,
)


TARGETS = ("objectness", "occupancy", "shell")


@dataclass(frozen=True)
class Arm:
    name: str
    checkpoint: Optional[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("dataset/finetune/front3d_rpn_data"))
    parser.add_argument("--split", type=Path, default=Path("dataset/finetune/front3d_rpn_data/3dfront_split.npz"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/feature_readout_probe"))
    parser.add_argument("--percent_train", type=float, default=0.10)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--alpha_threshold", type=float, default=0.01)
    parser.add_argument("--shell_alpha_threshold", type=float, default=0.02)
    parser.add_argument("--shell_smooth_kernel", type=int, default=3)
    parser.add_argument("--arms", nargs="*", default=["preset_main"])
    parser.add_argument("--stages", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--targets", nargs="*", default=list(TARGETS), choices=TARGETS)
    parser.add_argument("--samples_per_class", type=int, default=512)
    parser.add_argument("--eval_samples_per_class", type=int, default=2048)
    parser.add_argument("--readout_epochs", type=int, default=40)
    parser.add_argument("--readout_batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_train_scenes", type=int, default=0, help="Debug cap; 0 means no cap.")
    parser.add_argument("--max_val_scenes", type=int, default=0, help="Debug cap; 0 means no cap.")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def preset_arms(root: Path) -> Dict[str, Arm]:
    """Known checkpoints used in the current paper draft.

    Missing paths are filtered later, except for scratch.
    """
    base = root / "output/nerf_mae/results"
    specs = {
        "scratch": None,
        "coord_only": None,
        "joint_e300": base / "nerfmae_all_p1.0_e300_seed1" / "epoch_300.pt",
        "cosine_e300": base / "nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1" / "epoch_300.pt",
        "linear_e300": base / "nerfmae_alpha_rgba_curr_linear_ramp_p1.0_e300_seed1_abci3linear_det0" / "epoch_300.pt",
        "w05_e300": base / "nerfmae_alpha_rgba_curr_constant_rgb_half_p1.0_e300_seed1_abci3w05" / "epoch_300.pt",
        "occupancy_only_e300": base / "nerfmae_alpha_target_only_p1.0_e300_seed1_abci3ato_det0" / "epoch_300.pt",
        "shuffle_e300": base / "nerfmae_alpha_rgba_curr_cosine_ramp_alpha_shuffle_p1.0_e300_seed1" / "epoch_300.pt",
    }
    return {name: Arm(name, ckpt) for name, ckpt in specs.items()}


def resolve_arms(tokens: Sequence[str], root: Path) -> List[Arm]:
    presets = preset_arms(root)
    names: List[str] = []
    for token in tokens:
        if token == "preset_main":
            names.extend(["scratch", "joint_e300", "cosine_e300", "linear_e300", "w05_e300", "occupancy_only_e300", "shuffle_e300"])
        elif token == "preset_with_coord":
            names.extend(["coord_only", "scratch", "joint_e300", "cosine_e300", "linear_e300", "w05_e300", "occupancy_only_e300", "shuffle_e300"])
        elif "=" in token:
            name, value = token.split("=", 1)
            ckpt = Path(value)
            if not ckpt.is_absolute():
                ckpt = root / ckpt
            presets[name] = Arm(name, ckpt)
            names.append(name)
        else:
            names.append(token)

    arms: List[Arm] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        if name not in presets:
            raise KeyError(f"Unknown arm {name!r}. Use name=checkpoint_path for custom arms.")
        arm = presets[name]
        if arm.checkpoint is not None and not arm.checkpoint.exists():
            print(f"[warn] skip missing checkpoint for {name}: {arm.checkpoint}")
            continue
        arms.append(arm)
    return arms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_to_resolution(x: torch.Tensor, resolution: int) -> torch.Tensor:
    if x.shape[-3] > resolution or x.shape[-2] > resolution or x.shape[-1] > resolution:
        raise ValueError(f"input shape {tuple(x.shape)} exceeds resolution={resolution}")
    return F.pad(
        x,
        (
            0,
            resolution - x.shape[-1],
            0,
            resolution - x.shape[-2],
            0,
            resolution - x.shape[-3],
        ),
        mode="constant",
        value=0,
    )


def build_backbone(arm: Arm, resolution: int, device: torch.device) -> SwinTransformer_FPN_Pretrained_Skip:
    model = SwinTransformer_FPN_Pretrained_Skip(
        resolution=resolution,
        checkpoint_path=str(arm.checkpoint) if arm.checkpoint is not None else None,
        is_eval=arm.checkpoint is None,
    )
    # The FPN neck is intentionally unused, but keep module ownership simple.
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_stage_features(model: SwinTransformer_FPN_Pretrained_Skip, x: torch.Tensor, stages: Sequence[int]) -> Dict[int, torch.Tensor]:
    """Return encoder stage features as [C, W, L, H] tensors on CPU."""
    max_stage = max(stages)
    feats: Dict[int, torch.Tensor] = {}
    z = model.base.patch_partition(x)
    z = z + _pos_embed_like(model.base.pos_embed, z)
    for i in range(len(model.base.stages)):
        z = model.base.stages[i](z)
        if i in stages:
            feats[i] = torch.permute(z, [0, 4, 1, 2, 3])[0].contiguous().cpu()
        if i >= max_stage:
            break
    return feats


def valid_mask_for_feature(feature_shape: Tuple[int, int, int], original_shape: Tuple[int, int, int], resolution: int) -> torch.Tensor:
    fw, fl, fh = feature_shape
    ow, ol, oh = original_shape
    sx, sy, sz = resolution / fw, resolution / fl, resolution / fh
    xs = (torch.arange(fw, dtype=torch.float32) + 0.5) * sx
    ys = (torch.arange(fl, dtype=torch.float32) + 0.5) * sy
    zs = (torch.arange(fh, dtype=torch.float32) + 0.5) * sz
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    return (gx < ow) & (gy < ol) & (gz < oh)


def coordinate_features(feature_shape: Tuple[int, int, int], resolution: int) -> torch.Tensor:
    """Normalized [x,y,z] coordinate features for a feature-grid arm."""
    fw, fl, fh = feature_shape
    sx, sy, sz = resolution / fw, resolution / fl, resolution / fh
    xs = ((torch.arange(fw, dtype=torch.float32) + 0.5) * sx) / float(resolution)
    ys = ((torch.arange(fl, dtype=torch.float32) + 0.5) * sy) / float(resolution)
    zs = ((torch.arange(fh, dtype=torch.float32) + 0.5) * sz) / float(resolution)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    return torch.stack([gx, gy, gz], dim=0)


def downsample_occupancy(alpha: torch.Tensor, feature_shape: Tuple[int, int, int], threshold: float) -> torch.Tensor:
    mask = (alpha > threshold).float()[None, None]
    pooled = F.adaptive_max_pool3d(mask, feature_shape)[0, 0]
    return pooled > 0.5


def shell_from_occupancy(occ: torch.Tensor) -> torch.Tensor:
    occ_f = occ.float()[None, None]
    eroded = -F.max_pool3d(-occ_f, kernel_size=3, stride=1, padding=1)[0, 0]
    return occ & (eroded < 0.5)


def downsample_shell(alpha: torch.Tensor, feature_shape: Tuple[int, int, int], threshold: float, smooth_kernel: int) -> torch.Tensor:
    """Build a denoised shell target from alpha and downsample it to a feature grid."""
    if smooth_kernel < 1 or smooth_kernel % 2 == 0:
        raise ValueError("--shell_smooth_kernel must be a positive odd integer")
    alpha_f = alpha.float()[None, None]
    if smooth_kernel > 1:
        alpha_f = F.avg_pool3d(alpha_f, kernel_size=smooth_kernel, stride=1, padding=smooth_kernel // 2)
    occ_f = (alpha_f > threshold).float()
    eroded = -F.max_pool3d(-occ_f, kernel_size=3, stride=1, padding=1)
    shell = (occ_f > 0.5) & (eroded < 0.5)
    pooled = F.adaptive_max_pool3d(shell.float(), feature_shape)[0, 0]
    return pooled > 0.5


def objectness_from_boxes(boxes: torch.Tensor, feature_shape: Tuple[int, int, int], resolution: int) -> torch.Tensor:
    fw, fl, fh = feature_shape
    sx, sy, sz = resolution / fw, resolution / fl, resolution / fh
    xs = (torch.arange(fw, dtype=torch.float32) + 0.5) * sx
    ys = (torch.arange(fl, dtype=torch.float32) + 0.5) * sy
    zs = (torch.arange(fh, dtype=torch.float32) + 0.5) * sz
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    target = torch.zeros(feature_shape, dtype=torch.bool)
    if boxes is None or boxes.numel() == 0:
        return target
    boxes = boxes.float().cpu()
    for box in boxes:
        if box.numel() >= 7:
            cx, cy, cz, w, l, h, theta = box[:7]
            c, s = torch.cos(theta), torch.sin(theta)
            dx = gx - cx
            dy = gy - cy
            # Rotation convention only affects oriented boxes. This target is a
            # coarse objectness probe, so small convention differences are not
            # used for localization claims.
            xr = c * dx + s * dy
            yr = -s * dx + c * dy
            inside = (xr.abs() <= w / 2) & (yr.abs() <= l / 2) & ((gz - cz).abs() <= h / 2)
        else:
            xmin, ymin, zmin, xmax, ymax, zmax = box[:6]
            inside = (gx >= xmin) & (gx <= xmax) & (gy >= ymin) & (gy <= ymax) & (gz >= zmin) & (gz <= zmax)
        target |= inside
    return target


def sample_features(
    feat: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    samples_per_class: int,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flat_valid = valid.reshape(-1)
    y = target.reshape(-1) & flat_valid
    neg = (~target.reshape(-1)) & flat_valid
    pos_idx = torch.nonzero(y, as_tuple=False).flatten()
    neg_idx = torch.nonzero(neg, as_tuple=False).flatten()
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return torch.empty(0, feat.shape[0]), torch.empty(0)
    npos = min(samples_per_class, pos_idx.numel())
    nneg = min(samples_per_class, neg_idx.numel())
    pos_idx = pos_idx[torch.randperm(pos_idx.numel(), generator=rng)[:npos]]
    neg_idx = neg_idx[torch.randperm(neg_idx.numel(), generator=rng)[:nneg]]
    idx = torch.cat([pos_idx, neg_idx])
    labels = torch.cat([torch.ones(npos), torch.zeros(nneg)])
    # [C, W, L, H] -> [W*L*H, C]
    x = feat.permute(1, 2, 3, 0).reshape(-1, feat.shape[0])[idx].float()
    perm = torch.randperm(idx.numel(), generator=rng)
    return x[perm], labels[perm]


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order].astype(np.float64)
    total_pos = labels.sum()
    if total_pos <= 0:
        return float("nan")
    tp = np.cumsum(labels)
    precision = tp / (np.arange(labels.size) + 1)
    return float((precision * labels).sum() / total_pos)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.bool_)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    pos_ranks = ranks[labels].sum()
    return float((pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def accuracy_at_half(scores: np.ndarray, labels: np.ndarray) -> float:
    pred = scores >= 0.0
    return float((pred == labels.astype(np.bool_)).mean())


def train_linear_readout(x: torch.Tensor, y: torch.Tensor, args: argparse.Namespace, device: torch.device) -> nn.Linear:
    head = nn.Linear(x.shape[1], 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_cpu = x.contiguous()
    y_cpu = y.contiguous()
    n = x_cpu.shape[0]
    rng = torch.Generator().manual_seed(args.seed + 1009)
    for _ in range(args.readout_epochs):
        perm = torch.randperm(n, generator=rng)
        for start in range(0, n, args.readout_batch_size):
            idx = perm[start : start + args.readout_batch_size]
            xb = x_cpu[idx].to(device, non_blocking=True)
            yb = y_cpu[idx].to(device, non_blocking=True)
            logits = head(xb).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head.cpu().eval()


def eval_linear_readout(head: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        scores = head(x.float()).squeeze(1).numpy()
    labels = y.numpy()
    return {
        "balanced_ap": average_precision(scores, labels),
        "auroc": auroc(scores, labels),
        "acc_at_logit0": accuracy_at_half(scores, labels),
        "n": int(labels.size),
        "pos_frac": float(labels.mean()) if labels.size else float("nan"),
    }


def collect_samples(
    model: Optional[SwinTransformer_FPN_Pretrained_Skip],
    dataset: Front3DRPNDataset,
    args: argparse.Namespace,
    device: torch.device,
    eval_mode: bool,
) -> Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]]:
    rng = torch.Generator().manual_seed(args.seed + (2003 if eval_mode else 0))
    samples: Dict[Tuple[int, str], List[Tuple[torch.Tensor, torch.Tensor]]] = {
        (stage, target): [] for stage in args.stages for target in args.targets
    }
    cap = args.max_val_scenes if eval_mode else args.max_train_scenes
    limit = len(dataset) if cap <= 0 else min(cap, len(dataset))
    per_class = args.eval_samples_per_class if eval_mode else args.samples_per_class
    for idx in range(limit):
        rgbsigma, boxes, scene = dataset[idx]
        original_shape = tuple(int(v) for v in rgbsigma.shape[-3:])
        if model is None:
            feats = {
                stage: coordinate_features(
                    (
                        args.resolution // (4 * (2 ** stage)),
                        args.resolution // (4 * (2 ** stage)),
                        args.resolution // (4 * (2 ** stage)),
                    ),
                    args.resolution,
                )
                for stage in args.stages
            }
        else:
            padded = pad_to_resolution(rgbsigma, args.resolution).unsqueeze(0).to(device)
            feats = extract_stage_features(model, padded, args.stages)
        alpha = pad_to_resolution(rgbsigma[-1], args.resolution).cpu()
        for stage, feat in feats.items():
            fshape = tuple(int(v) for v in feat.shape[-3:])
            valid = valid_mask_for_feature(fshape, original_shape, args.resolution)
            occ = downsample_occupancy(alpha, fshape, args.alpha_threshold) & valid
            targets = {
                "occupancy": occ,
                "shell": downsample_shell(alpha, fshape, args.shell_alpha_threshold, args.shell_smooth_kernel) & valid,
                "objectness": objectness_from_boxes(boxes, fshape, args.resolution) & valid,
            }
            for target_name in args.targets:
                xs, ys = sample_features(feat, targets[target_name], valid, per_class, rng)
                if xs.numel() > 0:
                    samples[(stage, target_name)].append((xs, ys))
    packed: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}
    for key, chunks in samples.items():
        if chunks:
            packed[key] = (torch.cat([c[0] for c in chunks], dim=0), torch.cat([c[1] for c in chunks], dim=0))
        else:
            packed[key] = (torch.empty(0, 1), torch.empty(0))
    return packed


def make_dataset(args: argparse.Namespace, scenes: np.ndarray, percent_train: float) -> Front3DRPNDataset:
    return Front3DRPNDataset(
        features_path=str(args.data_root / "features"),
        boxes_path=str(args.data_root / "obb"),
        scene_list=list(scenes),
        normalize_density=True,
        flip_prob=0.0,
        rotate_prob=0.0,
        rot_scale_prob=0.0,
        preload=False,
        percent_train=percent_train,
    )


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arms = resolve_arms(args.arms, root)
    if not arms:
        raise SystemExit("No arms to evaluate.")
    with np.load(args.split) as split:
        train_scenes = split["train_scenes"]
        val_scenes = split["val_scenes"]
    train_set = make_dataset(args, train_scenes, args.percent_train)
    val_set = make_dataset(args, val_scenes, 1.0)
    print(f"[info] train scenes={len(train_set)} val scenes={len(val_set)} percent_train={args.percent_train}")

    all_rows: List[Dict[str, object]] = []
    manifest = {
        "percent_train": args.percent_train,
        "resolution": args.resolution,
        "alpha_threshold": args.alpha_threshold,
        "shell_alpha_threshold": args.shell_alpha_threshold,
        "shell_smooth_kernel": args.shell_smooth_kernel,
        "stages": args.stages,
        "targets": args.targets,
        "samples_per_class": args.samples_per_class,
        "eval_samples_per_class": args.eval_samples_per_class,
        "readout_epochs": args.readout_epochs,
        "arms": [{"name": a.name, "checkpoint": str(a.checkpoint) if a.checkpoint else None} for a in arms],
        "notes": [
            "coord_only uses normalized x/y/z coordinates at each feature-grid cell.",
            "balanced_ap is measured under class-balanced sampling; it is not detector AP.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for arm in arms:
        print(f"[arm] {arm.name} checkpoint={arm.checkpoint}")
        model = None if arm.name == "coord_only" else build_backbone(arm, args.resolution, device)
        train_samples = collect_samples(model, train_set, args, device, eval_mode=False)
        val_samples = collect_samples(model, val_set, args, device, eval_mode=True)
        if model is not None:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for stage in args.stages:
            for target in args.targets:
                x_train, y_train = train_samples[(stage, target)]
                x_val, y_val = val_samples[(stage, target)]
                row = {
                    "arm": arm.name,
                    "stage": stage,
                    "target": target,
                    "train_n": int(y_train.numel()),
                    "train_pos_frac": float(y_train.mean()) if y_train.numel() else float("nan"),
                    "val_n": int(y_val.numel()),
                    "val_pos_frac": float(y_val.mean()) if y_val.numel() else float("nan"),
                }
                if y_train.numel() == 0 or y_val.numel() == 0:
                    row.update({"balanced_ap": float("nan"), "auroc": float("nan"), "acc_at_logit0": float("nan")})
                    print("[warn] empty samples", row)
                else:
                    head = train_linear_readout(x_train, y_train, args, device)
                    metrics = eval_linear_readout(head, x_val, y_val)
                    row.update(metrics)
                    print(
                        f"[result] {arm.name} stage={stage} target={target} "
                        f"AP={metrics['balanced_ap']:.4f} AUROC={metrics['auroc']:.4f} "
                        f"n={metrics['n']}"
                    )
                all_rows.append(row)

    csv_path = args.output_dir / "readout_results.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "arm",
            "stage",
            "target",
            "balanced_ap",
            "auroc",
            "acc_at_logit0",
            "n",
            "pos_frac",
            "train_n",
            "train_pos_frac",
            "val_n",
            "val_pos_frac",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output_dir / "readout_results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"[done] wrote {csv_path}")


if __name__ == "__main__":
    main()
