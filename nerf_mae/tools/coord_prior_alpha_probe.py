#!/usr/bin/env python3
"""Train a tiny coordinate-only alpha prior diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--max-train-scenes", type=int, default=512)
    parser.add_argument("--max-val-scenes", type=int, default=64)
    parser.add_argument("--samples-per-scene", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def density_to_alpha(density: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


def sample_scene(path: Path, samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        rgbsigma = data["rgbsigma"]
    alpha = density_to_alpha(rgbsigma[..., -1].astype(np.float32))
    shape = np.asarray(alpha.shape, dtype=np.int64)
    flat_size = int(alpha.size)
    idx = rng.integers(0, flat_size, size=samples, endpoint=False)
    coords = np.stack(np.unravel_index(idx, alpha.shape), axis=1).astype(np.float32)
    denom = np.maximum(shape.astype(np.float32) - 1.0, 1.0)
    coords = coords / denom * 2.0 - 1.0
    target = alpha.reshape(-1)[idx].astype(np.float32)
    return coords, target[:, None]


def build_samples(
    features_path: Path,
    scenes: np.ndarray,
    max_scenes: int,
    samples_per_scene: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for scene in scenes[:max_scenes]:
        path = features_path / f"{str(scene)}.npz"
        if not path.exists():
            continue
        x, y = sample_scene(path, samples_per_scene, rng)
        xs.append(x)
        ys.append(y)
    if not xs:
        raise RuntimeError(f"no samples loaded from {features_path}")
    return torch.from_numpy(np.concatenate(xs, axis=0)), torch.from_numpy(np.concatenate(ys, axis=0))


class CoordMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def binary_stats(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> dict[str, float]:
    pred_occ = pred >= threshold
    target_occ = target >= threshold
    tp = torch.logical_and(pred_occ, target_occ).sum().item()
    fp = torch.logical_and(pred_occ, ~target_occ).sum().item()
    fn = torch.logical_and(~pred_occ, target_occ).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "pred_pos_rate": pred_occ.float().mean().item(),
    }


def average_precision(pred: torch.Tensor, target_occ: torch.Tensor) -> float:
    scores = pred.flatten()
    labels = target_occ.flatten().to(torch.float32)
    if labels.sum().item() == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    tp = torch.cumsum(sorted_labels, dim=0)
    precision = tp / torch.arange(1, sorted_labels.numel() + 1, dtype=torch.float32)
    return (precision * sorted_labels).sum().item() / labels.sum().item()


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, threshold: float) -> dict[str, object]:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            preds.append(model(x[start : start + batch_size]).cpu())
    pred = torch.cat(preds, dim=0)
    target = y.cpu()
    mse = torch.mean((pred - target) ** 2).item()
    target_occ = target >= threshold
    pred_clamped = pred.clamp(min=1e-6, max=1.0 - 1e-6)
    bce = torch.nn.functional.binary_cross_entropy(pred_clamped, target_occ.to(torch.float32)).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    pos_rate = target_occ.float().mean().item()
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    threshold_sweep = [binary_stats(pred, target, t) for t in thresholds]
    primary_stats = binary_stats(pred, target, threshold)
    best_iou = max(threshold_sweep, key=lambda item: item["iou"])
    return {
        "mse": mse,
        "mae": mae,
        "binary_bce": bce,
        "average_precision_occ": average_precision(pred, target_occ),
        "precision": primary_stats["precision"],
        "recall": primary_stats["recall"],
        "iou": primary_stats["iou"],
        "target_pos_rate": pos_rate,
        "pred_pos_rate": primary_stats["pred_pos_rate"],
        "pred_mean": pred.mean().item(),
        "target_mean": target.mean().item(),
        "best_iou_threshold": best_iou["threshold"],
        "best_iou": best_iou["iou"],
        "threshold_sweep": threshold_sweep,
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    with np.load(args.split) as split:
        train_scenes = split["train_scenes"]
        val_scenes = split["val_scenes"]

    x_train, y_train = build_samples(
        args.features_path, train_scenes, args.max_train_scenes, args.samples_per_scene, rng
    )
    x_val, y_val = build_samples(
        args.features_path, val_scenes, args.max_val_scenes, args.samples_per_scene, rng
    )
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    model = CoordMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(x_train.shape[0], device=device)
        epoch_losses = []
        for start in range(0, x_train.shape[0], args.batch_size):
            idx = perm[start : start + args.batch_size]
            pred = model(x_train[idx])
            loss = loss_fn(pred, y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(float(np.mean(epoch_losses)))

    result = {
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "epochs": args.epochs,
        "samples_per_scene": args.samples_per_scene,
        "max_train_scenes": args.max_train_scenes,
        "max_val_scenes": args.max_val_scenes,
        "final_train_mse": losses[-1],
        "val": evaluate(model, x_val, y_val, args.batch_size, args.threshold),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(result, f, indent=2)

    val = result["val"]
    lines = [
        "# Coordinate-Only Alpha Prior",
        "",
        f"- Train samples: `{result['train_samples']}`",
        f"- Val samples: `{result['val_samples']}`",
        f"- Final train MSE: `{result['final_train_mse']:.6f}`",
        f"- Val MSE: `{val['mse']:.6f}`",
        f"- Val MAE: `{val['mae']:.6f}`",
        f"- Val binary BCE: `{val['binary_bce']:.6f}`",
        f"- Val occupied AP: `{val['average_precision_occ']:.4f}`",
        f"- Val occupied IoU: `{val['iou']:.4f}`",
        f"- Best threshold / IoU: `{val['best_iou_threshold']:.3f}` / `{val['best_iou']:.4f}`",
        f"- Val precision / recall: `{val['precision']:.4f}` / `{val['recall']:.4f}`",
        f"- Target / predicted occupied rate: `{val['target_pos_rate']:.4f}` / `{val['pred_pos_rate']:.4f}`",
        f"- Target / predicted mean alpha: `{val['target_mean']:.6f}` / `{val['pred_mean']:.6f}`",
    ]
    args.out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
