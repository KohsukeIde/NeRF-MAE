#!/usr/bin/env python3
"""Minimal scaffold for a mask-predictability probe.

This script intentionally avoids entangling with a specific training launcher.  It saves
the exact protocol and a small logistic-probe helper.  In practice you can use it in
two modes:

1. `--make-protocol`: write a markdown protocol for the current repo.
2. `--fit-from-npz`: fit a logistic probe from saved features and patch masks.

Expected `.npz` for fit mode:
    features: [N, C] or [N, ..., C]
    mask:     [N] or [N, ...] with 1 for masked/removed

For extracting features from NeRF-MAE, add forward hooks to `stages.*` or the stage
outputs used by the decoder, save surface/empty token features and mask labels, then
call this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_logreg(x, y, lr=0.1, steps=500, l2=1e-4):
    x = x.astype(np.float64)
    y = y.astype(np.float64).reshape(-1)
    x = (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)
    w = np.zeros(x.shape[1], dtype=np.float64)
    b = 0.0
    for _ in range(steps):
        p = sigmoid(x @ w + b)
        grad_w = (x.T @ (p - y)) / len(y) + l2 * w
        grad_b = float((p - y).mean())
        w -= lr * grad_w
        b -= lr * grad_b
    p = sigmoid(x @ w + b)
    acc = float(((p > 0.5) == (y > 0.5)).mean())
    # AUC without sklearn.
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(p), dtype=np.float64) + 1
    pos = y > 0.5
    n_pos = pos.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        auc = float("nan")
    else:
        auc = float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return {"acc": acc, "auc": auc, "pos_frac": float(pos.mean())}


def write_protocol(path: Path):
    text = """# Mask-predictability probe protocol

Goal: test whether the full-grid NeRF-MAE encoder exposes the random mask pattern in its features.

1. Choose checkpoints: baseline_e300, cosine_ramp_e300, surface_cosine_jitter_e300.
2. Run the model in eval mode with normal pretraining masking.
3. Hook stage0/stage1/stage2/stage3 features before decoder upsampling.
4. Downsample / reshape the patch mask to each stage resolution.
5. Save `.npz` files with `features` and `mask` per stage / surface-region.
6. Fit linear/logistic probes with this script.
7. Report AUC. High AUC implies mask-pattern leakage into encoder features.

Interpretation:
- High mask AUC in baseline/current recipe: MixNeRF-MAE motivation is strong.
- MixNeRF e30/e100 should reduce mask AUC if it works.
- If mask AUC is low already, MixNeRF is less motivated.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-protocol", type=Path, default=None)
    ap.add_argument("--fit-from-npz", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()

    if args.make_protocol:
        write_protocol(args.make_protocol)
        return
    if not args.fit_from_npz:
        raise SystemExit("Use --make-protocol or --fit-from-npz")
    data = np.load(args.fit_from_npz)
    x = data["features"]
    y = data["mask"]
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1)
    # balanced subsample if very large
    n = min(len(y), 200000)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), n, replace=False) if len(y) > n else np.arange(len(y))
    result = fit_logreg(x[idx], y[idx], lr=args.lr, steps=args.steps)
    print(result)


if __name__ == "__main__":
    main()
