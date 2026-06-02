#!/usr/bin/env python3
"""Quick sanity checker for MixNeRF-style patch replacement.

This script does not instantiate NeRF-MAE.  It loads one or more `.npz` feature files
and applies the same patch-level replacement used by MixNeRF-MAE-lite, then reports
basic RGBA/alpha statistics.  It is useful before launching a GPU pretrain job.

The loader is intentionally permissive: it searches common keys (`rgbsigma`,
`features`, `data`, `arr_0`) and expects an array with 4 channels either first or last.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def load_volume(path: Path) -> np.ndarray:
    obj = np.load(path)
    if isinstance(obj, np.lib.npyio.NpzFile):
        for key in ("rgbsigma", "features", "data", "arr_0"):
            if key in obj:
                arr = obj[key]
                break
        else:
            key = obj.files[0]
            arr = obj[key]
    else:
        arr = obj
    arr = np.asarray(arr)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D volume, got {arr.shape} from {path}")
    if arr.shape[0] == 4:
        return arr
    if arr.shape[-1] == 4:
        return np.moveaxis(arr, -1, 0)
    raise ValueError(f"Cannot infer channel dim for {arr.shape} from {path}")


def patch_mask(shape, patch_size: int, mask_ratio: float, rng: np.random.Generator):
    _, h, w, d = shape
    gh, gw, gd = h // patch_size, w // patch_size, d // patch_size
    n = gh * gw * gd
    num = int(round(n * mask_ratio))
    m = np.zeros(n, dtype=np.float32)
    idx = rng.permutation(n)[:num]
    m[idx] = 1.0
    m = m.reshape(gh, gw, gd)
    m = np.repeat(np.repeat(np.repeat(m, patch_size, 0), patch_size, 1), patch_size, 2)
    return m[:h, :w, :d][None]


def crop_pair_to_common_patch_grid(a: np.ndarray, b: np.ndarray, patch_size: int):
    """Crop unequal scene volumes to a common patch-aligned spatial shape.

    NeRF-MAE training pads variable-size scenes in `transform()`, so raw feature
    files do not need to have identical shapes.  This standalone sanity script
    does not instantiate the trainer, so it uses a conservative shared crop for
    basic distribution checks.
    """
    if a.shape == b.shape:
        return a, b, False
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"channel mismatch: {a.shape} vs {b.shape}")
    spatial = tuple(min(a.shape[i], b.shape[i]) for i in range(1, 4))
    spatial = tuple((s // patch_size) * patch_size for s in spatial)
    if min(spatial) <= 0:
        raise ValueError(f"no common patch-aligned crop for {a.shape} vs {b.shape}")
    slices = (slice(None),) + tuple(slice(0, s) for s in spatial)
    return a[slices], b[slices], True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features", nargs="+", type=Path)
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if len(args.features) < 2:
        raise SystemExit("Provide at least two feature files to mix.")
    rng = np.random.default_rng(args.seed)
    vols = [load_volume(p) for p in args.features[:2]]
    a, b = vols
    original_shapes = (a.shape, b.shape)
    a, b, cropped = crop_pair_to_common_patch_grid(a, b, args.patch_size)
    m = patch_mask(a.shape, args.patch_size, args.mask_ratio, rng)
    mixed = a * (1 - m) + b * m

    def stats(name: str, x: np.ndarray):
        alpha = x[3]
        print(f"[{name}] shape={x.shape} mean={x.mean():.6f} std={x.std():.6f} alpha_mean={alpha.mean():.6f} alpha_occ001={(alpha>0.01).mean():.6f}")

    stats("A", a)
    stats("B", b)
    stats("mixed", mixed)
    if cropped:
        print(f"cropped_to_common_patch_grid=true original_shapes={original_shapes} cropped_shape={a.shape}")
    print(f"mask_mean={m.mean():.6f}")


if __name__ == "__main__":
    main()
