#!/usr/bin/env python3
"""Check that FCOS loads the pretrained MAE encoder from a checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nerf_mae.model.mae.swin_mae3d import SwinTransformer_MAE3D_New
from nerf_rpn.model.feature_extractor import SwinTransformer_FPN_Pretrained_Skip


ENCODER_PREFIXES = ("pos_embed", "patch_partition", "stages.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--skip-fcos-instantiate", action="store_true")
    return parser.parse_args()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"checkpoint does not contain state_dict: {path}")
    return checkpoint["state_dict"]


def make_mae_model(resolution: int) -> SwinTransformer_MAE3D_New:
    return SwinTransformer_MAE3D_New(
        patch_size=[4, 4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[4, 4, 4],
        stochastic_depth_prob=0.1,
        expand_dim=True,
        resolution=resolution,
    )


def is_prefix_key(key: str, prefix: str) -> bool:
    return key == prefix or key.startswith(prefix)


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(tuple(arr.shape)).encode("utf-8"))
    digest.update(arr.numpy().tobytes())
    return digest.hexdigest()


def prefix_digest(state: dict[str, torch.Tensor], keys: list[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def summarize_prefix(
    prefix: str,
    model_state: dict[str, torch.Tensor],
    checkpoint_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    keys = [key for key in model_state if is_prefix_key(key, prefix)]
    matched = [
        key
        for key in keys
        if key in checkpoint_state and tuple(model_state[key].shape) == tuple(checkpoint_state[key].shape)
    ]
    exact = [
        key
        for key in matched
        if torch.equal(model_state[key].detach().cpu(), checkpoint_state[key].detach().cpu())
    ]
    total_numel = sum(int(model_state[key].numel()) for key in keys)
    matched_numel = sum(int(model_state[key].numel()) for key in matched)
    exact_numel = sum(int(model_state[key].numel()) for key in exact)
    return {
        "prefix": prefix,
        "total_tensors": len(keys),
        "matched_shape_tensors": len(matched),
        "exact_loaded_tensors": len(exact),
        "total_numel": total_numel,
        "matched_shape_numel": matched_numel,
        "exact_loaded_numel": exact_numel,
        "matched_shape_tensor_ratio": len(matched) / len(keys) if keys else 1.0,
        "exact_loaded_tensor_ratio": len(exact) / len(keys) if keys else 1.0,
        "matched_shape_numel_ratio": matched_numel / total_numel if total_numel else 1.0,
        "exact_loaded_numel_ratio": exact_numel / total_numel if total_numel else 1.0,
        "model_digest": prefix_digest(model_state, keys),
        "checkpoint_digest": prefix_digest(checkpoint_state, matched),
        "unmatched_keys": [key for key in keys if key not in matched],
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_state = load_checkpoint(checkpoint_path)

    model = make_mae_model(args.resolution)
    missing, unexpected = model.load_state_dict(checkpoint_state, strict=False)
    model_state = model.state_dict()

    prefix_summaries = [
        summarize_prefix(prefix, model_state, checkpoint_state)
        for prefix in ENCODER_PREFIXES
    ]
    encoder_total_numel = sum(item["total_numel"] for item in prefix_summaries)
    encoder_exact_numel = sum(item["exact_loaded_numel"] for item in prefix_summaries)
    encoder_total_tensors = sum(item["total_tensors"] for item in prefix_summaries)
    encoder_exact_tensors = sum(item["exact_loaded_tensors"] for item in prefix_summaries)

    fcos_instantiated = False
    if not args.skip_fcos_instantiate:
        fcos = SwinTransformer_FPN_Pretrained_Skip(
            checkpoint_path=str(checkpoint_path),
            is_eval=False,
            resolution=args.resolution,
        )
        fcos_instantiated = hasattr(fcos, "base") and hasattr(fcos, "fpn_neck")

    encoder_missing = [
        key for key in missing if any(is_prefix_key(key, prefix) for prefix in ENCODER_PREFIXES)
    ]
    encoder_unexpected = [
        key for key in unexpected if any(is_prefix_key(key, prefix) for prefix in ENCODER_PREFIXES)
    ]
    unexpected_diagnostic = [
        key
        for key in unexpected
        if key.startswith("decomp_") or key.startswith("out.") or key.startswith("decoder")
    ]

    payload: dict[str, Any] = {
        "git_hash": git_head(),
        "checkpoint_path": str(checkpoint_path),
        "resolution": args.resolution,
        "checkpoint_key_count": len(checkpoint_state),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "encoder_missing": encoder_missing,
        "encoder_unexpected": encoder_unexpected,
        "unexpected_diagnostic_head_keys": unexpected_diagnostic,
        "encoder_exact_loaded_tensor_ratio": encoder_exact_tensors / encoder_total_tensors
        if encoder_total_tensors
        else 1.0,
        "encoder_exact_loaded_numel_ratio": encoder_exact_numel / encoder_total_numel
        if encoder_total_numel
        else 1.0,
        "prefix_summaries": prefix_summaries,
        "fcos_instantiated": fcos_instantiated,
        "pass": (
            not encoder_missing
            and not encoder_unexpected
            and encoder_exact_tensors == encoder_total_tensors
            and fcos_instantiated
        ),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# FCOS Checkpoint Load Sanity",
        "",
        f"- Git hash: `{payload['git_hash']}`",
        f"- Checkpoint: `{payload['checkpoint_path']}`",
        f"- FCOS instantiated: `{payload['fcos_instantiated']}`",
        f"- Pass: `{payload['pass']}`",
        f"- Missing keys: `{payload['missing_count']}`",
        f"- Unexpected keys: `{payload['unexpected_count']}`",
        f"- Encoder missing keys: `{len(encoder_missing)}`",
        f"- Encoder unexpected keys: `{len(encoder_unexpected)}`",
        f"- Encoder exact tensor ratio: `{payload['encoder_exact_loaded_tensor_ratio']:.6f}`",
        f"- Encoder exact numel ratio: `{payload['encoder_exact_loaded_numel_ratio']:.6f}`",
        "",
        "| prefix | exact tensors | total tensors | exact numel ratio |",
        "|---|---:|---:|---:|",
    ]
    for item in prefix_summaries:
        lines.append(
            "| {prefix} | {exact} | {total} | {ratio:.6f} |".format(
                prefix=item["prefix"],
                exact=item["exact_loaded_tensors"],
                total=item["total_tensors"],
                ratio=item["exact_loaded_numel_ratio"],
            )
        )
    if unexpected:
        lines.extend(["", "## Unexpected Keys", ""])
        lines.extend(f"- `{key}`" for key in unexpected[:50])
    if missing:
        lines.extend(["", "## Missing Keys", ""])
        lines.extend(f"- `{key}`" for key in missing[:50])

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"[info] wrote {args.out_json}")
    print(f"[info] wrote {args.out_md}")


if __name__ == "__main__":
    main()
