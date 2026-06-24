"""Compare Anchor-RPN MAE initialization modes.

This checks whether the legacy constructor path and the downstream-backbone
direct path produce the same encoder weights. It is intended as a sanity audit
before using Anchor-RPN as a detector-head breadth experiment.
"""

import argparse
import json
from pathlib import Path

import torch

from model.feature_extractor import SwinTransformer_FPN_Pretrained_Skip


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_constructor(checkpoint, resolution):
    return SwinTransformer_FPN_Pretrained_Skip(
        resolution=resolution,
        checkpoint_path=checkpoint,
        is_eval=False,
    )


def build_direct(checkpoint, resolution):
    model = SwinTransformer_FPN_Pretrained_Skip(
        resolution=resolution,
        checkpoint_path=None,
        is_eval=True,
    )
    state = torch_load(checkpoint)["state_dict"]
    missing, unexpected = model.base.load_state_dict(state, strict=False)
    return model, missing, unexpected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    ctor = build_constructor(args.checkpoint, args.resolution)
    direct, missing, unexpected = build_direct(args.checkpoint, args.resolution)

    ctor_state = ctor.base.state_dict()
    direct_state = direct.base.state_dict()
    common = sorted(set(ctor_state) & set(direct_state))
    mismatches = []
    max_abs = 0.0
    for key in common:
        a = ctor_state[key]
        b = direct_state[key]
        if a.shape != b.shape:
            mismatches.append({"key": key, "reason": "shape", "ctor": list(a.shape), "direct": list(b.shape)})
            continue
        diff = (a.detach().cpu() - b.detach().cpu()).abs()
        value = float(diff.max().item()) if diff.numel() else 0.0
        max_abs = max(max_abs, value)
        if value != 0.0:
            mismatches.append({"key": key, "reason": "value", "max_abs": value})

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "resolution": args.resolution,
        "common_base_keys": len(common),
        "constructor_only_keys": sorted(set(ctor_state) - set(direct_state))[:20],
        "direct_only_keys": sorted(set(direct_state) - set(ctor_state))[:20],
        "direct_missing_count": len(missing),
        "direct_unexpected_count": len(unexpected),
        "direct_unexpected_encoder_count": len(
            [k for k in unexpected if not k.startswith(("decoder", "out", "mask_token"))]
        ),
        "max_abs_diff_common": max_abs,
        "mismatch_count": len(mismatches),
        "mismatch_sample": mismatches[:20],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
