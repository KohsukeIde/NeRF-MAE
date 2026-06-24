#!/usr/bin/env python3
"""Anchor-RPN sanity diagnostics for MAE checkpoint transfer.

This script is intentionally read-only. It checks two preconditions before an
anchor-head breadth experiment is trusted:

1. MAE checkpoint keys load into the anchor backbone as expected.
2. The raw anchor generator covers ground-truth boxes on the target split.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from datasets import Front3DRPNDataset
from model.anchor import AnchorGenerator3D
from model.coder.misc import obb2hbb_3d
from model.feature_extractor import SwinTransformer_FPN_Pretrained_Skip
from model.utils import box_iou_3d


ANCHOR_SIZES = ((8,), (16,), (32,), (64,),)
ASPECT_RATIOS = (
    ((1.0, 1.0, 1.0), (1.0, 1.0, 2.0), (1.0, 2.0, 2.0), (1.0, 1.0, 3.0), (1.0, 3.0, 3.0)),
) * len(ANCHOR_SIZES)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_path", required=True)
    p.add_argument("--boxes_path", required=True)
    p.add_argument("--dataset_split", required=True)
    p.add_argument("--mae_checkpoint", default="")
    p.add_argument("--split", choices=["train_scenes", "val_scenes", "test_scenes"], default="val_scenes")
    p.add_argument("--resolution", type=int, default=160)
    p.add_argument("--limit_scenes", type=int, default=0)
    p.add_argument("--output", required=True)
    return p.parse_args()


def load_scene_list(path, split):
    data = np.load(path)
    return [str(x) for x in data[split]]


def audit_checkpoint(path, resolution):
    if not path:
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SwinTransformer_FPN_Pretrained_Skip(
        resolution=resolution,
        checkpoint_path=path,
        is_eval=False,
    )
    state = ckpt.get("state_dict", {})
    pos = state.get("pos_embed")
    return {
        "checkpoint": path,
        "state_dict_key_count": len(state),
        "pos_embed_shape": list(pos.shape) if pos is not None else None,
        "backbone_out_channels": (
            list(model.out_channels)
            if isinstance(model.out_channels, (list, tuple))
            else int(model.out_channels)
        ),
    }


def anchor_coverage(args):
    scenes = load_scene_list(args.dataset_split, args.split)
    if args.limit_scenes > 0:
        scenes = scenes[: args.limit_scenes]
    dataset = Front3DRPNDataset(
        features_path=args.features_path,
        boxes_path=args.boxes_path,
        scene_list=scenes,
        normalize_density=True,
        preload=False,
        percent_train=1.0,
    )
    generator = AnchorGenerator3D(ANCHOR_SIZES, ASPECT_RATIOS, is_normalized=False)

    all_max = []
    per_scene = []
    for idx in range(len(dataset)):
        rgbsigma, boxes, scene = dataset[idx]
        # Use the same FPN grid-scale convention as the model: anchor grids are
        # derived from feature map sizes, not from raw voxel resolution.
        spatial = torch.tensor(rgbsigma.shape[1:], dtype=torch.float32)
        grid_sizes = []
        for stride in [4, 8, 16, 32]:
            grid_sizes.append(tuple(torch.ceil(spatial / stride).to(torch.int64).tolist()))
        dummy_features = [
            torch.zeros((1, 256, *size), dtype=torch.float32) for size in grid_sizes
        ]
        anchors, _ = generator(rgbsigma.unsqueeze(0), dummy_features)
        anchors = anchors[0]
        if boxes.numel() == 0:
            continue
        gt = boxes[:, :7].float()
        gt_for_anchor_match = obb2hbb_3d(gt) if gt.shape[1] == 7 else gt
        ious = box_iou_3d(gt_for_anchor_match, anchors.float())
        max_iou = ious.max(dim=1).values.cpu().numpy()
        all_max.extend(max_iou.tolist())
        per_scene.append(
            {
                "scene": scene,
                "num_gt": int(gt.shape[0]),
                "num_anchors": int(anchors.shape[0]),
                "mean_max_iou": float(max_iou.mean()),
                "recall25_oracle": float((max_iou >= 0.25).mean()),
                "recall50_oracle": float((max_iou >= 0.50).mean()),
            }
        )

    arr = np.asarray(all_max, dtype=np.float32)
    return {
        "split": args.split,
        "scene_count": len(per_scene),
        "gt_count": int(arr.size),
        "mean_max_iou": float(arr.mean()) if arr.size else 0.0,
        "median_max_iou": float(np.median(arr)) if arr.size else 0.0,
        "p10_max_iou": float(np.percentile(arr, 10)) if arr.size else 0.0,
        "p90_max_iou": float(np.percentile(arr, 90)) if arr.size else 0.0,
        "oracle_recall25": float((arr >= 0.25).mean()) if arr.size else 0.0,
        "oracle_recall50": float((arr >= 0.50).mean()) if arr.size else 0.0,
        "per_scene": per_scene,
    }


def main():
    args = parse_args()
    result = {
        "checkpoint_audit": audit_checkpoint(args.mae_checkpoint, args.resolution),
        "anchor_coverage": anchor_coverage(args),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["anchor_coverage"], indent=2))
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
