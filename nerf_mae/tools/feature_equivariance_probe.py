"""Probe transform-aligned feature consistency for pretrained NeRF-MAE encoders.

This is a read-only mechanism diagnostic: it loads existing checkpoints, applies
two independently sampled coord-jitter transforms to the same scene, extracts
encoder features without MAE masking, inverse-aligns each feature map to the
canonical scene coordinates, and reports token-wise feature similarity.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nerf_mae.model.mae.shortcut_probe import SwinTransformer_MAE3D_Probe


@dataclass(frozen=True)
class CoordTransform:
    rotate90: bool
    flip_x: bool
    flip_y: bool
    shift_x: int
    shift_y: int


def parse_checkpoint_specs(values: Sequence[str]) -> List[Tuple[str, Path]]:
    specs: List[Tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"checkpoint spec must be label=path, got {value!r}")
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"empty checkpoint label in {value!r}")
        specs.append((label, Path(path).expanduser().resolve()))
    return specs


def density_to_alpha(density: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


def load_volume(features_dir: Path, scene: str, normalize_density: bool = True) -> torch.Tensor:
    path = features_dir / f"{scene}.npz"
    with np.load(path) as features:
        rgbsigma = features["rgbsigma"]
    if normalize_density:
        rgbsigma = rgbsigma.copy()
        rgbsigma[..., -1] = density_to_alpha(rgbsigma[..., -1])
    rgbsigma = np.transpose(rgbsigma, (3, 0, 1, 2))
    tensor = torch.from_numpy(rgbsigma)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    return tensor.float().contiguous()


def zero_fill_shift_channel_first(x: torch.Tensor, dim: int, shift: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    size = x.shape[dim]
    if abs(shift) >= size:
        return out
    src = [slice(None)] * x.ndim
    dst = [slice(None)] * x.ndim
    if shift > 0:
        src[dim] = slice(0, size - shift)
        dst[dim] = slice(shift, size)
    else:
        src[dim] = slice(-shift, size)
        dst[dim] = slice(0, size + shift)
    out[tuple(dst)] = x[tuple(src)]
    return out


def zero_fill_shift_channel_last(x: torch.Tensor, dim: int, shift: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    size = x.shape[dim]
    if abs(shift) >= size:
        return out
    src = [slice(None)] * x.ndim
    dst = [slice(None)] * x.ndim
    if shift > 0:
        src[dim] = slice(0, size - shift)
        dst[dim] = slice(shift, size)
    else:
        src[dim] = slice(-shift, size)
        dst[dim] = slice(0, size + shift)
    out[tuple(dst)] = x[tuple(src)]
    return out


def apply_transform_to_volume(x: torch.Tensor, transform: CoordTransform) -> torch.Tensor:
    # x: [C, H, W, D]. This mirrors BaseDataset.augment_rpn_inputs for z-up data.
    if transform.rotate90:
        x = torch.transpose(x, 1, 2)
        x = torch.flip(x, [1])
    if transform.flip_x:
        x = x.flip(dims=[1])
    if transform.flip_y:
        x = x.flip(dims=[2])
    if transform.shift_x:
        x = zero_fill_shift_channel_first(x, dim=1, shift=transform.shift_x)
    if transform.shift_y:
        x = zero_fill_shift_channel_first(x, dim=2, shift=transform.shift_y)
    return x.contiguous()


def inverse_align_feature(
    feat: torch.Tensor,
    transform: CoordTransform,
    input_resolution: int,
) -> torch.Tensor:
    # feat: [B, H, W, D, C].
    h, w = int(feat.shape[1]), int(feat.shape[2])
    stride_x = max(float(input_resolution) / float(h), 1.0)
    stride_y = max(float(input_resolution) / float(w), 1.0)
    shift_x = int(round(float(transform.shift_x) / stride_x))
    shift_y = int(round(float(transform.shift_y) / stride_y))

    if shift_y:
        feat = zero_fill_shift_channel_last(feat, dim=2, shift=-shift_y)
    if shift_x:
        feat = zero_fill_shift_channel_last(feat, dim=1, shift=-shift_x)
    if transform.flip_y:
        feat = feat.flip(dims=[2])
    if transform.flip_x:
        feat = feat.flip(dims=[1])
    if transform.rotate90:
        feat = torch.flip(feat, [1])
        feat = torch.transpose(feat, 1, 2)
    return feat.contiguous()


def inverse_align_mask(mask: torch.Tensor, transform: CoordTransform, input_resolution: int) -> torch.Tensor:
    # mask: [B, 1, H, W, D].
    feat = mask.permute(0, 2, 3, 4, 1).contiguous()
    feat = inverse_align_feature(feat, transform, input_resolution)
    return feat.permute(0, 4, 1, 2, 3).contiguous()


def sample_transform(
    rng: random.Random,
    rotate_prob: float,
    flip_prob: float,
    coord_shift_prob: float,
    coord_shift_max_voxels: int,
) -> CoordTransform:
    if rng.random() < coord_shift_prob and coord_shift_max_voxels > 0:
        shift_x = rng.randint(-coord_shift_max_voxels, coord_shift_max_voxels)
        shift_y = rng.randint(-coord_shift_max_voxels, coord_shift_max_voxels)
    else:
        shift_x = 0
        shift_y = 0
    return CoordTransform(
        rotate90=rng.random() < rotate_prob,
        flip_x=rng.random() < flip_prob,
        flip_y=rng.random() < flip_prob,
        shift_x=shift_x,
        shift_y=shift_y,
    )


def build_model(resolution: int, device: torch.device) -> SwinTransformer_MAE3D_Probe:
    model = SwinTransformer_MAE3D_Probe(
        patch_size=[4, 4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[4, 4, 4],
        stochastic_depth_prob=0.1,
        expand_dim=True,
        masking_prob=0.0,
        resolution=resolution,
        masking_strategy="random",
        probe_mode="baseline",
    )
    return model.to(device).eval()


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> Dict[str, object]:
    # These are locally produced training checkpoints with optimizer metadata.
    # PyTorch 2.6 defaults to weights_only=True, which rejects that metadata.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = clean_state_dict(checkpoint["state_dict"])
    incompatible = model.load_state_dict(state_dict, strict=False)
    return {
        "path": str(checkpoint_path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "epoch": checkpoint.get("epoch"),
    }


def pad_with_model(
    model: SwinTransformer_MAE3D_Probe,
    tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    padded, masks = model.transform([tensor])
    return padded[0].to(device), masks[0].to(device)


@torch.no_grad()
def extract_stage_features(
    model: SwinTransformer_MAE3D_Probe,
    volume: torch.Tensor,
    stages: Iterable[int],
) -> Dict[int, torch.Tensor]:
    # volume: [1, C, H, W, D].
    wanted = set(stages)
    feats: Dict[int, torch.Tensor] = {}
    x = model.patch_partition(volume)
    x = x + model.pos_embed.type_as(x).to(x.device).clone().detach()
    for i, stage in enumerate(model.stages):
        x = stage(x)
        if i in wanted:
            feats[i] = x.detach()
    return feats


def downsample_like(mask_or_alpha: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    return F.adaptive_avg_pool3d(mask_or_alpha.float(), output_size=tuple(spatial_shape))


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.shape[0] < 2:
        return float("nan")
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denom = torch.linalg.matrix_norm(xx) * torch.linalg.matrix_norm(yy)
    if float(denom.detach().cpu()) <= 0.0:
        return float("nan")
    return float((torch.linalg.matrix_norm(xy) ** 2 / denom).detach().cpu())


def compare_features(
    f1: torch.Tensor,
    f2: torch.Tensor,
    valid: torch.Tensor,
    region: torch.Tensor,
    max_tokens: int,
    rng: random.Random,
) -> Dict[str, float]:
    # f*: [1,H,W,D,C], valid/region: [1,1,H,W,D].
    mask = (valid > 0.5) & region
    token_mask = mask.squeeze(0).squeeze(0).reshape(-1)
    n_tokens = int(token_mask.sum().item())
    if n_tokens <= 1:
        return {
            "tokens": float(n_tokens),
            "cosine": float("nan"),
            "l2": float("nan"),
            "cka": float("nan"),
        }

    x = f1.reshape(-1, f1.shape[-1])[token_mask]
    y = f2.reshape(-1, f2.shape[-1])[token_mask]
    if x.shape[0] > max_tokens:
        indices = list(range(x.shape[0]))
        rng.shuffle(indices)
        take = torch.tensor(indices[:max_tokens], device=x.device, dtype=torch.long)
        x = x.index_select(0, take)
        y = y.index_select(0, take)

    cosine = F.cosine_similarity(x, y, dim=-1).mean()
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    l2 = torch.sqrt(((x_norm - y_norm) ** 2).sum(dim=-1).clamp_min(0.0)).mean()
    return {
        "tokens": float(n_tokens),
        "cosine": float(cosine.detach().cpu()),
        "l2": float(l2.detach().cpu()),
        "cka": linear_cka(x.float(), y.float()),
    }


def aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["label"]), int(row["stage"]), str(row["region"]))
        groups.setdefault(key, []).append(row)

    summary: List[Dict[str, object]] = []
    for (label, stage, region), items in sorted(groups.items()):
        out: Dict[str, object] = {
            "label": label,
            "stage": stage,
            "region": region,
            "n": len(items),
        }
        for metric in ("cosine", "l2", "cka", "tokens"):
            values = np.array([float(item[metric]) for item in items], dtype=np.float64)
            values = values[np.isfinite(values)]
            out[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
            out[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary.append(out)
    return summary


def write_markdown(path: Path, summary: List[Dict[str, object]], load_reports: Dict[str, object]) -> None:
    lines = [
        "# Feature Equivariance Probe",
        "",
        "Checkpoint load reports:",
        "",
    ]
    for label, report in load_reports.items():
        lines.append(
            f"- `{label}`: epoch={report.get('epoch')} missing={len(report.get('missing_keys', []))} "
            f"unexpected={len(report.get('unexpected_keys', []))} path=`{report.get('path')}`"
        )
    lines.extend(
        [
            "",
            "| label | stage | region | n | cosine | l2 | cka | tokens |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            "| {label} | {stage} | {region} | {n} | {cosine_mean:.4f} | "
            "{l2_mean:.4f} | {cka_mean:.4f} | {tokens_mean:.1f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset/pretrain"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split", default="val_scenes")
    parser.add_argument("--max-scenes", type=int, default=8)
    parser.add_argument("--num-pairs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--stages", default="0,1,2,3")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--rotate-prob", type=float, default=1.0)
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--coord-shift-prob", type=float, default=1.0)
    parser.add_argument("--coord-shift-max-voxels", type=int, default=8)
    parser.add_argument("--surface-threshold", type=float, default=0.01)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    split_file = args.split_file or (args.data_root / "nerfmae_split.npz")
    features_dir = args.data_root / "features"
    split = np.load(split_file, allow_pickle=True)
    scenes = []
    for scene in [str(item) for item in split[args.split].tolist()]:
        if (features_dir / f"{scene}.npz").is_file():
            scenes.append(scene)
        if len(scenes) >= args.max_scenes:
            break
    if not scenes:
        raise FileNotFoundError(
            f"no feature files found for split={args.split!r} under {features_dir}"
        )
    stages = [int(item) for item in args.stages.split(",") if item.strip()]
    checkpoints = parse_checkpoint_specs(args.checkpoint)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows: List[Dict[str, object]] = []
    load_reports: Dict[str, object] = {}

    for label, checkpoint_path in checkpoints:
        model = build_model(args.resolution, device)
        load_reports[label] = load_checkpoint(model, checkpoint_path)
        model.eval()

        for scene in scenes:
            volume_raw = load_volume(features_dir, scene)
            canonical_volume, _ = pad_with_model(model, volume_raw, device)
            alpha = canonical_volume[:, 3:4]
            full_valid_raw = torch.ones_like(volume_raw[:1])

            for pair_idx in range(args.num_pairs):
                t1 = sample_transform(
                    rng,
                    args.rotate_prob,
                    args.flip_prob,
                    args.coord_shift_prob,
                    args.coord_shift_max_voxels,
                )
                t2 = sample_transform(
                    rng,
                    args.rotate_prob,
                    args.flip_prob,
                    args.coord_shift_prob,
                    args.coord_shift_max_voxels,
                )
                v1_raw = apply_transform_to_volume(volume_raw, t1)
                v2_raw = apply_transform_to_volume(volume_raw, t2)
                m1_raw = apply_transform_to_volume(full_valid_raw, t1)
                m2_raw = apply_transform_to_volume(full_valid_raw, t2)
                v1, _ = pad_with_model(model, v1_raw, device)
                v2, _ = pad_with_model(model, v2_raw, device)
                m1, _ = pad_with_model(model, m1_raw, device)
                m2, _ = pad_with_model(model, m2_raw, device)

                feats1 = extract_stage_features(model, v1, stages)
                feats2 = extract_stage_features(model, v2, stages)
                for stage in stages:
                    f1 = inverse_align_feature(feats1[stage], t1, args.resolution)
                    f2 = inverse_align_feature(feats2[stage], t2, args.resolution)
                    spatial = f1.shape[1:4]
                    valid1 = inverse_align_mask(downsample_like(m1, spatial), t1, args.resolution)
                    valid2 = inverse_align_mask(downsample_like(m2, spatial), t2, args.resolution)
                    valid = (valid1 > 0.5) & (valid2 > 0.5)
                    surface_score = downsample_like(alpha, spatial)
                    surface = surface_score > args.surface_threshold
                    regions = {
                        "all": torch.ones_like(surface, dtype=torch.bool),
                        "surface": surface,
                        "empty": ~surface,
                    }
                    for region_name, region in regions.items():
                        metrics = compare_features(f1, f2, valid, region, args.max_tokens, rng)
                        rows.append(
                            {
                                "label": label,
                                "scene": scene,
                                "pair": pair_idx,
                                "stage": stage,
                                "region": region_name,
                                "transform1": asdict(t1),
                                "transform2": asdict(t2),
                                **metrics,
                            }
                        )

                del feats1, feats2, v1, v2, m1, m2
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del volume_raw, canonical_volume
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = aggregate_rows(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "data_root": str(args.data_root),
            "split_file": str(split_file),
            "split": args.split,
            "max_scenes": args.max_scenes,
            "num_pairs": args.num_pairs,
            "seed": args.seed,
            "stages": stages,
            "rotate_prob": args.rotate_prob,
            "flip_prob": args.flip_prob,
            "coord_shift_prob": args.coord_shift_prob,
            "coord_shift_max_voxels": args.coord_shift_max_voxels,
            "surface_threshold": args.surface_threshold,
            "max_tokens": args.max_tokens,
        },
        "load_reports": load_reports,
        "summary": summary,
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(args.output_md, summary, load_reports)
    print(f"[info] wrote {args.output_json}")
    print(f"[info] wrote {args.output_md}")


if __name__ == "__main__":
    main()
