"""Visibility-gated NeRF-MAE probes.

This wrapper is intentionally limited to low-risk V0/V1 variants.  It does not
modify Swin attention.  The goal is to test whether reducing decoder/encoder
participation of masked placeholder tokens improves transfer after the
participation probe showed strong masked-feature norms.

Modes
-----
VISGATE_MODE=off
    Original model.
VISGATE_MODE=feature_reset
    V0: after selected Swin stages, reset masked feature locations to zero before
    the next stage and before decoder use.
VISGATE_MODE=skip_gate
    V1: keep encoder propagation unchanged, but gate selected decoder skip
    features at masked locations.
VISGATE_MODE=reset_skip
    Apply both feature reset and skip gating.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

try:
    from model.mae.shortcut_probe import SwinTransformer_MAE3D_Probe
except ImportError:  # pragma: no cover
    from nerf_mae.model.mae.shortcut_probe import SwinTransformer_MAE3D_Probe


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _parse_stage_list(text: str) -> List[int]:
    stages: List[int] = []
    text = text.replace(":", ",")
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        stages.append(int(item))
    return sorted(set(stages))


class SwinTransformer_MAE3D_VisibilityGated(SwinTransformer_MAE3D_Probe):
    """Drop-in wrapper for V0/V1 visibility gating."""

    MODES = {"off", "feature_reset", "skip_gate", "reset_skip"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visgate_mode = _env_str("VISGATE_MODE", "off")
        if self.visgate_mode not in self.MODES:
            raise ValueError(f"Unsupported VISGATE_MODE={self.visgate_mode!r}")
        self.visgate_reset_stages = _parse_stage_list(_env_str("VISGATE_RESET_STAGES", "0,1,2"))
        self.visgate_skip_stages = _parse_stage_list(_env_str("VISGATE_SKIP_STAGES", "0,1,2"))
        self.visgate_log_stats = _env_flag("VISGATE_LOG_STATS", True)
        self.visibility_stats: Dict[str, object] = {}

    @staticmethod
    def _mask_to_shape(mask_patches: torch.Tensor, shape: Iterable[int]) -> torch.Tensor:
        """Convert patch mask [B,H,W,D,1] to bool [B,h,w,d]."""
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
        # Defensive fallback for odd resolutions.  Nearest keeps the mask discrete.
        x = F.interpolate(x, size=(target_h, target_w, target_d), mode="nearest")
        return x[:, 0] > 0.5

    @staticmethod
    def _gate_channels_first(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return feature * (~mask[:, None]).to(dtype=feature.dtype)

    @staticmethod
    def _gate_channels_last(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return feature * (~mask[..., None]).to(dtype=feature.dtype)

    def _record_stage_stats(self, stage: int, x: torch.Tensor, mask: torch.Tensor, prefix: str) -> None:
        if not self.visgate_log_stats:
            return
        with torch.no_grad():
            norm = x.float().pow(2).sum(dim=-1).sqrt()
            masked = mask
            visible = ~mask
            if masked.any() and visible.any():
                ratio = float((norm[masked].mean() / norm[visible].mean().clamp_min(1e-12)).detach().cpu())
            else:
                ratio = float("nan")
            self.visibility_stats[f"{prefix}_stage{stage}_mask_mean"] = float(mask.float().mean().detach().cpu())
            self.visibility_stats[f"{prefix}_stage{stage}_masked_visible_norm_ratio"] = ratio

    def forward_encoder_ecoder(self, x):
        if self.visgate_mode == "off":
            return super().forward_encoder_ecoder(x)

        x = self.patch_partition(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        x, mask_patches = self.window_masking_3d(
            x, p_remove=self.masking_prob, mask_token=self.mask_token
        )

        features = []
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            stage_mask = self._mask_to_shape(mask_patches, x.shape[1:4]).to(x.device)
            self._record_stage_stats(i, x, stage_mask, "pre")

            should_reset = self.visgate_mode in {"feature_reset", "reset_skip"} and i in self.visgate_reset_stages
            if should_reset:
                x = self._gate_channels_last(x, stage_mask)
                self._record_stage_stats(i, x, stage_mask, "post_reset")

            feature = torch.permute(x, [0, 4, 1, 2, 3]).contiguous()
            should_skip_gate = self.visgate_mode in {"skip_gate", "reset_skip"} and i in self.visgate_skip_stages
            if should_skip_gate:
                feature = self._gate_channels_first(feature, stage_mask)
                if self.visgate_log_stats:
                    self.visibility_stats[f"skip_stage{i}_gated"] = True
            features.append(feature)

        dec3 = self.decoder4(features[3], features[2])
        dec2 = self.decoder3(dec3, features[1])
        dec1 = self.decoder2(dec2, features[0])
        dec0 = self.decoder1(dec1)
        out = self.out(dec0)
        return out, mask_patches
