"""Pyramid target wrapper for NeRF-MAE shortcut probes.

This is an add-on module: it subclasses the existing
`SwinTransformer_MAE3D_Probe` from `shortcut_probe.py` and changes only the
*target volume* used by the reconstruction loss.  The visible input path, mask
sampling, dataset, and FCOS downstream path remain unchanged.

Motivation
----------
The current shortcut-probe results suggest that (i) target-side alpha structure
is a sample-efficient signal, (ii) full-resolution RGBA is useful at larger
budgets, and (iii) local/voxel-level interventions such as predicted-alpha gates
can hurt localization.  This module tests a scene-level coarse-to-fine target
hypothesis by replacing high-resolution alpha/RGB targets with pyramid targets
at early epochs and gradually restoring the full-resolution target.

Default scout variants
----------------------
PYR_MODE=alpha : alpha pyramid only; RGB target stays full-resolution.
PYR_MODE=rgb   : RGB pyramid only; alpha target stays full-resolution.
PYR_MODE=both  : alpha and RGB pyramid.

The default low-resolution alpha target uses max-pooling to preserve occupancy.
RGB uses average-pooling.  Targets are upsampled back to the original grid before
patchification, so no downstream code path needs to change.
"""
from __future__ import annotations

import math
import os
from typing import Tuple

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


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    return float(value)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return str(value)


class SwinTransformer_MAE3D_Pyramid(SwinTransformer_MAE3D_Probe):
    """Scene-level pyramid target wrapper.

    Environment variables
    ---------------------
    PYR_MODE:
        off | alpha | rgb | both | P_A | P_R | P_AR.  Default: off.
    PYR_SCALE:
        Downsampling factor for low-resolution targets. Default: 2.
    PYR_SCHEDULE:
        cosine | linear | hard.  Default: cosine.
    PYR_EPOCHS:
        Epochs over which to transition from low-res to full-res target.
        If <=0, uses total training epochs passed by the wrapper. Default: 0.
    PYR_ALPHA_POOL:
        max | avg.  Default: max.  max preserves occupancy/surface existence.
    PYR_RGB_POOL:
        avg.  Default: avg.
    PYR_UPSAMPLE:
        nearest | trilinear.  Default: nearest for alpha, trilinear for RGB.
        The env sets RGB upsample mode; alpha always defaults to nearest unless
        PYR_ALPHA_UPSAMPLE is explicitly set.
    PYR_LOG_STATS:
        If true, keep target blend stats in self.pyramid_stats for logging.

    Important ordering note
    -----------------------
    This wrapper constructs pyramid targets inside `forward_loss` from the
    already transformed target volume `x`.  Therefore if coordinate jitter is
    applied upstream to the full-resolution scene, low-resolution targets are
    produced *after* the full-resolution scene-level jitter, which preserves the
    intended scene-level-coherence interpretation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode = _env_str("PYR_MODE", "off").lower()
        aliases = {"p_a": "alpha", "pa": "alpha", "p_r": "rgb", "pr": "rgb", "p_ar": "both", "par": "both"}
        self.pyr_mode = aliases.get(mode, mode)
        self.pyr_scale = _env_int("PYR_SCALE", 2)
        self.pyr_schedule = _env_str("PYR_SCHEDULE", "cosine").lower()
        self.pyr_epochs = _env_int("PYR_EPOCHS", 0)
        self.pyr_alpha_pool = _env_str("PYR_ALPHA_POOL", "max").lower()
        self.pyr_rgb_pool = _env_str("PYR_RGB_POOL", "avg").lower()
        self.pyr_rgb_upsample = _env_str("PYR_UPSAMPLE", "trilinear").lower()
        self.pyr_alpha_upsample = _env_str("PYR_ALPHA_UPSAMPLE", "nearest").lower()
        self.pyr_log_stats = _env_flag("PYR_LOG_STATS", True)
        self._pyr_epoch = 1
        self._pyr_total_epochs = 1
        self._pyr_progress = 1.0
        self.pyramid_stats = {}

        if self.pyr_mode not in {"off", "alpha", "rgb", "both"}:
            raise ValueError(f"Unsupported PYR_MODE={self.pyr_mode!r}")
        if self.pyr_scale < 1:
            raise ValueError(f"PYR_SCALE must be >= 1, got {self.pyr_scale}")
        if self.pyr_schedule not in {"cosine", "linear", "hard"}:
            raise ValueError(f"Unsupported PYR_SCHEDULE={self.pyr_schedule!r}")
        if self.pyr_alpha_pool not in {"max", "avg"}:
            raise ValueError(f"Unsupported PYR_ALPHA_POOL={self.pyr_alpha_pool!r}")
        if self.pyr_rgb_pool not in {"avg"}:
            raise ValueError(f"Unsupported PYR_RGB_POOL={self.pyr_rgb_pool!r}")
        if self.pyr_rgb_upsample not in {"nearest", "trilinear"}:
            raise ValueError(f"Unsupported PYR_UPSAMPLE={self.pyr_rgb_upsample!r}")
        if self.pyr_alpha_upsample not in {"nearest", "trilinear"}:
            raise ValueError(f"Unsupported PYR_ALPHA_UPSAMPLE={self.pyr_alpha_upsample!r}")

    def set_pyramid_epoch(self, epoch: int, total_epochs: int) -> None:
        self._pyr_epoch = int(epoch)
        self._pyr_total_epochs = max(1, int(total_epochs))
        self._pyr_progress = self._compute_progress(epoch, total_epochs)

    def _compute_progress(self, epoch: int, total_epochs: int) -> float:
        if self.pyr_mode == "off" or self.pyr_scale == 1:
            return 1.0
        ramp_epochs = self.pyr_epochs if self.pyr_epochs > 0 else total_epochs
        if ramp_epochs <= 1:
            base = 1.0
        else:
            base = min(max((int(epoch) - 1) / float(ramp_epochs - 1), 0.0), 1.0)
        if self.pyr_schedule == "cosine":
            return float(0.5 * (1.0 - math.cos(math.pi * base)))
        if self.pyr_schedule == "hard":
            return float(1.0 if base >= 1.0 else 0.0)
        return float(base)

    @staticmethod
    def _interpolate_like(x: torch.Tensor, size: Tuple[int, int, int], mode: str) -> torch.Tensor:
        if mode == "nearest":
            return F.interpolate(x, size=size, mode="nearest")
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)

    def _pool_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        k = self.pyr_scale
        if k == 1:
            return alpha
        if self.pyr_alpha_pool == "max":
            return F.max_pool3d(alpha, kernel_size=k, stride=k)
        return F.avg_pool3d(alpha, kernel_size=k, stride=k)

    def _pool_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        k = self.pyr_scale
        if k == 1:
            return rgb
        return F.avg_pool3d(rgb, kernel_size=k, stride=k)

    def _pyramid_target_volume(self, x: torch.Tensor) -> torch.Tensor:
        if self.pyr_mode == "off" or self.pyr_scale == 1:
            return x
        if x.dim() != 5 or x.shape[1] < 4:
            raise ValueError(f"Expected target volume [B,4,H,W,D], got {tuple(x.shape)}")

        full_rgb = x[:, :3, ...]
        full_alpha = x[:, 3:4, ...]
        spatial_size = tuple(full_alpha.shape[-3:])
        t = float(self._pyr_progress)

        out_rgb = full_rgb
        out_alpha = full_alpha

        if self.pyr_mode in {"alpha", "both"}:
            low_alpha = self._pool_alpha(full_alpha)
            low_alpha = self._interpolate_like(low_alpha, spatial_size, self.pyr_alpha_upsample)
            out_alpha = (1.0 - t) * low_alpha + t * full_alpha

        if self.pyr_mode in {"rgb", "both"}:
            low_rgb = self._pool_rgb(full_rgb)
            low_rgb = self._interpolate_like(low_rgb, spatial_size, self.pyr_rgb_upsample)
            out_rgb = (1.0 - t) * low_rgb + t * full_rgb

        if self.pyr_log_stats:
            with torch.no_grad():
                self.pyramid_stats = {
                    "epoch": float(self._pyr_epoch),
                    "progress": float(t),
                    "mode": self.pyr_mode,
                    "scale": float(self.pyr_scale),
                    "alpha_low_mean": float(out_alpha.mean().detach().cpu()),
                    "rgb_low_mean": float(out_rgb.mean().detach().cpu()),
                }

        return torch.cat([out_rgb, out_alpha], dim=1)

    def forward_loss(self, x, pred, mask_batch, mask_patches, is_eval=False):
        if self.pyr_mode == "off":
            return super().forward_loss(x, pred, mask_batch, mask_patches, is_eval=is_eval)

        target_volume = self._pyramid_target_volume(x)
        target, valid_mask = self.patchify_3d(target_volume, mask_batch)
        full_target, _ = self.patchify_3d(x, mask_batch)
        pred = self.patchify_3d(pred)

        removed_mask = valid_mask.squeeze(-1).int() * mask_patches
        removed_mask = removed_mask.unsqueeze(-1).to(pred.dtype)
        valid_mask = valid_mask.to(pred.dtype)

        target_rgb = target[..., :3]
        target_alpha = target[..., 3].unsqueeze(-1)
        full_target_alpha = full_target[..., 3].unsqueeze(-1)
        target_alpha = self._apply_probe_alpha_target_corruption(target_alpha)

        pred_rgb = pred[..., :3]
        pred_alpha = self.alpha_activation(pred[..., 3].unsqueeze(-1))

        rgb_loss_map = (pred_rgb - target_rgb) ** 2
        alpha_loss_map = (pred_alpha - target_alpha) ** 2

        rgb_mask = self._build_rgb_mask(full_target_alpha, removed_mask)
        alpha_mask = self._build_alpha_mask(valid_mask, removed_mask)

        if rgb_mask is None or self.probe_rgb_weight <= 0:
            loss_rgb = pred_rgb.new_zeros(())
        else:
            loss_rgb = self._masked_mean(rgb_loss_map, rgb_mask)

        if alpha_mask is None or self.probe_alpha_weight <= 0:
            loss_alpha = pred_alpha.new_zeros(())
        else:
            loss_alpha = self._masked_mean(alpha_loss_map, alpha_mask)

        loss = self.probe_rgb_weight * loss_rgb + self.probe_alpha_weight * loss_alpha

        if is_eval:
            occupied_mask = full_target_alpha > self.probe_alpha_threshold
            return loss, loss_rgb, loss_alpha, pred, occupied_mask, target
        return loss, loss_rgb, loss_alpha
