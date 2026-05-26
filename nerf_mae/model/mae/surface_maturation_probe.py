"""Surface-Maturation probe wrapper for NeRF-MAE.

This module is intentionally an add-on: it does not edit the public
`shortcut_probe.py`.  It subclasses `SwinTransformer_MAE3D_Probe` and can be
activated from a wrapper entrypoint via environment variables.

Motivation
----------
The existing shortcut probes suggest that target-side alpha structure is a
sample-efficient signal, while full RGBA reconstruction is useful at larger
budgets.  This wrapper tests a less degenerate method: keep the same visible
input as NeRF-MAE, predict alpha, and use the model's own alpha confidence to
route RGB reconstruction.  RGB loss is never fully removed unless requested;
`SM_W_MIN` gives a floor to avoid the alpha-only chicken-and-egg failure mode.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch

try:
    from model.mae.shortcut_probe import SwinTransformer_MAE3D_Probe
except ImportError:  # pragma: no cover
    from nerf_mae.model.mae.shortcut_probe import SwinTransformer_MAE3D_Probe


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    return int(value)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return str(value)


class SwinTransformer_MAE3D_SurfaceMaturation(SwinTransformer_MAE3D_Probe):
    """Probe wrapper implementing surface-maturation RGB routing.

    Environment variables
    ---------------------
    SM_MODE:
        "off" | "surface_maturation".  Default: "off".
    SM_CONFIDENCE:
        "raw_alpha" or "binary_confidence".  Default: "raw_alpha".
        raw_alpha gates RGB mostly on predicted occupied/surface voxels.
        binary_confidence uses 2*|alpha-0.5| and opens RGB when the model is
        confident about either empty or occupied voxels.
    SM_W_MIN:
        RGB gate floor. Default: 0.05.  Use 0 for strict maturation.
    SM_TAU:
        Confidence threshold in the sigmoid gate. Default: 0.5.
    SM_K:
        Sigmoid steepness. Default: 20.
    SM_STOP_GATE_GRAD:
        If true, detach the gate from the RGB loss. Default: true.
    SM_RGB_MASK:
        Override RGB base mask: occupied | removed_occupied | removed_all.
        Default: uses self.probe_rgb_loss, normally "occupied".
    SM_INPUT_RGB_CURRICULUM:
        none | linear_release | cosine_release.  Optional input-side alpha
        curriculum: multiply visible RGB input by a scheduled scalar while
        keeping alpha input. Default: none.
    SM_INPUT_RGB_START / SM_INPUT_RGB_END / SM_INPUT_RGB_EPOCHS:
        Schedule parameters for input RGB release.
    SM_LOG_STATS:
        If true, store gate statistics in self.surface_stats for logging.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.surface_mode = _env_str("SM_MODE", "off")
        self.surface_confidence = _env_str("SM_CONFIDENCE", "raw_alpha")
        self.surface_w_min = _env_float("SM_W_MIN", 0.05)
        self.surface_tau = _env_float("SM_TAU", 0.5)
        self.surface_k = _env_float("SM_K", 20.0)
        self.surface_stop_gate_grad = _env_flag("SM_STOP_GATE_GRAD", True)
        self.surface_rgb_mask_override = _env_str("SM_RGB_MASK", "")
        self.surface_log_stats = _env_flag("SM_LOG_STATS", False)

        self.input_rgb_curriculum = _env_str("SM_INPUT_RGB_CURRICULUM", "none")
        self.input_rgb_start = _env_float("SM_INPUT_RGB_START", 0.0)
        self.input_rgb_end = _env_float("SM_INPUT_RGB_END", 1.0)
        self.input_rgb_epochs = _env_int("SM_INPUT_RGB_EPOCHS", 0)
        self._surface_epoch = 1
        self._surface_total_epochs = 1
        self._input_rgb_scale = 1.0
        self.surface_stats = {}

        if self.surface_mode not in {"off", "surface_maturation"}:
            raise ValueError(f"Unsupported SM_MODE={self.surface_mode!r}")
        if self.surface_confidence not in {"raw_alpha", "binary_confidence"}:
            raise ValueError(f"Unsupported SM_CONFIDENCE={self.surface_confidence!r}")
        if self.input_rgb_curriculum not in {"none", "linear_release", "cosine_release"}:
            raise ValueError(
                f"Unsupported SM_INPUT_RGB_CURRICULUM={self.input_rgb_curriculum!r}"
            )

    def set_surface_maturation_epoch(self, epoch: int, total_epochs: int) -> None:
        self._surface_epoch = int(epoch)
        self._surface_total_epochs = max(1, int(total_epochs))
        self._input_rgb_scale = self._compute_input_rgb_scale(epoch, total_epochs)

    def _compute_input_rgb_scale(self, epoch: int, total_epochs: int) -> float:
        if self.input_rgb_curriculum == "none":
            return 1.0
        ramp_epochs = self.input_rgb_epochs if self.input_rgb_epochs > 0 else total_epochs
        if ramp_epochs <= 1:
            progress = 1.0
        else:
            progress = min(max((int(epoch) - 1) / float(ramp_epochs - 1), 0.0), 1.0)
        if self.input_rgb_curriculum == "cosine_release":
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        return float(self.input_rgb_start + (self.input_rgb_end - self.input_rgb_start) * progress)

    def _apply_probe_input_corruption(self, x: torch.Tensor) -> torch.Tensor:
        x = super()._apply_probe_input_corruption(x)
        if self.input_rgb_curriculum != "none":
            x = x.clone()
            x[:, :3, ...] = x[:, :3, ...] * float(self._input_rgb_scale)
        return x

    def _alpha_confidence(self, pred_alpha: torch.Tensor) -> torch.Tensor:
        if self.surface_confidence == "raw_alpha":
            return pred_alpha.clamp(0.0, 1.0)
        # confidence for either empty or occupied prediction
        return (2.0 * (pred_alpha - 0.5).abs()).clamp(0.0, 1.0)

    def _surface_gate(self, pred_alpha: torch.Tensor) -> torch.Tensor:
        conf = self._alpha_confidence(pred_alpha)
        if self.surface_stop_gate_grad:
            conf = conf.detach()
        gate = torch.sigmoid(float(self.surface_k) * (conf - float(self.surface_tau)))
        w_min = float(self.surface_w_min)
        if w_min > 0.0:
            gate = w_min + (1.0 - w_min) * gate
        if self.surface_log_stats:
            with torch.no_grad():
                self.surface_stats = {
                    "gate_mean": float(gate.mean().detach().cpu()),
                    "gate_min": float(gate.min().detach().cpu()),
                    "gate_max": float(gate.max().detach().cpu()),
                    "conf_mean": float(conf.mean().detach().cpu()),
                    "input_rgb_scale": float(self._input_rgb_scale),
                }
        return gate

    def _build_surface_rgb_mask(
        self, target_alpha: torch.Tensor, removed_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.surface_rgb_mask_override:
            old = self.probe_rgb_loss
            try:
                self.probe_rgb_loss = self.surface_rgb_mask_override
                return self._build_rgb_mask(target_alpha, removed_mask)
            finally:
                self.probe_rgb_loss = old
        return self._build_rgb_mask(target_alpha, removed_mask)

    @staticmethod
    def _masked_gated_mean(
        loss_map: torch.Tensor,
        mask: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Average gated RGB loss over the base mask support.

        `_masked_mean(loss, mask * gate)` would divide by the gate mass, which
        turns the gate into a pure spatial reweighting. Surface-Maturation uses
        the gate as an actual RGB-loss floor/router, so the denominator is the
        ungated supervised support.
        """

        mask = mask.to(loss_map.dtype)
        gate = gate.to(loss_map.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (loss_map * mask * gate).sum() / denom

    def forward_loss(self, x, pred, mask_batch, mask_patches, is_eval=False):
        if self.surface_mode == "off":
            return super().forward_loss(x, pred, mask_batch, mask_patches, is_eval=is_eval)

        target, valid_mask = self.patchify_3d(x, mask_batch)
        pred = self.patchify_3d(pred)
        removed_mask = valid_mask.squeeze(-1).int() * mask_patches
        removed_mask = removed_mask.unsqueeze(-1).to(pred.dtype)
        valid_mask = valid_mask.to(pred.dtype)

        target_rgb = target[..., :3]
        target_alpha = target[..., 3].unsqueeze(-1)
        target_alpha = self._apply_probe_alpha_target_corruption(target_alpha)

        pred_rgb = pred[..., :3]
        pred_alpha = self.alpha_activation(pred[..., 3].unsqueeze(-1))

        rgb_loss_map = (pred_rgb - target_rgb) ** 2
        alpha_loss_map = (pred_alpha - target_alpha) ** 2

        base_rgb_mask = self._build_surface_rgb_mask(target_alpha, removed_mask)
        alpha_mask = self._build_alpha_mask(valid_mask, removed_mask)

        if base_rgb_mask is None or self.probe_rgb_weight <= 0:
            loss_rgb = pred_rgb.new_zeros(())
        else:
            gate = self._surface_gate(pred_alpha).to(rgb_loss_map.dtype)
            loss_rgb = self._masked_gated_mean(rgb_loss_map, base_rgb_mask, gate)

        loss_alpha = self._masked_mean(alpha_loss_map, alpha_mask)
        loss = self.probe_rgb_weight * loss_rgb + self.probe_alpha_weight * loss_alpha

        if is_eval:
            occupied_mask = target_alpha > self.probe_alpha_threshold
            return loss, loss_rgb, loss_alpha, pred, occupied_mask, target
        return loss, loss_rgb, loss_alpha
