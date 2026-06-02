"""MixNeRF-MAE-lite wrapper for NeRF-MAE.

This is an implementation-focused MVP for the *encoder-side* hypothesis:
NeRF-MAE uses a SimMIM/UNETR-style full-grid encoder in which masked positions are
visible to the encoder as zeros/mask placeholders.  MixNeRF-MAE-lite replaces the
would-be masked patches of a target scene with patches from another scene, preserving
3D Swin's dense grid while removing the explicit zero/mask-token shortcut.

The design is intentionally conservative:
- subclasses the existing shortcut-probe wrapper;
- keeps the public NeRF-MAE loss implementation via `forward_loss`;
- creates its own patch-level mask and disables the base model's internal masking
  when possible;
- computes reconstruction loss on the target scene at the mixed/removed positions.

Environment variables
---------------------
MIXNERF_MODE:
    off | mix. Default: off.
MIXNERF_MASK_RATIO:
    Fraction of target patches replaced by partner patches. Default: 0.75.
MIXNERF_PARTNER:
    roll | shuffle. Default: roll.
MIXNERF_FILL_MODE:
    partner | zeros | noise. Default: partner. `zeros` and `noise` are controls.
MIXNERF_PATCH_SIZE:
    Patch size used to upsample patch masks to voxel masks. Default: auto, then 4.
MIXNERF_DISABLE_INTERNAL_MASK:
    If true, temporarily sets known base-model mask attributes to 0 while encoding
    the mixed volume. Default: true. If no such attribute exists, this wrapper still
    uses its own mix mask for loss and logs a warning in `mixnerf_stats`.
MIXNERF_LOG_STATS:
    If true, records mask/fill stats in `self.mixnerf_stats`.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

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
    return default if value is None or value == "" else float(value)


def _env_int_or_none(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "" or value.lower() == "auto":
        return None
    return int(value)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else str(value)


class SwinTransformer_MAE3D_MixNeRF(SwinTransformer_MAE3D_Probe):
    """Drop-in MixNeRF-MAE-lite training wrapper.

    The wrapper constructs a target scene `A` and a partner scene `B` within the
    batch.  A random patch mask `m` defines which patches of A are replaced by B.
    The model encodes the mixed dense grid and reconstructs the original target A;
    reconstruction loss is applied on `m`, using the existing probe-aware loss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixnerf_mode = _env_str("MIXNERF_MODE", "off")
        self.mixnerf_mask_ratio = _env_float("MIXNERF_MASK_RATIO", 0.75)
        self.mixnerf_partner = _env_str("MIXNERF_PARTNER", "roll")
        self.mixnerf_fill_mode = _env_str("MIXNERF_FILL_MODE", "partner")
        self.mixnerf_patch_size = _env_int_or_none("MIXNERF_PATCH_SIZE")
        self.mixnerf_disable_internal_mask = _env_flag("MIXNERF_DISABLE_INTERNAL_MASK", True)
        self.mixnerf_log_stats = _env_flag("MIXNERF_LOG_STATS", False)
        self.mixnerf_stats: Dict[str, object] = {}

        if self.mixnerf_mode not in {"off", "mix"}:
            raise ValueError(f"Unsupported MIXNERF_MODE={self.mixnerf_mode!r}")
        if self.mixnerf_partner not in {"roll", "shuffle"}:
            raise ValueError(f"Unsupported MIXNERF_PARTNER={self.mixnerf_partner!r}")
        if self.mixnerf_fill_mode not in {"partner", "zeros", "noise"}:
            raise ValueError(f"Unsupported MIXNERF_FILL_MODE={self.mixnerf_fill_mode!r}")
        if not 0.0 <= self.mixnerf_mask_ratio < 1.0:
            raise ValueError("MIXNERF_MASK_RATIO must be in [0, 1).")

    def _infer_patch_size(self, x: torch.Tensor) -> int:
        if self.mixnerf_patch_size is not None:
            return int(self.mixnerf_patch_size)
        # NeRF-MAE paper/code use patch size 4.  Try common attribute names first.
        for attr in ("patch_size", "p", "patch_dim"):
            if hasattr(self, attr):
                value = getattr(self, attr)
                if isinstance(value, (tuple, list)):
                    return int(value[0])
                try:
                    return int(value)
                except Exception:
                    pass
        return 4

    def _make_partner_index(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if batch_size < 2:
            return torch.arange(batch_size, device=device)
        if self.mixnerf_partner == "roll":
            return torch.roll(torch.arange(batch_size, device=device), shifts=1, dims=0)
        return torch.randperm(batch_size, device=device)

    def _make_patch_mask(self, x: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        # x: [B, C, H, W, D]
        b, _, h, w, d = x.shape
        gh, gw, gd = h // patch_size, w // patch_size, d // patch_size
        if min(gh, gw, gd) <= 0:
            raise ValueError(f"Invalid patch grid for x.shape={tuple(x.shape)}, p={patch_size}")
        n = gh * gw * gd
        num_mask = int(round(n * self.mixnerf_mask_ratio))
        noise = torch.rand(b, n, device=x.device)
        ids = torch.argsort(noise, dim=1)
        patch_mask = torch.zeros(b, n, dtype=torch.float32, device=x.device)
        if num_mask > 0:
            patch_mask.scatter_(1, ids[:, :num_mask], 1.0)
        patch_mask = patch_mask.reshape(b, gh, gw, gd, 1)
        return patch_mask, (gh, gw, gd)

    @staticmethod
    def _patch_mask_to_voxel_mask(
        patch_mask: torch.Tensor,
        grid_shape: Tuple[int, int, int],
        patch_size: int,
        target_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        b = patch_mask.shape[0]
        gh, gw, gd = grid_shape
        mask = patch_mask.reshape(b, gh, gw, gd, 1).permute(0, 4, 1, 2, 3).contiguous()
        mask = mask.repeat_interleave(patch_size, dim=2)
        mask = mask.repeat_interleave(patch_size, dim=3)
        mask = mask.repeat_interleave(patch_size, dim=4)
        h, w, d = target_shape
        return mask[:, :, :h, :w, :d].contiguous()

    def _mix_input(self, x_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_size = self._infer_patch_size(x_target)
        patch_mask, patch_grid = self._make_patch_mask(x_target, patch_size)
        voxel_mask = self._patch_mask_to_voxel_mask(
            patch_mask, patch_grid, patch_size, tuple(x_target.shape[-3:])
        ).to(dtype=x_target.dtype)

        if self.mixnerf_fill_mode == "partner":
            partner_idx = self._make_partner_index(x_target.shape[0], x_target.device)
            filler = x_target[partner_idx]
        elif self.mixnerf_fill_mode == "zeros":
            filler = torch.zeros_like(x_target)
            partner_idx = torch.arange(x_target.shape[0], device=x_target.device)
        else:  # noise control matched per-channel roughly within current batch
            mean = x_target.mean(dim=(0, 2, 3, 4), keepdim=True)
            std = x_target.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-6)
            filler = torch.randn_like(x_target) * std + mean
            partner_idx = torch.arange(x_target.shape[0], device=x_target.device)

        x_mixed = x_target * (1.0 - voxel_mask) + filler * voxel_mask

        if self.mixnerf_log_stats:
            with torch.no_grad():
                self.mixnerf_stats.update(
                    {
                        "mode": self.mixnerf_mode,
                        "fill_mode": self.mixnerf_fill_mode,
                        "patch_size": patch_size,
                        "patch_grid": patch_grid,
                        "patch_mask_mean": float(patch_mask.float().mean().detach().cpu()),
                        "voxel_mask_mean": float(voxel_mask.float().mean().detach().cpu()),
                        "partner_mean_self_match": float((partner_idx == torch.arange(x_target.shape[0], device=x_target.device)).float().mean().detach().cpu()),
                    }
                )
        return x_mixed.contiguous(), patch_mask

    def _set_internal_masking(self, value: float) -> Dict[str, object]:
        """Best-effort temporary override for base-model internal masking.

        Current NeRF-MAE forks usually expose one of these attributes.  If none is
        found, the wrapper still runs but the base model may apply a second mask;
        `mixnerf_stats` records that case.
        """
        candidates = [
            "masking_prob",
            "mask_ratio",
            "masking_ratio",
            "mask_ratio_train",
            "mask_prob",
        ]
        old = {}
        for name in candidates:
            if hasattr(self, name):
                old[name] = getattr(self, name)
                try:
                    setattr(self, name, value)
                except Exception:
                    pass
        if self.mixnerf_log_stats:
            self.mixnerf_stats["internal_mask_attrs_overridden"] = sorted(old.keys())
            if not old:
                self.mixnerf_stats["internal_mask_warning"] = (
                    "No known mask attribute found. Verify that forward_encoder_ecoder "
                    "does not apply an additional internal mask."
                )
        return old

    def _restore_internal_masking(self, old: Dict[str, object]) -> None:
        for name, value in old.items():
            try:
                setattr(self, name, value)
            except Exception:
                pass

    def forward(self, x, is_eval=False):
        if self.mixnerf_mode == "off" or is_eval:
            return super().forward(x, is_eval=is_eval)

        padded_x, valid = self.transform(x)
        x_target = torch.cat(tuple(padded_x), dim=0)
        valid = torch.cat(tuple(valid), dim=0).to(x_target.device)

        x_target = x_target.contiguous()
        x_mixed, mix_mask_patches = self._mix_input(x_target)
        x_mixed = self._apply_probe_input_corruption(x_mixed)

        old_mask_attrs = {}
        if self.mixnerf_disable_internal_mask:
            old_mask_attrs = self._set_internal_masking(0.0)
        try:
            pred, base_mask_patches = self.forward_encoder_ecoder(x_mixed)
        finally:
            self._restore_internal_masking(old_mask_attrs)

        if self.mixnerf_log_stats:
            try:
                self.mixnerf_stats["base_mask_mean"] = float(base_mask_patches.float().mean().detach().cpu())
            except Exception:
                self.mixnerf_stats["base_mask_mean"] = "unavailable"

        # Use the mix mask as the reconstruction target mask.  It is patch-level and
        # matches the `forward_loss` convention: 1 means removed / to reconstruct.
        return self.forward_loss(x_target, pred, valid, mix_mask_patches, is_eval=False)
