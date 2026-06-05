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
    partner | zeros | noise | shuffle | shuffle_visible | mean | constant.
    Default: partner. `zeros`, `noise`, same-scene patch `shuffle`,
    visible-only same-scene `shuffle_visible`, and simple non-zero `mean` /
    `constant` are controls.
MIXNERF_CONSTANT_VALUE:
    Scalar used by `constant` fill. Default: 0.5.
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
        self.mixnerf_constant_value = _env_float("MIXNERF_CONSTANT_VALUE", 0.5)
        self.mixnerf_patch_size = _env_int_or_none("MIXNERF_PATCH_SIZE")
        self.mixnerf_disable_internal_mask = _env_flag("MIXNERF_DISABLE_INTERNAL_MASK", True)
        self.mixnerf_log_stats = _env_flag("MIXNERF_LOG_STATS", False)
        self.mixnerf_stats: Dict[str, object] = {}
        self._last_fill_stats: Dict[str, object] = {}

        if self.mixnerf_mode not in {"off", "mix"}:
            raise ValueError(f"Unsupported MIXNERF_MODE={self.mixnerf_mode!r}")
        if self.mixnerf_partner not in {"roll", "shuffle"}:
            raise ValueError(f"Unsupported MIXNERF_PARTNER={self.mixnerf_partner!r}")
        if self.mixnerf_fill_mode not in {
            "partner",
            "zeros",
            "noise",
            "shuffle",
            "shuffle_visible",
            "mean",
            "constant",
        }:
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
        self._last_fill_stats = {}
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
        elif self.mixnerf_fill_mode == "noise":
            mean = x_target.mean(dim=(0, 2, 3, 4), keepdim=True)
            std = x_target.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-6)
            filler = torch.randn_like(x_target) * std + mean
            partner_idx = torch.arange(x_target.shape[0], device=x_target.device)
        elif self.mixnerf_fill_mode == "mean":
            filler = x_target.mean(dim=(0, 2, 3, 4), keepdim=True).expand_as(x_target)
            partner_idx = torch.arange(x_target.shape[0], device=x_target.device)
        elif self.mixnerf_fill_mode == "constant":
            filler = torch.full_like(x_target, float(self.mixnerf_constant_value))
            partner_idx = torch.arange(x_target.shape[0], device=x_target.device)
        elif self.mixnerf_fill_mode == "shuffle_visible":
            filler = self._same_scene_visible_patch_shuffle(
                x_target, patch_size, patch_grid, patch_mask
            )
            partner_idx = torch.arange(x_target.shape[0], device=x_target.device)
        else:
            filler = self._same_scene_patch_shuffle(x_target, patch_size, patch_grid)
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
                        **self._last_fill_stats,
                    }
                )
        return x_mixed.contiguous(), patch_mask

    @staticmethod
    def _same_scene_patch_shuffle(
        x: torch.Tensor, patch_size: int, patch_grid: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Build a same-scene patch-shuffled filler.

        This is a non-zero, non-partner control: each scene supplies its own filler
        patches, but patch locations are shuffled so the masked location cannot be
        copied directly from the same spatial position.
        """
        b, c, h, w, d = x.shape
        gh, gw, gd = patch_grid
        hh, ww, dd = gh * patch_size, gw * patch_size, gd * patch_size
        cropped = x[:, :, :hh, :ww, :dd]
        patches = cropped.reshape(
            b, c, gh, patch_size, gw, patch_size, gd, patch_size
        )
        patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        patches = patches.reshape(b, gh * gw * gd, c, patch_size, patch_size, patch_size)

        shuffled = torch.empty_like(patches)
        for batch_idx in range(b):
            perm = torch.randperm(patches.shape[1], device=x.device)
            shuffled[batch_idx] = patches[batch_idx, perm]

        shuffled = shuffled.reshape(b, gh, gw, gd, c, patch_size, patch_size, patch_size)
        shuffled = shuffled.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        shuffled = shuffled.reshape(b, c, hh, ww, dd)
        if (hh, ww, dd) == (h, w, d):
            return shuffled
        out = x.clone()
        out[:, :, :hh, :ww, :dd] = shuffled
        return out

    def _same_scene_visible_patch_shuffle(
        self,
        x: torch.Tensor,
        patch_size: int,
        patch_grid: Tuple[int, int, int],
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build same-scene filler from visible patches only.

        `patch_mask == 1` marks positions to reconstruct.  For each masked
        location, this method samples a replacement patch from the same scene's
        visible patch set (`patch_mask == 0`).  This prevents the target masked
        patch from being copied into its own input location.
        """
        b, c, h, w, d = x.shape
        gh, gw, gd = patch_grid
        hh, ww, dd = gh * patch_size, gw * patch_size, gd * patch_size
        cropped = x[:, :, :hh, :ww, :dd]
        patches = cropped.reshape(
            b, c, gh, patch_size, gw, patch_size, gd, patch_size
        )
        patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        patches = patches.reshape(b, gh * gw * gd, c, patch_size, patch_size, patch_size)

        flat_mask = patch_mask.reshape(b, gh * gw * gd).to(dtype=torch.bool)
        filler_patches = patches.clone()
        visible_counts = []
        masked_counts = []
        fallback_count = 0

        for batch_idx in range(b):
            visible_idx = torch.nonzero(~flat_mask[batch_idx], as_tuple=False).flatten()
            masked_idx = torch.nonzero(flat_mask[batch_idx], as_tuple=False).flatten()
            visible_counts.append(int(visible_idx.numel()))
            masked_counts.append(int(masked_idx.numel()))
            if masked_idx.numel() == 0:
                continue
            if visible_idx.numel() == 0:
                # Degenerate case only possible with a 100% mask.  Fall back to
                # all non-identical patches so training does not crash.
                all_idx = torch.arange(patches.shape[1], device=x.device)
                source_pos = torch.randint(
                    low=0, high=max(1, patches.shape[1] - 1),
                    size=(masked_idx.numel(),),
                    device=x.device,
                )
                source_idx = all_idx[source_pos]
                source_idx = torch.where(source_idx >= masked_idx, source_idx + 1, source_idx)
                source_idx = source_idx.clamp_max(patches.shape[1] - 1)
                fallback_count += int(masked_idx.numel())
            else:
                source_pos = torch.randint(
                    low=0,
                    high=visible_idx.numel(),
                    size=(masked_idx.numel(),),
                    device=x.device,
                )
                source_idx = visible_idx[source_pos]
            filler_patches[batch_idx, masked_idx] = patches[batch_idx, source_idx]

        self._last_fill_stats.update(
            {
                "same_scene_fill_source": "visible_only",
                "self_replacement_rate": 0.0,
                "masked_source_rate": 0.0 if fallback_count == 0 else "fallback_non_identical",
                "visible_patch_count_min": min(visible_counts) if visible_counts else 0,
                "visible_patch_count_mean": float(sum(visible_counts) / max(1, len(visible_counts))),
                "masked_patch_count_min": min(masked_counts) if masked_counts else 0,
                "masked_patch_count_mean": float(sum(masked_counts) / max(1, len(masked_counts))),
            }
        )

        shuffled = filler_patches.reshape(
            b, gh, gw, gd, c, patch_size, patch_size, patch_size
        )
        shuffled = shuffled.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        shuffled = shuffled.reshape(b, c, hh, ww, dd)
        if (hh, ww, dd) == (h, w, d):
            return shuffled
        out = x.clone()
        out[:, :, :hh, :ww, :dd] = shuffled
        return out

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
