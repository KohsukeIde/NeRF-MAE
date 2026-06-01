"""Boundary-SDF auxiliary target probe for NeRF-MAE.

This wrapper keeps the public effective NeRF-MAE objective intact and adds a
small decoder-side SDF head for a denoised alpha-boundary target. It is intended
as a scout, not a final method implementation: downstream FCOS ignores decoder
heads, so the added ``sdf_out`` parameters are harmless unexpected keys during
backbone loading.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from einops import rearrange

try:
    from model.mae.shortcut_probe import SwinTransformer_MAE3D_Probe
    from model.mae.swin_mae3d import UnetOutBlock
except ImportError:
    from nerf_mae.model.mae.shortcut_probe import SwinTransformer_MAE3D_Probe
    from nerf_mae.model.mae.swin_mae3d import UnetOutBlock


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "")
    if value in {"", "''", '""'}:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "")
    if value in {"", "''", '""'}:
        return default
    return int(value)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "")
    if value in {"", "''", '""'}:
        return default
    return value


class SwinTransformer_MAE3D_BoundarySDF(SwinTransformer_MAE3D_Probe):
    """Swin MAE probe with an auxiliary smoothed-alpha boundary-distance head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.boundary_mode = _env_str("BOUNDARY_SDF_MODE", "sdf_aux")
        self.boundary_weight = _env_float("BOUNDARY_SDF_WEIGHT", 0.2)
        self.boundary_alpha_smooth_sigma = _env_float("BOUNDARY_ALPHA_SMOOTH_SIGMA", 1.0)
        self.boundary_alpha_threshold = _env_float("BOUNDARY_ALPHA_THRESHOLD", 0.02)
        self.boundary_distance_clip = _env_int("BOUNDARY_DISTANCE_CLIP", 16)
        self.boundary_loss = _env_str("BOUNDARY_SDF_LOSS", "mse")
        self.boundary_mask = _env_str("BOUNDARY_SDF_MASK", "removed")
        self.boundary_stats: dict[str, float] = {}

        if self.boundary_mode not in {"off", "sdf_aux"}:
            raise ValueError(f"Unsupported BOUNDARY_SDF_MODE={self.boundary_mode!r}")
        if self.boundary_loss not in {"mse", "smooth_l1"}:
            raise ValueError(f"Unsupported BOUNDARY_SDF_LOSS={self.boundary_loss!r}")
        if self.boundary_mask not in {"removed", "all"}:
            raise ValueError(f"Unsupported BOUNDARY_SDF_MASK={self.boundary_mask!r}")

        if self.boundary_mode != "off":
            self.sdf_out = UnetOutBlock(in_channels=self.embed_dim // 2, out_channels=1)

    def _patchify_scalar(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size[0]
        assert x.shape[1] == 1
        assert x.shape[2] == x.shape[3] == x.shape[4] and x.shape[2] % p == 0
        h = w = l = int(round(x.shape[2] // p))
        x = x.reshape(shape=(x.shape[0], 1, h, p, w, p, l, p))
        x = rearrange(x, "n c h p w q l r -> n h w l p q r c")
        return rearrange(x, "n h w l p q r c -> n h w l (p q r) c")

    def _smooth_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        sigma = float(self.boundary_alpha_smooth_sigma)
        if sigma <= 0:
            return alpha

        radius = max(1, int(round(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=alpha.device, dtype=alpha.dtype)
        kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum().clamp_min(1e-12)

        x = alpha
        kernels = (
            kernel.view(1, 1, -1, 1, 1),
            kernel.view(1, 1, 1, -1, 1),
            kernel.view(1, 1, 1, 1, -1),
        )
        pads = (
            (0, 0, 0, 0, radius, radius),
            (0, 0, radius, radius, 0, 0),
            (radius, radius, 0, 0, 0, 0),
        )
        for k, pad in zip(kernels, pads):
            x = F.pad(x, pad, mode="replicate")
            x = F.conv3d(x, k)
        return x

    @staticmethod
    def _erode(mask: torch.Tensor) -> torch.Tensor:
        inv = (~mask).to(torch.float32)
        has_empty_neighbor = F.max_pool3d(inv, kernel_size=3, stride=1, padding=1) > 0
        return mask & ~has_empty_neighbor

    def _approx_signed_distance(self, occupied: torch.Tensor) -> torch.Tensor:
        clip = max(1, int(self.boundary_distance_clip))
        eroded = self._erode(occupied)
        shell = occupied & ~eroded

        assigned = shell.clone()
        dilated = shell.to(torch.float32)
        dist = torch.full_like(dilated, float(clip))
        dist = torch.where(shell, torch.zeros_like(dist), dist)

        for step in range(1, clip + 1):
            dilated = F.max_pool3d(dilated, kernel_size=3, stride=1, padding=1)
            reached = dilated > 0
            new = reached & ~assigned
            dist = torch.where(new, torch.full_like(dist, float(step)), dist)
            assigned = assigned | new

        sign = torch.where(occupied, torch.full_like(dist, -1.0), torch.ones_like(dist))
        return (dist / float(clip)).clamp(0.0, 1.0) * sign

    def _boundary_target(self, target_alpha_volume: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        with torch.no_grad():
            smoothed = self._smooth_alpha(target_alpha_volume)
            occupied = smoothed > float(self.boundary_alpha_threshold)
            sdf = self._approx_signed_distance(occupied)
            eroded = self._erode(occupied)
            shell = occupied & ~eroded

            occ_ratio = occupied.to(torch.float32).mean().item()
            shell_count = shell.to(torch.float32).sum().item()
            occ_count = occupied.to(torch.float32).sum().item()
            shell_occ = shell_count / max(occ_count, 1.0)
            stats = {
                "sdf_occ_ratio": float(occ_ratio),
                "sdf_shell_occ": float(shell_occ),
                "sdf_target_abs_mean": float(sdf.abs().mean().item()),
            }
        return sdf, stats

    def forward_encoder_ecoder(self, x):
        if self.boundary_mode == "off":
            return super().forward_encoder_ecoder(x)

        x = self.patch_partition(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        x, mask_patches = self.window_masking_3d(
            x,
            p_remove=self.masking_prob,
            mask_token=self.mask_token,
        )

        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(torch.permute(x, [0, 4, 1, 2, 3]).contiguous())

        dec3 = self.decoder4(features[3], features[2])
        dec2 = self.decoder3(dec3, features[1])
        dec1 = self.decoder2(dec2, features[0])
        dec0 = self.decoder1(dec1)

        out = self.out(dec0)
        sdf = self.sdf_out(dec0)
        return torch.cat([out, sdf], dim=1), mask_patches

    def forward_loss(self, x, pred, mask_batch, mask_patches, is_eval=False):
        if self.boundary_mode == "off":
            return super().forward_loss(x, pred, mask_batch, mask_patches, is_eval=is_eval)

        pred_main = pred[:, :4, ...]
        pred_sdf_volume = pred[:, 4:5, ...]

        target, valid_mask = self.patchify_3d(x, mask_batch)
        pred_main = self.patchify_3d(pred_main)

        removed_mask = valid_mask.squeeze(-1).int() * mask_patches
        removed_mask = removed_mask.unsqueeze(-1).to(pred_main.dtype)
        valid_mask = valid_mask.to(pred_main.dtype)

        target_rgb = target[..., :3]
        target_alpha = target[..., 3].unsqueeze(-1)
        target_alpha = self._apply_probe_alpha_target_corruption(target_alpha)
        pred_rgb = pred_main[..., :3]
        pred_alpha = self.alpha_activation(pred_main[..., 3].unsqueeze(-1))

        rgb_loss_map = (pred_rgb - target_rgb) ** 2
        alpha_loss_map = (pred_alpha - target_alpha) ** 2

        rgb_mask = self._build_rgb_mask(target_alpha, removed_mask)
        alpha_mask = self._build_alpha_mask(valid_mask, removed_mask)

        loss_rgb = self._masked_mean(rgb_loss_map, rgb_mask)
        loss_alpha = self._masked_mean(alpha_loss_map, alpha_mask)

        target_sdf_volume, sdf_stats = self._boundary_target(x[:, 3:4, ...])
        target_sdf = self._patchify_scalar(target_sdf_volume)
        pred_sdf = self._patchify_scalar(pred_sdf_volume)
        if self.boundary_loss == "smooth_l1":
            sdf_loss_map = F.smooth_l1_loss(pred_sdf, target_sdf, reduction="none", beta=0.1)
        else:
            sdf_loss_map = (pred_sdf - target_sdf) ** 2
        sdf_mask = removed_mask if self.boundary_mask == "removed" else valid_mask
        loss_sdf = self._masked_mean(sdf_loss_map, sdf_mask)

        loss = (
            self.probe_rgb_weight * loss_rgb
            + self.probe_alpha_weight * loss_alpha
            + self.boundary_weight * loss_sdf
        )
        self.boundary_stats = {
            **sdf_stats,
            "sdf_loss": float(loss_sdf.detach().item()),
            "sdf_weight": float(self.boundary_weight),
        }

        if is_eval:
            occupied_mask = target_alpha > self.probe_alpha_threshold
            return (
                loss,
                loss_rgb,
                loss_alpha,
                pred_main,
                occupied_mask,
                target,
            )
        return loss, loss_rgb, loss_alpha
