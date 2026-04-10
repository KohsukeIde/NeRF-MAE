"""Shortcut-probe wrapper for the public NeRF-MAE training code.

This module adds shortcut probes motivated by the NeRF-MAE shortcut
hypothesis discussed in the chat:

1. masked_only_rgb_loss
   Keep the original input, but only supervise RGB on masked occupied voxels.
2. alpha_only
   Zero RGB before encoding and optimize alpha only.
3. radiance_only
   Zero alpha before encoding and optimize RGB only on masked occupied voxels.
4. custom target-side probes
   Keep the public architecture but corrupt the alpha target used in the loss.

The implementation is intentionally conservative: it subclasses the public
`SwinTransformer_MAE3D_New` model instead of modifying the original model file.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch

try:  # Training entrypoint when executed from `nerf_mae/`
    from model.mae.swin_mae3d import SwinTransformer_MAE3D_New
except ImportError:  # Package-style import for tests or external use
    from nerf_mae.model.mae.swin_mae3d import SwinTransformer_MAE3D_New


class SwinTransformer_MAE3D_Probe(SwinTransformer_MAE3D_New):
    """Drop-in replacement for the public NeRF-MAE model with probe controls."""

    PROBE_MODES = (
        "baseline",
        "masked_only_rgb_loss",
        "alpha_only",
        "radiance_only",
        "custom",
    )
    INPUT_MODES = ("keep", "zero", "shuffle")
    TARGET_ALPHA_MODES = ("keep", "zero", "shuffle")
    RGB_LOSS_MODES = ("occupied", "removed_occupied", "removed_all", "none")
    ALPHA_LOSS_MODES = ("removed", "all", "none")

    def __init__(
        self,
        *args,
        probe_mode: str = "baseline",
        probe_rgb_input: str = "keep",
        probe_alpha_input: str = "keep",
        probe_alpha_target: str = "keep",
        probe_rgb_loss: str = "occupied",
        probe_alpha_loss: str = "removed",
        probe_alpha_threshold: float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._validate_choice("probe_mode", probe_mode, self.PROBE_MODES)
        self.probe_mode = probe_mode
        self.probe_alpha_threshold = float(probe_alpha_threshold)

        resolved = self._resolve_probe_defaults(
            probe_mode=probe_mode,
            probe_rgb_input=probe_rgb_input,
            probe_alpha_input=probe_alpha_input,
            probe_alpha_target=probe_alpha_target,
            probe_rgb_loss=probe_rgb_loss,
            probe_alpha_loss=probe_alpha_loss,
        )
        self.probe_rgb_input = resolved["probe_rgb_input"]
        self.probe_alpha_input = resolved["probe_alpha_input"]
        self.probe_alpha_target = resolved["probe_alpha_target"]
        self.probe_rgb_loss = resolved["probe_rgb_loss"]
        self.probe_alpha_loss = resolved["probe_alpha_loss"]

    @staticmethod
    def _validate_choice(name: str, value: str, valid: Iterable[str]) -> None:
        if value not in valid:
            raise ValueError(f"{name} must be one of {tuple(valid)}, got {value!r}")

    def _resolve_probe_defaults(
        self,
        *,
        probe_mode: str,
        probe_rgb_input: str,
        probe_alpha_input: str,
        probe_alpha_target: str,
        probe_rgb_loss: str,
        probe_alpha_loss: str,
    ) -> Dict[str, str]:
        for name, value, valid in (
            ("probe_rgb_input", probe_rgb_input, self.INPUT_MODES),
            ("probe_alpha_input", probe_alpha_input, self.INPUT_MODES),
            ("probe_alpha_target", probe_alpha_target, self.TARGET_ALPHA_MODES),
            ("probe_rgb_loss", probe_rgb_loss, self.RGB_LOSS_MODES),
            ("probe_alpha_loss", probe_alpha_loss, self.ALPHA_LOSS_MODES),
        ):
            self._validate_choice(name, value, valid)

        if probe_mode == "baseline":
            return {
                "probe_rgb_input": "keep",
                "probe_alpha_input": "keep",
                "probe_alpha_target": "keep",
                "probe_rgb_loss": "occupied",
                "probe_alpha_loss": "removed",
            }
        if probe_mode == "masked_only_rgb_loss":
            return {
                "probe_rgb_input": "keep",
                "probe_alpha_input": "keep",
                "probe_alpha_target": "keep",
                "probe_rgb_loss": "removed_occupied",
                "probe_alpha_loss": "removed",
            }
        if probe_mode == "alpha_only":
            return {
                "probe_rgb_input": "zero",
                "probe_alpha_input": "keep",
                "probe_alpha_target": "keep",
                "probe_rgb_loss": "none",
                "probe_alpha_loss": "removed",
            }
        if probe_mode == "radiance_only":
            return {
                "probe_rgb_input": "keep",
                "probe_alpha_input": "zero",
                "probe_alpha_target": "keep",
                "probe_rgb_loss": "removed_occupied",
                "probe_alpha_loss": "none",
            }
        return {
            "probe_rgb_input": probe_rgb_input,
            "probe_alpha_input": probe_alpha_input,
            "probe_alpha_target": probe_alpha_target,
            "probe_rgb_loss": probe_rgb_loss,
            "probe_alpha_loss": probe_alpha_loss,
        }

    @staticmethod
    def _masked_mean(loss_map: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return loss_map.new_zeros(())
        mask = mask.to(loss_map.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (loss_map * mask).sum() / denom

    @staticmethod
    def _shuffle_spatial_(x: torch.Tensor, channel_slice: slice) -> None:
        selected = x[:, channel_slice, ...].clone().contiguous()
        batch_size = selected.shape[0]
        num_channels = selected.shape[1]
        flat = selected.reshape(batch_size, num_channels, -1)
        flat_size = flat.shape[-1]

        for b in range(batch_size):
            perm = torch.randperm(flat_size, device=x.device)
            flat[b] = flat[b, :, perm]

        x[:, channel_slice, ...] = flat.reshape_as(selected)

    def _apply_probe_input_corruption(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        if self.probe_rgb_input == "zero":
            x[:, :3, ...] = 0
        elif self.probe_rgb_input == "shuffle":
            self._shuffle_spatial_(x, slice(0, 3))

        if self.probe_alpha_input == "zero":
            x[:, 3:4, ...] = 0
        elif self.probe_alpha_input == "shuffle":
            self._shuffle_spatial_(x, slice(3, 4))
        return x.contiguous()

    def _build_rgb_mask(
        self, target_alpha: torch.Tensor, removed_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        occupied_mask = (target_alpha > self.probe_alpha_threshold).to(target_alpha.dtype)
        if self.probe_rgb_loss == "occupied":
            return occupied_mask
        if self.probe_rgb_loss == "removed_occupied":
            return occupied_mask * removed_mask
        if self.probe_rgb_loss == "removed_all":
            return removed_mask
        if self.probe_rgb_loss == "none":
            return None
        raise ValueError(f"Unsupported probe_rgb_loss: {self.probe_rgb_loss}")

    def _build_alpha_mask(
        self, valid_mask: torch.Tensor, removed_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.probe_alpha_loss == "removed":
            return removed_mask
        if self.probe_alpha_loss == "all":
            return valid_mask
        if self.probe_alpha_loss == "none":
            return None
        raise ValueError(f"Unsupported probe_alpha_loss: {self.probe_alpha_loss}")

    def _apply_probe_alpha_target_corruption(self, target_alpha: torch.Tensor) -> torch.Tensor:
        if self.probe_alpha_target == "keep":
            return target_alpha

        target_alpha = target_alpha.clone()
        if self.probe_alpha_target == "zero":
            target_alpha.zero_()
            return target_alpha

        if self.probe_alpha_target == "shuffle":
            batch_size = target_alpha.shape[0]
            flat = target_alpha.reshape(batch_size, -1, target_alpha.shape[-1])
            for b in range(batch_size):
                perm = torch.randperm(flat.shape[1], device=flat.device)
                flat[b] = flat[b, perm]
            return flat.reshape_as(target_alpha)

        raise ValueError(f"Unsupported probe_alpha_target: {self.probe_alpha_target}")

    def forward_loss(self, x, pred, mask_batch, mask_patches, is_eval=False):
        """Compute the probe-aware reconstruction loss.

        Notes
        -----
        `x` is the *uncorrupted* target volume. Input corruption is only applied to
        the encoder pathway in `forward`.
        """
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

        rgb_mask = self._build_rgb_mask(target_alpha, removed_mask)
        alpha_mask = self._build_alpha_mask(valid_mask, removed_mask)

        loss_rgb = self._masked_mean(rgb_loss_map, rgb_mask)
        loss_alpha = self._masked_mean(alpha_loss_map, alpha_mask)
        loss = loss_rgb + loss_alpha

        if is_eval:
            occupied_mask = target_alpha > self.probe_alpha_threshold
            return (
                loss,
                loss_rgb,
                loss_alpha,
                pred,
                occupied_mask,
                target,
            )
        return loss, loss_rgb, loss_alpha

    def forward(self, x, is_eval=False):
        padded_x, mask = self.transform(x)
        x_target = torch.cat(tuple(padded_x), dim=0)
        mask = torch.cat(tuple(mask), dim=0).to(x_target.device)

        x_model = self._apply_probe_input_corruption(x_target)
        pred, mask_patches = self.forward_encoder_ecoder(x_model)
        if is_eval:
            loss, loss_rgb, loss_alpha, pred_patches, occ_mask, target_patches = self.forward_loss(
                x_target, pred, mask, mask_patches, is_eval=True
            )
            return (
                loss,
                loss_rgb,
                loss_alpha,
                pred_patches,
                occ_mask,
                target_patches,
            )
        return self.forward_loss(x_target, pred, mask, mask_patches, is_eval=False)
