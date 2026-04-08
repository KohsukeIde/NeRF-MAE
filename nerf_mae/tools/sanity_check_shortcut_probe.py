"""Tiny smoke test for the NeRF-MAE shortcut-probe wrapper.

This does not run the full encoder/decoder. It only checks that the probe-aware
loss path works on synthetic tensors with the expected patchified shapes.
"""

from __future__ import annotations

import argparse
import torch

try:
    from model.mae.shortcut_probe import SwinTransformer_MAE3D_Probe
except ImportError:
    from nerf_mae.model.mae.shortcut_probe import SwinTransformer_MAE3D_Probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe_mode",
        default="baseline",
        choices=["baseline", "masked_only_rgb_loss", "alpha_only", "radiance_only", "custom"],
    )
    parser.add_argument("--resolution", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    model = SwinTransformer_MAE3D_Probe(
        patch_size=[4, 4, 4],
        embed_dim=24,
        depths=[1, 1, 1, 1],
        num_heads=[1, 2, 4, 8],
        window_size=[2, 2, 2],
        stochastic_depth_prob=0.0,
        expand_dim=True,
        masking_prob=0.5,
        resolution=args.resolution,
        probe_mode=args.probe_mode,
    )
    model.eval()

    batch_size = 2
    x_target = torch.rand(batch_size, 4, args.resolution, args.resolution, args.resolution)
    # Make alpha sparse-ish so the occupancy threshold matters.
    x_target[:, 3:4] = (x_target[:, 3:4] > 0.7).float() * x_target[:, 3:4]
    pred = torch.rand_like(x_target)
    valid_mask = torch.ones_like(x_target)

    patches_per_axis = args.resolution // 4
    mask_patches = torch.randint(
        low=0,
        high=2,
        size=(batch_size, patches_per_axis, patches_per_axis, patches_per_axis, 4**3),
    )

    x_model = model._apply_probe_input_corruption(x_target)
    loss, loss_rgb, loss_alpha, pred_patches, occ_mask, target_patches = model.forward_loss(
        x_target,
        pred,
        valid_mask,
        mask_patches,
        is_eval=True,
    )

    print(f"probe_mode      : {model.probe_mode}")
    print(f"rgb_input_mode  : {model.probe_rgb_input}")
    print(f"alpha_input_mode: {model.probe_alpha_input}")
    print(f"rgb_loss_mode   : {model.probe_rgb_loss}")
    print(f"alpha_loss_mode : {model.probe_alpha_loss}")
    print(f"corrupted input : {tuple(x_model.shape)}")
    print(f"pred patches    : {tuple(pred_patches.shape)}")
    print(f"target patches  : {tuple(target_patches.shape)}")
    print(f"occupancy mask  : {tuple(occ_mask.shape)}")
    print(f"loss            : {loss.item():.6f}")
    print(f"loss_rgb        : {loss_rgb.item():.6f}")
    print(f"loss_alpha      : {loss_alpha.item():.6f}")


if __name__ == "__main__":
    main()
