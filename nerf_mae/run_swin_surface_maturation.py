"""Entry point for surface-maturation / input-alpha-curriculum scouts.

Run from the `nerf_mae/` directory exactly like `run_swin_mae3d.py`, but set
surface-maturation options via environment variables, e.g.

SM_MODE=surface_maturation SM_W_MIN=0.05 SM_TAU=0.5 SM_K=20 \
python run_swin_surface_maturation.py --mode train ...

This wrapper avoids invasive edits to `run_swin_mae3d.py`: it monkey-patches the
model class used by the existing trainer and calls the original `main()`.
"""
from __future__ import annotations

import logging
import os

import run_swin_mae3d as base
from model.mae.surface_maturation_probe import SwinTransformer_MAE3D_SurfaceMaturation

base.SwinTransformer_MAE3D = SwinTransformer_MAE3D_SurfaceMaturation

_orig_apply_probe_curriculum = base.Trainer.apply_probe_curriculum


def _apply_probe_curriculum_and_surface(self, epoch):
    _orig_apply_probe_curriculum(self, epoch)
    model = self._probe_model()
    if hasattr(model, "set_surface_maturation_epoch"):
        model.set_surface_maturation_epoch(epoch, self.args.num_epochs)
        if self.rank == 0 and (epoch == 1 or epoch == self.args.num_epochs or epoch % max(1, self.args.eval_interval) == 0):
            stats = getattr(model, "surface_stats", {})
            self.logger.info(
                "surface_maturation epoch=%s mode=%s rgb_scale=%.6f w_min=%s tau=%s k=%s stats=%s",
                epoch,
                os.environ.get("SM_MODE", "off"),
                getattr(model, "_input_rgb_scale", 1.0),
                os.environ.get("SM_W_MIN", "0.05"),
                os.environ.get("SM_TAU", "0.5"),
                os.environ.get("SM_K", "20"),
                stats,
            )


base.Trainer.apply_probe_curriculum = _apply_probe_curriculum_and_surface

if __name__ == "__main__":
    logging.info("Using surface-maturation wrapper entrypoint.")
    base.main()
