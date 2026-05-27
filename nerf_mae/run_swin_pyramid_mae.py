"""Entry point for Pyramid Alpha/RGB target curriculum scouts.

Run from the `nerf_mae/` directory exactly like `run_swin_mae3d.py`, but set
pyramid options via environment variables, e.g.

PYR_MODE=both PYR_SCALE=2 PYR_SCHEDULE=cosine \
python run_swin_pyramid_mae.py --mode train ...

This wrapper avoids invasive edits to `run_swin_mae3d.py`: it monkey-patches the
model class used by the existing trainer and calls the original `main()`.
"""
from __future__ import annotations

import logging
import os

import run_swin_mae3d as base
from model.mae.pyramid_probe import SwinTransformer_MAE3D_Pyramid

base.SwinTransformer_MAE3D = SwinTransformer_MAE3D_Pyramid

_orig_apply_probe_curriculum = base.Trainer.apply_probe_curriculum


def _apply_probe_curriculum_and_pyramid(self, epoch):
    _orig_apply_probe_curriculum(self, epoch)
    model = self._probe_model()
    if hasattr(model, "set_pyramid_epoch"):
        model.set_pyramid_epoch(epoch, self.args.num_epochs)
        if self.rank == 0 and (
            epoch == 1 or epoch == self.args.num_epochs or epoch % max(1, self.args.eval_interval) == 0
        ):
            stats = getattr(model, "pyramid_stats", {})
            self.logger.info(
                "pyramid_mae epoch=%s mode=%s scale=%s schedule=%s progress=%.6f stats=%s",
                epoch,
                os.environ.get("PYR_MODE", "off"),
                os.environ.get("PYR_SCALE", "2"),
                os.environ.get("PYR_SCHEDULE", "cosine"),
                getattr(model, "_pyr_progress", 1.0),
                stats,
            )


base.Trainer.apply_probe_curriculum = _apply_probe_curriculum_and_pyramid

if __name__ == "__main__":
    logging.info("Using pyramid-target wrapper entrypoint.")
    base.main()
