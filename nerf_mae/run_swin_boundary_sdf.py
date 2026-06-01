"""Entry point for Boundary-SDF auxiliary scouts."""

from __future__ import annotations

import logging
import os

import run_swin_mae3d as base
from model.mae.boundary_sdf_probe import SwinTransformer_MAE3D_BoundarySDF

base.SwinTransformer_MAE3D = SwinTransformer_MAE3D_BoundarySDF

_orig_apply_probe_curriculum = base.Trainer.apply_probe_curriculum


def _apply_probe_curriculum_and_boundary(self, epoch):
    _orig_apply_probe_curriculum(self, epoch)
    model = self._probe_model()
    if hasattr(model, "boundary_stats") and self.rank == 0:
        if epoch == 1 or epoch == self.args.num_epochs or epoch % max(1, self.args.eval_interval) == 0:
            self.logger.info(
                "boundary_sdf epoch=%s mode=%s weight=%s sigma=%s threshold=%s clip=%s stats=%s",
                epoch,
                os.environ.get("BOUNDARY_SDF_MODE", "sdf_aux"),
                os.environ.get("BOUNDARY_SDF_WEIGHT", "0.2"),
                os.environ.get("BOUNDARY_ALPHA_SMOOTH_SIGMA", "1.0"),
                os.environ.get("BOUNDARY_ALPHA_THRESHOLD", "0.02"),
                os.environ.get("BOUNDARY_DISTANCE_CLIP", "16"),
                getattr(model, "boundary_stats", {}),
            )


base.Trainer.apply_probe_curriculum = _apply_probe_curriculum_and_boundary

if __name__ == "__main__":
    logging.info("Using Boundary-SDF wrapper entrypoint.")
    base.main()
