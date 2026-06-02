"""Entry point for MixNeRF-MAE-lite scouts.

This wrapper keeps the public `run_swin_mae3d.py` trainer intact and monkey-patches
only the model class.  Run from the `nerf_mae/` directory with the same CLI as
`run_swin_mae3d.py`.  MixNeRF controls are environment variables documented in
`model/mae/mixnerf_probe.py`.
"""
from __future__ import annotations

import logging
import os

import run_swin_mae3d as base
from model.mae.mixnerf_probe import SwinTransformer_MAE3D_MixNeRF

base.SwinTransformer_MAE3D = SwinTransformer_MAE3D_MixNeRF

_orig_apply_probe_curriculum = base.Trainer.apply_probe_curriculum


def _apply_probe_curriculum_and_log_mixnerf(self, epoch):
    _orig_apply_probe_curriculum(self, epoch)
    if self.rank != 0:
        return
    try:
        model = self._probe_model()
    except Exception:
        model = getattr(self, "model", None)
    if model is None:
        return
    if hasattr(model, "mixnerf_stats") and (
        epoch == 1 or epoch == self.args.num_epochs or epoch % max(1, self.args.eval_interval) == 0
    ):
        self.logger.info(
            "mixnerf epoch=%s mode=%s mask_ratio=%s fill=%s stats=%s",
            epoch,
            os.environ.get("MIXNERF_MODE", "off"),
            os.environ.get("MIXNERF_MASK_RATIO", "0.75"),
            os.environ.get("MIXNERF_FILL_MODE", "partner"),
            getattr(model, "mixnerf_stats", {}),
        )


base.Trainer.apply_probe_curriculum = _apply_probe_curriculum_and_log_mixnerf

if __name__ == "__main__":
    logging.info("Using MixNeRF-MAE-lite wrapper entrypoint.")
    base.main()
