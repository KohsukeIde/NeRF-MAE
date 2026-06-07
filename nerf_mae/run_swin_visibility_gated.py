"""Entry point for Visibility-Gated NeRF-MAE scouts."""

from __future__ import annotations

import logging
import os

import run_swin_mae3d as base
from model.mae.visibility_gated_probe import SwinTransformer_MAE3D_VisibilityGated

base.SwinTransformer_MAE3D = SwinTransformer_MAE3D_VisibilityGated

_orig_apply_probe_curriculum = base.Trainer.apply_probe_curriculum


def _apply_probe_curriculum_and_log_visibility(self, epoch):
    _orig_apply_probe_curriculum(self, epoch)
    if self.rank != 0:
        return
    try:
        model = self._probe_model()
    except Exception:
        model = getattr(self, "model", None)
    if model is None:
        return
    if hasattr(model, "visibility_stats") and (
        epoch == 1 or epoch == self.args.num_epochs or epoch % max(1, self.args.eval_interval) == 0
    ):
        self.logger.info(
            "visibility_gate epoch=%s mode=%s reset_stages=%s skip_stages=%s stats=%s",
            epoch,
            os.environ.get("VISGATE_MODE", "off"),
            os.environ.get("VISGATE_RESET_STAGES", "0,1,2"),
            os.environ.get("VISGATE_SKIP_STAGES", "0,1,2"),
            getattr(model, "visibility_stats", {}),
        )


base.Trainer.apply_probe_curriculum = _apply_probe_curriculum_and_log_visibility

if __name__ == "__main__":
    logging.info("Using Visibility-Gated wrapper entrypoint.")
    base.main()
