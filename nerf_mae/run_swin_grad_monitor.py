"""Gradient-conflict monitoring entrypoint for NeRF-MAE.

This wrapper logs cosine similarity between RGB and alpha reconstruction gradients
without changing the training objective.  It monkey-patches Trainer.train_epoch,
so use it only for short diagnostic runs.

Environment variables:
  GM_INTERVAL: monitor every N iterations. default=50
  GM_MAX_BATCHES: stop monitoring after this many monitored batches per epoch. default=2
  GM_PARAM_FILTER: substring filter for parameter names. default="stages"
"""
from __future__ import annotations

import os
import torch

import run_swin_mae3d as base


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v in (None, "") else int(v)


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v in (None, "") else str(v)


def _grad_vector(grads):
    flats = []
    for g in grads:
        if g is not None:
            flats.append(g.detach().flatten())
    if not flats:
        return None
    return torch.cat(flats)


def _train_epoch_with_grad_monitor(self, epoch):
    self.apply_probe_curriculum(epoch)
    interval = max(1, _env_int("GM_INTERVAL", 50))
    max_batches = max(1, _env_int("GM_MAX_BATCHES", 2))
    name_filter = _env_str("GM_PARAM_FILTER", "stages")
    monitored = 0

    for i, batch in enumerate(self.train_loader):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        rgbsigma, _, _ = batch
        if torch.cuda.is_available():
            rgbsigma = [item.cuda() for item in rgbsigma]
        loss, loss_rgb, loss_alpha = self.model(rgbsigma)

        if i % interval == 0 and monitored < max_batches and torch.is_tensor(loss_alpha):
            params = []
            for name, p in self.model.named_parameters():
                if p.requires_grad and (not name_filter or name_filter in name):
                    params.append(p)
            try:
                g_rgb = torch.autograd.grad(loss_rgb, params, retain_graph=True, allow_unused=True)
                g_alpha = torch.autograd.grad(loss_alpha, params, retain_graph=True, allow_unused=True)
                v_rgb = _grad_vector(g_rgb)
                v_alpha = _grad_vector(g_alpha)
                if v_rgb is not None and v_alpha is not None:
                    denom = (v_rgb.norm() * v_alpha.norm()).clamp_min(1e-12)
                    cos = torch.dot(v_rgb, v_alpha) / denom
                    if self.rank == 0:
                        self.logger.info(
                            "grad_conflict epoch=%s iter=%s filter=%s cos_rgb_alpha=%.6f norm_rgb=%.6f norm_alpha=%.6f",
                            epoch,
                            i,
                            name_filter,
                            float(cos.detach().cpu()),
                            float(v_rgb.norm().detach().cpu()),
                            float(v_alpha.norm().detach().cpu()),
                        )
                    monitored += 1
            except RuntimeError as exc:
                if self.rank == 0:
                    self.logger.warning("grad_conflict failed at epoch=%s iter=%s: %s", epoch, i, exc)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        should_report = i % self.args.log_interval == 0 or self.args.wandb
        report_loss = loss.detach()
        report_loss_rgb = loss_rgb.detach()
        report_loss_alpha = loss_alpha.detach() if torch.is_tensor(loss_alpha) else None

        if should_report and self.world_size > 1:
            report_loss = report_loss.clone()
            base.dist.all_reduce(report_loss)
            report_loss /= self.world_size

            report_loss_rgb = report_loss_rgb.clone()
            base.dist.all_reduce(report_loss_rgb)
            report_loss_rgb /= self.world_size

            if report_loss_alpha is not None:
                report_loss_alpha = report_loss_alpha.clone()
                base.dist.all_reduce(report_loss_alpha)
                report_loss_alpha /= self.world_size

        if i % self.args.log_interval == 0 and self.rank == 0:
            self.logger.info(
                f"epoch {epoch} [{i}/{len(self.train_loader)}] "
                f"lr: {self.scheduler.get_last_lr()[0]:.6f} "
                f"loss: {report_loss.item():.4f}"
            )
            if self.args.wandb:
                base.wandb.log({
                    "lr": self.scheduler.get_last_lr()[0],
                    "loss_recon": report_loss.item(),
                    "loss_rgb": report_loss_rgb.item(),
                    "epoch": epoch,
                    "iter": i,
                })
                base.wandb.log({
                    "loss_alpha": report_loss_alpha.item()
                    if report_loss_alpha is not None
                    else 0.0
                })


base.Trainer.train_epoch = _train_epoch_with_grad_monitor

if __name__ == "__main__":
    base.main()
