# Encoder mask-path report

Repo: `/groups/gag51404/ide/vgi/NeRF-MAE`

## Pattern summary

| file | pattern | count | sample |
|---|---|---:|---|
| `nerf_mae/model/mae/swin_mae3d.py` | `forward_loss` | 3 | ` loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()         return loss      def forward_loss(self, x, pred, mask_batch, mask_patches, is_eval=False):         # for both RGB` |
| `nerf_mae/model/mae/swin_mae3d.py` | `forward` | 5 | `ative_position_index, self.window_size  # type: ignore[arg-type]         )      def forward(self, x: Tensor):         """         Args:             x (Tensor): Tensor with ` |
| `nerf_mae/model/mae/swin_mae3d.py` | `forward_encoder_decoder` | 2 | `     masks.append(mask_rgbsigma)         return padded_rgbsigma, masks      def forward_encoder_ecoder(self, x):         # print("x", x.shape)          # print("enc0", self.encoder1(` |
| `nerf_mae/model/mae/swin_mae3d.py` | `masking_prob` | 10 | ` 256,         input_dim: int = 4,         decoder_embed_dim: int = 768,         masking_prob=0.50,         resolution=160,         masking_strategy=None,     ):         sup` |
| `nerf_mae/model/mae/swin_mae3d.py` | `mask_remove` | 55 | `     x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()         x, mask_patches = self.window_masking_3d(             x,             p_remove=self.masking_prob` |
| `nerf_mae/model/mae/swin_mae3d.py` | `occupied_rgb` | 3 | `= pred[..., :3]         pred_alpha = pred[..., 3].unsqueeze(-1)          mask = target_alpha > 0.01          mask_remove = mask_remove.unsqueeze(-1).int()          # print("pred_r` |
| `nerf_mae/model/mae/swin_mae3d.py` | `swin_stages` | 39 | `ae.model.mae.unetr_block import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock   # Swin Transformer FPN   def shifted_window_attention(  # changed to 3D     input: Ten` |
| `nerf_mae/model/mae/swin_mae3d.py` | `pos_embed` | 14 | `   self.num_patches = int(round(self.resolution // patch_size[0]))         self.pos_embed = nn.Parameter(             torch.zeros(                 1, self.num_patches, s` |
| `nerf_mae/model/mae/shortcut_probe.py` | `forward_loss` | 1 | `e ValueError(f"Unsupported probe_alpha_target: {self.probe_alpha_target}")      def forward_loss(self, x, pred, mask_batch, mask_patches, is_eval=False):         """Compute the` |
| `nerf_mae/model/mae/shortcut_probe.py` | `forward` | 1 | `           target,             )         return loss, loss_rgb, loss_alpha      def forward(self, x, is_eval=False):         padded_x, mask = self.transform(x)         x_ta` |
| `nerf_mae/model/mae/shortcut_probe.py` | `forward_encoder_decoder` | 3 | ` self._apply_probe_input_corruption(x_target)         pred, mask_patches = self.forward_encoder_ecoder(x_model)         if is_eval:             loss, loss_rgb, loss_alpha, pred_patch` |
| `nerf_mae/model/mae/shortcut_probe.py` | `masking_prob` | 1 | `mask_patches = self.window_masking_3d(             x,             p_remove=self.masking_prob,             mask_token=self.mask_token,         )          features = []      ` |
| `nerf_mae/model/mae/shortcut_probe.py` | `mask_remove` | 18 | `ontiguous()      def _build_rgb_mask(         self, target_alpha: torch.Tensor, removed_mask: torch.Tensor     ) -> Optional[torch.Tensor]:         occupied_mask = (target_` |
| `nerf_mae/model/mae/shortcut_probe.py` | `occupied_rgb` | 9 | `p", "zero", "shuffle")     TARGET_ALPHA_MODES = ("keep", "zero", "shuffle")     RGB_LOSS_MODES = (         "occupied",         "target_alpha",         "removed_target_alpha",` |
| `nerf_mae/model/mae/shortcut_probe.py` | `swin_stages` | 8 | `s.  The implementation is intentionally conservative: it subclasses the public `SwinTransformer_MAE3D_New` model instead of modifying the original model file. """  ` |
| `nerf_mae/model/mae/shortcut_probe.py` | `pos_embed` | 1 | `ard_encoder_ecoder(x)          x = self.patch_partition(x)         x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()         x, mask_patches = self.window_` |
| `nerf_mae/run_swin_mae3d.py` | `forward_loss` | 0 | `` |
| `nerf_mae/run_swin_mae3d.py` | `forward` | 0 | `` |
| `nerf_mae/run_swin_mae3d.py` | `forward_encoder_decoder` | 0 | `` |
| `nerf_mae/run_swin_mae3d.py` | `masking_prob` | 3 | `help="Input dimension for backbone."     )      parser.add_argument(         "--masking_prob", type=float, default=0.5, help="Input dimension for backbone."     )      pars` |
| `nerf_mae/run_swin_mae3d.py` | `mask_remove` | 1 | `lpha,                     pred,                     mask,                     # mask_patches,                     target,                 ) = self.model(rgbsigma, is_eval=T` |
| `nerf_mae/run_swin_mae3d.py` | `occupied_rgb` | 0 | `` |
| `nerf_mae/run_swin_mae3d.py` | `swin_stages` | 21 | `DistributedSampler  # from model.fcos.fcos import FCOSOverNeRF # from model.mae.swin_mae3d import SwinTransformer_MAE3D  from model.mae.shortcut_probe import SwinTr` |
| `nerf_mae/run_swin_mae3d.py` | `pos_embed` | 5 | `ate-jitter diagnostics.",     )     parser.add_argument(         "--disable_abs_pos_embed",         action="store_true",         help="Zero and freeze the absolute sinus` |

## Checklist

- [ ] Confirm whether `forward_encoder_ecoder` receives a full dense grid after masking.
- [ ] Confirm whether masked positions are zeros, learned mask tokens, or otherwise marked.
- [ ] Confirm whether the removed mask is used only in the loss or also in the encoder.
- [ ] Confirm whether patch-grid shape is preserved through all Swin stages.
- [ ] Confirm whether disabling base internal masking sets base returned mask mean to 0 during MixNeRF runs.
