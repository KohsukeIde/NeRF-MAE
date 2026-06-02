# Mask-predictability probe protocol

Goal: test whether the full-grid NeRF-MAE encoder exposes the random mask pattern in its features.

1. Choose checkpoints: baseline_e300, cosine_ramp_e300, surface_cosine_jitter_e300.
2. Run the model in eval mode with normal pretraining masking.
3. Hook stage0/stage1/stage2/stage3 features before decoder upsampling.
4. Downsample / reshape the patch mask to each stage resolution.
5. Save `.npz` files with `features` and `mask` per stage / surface-region.
6. Fit linear/logistic probes with this script.
7. Report AUC. High AUC implies mask-pattern leakage into encoder features.

Interpretation:
- High mask AUC in baseline/current recipe: MixNeRF-MAE motivation is strong.
- MixNeRF e30/e100 should reduce mask AUC if it works.
- If mask AUC is low already, MixNeRF is less motivated.
