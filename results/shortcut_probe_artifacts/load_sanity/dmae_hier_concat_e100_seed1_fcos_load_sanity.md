# FCOS Checkpoint Load Sanity

- Git hash: `6dff2390a647c6f1762ba8288466951b5beb1b9a`
- Checkpoint: `/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_dmae_hier_concat_p1.0_e100_seed1_abci3dmae_e100_det0_1n8g/epoch_100.pt`
- FCOS instantiated: `True`
- Pass: `True`
- Missing keys: `2`
- Unexpected keys: `8`
- Encoder missing keys: `0`
- Encoder unexpected keys: `0`
- Encoder exact tensor ratio: `1.000000`
- Encoder exact numel ratio: `1.000000`

| prefix | exact tensors | total tensors | exact numel ratio |
|---|---:|---:|---:|
| pos_embed | 1 | 1 | 1.000000 |
| patch_partition | 4 | 4 | 1.000000 |
| stages. | 345 | 345 | 1.000000 |

## Unexpected Keys

- `decomp_structure_head.0.weight`
- `decomp_structure_head.0.bias`
- `decomp_structure_head.3.weight`
- `decomp_structure_head.3.bias`
- `decomp_structure_head.6.weight`
- `decomp_structure_head.6.bias`
- `decomp_rgb_head.weight`
- `decomp_rgb_head.bias`

## Missing Keys

- `out.conv.weight`
- `out.conv.bias`
