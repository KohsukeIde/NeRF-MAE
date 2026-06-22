# Anchor-head low-label jobs (20260623)

- Purpose: second detector-head breadth check for Front3D 10% labels.
- Head: anchor-based NeRF-RPN via `run_rpn.py`.
- Arms: scratch, joint e300, structure-first/cosine e300.
- Seeds: 1, 2, 3.
- Percent train: 0.1.
- Anchor epochs: 200.
- Deterministic: 0.
- CSV: /groups/gag51404/ide/vgi/NeRF-MAE/results/shortcut_probe_artifacts/anchor_head_lowlabel_jobs_20260623.csv

```csv
job_id,job_name,arm,seed,percent_train,save_name,checkpoint
1935738.pbs1,anc_s_p10_s1,scratch,1,0.1,front3d_anchor_scratch_p10_seed1_rpn200,
1935739.pbs1,anc_j_p10_s1,joint_e300,1,0.1,front3d_anchor_joint_e300_p10_seed1_rpn200,/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_all_p1.0_e300_seed1_abci3clean/epoch_300.pt
1935740.pbs1,anc_c_p10_s1,cosine_e300,1,0.1,front3d_anchor_cosine_e300_p10_seed1_rpn200,/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1_abci3clean/epoch_300.pt
1935741.pbs1,anc_s_p10_s2,scratch,2,0.1,front3d_anchor_scratch_p10_seed2_rpn200,
1935742.pbs1,anc_j_p10_s2,joint_e300,2,0.1,front3d_anchor_joint_e300_p10_seed2_rpn200,/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_all_p1.0_e300_seed1_abci3clean/epoch_300.pt
1935743.pbs1,anc_c_p10_s2,cosine_e300,2,0.1,front3d_anchor_cosine_e300_p10_seed2_rpn200,/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1_abci3clean/epoch_300.pt
1935744.pbs1,anc_s_p10_s3,scratch,3,0.1,front3d_anchor_scratch_p10_seed3_rpn200,
1935745.pbs1,anc_j_p10_s3,joint_e300,3,0.1,front3d_anchor_joint_e300_p10_seed3_rpn200,/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_all_p1.0_e300_seed1_abci3clean/epoch_300.pt
1935746.pbs1,anc_c_p10_s3,cosine_e300,3,0.1,front3d_anchor_cosine_e300_p10_seed3_rpn200,/groups/gag51404/ide/vgi/NeRF-MAE/output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1_abci3clean/epoch_300.pt
```
