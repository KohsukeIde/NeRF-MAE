# Alpha Boundary Target Audit v2

Rows: 420

| variant | scenes | occ ratio mean | shell/occ mean | components median | largest comp mean | top5 comp mean | small comp mean | raw IoU mean | raw recall mean | sdf inside p90 mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_thr001 | 60 | 0.308927 | 0.6611 | 562.5 | 0.9923 | 0.9970 | 0.0000 | 1.0000 | 1.0000 | 3.411 |
| smooth075_thr001 | 60 | 0.409878 | 0.3535 | 151.5 | 0.9939 | 0.9983 | 0.0000 | 0.7530 | 0.9980 | 6.781 |
| smooth100_thr001 | 60 | 0.434797 | 0.3000 | 56.5 | 0.9943 | 0.9988 | 0.0000 | 0.7092 | 0.9968 | 7.828 |
| smooth100_thr001_close1_min64 | 60 | 0.409860 | 0.2678 | 29.0 | 0.9957 | 0.9994 | 0.0003 | 0.6315 | 0.8968 | 6.330 |
| smooth100_thr001_min256 | 60 | 0.434312 | 0.2992 | 56.5 | 0.9943 | 0.9988 | 0.0012 | 0.7091 | 0.9960 | 7.829 |
| smooth100_thr001_min64 | 60 | 0.434578 | 0.2996 | 56.5 | 0.9943 | 0.9988 | 0.0005 | 0.7091 | 0.9964 | 7.829 |
| smooth100_thr002 | 60 | 0.397898 | 0.3361 | 48.0 | 0.9936 | 0.9988 | 0.0000 | 0.7607 | 0.9869 | 7.015 |

Decision guide:
- A usable Boundary-SDF target should reduce fragmentation without deleting most raw low-threshold support.
- Prefer variants with lower component counts, lower shell/occupied ratio, and reasonable raw-reference recall.
- If only aggressive filtering looks clean but raw recall collapses, the target is likely too biased for a main method.
- Signed-distance maps should show coherent bands around surfaces in the PNG comparisons.
