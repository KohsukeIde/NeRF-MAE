# Alpha Boundary Target Audit v2

Rows: 140

| variant | scenes | occ ratio mean | shell/occ mean | components median | largest comp mean | top5 comp mean | small comp mean | raw IoU mean | raw recall mean | sdf inside p90 mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_thr001 | 20 | 0.314632 | 0.6869 | 440.5 | 0.9937 | 0.9976 | 0.0000 | 1.0000 | 1.0000 | 3.235 |
| smooth075_thr001 | 20 | 0.426111 | 0.3599 | 113.5 | 0.9954 | 0.9989 | 0.0000 | 0.7363 | 0.9983 | 6.527 |
| smooth100_thr001 | 20 | 0.453589 | 0.3031 | 39.5 | 0.9957 | 0.9993 | 0.0000 | 0.6914 | 0.9974 | 7.611 |
| smooth100_thr001_close1_min64 | 20 | 0.427339 | 0.2688 | 21.0 | 0.9964 | 0.9997 | 0.0002 | 0.6158 | 0.8955 | 6.260 |
| smooth100_thr001_min256 | 20 | 0.453167 | 0.3024 | 39.5 | 0.9957 | 0.9993 | 0.0010 | 0.6913 | 0.9967 | 7.615 |
| smooth100_thr001_min64 | 20 | 0.453390 | 0.3027 | 39.5 | 0.9957 | 0.9993 | 0.0004 | 0.6913 | 0.9970 | 7.615 |
| smooth100_thr002 | 20 | 0.416202 | 0.3406 | 35.0 | 0.9945 | 0.9993 | 0.0000 | 0.7417 | 0.9892 | 6.796 |

Decision guide:
- A usable Boundary-SDF target should reduce fragmentation without deleting most raw low-threshold support.
- Prefer variants with lower component counts, lower shell/occupied ratio, and reasonable raw-reference recall.
- If only aggressive filtering looks clean but raw recall collapses, the target is likely too biased for a main method.
- Signed-distance maps should show coherent bands around surfaces in the PNG comparisons.
