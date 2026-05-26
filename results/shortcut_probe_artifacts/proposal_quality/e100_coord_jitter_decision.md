# Proposal Quality Summary

| label | AP@50 | AP@75 | AP75/AP50 | R50@300 | mean IoU | frac IoU>=0.5 | center err >=0.5 | size err >=0.5 | first TP rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_coord_jitter | 0.5564 | 0.1015 | 0.1824 | 0.6765 | 0.0632 | 0.0180 | 0.0474 | 0.1816 | 1.2941 |
| cosine_coord_jitter | 0.6219 | 0.1031 | 0.1657 | 0.7279 | 0.0635 | 0.0196 | 0.0498 | 0.1818 | 1.0588 |
| dmae_hier_concat | 0.5778 | 0.1055 | 0.1826 | 0.6912 | 0.0649 | 0.0184 | 0.0501 | 0.2607 | 1.3529 |
| dmae_hier_concat_coord_jitter | 0.5212 | 0.0858 | 0.1646 | 0.6838 | 0.0702 | 0.0182 | 0.0500 | 0.2589 | 1.2941 |

## baseline_coord_jitter

- Proposal IoU: mean=0.0632, median=0.0023, p90=0.1989, frac>=0.25/0.5/0.75=0.0657/0.0180/0.0057
- Ranking: tp50 score=0.5904, fp50 score=0.0521, first TP rank=1.2941, top50/top100/top300 TP=0.1071/0.0535/0.0180
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1655/0.0474, size rel err IoU>=0.25/0.5=0.2630/0.1816
- Level share: L0=0.5537, L1=0.3220, L2=0.0898, L3=0.0345
- TP50 level share: L0=0.2717, L1=0.4565, L2=0.2717, L3=0.0000

## cosine_coord_jitter

- Proposal IoU: mean=0.0635, median=0.0047, p90=0.2011, frac>=0.25/0.5/0.75=0.0641/0.0196/0.0049
- Ranking: tp50 score=0.6006, fp50 score=0.0433, first TP rank=1.0588, top50/top100/top300 TP=0.1153/0.0588/0.0196
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1598/0.0498, size rel err IoU>=0.25/0.5=0.2469/0.1818
- Level share: L0=0.5663, L1=0.3025, L2=0.1014, L3=0.0298
- TP50 level share: L0=0.2200, L1=0.4700, L2=0.3100, L3=0.0000

## dmae_hier_concat

- Proposal IoU: mean=0.0649, median=0.0049, p90=0.2051, frac>=0.25/0.5/0.75=0.0665/0.0184/0.0057
- Ranking: tp50 score=0.6053, fp50 score=0.0455, first TP rank=1.3529, top50/top100/top300 TP=0.1094/0.0547/0.0184
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1600/0.0501, size rel err IoU>=0.25/0.5=0.3133/0.2607
- Level share: L0=0.5139, L1=0.3408, L2=0.1137, L3=0.0316
- TP50 level share: L0=0.2340, L1=0.5213, L2=0.2447, L3=0.0000

## dmae_hier_concat_coord_jitter

- Proposal IoU: mean=0.0702, median=0.0095, p90=0.2214, frac>=0.25/0.5/0.75=0.0767/0.0182/0.0047
- Ranking: tp50 score=0.5743, fp50 score=0.0584, first TP rank=1.2941, top50/top100/top300 TP=0.1094/0.0547/0.0182
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1657/0.0500, size rel err IoU>=0.25/0.5=0.3284/0.2589
- Level share: L0=0.5314, L1=0.2984, L2=0.1341, L3=0.0361
- TP50 level share: L0=0.2903, L1=0.4516, L2=0.2581, L3=0.0000

