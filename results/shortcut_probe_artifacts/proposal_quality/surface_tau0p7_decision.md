# Proposal Quality Summary

| label | AP@50 | AP@75 | AP75/AP50 | R50@300 | mean IoU | frac IoU>=0.5 | TP50 fail75 | center err >=0.5 | size err >=0.5 | first TP rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_coord_jitter | 0.5564 | 0.1015 | 0.1824 | 0.6765 | 0.0632 | 0.0180 | 0.6848 | 0.0474 | 0.1816 | 1.2941 |
| cosine_coord_jitter | 0.6219 | 0.1031 | 0.1657 | 0.7279 | 0.0635 | 0.0196 | 0.7500 | 0.0498 | 0.1818 | 1.0588 |
| surface_tau0p7_k30 | 0.5973 | 0.0919 | 0.1539 | 0.7279 | 0.0609 | 0.0194 | 0.6869 | 0.0519 | 0.1775 | 1.2941 |

## baseline_coord_jitter

- Proposal IoU: mean=0.0632, median=0.0023, p90=0.1989, frac>=0.25/0.5/0.75=0.0657/0.0180/0.0057
- IoU histogram: [0,0.05)=0.6859, [0.05,0.10)=0.1014, [0.10,0.25)=0.1471, [0.25,0.50)=0.0476, [0.50,0.75)=0.0124, [0.75,1.00]=0.0057
- Ranking: tp50 score=0.5904, fp50 score=0.0521, first TP rank=1.2941, top50/top100/top300 TP=0.1071/0.0535/0.0180
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1655/0.0474, size rel err IoU>=0.25/0.5=0.2630/0.1816
- Level share: L0=0.5537, L1=0.3220, L2=0.0898, L3=0.0345
- TP50 level share: L0=0.2717, L1=0.4565, L2=0.2717, L3=0.0000
- Score-IoU calibration: Q0: score=0.1735, IoU=0.1534, frac50=0.0892, frac75=0.0284; Q1: score=0.0488, IoU=0.0544, frac50=0.0000, frac75=0.0000; Q2: score=0.0359, IoU=0.0466, frac50=0.0000, frac75=0.0000; Q3: score=0.0288, IoU=0.0319, frac50=0.0000, frac75=0.0000; Q4: score=0.0222, IoU=0.0299, frac50=0.0010, frac75=0.0000
- Object size AP@50: small=0.2066, medium=0.2033, large=0.2453; AP@75: small=0.0609, medium=0.0437, large=0.0323

## cosine_coord_jitter

- Proposal IoU: mean=0.0635, median=0.0047, p90=0.2011, frac>=0.25/0.5/0.75=0.0641/0.0196/0.0049
- IoU histogram: [0,0.05)=0.6816, [0.05,0.10)=0.1073, [0.10,0.25)=0.1471, [0.25,0.50)=0.0445, [0.50,0.75)=0.0147, [0.75,1.00]=0.0049
- Ranking: tp50 score=0.6006, fp50 score=0.0433, first TP rank=1.0588, top50/top100/top300 TP=0.1153/0.0588/0.0196
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1598/0.0498, size rel err IoU>=0.25/0.5=0.2469/0.1818
- Level share: L0=0.5663, L1=0.3025, L2=0.1014, L3=0.0298
- TP50 level share: L0=0.2200, L1=0.4700, L2=0.3100, L3=0.0000
- Score-IoU calibration: Q0: score=0.1654, IoU=0.1577, frac50=0.0971, frac75=0.0245; Q1: score=0.0399, IoU=0.0581, frac50=0.0010, frac75=0.0000; Q2: score=0.0276, IoU=0.0392, frac50=0.0000, frac75=0.0000; Q3: score=0.0216, IoU=0.0317, frac50=0.0000, frac75=0.0000; Q4: score=0.0166, IoU=0.0307, frac50=0.0000, frac75=0.0000
- Object size AP@50: small=0.1846, medium=0.2943, large=0.2639; AP@75: small=0.0455, medium=0.0795, large=0.0272

## surface_tau0p7_k30

- Proposal IoU: mean=0.0609, median=0.0000, p90=0.1992, frac>=0.25/0.5/0.75=0.0657/0.0194/0.0061
- IoU histogram: [0,0.05)=0.7059, [0.05,0.10)=0.0904, [0.10,0.25)=0.1380, [0.25,0.50)=0.0463, [0.50,0.75)=0.0133, [0.75,1.00]=0.0061
- Ranking: tp50 score=0.5738, fp50 score=0.0612, first TP rank=1.2941, top50/top100/top300 TP=0.1129/0.0565/0.0194
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1598/0.0519, size rel err IoU>=0.25/0.5=0.2756/0.1775
- Level share: L0=0.6200, L1=0.2588, L2=0.0900, L3=0.0312
- TP50 level share: L0=0.2323, L1=0.4949, L2=0.2727, L3=0.0000
- Score-IoU calibration: Q0: score=0.1864, IoU=0.1530, frac50=0.0941, frac75=0.0304; Q1: score=0.0606, IoU=0.0487, frac50=0.0000, frac75=0.0000; Q2: score=0.0443, IoU=0.0396, frac50=0.0010, frac75=0.0000; Q3: score=0.0360, IoU=0.0310, frac50=0.0010, frac75=0.0000; Q4: score=0.0285, IoU=0.0320, frac50=0.0010, frac75=0.0000
- Object size AP@50: small=0.1833, medium=0.2232, large=0.2656; AP@75: small=0.0496, medium=0.0173, large=0.0527

