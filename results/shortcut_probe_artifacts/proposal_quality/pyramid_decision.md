# Proposal Quality Summary

| label | AP@50 | AP@75 | AP75/AP50 | R50@300 | mean IoU | frac IoU>=0.5 | TP50 fail75 | center err >=0.5 | size err >=0.5 | first TP rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_coord_jitter | 0.5564 | 0.1015 | 0.1824 | 0.6765 | 0.0632 | 0.0180 | 0.6848 | 0.0474 | 0.1816 | 1.2941 |
| cosine_coord_jitter | 0.6219 | 0.1031 | 0.1657 | 0.7279 | 0.0635 | 0.0196 | 0.7500 | 0.0498 | 0.1818 | 1.0588 |
| surface_tau0p7 | 0.5973 | 0.0919 | 0.1539 | 0.7279 | 0.0609 | 0.0194 | 0.6869 | 0.0519 | 0.1775 | 1.2941 |
| shuffle_coord_jitter | 0.4138 | 0.0574 | 0.1388 | 0.6103 | 0.0605 | 0.0163 | 0.7590 | 0.0576 | 0.1845 | 4.2941 |
| pyramid_alpha | 0.5694 | 0.0978 | 0.1718 | 0.7206 | 0.0660 | 0.0192 | 0.7347 | 0.0488 | 0.3187 | 1.2941 |
| pyramid_rgb | 0.5447 | 0.1163 | 0.2136 | 0.6985 | 0.0655 | 0.0186 | 0.6947 | 0.0499 | 0.1782 | 3.0000 |
| pyramid_both | 0.5677 | 0.0521 | 0.0918 | 0.6985 | 0.0641 | 0.0186 | 0.7895 | 0.0509 | 0.1663 | 1.0588 |

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

## surface_tau0p7

- Proposal IoU: mean=0.0609, median=0.0000, p90=0.1992, frac>=0.25/0.5/0.75=0.0657/0.0194/0.0061
- IoU histogram: [0,0.05)=0.7059, [0.05,0.10)=0.0904, [0.10,0.25)=0.1380, [0.25,0.50)=0.0463, [0.50,0.75)=0.0133, [0.75,1.00]=0.0061
- Ranking: tp50 score=0.5738, fp50 score=0.0612, first TP rank=1.2941, top50/top100/top300 TP=0.1129/0.0565/0.0194
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1598/0.0519, size rel err IoU>=0.25/0.5=0.2756/0.1775
- Level share: L0=0.6200, L1=0.2588, L2=0.0900, L3=0.0312
- TP50 level share: L0=0.2323, L1=0.4949, L2=0.2727, L3=0.0000
- Score-IoU calibration: Q0: score=0.1864, IoU=0.1530, frac50=0.0941, frac75=0.0304; Q1: score=0.0606, IoU=0.0487, frac50=0.0000, frac75=0.0000; Q2: score=0.0443, IoU=0.0396, frac50=0.0010, frac75=0.0000; Q3: score=0.0360, IoU=0.0310, frac50=0.0010, frac75=0.0000; Q4: score=0.0285, IoU=0.0320, frac50=0.0010, frac75=0.0000
- Object size AP@50: small=0.1833, medium=0.2232, large=0.2656; AP@75: small=0.0496, medium=0.0173, large=0.0527

## shuffle_coord_jitter

- Proposal IoU: mean=0.0605, median=0.0001, p90=0.1995, frac>=0.25/0.5/0.75=0.0657/0.0163/0.0039
- IoU histogram: [0,0.05)=0.6988, [0.05,0.10)=0.0908, [0.10,0.25)=0.1447, [0.25,0.50)=0.0494, [0.50,0.75)=0.0124, [0.75,1.00]=0.0039
- Ranking: tp50 score=0.5321, fp50 score=0.0580, first TP rank=4.2941, top50/top100/top300 TP=0.0941/0.0482/0.0163
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1629/0.0576, size rel err IoU>=0.25/0.5=0.2763/0.1845
- Level share: L0=0.4976, L1=0.3120, L2=0.1561, L3=0.0343
- TP50 level share: L0=0.1928, L1=0.5301, L2=0.2771, L3=0.0000
- Score-IoU calibration: Q0: score=0.1799, IoU=0.1532, frac50=0.0804, frac75=0.0196; Q1: score=0.0555, IoU=0.0511, frac50=0.0000, frac75=0.0000; Q2: score=0.0392, IoU=0.0413, frac50=0.0000, frac75=0.0000; Q3: score=0.0305, IoU=0.0299, frac50=0.0000, frac75=0.0000; Q4: score=0.0233, IoU=0.0269, frac50=0.0010, frac75=0.0000
- Object size AP@50: small=0.1473, medium=0.1856, large=0.1762; AP@75: small=0.0339, medium=0.0099, large=0.0343

## pyramid_alpha

- Proposal IoU: mean=0.0660, median=0.0041, p90=0.2085, frac>=0.25/0.5/0.75=0.0696/0.0192/0.0051
- IoU histogram: [0,0.05)=0.6731, [0.05,0.10)=0.1020, [0.10,0.25)=0.1553, [0.25,0.50)=0.0504, [0.50,0.75)=0.0141, [0.75,1.00]=0.0051
- Ranking: tp50 score=0.5705, fp50 score=0.0517, first TP rank=1.2941, top50/top100/top300 TP=0.1118/0.0565/0.0192
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1664/0.0488, size rel err IoU>=0.25/0.5=0.3433/0.3187
- Level share: L0=0.5116, L1=0.3376, L2=0.1161, L3=0.0347
- TP50 level share: L0=0.2755, L1=0.4490, L2=0.2755, L3=0.0000
- Score-IoU calibration: Q0: score=0.1773, IoU=0.1644, frac50=0.0931, frac75=0.0255; Q1: score=0.0488, IoU=0.0555, frac50=0.0010, frac75=0.0000; Q2: score=0.0342, IoU=0.0442, frac50=0.0000, frac75=0.0000; Q3: score=0.0269, IoU=0.0375, frac50=0.0020, frac75=0.0000; Q4: score=0.0212, IoU=0.0285, frac50=0.0000, frac75=0.0000
- Object size AP@50: small=0.2152, medium=0.2372, large=0.1893; AP@75: small=0.0642, medium=0.0367, large=0.0326

## pyramid_rgb

- Proposal IoU: mean=0.0655, median=0.0076, p90=0.2045, frac>=0.25/0.5/0.75=0.0688/0.0186/0.0057
- IoU histogram: [0,0.05)=0.6702, [0.05,0.10)=0.1135, [0.10,0.25)=0.1475, [0.25,0.50)=0.0502, [0.50,0.75)=0.0129, [0.75,1.00]=0.0057
- Ranking: tp50 score=0.5961, fp50 score=0.0453, first TP rank=3.0000, top50/top100/top300 TP=0.1082/0.0553/0.0186
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1660/0.0499, size rel err IoU>=0.25/0.5=0.2602/0.1782
- Level share: L0=0.5835, L1=0.2853, L2=0.0937, L3=0.0375
- TP50 level share: L0=0.2526, L1=0.4947, L2=0.2421, L3=0.0105
- Score-IoU calibration: Q0: score=0.1728, IoU=0.1574, frac50=0.0902, frac75=0.0284; Q1: score=0.0398, IoU=0.0552, frac50=0.0020, frac75=0.0000; Q2: score=0.0274, IoU=0.0421, frac50=0.0010, frac75=0.0000; Q3: score=0.0212, IoU=0.0397, frac50=0.0000, frac75=0.0000; Q4: score=0.0166, IoU=0.0329, frac50=0.0000, frac75=0.0000
- Object size AP@50: small=0.1744, medium=0.2470, large=0.2070; AP@75: small=0.0490, medium=0.0483, large=0.0601

## pyramid_both

- Proposal IoU: mean=0.0641, median=0.0030, p90=0.2090, frac>=0.25/0.5/0.75=0.0688/0.0186/0.0039
- IoU histogram: [0,0.05)=0.6839, [0.05,0.10)=0.0992, [0.10,0.25)=0.1480, [0.25,0.50)=0.0502, [0.50,0.75)=0.0147, [0.75,1.00]=0.0039
- Ranking: tp50 score=0.6142, fp50 score=0.0542, first TP rank=1.0588, top50/top100/top300 TP=0.1082/0.0553/0.0186
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1606/0.0509, size rel err IoU>=0.25/0.5=0.2654/0.1663
- Level share: L0=0.5418, L1=0.3184, L2=0.1004, L3=0.0394
- TP50 level share: L0=0.2737, L1=0.4632, L2=0.2632, L3=0.0000
- Score-IoU calibration: Q0: score=0.1837, IoU=0.1593, frac50=0.0912, frac75=0.0196; Q1: score=0.0508, IoU=0.0558, frac50=0.0010, frac75=0.0000; Q2: score=0.0367, IoU=0.0405, frac50=0.0000, frac75=0.0000; Q3: score=0.0291, IoU=0.0342, frac50=0.0000, frac75=0.0000; Q4: score=0.0228, IoU=0.0307, frac50=0.0010, frac75=0.0000
- Object size AP@50: small=0.1709, medium=0.2870, large=0.2122; AP@75: small=0.0168, medium=0.0222, large=0.0296

