# Proposal Quality Summary

| label | AP@50 | AP@75 | AP75/AP50 | R50@300 | mean IoU | frac IoU>=0.5 | center err >=0.5 | size err >=0.5 | first TP rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_ft1 | 0.4695 | 0.0869 | 0.1851 | 0.6618 | 0.0653 | 0.0176 | 0.0525 | 0.3178 | 1.5882 |
| baseline_ft2 | 0.4862 | 0.0916 | 0.1883 | 0.6250 | 0.0605 | 0.0167 | 0.0484 | 0.1658 | 1.4118 |
| baseline_ft3 | 0.5258 | 0.0947 | 0.1802 | 0.6985 | 0.0610 | 0.0186 | 0.0524 | 0.3017 | 1.8235 |
| cosine_ft1 | 0.5539 | 0.1135 | 0.2049 | 0.7059 | 0.0641 | 0.0192 | 0.0520 | 0.2867 | 1.5882 |
| cosine_ft2 | 0.5704 | 0.1176 | 0.2061 | 0.6838 | 0.0611 | 0.0184 | 0.0482 | 0.1678 | 1.2941 |
| cosine_ft3 | 0.5928 | 0.0891 | 0.1502 | 0.7132 | 0.0602 | 0.0190 | 0.0510 | 0.3216 | 1.1765 |
| shuffle_ft1 | 0.4162 | 0.0326 | 0.0783 | 0.5956 | 0.0640 | 0.0159 | 0.0571 | 0.1909 | 1.8824 |
| shuffle_ft2 | 0.4187 | 0.0750 | 0.1792 | 0.5956 | 0.0575 | 0.0159 | 0.0497 | 0.1775 | 3.0588 |
| shuffle_ft3 | 0.4532 | 0.0282 | 0.0623 | 0.6397 | 0.0690 | 0.0173 | 0.0547 | 0.3265 | 1.5882 |

## baseline_ft1

- Proposal IoU: mean=0.0653, median=0.0044, p90=0.2058, frac>=0.25/0.5/0.75=0.0690/0.0176/0.0053
- Ranking: tp50 score=0.5716, fp50 score=0.0589, first TP rank=1.5882, top50/top100/top300 TP=0.1024/0.0524/0.0176
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1617/0.0525, size rel err IoU>=0.25/0.5=0.3171/0.3178
- Level share: L0=0.5090, L1=0.3227, L2=0.1257, L3=0.0425
- TP50 level share: L0=0.2667, L1=0.5000, L2=0.2222, L3=0.0111

## baseline_ft2

- Proposal IoU: mean=0.0605, median=0.0026, p90=0.1916, frac>=0.25/0.5/0.75=0.0649/0.0167/0.0053
- Ranking: tp50 score=0.6145, fp50 score=0.0417, first TP rank=1.4118, top50/top100/top300 TP=0.0976/0.0500/0.0167
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1648/0.0484, size rel err IoU>=0.25/0.5=0.2673/0.1658
- Level share: L0=0.5273, L1=0.3122, L2=0.1233, L3=0.0373
- TP50 level share: L0=0.2235, L1=0.4941, L2=0.2588, L3=0.0235

## baseline_ft3

- Proposal IoU: mean=0.0610, median=0.0011, p90=0.2002, frac>=0.25/0.5/0.75=0.0665/0.0186/0.0049
- Ranking: tp50 score=0.5757, fp50 score=0.0386, first TP rank=1.8235, top50/top100/top300 TP=0.1082/0.0553/0.0186
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1630/0.0524, size rel err IoU>=0.25/0.5=0.3284/0.3017
- Level share: L0=0.5863, L1=0.2692, L2=0.1082, L3=0.0363
- TP50 level share: L0=0.2632, L1=0.4632, L2=0.2737, L3=0.0000

## cosine_ft1

- Proposal IoU: mean=0.0641, median=0.0039, p90=0.2069, frac>=0.25/0.5/0.75=0.0694/0.0192/0.0059
- Ranking: tp50 score=0.5827, fp50 score=0.0576, first TP rank=1.5882, top50/top100/top300 TP=0.1106/0.0553/0.0192
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1622/0.0520, size rel err IoU>=0.25/0.5=0.3034/0.2867
- Level share: L0=0.5343, L1=0.3251, L2=0.1041, L3=0.0365
- TP50 level share: L0=0.2449, L1=0.5000, L2=0.2551, L3=0.0000

## cosine_ft2

- Proposal IoU: mean=0.0611, median=0.0009, p90=0.1966, frac>=0.25/0.5/0.75=0.0625/0.0184/0.0061
- Ranking: tp50 score=0.6339, fp50 score=0.0410, first TP rank=1.2941, top50/top100/top300 TP=0.1106/0.0553/0.0184
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1586/0.0482, size rel err IoU>=0.25/0.5=0.2667/0.1678
- Level share: L0=0.5396, L1=0.2957, L2=0.1243, L3=0.0404
- TP50 level share: L0=0.2234, L1=0.5106, L2=0.2660, L3=0.0000

## cosine_ft3

- Proposal IoU: mean=0.0602, median=0.0003, p90=0.1974, frac>=0.25/0.5/0.75=0.0639/0.0190/0.0055
- Ranking: tp50 score=0.6041, fp50 score=0.0374, first TP rank=1.1765, top50/top100/top300 TP=0.1129/0.0571/0.0190
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1568/0.0510, size rel err IoU>=0.25/0.5=0.3381/0.3216
- Level share: L0=0.5443, L1=0.3031, L2=0.1141, L3=0.0384
- TP50 level share: L0=0.2165, L1=0.4742, L2=0.3093, L3=0.0000

## shuffle_ft1

- Proposal IoU: mean=0.0640, median=0.0032, p90=0.2113, frac>=0.25/0.5/0.75=0.0727/0.0159/0.0027
- Ranking: tp50 score=0.5287, fp50 score=0.0719, first TP rank=1.8824, top50/top100/top300 TP=0.0953/0.0476/0.0159
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1703/0.0571, size rel err IoU>=0.25/0.5=0.2746/0.1909
- Level share: L0=0.4590, L1=0.3547, L2=0.1410, L3=0.0453
- TP50 level share: L0=0.2840, L1=0.4691, L2=0.2346, L3=0.0123

## shuffle_ft2

- Proposal IoU: mean=0.0575, median=0.0001, p90=0.1959, frac>=0.25/0.5/0.75=0.0655/0.0159/0.0043
- Ranking: tp50 score=0.5438, fp50 score=0.0441, first TP rank=3.0588, top50/top100/top300 TP=0.0941/0.0471/0.0159
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1669/0.0497, size rel err IoU>=0.25/0.5=0.2594/0.1775
- Level share: L0=0.5059, L1=0.3371, L2=0.1157, L3=0.0414
- TP50 level share: L0=0.2469, L1=0.4691, L2=0.2840, L3=0.0000

## shuffle_ft3

- Proposal IoU: mean=0.0690, median=0.0103, p90=0.2111, frac>=0.25/0.5/0.75=0.0737/0.0173/0.0025
- Ranking: tp50 score=0.5045, fp50 score=0.0864, first TP rank=1.5882, top50/top100/top300 TP=0.1000/0.0506/0.0173
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1702/0.0547, size rel err IoU>=0.25/0.5=0.3607/0.3265
- Level share: L0=0.5675, L1=0.2753, L2=0.1190, L3=0.0382
- TP50 level share: L0=0.3295, L1=0.4205, L2=0.2500, L3=0.0000

