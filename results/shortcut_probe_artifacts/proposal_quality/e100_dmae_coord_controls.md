# Proposal Quality Summary

| label | AP@50 | AP@75 | AP75/AP50 | R50@300 | mean IoU | frac IoU>=0.5 | center err >=0.5 | size err >=0.5 | first TP rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_no_pos | 0.5371 | 0.0899 | 0.1674 | 0.6912 | 0.0670 | 0.0186 | 0.0512 | 0.3113 | 3.0000 |
| alpha_coord_jitter | 0.4954 | 0.0622 | 0.1255 | 0.6324 | 0.0634 | 0.0169 | 0.0578 | 0.1716 | 1.6875 |
| cosine_coord_jitter | 0.6219 | 0.1031 | 0.1657 | 0.7279 | 0.0635 | 0.0196 | 0.0498 | 0.1818 | 1.0588 |
| dmae_gate | 0.5045 | 0.0832 | 0.1650 | 0.6544 | 0.0613 | 0.0175 | 0.0491 | 0.2988 | 2.5882 |
| dmae_concat | 0.5778 | 0.1055 | 0.1826 | 0.6912 | 0.0649 | 0.0184 | 0.0501 | 0.2607 | 1.3529 |
| dmae_film | 0.5443 | 0.0709 | 0.1302 | 0.7059 | 0.0686 | 0.0188 | 0.0525 | 0.2563 | 1.4706 |

## baseline_no_pos

- Proposal IoU: mean=0.0670, median=0.0066, p90=0.2127, frac>=0.25/0.5/0.75=0.0749/0.0186/0.0049
- Ranking: tp50 score=0.5565, fp50 score=0.0577, first TP rank=3.0000, top50/top100/top300 TP=0.1071/0.0541/0.0186
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1711/0.0512, size rel err IoU>=0.25/0.5=0.3245/0.3113
- Level share: L0=0.4853, L1=0.3198, L2=0.1561, L3=0.0388
- TP50 level share: L0=0.2421, L1=0.4947, L2=0.2526, L3=0.0105

## alpha_coord_jitter

- Proposal IoU: mean=0.0634, median=0.0036, p90=0.1996, frac>=0.25/0.5/0.75=0.0667/0.0169/0.0035
- Ranking: tp50 score=0.5936, fp50 score=0.0431, first TP rank=1.6875, top50/top100/top300 TP=0.0988/0.0500/0.0169
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1614/0.0578, size rel err IoU>=0.25/0.5=0.2435/0.1716
- Level share: L0=0.5024, L1=0.3600, L2=0.1071, L3=0.0306
- TP50 level share: L0=0.2674, L1=0.4535, L2=0.2791, L3=0.0000

## cosine_coord_jitter

- Proposal IoU: mean=0.0635, median=0.0047, p90=0.2011, frac>=0.25/0.5/0.75=0.0641/0.0196/0.0049
- Ranking: tp50 score=0.6006, fp50 score=0.0433, first TP rank=1.0588, top50/top100/top300 TP=0.1153/0.0588/0.0196
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1598/0.0498, size rel err IoU>=0.25/0.5=0.2469/0.1818
- Level share: L0=0.5663, L1=0.3025, L2=0.1014, L3=0.0298
- TP50 level share: L0=0.2200, L1=0.4700, L2=0.3100, L3=0.0000

## dmae_gate

- Proposal IoU: mean=0.0613, median=0.0007, p90=0.2025, frac>=0.25/0.5/0.75=0.0676/0.0175/0.0053
- Ranking: tp50 score=0.5843, fp50 score=0.0654, first TP rank=2.5882, top50/top100/top300 TP=0.1024/0.0518/0.0175
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1647/0.0491, size rel err IoU>=0.25/0.5=0.3328/0.2988
- Level share: L0=0.5825, L1=0.2808, L2=0.1063, L3=0.0304
- TP50 level share: L0=0.2697, L1=0.4382, L2=0.2921, L3=0.0000

## dmae_concat

- Proposal IoU: mean=0.0649, median=0.0049, p90=0.2051, frac>=0.25/0.5/0.75=0.0665/0.0184/0.0057
- Ranking: tp50 score=0.6053, fp50 score=0.0455, first TP rank=1.3529, top50/top100/top300 TP=0.1094/0.0547/0.0184
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1600/0.0501, size rel err IoU>=0.25/0.5=0.3133/0.2607
- Level share: L0=0.5139, L1=0.3408, L2=0.1137, L3=0.0316
- TP50 level share: L0=0.2340, L1=0.5213, L2=0.2447, L3=0.0000

## dmae_film

- Proposal IoU: mean=0.0686, median=0.0103, p90=0.2115, frac>=0.25/0.5/0.75=0.0688/0.0188/0.0045
- Ranking: tp50 score=0.5731, fp50 score=0.0511, first TP rank=1.4706, top50/top100/top300 TP=0.1094/0.0553/0.0188
- Localization among matched proposals: center norm err IoU>=0.25/0.5=0.1679/0.0525, size rel err IoU>=0.25/0.5=0.3196/0.2563
- Level share: L0=0.4496, L1=0.3657, L2=0.1449, L3=0.0398
- TP50 level share: L0=0.2188, L1=0.5312, L2=0.2500, L3=0.0000

