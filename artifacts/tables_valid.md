
#### 30 s head - P(S >= P0) on valid

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.3360 | 0.1033 | 0.0191 |  |
| + clock-aware vol model | 0.3280 | 0.1030 | 0.0216 | -0.0080 [-0.0089, -0.0070] |
| + no-move atom & fat-tailed law | 0.3215 | 0.1015 | 0.0084 | -0.0146 [-0.0163, -0.0130] |
| + drift (structural model) | 0.3169 | 0.0999 | 0.0060 | -0.0192 [-0.0215, -0.0172] |
| + calibration from prior | 0.3151 | 0.0993 | 0.0041 | -0.0209 [-0.0240, -0.0184] |
| direct GBM (rival) | 0.3159 | 0.0994 | 0.0054 | -0.0201 [-0.0233, -0.0174] |
| **final: blend** | 0.3148 | 0.0992 | 0.0047 | -0.0212 [-0.0243, -0.0185] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 1.5%, e∈(0, 30]: 4.2%, e∈(30, 60]: 5.4%, e∈(60, 120]: 6.3%, e∈(120, 180]: 7.1%, e∈(180, 240]: 7.1%, e∈(240, 270]: 10.8%

quantile coverage (nominal 5/10/25/50/75/90/95%): 4.5/9.1/24.7/51.6/76.0/91.0/95.6; pinball skill vs naive Gaussian 2.63%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 2.1% | 0.0614 | 0.0491 | 20.0% | 0.0009 |
| -10 bps | 5.7% | 0.1531 | 0.1258 | 17.8% | 0.0015 |
| -5 bps | 10.6% | 0.2651 | 0.2189 | 17.4% | 0.0083 |
| +5 bps | 10.7% | 0.2580 | 0.2157 | 16.4% | 0.0052 |
| +10 bps | 5.7% | 0.1504 | 0.1250 | 16.9% | 0.0010 |
| +20 bps | 2.1% | 0.0615 | 0.0491 | 20.1% | 0.0010 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 13.7% | 13.6% | 0.6400 | 0.5210 | 18.6% | 0.929 |
| ±10 bps | 6.4% | 6.3% | 0.3310 | 0.2722 | 17.8% | 0.979 |
| ±20 bps | 2.1% | 2.1% | 0.1260 | 0.1007 | 20.1% | 0.995 |

#### 60 s head - P(S >= P0) on valid

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.4344 | 0.1389 | 0.0196 |  |
| + clock-aware vol model | 0.4238 | 0.1382 | 0.0210 | -0.0106 [-0.0117, -0.0094] |
| + no-move atom & fat-tailed law | 0.4209 | 0.1372 | 0.0081 | -0.0135 [-0.0149, -0.0122] |
| + drift (structural model) | 0.4176 | 0.1360 | 0.0066 | -0.0169 [-0.0187, -0.0151] |
| + calibration from prior | 0.4170 | 0.1358 | 0.0055 | -0.0174 [-0.0195, -0.0155] |
| direct GBM (rival) | 0.4181 | 0.1360 | 0.0072 | -0.0163 [-0.0185, -0.0142] |
| **final: blend** | 0.4168 | 0.1357 | 0.0063 | -0.0176 [-0.0198, -0.0156] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 0.5%, e∈(0, 30]: 2.0%, e∈(30, 60]: 2.9%, e∈(60, 120]: 4.2%, e∈(120, 180]: 5.0%, e∈(180, 240]: 6.0%

quantile coverage (nominal 5/10/25/50/75/90/95%): 4.6/9.1/24.1/50.6/76.2/91.0/95.5; pinball skill vs naive Gaussian 2.43%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 3.6% | 0.1075 | 0.0860 | 20.0% | 0.0009 |
| -10 bps | 9.8% | 0.2414 | 0.2034 | 15.7% | 0.0030 |
| -5 bps | 17.6% | 0.3760 | 0.3247 | 13.6% | 0.0115 |
| +5 bps | 17.6% | 0.3641 | 0.3184 | 12.5% | 0.0073 |
| +10 bps | 9.8% | 0.2350 | 0.2006 | 14.6% | 0.0012 |
| +20 bps | 3.6% | 0.1057 | 0.0853 | 19.4% | 0.0011 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 21.1% | 21.0% | 0.8122 | 0.6853 | 15.6% | 0.877 |
| ±10 bps | 10.6% | 10.6% | 0.4975 | 0.4177 | 16.0% | 0.945 |
| ±20 bps | 3.7% | 3.7% | 0.2161 | 0.1729 | 20.0% | 0.979 |