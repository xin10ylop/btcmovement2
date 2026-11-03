
#### 30 s head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.3421 | 0.1025 | 0.0283 |  |
| + clock-aware vol model | 0.3293 | 0.1022 | 0.0292 | -0.0128 [-0.0140, -0.0117] |
| + no-move atom & fat-tailed law | 0.3172 | 0.0993 | 0.0104 | -0.0249 [-0.0265, -0.0234] |
| + drift (structural model) | 0.3098 | 0.0968 | 0.0057 | -0.0323 [-0.0342, -0.0305] |
| + calibration from prior | 0.3064 | 0.0956 | 0.0021 | -0.0357 [-0.0385, -0.0331] |
| direct GBM (rival) | 0.3066 | 0.0956 | 0.0031 | -0.0355 [-0.0386, -0.0326] |
| **final: blend** | 0.3057 | 0.0954 | 0.0024 | -0.0364 [-0.0395, -0.0337] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 3.3%, e∈(0, 30]: 7.5%, e∈(30, 60]: 9.5%, e∈(60, 120]: 11.4%, e∈(120, 180]: 11.8%, e∈(180, 240]: 11.6%, e∈(240, 270]: 16.1%

quantile coverage (nominal 5/10/25/50/75/90/95%): 4.8/9.4/24.8/52.7/76.7/90.8/95.3; pinball skill vs naive Gaussian 4.25%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 1.1% | 0.0333 | 0.0256 | 23.1% | 0.0007 |
| -10 bps | 3.5% | 0.1058 | 0.0821 | 22.4% | 0.0012 |
| -5 bps | 7.7% | 0.2193 | 0.1704 | 22.3% | 0.0051 |
| +5 bps | 7.4% | 0.2120 | 0.1649 | 22.2% | 0.0042 |
| +10 bps | 3.4% | 0.1038 | 0.0804 | 22.5% | 0.0012 |
| +20 bps | 1.0% | 0.0340 | 0.0254 | 25.1% | 0.0006 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 9.0% | 9.2% | 0.5055 | 0.3886 | 23.1% | 0.958 |
| ±10 bps | 3.7% | 3.7% | 0.2217 | 0.1715 | 22.7% | 0.990 |
| ±20 bps | 1.0% | 1.1% | 0.0679 | 0.0516 | 24.1% | 0.998 |

#### 60 s head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.4422 | 0.1386 | 0.0301 |  |
| + clock-aware vol model | 0.4239 | 0.1378 | 0.0267 | -0.0183 [-0.0196, -0.0169] |
| + no-move atom & fat-tailed law | 0.4184 | 0.1360 | 0.0102 | -0.0238 [-0.0251, -0.0224] |
| + drift (structural model) | 0.4124 | 0.1337 | 0.0047 | -0.0298 [-0.0314, -0.0281] |
| + calibration from prior | 0.4111 | 0.1331 | 0.0024 | -0.0311 [-0.0330, -0.0291] |
| direct GBM (rival) | 0.4117 | 0.1332 | 0.0034 | -0.0305 [-0.0328, -0.0283] |
| **final: blend** | 0.4104 | 0.1329 | 0.0029 | -0.0317 [-0.0339, -0.0296] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 1.7%, e∈(0, 30]: 4.1%, e∈(30, 60]: 6.0%, e∈(60, 120]: 7.6%, e∈(120, 180]: 8.6%, e∈(180, 240]: 9.5%

quantile coverage (nominal 5/10/25/50/75/90/95%): 4.8/9.5/24.4/51.4/76.2/90.6/95.2; pinball skill vs naive Gaussian 4.01%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 1.9% | 0.0611 | 0.0461 | 24.6% | 0.0008 |
| -10 bps | 6.2% | 0.1766 | 0.1394 | 21.0% | 0.0016 |
| -5 bps | 13.3% | 0.3274 | 0.2659 | 18.8% | 0.0073 |
| +5 bps | 13.0% | 0.3201 | 0.2602 | 18.7% | 0.0054 |
| +10 bps | 6.1% | 0.1747 | 0.1371 | 21.5% | 0.0015 |
| +20 bps | 1.8% | 0.0616 | 0.0461 | 25.1% | 0.0008 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 15.0% | 15.3% | 0.7073 | 0.5644 | 20.2% | 0.913 |
| ±10 bps | 6.4% | 6.5% | 0.3627 | 0.2839 | 21.7% | 0.969 |
| ±20 bps | 1.8% | 1.9% | 0.1231 | 0.0925 | 24.9% | 0.989 |

#### 5-minute close head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.5289 | 0.1679 | 0.0402 |  |
| + clock-aware vol model | 0.5073 | 0.1650 | 0.0157 | -0.0216 [-0.0238, -0.0195] |
| + no-move atom & fat-tailed law | 0.4935 | 0.1646 | 0.0159 | -0.0354 [-0.0394, -0.0317] |
| + drift (structural model) | 0.4912 | 0.1637 | 0.0188 | -0.0377 [-0.0417, -0.0340] |
| + calibration from prior | 0.4912 | 0.1637 | 0.0188 | -0.0377 [-0.0417, -0.0340] |
| direct GBM (rival) | 0.4896 | 0.1639 | 0.0095 | -0.0393 [-0.0442, -0.0346] |
| **final: blend** | 0.4888 | 0.1636 | 0.0142 | -0.0401 [-0.0445, -0.0359] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 0.3%, e∈(0, 30]: 0.8%, e∈(30, 60]: 2.0%, e∈(60, 120]: 3.8%, e∈(120, 180]: 6.7%, e∈(180, 240]: 11.7%, e∈(240, 270]: 19.7%, e∈(270, 299]: 29.9%

quantile coverage (nominal 5/10/25/50/75/90/95%): 5.1/10.0/25.1/51.2/75.2/89.8/94.6; pinball skill vs naive Gaussian 3.62%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 4.1% | 0.1370 | 0.1029 | 24.9% | 0.0011 |
| -10 bps | 12.3% | 0.2991 | 0.2436 | 18.6% | 0.0023 |
| -5 bps | 22.9% | 0.4351 | 0.3711 | 14.7% | 0.0081 |
| +5 bps | 22.4% | 0.4327 | 0.3680 | 15.0% | 0.0063 |
| +10 bps | 12.2% | 0.3053 | 0.2446 | 19.9% | 0.0017 |
| +20 bps | 4.0% | 0.1387 | 0.1031 | 25.7% | 0.0018 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 24.3% | 24.7% | 0.8586 | 0.7017 | 18.3% | 0.831 |
| ±10 bps | 12.5% | 12.7% | 0.6031 | 0.4761 | 21.1% | 0.893 |
| ±20 bps | 4.0% | 4.1% | 0.2744 | 0.2028 | 26.1% | 0.929 |