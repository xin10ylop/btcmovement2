
#### 10 s head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.2169 | 0.0619 | 0.0314 |  |
| + clock-aware vol model | 0.2174 | 0.0607 | 0.0269 | +0.0005 [-0.0004, +0.0015] |
| + no-move atom & fat-tailed law | 0.1919 | 0.0567 | 0.0063 | -0.0250 [-0.0269, -0.0231] |
| + drift (structural model) | 0.1850 | 0.0548 | 0.0059 | -0.0319 [-0.0338, -0.0302] |
| + calibration from prior | 0.1776 | 0.0527 | 0.0013 | -0.0393 [-0.0427, -0.0362] |
| direct GBM (rival) | 0.1771 | 0.0524 | 0.0014 | -0.0398 [-0.0434, -0.0365] |
| **final: blend** | 0.1767 | 0.0524 | 0.0012 | -0.0403 [-0.0438, -0.0369] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 6.9%, e∈(0, 30]: 15.8%, e∈(30, 60]: 18.6%, e∈(60, 120]: 19.3%, e∈(120, 180]: 19.4%, e∈(180, 240]: 18.8%, e∈(240, 270]: 19.0%, e∈(270, 299]: 31.6%

quantile coverage (nominal 5/10/25/50/75/90/95%): 4.9/9.2/27.6/56.8/78.0/90.9/95.1; pinball skill vs naive Gaussian 6.50%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 0.4% | 0.0146 | 0.0110 | 24.6% | 0.0011 |
| -10 bps | 1.4% | 0.0500 | 0.0365 | 27.1% | 0.0046 |
| -5 bps | 3.0% | 0.1160 | 0.0812 | 30.1% | 0.0107 |
| +5 bps | 2.9% | 0.1106 | 0.0753 | 32.0% | 0.0012 |
| +10 bps | 1.3% | 0.0486 | 0.0346 | 28.7% | 0.0004 |
| +20 bps | 0.4% | 0.0147 | 0.0105 | 28.1% | 0.0002 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 3.6% | 3.7% | 0.2789 | 0.1874 | 32.8% | 0.989 |
| ±10 bps | 1.4% | 1.5% | 0.1058 | 0.0747 | 29.4% | 0.999 |
| ±20 bps | 0.4% | 0.4% | 0.0297 | 0.0216 | 27.2% | 1.000 |

#### 15 s head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.2576 | 0.0745 | 0.0303 |  |
| + clock-aware vol model | 0.2534 | 0.0737 | 0.0283 | -0.0042 [-0.0051, -0.0032] |
| + no-move atom & fat-tailed law | 0.2323 | 0.0701 | 0.0072 | -0.0253 [-0.0272, -0.0234] |
| + drift (structural model) | 0.2247 | 0.0678 | 0.0063 | -0.0329 [-0.0349, -0.0309] |
| + calibration from prior | 0.2188 | 0.0661 | 0.0019 | -0.0388 [-0.0421, -0.0357] |
| direct GBM (rival) | 0.2185 | 0.0659 | 0.0015 | -0.0391 [-0.0427, -0.0357] |
| **final: blend** | 0.2179 | 0.0658 | 0.0014 | -0.0397 [-0.0432, -0.0364] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 5.2%, e∈(0, 30]: 12.2%, e∈(30, 60]: 14.8%, e∈(60, 120]: 16.1%, e∈(120, 180]: 16.4%, e∈(180, 240]: 15.7%, e∈(240, 270]: 16.8%, e∈(270, 299]: 29.6%

quantile coverage (nominal 5/10/25/50/75/90/95%): 5.0/9.4/26.0/54.6/76.9/90.5/94.9; pinball skill vs naive Gaussian 5.40%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 0.6% | 0.0195 | 0.0147 | 24.6% | 0.0004 |
| -10 bps | 1.9% | 0.0652 | 0.0483 | 26.0% | 0.0008 |
| -5 bps | 4.3% | 0.1464 | 0.1054 | 28.0% | 0.0022 |
| +5 bps | 4.1% | 0.1401 | 0.1011 | 27.9% | 0.0014 |
| +10 bps | 1.9% | 0.0635 | 0.0472 | 25.7% | 0.0007 |
| +20 bps | 0.6% | 0.0197 | 0.0145 | 26.3% | 0.0003 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 5.1% | 5.3% | 0.3485 | 0.2486 | 28.7% | 0.982 |
| ±10 bps | 2.0% | 2.1% | 0.1378 | 0.1017 | 26.2% | 0.997 |
| ±20 bps | 0.6% | 0.6% | 0.0397 | 0.0296 | 25.6% | 0.999 |

#### 30 s head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.3421 | 0.1025 | 0.0283 |  |
| + clock-aware vol model | 0.3290 | 0.1019 | 0.0273 | -0.0131 [-0.0142, -0.0120] |
| + no-move atom & fat-tailed law | 0.3166 | 0.0992 | 0.0072 | -0.0255 [-0.0272, -0.0239] |
| + drift (structural model) | 0.3090 | 0.0966 | 0.0053 | -0.0331 [-0.0351, -0.0313] |
| + calibration from prior | 0.3055 | 0.0954 | 0.0029 | -0.0366 [-0.0395, -0.0338] |
| direct GBM (rival) | 0.3057 | 0.0954 | 0.0017 | -0.0364 [-0.0396, -0.0334] |
| **final: blend** | 0.3048 | 0.0952 | 0.0018 | -0.0373 [-0.0405, -0.0344] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 3.5%, e∈(0, 30]: 7.6%, e∈(30, 60]: 9.6%, e∈(60, 120]: 11.6%, e∈(120, 180]: 12.0%, e∈(180, 240]: 11.8%, e∈(240, 270]: 17.2%

quantile coverage (nominal 5/10/25/50/75/90/95%): 5.1/9.8/24.9/52.4/75.9/90.1/94.8; pinball skill vs naive Gaussian 4.44%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 1.1% | 0.0333 | 0.0255 | 23.4% | 0.0008 |
| -10 bps | 3.5% | 0.1058 | 0.0819 | 22.6% | 0.0010 |
| -5 bps | 7.7% | 0.2193 | 0.1700 | 22.5% | 0.0026 |
| +5 bps | 7.4% | 0.2120 | 0.1646 | 22.4% | 0.0024 |
| +10 bps | 3.4% | 0.1038 | 0.0803 | 22.6% | 0.0014 |
| +20 bps | 1.0% | 0.0340 | 0.0254 | 25.2% | 0.0006 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 9.0% | 9.2% | 0.5055 | 0.3874 | 23.4% | 0.959 |
| ±10 bps | 3.7% | 3.7% | 0.2217 | 0.1710 | 22.9% | 0.990 |
| ±20 bps | 1.0% | 1.1% | 0.0679 | 0.0514 | 24.4% | 0.998 |

#### 60 s head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.4422 | 0.1386 | 0.0301 |  |
| + clock-aware vol model | 0.4234 | 0.1376 | 0.0247 | -0.0187 [-0.0200, -0.0174] |
| + no-move atom & fat-tailed law | 0.4179 | 0.1358 | 0.0066 | -0.0242 [-0.0256, -0.0229] |
| + drift (structural model) | 0.4118 | 0.1335 | 0.0038 | -0.0304 [-0.0320, -0.0287] |
| + calibration from prior | 0.4104 | 0.1329 | 0.0032 | -0.0318 [-0.0338, -0.0298] |
| direct GBM (rival) | 0.4109 | 0.1330 | 0.0025 | -0.0313 [-0.0337, -0.0290] |
| **final: blend** | 0.4096 | 0.1327 | 0.0020 | -0.0325 [-0.0348, -0.0303] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 1.9%, e∈(0, 30]: 4.2%, e∈(30, 60]: 6.3%, e∈(60, 120]: 7.8%, e∈(120, 180]: 8.7%, e∈(180, 240]: 9.8%

quantile coverage (nominal 5/10/25/50/75/90/95%): 5.0/9.9/24.8/51.2/75.4/90.0/94.8; pinball skill vs naive Gaussian 4.13%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 1.9% | 0.0611 | 0.0459 | 24.8% | 0.0011 |
| -10 bps | 6.2% | 0.1766 | 0.1391 | 21.2% | 0.0009 |
| -5 bps | 13.3% | 0.3274 | 0.2650 | 19.1% | 0.0026 |
| +5 bps | 13.0% | 0.3201 | 0.2597 | 18.9% | 0.0030 |
| +10 bps | 6.1% | 0.1747 | 0.1369 | 21.6% | 0.0023 |
| +20 bps | 1.8% | 0.0616 | 0.0461 | 25.2% | 0.0008 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 15.0% | 15.3% | 0.7073 | 0.5627 | 20.4% | 0.913 |
| ±10 bps | 6.4% | 6.5% | 0.3627 | 0.2833 | 21.9% | 0.969 |
| ±20 bps | 1.8% | 1.9% | 0.1231 | 0.0922 | 25.1% | 0.989 |

#### 5-minute close head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.5289 | 0.1679 | 0.0402 |  |
| + clock-aware vol model | 0.5025 | 0.1649 | 0.0152 | -0.0264 [-0.0296, -0.0234] |
| + no-move atom & fat-tailed law | 0.4913 | 0.1646 | 0.0162 | -0.0376 [-0.0419, -0.0335] |
| + drift (structural model) | 0.4885 | 0.1635 | 0.0161 | -0.0404 [-0.0448, -0.0363] |
| + calibration from prior | 0.4881 | 0.1635 | 0.0148 | -0.0408 [-0.0452, -0.0366] |
| direct GBM (rival) | 0.4880 | 0.1637 | 0.0095 | -0.0409 [-0.0459, -0.0362] |
| **final: blend** | 0.4867 | 0.1633 | 0.0125 | -0.0422 [-0.0470, -0.0377] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 0.4%, e∈(0, 30]: 0.9%, e∈(30, 60]: 2.0%, e∈(60, 120]: 3.9%, e∈(120, 180]: 6.8%, e∈(180, 240]: 12.1%, e∈(240, 270]: 21.3%, e∈(270, 299]: 33.3%

quantile coverage (nominal 5/10/25/50/75/90/95%): 5.0/9.9/25.0/50.9/74.8/89.6/94.6; pinball skill vs naive Gaussian 3.72%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 4.1% | 0.1370 | 0.1027 | 25.0% | 0.0020 |
| -10 bps | 12.3% | 0.2991 | 0.2432 | 18.7% | 0.0025 |
| -5 bps | 22.9% | 0.4351 | 0.3705 | 14.8% | 0.0041 |
| +5 bps | 22.4% | 0.4327 | 0.3677 | 15.0% | 0.0025 |
| +10 bps | 12.2% | 0.3053 | 0.2447 | 19.8% | 0.0059 |
| +20 bps | 4.0% | 0.1387 | 0.1030 | 25.7% | 0.0028 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 24.3% | 24.7% | 0.8586 | 0.7002 | 18.4% | 0.831 |
| ±10 bps | 12.5% | 12.7% | 0.6031 | 0.4757 | 21.1% | 0.893 |
| ±20 bps | 4.0% | 4.1% | 0.2744 | 0.2023 | 26.3% | 0.929 |