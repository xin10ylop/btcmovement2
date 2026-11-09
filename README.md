# BTC 5-minute reference-relative prediction

A from-scratch system that continuously predicts how Bitcoin will trade **relative to the
fixed opening price of the current 5-minute candle**, at five horizons:

| head | question it answers at every second `e` of the window |
|---|---|
| **10s / 15s** | distribution of the price 10 s / 15 s from now, relative to the window's fixed reference P0 |
| **30s** | the same, 30 s ahead |
| **60s** | the same, 60 s ahead |
| **close (5m)** | distribution of the 5-minute candle **close** relative to P0 (at `e = 0` this is the pure 5-minute forecast) |

For each head it outputs `P(S >= P0)`, quantiles of the target price in dollars,
`P(touch P0·(1±k))` within the horizon, and first-passage probabilities
(up-level first / down-level first / neither) for symmetric bands around P0.

**Headline (untouched 6-month holdout, May-Nov 2026, regime-aware walk-forward model):**
log-loss skill vs a naive Gaussian random walk of **18.6% / 15.4% / 10.9% / 7.4% / 8.0%** for
the 10s / 15s / 30s / 60s / close heads, positive in every month and every market regime,
calibrated (ECE ≤ 0.0125). Directional AUC 0.66 at 10 s decaying to 0.54 for the 5-minute
direction at the open - an honest picture of where BTC is and is not predictable.
See section 6.

**Cross-asset add-on (ETH, pre-registered, adopted on validation):** BTC spot is the slowest of
the BTC/ETH spot and perp markets, so ETH's perpetual carries information BTC has not priced
yet. Adding 20 causal ETH features lifts holdout skill to **19.0% / 15.8% / 11.1% / 7.5% / 8.0%**
(paired gain significant on the 10s-60s heads, better in 7 of 7 months each; on the close head
the same small gain as on validation, −0.0001 log-loss, but its CI includes zero). See section 6b.

---

## 1. Definitions (the reference never moves)

* Window = `[T0, T0+300s)`, `T0` on a UTC 5-minute boundary.
  **P0 = the first trade of the window** (open of the first Binance BTCUSDT spot 1s bar,
  identical to the Binance 5m candle open). P0 is fixed for all 300 seconds.
* Decision at elapsed second `e`: everything strictly before `T0+e` is known, plus P0.
  Current price `S_e` = P0 at `e=0`, else the close of bar `T0+e-1`.
* Targets: 30s/60s heads use `S_{e+h}` (only while `e+h <= 300`, so a target never
  leaves the window and never mixes two references); the close head uses the 5m close.
  "Up" means `S_target >= P0` (ties count as up, the convention of 5-minute up/down markets).
* Touch uses every trade in the horizon (1s highs/lows).

`tests/test_no_lookahead.py` replaces every bar from the decision time onward with
garbage and asserts that no feature changes (including `e = 0`, where only P0 is known).

## 2. Data

Binance public archive (`data.binance.vision`), 2025-01-01 → 2026-11-02, 671 days,
SHA-256 verified:
* spot BTCUSDT 1-second klines (OHLC, volume, taker-buy volume, trade count) – the reference market;
* USD-M perpetual BTCUSDT aggTrades, aggregated to 1s bars (price, taker buy/sell volume,
  trade count, largest aggressive order) – the price-discovery market;
* (cross-asset add-on) the same two series for **ETHUSDT**, on the identical second grid
  (both timelines start 2025-01-01), stored separately (`/home/user/data_eth`).

The archive has no USD-M aggTrades file for 2026-10-28 (404 for both symbols); that holdout day
has no perp data in either market (cross-asset features are NaN there, 0.55% of holdout rows).

Splits (chronological, 1-day embargo):

| split | dates | use |
|---|---|---|
| train | 2025-01-02 → 2025-12-31 | fitting; clock seasonality estimated here only |
| valid | 2026-01-02 → 2026-04-30 | early stopping and every design decision |
| **test** | 2026-05-02 → 2026-11-02 | untouched final holdout, evaluated once |

All research in `research/` uses 2025 data only.

## 3. What the data says about BTC at this scale (research findings, 2025 only)

1. **There is no reference-price effect.** Conditional on distance from P0, the
   probability that the remaining move is up is 0.50 in every bucket and at every
   elapsed time (`research/eda4_reference.py`). The market does not know about the 5-minute
   open. Therefore the reference must only decide *where the predictive distribution is
   read*, never *what the dynamics are* - this is the core architectural principle.
2. **Volatility runs on a clock.** The first second of each hour has 2.5x average activity,
   every 5-minute boundary ~1.8x, decaying to ~0.75x by the end of each window. 9-11 ET on
   weekdays is 2.5-3x; weekends 0.3x; Sunday 18:00 ET (CME reopen) jumps to 1.6x.
   The profile is stable between H1 and H2 2025. Consequence: at `e = 285` only ~3.7% of
   the window's variance remains, not 5% - a `sigma*sqrt(tau)` model is systematically
   overconfident about remaining uncertainty early and underconfident late.
3. **Spot lags the perpetual.** Perp returns lead spot (lag-1 correlation 0.18) and the
   perp-spot basis deviation predicts spot's next move. The best estimate of the "efficient
   price" is not the last spot trade - so there is genuine short-horizon drift, and it
   saturates with horizon (it is a gap that closes), which is why it can be shared across heads.
4. **Fat tails and a "no-move" atom.** Increments are heavily fat-tailed relative to
   trailing vol, and the price frequently does not move at all: with 1 s left in the window
   there is a 48% chance the price is unchanged (24% with 6-10 s left, 7.5% over any 30 s),
   and in quiet regimes ~3x more often than in active ones. Because ties resolve "up", this
   atom is first-order for `P(close >= P0)` at the end of the window - a Gaussian with a
   *good* volatility estimate is worse than the naive model there.
5. **The market's mechanics drift, not just its volatility.** Monthly averages of online
   microstructure estimates (`research`/`btcmodel/regime.py`):

   | period | perp->spot lead-lag beta | spot 1s autocorr. | 5m variance ratio | ann. vol |
   |---|---|---|---|---|
   | 2025 (train) | 0.16-0.19 | 0.10-0.15 | 1.41-1.86 | 20-48% |
   | Jan-Apr 2026 (valid) | 0.13-0.15 | 0.085-0.10 | 1.29-1.49 | 29-57% |
   | May-Nov 2026 (test) | 0.12-0.15 | 0.10-0.12 | 1.56-1.77 | 24-38% |

   Spot became structurally more efficient (it lags the perp less), so the main directional
   signal is weaker in 2026 than in the training year. A regime-blind model would keep
   trusting the 2025 strength of that signal.
6. **ETH leads BTC, not the other way round** (2025-only linear probe from the ETH
   replication, xin10ylop/eth3 `research/r2_btc_leads_eth.py`): at 1 s,
   corr(BTC spot[t+1], ETH perp[t]) = 0.157 vs corr(ETH spot[t+1], BTC perp[t]) = 0.094. Adding
   ETH perp/spot moves and a "BTC lagging ETH" gap to BTC's own signals raised the next-10s OOS
   R² from 5.07% to 5.91% (next-60s 1.99% → 2.30%); BTC added almost nothing to ETH. The
   perpetuals lead both spots and BTC spot is the slowest of the four markets.

## 4. Model

Write `x = log(S_e/P0)` (known) and `R = log(S_target/S_e)` (unknown). Then
`log(S_target/P0) = x + R`, so every reference-relative question is a question about the
conditional law of the increment `R`, evaluated at a reference-dependent point:

```
P(S_target >= P0)        = P(R >= -x)
P(S_target >= P0(1+k))   = P(R >= log(1+k) - x)
quantiles of S_target/P0 = x + quantiles of R
touch / first passage    = law of the running max / min of the increment path
```

Each head models `R = sigma * (mu + eps)`:

* **scale `sigma`** - LightGBM estimate of `E|R|` as a multiplicative correction of the
  naive trailing-vol scale. Its key input is a *clock projection*: seasonally adjusted
  realized variance times the integrated clock profile over exactly the seconds that remain
  in the horizon (`lproj_*`), plus trade intensity, ranges, jump ratios, window range.
* **drift `mu`** - LightGBM regression of the standardized increment on lead-lag
  (perp vs spot), basis deviations, taker-flow imbalance (spot and perp), multi-scale
  returns, window path statistics, multi-timeframe candle opens. The close head also
  receives the 30s/60s heads' drift (*cross-horizon stacking*: the gap is learned where its
  signal-to-noise ratio is highest and transferred to the long horizon).
* **no-move atom `pi0`** - LightGBM `P(R = 0 | state)`. The continuous part lives on
  `R != 0` with scale `sigma_c = sigma/(1-pi0)`; by the identity
  `E[R/sigma_c | R != 0] = E[R]/sigma = mu` the drift carries over unchanged, and
  `P(S_T >= P0) = pi0*1{x >= 0} + (1-pi0)*P_c(R >= -x)`. Quantiles are those of the
  atom + continuous mixture.
* **residual law `eps`** - empirical, fat-tailed, fitted on non-zero moves, bucketed by
  horizon length.
* **regime state** (`btcmodel/regime.py`) - online, causal estimates of the market's
  current mechanics: lead-lag beta and the realized *catch-up coefficient* of spot to the
  basis gap (how strong the main drift signal is right now), 1s autocorrelation, variance
  ratios (trending vs mean-reverting), 7-day vol and the 1d/7d vol ratio, vol-of-vol,
  vol-standardized 4h/1d trend, perp premium level, activity regime. They enter the scale,
  atom, drift, calibration and direct models, so that the models condition on the regime
  instead of averaging over it.
* **cross-asset add-on** (`btcmodel/xasset.py`, opt-in `BTC_XASSET=1`) - 20 strictly causal
  ETH features: ETH perp moves over 1/2/3/5/10/30/60/300 s (ETH vol units), the "BTC lagging
  ETH" gap (BTC spot move − rolling-1h beta × ETH perp move) over the same lags (BTC vol units),
  ETH spot moves over 1/5/30 s, and the 1h beta. They enter the **drift model and the direct
  GBM** only (directional information); scale, atom and calibration are unchanged.
* **calibration from the prior** - a small LightGBM whose `init_score` is the structural
  logit; it can only move away from the structural answer when that lowers out-of-sample
  loss (early stopping on the validation set).
* **touch / first passage** - closed-form Brownian priors with the head's own scale and
  drift (reflection principle; method-of-images double-barrier formula with Girsanov drift,
  verified by Monte Carlo), then a boosted correction. The touch corrector is
  *monotone-constrained* in the distance to the level, so touch probabilities are coherent
  across thresholds.

**Cross-fitting.** Each stage is trained on out-of-fold predictions of the previous stage
(4 contiguous time blocks with a purge gap), so no stage ever learns from in-sample-
optimistic inputs.

## 5. Evaluation protocol

* Effective sample size is the number of windows/days, not decision points (all 60
  decision points of a window share one outcome). Every confidence interval is a
  **day-block bootstrap** of the loss difference.
* The honest benchmark is *not* 0.5: late in a window `P(close >= P0)` is nearly known from `x`
  alone. Every model is compared with a **naive Gaussian random walk** using trailing vol,
  and each architectural idea is added one rung at a time (ladder below).
* The directional (increment) AUC is reported separately; it is the pure measure of
  "which way next" skill, free of the trivially-known distance to the reference.


## 6. Results on the untouched holdout (2026-05-02 → 2026-11-02, 53,280 windows)

Final model = **walk-forward, regime-aware** (trained through 2026-04-30, i.e. exactly what
would have been deployed on 1 May 2026), scored once on six months it never saw.
Log-loss of `P(S_target >= P0)` at every 5th second of every window; 95% CIs from a
day-block bootstrap.

| head | naive Gaussian | **final** | skill | ΔLL [95% CI] | ECE | directional AUC | drift R² |
|---|---|---|---|---|---|---|---|
| 10 s | 0.2169 | **0.1767** | **18.6%** | −0.0403 [−0.0438, −0.0369] | 0.0012 | **0.662** | 6.8% |
| 15 s | 0.2576 | **0.2179** | **15.4%** | −0.0397 [−0.0432, −0.0364] | 0.0014 | 0.650 | 6.0% |
| 30 s | 0.3421 | **0.3048** | **10.9%** | −0.0373 [−0.0405, −0.0344] | 0.0018 | 0.620 | 4.3% |
| 60 s | 0.4422 | **0.4096** | **7.4%** | −0.0325 [−0.0348, −0.0303] | 0.0020 | 0.589 | 2.6% |
| 5 m close | 0.5289 | **0.4867** | **8.0%** | −0.0422 [−0.0470, −0.0377] | 0.0125 | 0.566 | 1.7% |

* **Consistency:** positive skill in **7 of 7 months for all five heads**, and significant in
  every regime stratum (low/mid/high vol, weekday/weekend, US session, weak/strong trend,
  rising vol, and the 5% most volatile days).
* **Where the edge lives** (`artifacts/figures/skill_by_elapsed_test_wf5.png`): at the window
  open the 5-minute direction is barely predictable (AUC 0.540, skill 0.36%), the 10 s
  direction clearly is (AUC 0.658, skill 6.9%). Skill grows as the window unfolds and reaches
  **~36% in the final 15 seconds** - that is where clock-aware volatility, the no-move atom,
  fat tails and the close-pressure regime matter most.
* **Magnitude:** the predicted expected absolute move beats the naive scale by ~7% in MSE
  (Spearman with realized |move| 0.57 vs 0.54 on the close head). Pinball (quantile) skill vs
  a naive Gaussian: 6.5% / 5.4% / 4.4% / 4.1% / 3.7% (10s → close). Close-head quantile
  coverage is within 0.5 pp of nominal at every level (5.0/9.9/25.0/50.9/74.8/89.6/94.6%).
  The 10 s / 15 s heads cover the tails correctly but *over*-cover the central levels
  (10 s median: 56.8%) - not miscalibration but the no-move atom: when a quantile lands on
  "price unchanged", `P(Y <= q)` jumps past the nominal level, as it must for a distribution
  with a point mass (17% of 10-second intervals have no price change).
* **Touch probabilities** (trading at `P0·(1±k)`, k = 5/10/20 bps, within the horizon):
  15-32% log-loss skill vs the Brownian reflection formula across all heads and levels
  (close head 15-26%), ECE ≤ 0.011.
* **First passage** (±k band around P0: up first / down first / neither): 18-33% skill vs the
  Brownian double-barrier formula (close head 18-26%).

### What regime handling is worth (paired, identical holdout rows)

Regime-blind model trained on 2025 only (frozen) vs the final regime-aware walk-forward model
(`artifacts/compare_test_frozen_vs_wf5.txt`):

| head | rows | regime-blind skill | regime-aware skill | ΔLL [95% CI] |
|---|---|---|---|---|
| close | all | 7.58% | **7.98%** | −0.0021 [−0.0027, −0.0015] |
| close | last 60 s | 23.1% | **25.2%** | −0.0079 [−0.0100, −0.0059] |
| close | last 30 s | 28.7% | **31.9%** | −0.0104 [−0.0132, −0.0076] |
| 30 s | last 30 s | 21.0% | **24.8%** | −0.0140 [−0.0167, −0.0114] |
| 60 s | last 60 s | 14.6% | **15.7%** | −0.0046 [−0.0055, −0.0037] |

Better on every slice of every head, with gains concentrated 5-10x in the final seconds -
exactly where the close-pressure regime acts. The mechanism found on validation data
confirmed itself out of sample.

### Online recalibration (causal, adopted per pre-registered rule 4)

Each day a 2-parameter logistic map per elapsed-time bucket is refit on the previous 14 days
of *already resolved* windows and applied to the next day (`scripts/regime_eval.py`).

| head | ΔLL [95% CI] | ECE before → after | adopted |
|---|---|---|---|
| 10 s | −0.00015 [−0.00023, −0.00008] | 0.0012 → 0.0011 | yes |
| 15 s | −0.00017 [−0.00027, −0.00008] | 0.0014 → 0.0011 | yes |
| 30 s | −0.00015 [−0.00024, −0.00007] | 0.0018 → 0.0016 | yes |
| 60 s | −0.00001 [−0.00013, +0.00010] | 0.0020 → 0.0019 | no (CI includes 0) |
| close | −0.00098 [−0.00160, −0.00041] | **0.0125 → 0.0032** | yes |

The regime-aware model needs about half the online repair the regime-blind one needed
(close head: −0.0010 vs −0.0019), i.e. it already absorbs most of the drift itself.
Regime-stratified skill (`artifacts/regime_test_wf5.txt`): significant in all 50
head x stratum cells; even on the 5% most volatile days skill is 2.4-7.0%.

### The ladder (close head, holdout) - what each idea contributes

| rung | log-loss | ΔLL vs naive |
|---|---|---|
| naive Gaussian random walk, trailing 5-min vol | 0.5289 | — |
| + clock-aware learned volatility | 0.5025 | −0.0264 |
| + no-move atom & empirical fat-tailed law | 0.4913 | −0.0376 |
| + drift (perp-led efficient-price gap, flows, stacking) | 0.4885 | −0.0404 |
| + calibration from the structural prior | 0.4881 | −0.0408 |
| direct GBM on all features (rival) | 0.4880 | −0.0409 |
| **final: equal-weight logit blend (pre-registered)** | **0.4867** | **−0.0422** |

Full tables for every head: `artifacts/tables_test_wf5.md`. Figures: `artifacts/figures/`.

## 6b. Cross-asset add-on: what ETH is worth (pre-registered, paired)

Protocol (`artifacts/PREREGISTRATION.md`, amendment 2, committed before any model with these
features was fitted): feature set frozen from the 2025 probe; decision on **validation only**
with two arms identical except for the ETH features - A `_r5` (regime-aware) vs B `_rx5` (+ ETH),
both fitted on 2025, early-stopped and scored on Jan-Apr 2026. Rule: adopt iff the close head's
blended log-loss improves with a day-block CI excluding zero **and** no shorter head is worse.

**Validation decision** (`artifacts/compare_valid_r5_vs_rx5.txt`, 119 days) - **adopted**:

| head | skill A | skill B | ΔLL B−A [95% CI] | drift R² A → B | dir. AUC A → B |
|---|---|---|---|---|---|
| 10 s | 12.09% | 12.44% | −0.00076 [−0.00086, −0.00067] | 5.46% → 6.14% | 0.630 → 0.639 |
| 15 s | 9.59% | 9.86% | −0.00069 [−0.00078, −0.00060] | 4.54% → 5.11% | 0.616 → 0.623 |
| 30 s | 6.28% | 6.46% | −0.00061 [−0.00073, −0.00051] | 2.88% → 3.24% | 0.587 → 0.593 |
| 60 s | 4.05% | 4.13% | −0.00035 [−0.00047, −0.00024] | 1.56% → 1.76% | 0.560 → 0.564 |
| close | 5.59% | 5.61% | **−0.00011 [−0.00022, −0.00001]** | 1.10% → 1.25% | 0.548 → 0.552 |

(The validation skills are lower than the holdout's because these research arms are fitted on
2025 only and cannot know the close-pressure regime that appeared in Feb 2026.) The drift R²
gain at 10 s (+0.69 pp, ×1.13) reproduces the 2025 probe (+0.85 pp, ×1.17) out of sample.

**Holdout** (WF-x5 = the WF-v5 recipe + ETH features, refit walk-forward through 2026-04-30,
scored once; paired with the current WF-v5 on identical rows, `artifacts/compare_test_wf5_vs_wfx5.txt`,
breakdown `artifacts/xasset_test_wf5_vs_wfx5.md`):

| head | naive Gaussian | WF-v5 | **WF-x5** | skill v5 → x5 | ΔLL x5 − v5 [95% CI] | ECE x5 | dir. AUC v5 → x5 | drift R² v5 → x5 | months better |
|---|---|---|---|---|---|---|---|---|---|
| 10 s | 0.2169 | 0.1767 | **0.1757** | 18.6% → **19.0%** | −0.00094 [−0.00103, −0.00086] | 0.0011 | 0.662 → **0.672** | 6.8% → **7.7%** | 7/7 |
| 15 s | 0.2576 | 0.2179 | **0.2170** | 15.4% → **15.8%** | −0.00090 [−0.00100, −0.00082] | 0.0015 | 0.650 → 0.658 | 6.0% → 6.7% | 7/7 |
| 30 s | 0.3421 | 0.3048 | **0.3040** | 10.9% → **11.1%** | −0.00082 [−0.00093, −0.00073] | 0.0018 | 0.620 → 0.627 | 4.3% → 4.7% | 7/7 |
| 60 s | 0.4422 | 0.4096 | **0.4090** | 7.4% → **7.5%** | −0.00058 [−0.00071, −0.00047] | 0.0019 | 0.589 → 0.593 | 2.6% → 2.8% | 7/7 |
| 5 m close | 0.5289 | 0.4867 | 0.4866 | 8.0% → 8.0% | −0.00009 [−0.00021, +0.00002] | 0.0124 | 0.566 → 0.569 | 1.7% → 1.8% | 5/7 |

* **Short heads: clean replication.** Every 10s-60s gain is significant and larger than on
  validation, positive in every holdout month, and largest at the window open (10 s at `e = 0`:
  skill 6.9% → 7.5%, ΔLL −0.0041), where the head knows least about its own window. Touch and
  first-passage skill rise by 0.1-0.4 pp on every short head with unchanged calibration.
* **Close head: not shown either way.** The gain is the same size as on validation but its CI
  now includes zero. Where it acts moved: on validation (models fitted on 2025) it helped early
  in the window and hurt in the final 30 s (+0.0003, significant); on the holdout (walk-forward,
  fitted on data that include the 2026 close-pressure regime) it is neutral before the last
  minute and helps in it (final 30 s: −0.00067 [−0.00086, −0.00048]). Per the pre-registered rule
  11 the add-on stays adopted (the close head is not significantly worse), and the cross-asset
  models are the deployment default.
* **Regimes and recalibration** (`artifacts/regime_test_wfx5.txt`): skill stays significant in
  every head x regime stratum (minimum 2.35%, on the 5% most volatile days). The online
  recalibrator passes rule 4 for exactly the same heads as with WF-v5 (10s/15s/30s/close; close
  ECE 0.0124 → 0.0034, ΔLL −0.00091 [−0.00153, −0.00035]; 60 s not adopted), so
  `OnlineRecalibrator()` defaults are unchanged. After recalibration the close-head difference
  between the two models shrinks further (0.48569 vs 0.48573).
* **Data check.** Before any of this, the packaged WF-v5 models were re-scored on the
  re-downloaded holdout: naive, scale model, direct GBM and touch layer reproduce the committed
  numbers bit for bit (the structured path differs by ≤ 5e-6 log-loss from the compacted
  residual law), so both arms see exactly the original data.
* **Live = batch.** `tests/test_predictor_consistency.py` feeds raw BTC + ETH second buffers to
  the live `Predictor`: max |batch − live| = 0.0 on all five WF-x5 heads, and with the ETH feed
  stale the regime-only fallback reproduces WF-v5 exactly (`artifacts/predictor_consistency_wfx5.txt`).

## 7. Honest limitations

* **The 5-minute direction at the open is barely predictable** (AUC ~0.54 at `e = 0` on the
  holdout). That is what an efficient market looks like; the edge of this system lies in
  volatility, tails, the no-move atom, path probabilities and the short-horizon drift - and
  therefore grows as the window unfolds.
* **Reference feed.** P0 and the close are Binance BTCUSDT spot. If your contracts settle on
  another feed (e.g. an oracle aggregating several venues), retrain with that feed as the
  "spot" series; the architecture does not change.
* **No costs, fees, latency or market impact** are modelled - these are probability
  forecasts, not a trading strategy. A forecast edge in log-loss is an edge in Kelly growth
  only against a counterparty pricing with the weaker forecast and before costs.
* **The cross-asset models need a second live feed.** ETH spot and perp 1s bars must arrive on
  the same second grid as BTC; feed latency between venues is not modelled. In training the ETH
  side was essentially never missing (0.0005% of rows), so the predictor does not feed NaNs to
  these models: when ETH data are missing, stale or gapped it answers with a regime-only model
  set (`fallback=`, labelled in the output). The ETH lead is a market-structure property and
  can weaken like the perp->spot lead did in 2026 - monitor it with the monthly refit.
* **Regimes drift.** The close-pressure regime appeared in Feb 2026 and may change or
  vanish. The model reads its current intensity online (`reg_cp*`), but nothing replaces
  retraining: refit monthly (`scripts/train_production.py`), run the online recalibrator,
  and monitor rolling log-loss vs the naive baseline (alarm if skill < 0 over 3 days).

## 8. Usage

```bash
pip install -r requirements.txt
python -m btcmodel.data 2025-01-01 2026-11-02      # download + compact (~20 min)
python -m btcmodel.timeline                         # continuous 1s timeline
python -m btcmodel.dataset                          # features/targets per split
python scripts/build_targets_extra.py               # 10s/15s targets
python scripts/build_regime.py                      # regime-state features
python scripts/train_eval.py valid h30              # research protocol (train 2025, valid 2026H1)
python scripts/evaluate_saved.py test h30           # frozen models on the holdout
BTC_REGIME=1 python scripts/train_production.py --end 2026-04-30 --tag _wf5   # walk-forward
BTC_REGIME=1 python scripts/train_production.py     # deployment refit on everything
python -m tests.test_no_lookahead
python -m tests.test_regime_no_lookahead
python -m tests.test_predictor_consistency _wf5 test h10 h15 h30 h60 close
```

Cross-asset add-on (section 6b; staged driver: `scripts/run_xasset_pipeline.sh`):

```bash
BTC_ASSET=ETH python -m btcmodel.data 2025-01-01 2026-11-02 6   # ETHUSDT spot 1s + perp aggTrades
BTC_ASSET=ETH python -m btcmodel.timeline                      # same second grid as BTC
python scripts/build_xasset.py                                 # <split>_xasset.parquet (BTC cache)
python -m tests.test_xasset_no_lookahead
BTC_REGIME=1 BTC_XASSET=1 BTC_EVENTS=0 BTC_TAG=_rx5 python scripts/train_eval.py valid h10   # research arm
BTC_REGIME=1 BTC_XASSET=1 python scripts/train_production.py h10 --end 2026-04-30 --tag _wfx5 --resume
BTC_REGIME=1 BTC_XASSET=1 BTC_TAG=_wfx5 python scripts/evaluate_saved.py test h10
python scripts/compare_versions.py test _wf5 _wfx5             # paired, identical rows
python scripts/xasset_report.py test _wf5 _wfx5                # by elapsed time and month
python -m tests.test_predictor_consistency _wfx5 test h10 h15 h30 h60 close --fallback _wf5
```

Four packaged model sets are included (same architecture and hyper-parameters; the `x` sets add
the 20 ETH features):

| directory | trained on | use |
|---|---|---|
| `models/wf5_20260430/` | 2025-01 → 2026-04-30 | **the holdout-evaluated models** - every number in section 6 |
| `models/prod_20261102/` | 2025-01 → 2026-11-02 (2025 thinned to every other window = recency tilt; early stopping on 2026-10-03 → 11-02) | deployment, BTC only - the same recipe refit on the newest data; no untouched data remains to score it, so its quality is inferred from the wf5 holdout. **Fallback** for the `prodx` set |
| `models/wfx5_20260430/` | 2025-01 → 2026-04-30, + ETH | the holdout-evaluated cross-asset models - every WF-x5 number in section 6b |
| `models/prodx_20261102/` | as `prod_20261102`, + ETH | **deployment default** (cross-asset); needs the ETH feed, falls back to `prod_20261102` without it |

Live inference (`btcmodel/predictor.py`):

```python
from btcmodel.predictor import Predictor
from btcmodel.recalibration import OnlineRecalibrator

rc = OnlineRecalibrator()                       # optional, adopted for 10s/15s/30s/close
pred = Predictor.load_dir("models/prodx_20261102", recalibrator=rc,   # cross-asset (default)
                          fallback="models/prod_20261102")            # BTC-only if ETH is missing
pred.ingest(arrays, t0_epoch,   # >= 9 days of 1s spot klines + perp 1s bars (fields as in
            xarrays=eth_arrays) # btcmodel/timeline.FIELDS), buffer starting on an hour boundary;
                                # eth_arrays: the same fields for ETHUSDT on the same second grid
                                # (BTC-only sets: Predictor.load_dir("models/prod_20261102"), no xarrays)
out = pred.predict(now_epoch_second, p0=window_open_price)
out["model"]                                          # "cross-asset" or "fallback (...)"
out["heads"]["close"]["p_above_ref"]                  # P(5m close >= P0)
out["heads"]["h10"]["p_above_ref"]                    # P(price in 10 s >= P0)
out["heads"]["h30"]["quantiles_usd"]                  # price distribution 30 s ahead, in $
out["heads"]["close"]["touch"]["+10bps"]              # P(trade at >= P0*1.001 before the close)
out["heads"]["close"]["first_passage"]["+/-10bps"]    # up-first / down-first / neither
# once outcomes are known:  rc.add(head, e, p_blend, y, day);  once a day:  rc.refit(today)
```

`python scripts/demo_live.py` replays a holdout window second by second through this exact
interface (it only ever sees data up to the decision second). To refresh the deployment models
later (recommended monthly): download the new days, rebuild the caches, then run
`BTC_REGIME=1 python scripts/train_production.py --old-thin 2 --resume` and
`python scripts/package_models.py prod models/prod_<date>`. Needs ~10 GB RAM per head and
~1 h per head on 4 CPUs; each head runs in its own process (`--resume` skips finished heads).
With the cross-asset add-on (`BTC_XASSET=1`) the full-history refit of the 10 s head peaks at
~14 GB resident: on a 15 GB machine enable a few GB of swap (it was OOM-killed without it).

## 9. Repository map

| path | content |
|---|---|
| `btcmodel/data.py`, `timeline.py` | download, checksum, compaction, continuous 1s timeline |
| `btcmodel/features.py` | causal features, reference-relative targets, window path stats |
| `btcmodel/regime.py` | online regime-state features (incl. close pressure) |
| `btcmodel/xasset.py` | cross-asset (ETH) add-on features, causal |
| `btcmodel/models.py` | head model: scale, no-move atom, drift, residual law, calibration |
| `btcmodel/events.py` | touch and first-passage layers |
| `btcmodel/prob.py` | Brownian barrier formulas, scoring rules, day-block bootstrap |
| `btcmodel/predictor.py` | live inference (`Predictor.load_dir`) |
| `btcmodel/recalibration.py` | causal online recalibrator |
| `models/wf5_20260430/` | packaged holdout-evaluated models + clock seasonality |
| `models/prod_20261102/` | packaged deployment models, refit through 2026-11-02 |
| `models/wfx5_20260430/`, `models/prodx_20261102/` | the same two sets with the cross-asset add-on |
| `research/` | the studies behind every design decision (2025/validation data only) |
| `scripts/` | training, frozen evaluation, walk-forward, robustness, regime analysis, figures |
| `artifacts/` | results (JSON/text), tables, figures, pre-registration |
| `scripts/build_xasset.py`, `run_xasset_pipeline.sh`, `xasset_report.py` | cross-asset cache, staged pipeline, paired report |
| `tests/` | lookahead tests (incl. cross-asset), live-vs-batch consistency (incl. ETH buffer and fallback) |
