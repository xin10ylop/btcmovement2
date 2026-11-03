# BTC 5-minute reference-relative prediction

A from-scratch system that continuously predicts how Bitcoin will trade **relative to the
fixed opening price of the current 5-minute candle**, at three horizons:

| head | question it answers at every second `e` of the window |
|---|---|
| **30s** | distribution of the price 30 s from now, relative to the window's fixed reference P0 |
| **60s** | the same, 60 s ahead |
| **close (5m)** | distribution of the 5-minute candle **close** relative to P0 (at `e = 0` this is the pure 5-minute forecast) |

For each head it outputs `P(S >= P0)`, quantiles of the target price in dollars,
`P(touch P0·(1±k))` within the horizon, and first-passage probabilities
(up-level first / down-level first / neither) for symmetric bands around P0.

> RESULTS_PLACEHOLDER

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
  trade count, largest aggressive order) – the price-discovery market.

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
4. **Fat tails and ties.** Increments are heavily fat-tailed relative to trailing vol; in
   quiet regimes the price often does not move at all (7.5% of 30s increments in 2025 were
   exactly zero), which matters because ties resolve "up".

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
* **residual law `eps`** - empirical, fat-tailed, bucketed by horizon length.
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

> LADDER_PLACEHOLDER

## 6. Usage

```bash
pip install -r requirements.txt
python -m btcmodel.data 2025-01-01 2026-11-02      # download + compact (≈20 min)
python -m btcmodel.timeline                         # continuous 1s timeline
python -m btcmodel.dataset                          # features/targets per split
python scripts/train_eval.py valid h30              # train + evaluate one head (h30|h60|close)
python scripts/train_eval.py test  close            # final holdout
python -m tests.test_no_lookahead
```

Live inference: `btcmodel/predictor.py` (`Predictor.ingest(...)`, `Predictor.predict(epoch_second, p0)`).
