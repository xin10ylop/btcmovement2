# Pre-registered decisions (written before the corresponding results were seen)

1. **Final P(S >= P0) forecast** = equal-weight logit blend of the structured head model and
   the direct GBM. Chosen on the 30s validation results (v2); weight 0.5 is not tuned further.
2. **v3 vs v4 (regime-state features)** - decided on VALIDATION only: v4 is adopted if its
   blended log-loss is lower on the close head and not higher on the 30s and 60s heads;
   otherwise v3 is kept. Written before v4 validation results existed.
3. **Holdout (test, 2026-05-02 .. 2026-11-02)** is scored with frozen models selected by the
   rules above. The walk-forward refit (data through 2026-04-30, early stopping on the last
   30 days before that) is reported alongside as the deployment-realistic variant; it does
   not change the selection.
4. **Online recalibration** (14-day causal window) is reported as an add-on; it is adopted
   for deployment only if it improves the holdout log-loss with a day-block bootstrap CI
   excluding zero.

## Amendment (before ANY holdout/test evaluation was run)

Discovery on validation data (research/eda5_close_pressure.py, train+valid only): from
February 2026 the price is pushed toward the window's own reference in the final 5-10 s
(+0.12..+0.20 sigma, t = 3..7.5; crossing rate ~13% -> ~25%; mid-window placebo null). This
regime did not exist in 2025, so no model fitted on 2025 can learn it. The v3-vs-v4 rule
above is therefore moot (v4 was stopped) and replaced by:

5. **Final model ("WF-v5")** = same architecture + regime-state features (incl. online
   close-pressure intensity `reg_cp*`), refit walk-forward on all data through 2026-04-30
   (early stopping on the last 30 days before that). This is what would have been deployed
   on 2026-05-01.
6. **Reference ("v3 frozen")** = the 2025-trained model (no regime features), scored on the
   same holdout, to measure what regime handling is worth.
7. Online recalibration remains an add-on under rule 4.

## Amendment 2 - cross-asset (ETH) add-on (written before any model with these features was fitted or scored)

Motivation: a 2025-only linear probe in the ETH replication (xin10ylop/eth3,
`research/r2_btc_leads_eth.py`) found that ETH adds a lot to BTC's own short-horizon signals
(BTC spot next-10s OOS R2 5.07% -> 5.91%, next-60s 1.99% -> 2.30%) while BTC adds almost nothing
to ETH; the perpetuals lead both spots and BTC spot is the slowest of the four markets.
Nothing below has been looked at on 2026 data.

8. **Features** = exactly `XASSET_FEATS` in `btcmodel/xasset.py` (20): the probe's set - ETH perp
   moves over 1/2/3/5/10/30/60/300 s (ETH vol units), the "BTC lagging ETH" gap (BTC spot move
   minus rolling-1h beta x ETH perp move) over the same 8 lags (BTC vol units), ETH spot moves over
   1/5/30 s - plus the rolling-1h beta itself. Strictly causal (bars < decision second of both
   markets; `tests/test_xasset_no_lookahead.py`). Data: Binance ETHUSDT spot 1s klines + USD-M
   perp aggTrades, same period and second grid as BTC. They enter the **drift model of every head
   and the direct GBM** only. Everything else - other features, hyper-parameters, cross-fitting,
   stacking, equal-weight blend, splits - is unchanged. The definitions are frozen: they are not
   modified after any validation or holdout result is seen.
9. **Validation arms** (research protocol: fit on train 2025, early stopping on valid, scored on
   valid; all five heads with stacking h10 -> h15 -> h30 -> h60 -> close inside each arm; both with
   `BTC_REGIME=1`, same code, data and seeds; touch / first-passage layers skipped in both arms,
   `BTC_EVENTS=0`, because the rule below is about P(S_target >= P0)):
   * A = `_r5`: the current regime-aware feature set (as WF-v5);
   * B = `_rx5`: A + the cross-asset features.
10. **Adoption rule (VALIDATION only)**, `python scripts/compare_versions.py valid _r5 _rx5`,
    blended forecast (`E_blend`), all rows, day-block bootstrap: the add-on is **adopted iff**
    (a) close head: dLL(B - A) < 0 with the 95% CI entirely below zero, **and**
    (b) h10, h15, h30, h60: dLL(B - A) <= 0 (point estimate; "not worse").
    Applied once. If rejected, the decision is documented, the holdout is NOT used for this
    question and nothing is deployed.
11. **If adopted**: final model "WF-x5" = WF-v5 recipe + cross-asset features, refit walk-forward
    through 2026-04-30 (early stopping on the last 30 days before that), scored once on the
    holdout (2026-05-02 .. 2026-11-02) and compared on identical rows with the current
    walk-forward model `_wf5` (`compare_versions.py test _wf5 _wfx5`). The `_wf5` reference is the
    packaged `models/wf5_20260430`; its holdout predictions are regenerated here and must
    reproduce the committed `artifacts/results_*_test_wf5.json` (check that the re-downloaded data
    are identical). The holdout result is reported as is and does not change rule 10; but if the
    close head is significantly WORSE on the holdout (CI entirely above zero) this is reported as a
    failed replication and the default deployment stays on the regime-only models.
    Online recalibration remains governed by rule 4.

### Decision under rule 10 (recorded after the validation comparison, before any holdout scoring of B)

`artifacts/compare_valid_r5_vs_rx5.txt` (validation, 119 days, E_blend, all rows, dLL = B - A):
h10 -0.00076 [-0.00086, -0.00067], h15 -0.00069 [-0.00078, -0.00060], h30 -0.00061 [-0.00073, -0.00051],
h60 -0.00035 [-0.00047, -0.00024], **close -0.00011 [-0.00022, -0.00001]** (full precision
-0.000113 [-0.000219, -0.000008]; 10,000 resamples [-0.000213, -0.000013]).
(a) holds (close CI entirely below zero) and (b) holds (every shorter head < 0): **ADOPTED**.
Noted, not part of the rule: the close-head gain is small, and in its final 30 s B is significantly
worse (+0.00032 [+0.00013, +0.00053]); the gain is concentrated early in the window (close at e = 0:
-0.00029 [-0.00052, -0.00006]; h10 at e = 0: -0.00337). Proceeding to rule 11 (WF-x5).
