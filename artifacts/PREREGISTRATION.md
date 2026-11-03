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
