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
