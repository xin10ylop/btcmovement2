#!/bin/bash
# Cross-asset (ETH) add-on for the BTC model, end to end (PREREGISTRATION amendment 2).
#   bash scripts/run_xasset_pipeline.sh [first_stage] [last_stage]
# Assumes the raw downloads exist:
#   python -m btcmodel.data 2025-01-01 2026-11-02 6
#   BTC_ASSET=ETH python -m btcmodel.data 2025-01-01 2026-11-02 6
# Stages:
#   timelines  BTC and ETH 1s timelines (identical second grid)
#   caches     BTC features/targets, 10s/15s targets, regime and cross-asset caches
#   tests      lookahead tests (base features, regime, cross-asset)
#   wf5ref     holdout predictions of the CURRENT walk-forward model (models/wf5_20260430),
#              reports to $D/wf5_repro (must reproduce artifacts/results_*_test_wf5.json)
#   validA     research protocol, arm A = regime-aware (_r5)       } adoption rule 10
#   validB     research protocol, arm B = + cross-asset (_rx5)     } (validation only)
#   decide     paired comparison on validation
#   --- only if adopted ---
#   wfx        walk-forward refit through 2026-04-30 with cross-asset features (_wfx5)
#   wfx_eval   score WF-x5 once on the holdout (and on valid), paired comparison vs _wf5
#   package    models/wfx5_20260430
#   prod       deployment refit through 2026-11-02 + package models/prodx_20261102
# Every head trains in its own process (memory). Progress: $D/xasset_pipeline.log
set -euo pipefail
cd "$(dirname "$0")/.."
export BTC_ASSET=BTC
D=$(python -c "from btcmodel.config import DATA_DIR; print(DATA_DIR)")
LOG=$D/xasset_pipeline.log
HEADS="h10 h15 h30 h60 close"
STAGES="timelines caches tests wf5ref validA validB decide wfx wfx_eval package prod"
START=${1:-timelines}
STOP=${2:-decide}
run=0
for S in $STAGES; do
  [ "$S" = "$START" ] && run=1
  [ $run = 1 ] || continue
  echo "STAGE $S start $(date +%H:%M)" >> "$LOG"
  case $S in
    timelines) python -m btcmodel.timeline >> "$LOG" 2>&1
               BTC_ASSET=ETH python -m btcmodel.timeline >> "$LOG" 2>&1 ;;
    caches)    python -m btcmodel.dataset > "$D/dataset.log" 2>&1
               python scripts/build_targets_extra.py >> "$LOG" 2>&1
               python scripts/build_regime.py > "$D/regime_build.log" 2>&1
               python scripts/build_xasset.py > "$D/xasset_build.log" 2>&1 ;;
    tests)     python -m tests.test_xasset_no_lookahead >> "$LOG" 2>&1
               python -m tests.test_regime_no_lookahead >> "$LOG" 2>&1 ;;
    wf5ref)    python - >> "$LOG" 2>&1 <<'EOF'
import gzip, os, pickle
from btcmodel.config import MODEL_DIR
for h in ["h10", "h15", "h30", "h60", "close"]:
    M = pickle.load(gzip.open(f"models/wf5_20260430/{h}.pkl.gz", "rb"))
    pickle.dump(M, open(os.path.join(MODEL_DIR, f"models_{h}_valid_wf5.pkl"), "wb"))
print("unpacked models/wf5_20260430 -> models_<head>_valid_wf5.pkl")
EOF
               # reports go to $D/wf5_repro (the committed artifacts/*_test_wf5.* come from the
               # uncompacted models); the paired comparison only needs the predictions
               for h in $HEADS; do
                 [ -f "$D/models/preds_${h}_test_wf5.pkl" ] && [ -f "$D/stack/${h}_test_test_wf5.parquet" ] && continue
                 BTC_ART_DIR=$D/wf5_repro BTC_REGIME=1 BTC_TAG=_wf5 python scripts/evaluate_saved.py test $h > "$D/test_wf5_$h.log" 2>&1
               done ;;
    validA)    for h in $HEADS; do
                 BTC_REGIME=1 BTC_EVENTS=0 BTC_TAG=_r5 python scripts/train_eval.py valid $h > "$D/tv_r5_$h.log" 2>&1
               done ;;
    validB)    for h in $HEADS; do
                 BTC_REGIME=1 BTC_XASSET=1 BTC_EVENTS=0 BTC_TAG=_rx5 python scripts/train_eval.py valid $h > "$D/tv_rx5_$h.log" 2>&1
               done ;;
    decide)    python scripts/compare_versions.py valid _r5 _rx5 >> "$LOG" 2>&1 ;;
    wfx)       for h in $HEADS; do
                 BTC_REGIME=1 BTC_XASSET=1 python scripts/train_production.py $h --end 2026-04-30 --tag _wfx5 --resume >> "$D/wfx5_train.log" 2>&1
               done ;;
    wfx_eval)  for s in test valid; do for h in $HEADS; do
                 BTC_REGIME=1 BTC_XASSET=1 BTC_TAG=_wfx5 python scripts/evaluate_saved.py $s $h > "$D/${s}_wfx5_$h.log" 2>&1
               done; done
               python scripts/compare_versions.py test _wf5 _wfx5 >> "$LOG" 2>&1 ;;
    package)   python scripts/package_models.py valid_wfx5 models/wfx5_20260430 >> "$LOG" 2>&1 ;;
    prod)      for h in $HEADS; do
                 BTC_REGIME=1 BTC_XASSET=1 python scripts/train_production.py $h --old-thin 2 --resume >> "$D/prodx_train.log" 2>&1
               done
               python scripts/package_models.py prod models/prodx_20261102 >> "$LOG" 2>&1 ;;
  esac
  echo "STAGE $S done $(date +%H:%M)" >> "$LOG"
  [ "$S" = "$STOP" ] && break
done
echo "PIPELINE STOPPED after $S $(date +%H:%M)" >> "$LOG"
