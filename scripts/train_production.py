"""Refit every head for deployment on ALL available data.

The research protocol (train 2025 / valid Jan-Apr 2026 / test May-Nov 2026) is what the
reported numbers come from.  For live use, more recent data matters (regimes drift), so
this script refits with the same architecture and hyper-parameters on everything except
the most recent 30 days, which serve only as the early-stopping set.

    python scripts/train_production.py            # all heads, in order (stacking!)
Outputs /home/user/data/models/prod_<head>.pkl  (load with btcmodel.predictor.Predictor).
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_eval import load_head, MODEL_DIR, STACK_DIR, HEAD_ORDER  # noqa: E402
from btcmodel.models import HeadModel, _fit, VOL_FEATS, DIR_FEATS  # noqa: E402
from btcmodel.events import touch_rows, TouchModel, fp_rows, FirstPassageModel  # noqa: E402

ES_DAYS = 30


def main(heads):
    for head in heads:
        d = pd.concat([load_head(s, head) for s in ("train", "valid", "test")], ignore_index=True)
        last = d["day"].max()
        es = d["day"].values > last - ES_DAYS
        gap = (d["day"].values > last - ES_DAYS - 1) & ~es          # 1-day embargo
        tr, va = d[~es & ~gap].reset_index(drop=True), d[es].reset_index(drop=True)
        del d
        extra = []
        for sh in [h for h in HEAD_ORDER if HEAD_ORDER.index(h) < HEAD_ORDER.index(head)]:
            for x, key in ((tr, "tr"), (va, "va")):
                s = pd.read_parquet(os.path.join(STACK_DIR, f"prod_{sh}_{key}.parquet"))
                x[f"mu_{sh}_px"] = pd.merge(x[["T0", "e"]], s, on=["T0", "e"], how="left")["drift_px"].values
                x[f"mu_{sh}_px"] = (x[f"mu_{sh}_px"] / x["naive_scale"]).astype(np.float32)
            extra.append(f"mu_{sh}_px")
        hm = HeadModel(head, extra_dir_feats=extra).fit(tr, va)
        for x, key in ((tr, "tr"), (va, "va")):
            pd.DataFrame({"T0": x["T0"].values, "e": x["e"].values,
                          "drift_px": x["mu"].values * x["sig"].values}).to_parquet(
                os.path.join(STACK_DIR, f"prod_{head}_{key}.parquet"))
        feats = sorted(set(VOL_FEATS + DIR_FEATS + ["lH", "z_naive", "lproj_300", "lsint"]))
        ytr, yva = (tr[f"{head}__y"].values >= 0).astype(int), (va[f"{head}__y"].values >= 0).astype(int)
        dm = _fit(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000),
                  tr[feats], ytr, va[feats], yva, es=100)
        sub = lambda x, f: x.iloc[::f].reset_index(drop=True)
        tm = TouchModel().fit(touch_rows(sub(tr, 4), head, hm), touch_rows(sub(va, 2), head, hm))
        fm = FirstPassageModel().fit(fp_rows(sub(tr, 4), head, hm), fp_rows(sub(va, 2), head, hm))
        with open(os.path.join(MODEL_DIR, f"prod_{head}.pkl"), "wb") as fh:
            pickle.dump({"head": hm, "touch": tm, "fp": fm, "direct": dm}, fh)
        print(f"saved prod_{head}.pkl", flush=True)
        del tr, va


if __name__ == "__main__":
    main(sys.argv[1:] or HEAD_ORDER)
