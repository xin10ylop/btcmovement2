"""The live Predictor (raw buffer -> prediction) must reproduce the batch pipeline."""
import pickle
import sys

import numpy as np

from btcmodel.dataset import load_season
from btcmodel.predictor import Predictor
from btcmodel.timeline import load_timeline, FIELDS


def run(split="valid", heads=("h30",), n=6, seed=0):
    tl = load_timeline()
    models = {h: pickle.load(open(f"/home/user/data/models/models_{h}_{split}.pkl", "rb")) for h in heads}
    preds = {h: pickle.load(open(f"/home/user/data/models/preds_{h}_{split}.pkl", "rb")) for h in heads}
    pr = Predictor(models, load_season())
    rng = np.random.default_rng(seed)
    P = preds[heads[0]]
    worst = 0.0
    for k in rng.choice(len(P["y"]), n, replace=False):
        T0, e = int(P["T0"][k]), int(P["e"][k])
        start = (T0 - 2 * 86400) // 3600 * 3600          # hour-aligned buffer start, 2 days back
        end = T0 + e                                       # only what is known at decision time
        arrays = {f: np.asarray(tl[f][start:end]) for f in FIELDS}
        pr.ingest(arrays, tl.t0_epoch + start)
        out = pr.predict(tl.t0_epoch + T0 + e, p0=float(np.asarray(tl["s_open"][T0])))
        for h in heads:
            Ph = preds[h]
            j = np.flatnonzero((Ph["T0"] == T0) & (Ph["e"] == e))
            if len(j) == 0 or h not in out["heads"]:
                continue
            batch = Ph["p"]["E_blend"][j[0]] if "E_blend" in Ph["p"] else None
            live = out["heads"][h]["p_above_ref"]
            bm, lm = Ph["p"]["M_final"][j[0]], out["heads"][h]["p_above_ref_structured"]
            print(f"{h} T0={T0} e={e}: batch structured={bm:.6f} live structured={lm:.6f}"
                  + (f" | batch blend={batch:.6f} live blend={live:.6f}" if batch is not None else ""))
            worst = max(worst, abs(bm - lm))
    print("max |batch - live| =", worst)
    assert worst < 1e-4


if __name__ == "__main__":
    run(heads=tuple(sys.argv[1:]) or ("h30",))
