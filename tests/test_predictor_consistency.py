"""The live Predictor (raw buffer -> prediction) must reproduce the batch pipeline."""
import os
import pickle
import sys

import numpy as np

from btcmodel.dataset import load_season
from btcmodel.predictor import Predictor
from btcmodel.timeline import load_timeline, FIELDS


def run(split="valid", heads=("h30",), n=6, seed=0, tag="", model_split="valid", buffer_days=2):
    tl = load_timeline()
    from btcmodel.config import MODEL_DIR
    models = {h: pickle.load(open(os.path.join(MODEL_DIR, f"models_{h}_{model_split}{tag}.pkl"), "rb")) for h in heads}
    preds = {h: pickle.load(open(os.path.join(MODEL_DIR, f"preds_{h}_{split}{tag}.pkl"), "rb")) for h in heads}
    pr = Predictor(models, load_season())
    xtl = None
    if pr.needs_xasset:                                   # cross-asset models: second market buffer
        from btcmodel.config import XDATA_DIR
        xtl = load_timeline(XDATA_DIR)
        assert xtl.d0 == tl.d0
    rng = np.random.default_rng(seed)
    P = preds[heads[0]]
    worst = 0.0
    for k in rng.choice(len(P["y"]), n, replace=False):
        T0, e = int(P["T0"][k]), int(P["e"][k])
        start = (T0 - buffer_days * 86400) // 3600 * 3600   # hour-aligned buffer start
        end = T0 + e                                       # only what is known at decision time
        arrays = {f: np.asarray(tl[f][start:end]) for f in FIELDS}
        xarrays = None if xtl is None else {f: np.asarray(xtl[f][start:end]) for f in FIELDS}
        pr.ingest(arrays, tl.t0_epoch + start, xarrays=xarrays)
        out = pr.predict(tl.t0_epoch + T0 + e, p0=float(np.asarray(tl["s_open"][T0])))
        for h in heads:
            Ph = preds[h]
            j = np.flatnonzero((Ph["T0"] == T0) & (Ph["e"] == e))
            if len(j) == 0 or h not in out["heads"]:
                continue
            batch = Ph["p"]["E_blend"][j[0]] if "E_blend" in Ph["p"] else None
            if batch is not None:
                worst = max(worst, abs(batch - out["heads"][h]["p_above_ref_blend"]))
            live = out["heads"][h]["p_above_ref_blend"]
            bm, lm = Ph["p"]["M_final"][j[0]], out["heads"][h]["p_above_ref_structured"]
            print(f"{h} T0={T0} e={e}: batch structured={bm:.6f} live structured={lm:.6f}"
                  + (f" | batch blend={batch:.6f} live blend={live:.6f}" if batch is not None else ""))
            worst = max(worst, abs(bm - lm))
    print("max |batch - live| =", worst)
    assert worst < 1e-4


def run_fallback(split="test", heads=("h30",), tag="_wfx5", fb_tag="_wf5", n=3, seed=1, buffer_days=9):
    """Cross-asset models with a regime-only fallback: with the second market cut short (feed
    down) the answer must come from the fallback and reproduce ITS batch predictions; with the
    second market present it must come from the cross-asset models."""
    tl = load_timeline()
    from btcmodel.config import MODEL_DIR, XDATA_DIR
    xtl = load_timeline(XDATA_DIR)
    ld = lambda t: {h: pickle.load(open(os.path.join(MODEL_DIR, f"models_{h}_valid{t}.pkl"), "rb")) for h in heads}
    fb = Predictor(ld(fb_tag), load_season())
    pr = Predictor(ld(tag), load_season(), fallback=fb)
    assert pr.needs_xasset and not fb.needs_xasset
    preds = {h: pickle.load(open(os.path.join(MODEL_DIR, f"preds_{h}_{split}{fb_tag}.pkl"), "rb")) for h in heads}
    rng = np.random.default_rng(seed)
    P = preds[heads[0]]
    worst = 0.0
    for k in rng.choice(len(P["y"]), n, replace=False):
        T0, e = int(P["T0"][k]), int(P["e"][k])
        start = (T0 - buffer_days * 86400) // 3600 * 3600
        arrays = {f: np.asarray(tl[f][start:T0 + e]) for f in FIELDS}
        stale = {f: np.asarray(xtl[f][start:T0 + e - 120]) for f in FIELDS}   # ETH feed 2 min behind
        pr.ingest(arrays, tl.t0_epoch + start, xarrays=stale)
        out = pr.predict(tl.t0_epoch + T0 + e, p0=float(np.asarray(tl["s_open"][T0])))
        assert out["model"].startswith("fallback"), out["model"]
        for h in heads:
            j = np.flatnonzero((preds[h]["T0"] == T0) & (preds[h]["e"] == e))
            if len(j) and h in out["heads"]:
                worst = max(worst, abs(preds[h]["p"]["E_blend"][j[0]] - out["heads"][h]["p_above_ref_blend"]))
        pr.ingest(arrays, tl.t0_epoch + start, xarrays={f: np.asarray(xtl[f][start:T0 + e]) for f in FIELDS})
        assert pr.predict(tl.t0_epoch + T0 + e, p0=float(np.asarray(tl["s_open"][T0])))["model"] == "cross-asset"
        print(f"fallback ok T0={T0} e={e}")
    print("fallback: max |batch(fallback) - live| =", worst)
    assert worst < 1e-4


if __name__ == "__main__":
    # python -m tests.test_predictor_consistency [tag] [split] [heads...] [--fallback <tag>]
    # e.g. python -m tests.test_predictor_consistency _wf5 test h10 h15 h30 h60 close
    #      python -m tests.test_predictor_consistency _wfx5 test h10 h15 h30 h60 close --fallback _wf5
    args = sys.argv[1:]
    fb = None
    if "--fallback" in args:
        i = args.index("--fallback")
        fb = args[i + 1]
        args = args[:i] + args[i + 2:]
    tag = args[0] if len(args) > 0 else ""
    split = args[1] if len(args) > 1 else "valid"
    heads = tuple(args[2:]) or ("h30",)
    run(split=split, heads=heads, tag=tag, buffer_days=9 if tag else 2)
    if fb is not None:
        run_fallback(split=split, heads=heads, tag=tag, fb_tag=fb)
