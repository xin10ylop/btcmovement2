"""Cross-asset add-on features: what a second market (ETH by default) says about BTC.

Motivation (2025-only linear probe, xin10ylop/eth3 research/r2_btc_leads_eth.py): ETH adds a
lot to BTC's own short-horizon signals (BTC spot next-10s OOS R2 5.07% -> 5.91%) while BTC
adds almost nothing to ETH.  At 1 s, corr(BTC spot[t+1], ETH perp[t]) = 0.157: the
perpetuals lead both spots and BTC spot is the slowest of the four markets.

Features (exactly the probe's set, plus the beta it uses), all causal: functions of bars
<= c = T0 + e - 1 of BOTH timelines only (never the decision second itself or later):

  xe_pr_{L}   ETH perp log move over the last L s, in ETH vol units (v_E * sqrt(L)),
              L in 1, 2, 3, 5, 10, 30, 60, 300
  xe_gap_{L}  "BTC lagging ETH" gap: BTC spot move over L s minus beta_1h x ETH perp move
              over L s, in BTC vol units (v_B * sqrt(L)), same L
  xe_sr_{L}   ETH spot log move over L s in ETH vol units, L in 1, 5, 30
  xe_beta_1h  rolling 1-hour regression slope of BTC spot 1s returns on ETH spot 1s returns

v_X = sqrt(max(RV_300, RV_3600 / 4)) per second, floored at 2e-6 (same definition as the
primary feature engine's `v`).  Where the second market has a data gap (day missing, or no
spot / perp trade in the last 300 s) the features are NaN; LightGBM routes them explicitly.

The two timelines must share their index (both start on the same UTC day).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import Roll, ffill_nan

DAY = 86400
LAGS = (1, 2, 3, 5, 10, 30, 60, 300)
SPOT_LAGS = (1, 5, 30)
XASSET_FEATS = ([f"xe_pr_{L}" for L in LAGS] + [f"xe_gap_{L}" for L in LAGS]
                + [f"xe_sr_{L}" for L in SPOT_LAGS] + ["xe_beta_1h"])


def _present_sec(src, a, b):
    if hasattr(src, "present_sec"):
        p = np.asarray(src.present_sec[a:min(b, src.n)], dtype=bool)
    else:
        d0, d1 = a // DAY, (min(b, src.n) - 1) // DAY + 1
        p = np.repeat(np.asarray(src.present[d0:d1], dtype=bool), DAY)[a - d0 * DAY:min(b, src.n) - d0 * DAY]
    return np.concatenate([p, np.zeros(b - a - len(p), dtype=bool)])


def _get(src, k, a, b, fill):
    x = np.asarray(src[k][a:min(b, src.n)], dtype=np.float64)
    if len(x) < b - a:                     # past the end of the source: never read (c < b - a)
        x = np.concatenate([x, np.full(b - a - len(x), fill)])
    return x


def _vol(R2, c):
    v = np.sqrt(np.maximum(R2.sum(c, 300) / 300, 0.25 * R2.sum(c, 3600) / 3600))
    return np.maximum(v, 2e-6)


def xasset_features(tl, xtl, T0_abs: np.ndarray, e: np.ndarray, chunk_days: int = 31,
                    warm_days: int = 1) -> pd.DataFrame:
    """Cross-asset features for decision points (T0_abs, e) given in the shared timeline index.

    tl: primary (reference) market timeline; xtl: the second market's timeline.  Both are
    timeline-like sources (Timeline, predictor buffers): __getitem__(field), n, and
    `present` (per day) or `present_sec` (per second)."""
    assert getattr(tl, "t0_epoch", None) == getattr(xtl, "t0_epoch", None), "timelines must share their start"
    order = np.argsort(T0_abs, kind="stable")
    T0s, es = np.asarray(T0_abs, np.int64)[order], np.asarray(e, np.int64)[order]
    day = T0s // DAY
    parts = []
    for c0 in range(day.min(), day.max() + 1, chunk_days):
        sel = (day >= c0) & (day < c0 + chunk_days)
        if not sel.any():
            continue
        a = max(0, (c0 - warm_days) * DAY)
        b = min(tl.n, (c0 + chunk_days) * DAY + 600)
        lsB = np.log(_get(tl, "s_close", a, b, np.nan))
        lsE = np.log(_get(xtl, "s_close", a, b, np.nan))
        lpE = np.log(ffill_nan(_get(xtl, "p_close", a, b, np.nan)))
        bad = ~np.isfinite(lpE)
        lpE[bad] = lsE[bad]
        rsB = np.nan_to_num(np.diff(lsB, prepend=lsB[0]))
        rsE = np.nan_to_num(np.diff(lsE, prepend=lsE[0]))
        # data-quality of the second market: day present and trades on both its markets
        okE = _present_sec(xtl, a, b)
        sE = Roll((_get(xtl, "s_count", a, b, 0.0) > 0).astype(np.float64))
        pE = Roll((_get(xtl, "p_ntr", a, b, 0.0) > 0).astype(np.float64))
        R2B, R2E, RBE = Roll(rsB ** 2), Roll(rsE ** 2), Roll(rsB * rsE)
        c = (T0s[sel] + es[sel] - 1) - a          # local index of the last known bar
        F = {}
        vB, vE = _vol(R2B, c), _vol(R2E, c)
        beta = RBE.sum(c, 3600) / (R2E.sum(c, 3600) + 1e-18)
        for L in LAGS:
            cl = np.maximum(c - L, 0)
            dpE = lpE[c] - lpE[cl]
            dsB = lsB[c] - lsB[cl]
            F[f"xe_pr_{L}"] = dpE / (vE * np.sqrt(L))
            F[f"xe_gap_{L}"] = (dsB - beta * dpE) / (vB * np.sqrt(L))
        for L in SPOT_LAGS:
            F[f"xe_sr_{L}"] = (lsE[c] - lsE[np.maximum(c - L, 0)]) / (vE * np.sqrt(L))
        F["xe_beta_1h"] = beta
        good = (okE[c] & okE[np.maximum(c - 3600, 0)] & (sE.sum(c, 300) > 0) & (pE.sum(c, 300) > 0)
                & np.isfinite(lsE[c]) & np.isfinite(lsE[np.maximum(c - 3600, 0)]))
        df = pd.DataFrame({k: np.where(good, v, np.nan).astype(np.float32) for k, v in F.items()})
        df["_pos"] = np.flatnonzero(sel)
        parts.append(df)
    res = pd.concat(parts, ignore_index=True).sort_values("_pos")
    res.index = order[res["_pos"].values]
    return res.drop(columns="_pos").sort_index()[XASSET_FEATS]
