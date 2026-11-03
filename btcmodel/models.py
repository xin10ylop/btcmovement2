"""Reference-relative predictive distribution models.

Architecture (one instance per head: 30s, 60s, window close)
------------------------------------------------------------
Let x = log(S_now / P0) (known), R = log(S_target / S_now) the future increment.
Then log(S_target / P0) = x + R, so EVERY reference-relative question is a question
about the conditional law of R, read at a reference-dependent point:

    P(S_target >= P0)        = P(R >= -x)
    P(S_target >= P0(1+k))   = P(R >= log(1+k) - x)
    quantiles of S_target/P0 = x + quantiles of R
    touch / first-passage    = law of the running max/min of the increment path

The fixed reference therefore never contaminates the model of market dynamics -
it only decides where the distribution is read.  The law of R is modelled as

    R = sigma_hat * (mu_hat + eps)

  * sigma_hat : scale model (expected |R|), seasonality-aware term structure
  * mu_hat    : standardized drift model (efficient-price gap, flow, lead-lag)
  * eps       : empirical, fat-tailed residual law (bucketed by horizon length)

and then a gradient-boosted *calibration-from-prior* layer (LightGBM init_score =
structural logit) corrects what the structure misses.  It can only move away from
the structural prior if that improves out-of-sample loss (early stopping).

Every stage is trained on out-of-fold (cross-fitted, time-blocked, purged)
predictions of the previous stage so that no stage learns from in-sample-optimistic
inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import logit, expit

from .prob import p_touch, p_double_barrier

SQ2PI = np.sqrt(2 / np.pi)

VOL_FEATS = [
    "lrv_10", "lrv_30", "lrv_60", "lrv_120", "lrv_300", "lrv_900", "lrv_3600", "lrv_14400", "lrv_86400",
    "lrvsa_60", "lrvsa_300", "lrvsa_3600", "lrvp_30", "lrvp_300", "lpk_60", "lpk_300", "lpk_3600",
    "jump_300", "jump_3600", "lcnt_10", "lcnt_60", "lcnt_300", "lcnt_3600", "lvol_10", "lvol_60",
    "lvol_300", "lvol_3600", "lcnt_ratio", "lvol_ratio", "seas_now", "maxq_10",
    "e", "how", "hod_et", "dow_et", "moh", "som", "w_in_hour",
    "abs_x_v", "wrange_v", "ncross",
]
DIR_FEATS = [
    "r_1", "r_2", "r_3", "r_5", "r_10", "r_20", "r_30", "r_60", "r_120", "r_300", "r_900", "r_3600",
    "pls_1", "pls_2", "pls_5", "pls_10", "pls_30", "pls_60", "pls_300",
    "bdev_10", "bdev_60", "bdev_300", "bdev_1800",
    "simb_1", "pimb_1", "simb_5", "pimb_5", "simb_10", "pimb_10", "simb_30", "pimb_30", "simb_60",
    "pimb_60", "simb_300", "pimb_300", "simb_900", "pimb_900", "netflow_10", "netflow_60", "netflow_300",
    "rngpos_60", "rngpos_300", "e", "x_v", "wmax_v", "wmin_v", "ncross", "fup", "tsc",
    "prev_ret_v", "prev2_ret_v", "mtf_15m", "mtf_1h", "mtf_4h", "mtf_1d", "round_1000", "round_100",
    "lrvsa_300", "lcnt_ratio", "lvol_ratio", "jump_300", "hod_et", "som",
]

LGB_BASE = dict(
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000, feature_fraction=0.7,
    bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, max_bin=127, verbose=-1, num_threads=4,
)


def add_derived(df: pd.DataFrame, head: str) -> pd.DataFrame:
    """Head-specific derived columns (all causal)."""
    H = df[f"{head}__H"].values.astype(np.float64)
    v = df["v"].values.astype(np.float64)
    out = {}
    out["lH"] = np.log(H)
    # naive scale: per-second trailing vol * sqrt(H) * E|N(0,1)|
    out["naive_scale"] = v * np.sqrt(H) * SQ2PI
    sint = df[f"{head}__sint"].values.astype(np.float64)
    # clock projection: seasonally-adjusted trailing variance x expected seasonal variance ahead
    out["lproj_300"] = df["lrvsa_300"].values + np.log(sint) - np.log(v ** 2)
    out["lproj_3600"] = df["lrvsa_3600"].values + np.log(sint) - np.log(v ** 2)
    out["lsint"] = np.log(sint)
    out["abs_x_v"] = np.abs(df["x_v"].values)
    out["wrange_v"] = df["wmax_v"].values - df["wmin_v"].values
    # distance to reference in units of the naive horizon scale
    out["z_naive"] = df["xcur"].values / (v * np.sqrt(H))
    for k, a in out.items():
        df[k] = a.astype(np.float32) if k not in ("naive_scale",) else a
    return df


def top_gain(m, names, k=8):
    g = m.feature_importance("gain")
    g = g / max(g.sum(), 1e-12)
    o = np.argsort(-g)[:k]
    return ", ".join(f"{names[i]}={g[i]:.2f}" for i in o)


def time_folds(day: np.ndarray, k: int = 4, purge_days: int = 1):
    """Contiguous time blocks with a purge gap; yields (train_idx, pred_idx)."""
    ud = np.unique(day)
    blocks = np.array_split(ud, k)
    for b in blocks:
        lo, hi = b.min(), b.max()
        pred = (day >= lo) & (day <= hi)
        train = (day < lo - purge_days) | (day > hi + purge_days)
        yield np.flatnonzero(train), np.flatnonzero(pred)


def _fit(params, Xtr, ytr, Xva=None, yva=None, rounds=3000, init_tr=None, init_va=None,
         weight=None, es=100, fixed_rounds=None, monotone=None):
    p = dict(LGB_BASE)
    p.update(params)
    if monotone is not None:
        p["monotone_constraints"] = monotone
        p["monotone_constraints_method"] = "advanced"
    dtr = lgb.Dataset(Xtr, ytr, init_score=init_tr, weight=weight, free_raw_data=True)
    if fixed_rounds is not None:
        return lgb.train(p, dtr, num_boost_round=max(fixed_rounds, 1))
    dva = lgb.Dataset(Xva, yva, init_score=init_va, reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(es, verbose=False)])
    return m


class EmpiricalDist:
    """Bucketed empirical law of standardized residuals (quantile table)."""

    LEVELS = np.linspace(0, 1, 4001)

    def __init__(self, edges: np.ndarray):
        self.edges = edges

    def bucket(self, H):
        return np.clip(np.searchsorted(self.edges, H, side="right") - 1, 0, len(self.edges) - 2)

    def fit(self, eps, H):
        b = self.bucket(H)
        self.q = np.stack([np.quantile(eps[b == j], self.LEVELS) for j in range(len(self.edges) - 1)])
        self.sorted = [np.sort(eps[b == j]) for j in range(len(self.edges) - 1)]
        return self

    def sf(self, z, H):
        """P(eps >= z) (inclusive, so that ties count as 'up')."""
        b = self.bucket(H)
        out = np.empty(len(z))
        for j in range(len(self.edges) - 1):
            m = b == j
            if m.any():
                s = self.sorted[j]
                out[m] = 1.0 - np.searchsorted(s, z[m], side="left") / len(s)
        return out

    def ppf(self, a, H):
        b = self.bucket(H)
        return np.array([np.interp(a, self.LEVELS, self.q[j]) for j in b]) if np.ndim(a) == 0 else None

    def ppf_many(self, alphas, H):
        b = self.bucket(H)
        out = np.empty((len(H), len(alphas)))
        for j in range(len(self.edges) - 1):
            m = b == j
            out[m] = np.interp(alphas, self.LEVELS, self.q[j])[None, :]
        return out


class HeadModel:
    """Scale + drift + residual law + calibrated event layers for one horizon head."""

    def __init__(self, head: str, extra_dir_feats: list[str] | None = None, seed: int = 0):
        self.head = head
        self.extra = extra_dir_feats or []
        self.seed = seed
        edges = np.array([1, 11, 31, 61, 121, 201, 301]) if head == "close" else np.array([0, 10 ** 6])
        self.dist = EmpiricalDist(edges)

    # ------------------------------------------------------------------ helpers
    def _vf(self):
        return VOL_FEATS + ["lH", "lproj_300", "lproj_3600", "lsint"]

    def _df(self):
        return DIR_FEATS + ["lH", "z_naive"] + self.extra

    def _scale_target(self, df):
        return np.clip(np.abs(df[f"{self.head}__inc"].values) / df["naive_scale"].values, 0, 25)

    # ------------------------------------------------------------------ fitting
    def fit(self, tr: pd.DataFrame, va: pd.DataFrame, log=print):
        h = self.head
        P = {"seed": self.seed}
        # ---- scale (multiplicative correction of the naive scale) ----
        ys_tr, ys_va = self._scale_target(tr), self._scale_target(va)
        m = _fit(P, tr[self._vf()], ys_tr, va[self._vf()], ys_va)
        self.scale_rounds = m.best_iteration
        self.scale_model = m
        sc_va = m.predict(va[self._vf()], num_iteration=m.best_iteration)
        sc_oof = np.empty(len(tr))
        for itr, ipr in time_folds(tr["day"].values):
            mm = _fit(P, tr[self._vf()].iloc[itr], ys_tr[itr], fixed_rounds=self.scale_rounds)
            sc_oof[ipr] = mm.predict(tr[self._vf()].iloc[ipr])
        tr["sig"] = np.maximum(sc_oof, 0.05) * tr["naive_scale"].values
        va["sig"] = np.maximum(sc_va, 0.05) * va["naive_scale"].values
        log(f"[{h}] scale: {self.scale_rounds} rounds; valid MSE ratio: const={np.mean((ys_va - ys_tr.mean()) ** 2):.4f} "
            f"model={np.mean((ys_va - sc_va) ** 2):.4f}")
        log(f"[{h}] scale top features (gain share): {top_gain(m, self._vf())}")

        # ---- drift (standardized) ----
        u_tr = np.clip(tr[f"{h}__inc"].values / tr["sig"].values, -8, 8)
        u_va = np.clip(va[f"{h}__inc"].values / va["sig"].values, -8, 8)
        for d in (tr, va):
            d["lsig_rel"] = np.log(d["sig"].values / d["naive_scale"].values).astype(np.float32)
        dfeat = self._df() + ["lsig_rel"]
        m = _fit(dict(P, learning_rate=0.05, num_leaves=31, min_data_in_leaf=5000), tr[dfeat], u_tr,
                 va[dfeat], u_va)
        self.drift_rounds = max(m.best_iteration, 1)
        self.drift_model = m
        mu_va = m.predict(va[dfeat], num_iteration=self.drift_rounds)
        mu_oof = np.empty(len(tr))
        for itr, ipr in time_folds(tr["day"].values):
            mm = _fit(dict(P, learning_rate=0.05, num_leaves=31, min_data_in_leaf=5000),
                      tr[dfeat].iloc[itr], u_tr[itr], fixed_rounds=self.drift_rounds)
            mu_oof[ipr] = mm.predict(tr[dfeat].iloc[ipr])
        tr["mu"], va["mu"] = mu_oof, mu_va
        r2 = 1 - np.mean((u_va - mu_va) ** 2) / np.mean((u_va - u_va.mean()) ** 2)
        log(f"[{h}] drift: {self.drift_rounds} rounds; valid R2(u)={r2 * 100:.3f}%  corr={np.corrcoef(mu_va, u_va)[0, 1]:.4f}")
        log(f"[{h}] drift top features (gain share): {top_gain(m, dfeat)}")
        self.dfeat = dfeat

        # ---- residual law from OOF residuals ----
        eps = np.clip(tr[f"{h}__inc"].values / tr["sig"].values - tr["mu"].values, -50, 50)
        self.dist.fit(eps, tr[f"{h}__H"].values)

        # ---- P(up) calibration from structural prior ----
        for d in (tr, va):
            d["p_struct"] = np.clip(self.struct_up(d), 1e-5, 1 - 1e-5)
            d["z_struct"] = (-d["xcur"].values / d["sig"].values - d["mu"].values).astype(np.float32)
        cf = self.calib_feats()
        y_tr, y_va = (tr[f"{h}__y"].values >= 0).astype(int), (va[f"{h}__y"].values >= 0).astype(int)
        m = _fit(dict(P, objective="binary", learning_rate=0.04, num_leaves=15, min_data_in_leaf=20000,
                      lambda_l2=50.0), tr[cf], y_tr, va[cf], y_va,
                 init_tr=logit(tr["p_struct"].values), init_va=logit(va["p_struct"].values), es=200)
        self.calib_up = m
        self.calib_rounds = m.best_iteration
        log(f"[{h}] up-calibration: {self.calib_rounds} rounds; top features: {top_gain(m, cf)}")
        return self

    @staticmethod
    def calib_feats():
        return ["z_struct", "mu", "lsig_rel", "e", "lH", "lrvsa_300", "lcnt_ratio", "jump_300",
                "som", "hod_et", "x_v", "ncross"]

    # ------------------------------------------------------------------ predicting
    def prepare(self, d: pd.DataFrame) -> pd.DataFrame:
        sc = self.scale_model.predict(d[self._vf()], num_iteration=self.scale_rounds)
        d["sig"] = np.maximum(sc, 0.05) * d["naive_scale"].values
        d["lsig_rel"] = np.log(d["sig"].values / d["naive_scale"].values).astype(np.float32)
        d["mu"] = self.drift_model.predict(d[self.dfeat], num_iteration=self.drift_rounds)
        d["p_struct"] = np.clip(self.struct_up(d), 1e-5, 1 - 1e-5)
        d["z_struct"] = (-d["xcur"].values / d["sig"].values - d["mu"].values).astype(np.float32)
        return d

    def struct_up(self, d):
        z = -d["xcur"].values / d["sig"].values - d["mu"].values
        return self.dist.sf(z, d[f"{self.head}__H"].values)

    def struct_terminal_sf(self, d, level_log):
        """P(log(S_T/P0) >= level) from the structural law (level array or scalar)."""
        z = (level_log - d["xcur"].values) / d["sig"].values - d["mu"].values
        return self.dist.sf(np.asarray(z, float), d[f"{self.head}__H"].values)

    def p_up(self, d):
        if self.calib_rounds <= 0:
            return d["p_struct"].values
        return expit(logit(d["p_struct"].values) +
                     self.calib_up.predict(d[self.calib_feats()], num_iteration=self.calib_rounds,
                                           raw_score=True))

    def quantiles(self, d, alphas):
        """Quantiles of log(S_target/P0)."""
        q = self.dist.ppf_many(np.asarray(alphas), d[f"{self.head}__H"].values)
        return d["xcur"].values[:, None] + d["sig"].values[:, None] * (d["mu"].values[:, None] + q)

    # ---- BM-equivalent parameters for path events ----
    def bm_params(self, d):
        H = d[f"{self.head}__H"].values.astype(float)
        s = d["sig"].values / SQ2PI / np.sqrt(H)          # per-second diffusion
        mu = d["mu"].values * d["sig"].values / H          # per-second drift
        return mu, s, H
