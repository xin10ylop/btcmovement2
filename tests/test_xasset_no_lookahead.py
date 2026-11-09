"""Cross-asset features must not change when the future of EITHER market is replaced by
garbage, and a live-style buffer that ends at the decision second must give the same values.

    python -m tests.test_xasset_no_lookahead
"""
import numpy as np

from btcmodel.config import XDATA_DIR
from btcmodel.timeline import load_timeline
from btcmodel.xasset import xasset_features, XASSET_FEATS

FIELDS_B = ["s_close"]
FIELDS_E = ["s_close", "p_close", "s_count", "p_ntr"]


class Slice:
    """Local slice [lo, hi) of a timeline; bars >= cut (local) optionally replaced by garbage."""

    def __init__(self, tl, fields, lo, hi, cut=None, rng=None, end=None):
        self.t0_epoch = tl.t0_epoch + lo                  # local index 0 = absolute lo
        stop = hi if end is None else end                 # live buffer: nothing after `end`
        self.n = stop - lo
        self.present_sec = np.repeat(tl.present, 86400)[lo:stop]
        self.d = {}
        for k in fields:
            x = np.array(tl[k][lo:stop])
            if cut is not None:
                if x.dtype.kind == "f":
                    x[cut:] = x[cut:] * np.exp(rng.normal(0, 0.05, len(x) - cut))
                else:
                    x[cut:] = rng.integers(0, 1000, len(x) - cut)
            self.d[k] = x

    def __getitem__(self, k):
        return self.d[k]


def test():
    tl, xtl = load_timeline(), load_timeline(XDATA_DIR)
    assert tl.d0 == xtl.d0
    rng = np.random.default_rng(7)
    for trial in range(8):
        T0 = int((rng.integers(3 * 86400, min(tl.n, xtl.n) - 2 * 86400) // 300) * 300)
        e = int(rng.choice([0, 1, 2, 5, 150, 299]))
        a = xasset_features(tl, xtl, np.array([T0]), np.array([e]))
        assert np.isfinite(a.values).all(), (T0, e, a.to_dict())
        lo, hi = (T0 - 2 * 86400) // 86400 * 86400, T0 + 2 * 86400
        cut = T0 + e - lo
        # 1) garbage future in both markets (decision second itself included)
        b = xasset_features(Slice(tl, FIELDS_B, lo, hi, cut, rng), Slice(xtl, FIELDS_E, lo, hi, cut, rng),
                            np.array([T0 - lo]), np.array([e]))
        # 2) live-style buffers that stop at the decision second (primary padded to the window end)
        live_b = Slice(tl, FIELDS_B, lo, hi, end=T0 + 300)
        live_x = Slice(xtl, FIELDS_E, lo, hi, end=T0 + e)
        c = xasset_features(live_b, live_x, np.array([T0 - lo]), np.array([e]))
        for name, other in (("garbage future", b), ("live buffer", c)):
            rel = np.abs(a.values - other.values) / (np.abs(a.values) + 1e-9)
            assert np.nanmax(rel) < 1e-5, (name, T0, e, dict(zip(XASSET_FEATS, rel[0])))
        # 3) sanity: the decision second's own bar DOES matter for the next decision second
        if e < 299:
            d = xasset_features(Slice(tl, FIELDS_B, lo, hi, cut + 1, rng), Slice(xtl, FIELDS_E, lo, hi, cut, rng),
                                np.array([T0 - lo]), np.array([e + 1]))
            n = xasset_features(tl, xtl, np.array([T0]), np.array([e + 1]))
            assert np.nanmax(np.abs(d.values - n.values)) > 0, "bar T0+e should be known at e+1"
        print(f"ok T0={T0} e={e}  xe_pr_1={a['xe_pr_1'].iloc[0]:+.3f} xe_gap_10={a['xe_gap_10'].iloc[0]:+.3f} "
              f"beta={a['xe_beta_1h'].iloc[0]:.3f}")


if __name__ == "__main__":
    test()
