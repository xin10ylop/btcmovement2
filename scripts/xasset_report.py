"""Holdout report for the cross-asset add-on: WF-v5 (A) vs WF-x5 (B) on identical rows.

    python scripts/xasset_report.py [split=test] [A=_wf5] [B=_wfx5]

Per head: log-loss skill vs the naive Gaussian, paired dLL with day-block CI, ECE, directional
AUC and drift R2 for both versions; then dLL by elapsed-time bucket and by calendar month.
Writes artifacts/xasset_<split><A>_vs<B>.md."""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btcmodel.config import MODEL_DIR as MD, ART_DIR  # noqa: E402
from btcmodel.prob import logloss, block_bootstrap_diff, ece, auc  # noqa: E402

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
A = sys.argv[2] if len(sys.argv) > 2 else "_wf5"
B = sys.argv[3] if len(sys.argv) > 3 else "_wfx5"
HEADS = ["h10", "h15", "h30", "h60", "close"]


def r2(P):
    u = np.clip(P["inc"] / P["sig"], -8, 8)
    return 1 - np.mean((u - P["mu"]) ** 2) / np.mean((u - u.mean()) ** 2)


def main():
    out = [f"# Cross-asset add-on on {SPLIT}: A = `{A}` vs B = `{B}` (negative dLL = B better)\n"]
    main_rows, e_rows, m_rows = [], [], []
    for h in HEADS:
        fa, fb = os.path.join(MD, f"preds_{h}_{SPLIT}{A}.pkl"), os.path.join(MD, f"preds_{h}_{SPLIT}{B}.pkl")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        Pa, Pb = pickle.load(open(fa, "rb")), pickle.load(open(fb, "rb"))
        assert np.array_equal(Pa["T0"], Pb["T0"]) and np.array_equal(Pa["e"], Pb["e"])
        y, day, e = Pb["y"], Pb["day"], Pb["e"]
        pa, pb, p0 = Pa["p"]["E_blend"], Pb["p"]["E_blend"], Pb["p"]["B0_naive"]
        la, lb, l0 = logloss(y, pa), logloss(y, pb), logloss(y, p0)
        d, lo, hi, _ = block_bootstrap_diff(lb, la, day, n_boot=1000)
        nz = Pb["inc"] != 0
        up = (Pb["inc"][nz] > 0).astype(int)
        main_rows.append({"head": h, "naive LL": l0.mean(), "A LL": la.mean(), "B LL": lb.mean(),
                          "A skill %": 100 * (1 - la.mean() / l0.mean()), "B skill %": 100 * (1 - lb.mean() / l0.mean()),
                          "dLL B-A": d, "CI lo": lo, "CI hi": hi, "ECE A": ece(y, pa), "ECE B": ece(y, pb),
                          "AUC A": auc(up, Pa["mu"][nz]), "AUC B": auc(up, Pb["mu"][nz]),
                          "R2 A %": 100 * r2(Pa), "R2 B %": 100 * r2(Pb)})
        eb = pd.cut(e, [-1, 0, 60, 120, 180, 240, 270, 299])
        for b in eb.categories:
            m = np.asarray(eb == b)
            if m.sum() < 1000:
                continue
            dd, l_, h_, _ = block_bootstrap_diff(lb[m], la[m], day[m], n_boot=1000)
            e_rows.append({"head": h, "elapsed": str(b), "A skill %": 100 * (1 - la[m].mean() / l0[m].mean()),
                           "B skill %": 100 * (1 - lb[m].mean() / l0[m].mean()), "dLL B-A": dd, "CI lo": l_, "CI hi": h_})
        mon = (np.datetime64("2025-01-01") + day.astype("timedelta64[D]")).astype("datetime64[M]").astype(str)
        for mo in np.unique(mon):
            m = mon == mo
            m_rows.append({"head": h, "month": mo, "dLL B-A": (lb[m] - la[m]).mean(),
                           "B skill %": 100 * (1 - lb[m].mean() / l0[m].mean())})
    f = lambda df: df.to_string(index=False, float_format=lambda x: f"{x:.5f}" if abs(x) < 1 else f"{x:.3f}")
    out.append("## P(S_target >= P0), blended forecast, all rows\n```\n" + f(pd.DataFrame(main_rows)) + "\n```\n")
    out.append("## dLL by elapsed time in the window\n```\n" + f(pd.DataFrame(e_rows)) + "\n```\n")
    mt = pd.DataFrame(m_rows).pivot(index="month", columns="head", values="dLL B-A")
    out.append("## dLL by month (B better where negative)\n```\n" + f(mt.reset_index()[["month"] + [h for h in HEADS if h in mt]])
               + f"\n```\nmonths with B better: " + ", ".join(f"{h} {int((mt[h] < 0).sum())}/{mt[h].notna().sum()}"
                                                          for h in HEADS if h in mt) + "\n")
    s = "\n".join(out)
    print(s)
    open(os.path.join(ART_DIR, f"xasset_{SPLIT}{A}_vs{B}.md"), "w").write(s)


if __name__ == "__main__":
    main()
