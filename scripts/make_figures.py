"""Figures for the report (from the cached evaluation predictions)."""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "artifacts", "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, ROOT)

C = {"B0_naive": "#9aa0a6", "B1_volmodel": "#7fa7d9", "B3_struct": "#3b6fb6", "M_final": "#d9480f",
     "D_directGBM": "#6c9a3b"}


def main(split):
    P = {}
    for head in ["h30", "h60", "close"]:
        f = f"/home/user/data/models/preds_{head}_{split}.pkl"
        if os.path.exists(f):
            with open(f, "rb") as fh:
                P[head] = pickle.load(fh)
    from btcmodel.prob import reliability, logloss
    # 1. reliability diagrams
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, head in zip(axs, [h for h in ["h30", "h60", "close"] if h in P]):
        y = P[head]["y"]
        for k in ["B0_naive", "M_final"]:
            r = reliability(y, P[head]["p"][k], nb=25)
            ax.plot(r[:, 0], r[:, 1], "o-", ms=3, color=C[k], label=k)
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(f"{head}: P(S_target >= P0) reliability ({split})")
        ax.set_xlabel("predicted"); ax.set_ylabel("observed")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"reliability_{split}.png"), dpi=130); plt.close(fig)
    # 2. log-loss skill vs naive by elapsed second
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for head, col in [(h, c) for h, c in [("h30", "#7fa7d9"), ("h60", "#3b6fb6"), ("close", "#d9480f")] if h in P]:
        e, y = P[head]["e"], P[head]["y"]
        l0 = logloss(y, P[head]["p"]["B0_naive"]); l1 = logloss(y, P[head]["p"]["M_final"])
        bins = np.arange(0, 301, 15)
        g = np.digitize(e, bins) - 1
        sk = [100 * (1 - l1[g == j].mean() / l0[g == j].mean()) if (g == j).any() else np.nan for j in range(len(bins) - 1)]
        ax.plot(bins[:-1] + 7.5, sk, "o-", color=col, label=head)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("elapsed seconds in the 5-minute window"); ax.set_ylabel("log-loss skill vs naive Gaussian (%)")
    ax.set_title(f"Where the edge comes from ({split})"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"skill_by_elapsed_{split}.png"), dpi=130); plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "valid")
