"""Compute cross-asset (ETH) features for every cached decision point (same row order).

    BTC_XDATA_DIR=/home/user/data_eth python scripts/build_xasset.py [train valid test]

Writes <cache>/<split>_xasset.parquet, merged by train_eval.load_head when BTC_XASSET=1.
Needs the second market's timeline (BTC_ASSET=ETH python -m btcmodel.timeline)."""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btcmodel.config import XDATA_DIR  # noqa: E402
from btcmodel.dataset import CACHE  # noqa: E402
from btcmodel.timeline import load_timeline  # noqa: E402
from btcmodel.xasset import xasset_features  # noqa: E402

if __name__ == "__main__":
    tl, xtl = load_timeline(), load_timeline(XDATA_DIR)
    assert tl.d0 == xtl.d0, (tl.d0, xtl.d0)
    print(f"primary {tl.d0}..{tl.d1} ({tl.present.sum()} days), second {xtl.d0}..{xtl.d1} "
          f"({xtl.present.sum()} days, from {xtl.tl_dir})")
    for split in (sys.argv[1:] or ["train", "valid", "test"]):
        t = time.time()
        d = pd.read_parquet(os.path.join(CACHE, f"{split}.parquet"), columns=["T0", "e"])
        r = xasset_features(tl, xtl, d["T0"].values.astype(np.int64), d["e"].values.astype(np.int64))
        assert len(r) == len(d)
        r.reset_index(drop=True).to_parquet(os.path.join(CACHE, f"{split}_xasset.parquet"))
        print(split, r.shape, f"{time.time() - t:.0f}s, rows with missing second market: "
              f"{r['xe_pr_1'].isna().mean():.4%}")
        print(r.describe().T[["mean", "std", "min", "max"]].to_string())
