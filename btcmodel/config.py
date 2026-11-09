"""Asset configuration: one switch for symbol, data, models and artifacts.

    BTC_ASSET=ETH python scripts/train_eval.py valid h30     # everything for ETHUSDT

Defaults (BTC) reproduce the original paths exactly:
  BTC: data /home/user/data,      artifacts artifacts/
  ETH: data /home/user/data_eth,  artifacts artifacts/eth/
"""
import os

ASSET = os.environ.get("BTC_ASSET", "BTC").upper()
SYMBOL = f"{ASSET}USDT"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("BTC_DATA_DIR",
                          "/home/user/data" if ASSET == "BTC" else f"/home/user/data_{ASSET.lower()}")
MODEL_DIR = os.path.join(DATA_DIR, "models")
STACK_DIR = os.path.join(DATA_DIR, "stack")
ART_DIR = os.environ.get("BTC_ART_DIR") or (
    os.path.join(ROOT, "artifacts") if ASSET == "BTC" else os.path.join(ROOT, "artifacts", ASSET.lower()))
FIG_DIR = os.path.join(ART_DIR, "figures")
for _d in (MODEL_DIR, STACK_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

# Cross-asset add-on (btcmodel/xasset.py): the second market whose 1s timeline is read
# alongside the primary one.  Both timelines must start on the same day (identical
# second indexing).  Only used when BTC_XASSET=1 (training) or when a model needs it (live).
XASSET = os.environ.get("BTC_XASSET_SYMBOL", "ETH" if ASSET == "BTC" else "BTC").upper()
XDATA_DIR = os.environ.get("BTC_XDATA_DIR",
                           "/home/user/data" if XASSET == "BTC" else f"/home/user/data_{XASSET.lower()}")
