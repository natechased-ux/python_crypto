#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fib5.py (Kraken-ready)
- Zones: FIB (Tiny/Short/Medium/Long), EMA (1H/1D), SWING High/Low
- Liquidity confirm: Absorption (reversal) / Imbalance (continuation)
- Continuation exits: liquidity-based (TP at next strong opposite cluster; SL beyond defended cluster) with ATR fallback
- Auto-tuner: per-coin alerts/day target + outcome-aware win-rate shaping, warmup mode
- Outcome monitor: fills hit_tp/hit_sl/outcome + MAE/MFE (% and ATR units)
- Diagnostics: zone printouts + debug candidates
- Data source: Kraken spot (WS + REST)

Run:
  pip install pandas requests websocket-client python-dateutil
  python fib5_kraken.py
"""

import os, time, json, csv, threading, collections
import pandas as pd
import numpy as np   # <-- add this
import numpy as np                    # used by np.where(...) in allowlist
from itertools import combinations    # used to build singles/pairs/triples


# ===== Exit Learning (EWMA + Quantiles) =====
EXITS_JSON_PATH = "exits_profile.json"
EXITS_LEARN_WINDOW_HOURS = 720   # 30 days
EXITS_MIN_SAMPLES = 40
EXITS_EWMA_TAU_HOURS = 72        # recency weight
EXITS_TP_QUANT = 0.65
EXITS_SL_QUANT = 0.60
EXITS_WR_FLOOR = 0.48            # min WR for learned exits
EXITS_LEARN_EVERY_SEC = 3*3600   # refresh every 3h
EXPLORATION_RATE = 0.10
EXPLORATION_TP_JITTER = 0.3      # +/- R
EXPLORATION_SL_JITTER = 0.2      # +/- R

def _now_utc_pd():
    import pandas as _pd
    return _pd.Timestamp.now(tz="UTC")

def _safe_json_read(path):
    try:
        import json
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _safe_json_write(path, data):
    try:
        import json
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print("[exits] save failed:", e)


# ===== Adaptive Regime & Combo Miner =====
ADAPTIVE_JSON_PATH = "adaptive_rules.json"
ADAPTIVE_MIN_TRADES = 25
ADAPTIVE_TOP_K = 8
ADAPTIVE_WINDOW_HOURS = 72
SENTINEL_WINDOW_HOURS = 24
SENTINEL_MIN_CLOSED = 40
SENTINEL_BAD_NETR = -5.0
SENTINEL_BAD_WR = 0.45
SENTINEL_HYSTERESIS_STEPS = 8
SLOW_FLOW_MULT = 1.3
BASE_FLOW_MULT = 1.0
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser
import urllib.parse as _url

import requests
import pandas as pd

# ========================= Env / Config =========================
BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw")
CHAT_ID   = os.getenv("TG_CHAT_ID", "7967738614")
KRAKEN_BASE = "https://api.kraken.com"

# Scan cadence
INTERVAL_SECONDS  = int(os.getenv("INTERVAL_SECONDS", "5"))
COOLDOWN_MINUTES  = int(os.getenv("COOLDOWN_MINUTES", "30"))

# Paths
# Make fib6 fully independent from fib5
LOG_PATH   = os.getenv("FIB6_TRADE_LOG_PATH", "live_trade_log_fib6.csv")
THR_PATH   = os.getenv("FIB6_LIQ_THRESHOLDS_PATH", "thresholds_fib6.json")
STATE_PATH = os.getenv("FIB6_STATE_PATH", "bot_state_fib6.json")


# Targets / Tuning
TARGET_ALERTS_PER_DAY = float(os.getenv("TARGET_ALERTS_PER_DAY", "1.0")) # per coin
TARGET_BAND           = float(os.getenv("TARGET_BAND", "1.5"))
AUTOTUNE_INTERVAL_MIN = int(os.getenv("AUTOTUNE_INTERVAL_MIN", "10"))
AUTOTUNE_MIN_SCALE    = float(os.getenv("AUTOTUNE_MIN_SCALE", "0.85"))
AUTOTUNE_MAX_SCALE    = float(os.getenv("AUTOTUNE_MAX_SCALE", "1.15"))

WINRATE_LOOKBACK_H = int(os.getenv("WINRATE_LOOKBACK_H", "72"))
WINRATE_TARGET     = float(os.getenv("WINRATE_TARGET", "0.75"))
WINRATE_BAND       = float(os.getenv("WINRATE_BAND", "0.10"))

# Warmup (permissive early)
WARMUP_HOURS                 = int(os.getenv("WARMUP_HOURS", "6"))
WARMUP_MIN_ALERTS_PER_COIN   = int(os.getenv("WARMUP_MIN_ALERTS_PER_COIN", "6"))
WARMUP_TARGET_ALERTS_PER_DAY = float(os.getenv("WARMUP_TARGET_ALERTS_PER_DAY", "10.0"))
WARMUP_TARGET_BAND           = float(os.getenv("WARMUP_TARGET_BAND", "2.0"))
WARMUP_FLOOR_MULT            = float(os.getenv("WARMUP_FLOOR_MULT", "0.30"))

# Feature flags
ENABLE_SWING_ZONES   = os.getenv("ENABLE_SWING_ZONES", "true").lower() == "true"
ENABLE_CONTINUATION  = os.getenv("ENABLE_CONTINUATION", "false").lower() == "true"
WEEKEND_DISABLE_CONT = os.getenv("WEEKEND_DISABLE_CONT", "false").lower() == "true"
ENABLE_STATUS_CMDS   = os.getenv("ENABLE_STATUS_CMDS", "true").lower() == "true"

# ------ Market Agreement (live flow-weighted ratio) ------
MARKET_AGREEMENT_ENABLED   = bool(int(os.getenv("MARKET_AGREEMENT_ENABLED", "1")))
AGREEMENT_LOOKBACK_SEC     = int(os.getenv("AGREEMENT_LOOKBACK_SEC", "60"))   # 60s window
AGREEMENT_MIN_COINS        = int(os.getenv("AGREEMENT_MIN_COINS",  "6"))     # minimum breadth
AGREEMENT_MIN_TOTAL_USD    = float(os.getenv("AGREEMENT_MIN_TOTAL_USD", "200000"))  # total |delta| in window
AGREEMENT_MIN_RATIO        = float(os.getenv("AGREEMENT_MIN_RATIO", "0.20")) # market flow skew threshold

# Outcome monitor cadence
OUTCOME_POLL_SECONDS    = int(os.getenv("OUTCOME_POLL_SECONDS", "60"))
OUTCOME_GRANULARITY_SEC = int(os.getenv("OUTCOME_GRANULARITY_SEC", "60"))

# Diagnostics
ZONE_DIAG_INTERVAL_SEC = int(os.getenv("ZONE_DIAG_INTERVAL_SEC", "60"))
HEALTH_ALERT_NOALERTS_H= int(os.getenv("HEALTH_ALERT_NOALERTS_H", "12"))
DEBUG_TOP_N            = int(os.getenv("DEBUG_TOP_N", "5"))

# Zones / bands
GOLDEN_MIN   = 0.618
GOLDEN_MAX   = 0.66
GOLDEN_TOL   = 0.0025       # ±0.25% pad
EMA_BAND_PCT = 0.005        # ±0.5% band
SL_BUFFER    = 0.01

# Swing zone settings
SWING_LOOKBACK_BARS_1H = int(os.getenv("SWING_LOOKBACK_BARS_1H", "60"))
SWING_FRAC_LEFT        = int(os.getenv("SWING_FRAC_LEFT", "3"))
SWING_FRAC_RIGHT       = int(os.getenv("SWING_FRAC_RIGHT", "3"))
SWING_ZONE_ATR_MULT    = float(os.getenv("SWING_ZONE_ATR_MULT", "0.25")) # half-band = mult*ATR
SWING_ZONE_MIN_PCT     = float(os.getenv("SWING_ZONE_MIN_PCT", "0.0015"))

# Coins (Kraken symbols)
COINS = [
    "XBT/USD","ETH/USD",
    "SOL/USD","XRP/USD","MATIC/USD","ADA/USD","AVAX/USD","DOGE/USD","DOT/USD",
    "LINK/USD","ATOM/USD","NEAR/USD","ARB/USD","OP/USD",
    "INJ/USD","AAVE/USD","LTC/USD","BCH/USD","ETC/USD","ALGO/USD","FIL/USD","ICP/USD",
    "RNDR/USD","STX/USD","GRT/USD","SEI/USD","HBAR/USD","JUP/USD","TIA/USD",
    "UNI/USD","MKR/USD","FET/USD","CRV/USD","SAND/USD","MANA/USD","AXS/USD",
    "PEPE/USD","XLM/USD"
]

# ========================= Telegram =========================
def send_telegram(msg: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("[TG] skipped (set TG_BOT_TOKEN & TG_CHAT_ID)")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=12
        )
    except Exception as e:
        print("[TG] error:", e)

# --- Quintile support (global & per-coin) ---
# --- Quintile helpers ---
def _load_quintile_edges(log_path: str, lookback_h: int = 168):
    import pandas as pd, numpy as np
    try:
        df = pd.read_csv(log_path, encoding="utf-8-sig")
    except Exception:
        return None, None
    if "timestamp_utc" not in df or "delta_usd" not in df:
        return None, None
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df[df["timestamp_utc"] >= (pd.Timestamp.utcnow() - pd.Timedelta(hours=lookback_h))]
    df["abs_delta"] = pd.to_numeric(df["delta_usd"], errors="coerce").abs()
    df = df.dropna(subset=["abs_delta"])
    if df.empty:
        return None, None
    # Global edges (q20,q40,q60,q80)
    g_edges = np.quantile(df["abs_delta"], [0.2, 0.4, 0.6, 0.8])
    # Per-coin edges
    pc_edges = {}
    for sym, g in df.groupby("symbol"):
        x = pd.to_numeric(g["abs_delta"], errors="coerce").dropna()
        if len(x) >= 20:
            pc_edges[sym] = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    return g_edges, pc_edges

def _assign_quintile(x: float, edges):
    if edges is None or not isinstance(x, (int, float)):
        return None
    q20,q40,q60,q80 = edges
    if x <= q20: return 1
    if x <= q40: return 2
    if x <= q60: return 3
    if x <= q80: return 4
    return 5



# ========================= Utils =========================
def fmt_price(px: float) -> str:
    if px >= 100: return f"{px:.2f}"
    if px >= 10:  return f"{px:.3f}"
    if px >= 1:   return f"{px:.4f}"
    if px >= 0.1: return f"{px:.5f}"
    return f"{px:.9f}"

def fmt_dollars(n: float) -> str:
    n = float(n)
    absn = abs(n)
    if absn >= 1_000_000_000: s = f"{n/1_000_000_000:.2f}B"
    elif absn >= 1_000_000:   s = f"{n/1_000_000:.2f}M"
    elif absn >= 1_000:       s = f"{n/1_000:.2f}K"
    else:                     s = f"{n:.0f}"
    return f"${s.rstrip('0').rstrip('.')}"

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr_wilder_df(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    tr = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def atr_wilder_value(df: pd.DataFrame, length: int = 14) -> float:
    return float(atr_wilder_df(df, length).iloc[-1])

def rsi_wilder(series: pd.Series, length: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    delta = s.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0.0, pd.NA)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(method="bfill").fillna(50.0)

def stoch_rsi_kd(series_close: pd.Series, rsi_len: int = 14, stoch_len: int = 14, k_len: int = 3, d_len: int = 3) -> tuple[float,float]:
    rsi = rsi_wilder(series_close, rsi_len)
    low = rsi.rolling(stoch_len, min_periods=max(2, stoch_len//2)).min()
    high = rsi.rolling(stoch_len, min_periods=max(2, stoch_len//2)).max()
    denom = (high - low)
    stoch = ((rsi - low) / denom.where(denom != 0, pd.NA)).fillna(0.5)
    k = stoch.rolling(k_len, min_periods=1).mean() * 100.0
    d = k.rolling(d_len, min_periods=1).mean()
    return float(k.iloc[-1]), float(d.iloc[-1])

def fib_golden_zone(lo: float, hi: float) -> tuple[float,float]:
    r618 = lo + GOLDEN_MIN * (hi - lo)
    r660 = lo + GOLDEN_MAX * (hi - lo)
    zmin, zmax = min(r618, r660), max(r618, r660)
    pad = ((zmin+zmax)/2) * GOLDEN_TOL
    return zmin - pad, zmax + pad

import re
MUTE_CONTINUATION_ALERTS = os.getenv("MUTE_CONTINUATION_ALERTS","true").lower()=="true"

def _extract_mode(msg: str):
    m = re.search(r"—\s*(CONTINUATION|REVERSAL)\s*—", msg)
    return m.group(1) if m else None

def _extract_side(msg: str):
    m = re.search(r"\*(LONG|SHORT)\*\s+on\s+\*", msg)
    return m.group(1) if m else None


def price_in_zone(px, zmin, zmax): return (zmin is not None) and (zmax is not None) and (zmin <= px <= zmax)

def detect_reversal(prev_px: float, curr_px: float, zmin: float, zmax: float):
    if curr_px < zmin and prev_px >= zmin: return "LONG"
    if curr_px > zmax and prev_px <= zmax: return "SHORT"
    if zmin <= curr_px <= zmax:
        if curr_px >= prev_px: return "LONG"
        if curr_px <= prev_px: return "SHORT"
    return None

def zone_edge(curr: float, zmin: float, zmax: float) -> str:
    mid = 0.5 * (zmin + zmax)
    one_third = (zmax - zmin) / 3.0 if zmax > zmin else 0.0
    if curr >= mid + one_third: return "upper"
    if curr <= mid - one_third: return "lower"
    return "middle"

def choose_mode(liq_reason: str, delta_usd: float, edge: str) -> str:
    has_abs = "Absorption" in liq_reason
    has_imb = "Imbalance" in liq_reason
    if has_abs and not has_imb: return "REVERSAL"
    if has_imb and not has_abs: return "CONTINUATION"
    if has_abs and has_imb:
        if (edge == "upper" and delta_usd > 0) or (edge == "lower" and delta_usd < 0):
            return "CONTINUATION"
        return "REVERSAL"
    return "REVERSAL"

def is_weekend_utc(now: datetime) -> bool:
    return now.weekday() >= 5  # 5=Sat, 6=Sun

# ========================= Data fetch (Kraken REST) =========================
def _kr_pair_for_rest(pair: str) -> str:
    # "XBT/USD" -> "XBTUSD"; Kraken REST likes the compact form
    return pair.replace("/", "")

def fetch_candles(product_id: str, granularity: int) -> pd.DataFrame:
    """
    Kraken OHLC: /0/public/OHLC?pair=XBTUSD&interval=60
    Kraken rows are [time, open, high, low, close, vwap, volume, count] (8),
    sometimes vwap omitted (7). We normalize and emit:
    ["time","low","high","open","close","volume"].
    """
    itv_map = {60: 1, 300: 5, 900: 15, 1800: 30, 3600: 60, 21600: 240, 86400: 1440}
    interval = itv_map.get(granularity, 60)

    pair_q = _kr_pair_for_rest(product_id)
    url = f"{KRAKEN_BASE}/0/public/OHLC?pair={_url.quote(pair_q)}&interval={interval}"
    r = requests.get(url, timeout=15); r.raise_for_status()
    res = r.json().get("result", {})

    ohlc = next((v for k, v in res.items() if k != "last"), None)
    if not ohlc:
        return pd.DataFrame(columns=["time","low","high","open","close","volume"])

    df = pd.DataFrame(ohlc)

    # Normalize columns
    if df.shape[1] == 8:
        df.columns = ["time","open","high","low","close","vwap","volume","count"]
    elif df.shape[1] == 7:
        df.columns = ["time","open","high","low","close","volume","count"]
        df["vwap"] = pd.NA
    else:
        # pad/trim defensively to 8
        while df.shape[1] < 8:
            df[df.shape[1]] = pd.NA
        df = df.iloc[:, :8]
        df.columns = ["time","open","high","low","close","vwap","volume","count"]

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)

    # Emit the 6 columns your pipeline expects, in your expected order
    return df[["time","low","high","open","close","volume"]]


def fetch_trades(product_id: str, limit: int = 300) -> pd.DataFrame:
    """
    Kraken Trades: /0/public/Trades?pair=XBTUSD
    Payload rows are typically 6 fields:
        [price, volume, time, side, ordertype, misc]
    but may include a 7th field (e.g., trade id). We normalize both.
    """
    pair_q = _kr_pair_for_rest(product_id)
    url = f"{KRAKEN_BASE}/0/public/Trades?pair={_url.quote(pair_q)}"
    r = requests.get(url, timeout=12); r.raise_for_status()
    res = r.json().get("result", {})

    arr = next((v for k, v in res.items() if k != "last"), None)
    if not arr:
        return pd.DataFrame()

    # Determine row width dynamically
    # Common: 6 fields => ["price","volume","time","side","ordertype","misc"]
    # Some:   7 fields => add "trade_id"
    first = arr[0] if len(arr) else []
    n = len(first)

    if n >= 7:
        cols = ["price","volume","time","side","ordertype","misc","trade_id"]
        df = pd.DataFrame(arr, columns=cols[:n])  # if >7, we’ll truncate
    elif n == 6:
        cols = ["price","volume","time","side","ordertype","misc"]
        df = pd.DataFrame(arr, columns=cols)
    else:
        # Fallback: build without column names, then coerce what we can
        df = pd.DataFrame(arr)
        # Ensure at least first 6 cols exist
        while df.shape[1] < 6:
            df[df.shape[1]] = pd.NA
        df = df.iloc[:, :6]
        df.columns = ["price","volume","time","side","ordertype","misc"]

    # Types & normalization
    df["time"]   = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    df["price"]  = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")  # base volume
    # Map taker side: Kraken uses 'b'/'s'
    df["side"]   = df["side"].map({"b":"buy","s":"sell"}).fillna("buy")

    # Notional in quote currency for ΔCVD
    df["notional"] = df["price"] * df["volume"]

    df = df.sort_values("time").reset_index(drop=True)
    return df


def _kr_pair_for_rest(pair: str) -> str:
    # "XBT/USD" -> "XBTUSD"
    return pair.replace("/", "")

def fetch_book(product_id: str, depth: int = 200) -> dict:
    """
    Kraken Depth -> {"bids":[[px,vol,ts],...], "asks":[[px,vol,ts],...]}
    We ALWAYS return {"bids":[(px,vol)...], "asks":[(px,vol)...]}.
    """
    pair_q = _kr_pair_for_rest(product_id)
    url = f"{KRAKEN_BASE}/0/public/Depth?pair={_url.quote(pair_q)}&count={depth}"
    r = requests.get(url, timeout=12); r.raise_for_status()
    res = r.json().get("result", {})
    dp = next((v for k, v in res.items()), None)
    if not dp:
        return {"bids": [], "asks": []}
    bids = [(float(row[0]), float(row[1])) for row in dp.get("bids", [])[:depth]]
    asks = [(float(row[0]), float(row[1])) for row in dp.get("asks", [])[:depth]]
    return {"bids": bids, "asks": asks}

def ensure_book_dict(book):
    """If any old code still returns a tuple (bids, asks), convert it into the dict format."""
    if isinstance(book, tuple) and len(book) == 2:
        bids, asks = book
        return {"bids": bids, "asks": asks}
    return book
# ======= GLOBAL SAFETY WRAPPERS (put once, after fetch_* defs) =======
# Wrap fetch_book so ANY caller gets a dict, even if some code expects a tuple.
try:
    _fetch_book_raw = fetch_book
except NameError:
    pass
else:
    def _ensure_book_dict(book):
        if isinstance(book, tuple) and len(book) == 2:
            bids, asks = book
            return {"bids": bids, "asks": asks}
        return book

    def fetch_book(product_id: str, depth: int = 200):
        book = _fetch_book_raw(product_id, depth=depth)
        # If the underlying returns a tuple or odd shape, coerce to the dict we expect
        book = _ensure_book_dict(book)
        if not isinstance(book, dict):
            # last-resort normalization
            try:
                bids, asks = book  # may raise
                book = {"bids": bids, "asks": asks}
            except Exception:
                book = {"bids": [], "asks": []}
        # Only keep (px, vol) pairs
        def _clean(side):
            clean = []
            for row in book.get(side, []):
                try:
                    clean.append((float(row[0]), float(row[1])))
                except Exception:
                    continue
            return clean
        return {"bids": _clean("bids"), "asks": _clean("asks")}
    

# Wrap fetch_candles so ANY caller gets 6 normalized columns, even if some old code path constructs DF directly.
try:
    _fetch_candles_raw = fetch_candles
except NameError:
    pass
else:
    def fetch_candles(product_id: str, granularity: int):
        df = _fetch_candles_raw(product_id, granularity)
        # If some old path built a different shape, coerce it here
        expected = ["time","low","high","open","close","volume"]
        if not isinstance(df, pd.DataFrame) or any(col not in df.columns for col in expected):
            # Attempt to rebuild via Kraken REST (fallback)
            try:
                # Do the robust Kraken OHLC call again
                pair_q = product_id.replace("/", "")
                itv_map = {60: 1, 300: 5, 900: 15, 1800: 30, 3600: 60, 21600: 240, 86400: 1440}
                interval = itv_map.get(granularity, 60)
                url = f"{KRAKEN_BASE}/0/public/OHLC?pair={_url.quote(pair_q)}&interval={interval}"
                r = requests.get(url, timeout=15); r.raise_for_status()
                res = r.json().get("result", {})
                ohlc = next((v for k, v in res.items() if k != "last"), None)
                tmp = pd.DataFrame(ohlc)
                if tmp.shape[1] == 8:
                    tmp.columns = ["time","open","high","low","close","vwap","volume","count"]
                elif tmp.shape[1] == 7:
                    tmp.columns = ["time","open","high","low","close","volume","count"]
                    tmp["vwap"] = pd.NA
                else:
                    while tmp.shape[1] < 8:
                        tmp[tmp.shape[1]] = pd.NA
                    tmp = tmp.iloc[:, :8]
                    tmp.columns = ["time","open","high","low","close","vwap","volume","count"]
                tmp["time"] = pd.to_datetime(tmp["time"], unit="s", utc=True)
                for col in ["open","high","low","close","volume"]:
                    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
                df = tmp[["time","low","high","open","close","volume"]].sort_values("time").reset_index(drop=True)
            except Exception:
                # As absolute last resort, return an empty but correctly-shaped frame
                df = pd.DataFrame(columns=expected)
        else:
            # Ensure correct order
            df = df[expected].copy()
        return df

# ========================= Streaming (Kraken WS) =========================
try:
    import websocket  # websocket-client
except Exception:
    websocket = None

class TradeRec:
    __slots__ = ("ts","side","price","notional")
    def __init__(self, ts: float, side: str, price: float, notional: float):
        self.ts = ts; self.side = side; self.price = price; self.notional = notional

class TradeStream:
    def __init__(self, maxlen=12000):
        self.buffers = collections.defaultdict(lambda: collections.deque(maxlen=maxlen))
        self.locks   = collections.defaultdict(threading.Lock)
        self.counts  = collections.defaultdict(int)
        self.sums    = collections.defaultdict(float)
        self.last_ts = collections.defaultdict(float)
    def add(self, symbol: str, t: TradeRec):
        with self.locks[symbol]:
            self.buffers[symbol].append(t)
            self.counts[symbol] += 1
            self.sums[symbol]   += t.notional
            self.last_ts[symbol] = t.ts
    def window(self, symbol: str, lookback_s: int):
        cutoff = time.time() - lookback_s
        buys = sells = 0.0
        n = 0
        with self.locks[symbol]:
            for t in reversed(self.buffers[symbol]):
                if t.ts < cutoff: break
                n += 1
                if t.side == "buy": buys += t.notional
                else:               sells += t.notional
        return {"buys":buys, "sells":sells, "delta":buys-sells, "n":n}
    def calib_stats(self, symbol: str):
        return self.counts[symbol], self.sums[symbol]

class OrderBooks:
    def __init__(self):
        self.bids = collections.defaultdict(list)
        self.asks = collections.defaultdict(list)
        self.mid  = {}
        self.last_ts = collections.defaultdict(float)
        self.locks = collections.defaultdict(threading.Lock)
    def update_snapshot(self, symbol: str, bids, asks):
        with self.locks[symbol]:
            self.bids[symbol] = bids
            self.asks[symbol] = asks
            if bids and asks:
                self.mid[symbol] = 0.5*(float(bids[0][0]) + float(asks[0][0]))
            self.last_ts[symbol] = time.time()
    def apply_changes(self, symbol: str, changes):
        with self.locks[symbol]:
            bd = {float(px): float(sz) for px,sz in self.bids[symbol]}
            ad = {float(px): float(sz) for px,sz in self.asks[symbol]}
            for side, px_s, sz_s in changes:
                px = float(px_s); sz = float(sz_s)
                if side == "buy":
                    if sz == 0: bd.pop(px, None)
                    else: bd[px] = sz
                else:
                    if sz == 0: ad.pop(px, None)
                    else: ad[px] = sz
            bids = sorted([(p,s) for p,s in bd.items()], key=lambda x:x[0], reverse=True)[:200]
            asks = sorted([(p,s) for p,s in ad.items()], key=lambda x:x[0])[:200]
            self.bids[symbol] = bids
            self.asks[symbol] = asks
            if bids and asks:
                self.mid[symbol] = 0.5*(bids[0][0] + asks[0][0])
            self.last_ts[symbol] = time.time()
    def sum_nearby(self, symbol: str, pct=0.01):
        with self.locks[symbol]:
            mid = self.mid.get(symbol)
            if mid is None: return 0.0, 0.0, None
            lo, hi = mid*(1-pct), mid*(1+pct)
            def tot(arr):
                t=0.0
                for row in arr:
                    px=float(row[0]); sz=float(row[1])
                    if lo <= px <= hi: t += px*sz
                return t
            return tot(self.bids[symbol]), tot(self.asks[symbol]), mid


# ========================= Market Flow Aggregator (live) =========================
class MarketFlow:
    """Tracks a rolling window of delta_usd per coin and produces a market flow score in [-1,+1].
    Score = (sum(delta_usd) / sum(|delta_usd|)) over the window across all coins."""
    def __init__(self, lookback_sec=60):
        self.lookback = lookback_sec
        self.locks = collections.defaultdict(threading.Lock)
        self.buffers = collections.defaultdict(lambda: collections.deque())

    def _purge_old(self, symbol, now_ts):
        buf = self.buffers[symbol]
        while buf and now_ts - buf[0][0] > self.lookback:
            buf.popleft()

    def register(self, symbol: str, delta_usd: float, now_ts: float = None):
        if not symbol:
            return
        if now_ts is None:
            now_ts = time.time()
        with self.locks[symbol]:
            self._purge_old(symbol, now_ts)
            self.buffers[symbol].append((now_ts, float(delta_usd)))

    def snapshot(self):
        now_ts = time.time()
        sum_delta = 0.0
        sum_abs = 0.0
        active = 0
        for sym, lock in self.locks.items():
            with lock:
                buf = self.buffers[sym]
                while buf and now_ts - buf[0][0] > self.lookback:
                    buf.popleft()
                if not buf:
                    continue
                coin_sum = sum(v for _, v in buf)
                active += 1
                sum_delta += coin_sum
                sum_abs += abs(coin_sum)
        score = (sum_delta / sum_abs) if sum_abs > 0 else 0.0
        return score, active, sum_abs

    def gate(self, side: str):
        score, n_coins, sum_abs = self.snapshot()
        now = time.time()
        # breadth & liquidity checks first
        if n_coins < AGREEMENT_MIN_COINS:
            self.last_pass["LONG"] = None; self.last_pass["SHORT"] = None
            return False, f"breadth {n_coins}<{AGREEMENT_MIN_COINS}"
        if sum_abs < AGREEMENT_MIN_TOTAL_USD:
            self.last_pass["LONG"] = None; self.last_pass["SHORT"] = None
            return False, f"flow ${sum_abs:,.0f}<{AGREEMENT_MIN_TOTAL_USD:,.0f}"
        s_upper = str(side).upper()
        if s_upper == "LONG":
            if score >= AGREEMENT_MIN_RATIO:
                if self.last_pass["LONG"] is None:
                    self.last_pass["LONG"] = now
                if now - self.last_pass["LONG"] >= AGREEMENT_PERSISTENCE_SEC:
                    return True, f"market LONG (score {score:+.2f})"
                return False, f"waiting LONG persistence ({now - self.last_pass['LONG']:.1f}/{AGREEMENT_PERSISTENCE_SEC}s, score {score:+.2f})"
            else:
                self.last_pass["LONG"] = None
                return False, f"market {score:+.2f}<{AGREEMENT_MIN_RATIO:.2f}"
        else:
            if score <= -AGREEMENT_MIN_RATIO:
                if self.last_pass["SHORT"] is None:
                    self.last_pass["SHORT"] = now
                if now - self.last_pass["SHORT"] >= AGREEMENT_PERSISTENCE_SEC:
                    return True, f"market SHORT (score {score:+.2f})"
                return False, f"waiting SHORT persistence ({now - self.last_pass['SHORT']:.1f}/{AGREEMENT_PERSISTENCE_SEC}s, score {score:+.2f})"
            else:
                self.last_pass["SHORT"] = None
                return False, f"market {score:+.2f}>-{AGREEMENT_MIN_RATIO:.2f}"
# Global instance
MARKET_FLOW = MarketFlow(lookback_sec=AGREEMENT_LOOKBACK_SEC)

class KrakenWS:
    URL = "wss://ws.kraken.com"

    def __init__(self, symbols, trades: TradeStream, books: OrderBooks):
        self.symbols = symbols
        self.trades  = trades
        self.books   = books
        self.ws      = None
        self.thread  = None

    def _on_open(self, ws):
        ws.send(json.dumps({"event":"subscribe","pair": self.symbols,"subscription":{"name":"trade"}}))
        ws.send(json.dumps({"event":"subscribe","pair": self.symbols,"subscription":{"name":"book","depth":100}}))

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
        except Exception:
            return
        if isinstance(data, dict):
            return
        if not isinstance(data, list) or len(data) < 2:
            return

        chan = data[-2] if len(data) >= 3 else ""
        pair = data[-1] if len(data) >= 4 else ""

        if chan == "trade" and isinstance(data[1], list):
            for t in data[1]:
                try:
                    px   = float(t[0]); vol = float(t[1]); ts = float(t[2])
                    side = "buy" if t[3] == "b" else "sell"
                    self.trades.add(pair, TradeRec(ts, side, px, px*vol))
                except Exception:
                    continue
            return

        if isinstance(data[1], dict) and chan.startswith("book"):
            payload = data[1]
            if "as" in payload or "bs" in payload:
                try:
                    asks = [(float(a[0]), float(a[1])) for a in payload.get("as", [])][:200]
                    bids = [(float(b[0]), float(b[1])) for b in payload.get("bs", [])][:200]
                    asks.sort(key=lambda x: x[0]); bids.sort(key=lambda x: x[0], reverse=True)
                    self.books.update_snapshot(pair, bids, asks)
                except Exception:
                    pass

            updated = False
            if "a" in payload:
                try:
                    with self.books.locks[pair]:
                        cur = {float(px): float(sz) for px, sz in self.books.asks.get(pair, [])}
                        for a in payload["a"]:
                            px = float(a[0]); sz = float(a[1])
                            if sz == 0: cur.pop(px, None)
                            else:       cur[px] = sz
                        asks = sorted(cur.items(), key=lambda x: x[0])[:200]
                        self.books.asks[pair] = asks
                        updated = True
                except Exception:
                    pass

            if "b" in payload:
                try:
                    with self.books.locks[pair]:
                        cur = {float(px): float(sz) for px, sz in self.books.bids.get(pair, [])}
                        for b in payload["b"]:
                            px = float(b[0]); sz = float(b[1])
                            if sz == 0: cur.pop(px, None)
                            else:       cur[px] = sz
                        bids = sorted(cur.items(), key=lambda x: x[0], reverse=True)[:200]
                        self.books.bids[pair] = bids
                        updated = True
                except Exception:
                    pass

            if updated:
                with self.books.locks[pair]:
                    bids = self.books.bids.get(pair, []); asks = self.books.asks.get(pair, [])
                    if bids and asks:
                        self.books.mid[pair] = 0.5*(bids[0][0] + asks[0][0])
                        self.books.last_ts[pair] = time.time()

    def _on_error(self, ws, err):
        print("[KrakenWS error]", err)

    def _on_close(self, ws, *args):
        print("[KrakenWS closed]")

    def start(self):
        if websocket is None:
            print("[KrakenWS] websocket-client not installed (pip install websocket-client)")
            return
        self.ws = websocket.WebSocketApp(
            self.URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        self.thread = threading.Thread(target=self.ws.run_forever, kwargs={"ping_interval":20,"ping_timeout":10}, daemon=True)
        self.thread.start()

# ========================= Rules / Playbook (fallbacks) =========================
# ========================= Rules / Playbook (with safe shim) =========================
try:
    from coin_rules import set_all_coins as _set_all_coins, load_rules as _load_rules, get_styles_for as _get_styles_for
    HAVE_RULES = True
except Exception:
    HAVE_RULES = False
    def _set_all_coins(_): pass
    def _load_rules(): pass
    def _get_styles_for(_product_id, _zone_source): return None

def set_all_coins(coins):  # proxy so the rest of the code doesn't change
    try: _set_all_coins(coins)
    except Exception: pass

def load_rules():
    try: _load_rules()
    except Exception: pass

def get_styles_for(product_id: str, zone_source: str) -> dict:
    """
    Normalizes coin_rules.get_styles_for(...) so callers always get a dict.
    If coin_rules returns a tuple, we take the first element if it's a dict.
    Fallback defaults are provided based on the zone family (FIB/SWING => REVERSAL, EMA => SCALP).
    """
    # sensible defaults
    defaults_rev = {"style":"REVERSAL","tp1_pct":0.7,"tp1_size_pct":60,"tp2_pct":1.6,"timebox_h":6,"sl_mode":"zone"}
    defaults_sca = {"style":"SCALP","tp_pct":0.9,"sl_pct":0.7,"timebox_h":4}
    base = defaults_rev if (zone_source.startswith("FIB") or zone_source.startswith("SWING")) else defaults_sca

    try:
        res = _get_styles_for(product_id, zone_source)
    except Exception:
        return base

    if isinstance(res, dict):
        return {**base, **res}
    if isinstance(res, tuple):
        # common case: (dict, something)
        if len(res) > 0 and isinstance(res[0], dict):
            return {**base, **res[0]}
        # unknown tuple shape — fall back
        return base
    if res is None:
        return base
    # any other type — fall back
    return base


# ========================= Thresholds (tiers tuned for Kraken) =========================
DEFAULTS = dict(
    lookback_s=60,
    delta_usd=4000.0,
    burst_sells_long=1200000.0,
    burst_buys_short=1200000.0,
    near_pct=0.006,
    thick_ratio=1.6,
    cluster_min=2000000.0,
)

MAJORS = {"XBT/USD","ETH/USD"}
LIQUID = {"SOL/USD","XRP/USD","MATIC/USD","ADA/USD","AVAX/USD","DOGE/USD","DOT/USD","LINK/USD","ATOM/USD","NEAR/USD","ARB/USD","OP/USD","LTC/USD","BCH/USD","FIL/USD","ICP/USD"}
MIDS   = set(COINS) - MAJORS - LIQUID
SMALLS = set()

TIER_BASES = {
    "MAJORS": dict(delta_usd=12000.0, burst_sells_long=3000000.0, burst_buys_short=3000000.0, near_pct=0.004, thick_ratio=1.6, cluster_min=8000000.0),
    "LIQUID": dict(delta_usd=4000.0,  burst_sells_long=1200000.0, burst_buys_short=1200000.0, near_pct=0.006, thick_ratio=1.6, cluster_min=3000000.0),
    "MIDS":   dict(delta_usd=2000.0,  burst_sells_long= 800000.0, burst_buys_short= 800000.0, near_pct=0.008, thick_ratio=1.6, cluster_min=1500000.0),
    "SMALLS": dict(delta_usd=1000.0,  burst_sells_long= 400000.0, burst_buys_short= 400000.0, near_pct=0.012, thick_ratio=1.5, cluster_min= 800000.0)
}

def build_thresholds(coins):
    out = {}
    for c in coins:
        base = DEFAULTS.copy()
        if c in MAJORS: base.update(TIER_BASES["MAJORS"])
        elif c in LIQUID: base.update(TIER_BASES["LIQUID"])
        elif c in MIDS:   base.update(TIER_BASES["MIDS"])
        else:             base.update(TIER_BASES["SMALLS"])
        out[c] = base
    return out

def load_custom_thresholds(path):
    try:
        with open(path, "r") as f: return json.load(f)
    except Exception:
        return {}

def save_custom_thresholds(path, data):
    tmp = path+".tmp"
    with open(tmp, "w") as f: json.dump(data, f, indent=2)
    os.replace(tmp, path)

LIQ_THRESHOLDS = build_thresholds(COINS)
LIQ_THRESHOLDS.update(load_custom_thresholds(THR_PATH))
# Ensure every coin has all required keys (backfill from DEFAULTS)
for coin in COINS:
    LIQ_THRESHOLDS.setdefault(coin, {})
    for k, v in DEFAULTS.items():
        LIQ_THRESHOLDS[coin].setdefault(k, v)


# Early-flow auto-calibration
AUTO_CALIBRATE   = True
CALIB_WARMUP_SEC = 600
CALIB_MIN_TRADES = 200
WS_START_TS      = time.time()

def maybe_autocalibrate(symbol, elapsed_sec, trades_stream, thresholds):
    if not AUTO_CALIBRATE or elapsed_sec < CALIB_WARMUP_SEC:
        return thresholds[symbol]
    cnt, sumn = trades_stream.calib_stats(symbol)
    if cnt < CALIB_MIN_TRADES:
        return thresholds[symbol]
    nominal_flow = 5000.0
    avg_flow = max(1.0, sumn / max(1.0, elapsed_sec))
    scale = max(0.5, min(2.5, avg_flow / nominal_flow))
    cfg = thresholds[symbol].copy()
    for k in ("delta_usd","burst_sells_long","burst_buys_short","cluster_min"):
        cfg[k] *= scale
    return cfg

# ========================= Liquidity Confirm =========================
trades_stream = TradeStream()
books_stream  = OrderBooks()
ws_client     = KrakenWS(COINS, trades_stream, books_stream)
if websocket is not None: ws_client.start()

def liquidity_confirm(product_id: str, side_hint: str):
    cfg = maybe_autocalibrate(product_id, time.time()-WS_START_TS, trades_stream, LIQ_THRESHOLDS)
    lookback_s   = int(cfg.get("lookback_s",   DEFAULTS["lookback_s"]))
    delta_th     =     cfg.get("delta_usd",    DEFAULTS["delta_usd"])
    burst_sell_L =     cfg.get("burst_sells_long", DEFAULTS["burst_sells_long"])
    burst_buy_S  =     cfg.get("burst_buys_short", DEFAULTS["burst_buys_short"])
    near_pct     =     cfg.get("near_pct",     DEFAULTS["near_pct"])
    thick_ratio  =     cfg.get("thick_ratio",  DEFAULTS["thick_ratio"])


    # Stream first
    s = trades_stream.window(product_id, lookback_s)
    b_near = a_near = mid = None
    stream_ready = s["n"] >= 25 and books_stream.mid.get(product_id) is not None

    if stream_ready:
        b_near, a_near, mid = books_stream.sum_nearby(product_id, pct=near_pct)
        buys, sells, delta = s["buys"], s["sells"], s["delta"]
        src = "stream"
        book_ts = books_stream.last_ts[product_id]
        tr_ts   = trades_stream.last_ts[product_id]
        if (time.time() - max(book_ts, tr_ts)) > 5.0:  # latency guard
            stream_ready = False

    if not stream_ready:
        tdf = fetch_trades(product_id, limit=300)
        if tdf.empty: return (False, "", {})
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(seconds=lookback_s)
        win = tdf[tdf["time"] >= cutoff]
        buys  = float(win.loc[win["side"]=="buy","notional"].sum())
        sells = float(win.loc[win["side"]=="sell","notional"].sum())
        delta = buys - sells

    # ORDER BOOK — FORCE DICT
        book  = ensure_book_dict(fetch_book(product_id))
        assert isinstance(book, dict), f"fetch_book() must return dict, got {type(book)}"
        bids, asks = book.get("bids", []), book.get("asks", [])
        if not bids or not asks: return (False, "", {})
        mid = 0.5*(bids[0][0] + asks[0][0])

    # near-book notional
        def tot(levels, md, pct):
            lo, hi = md*(1-pct), md*(1+pct)
            return sum(px*sz for px, sz in levels if lo<=px<=hi)
        b_near, a_near = tot(bids, mid, near_pct), tot(asks, mid, near_pct)
        src = "rest"


    if b_near == 0 and a_near == 0: return (False, "", {})

    ok_abs = ok_imb = False; reasons = []
    if side_hint == "LONG":
        if (sells >= burst_sell_L) and (a_near and a_near>0) and (b_near / max(a_near,1e-9) >= thick_ratio) and (delta > -0.4*burst_sell_L):
            ok_abs = True; reasons.append("Absorption")
        if delta >= delta_th:
            ok_imb = True; reasons.append("Imbalance")
    else:
        if (buys >= burst_buy_S) and (b_near and b_near>0) and (a_near / max(b_near,1e-9) >= thick_ratio) and (delta < 0.4*burst_buy_S):
            ok_abs = True; reasons.append("Absorption")
        if delta <= -delta_th:
            ok_imb = True; reasons.append("Imbalance")

    ok = ok_abs or ok_imb
    diag = {
        "delta_usd": delta, "buys_usd": buys, "sells_usd": sells,
        "bid_near_usd": b_near or 0.0, "ask_near_usd": a_near or 0.0,
        "mid": mid or 0.0, "source": src
    }
    return (ok, "+".join(reasons), diag)

# ========================= Liquidity clusters (continuation exits) =========================
def find_cluster_tp_sl(symbol: str, side: str, mid: float, cfg: dict, books: OrderBooks, bias_pct: float = 0.004):
    cluster_min = cfg.get("cluster_min", 2_000_000.0)
    with books.locks[symbol]:
        bids = books.bids.get(symbol, []); asks = books.asks.get(symbol, [])
    if not bids or not asks or mid is None:
        return None, None

    def first_above(levels, start):
        for px, sz in levels:
            if px <= start: continue
            if px*sz >= cluster_min: return px
        return None
    def first_below(levels, start):
        for px, sz in reversed(levels):
            if px >= start: continue
            if px*sz >= cluster_min: return px
        return None

    if side == "LONG":
        tp_px = first_above(asks, mid)
        sl_ref = first_below(bids, mid)
        if tp_px is None or sl_ref is None: return None, None
        return tp_px, sl_ref*(1-bias_pct)
    else:
        tp_px = first_below(bids, mid)
        sl_ref = first_above(asks, mid)
        if tp_px is None or sl_ref is None: return None, None
        return tp_px, sl_ref*(1+bias_pct)

# ========================= Logging / State =========================
def init_log_file():
    cols = [
        "timestamp_utc","symbol","side","mode","entry","sl","tp1","tp2","stop_type",
        "zone","zones_active","strength",
        "delta_usd","near_bids_usd","near_asks_usd","buys_usd","sells_usd","liq_reason","mid","src",
        "outcome","hit_tp","hit_sl","mae_pct","mfe_pct","mae_atr","mfe_atr"
    ]
    if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        with open(LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:   # <-- encoding
            csv.writer(f).writerow(cols)

def ensure_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "timestamp_utc","symbol","side","mode","entry","sl","tp1","tp2","stop_type",
        "zone","zones_active","strength",
        "delta_usd","near_bids_usd","near_asks_usd","buys_usd","sells_usd","liq_reason","mid","src",
        "outcome","hit_tp","hit_sl","mae_pct","mfe_pct","mae_atr","mfe_atr"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c in {"timestamp_utc","mode","zone","zones_active","strength","liq_reason","outcome","src","stop_type"} else 0
    return df

def log_trade(symbol, side, mode, entry, sl, tp1, tp2, stop_type, zone, zones, stars, reason, diag, now):
    init_log_file()
    exists = os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0
    fieldnames = [
        "timestamp_utc","symbol","side","mode","entry","sl","tp1","tp2","stop_type",
        "zone","zones_active","strength",
        "delta_usd","near_bids_usd","near_asks_usd","buys_usd","sells_usd","liq_reason","mid","src",
        "outcome","hit_tp","hit_sl","mae_pct","mfe_pct","mae_atr","mfe_atr"
    ]
    row = {
        "timestamp_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol, "side": side, "mode": mode, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2 if tp2 else "",
        "stop_type": stop_type,
        "zone": zone, "zones_active": zones, "strength": stars,
        "delta_usd": round(diag.get("delta_usd",0),2),
        "near_bids_usd": round(diag.get("bid_near_usd",0),2),
        "near_asks_usd": round(diag.get("ask_near_usd",0),2),
        "buys_usd": round(diag.get("buys_usd",0),2),
        "sells_usd": round(diag.get("sells_usd",0),2),
        "liq_reason": reason, "mid": round(diag.get("mid",0),6), "src": diag.get("source",""),
        "outcome": "", "hit_tp": 0, "hit_sl": 0, "mae_pct": "", "mfe_pct": "", "mae_atr": "", "mfe_atr": ""
    }
    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:  # <-- add encoding
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists: w.writeheader()
        w.writerow(row)

def load_state():
    try:
        with open(STATE_PATH,"r") as f: return json.load(f)
    except Exception:
        return {"last_alert_ts": {}, "last_alert_key": {}}

def save_state(st):
    tmp = STATE_PATH+".tmp"
    with open(tmp,"w") as f: json.dump(st, f)
    os.replace(tmp, STATE_PATH)

STATE = load_state()

# ========================= Outcome Monitor (thread) =========================
class OutcomeMonitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)

    def run(self):
        while True:
            try:
                self.tick()
            except Exception as e:
                print("[Outcome] error:", e)
            time.sleep(OUTCOME_POLL_SECONDS)

    def tick(self):
    # 1) file presence + read with UTF-8
        if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
            return
        df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
        if df.empty or "timestamp_utc" not in df.columns:
            return

    # 2) keep original timestamp_utc INTACT; parse into a separate column
    # first pass: tolerant parser
        ts1 = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    # second pass only on failures: strip/replace variants (handles 'Z' explicitly)
        bad = ts1.isna()
        if bad.any():
            ts2 = pd.to_datetime(
                df.loc[bad, "timestamp_utc"].astype(str).str.strip().str.replace("Z","", regex=False),
                utc=True, errors="coerce"
            )
            ts1 = ts1.where(~bad, ts2)

    # DO NOT write ts1 back to df["timestamp_utc"]; keep it separate
        df["ts_parsed"] = ts1

    # 3) work only on rows with a valid parsed timestamp AND still open outcome
        df = ensure_log_columns(df)
        # rows are "open" if outcome is missing, NaN, or empty/whitespace
        open_mask = df["outcome"].isna() | df["outcome"].astype(str).str.strip().eq("")

        valid_mask = df["ts_parsed"].notna()
        idxs = df[open_mask & valid_mask].index.tolist()
        if not idxs:
            return

        updated = False
        for idx in idxs:
            row = df.loc[idx]
            sym   = str(row["symbol"])
            side  = str(row["side"]).upper()

        # Safe numeric coercion; skip row if any required field is missing
            try:
                entry = float(row["entry"]); sl = float(row["sl"]); tp1 = float(row["tp1"])
            except Exception:
                continue

        # Use ts_parsed for time math; add a small pre-buffer
            start_ts = pd.to_datetime(row["ts_parsed"], utc=True)
            start    = start_ts - timedelta(minutes=2)

        # Fetch 1m candles and filter from start
            dfc = fetch_candles(sym, OUTCOME_GRANULARITY_SEC)
            if dfc.empty: continue
            dfc = dfc[dfc["time"] >= start]
            if dfc.empty: continue

            highs = dfc["high"].astype(float); lows = dfc["low"].astype(float)

        # First-touch logic
            hit_tp = hit_sl = 0; outcome = "Open"
            for _, c in dfc.iterrows():
                h = float(c["high"]); l = float(c["low"])
                if side == "LONG":
                    if h >= tp1: hit_tp,outcome=1,"TP"; break
                    if l <= sl:  hit_sl,outcome=1,"SL"; break
                else:
                    if l <= tp1: hit_tp,outcome=1,"TP"; break
                    if h >= sl:  hit_sl,outcome=1,"SL"; break
            else:
            # still open—don’t touch this row
                continue

        # MAE/MFE (%)
            if side == "LONG":
                mfe_pct = (highs.max()-entry)/entry*100.0
                mae_pct = (entry-lows.min())/entry*100.0
            else:
                mfe_pct = (entry-lows.min())/entry*100.0
                mae_pct = (highs.max()-entry)/entry*100.0

        # ATR units (1H)
            df1h = fetch_candles(sym, 3600)
            atr1h = atr_wilder_value(df1h, 14) if not df1h.empty else max(1e-9, entry*0.005)
            mfe_atr = (mfe_pct/100.0) * entry / atr1h
            mae_atr = (mae_pct/100.0) * entry / atr1h

        # Write ONLY the outcome-related fields; DO NOT touch timestamp_utc
            df.at[idx,"outcome"] = outcome
            df.at[idx,"hit_tp"]  = hit_tp
            df.at[idx,"hit_sl"]  = hit_sl
            df.at[idx,"mfe_pct"] = round(float(mfe_pct),4)
            df.at[idx,"mae_pct"] = round(float(mae_pct),4)
            df.at[idx,"mfe_atr"] = round(float(mfe_atr),3)
            df.at[idx,"mae_atr"] = round(float(mae_atr),3)
            updated = True

        if updated:
        # Don’t persist ts_parsed to the CSV — keep the original strings
            out = df.drop(columns=["ts_parsed"], errors="ignore")
            tmp = LOG_PATH + ".tmp"
            out.to_csv(tmp, index=False, encoding="utf-8-sig")
            os.replace(tmp, LOG_PATH)
            print("[Outcome] updated", LOG_PATH)


# ========================= Warmup & Regime =========================
def is_global_warmup() -> bool:
    return (time.time() - WS_START_TS) < (WARMUP_HOURS * 3600)

def coin_in_warmup(symbol: str, df_recent_24h: pd.DataFrame) -> bool:
    return int(df_recent_24h[df_recent_24h["symbol"] == symbol]["symbol"].count()) < WARMUP_MIN_ALERTS_PER_COIN

def regime_bias_1d(df1d: pd.DataFrame) -> str:
    closes = df1d["close"].astype(float)
    if len(closes) < 22: return "neutral"
    m = closes.rolling(20).mean(); sd = closes.rolling(20).std()
    bbw = (2*sd)/m
    p = (bbw.rank(pct=True).iloc[-1] if bbw.notna().any() else 0.5)
    if p >= 0.65: return "trend"
    if p <= 0.35: return "range"
    return "neutral"

# ========================= AutoTuner (thread) =========================
class AutoTuner(threading.Thread):
    def __init__(self, thresholds: dict, lock: threading.Lock):
        super().__init__(daemon=True)
        self.thresholds = thresholds
        self.lock = lock
        self.running = True
    def run(self):
        while self.running:
            try: self.tune_once()
            except Exception as e: print("[AUTOTUNE] error:", e)
            time.sleep(AUTOTUNE_INTERVAL_MIN * 60)
    def stop(self): self.running = False
    def tune_once(self):
        if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
            return
        df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
        if "timestamp_utc" not in df.columns or df.empty:
            return
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp_utc"])
        df = ensure_log_columns(df)
        cutoff_24h = pd.Timestamp.utcnow() - timedelta(hours=24)
        rec_24 = df[df["timestamp_utc"] >= cutoff_24h]
        cutoff_wr = pd.Timestamp.utcnow() - timedelta(hours=WINRATE_LOOKBACK_H)
        wr_df = df[df["timestamp_utc"] >= cutoff_wr].copy()
        for col in ("hit_tp","hit_sl"):
            if col in wr_df.columns: wr_df[col] = wr_df[col].fillna(0).astype(int)
        def coin_winrate(g: pd.DataFrame) -> float:
            total = int((g["hit_tp"]==1).sum() + (g["hit_sl"]==1).sum())
            return (int((g["hit_tp"]==1).sum()) / total) if total>0 else float("nan")
        wr = wr_df.groupby("symbol").apply(coin_winrate).rename("winrate").reset_index()
        wr_map = dict(zip(wr["symbol"], wr["winrate"]))
        counts = rec_24.groupby("symbol")["symbol"].count().to_dict()
        global_warm = is_global_warmup()
        updated = False
        with self.lock:
            for coin in COINS:
                cfg  = self.thresholds.get(coin, DEFAULTS.copy()).copy()
                if coin in MAJORS: base = TIER_BASES["MAJORS"]
                elif coin in LIQUID: base = TIER_BASES["LIQUID"]
                elif coin in MIDS:   base = TIER_BASES["MIDS"]
                else:                base = TIER_BASES["SMALLS"]

                if global_warm or coin_in_warmup(coin, rec_24):
                    T, BAND, floor_mult = WARMUP_TARGET_ALERTS_PER_DAY, WARMUP_TARGET_BAND, WARMUP_FLOOR_MULT
                else:
                    T, BAND, floor_mult = TARGET_ALERTS_PER_DAY, TARGET_BAND, 0.50

                lower_cnt, upper_cnt = T / BAND, T * BAND
                c = counts.get(coin, 0)

                if c > upper_cnt:
                    ratio = min(3.0, c / max(1e-9, T))
                    scale = max(AUTOTUNE_MIN_SCALE, min(AUTOTUNE_MAX_SCALE, 1.0 + 0.12 * (ratio - 1)))
                    for k in ("delta_usd","burst_sells_long","burst_buys_short","cluster_min"): cfg[k] *= scale
                    updated = True
                elif c < lower_cnt:
                    need  = T / max(1.0, c if c>0 else 1.0)
                    scale = max(AUTOTUNE_MIN_SCALE, min(AUTOTUNE_MAX_SCALE, 1.0 - 0.10 * (need - 1)))
                    for k in ("delta_usd","burst_sells_long","burst_buys_short","cluster_min"): cfg[k] *= scale
                    updated = True

                wr_val = wr_map.get(coin, float("nan"))
                lower_wr, upper_wr = WINRATE_TARGET - WINRATE_BAND, WINRATE_TARGET + WINRATE_BAND
                if not pd.isna(wr_val):
                    if wr_val < lower_wr:
                        for k in ("delta_usd","burst_sells_long","burst_buys_short","cluster_min"): cfg[k] *= 1.06
                        updated = True
                    elif wr_val > upper_wr:
                        for k in ("delta_usd","burst_sells_long","burst_buys_short","cluster_min"): cfg[k] *= 0.96
                        updated = True

                cfg["delta_usd"]        = max(floor_mult * base["delta_usd"],        cfg["delta_usd"])
                cfg["burst_sells_long"] = max(floor_mult * base["burst_sells_long"], cfg["burst_sells_long"])
                cfg["burst_buys_short"] = max(floor_mult * base["burst_buys_short"], cfg["burst_buys_short"])
                cfg["cluster_min"]      = max(floor_mult * base["cluster_min"],      cfg["cluster_min"])
                self.thresholds[coin] = cfg
        if updated:
            persisted = load_custom_thresholds(THR_PATH)
            for coin in COINS:
                cfg = self.thresholds[coin]
                persisted[coin] = {
                    "delta_usd": cfg["delta_usd"],
                    "burst_sells_long": cfg["burst_sells_long"],
                    "burst_buys_short": cfg["burst_buys_short"],
                    "near_pct": cfg["near_pct"],
                    "thick_ratio": cfg["thick_ratio"],
                    "cluster_min": cfg["cluster_min"],
                    "lookback_s": cfg.get("lookback_s", DEFAULTS["lookback_s"]),
                }
            save_custom_thresholds(THR_PATH, persisted)
            print("[AUTOTUNE] thresholds updated →", THR_PATH)

# ========================= Swing utils =========================
def assign_stars(trade):
    zones = str(trade.get("zones_active","")).split(",")
    zones = [z.strip().upper() for z in zones if z.strip()]
    zone_count = len(zones)
    has_fib = any("FIB" in z for z in zones) or ("FIB" in str(trade.get("zone","")).upper())
    
    if has_fib and zone_count >= 3:
        return "★★★"
    elif has_fib or zone_count >= 3:
        return "★★"
    else:
        return "★"


def find_last_swing_high(df: pd.DataFrame, left=3, right=3, lookback=60) -> float | None:
    highs = df["high"].astype(float).tail(lookback + left + right + 1).reset_index(drop=True)
    idx_end = len(highs) - right - 1
    for i in range(idx_end - 1, left - 1, -1):
        window = highs.iloc[i - left : i + right + 1]
        if highs.iloc[i] == window.max(): return float(highs.iloc[i])
    return None

def find_last_swing_low(df: pd.DataFrame, left=3, right=3, lookback=60) -> float | None:
    lows = df["low"].astype(float).tail(lookback + left + right + 1).reset_index(drop=True)
    idx_end = len(lows) - right - 1
    for i in range(idx_end - 1, left - 1, -1):
        window = lows.iloc[i - left : i + right + 1]
        if lows.iloc[i] == window.min(): return float(lows.iloc[i])
    return None

# ========================= Zone diagnostics =========================
def zones_touching_now(product_id: str) -> list[str]:
    zones_hit = []
    df1h = fetch_candles(product_id, 3600)
    df6h = fetch_candles(product_id, 21600)
    df1d = fetch_candles(product_id, 86400)
    if len(df1h) < 10: return zones_hit
    curr = float(df1h["close"].iloc[-1])

    # FIBs
    lo = df1h["low"].tail(2*24).min(); hi = df1h["high"].tail(2*24).max()
    if price_in_zone(curr, *fib_golden_zone(lo, hi)): zones_hit.append("FIB_TINY")
    lo = df1h["low"].tail(7*24).min(); hi = df1h["high"].tail(7*24).max()
    if price_in_zone(curr, *fib_golden_zone(lo, hi)): zones_hit.append("FIB_SHORT")
    lo = df6h["low"].tail(14*4).min(); hi = df6h["high"].tail(14*4).max()
    if price_in_zone(curr, *fib_golden_zone(lo, hi)): zones_hit.append("FIB_MEDIUM")
    lo = df6h["low"].tail(30*4).min(); hi = df6h["high"].tail(30*4).max()
    if price_in_zone(curr, *fib_golden_zone(lo, hi)): zones_hit.append("FIB_LONG")

    # EMA 1H/1D
    for per, name in [(50,"EMA50_1H"),(200,"EMA200_1H")]:
        if len(df1h)>=per:
            v=float(ema(df1h["close"], per).iloc[-1]); b=(v*(1-EMA_BAND_PCT), v*(1+EMA_BAND_PCT))
            if price_in_zone(curr, *b): zones_hit.append(name)
    for per, name in [(20,"EMA20_1D"),(50,"EMA50_1D"),(100,"EMA100_1D"),(200,"EMA200_1D")]:
        if len(df1d)>=per:
            v=float(ema(df1d["close"], per).iloc[-1]); b=(v*(1-EMA_BAND_PCT), v*(1+EMA_BAND_PCT))
            if price_in_zone(curr, *b): zones_hit.append(name)

    # SWING
    if ENABLE_SWING_ZONES:
        atr1h = atr_wilder_value(df1h)
        hi_s = find_last_swing_high(df1h, SWING_FRAC_LEFT, SWING_FRAC_RIGHT, SWING_LOOKBACK_BARS_1H)
        lo_s = find_last_swing_low(df1h, SWING_FRAC_LEFT, SWING_FRAC_RIGHT, SWING_LOOKBACK_BARS_1H)
        def hb(px): return max(SWING_ZONE_ATR_MULT * atr1h, px * SWING_ZONE_MIN_PCT)
        if hi_s is not None:
            band=(hi_s-hb(hi_s), hi_s+hb(hi_s))
            if price_in_zone(curr, *band): zones_hit.append("SWING_HIGH_ZONE")
        if lo_s is not None:
            band=(lo_s-hb(lo_s), lo_s+hb(lo_s))
            if price_in_zone(curr, *band): zones_hit.append("SWING_LOW_ZONE")

    return zones_hit

def print_zone_diagnostics(coins):
    hits=[]
    for c in coins:
        try:
            z=zones_touching_now(c)
            if z: hits.append(f"{c}({','.join(z)})")
        except Exception: continue
    if hits: print("[Zones]", " | ".join(hits))
    else:    print("[Zones] (none currently)")

def debug_top_candidates(coins):
    rows = []
    for sym in coins:
        try:
            df1h = fetch_candles(sym, 3600)
            if df1h.empty: continue
            curr = float(df1h["close"].iloc[-1])
            lo = df1h["low"].tail(2*24).min(); hi = df1h["high"].tail(2*24).max()
            zmin, zmax = fib_golden_zone(lo, hi)
            if not (zmin <= curr <= zmax): continue

            # 60s ΔCVD (REST for debug)
            # 60s ΔCVD (REST for debug)
            tdf = fetch_trades(sym, limit=250)
            if tdf.empty: continue
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(seconds=60)
            win = tdf[tdf["time"] >= cutoff]
            buys = float(win.loc[win["side"]=="buy","notional"].sum())
            sells= float(win.loc[win["side"]=="sell","notional"].sum())
            delta = buys - sells

# ORDER BOOK — FORCE DICT
            book = ensure_book_dict(fetch_book(sym))
            assert isinstance(book, dict), f"fetch_book() must return dict, got {type(book)}"
            bids, asks = book.get("bids", [])[:150], book.get("asks", [])[:150]
            if not bids or not asks: continue
            mid = 0.5*(bids[0][0] + asks[0][0])

            near_pct = LIQ_THRESHOLDS[sym]["near_pct"]
            def tot(levels):
                lo,hi = mid*(1-near_pct), mid*(1+near_pct)
                return sum(px*sz for px,sz in levels if lo<=px<=hi)
            b_near, a_near = tot(bids), tot(asks)


            rows.append((sym, delta, b_near, a_near, zmin, zmax, curr))
        except Exception:
            continue
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    if rows:
        print("[Debug candidates]")
        for sym, delta, b_near, a_near, zmin, zmax, curr in rows[:DEBUG_TOP_N]:
            print(f"  {sym}: ΔCVD={int(delta):>8}  nearB={int(b_near):>8}  nearA={int(a_near):>8}  band=({fmt_price(zmin)}..{fmt_price(zmax)}) px={fmt_price(curr)}")



# ========================= 70% Allowlist Config =========================
ALLOWLIST_PATH = os.getenv("ALLOWLIST_OVER70_PATH", "allowlist_over70.csv")
ALLOWLIST_REFRESH_SEC = int(os.getenv("ALLOWLIST_REFRESH_SEC", "900"))
ALLOWLIST_MIN_SUPPORT = int(os.getenv("ALLOWLIST_MIN_SUPPORT", "5"))
ALLOWLIST_WINRATE = float(os.getenv("ALLOWLIST_WINRATE", "0.70"))
ALLOWLIST_MAX_COMBO = int(os.getenv("ALLOWLIST_MAX_COMBO", "3"))
NUMERIC_TO_BIN = ["delta_usd","near_bids_usd","near_asks_usd","buys_usd","sells_usd"]

class ComboAllowlist(threading.Thread):
    """Auto-refreshing allowlist of 70%+ raw win combos per coin from the log CSV."""
    def __init__(self, log_path: str, out_path: str):
        super().__init__(daemon=True)
        self.log_path = log_path
        self.out_path = out_path
        self._lock = threading.RLock()
        self._allow = {}
        self._bin_edges = {}
        self._last_build = 0.0

    def should_alert(self, symbol: str, feat: dict) -> tuple[bool, str]:
        with self._lock:
            combos = self._allow.get(symbol, [])
        for combo in combos:
            if all(str(feat.get(f)) == str(v) for (f,v) in combo):
                why = " + ".join([f"{f}={v}" for f,v in combo])
                return True, f"70%+ combo matched: {why}"
        return False, "no 70%+ combo (n≥5)"

    def bin_value(self, symbol: str, key: str, value):
        try:
            x = float(value)
        except Exception:
            return None
        with self._lock:
            cuts = self._bin_edges.get(symbol, {}).get(key)
        if not cuts: return None
        cuts = sorted(cuts)
        if len(cuts) == 1:
            return f"(-inf,{cuts[0]}]" if x <= cuts[0] else f"({cuts[0]},inf)"
        c1, c2 = cuts[0], cuts[1]
        if x <= c1:   return f"(-inf,{c1}]"
        elif x <= c2: return f"({c1},{c2}]"
        else:         return f"({c2},inf)"

    def run(self):
        while True:
            try:
                self._maybe_rebuild()
            except Exception as e:
                print("[ALLOWLIST] error:", e)
            time.sleep(5)

    def _maybe_rebuild(self):
        now = time.time()
        if now - self._last_build < ALLOWLIST_REFRESH_SEC:
            return
        self._rebuild()
        self._last_build = now

    def _rebuild(self):
        if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
            return
        df = pd.read_csv(self.log_path, encoding="utf-8-sig")
        if "symbol" not in df.columns:
            return

        # normalize outcomes
        if "hit_tp" not in df.columns and "outcome" in df.columns:
            df["hit_tp"] = df["outcome"].astype(str).str.upper().eq("TP").astype(int)
        if "hit_sl" not in df.columns and "outcome" in df.columns:
            df["hit_sl"] = df["outcome"].astype(str).str.upper().eq("SL").astype(int)
        for c in ("hit_tp","hit_sl"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        df["is_win"] = np.where(df["hit_tp"]==1, 1, np.where(df["hit_sl"]==1, 0, np.nan))
        closed = df.dropna(subset=["is_win"]).copy()
        if closed.empty: return
        closed["is_win"] = closed["is_win"].astype(int)

        # cat features
        ignore = {"is_win","hit_tp","hit_sl","outcome","entry","tp","tp1","tp2","sl","timestamp","time","signal_time","mid","src"}
        cat_feats = []
        for c in closed.columns:
            if c in ignore or c == "symbol": continue
            s = closed[c]; nun = s.nunique(dropna=True)
            if s.dtype == object or s.dtype == bool or nun <= 12:
                cat_feats.append(c)

        # per-coin tertiles for numeric
        bin_edges = {}
        pieces = []
        for sym, g in closed.groupby("symbol"):
            g = g.copy()
            edges = {}
            for num in NUMERIC_TO_BIN:
                if num not in g.columns: continue
                s = pd.to_numeric(g[num], errors="coerce").dropna()
                if len(s) < 25: continue
                try:
                    cuts = list(pd.Series(s).quantile([1/3, 2/3]).astype(float).values)
                except Exception:
                    continue
                edges[num] = cuts
                def label(v):
                    if pd.isna(v): return None
                    v = float(v)
                    if len(cuts) == 1:
                        return f"(-inf,{cuts[0]}]" if v <= cuts[0] else f"({cuts[0]},inf)"
                    c1, c2 = cuts
                    if v <= c1:   return f"(-inf,{c1}]"
                    elif v <= c2: return f"({c1},{c2}]"
                    else:         return f"({c2},inf)"
                g[f"bin_{num}"] = pd.to_numeric(g[num], errors="coerce").map(label)
                if f"bin_{num}" not in cat_feats:
                    cat_feats.append(f"bin_{num}")
            pieces.append(g); bin_edges[sym] = edges

        if not pieces: return
        cb = pd.concat(pieces, ignore_index=True)

        # prune
        cat_pruned = []
        for c in cat_feats:
            nun = cb[c].nunique(dropna=True)
            if nun == 0 or nun > 20: continue
            if cb[c].notna().sum() < 10: continue
            cat_pruned.append(c)

        rows = []
        for sym, g in cb.groupby("symbol"):
            if len(g) < 10: continue
            feats = [c for c in cat_pruned if g[c].notna().sum() >= ALLOWLIST_MIN_SUPPORT]
            for k in range(1, ALLOWLIST_MAX_COMBO+1):
                for fs in combinations(feats, k):
                    grp = g.groupby(list(fs), dropna=False)["is_win"].agg(count="size", win_rate="mean").reset_index()
                    ok = grp[(grp["count"] >= ALLOWLIST_MIN_SUPPORT) & (grp["win_rate"] > ALLOWLIST_WINRATE)]
                    if ok.empty: continue
                    for _, r in ok.iterrows():
                        combo = tuple((f, r[f]) for f in fs)
                        rows.append({
                            "symbol": sym,
                            "combo_size": k,
                            "features": ";".join(f for f,_ in combo),
                            "values":   ";".join(str(v) for _,v in combo),
                            "count": int(r["count"]),
                            "win_rate": float(r["win_rate"])
                        })
        allow_df = pd.DataFrame(rows).sort_values(["win_rate","count","combo_size"], ascending=[False, False, True])

        # persist
        try:
            tmp = self.out_path + ".tmp"
            allow_df.to_csv(tmp, index=False, encoding="utf-8-sig")
            os.replace(tmp, self.out_path)
            print(f"[ALLOWLIST] saved {len(allow_df)} rows → {self.out_path}")
        except Exception as e:
            print("[ALLOWLIST] save failed:", e)

        # in-memory
        allow_map = {}
        for _, r in allow_df.iterrows():
            sym = r["symbol"]
            feats = r["features"].split(";")
            vals  = r["values"].split(";")
            combo = tuple(zip(feats, vals))
            allow_map.setdefault(sym, []).append(combo)

        with self._lock:
            self._allow = allow_map
            self._bin_edges = bin_edges

# Global allowlist instance (started in __main__)
ALLOW = ComboAllowlist(LOG_PATH, ALLOWLIST_PATH)
# ========================= Check Signal =========================
def check_signal(product_id: str, cooldowns: dict, last_alert_key: dict):
    # === Exit-tuning flags (dynamic by default) ===
    USE_FIXED_TP1   = False   # keep dynamic TP unless you flip this
    USE_SL_CAP      = False   # keep dynamic SL unless you flip this
    TP1_PCT         = 0.028   # used only if USE_FIXED_TP1 = True  (~+2.8%)
    SL_PCT_MAX      = 0.008   # used only if USE_SL_CAP = True    (~0.8%)

    # Candles
    df1h = fetch_candles(product_id, 3600)
    df6h = fetch_candles(product_id, 21600)
    df1d = fetch_candles(product_id, 86400)
    if len(df1h) < 7*24 + 2:
        return None

    curr, prev = float(df1h["close"].iloc[-1]), float(df1h["close"].iloc[-2])
    df1h_atr = atr_wilder_value(df1h)
    regime = regime_bias_1d(df1d)

    # -----------------------
    # Build zones & candidates
    # -----------------------
    entries=[]; zone_flags={}
    # FIBs
    lo = df1h["low"].tail(2*24).min(); hi = df1h["high"].tail(2*24).max()
    zmin, zmax = fib_golden_zone(lo, hi); side = detect_reversal(prev, curr, zmin, zmax)
    entries.append(("FIB_TINY", lo, hi, zmin, zmax, side)); zone_flags["FIB_TINY"] = price_in_zone(curr, zmin, zmax)

    lo = df1h["low"].tail(7*24).min(); hi = df1h["high"].tail(7*24).max()
    zmin, zmax = fib_golden_zone(lo, hi); side = detect_reversal(prev, curr, zmin, zmax)
    entries.append(("FIB_SHORT", lo, hi, zmin, zmax, side)); zone_flags["FIB_SHORT"] = price_in_zone(curr, zmin, zmax)

    lo = df6h["low"].tail(14*4).min(); hi = df6h["high"].tail(14*4).max()
    zmin, zmax = fib_golden_zone(lo, hi); side = detect_reversal(prev, curr, zmin, zmax)
    entries.append(("FIB_MEDIUM", lo, hi, zmin, zmax, side)); zone_flags["FIB_MEDIUM"] = price_in_zone(curr, zmin, zmax)

    lo = df6h["low"].tail(30*4).min(); hi = df6h["high"].tail(30*4).max()
    zmin, zmax = fib_golden_zone(lo, hi); side = detect_reversal(prev, curr, zmin, zmax)
    entries.append(("FIB_LONG", lo, hi, zmin, zmax, side)); zone_flags["FIB_LONG"] = price_in_zone(curr, zmin, zmax)

    # SWING zones
    if ENABLE_SWING_ZONES:
        hi_s = find_last_swing_high(df1h, SWING_FRAC_LEFT, SWING_FRAC_RIGHT, SWING_LOOKBACK_BARS_1H)
        lo_s = find_last_swing_low(df1h,  SWING_FRAC_LEFT, SWING_FRAC_RIGHT, SWING_LOOKBACK_BARS_1H)
        def hb(px): return max(SWING_ZONE_ATR_MULT * df1h_atr, px * SWING_ZONE_MIN_PCT)
        if hi_s is not None:
            band=(hi_s-hb(hi_s), hi_s+hb(hi_s)); side=detect_reversal(prev, curr, *band)
            entries.append(("SWING_HIGH_ZONE", hi_s, hi_s, band[0], band[1], side)); zone_flags["SWING_HIGH_ZONE"]=price_in_zone(curr,*band)
        if lo_s is not None:
            band=(lo_s-hb(lo_s), lo_s+hb(lo_s)); side=detect_reversal(prev, curr, *band)
            entries.append(("SWING_LOW_ZONE", lo_s, lo_s, band[0], band[1], side)); zone_flags["SWING_LOW_ZONE"]=price_in_zone(curr,*band)

    # EMA 1H
    for per, name in [(50,"EMA50_1H"), (200,"EMA200_1H")]:
        if len(df1h) >= per:
            val = float(ema(df1h["close"], per).iloc[-1])
            zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
            side = detect_reversal(prev, curr, zmin, zmax)
            entries.append((name, val, val, zmin, zmax, side)); zone_flags[name] = price_in_zone(curr, zmin, zmax)

    # EMA 1D
    for per, name in [(20,"EMA20_1D"),(50,"EMA50_1D"),(100,"EMA100_1D"),(200,"EMA200_1D")]:
        if len(df1d) >= per:
            val = float(ema(df1d["close"], per).iloc[-1])
            zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
            side = detect_reversal(prev, curr, zmin, zmax)
            entries.append((name, val, val, zmin, zmax, side)); zone_flags[name] = price_in_zone(curr, zmin, zmax)

    # Keep only directional
    entries = [e for e in entries if e[5] is not None]
    if not entries:
        return None

    # Priority
    priority = {
        "EMA200_1D":12,"EMA100_1D":11,"EMA50_1D":10,"EMA20_1D":9,
        "FIB_LONG":8,"FIB_MEDIUM":7,
        "SWING_HIGH_ZONE":6,"SWING_LOW_ZONE":6,
        "EMA200_1H":5,"EMA50_1H":4,
        "FIB_SHORT":3,"FIB_TINY":2
    }
    entries.sort(key=lambda e: priority.get(e[0], 0), reverse=True)
    zone_source, lo, hi, zmin, zmax, side = entries[0]
    zone_count = sum(1 for k,v in zone_flags.items() if v)
    zones_list = ", ".join([k for k,v in zone_flags.items() if v])

    # --------------
    # Cooldown/dedupe
    # --------------
    now = datetime.now(timezone.utc)
    last = cooldowns.get(product_id)
    if last and (now - last).total_seconds() < COOLDOWN_MINUTES*60:
        return None

    edge_now = zone_edge(curr, zmin, zmax)
    key = f"{product_id}|{zone_source}|{edge_now}|{side}"
    if last_alert_key.get(product_id) == key:
        return None

    # -------------------------
    # Liquidity confirm (gate)
    # -------------------------
    liq_ok, liq_reason, liq_diag = liquidity_confirm(product_id, side)
    if not liq_ok:
        return None

    delta_usd = float(liq_diag.get("delta_usd",0.0))
    try:
        MARKET_FLOW.register(product_id, delta_usd)
    except Exception:
        pass
    thr = LIQ_THRESHOLDS[product_id]["delta_usd"]

    # Candidate from liquidity (existing behavior)
    candidate_mode = choose_mode(liq_reason, delta_usd, edge_now)

    # Strong continuation evidence for FIB/SWING
    def strong_continuation_evidence():
        big_flow = abs(delta_usd) >= 1.75 * thr
        edge_align_long  = (delta_usd > 0) and (edge_now in {"upper","middle"})
        edge_align_short = (delta_usd < 0) and (edge_now in {"lower","middle"})
        edge_ok = edge_align_long or edge_align_short
        regime_ok = (regime == "trend")
        weekend_ok = (not WEEKEND_DISABLE_CONT) or (not is_weekend_utc(datetime.now(timezone.utc)))
        return big_flow and edge_ok and regime_ok and weekend_ok

    is_fib   = zone_source.startswith("FIB")
    is_swing = zone_source.startswith("SWING")
    is_ema   = zone_source.startswith("EMA")

    # ---------------------
    # Final mode selection
    # ---------------------
    if is_ema:
        mode = "CONTINUATION"
    elif is_fib or is_swing:
        if candidate_mode == "CONTINUATION" and strong_continuation_evidence():
            mode = "CONTINUATION"
        else:
            mode = "REVERSAL"
    else:
        mode = candidate_mode

    if ENABLE_CONTINUATION and mode == "CONTINUATION":
        if (delta_usd >= 0 and edge_now in {"upper","middle"}):
            side = "LONG"
        elif (delta_usd < 0 and edge_now in {"lower","middle"}):
            side = "SHORT"
        if regime == "range" and abs(delta_usd) < thr * 1.25:
            return None
        if WEEKEND_DISABLE_CONT and is_weekend_utc(datetime.now(timezone.utc)):
            return None

    # ----------
    # Book guard
    # ----------
    b_near, a_near, mid = books_stream.sum_nearby(product_id, pct=LIQ_THRESHOLDS[product_id]["near_pct"])
    if mid is None: mid = curr
    if b_near==0 and a_near==0:
        return None

    # --- Flow features & quintiles (annotation only; NO gating) ---
    abs_delta   = abs(float(liq_diag.get("delta_usd", 0.0)))
    thr_norm    = max(1.0, LIQ_THRESHOLDS[product_id]["delta_usd"])
    delta_ratio = abs_delta / thr_norm
    flow_aligned = (delta_usd > 0 and side=="LONG") or (delta_usd < 0 and side=="SHORT")
    book_aligned = (b_near > a_near) if side=="LONG" else (a_near > b_near)

    # refresh quintile edges every 5 minutes
    # refresh quintile edges every 5 minutes
    if ("quintile_edges" not in STATE) or (now.timestamp() - STATE.get("q_last_ts", 0)) > 300:
        g_edges, pc_edges = _load_quintile_edges(LOG_PATH, lookback_h=int(os.getenv("Q_LOOKBACK_H","168")))
    # >>> FIX: coerce numpy arrays to JSON-safe lists
        def _tolist(x):
            import numpy as np
            if x is None: return None
            if isinstance(x, np.ndarray): return x.tolist()
            try:
            # handles sequences like array-like or list of numpy scalars
                return [float(v) for v in x]
            except Exception:
                return x

        g_edges = _tolist(g_edges)
        pc_edges = {} if not pc_edges else {sym: _tolist(edges) for sym, edges in pc_edges.items()}

        STATE["quintile_edges"] = {"global": g_edges, "per_coin": pc_edges}
        STATE["q_last_ts"] = float(now.timestamp())
        save_state(STATE)


    edges = STATE.get("quintile_edges", {})
    g_edges = edges.get("global")
    pc_edges = edges.get("per_coin", {})
    abs_delta_q_global = _assign_quintile(abs_delta, g_edges)
    abs_delta_q_coin   = _assign_quintile(abs_delta, pc_edges.get(product_id))

    # ----
    # Exits (dynamic by default)
    # ----
    style = get_styles_for(product_id, zone_source).get("style","SCALP")
    entry = curr; tp2_price = None; stop_type = "ZONE/ATR"

    if mode == "CONTINUATION":
        tp_liq, sl_liq = find_cluster_tp_sl(product_id, side, mid, LIQ_THRESHOLDS[product_id], books_stream, bias_pct=0.004)
        if tp_liq and sl_liq:
            tp1_price = tp_liq; sl_price = sl_liq; stop_type = "LIQ"
        else:
            if side=="LONG": tp1_price = entry + 2.0*df1h_atr; sl_price = entry - 1.2*df1h_atr
            else:            tp1_price = entry - 2.0*df1h_atr; sl_price = entry + 1.2*df1h_atr
            stop_type = "ATR"
        msg_plan = f"CONTINUATION • TP {'↑' if side=='LONG' else '↓'}@liquidity (fallback ATR) | stop:{stop_type}"

    else:  # REVERSAL
        params = get_styles_for(product_id, zone_source)
        if zone_source.startswith("EMA") and style == "SCALP":
            tp_pct = params.get("tp_pct",0.9); sl_pct=params.get("sl_pct",0.7); timebox_h=params.get("timebox_h",4)
            if side=="LONG": tp1_price=entry*(1+tp_pct/100); sl_price=entry*(1-sl_pct/100)
            else:            tp1_price=entry*(1-tp_pct/100); sl_price=entry*(1+sl_pct/100)
            msg_plan = f"SCALP • TP {tp_pct:.2f}% | SL {sl_pct:.2f}% | {timebox_h}h"
        else:
            tp1_pct = params.get("tp1_pct",0.7); tp2_pct=params.get("tp2_pct",1.6); timebox_h=params.get("timebox_h",6); sl_mode=params.get("sl_mode","zone")
            if side=="LONG":
                tp1_price=entry*(1+tp1_pct/100); tp2_price=entry*(1+tp2_pct/100)
                sl_price = entry*(1-params.get("atr_sl_pct",0.70)/100) if sl_mode=="atr" else zmin*(1-SL_BUFFER)
            else:
                tp1_price=entry*(1-tp1_pct/100); tp2_price=entry*(1-tp2_pct/100)
                sl_price = entry*(1+params.get("atr_sl_pct",0.70)/100) if sl_mode=="atr" else zmax*(1+SL_BUFFER)
            msg_plan = f"REVERSAL • TP1 {tp1_pct:.2f}% + TP2 {tp2_pct:.2f}% | SL_{sl_mode.upper()} | {timebox_h}h"

    # Optional overrides (OFF by default)
    if USE_FIXED_TP1:
        tp2_price = None
        tp1_price = entry * (1 + TP1_PCT) if side == "LONG" else entry * (1 - TP1_PCT)

    if USE_SL_CAP:
        if side == "LONG":
            sl_cap = entry * (1 - SL_PCT_MAX)
            sl_price = max(sl_price, sl_cap)
        else:
            sl_cap = entry * (1 + SL_PCT_MAX)
            sl_price = min(sl_price, sl_cap)

    # -----------------
    # Stars (Fib/Conf)
    # -----------------
    active_fib = any(("FIB" in k) and v for k, v in zone_flags.items()) or ("FIB" in zone_source)
    confluence = zone_count
    if active_fib and confluence >= 3:
        stars = "★★★"
    elif active_fib or confluence >= 3:
        stars = "★★☆"
    else:
        stars = "★☆☆"

    # -------------
    # Alert message
    # -------------
    msg = f"{product_id} | Entry: ${fmt_price(entry)} | TP: ${fmt_price(tp1_price)} | SL: ${fmt_price(sl_price)}"

    # Log & cooldown

    # Log & cooldown
    # -------------
    # add the flow/quintile features into liq_diag so they persist to CSV
    liq_diag["abs_delta"] = abs_delta
    liq_diag["delta_ratio"] = round(delta_ratio, 3)
    liq_diag["abs_delta_q_global"] = abs_delta_q_global
    liq_diag["abs_delta_q_coin"] = abs_delta_q_coin
    liq_diag["flow_aligned"] = int(bool(flow_aligned))
    liq_diag["book_aligned"] = int(bool(book_aligned))

    stop_type = "LIQ" if "liquidity" in msg_plan.lower() else stop_type
    log_trade(product_id, side, mode, entry, sl_price, tp1_price, tp2_price, stop_type, zone_source, zones_list, stars, liq_reason, liq_diag, now)

    cooldowns[product_id] = now
    last_alert_key[product_id] = key
    STATE["last_alert_ts"][product_id] = now.timestamp()
    STATE["last_alert_key"][product_id] = key
    save_state(STATE)
    # -------- Features for 70% allowlist --------
    feat = {
        "side": side,
        "mode": mode,
        "stop_type": stop_type,
        "zone": zone_source,
        "strength": stars,
        "liq_reason": liq_reason or "",
    }
    # binned numerics (per-coin tertiles)
    feat["bin_delta_usd"]     = ALLOW.bin_value(product_id, "delta_usd", abs_delta) if 'abs_delta' in locals() else None
    feat["bin_buys_usd"]      = ALLOW.bin_value(product_id, "buys_usd",  liq_diag.get("buys_usd") if 'liq_diag' in locals() else None)
    feat["bin_sells_usd"]     = ALLOW.bin_value(product_id, "sells_usd", liq_diag.get("sells_usd") if 'liq_diag' in locals() else None)
    feat["bin_near_bids_usd"] = ALLOW.bin_value(product_id, "near_bids_usd", liq_diag.get("bid_near_usd") if 'liq_diag' in locals() else None)
    feat["bin_near_asks_usd"] = ALLOW.bin_value(product_id, "near_asks_usd", liq_diag.get("ask_near_usd") if 'liq_diag' in locals() else None)

    return msg, feat




# ========================= Status / Commands =========================
def handle_command(cmd: str):
    if not ENABLE_STATUS_CMDS: return None
    cmd = cmd.strip()
    if cmd == "!status":
        hits=[]
        for c in COINS:
            try:
                z=zones_touching_now(c)
                if z: hits.append(f"{c}({','.join(z)})")
            except Exception: pass
        body = "[Status]\n" + (" | ".join(hits) if hits else "(no coins in zones)")
        return body
    if cmd.startswith("!thresholds"):
        parts = cmd.split()
        if len(parts)==2 and parts[1] in LIQ_THRESHOLDS:
            th = LIQ_THRESHOLDS[parts[1]]
            return f"[Thresholds {parts[1]}]\nΔ:{fmt_dollars(th['delta_usd'])}  burstL:{fmt_dollars(th['burst_sells_long'])} burstS:{fmt_dollars(th['burst_buys_short'])}\nnear_pct:{th['near_pct']:.3%}  thick_ratio:{th['thick_ratio']:.2f}  cluster_min:{fmt_dollars(th['cluster_min'])}"
        else:
            return "Usage: !thresholds COIN (e.g., !thresholds XRP/USD)"
    return None


def _stoch_gate_rev(product_id: str, side_txt: str):
    # Returns: allowed(bool), k15,d15,k1h,d1h, juice(bool for SHORTs)
    try:
        df1h = fetch_candles(product_id, 3600)
        df15 = fetch_candles(product_id, 900)
        if len(df1h) < 30 or len(df15) < 30:
            return False, float("nan"), float("nan"), float("nan"), float("nan"), False
        k1h, d1h = stoch_rsi_kd(df1h["close"])
        k15, d15 = stoch_rsi_kd(df15["close"])
        allowed = False
        juice = False
        if side_txt == "LONG":
            # Gate: 1h oversold + (15m K>D or 15m in Mid) and NOT 15m oversold
            cond1 = (k1h < 20.0)
            cond2 = (k15 >= 20.0)  # avoid oversold on 15m
            cond3 = (k15 > d15) or (20.0 <= k15 <= 80.0)
            allowed = bool(cond1 and cond2 and cond3)
        elif side_txt == "SHORT":
            # Gate: require 15m K < D
            allowed = bool(k15 < d15)
            # Juice: if 1h supportive (K<D) or 1h Mid (20-80)
            juice = bool((k1h < d1h) or (20.0 <= k1h <= 80.0))
        return allowed, k15, d15, k1h, d1h, juice
    except Exception:
        return False, float("nan"), float("nan"), float("nan"), float("nan"), False

# ========================= Main Loop =========================

def main_loop():
    # Make sure log header exists
    init_log_file()
    cooldowns = {}  # coin -> last alert time
    last_health = 0.0

    while True:
        start = time.time()
        for coin in COINS:
            try:
                out = check_signal(coin, cooldowns, STATE.get("last_alert_key", {}))
                if out:
                    alert, feat = out
                    mode_txt = _extract_mode(alert)
                    if mode_txt == "REVERSAL":
                        side_txt = _extract_side(alert) or ""
                        allowed, k15, d15, k1h, d1h, juice = _stoch_gate_rev(coin, side_txt)
                        stoch_line = f"\nStoch: 15m K/D={k15:.1f}/{d15:.1f}  1h K/D={k1h:.1f}/{d1h:.1f}  Gate:{'PASS' if allowed else 'MUTED'}"
                        if side_txt == "SHORT" and juice:
                            stoch_line += "  Juice:+"
                        alert = alert + stoch_line
                        if allowed:
                            ok, why = ALLOW.should_alert(coin, feat)
                            if ok:
                                if MARKET_AGREEMENT_ENABLED:
                                    ok_m, why_m = MARKET_FLOW.gate(side_txt)
                                    if not ok_m:
                                        print(f"[MUTED MARKET] {coin} {side_txt}: {why_m}")
                                        continue
                                    alert = alert + f"\n🧭 {why_m}"
                                send_telegram(alert + f"\n🔒 {why}")
                            else:
                                print(f"[MUTED 70%] {coin} {side_txt}: {why}")
                    else:
                        if not (MUTE_CONTINUATION_ALERTS and mode_txt == "CONTINUATION"):
                            ok, why = ALLOW.should_alert(coin, feat)
                            if ok:
                                side_txt2 = _extract_side(alert) or ""
                                if MARKET_AGREEMENT_ENABLED and side_txt2:
                                    ok_m, why_m = MARKET_FLOW.gate(side_txt2)
                                    if not ok_m:
                                        print(f"[MUTED MARKET] {coin} {side_txt2}: {why_m}")
                                        continue
                                    alert = alert + f"\n🧭 {why_m}"
                                send_telegram(alert + f"\n🔒 {why}")
                            else:
                                print(f"[MUTED 70%] {coin}: {why}")
            except Exception as e:
                print("[Loop] error on", coin, ":", e)

        # health ping
        if time.time() - last_health > 120:
            print("[health] alive", datetime.utcnow().strftime("%H:%M:%S"), flush=True)
            last_health = time.time()

        # pacing
        dt = time.time() - start
        if dt < INTERVAL_SECONDS:
            time.sleep(INTERVAL_SECONDS - dt)

# ========================= Entrypoint =========================
if __name__ == "__main__":
    if not os.path.exists(STATE_PATH):
        save_state(STATE)

    # Start the outcome updater so open trades get updated in the CSV/Excel
    OutcomeMonitor().start()

    # (Optional) also start the AutoTuner to keep thresholds fresh)
    tuner = AutoTuner(LIQ_THRESHOLDS, threading.Lock())
    
    # Start 70% allowlist refresher
    ALLOW.start()
    tuner.start()

    # Kraken WS started in class instantiation
    main_loop()
