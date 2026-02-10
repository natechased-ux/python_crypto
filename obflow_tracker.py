
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obflow_tracker.py
-----------------
Purpose: Track orderbook-flow and context features to CSV every 5 minutes.
NO alerts, NO trading logic — just logging for later analysis.

What it logs per symbol (row):
- ID = "<symbol>|<timestamp_utc>"
- timestamp_utc
- price (latest close from 1m/5m/15m fallback)
- Orderflow lookbacks (net taker notional: buys_usd - sells_usd):
    lookback_5m, lookback_15m, lookback_30min, lookback_1hr, lookback_4hr
- Cumulative orderflow across ALL COINS (same windows):
    ..._cum
- Ratios: symbol_flow / cumulative_flow for each window (NaN if denom≈0)
- Level proximity flags (1 if |price-level|/price <= 0.25%):
    EMAs (1H: 10/20/50/200; 1D: 20/50/100/200),
    top-of-book (best bid/ask),
    last swing high/low (1H),
    Fib zones (Tiny/Short/Medium/Long): flags for near_min, near_max, and in_zone
- Stoch RSI (K,D) for 1H and 15m
- Future returns (percent from entry) backfilled when enough time has passed:
    ret_15m, ret_30m, ret_1h, ret_2h, ret_4h

Config:
- TRACK_SYMBOLS: env CSV (default: ["XRP/USD"])
- COINS for cumulative: same as in source script or env COINS_CSV
- LOG_CSV: path (default: "obflow_log.csv")
- PING_EVERY_SEC: default 300 (5 minutes)

Dependencies:
    pip install pandas requests python-dateutil websocket-client
"""
import os, csv, time, json, threading, collections, math
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import requests
import urllib.parse as _url

# --------------------- Config ---------------------
KRAKEN_BASE = "https://api.kraken.com"

# Universe for cumulative market flow (can be wider than TRACK_SYMBOLS)
DEFAULT_COINS = [
    "XBT/USD","ETH/USD","SOL/USD","XRP/USD","MATIC/USD","ADA/USD","AVAX/USD","DOGE/USD","DOT/USD",
    "LINK/USD","ATOM/USD","NEAR/USD","ARB/USD","OP/USD","INJ/USD","AAVE/USD","LTC/USD","BCH/USD",
    "ETC/USD","ALGO/USD","FIL/USD","ICP/USD","RNDR/USD","STX/USD","GRT/USD","SEI/USD","HBAR/USD",
    "JUP/USD","TIA/USD","UNI/USD","MKR/USD","FET/USD","CRV/USD","SAND/USD","MANA/USD","AXS/USD",
    "PEPE/USD","XLM/USD"
]
COINS = [s.strip() for s in os.getenv("COINS_CSV", ",".join(DEFAULT_COINS)).split(",") if s.strip()]

# Symbols to actively TRACK (rows written for these)
TS_ENV = os.getenv("TRACK_SYMBOLS", "").strip()
TRACK_SYMBOLS = [s.strip() for s in TS_ENV.split(",") if s.strip()] if TS_ENV else COINS.copy()

COINS = [s.strip() for s in os.getenv("COINS_CSV", ",".join(DEFAULT_COINS)).split(",") if s.strip()]

LOG_CSV = os.getenv("LOG_CSV", "obflow_log.csv")
PING_EVERY_SEC = int(os.getenv("PING_EVERY_SEC", "300"))  # 5 minutes
NEAR_PCT = float(os.getenv("NEAR_PCT", "0.0025"))  # 0.25%

# ------------------ Helpers reused ------------------
def _kr_pair_for_rest(pair: str) -> str:
    return pair.replace("/", "")

def fetch_candles(product_id: str, granularity: int) -> pd.DataFrame:
    """
    Kraken OHLC: /0/public/OHLC?pair=XBTUSD&interval=60
    Normalize to columns: ["time","low","high","open","close","volume"], ascending by time.
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
    if df.shape[1] == 8:
        df.columns = ["time","open","high","low","close","vwap","volume","count"]
    elif df.shape[1] == 7:
        df.columns = ["time","open","high","low","close","volume","count"]
        df["vwap"] = pd.NA
    else:
        while df.shape[1] < 8:
            df[df.shape[1]] = pd.NA
        df = df.iloc[:, :8]
        df.columns = ["time","open","high","low","close","vwap","volume","count"]
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["time","low","high","open","close","volume"]].sort_values("time").reset_index(drop=True)

def fetch_trades(product_id: str, limit: int = 1000) -> pd.DataFrame:
    """
    Kraken Trades: /0/public/Trades?pair=XBTUSD
    Return normalized columns: ["price","volume","time","side","ordertype","misc","notional"]
    side: 'buy' or 'sell', time is UTC ts.
    """
    pair_q = _kr_pair_for_rest(product_id)
    url = f"{KRAKEN_BASE}/0/public/Trades?pair={_url.quote(pair_q)}"
    r = requests.get(url, timeout=12); r.raise_for_status()
    res = r.json().get("result", {})
    arr = next((v for k, v in res.items() if k != "last"), None)
    if not arr:
        return pd.DataFrame(columns=["price","volume","time","side","ordertype","misc","notional"])
    # Trim if too long
    if len(arr) > limit:
        arr = arr[-limit:]
    first = arr[0] if len(arr) else []
    n = len(first)
    if n >= 7:
        cols = ["price","volume","time","side","ordertype","misc","trade_id"]
        df = pd.DataFrame(arr, columns=cols[:n])
    elif n == 6:
        cols = ["price","volume","time","side","ordertype","misc"]
        df = pd.DataFrame(arr, columns=cols)
    else:
        df = pd.DataFrame(arr)
        while df.shape[1] < 6:
            df[df.shape[1]] = pd.NA
        df = df.iloc[:, :6]
        df.columns = ["price","volume","time","side","ordertype","misc"]
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["side"] = df["side"].map({"b":"buy","s":"sell"}).fillna("buy")
    df["notional"] = df["price"] * df["volume"]
    return df.sort_values("time").reset_index(drop=True)

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi_wilder(series: pd.Series, length: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    delta = s.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0.0, pd.NA)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(method="bfill").fillna(50.0)

def stoch_rsi_kd(series_close: pd.Series, rsi_len: int = 14, stoch_len: int = 14, k_len: int = 3, d_len: int = 3) -> Tuple[float,float]:
    rsi = rsi_wilder(series_close, rsi_len)
    low = rsi.rolling(stoch_len, min_periods=max(2, stoch_len//2)).min()
    high = rsi.rolling(stoch_len, min_periods=max(2, stoch_len//2)).max()
    denom = (high - low)
    stoch = ((rsi - low) / denom.where(denom != 0, pd.NA)).fillna(0.5)
    k = stoch.rolling(k_len, min_periods=1).mean() * 100.0
    d = k.rolling(d_len, min_periods=1).mean()
    return float(k.iloc[-1]), float(d.iloc[-1])

def atr_wilder_df(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    tr = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

# ---------- Feature builders ----------
def _fib_golden_zone(lo: float, hi: float, tol: float = NEAR_PCT) -> Tuple[float,float]:
    r618 = lo + 0.618 * (hi - lo)
    r660 = lo + 0.660 * (hi - lo)
    zmin, zmax = min(r618, r660), max(r618, r660)
    pad = ((zmin+zmax)/2) * tol
    return zmin - pad, zmax + pad

def _price_near_level(price: float, level: float, pct: float = NEAR_PCT) -> int:
    if not (np.isfinite(price) and np.isfinite(level)):
        return 0
    return int(abs(price - level) / price <= pct)

def _calc_flows_for_windows(sym: str, windows_s: Dict[str,int]) -> Dict[str, float]:
    """Return net delta (buys_usd - sells_usd) for each window for a symbol.
    Single trades fetch, then slice for each window."""
    tdf = fetch_trades(sym, limit=2000)
    out = {k: 0.0 for k in windows_s}
    if tdf.empty or "time" not in tdf:
        return out
    now = pd.Timestamp.utcnow()
    for name, sec in windows_s.items():
        cutoff = now - pd.Timedelta(seconds=sec)
        win = tdf[tdf["time"] >= cutoff]
        buys = float(win.loc[win["side"]=="buy","notional"].sum())
        sells = float(win.loc[win["side"]=="sell","notional"].sum())
        out[name] = buys - sells
    return out

def _cumulative_flows(windows_s: Dict[str,int]) -> Dict[str, float]:
    """Sum net flow across ALL COINS for each window."""
    sums = {k: 0.0 for k in windows_s}
    now = pd.Timestamp.utcnow()
    # Fetch per-coin once and aggregate
    for coin in COINS:
        tdf = fetch_trades(coin, limit=2000)
        if tdf.empty: 
            continue
        for name, sec in windows_s.items():
            cutoff = now - pd.Timedelta(seconds=sec)
            win = tdf[tdf["time"] >= cutoff]
            buys = float(win.loc[win["side"]=="buy","notional"].sum())
            sells = float(win.loc[win["side"]=="sell","notional"].sum())
            sums[name] += (buys - sells)
    return sums

def _last_price(sym: str) -> Optional[float]:
    # Try 60s, fallback to 300/900
    for g in (60, 300, 900):
        df = fetch_candles(sym, g)
        if not df.empty:
            return float(df["close"].iloc[-1])
    return None

def _level_flags_and_stoch(sym: str, price: float) -> Dict[str, int | float]:
    out: Dict[str, int | float] = {}
    df1h = fetch_candles(sym, 3600)
    df1d = fetch_candles(sym, 86400)
    df15 = fetch_candles(sym, 900)

    # EMAs 1H
    for per in (10,20,50,200):
        key = f"near_ema{per}_1h"
        if len(df1h) >= per:
            val = float(ema(df1h["close"], per).iloc[-1])
            out[key] = _price_near_level(price, val)
        else:
            out[key] = 0

    # EMAs 1D
    for per in (20,50,100,200):
        key = f"near_ema{per}_1d"
        if len(df1d) >= per:
            val = float(ema(df1d["close"], per).iloc[-1])
            out[key] = _price_near_level(price, val)
        else:
            out[key] = 0

    # Top of book levels as "nodes"
    bid_top = ask_top = None
    try:
        pair_q = _kr_pair_for_rest(sym)
        url = f"{KRAKEN_BASE}/0/public/Depth?pair={_url.quote(pair_q)}&count=1"
        r = requests.get(url, timeout=10); r.raise_for_status()
        res = r.json().get("result", {})
        dp = next((v for k, v in res.items()), None) or {}
        bids = dp.get("bids", [])
        asks = dp.get("asks", [])
        if bids:
            bid_top = float(bids[0][0])
        if asks:
            ask_top = float(asks[0][0])
    except Exception:
        pass
    out["near_top_bid"] = _price_near_level(price, bid_top) if bid_top is not None else 0
    out["near_top_ask"] = _price_near_level(price, ask_top) if ask_top is not None else 0

    # Swing highs/lows (1H)
    def _find_last_swing(series: pd.Series, left=3, right=3) -> Tuple[Optional[float], Optional[float]]:
        highs = df1h["high"].astype(float)
        lows = df1h["low"].astype(float)
        if len(highs) < left+right+1:
            return None, None
        # scan from back
        idx_end = len(highs) - right - 1
        hi_val = lo_val = None
        for i in range(idx_end - 1, left - 1, -1):
            window_h = highs.iloc[i-left:i+right+1]
            if highs.iloc[i] == window_h.max():
                hi_val = float(highs.iloc[i]); break
        for i in range(idx_end - 1, left - 1, -1):
            window_l = lows.iloc[i-left:i+right+1]
            if lows.iloc[i] == window_l.min():
                lo_val = float(lows.iloc[i]); break
        return hi_val, lo_val

    hi_s, lo_s = (None, None)
    if not df1h.empty:
        hi_s, lo_s = _find_last_swing(df1h["close"])
    out["near_swing_high_1h"] = _price_near_level(price, hi_s) if hi_s is not None else 0
    out["near_swing_low_1h"]  = _price_near_level(price, lo_s) if lo_s is not None else 0

    # Fib zones (Tiny: last 2d 1H; Short: last 7d 1H; Medium: 14*6H; Long: 30*6H)
    def _near_fib_flags(df_lohi: pd.DataFrame, lo_bars: int, hi_bars: int, key: str):
        if df_lohi.empty:
            out[f"near_{key}_min"] = 0
            out[f"near_{key}_max"] = 0
            out[f"in_{key}_zone"] = 0
            return
        lo = float(df_lohi["low"].tail(lo_bars).min())
        hi = float(df_lohi["high"].tail(hi_bars).max())
        zmin, zmax = _fib_golden_zone(lo, hi, tol=NEAR_PCT)
        out[f"near_{key}_min"] = _price_near_level(price, zmin)
        out[f"near_{key}_max"] = _price_near_level(price, zmax)
        out[f"in_{key}_zone"] = int(zmin <= price <= zmax)

    if not df1h.empty:
        _near_fib_flags(df1h, 2*24, 2*24, "fib_tiny")
        _near_fib_flags(df1h, 7*24, 7*24, "fib_short")
    df6h = fetch_candles(sym, 21600)
    if not df6h.empty:
        # 14*6H and 30*6H windows
        _near_fib_flags(df6h, 14*4, 14*4, "fib_medium")
        _near_fib_flags(df6h, 30*4, 30*4, "fib_long")

    # Stoch RSI
    if len(df1h) >= 30:
        k1h, d1h = stoch_rsi_kd(df1h["close"])
    else:
        k1h = d1h = float("nan")
    if len(df15) >= 30:
        k15, d15 = stoch_rsi_kd(df15["close"])
    else:
        k15 = d15 = float("nan")
    out["stoch_k_1h"], out["stoch_d_1h"] = k1h, d1h
    out["stoch_k_15m"], out["stoch_d_15m"] = k15, d15
    return out

# ---------- CSV I/O ----------
LOG_COLUMNS = [
    "ID","timestamp_utc","symbol","price",
    "lookback_5m","lookback_15m","lookback_30min","lookback_1hr","lookback_4hr",
    "lookback_5m_cum","lookback_15m_cum","lookback_30min_cum","lookback_1hr_cum","lookback_4hr_cum",
    "lookback_5m_ratio","lookback_15m_ratio","lookback_30min_ratio","lookback_1hr_ratio","lookback_4hr_ratio",
    # level flags
    "near_ema10_1h","near_ema20_1h","near_ema50_1h","near_ema200_1h",
    "near_ema20_1d","near_ema50_1d","near_ema100_1d","near_ema200_1d",
    "near_top_bid","near_top_ask",
    "near_swing_high_1h","near_swing_low_1h",
    "near_fib_tiny_min","near_fib_tiny_max","in_fib_tiny_zone",
    "near_fib_short_min","near_fib_short_max","in_fib_short_zone",
    "near_fib_medium_min","near_fib_medium_max","in_fib_medium_zone",
    "near_fib_long_min","near_fib_long_max","in_fib_long_zone",
    # stoch
    "stoch_k_1h","stoch_d_1h","stoch_k_15m","stoch_d_15m",
    # future returns (to be backfilled)
    "ret_15m","ret_30m","ret_1h","ret_2h","ret_4h"
]

def _init_csv():
    if not os.path.exists(LOG_CSV) or os.path.getsize(LOG_CSV) == 0:
        with open(LOG_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(LOG_COLUMNS)

def _append_row(row: Dict[str, object]):
    exists = os.path.exists(LOG_CSV) and os.path.getsize(LOG_CSV) > 0
    with open(LOG_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if not exists: w.writeheader()
        w.writerow(row)

# ---------- Backfill Thread ----------
class Backfill(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)

    def run(self):
        while True:
            try:
                self._tick()
            except Exception as e:
                print("[Backfill] error:", e)
            time.sleep(60)

    def _tick(self):
        if not os.path.exists(LOG_CSV) or os.path.getsize(LOG_CSV) == 0:
            return
        df = pd.read_csv(LOG_CSV, encoding="utf-8-sig")
        if df.empty: return
        # Parse ts
        df["ts"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        now = pd.Timestamp.utcnow()
        horizons = {
            "ret_15m": 15, "ret_30m": 30, "ret_1h": 60, "ret_2h": 120, "ret_4h": 240
        }
        # For rows needing backfill and with elapsed >= horizon, fill return = (close(t+H) - entry)/entry*100
        updated = False
        for idx, row in df.iterrows():
            entry_px = row.get("price", np.nan)
            if not np.isfinite(entry_px):
                continue
            ts = row.get("ts")
            sym = str(row.get("symbol", ""))
            if not (pd.notna(ts) and sym):
                continue
            for col, mins in horizons.items():
                if pd.notna(row.get(col)) and str(row.get(col)) != "":
                    continue  # already filled
                if now < ts + pd.Timedelta(minutes=mins):
                    continue  # not yet eligible
                # fetch candles at a reasonable granularity for the horizon
                gran = 60 if mins <= 60 else 300 if mins <= 240 else 900
                cdf = fetch_candles(sym, gran)
                if cdf.empty: 
                    continue
                # find the first candle whose time >= ts + mins
                target = ts + pd.Timedelta(minutes=mins)
                sub = cdf[cdf["time"] >= target]
                if sub.empty:
                    # use last available
                    last_close = float(cdf["close"].iloc[-1])
                else:
                    last_close = float(sub["close"].iloc[0])
                ret_pct = (last_close - entry_px) / entry_px * 100.0
                df.at[idx, col] = round(float(ret_pct), 5)
                updated = True
        if updated:
            out = df.drop(columns=["ts"], errors="ignore")
            tmp = LOG_CSV + ".tmp"
            out.to_csv(tmp, index=False, encoding="utf-8-sig")
            os.replace(tmp, LOG_CSV)
            print("[Backfill] returns updated")

# ---------- Main loop ----------
def _one_tick():
    windows_s = {
        "lookback_5m": 5*60,
        "lookback_15m": 15*60,
        "lookback_30min": 30*60,
        "lookback_1hr": 60*60,
        "lookback_4hr": 4*60*60,
    }
    # Precompute cumulative flows over all coins once per tick
    cum = _cumulative_flows(windows_s)

    for sym in TRACK_SYMBOLS:
        try:
            price = _last_price(sym)
            if price is None:
                print(f"[skip] no price for {sym}")
                continue
            # per-symbol flows
            flows = _calc_flows_for_windows(sym, windows_s)
            ratios = {}
            for k in windows_s:
                denom = cum.get(k, 0.0)
                ratios[k + "_ratio"] = (flows.get(k, 0.0) / denom) if abs(denom) > 1e-9 else np.nan

            # levels + stoch
            levels = _level_flags_and_stoch(sym, price)

            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            row = {
                "ID": f"{sym}|{now}",
                "timestamp_utc": now,
                "symbol": sym,
                "price": round(float(price), 8),
                # flows
                "lookback_5m": round(float(flows["lookback_5m"]), 2),
                "lookback_15m": round(float(flows["lookback_15m"]), 2),
                "lookback_30min": round(float(flows["lookback_30min"]), 2),
                "lookback_1hr": round(float(flows["lookback_1hr"]), 2),
                "lookback_4hr": round(float(flows["lookback_4hr"]), 2),
                "lookback_5m_cum": round(float(cum["lookback_5m"]), 2),
                "lookback_15m_cum": round(float(cum["lookback_15m"]), 2),
                "lookback_30min_cum": round(float(cum["lookback_30min"]), 2),
                "lookback_1hr_cum": round(float(cum["lookback_1hr"]), 2),
                "lookback_4hr_cum": round(float(cum["lookback_4hr"]), 2),
                "lookback_5m_ratio": ratios["lookback_5m_ratio"],
                "lookback_15m_ratio": ratios["lookback_15m_ratio"],
                "lookback_30min_ratio": ratios["lookback_30min_ratio"],
                "lookback_1hr_ratio": ratios["lookback_1hr_ratio"],
                "lookback_4hr_ratio": ratios["lookback_4hr_ratio"],
                # level flags
                "near_ema10_1h": levels["near_ema10_1h"],
                "near_ema20_1h": levels["near_ema20_1h"],
                "near_ema50_1h": levels["near_ema50_1h"],
                "near_ema200_1h": levels["near_ema200_1h"],
                "near_ema20_1d": levels["near_ema20_1d"],
                "near_ema50_1d": levels["near_ema50_1d"],
                "near_ema100_1d": levels["near_ema100_1d"],
                "near_ema200_1d": levels["near_ema200_1d"],
                "near_top_bid": levels["near_top_bid"],
                "near_top_ask": levels["near_top_ask"],
                "near_swing_high_1h": levels["near_swing_high_1h"],
                "near_swing_low_1h": levels["near_swing_low_1h"],
                "near_fib_tiny_min": levels["near_fib_tiny_min"],
                "near_fib_tiny_max": levels["near_fib_tiny_max"],
                "in_fib_tiny_zone": levels["in_fib_tiny_zone"],
                "near_fib_short_min": levels["near_fib_short_min"],
                "near_fib_short_max": levels["near_fib_short_max"],
                "in_fib_short_zone": levels["in_fib_short_zone"],
                "near_fib_medium_min": levels["near_fib_medium_min"],
                "near_fib_medium_max": levels["near_fib_medium_max"],
                "in_fib_medium_zone": levels["in_fib_medium_zone"],
                "near_fib_long_min": levels["near_fib_long_min"],
                "near_fib_long_max": levels["near_fib_long_max"],
                "in_fib_long_zone": levels["in_fib_long_zone"],
                "stoch_k_1h": round(float(levels["stoch_k_1h"]), 4) if np.isfinite(levels["stoch_k_1h"]) else "",
                "stoch_d_1h": round(float(levels["stoch_d_1h"]), 4) if np.isfinite(levels["stoch_d_1h"]) else "",
                "stoch_k_15m": round(float(levels["stoch_k_15m"]), 4) if np.isfinite(levels["stoch_k_15m"]) else "",
                "stoch_d_15m": round(float(levels["stoch_d_15m"]), 4) if np.isfinite(levels["stoch_d_15m"]) else "",
                # future returns placeholders
                "ret_15m": "", "ret_30m": "", "ret_1h": "", "ret_2h": "", "ret_4h": ""
            }
            _append_row(row)
            print(f"[log] {sym} @ {now}")
        except Exception as e:
            print("[tick] error for", sym, ":", e)

def main():
    _init_csv()
    # Start backfill thread
    Backfill().start()

    print(f"[tracker] running. Tracking {len(TRACK_SYMBOLS)} symbols:", ", ".join(TRACK_SYMBOLS))
    while True:
        t0 = time.time()
        try:
            _one_tick()
        except Exception as e:
            print("[main] tick error:", e)
        # pacing
        dt = time.time() - t0
        if dt < PING_EVERY_SEC:
            time.sleep(PING_EVERY_SEC - dt)

if __name__ == "__main__":
    main()
