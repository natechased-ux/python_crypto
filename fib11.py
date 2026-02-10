#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Order-Flow + Level Touch (Enhanced Logging, Complete)
----------------------------------------------------
- Uses curated 38-coin list (same as before) on Kraken spot.
- Multi-lookback order-flow deltas: 60s, 5m, 15m, 30m, 1h.
- Triggers when strongest lookback delta exceeds threshold AND price is inside a key level band:
  * Fibonacci golden zones (tiny=2d, short=7d)
  * EMA bands (1H: 50/200, 1D: 20/50/100/200)
  * Volume nodes from recent trades: VPOC + top HVNs
- Opens **virtual trade** (no TP/SL yet). Every 5m for 2h, writes sample row with ret%, MAE/MFE.
- At entry, logs:
  * Full lookback flow deltas (buys/sells/delta)
  * Snapshot of **all key levels** (FIB low/high/mid, EMA mids, VPOC/HVNs)
  * Meta: best lookback, best delta, distances to every level (pct & ATR), BB-width & percentile,
    day-of-week/hour, weekend flag.

CSV columns:
  timestamp_utc, symbol, side, level, entry, atr_1h,
  lookbacks_json, levels_json, meta_json,
  sample_ts, sample_px, ret_pct, mae_pct, mfe_pct
"""

import os, time, json, math, threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import requests
import pandas as pd
import numpy as np

# ======================== Config ========================
SYMBOLS = [
    "XBT/USD","ETH/USD",
    "SOL/USD","XRP/USD","MATIC/USD","ADA/USD","AVAX/USD","DOGE/USD","DOT/USD",
    "LINK/USD","ATOM/USD","NEAR/USD","ARB/USD","OP/USD",
    "INJ/USD","AAVE/USD","LTC/USD","BCH/USD","ETC/USD","ALGO/USD","FIL/USD","ICP/USD",
    "RNDR/USD","STX/USD","GRT/USD","SEI/USD","SUI/USD","APT/USD","TIA/USD",
    "UNI/USD","MKR/USD","FET/USD","CRV/USD","SAND/USD","MANA/USD","AXS/USD","PEPE/USD","XLM/USD",
]

LOOKBACKS_S = [60, 300, 900, 1800, 3600]
LEVEL_TOL_PCT = 0.005     # ±0.5% around a level
EMA_BAND_PCT  = 0.005     # ±0.5% around EMA mid

EMA_PERIODS_1H = [50, 200]
EMA_PERIODS_1D = [20, 50, 100, 200]

PROFILE_LOOKBACK_H = 6
PROFILE_BIN_PCT = 0.001
TOP_HVNS = 3

ATR_LEN = 14

TRACK_WINDOW_MIN = 120
SAMPLE_EVERY_MIN = 5
COOLDOWN_MIN = 30

# ---- Multi-lookback thresholds (JSON) ----
import json, math, os

THRESH_JSON_PATH = os.getenv("FLOW_THRESHOLDS_PATH", "thresholds_multi_lookback.json")

def _load_thresholds(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

THRESH = _load_thresholds(THRESH_JSON_PATH)

# Fallbacks if a symbol/lookback/key is missing in JSON
DEFAULT_DELTA_USD = 2500.0
def threshold_for(symbol: str, lb_s: int, key: str = "delta_usd") -> float:
    sym_cfg = THRESH.get(symbol, {})
    per_lb = sym_cfg.get("thresholds", {}).get(str(lb_s), {})
    val = per_lb.get(key)
    if val is None:
        # fallback to base (60s) scaled ~sqrt(time)
        base = sym_cfg.get("thresholds", {}).get("60", {}).get(key, DEFAULT_DELTA_USD)
        val = float(base) * math.sqrt(lb_s / 60.0)
    return float(val)

def thick_ratio_for(symbol: str, default: float = 1.6) -> float:
    sym_cfg = THRESH.get(symbol, {})
    return float(sym_cfg.get("thick_ratio", default))


TIER = {
    "MAJORS": {"set": {"XBT/USD", "ETH/USD"}, "delta_usd": 12000},
    "LIQUID": {"set": {"SOL/USD", "XRP/USD", "LINK/USD"}, "delta_usd": 4000},
}
DEFAULT_DELTA_USD = 2500

CSV_PATH = os.getenv("FLOW_LEVELS_LOG", "live_flow_levels_log.csv")
KRAKEN_BASE = "https://api.kraken.com"

# ======================== Utils & Fetch ========================

def fmt_px(x: float) -> str:
    if x >= 100: return f"{x:.2f}"
    if x >= 10:  return f"{x:.3f}"
    if x >= 1:   return f"{x:.4f}"
    if x >= 0.1: return f"{x:.5f}"
    return f"{x:.9f}"


def _kr_pair_for_rest(pair: str) -> str:
    return pair.replace("/", "")


def fetch_candles(symbol: str, granularity_s: int) -> pd.DataFrame:
    itv_map = {60:1, 300:5, 900:15, 1800:30, 3600:60, 21600:240, 86400:1440}
    interval = itv_map.get(granularity_s, 60)
    pair_q = _kr_pair_for_rest(symbol)
    url = f"{KRAKEN_BASE}/0/public/OHLC?pair={pair_q}&interval={interval}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    res = r.json().get("result", {})
    arr = next((v for k, v in res.items() if k != "last"), [])
    if not arr:
        return pd.DataFrame(columns=["time","low","high","open","close","volume"])
    df = pd.DataFrame(arr)
    if df.shape[1] == 8:
        df.columns = ["time","open","high","low","close","vwap","volume","count"]
    elif df.shape[1] == 7:
        df.columns = ["time","open","high","low","close","volume","count"]
        df["vwap"] = np.nan
    else:
        while df.shape[1] < 8:
            df[df.shape[1]] = np.nan
        df = df.iloc[:, :8]
        df.columns = ["time","open","high","low","close","vwap","volume","count"]
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("time").reset_index(drop=True)[["time","low","high","open","close","volume"]]


def fetch_trades(symbol: str) -> pd.DataFrame:
    pair_q = _kr_pair_for_rest(symbol)
    url = f"{KRAKEN_BASE}/0/public/Trades?pair={pair_q}"
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    res = r.json().get("result", {})
    arr = next((v for k, v in res.items() if k != "last"), [])
    if not arr:
        return pd.DataFrame(columns=["time","price","volume","side","notional"]) 
    first = arr[0]
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
            df[df.shape[1]] = np.nan
        df = df.iloc[:, :6]
        df.columns = ["price","volume","time","side","ordertype","misc"]
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["side"] = df["side"].map({"b":"buy","s":"sell"}).fillna("buy")
    df["notional"] = df["price"] * df["volume"]
    return df.dropna(subset=["time","price","volume"]).sort_values("time").reset_index(drop=True)

# ======================== Indicators & Levels ========================

def ema(series: pd.Series, length: int) -> pd.Series:
    return pd.Series(series, dtype=float).ewm(span=length, adjust=False).mean()


def atr_wilder(df: pd.DataFrame, length: int = 14) -> float:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    tr = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1/length, adjust=False).mean().iloc[-1])


def fib_golden_zone(lo: float, hi: float, tol_pct: float) -> Tuple[float,float]:
    r618 = lo + 0.618*(hi-lo)
    r660 = lo + 0.66*(hi-lo)
    zmin, zmax = min(r618, r660), max(r618, r660)
    pad = ((zmin+zmax)/2.0)*tol_pct
    return zmin - pad, zmax + pad


def bb_width(df: pd.DataFrame, length: int = 20, mult: float = 2.0) -> pd.Series:
    close = df["close"].astype(float)
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    upper = ma + mult*sd
    lower = ma - mult*sd
    width = (upper - lower) / ma
    return width


def build_volume_profile(trades: pd.DataFrame, bin_pct: float) -> Tuple[float, List[Tuple[float,float]]]:
    if trades.empty:
        return np.nan, []
    px = trades["price"].astype(float).to_numpy()
    notional = trades["notional"].astype(float).to_numpy()
    mid = float(np.nanmedian(px))
    width = max(1e-9, bin_pct * mid)
    bins = np.round(px / width) * width
    bucket = {}
    for b, n in zip(bins, notional):
        bucket[b] = bucket.get(b, 0.0) + float(n)
    items = sorted(bucket.items(), key=lambda x: x[1], reverse=True)
    if not items:
        return np.nan, []
    vpoc_price = float(items[0][0])
    hvns = [(float(p), float(v)) for p, v in items[1:1+TOP_HVNS]]
    return vpoc_price, hvns


def flow_deltas(trades: pd.DataFrame, lookbacks_s: List[int]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str,float]] = {}
    if trades.empty:
        for lb in lookbacks_s:
            out[lb] = {"buys":0.0, "sells":0.0, "delta":0.0, "n":0}
        return out
    now = trades["time"].max()
    for lb in lookbacks_s:
        cut = now - pd.Timedelta(seconds=lb)
        win = trades[trades["time"] >= cut]
        buys  = float(win.loc[win["side"]=="buy", "notional"].sum())
        sells = float(win.loc[win["side"]=="sell","notional"].sum())
        out[lb] = {"buys":buys, "sells":sells, "delta":buys-sells, "n": int(len(win))}
    return out


def ema_levels(df1h: pd.DataFrame, df1d: pd.DataFrame) -> List[Tuple[str, Tuple[float,float]]]:
    levels = []
    if len(df1h) > max(EMA_PERIODS_1H):
        closes = df1h["close"].astype(float)
        for p in EMA_PERIODS_1H:
            val = float(ema(closes, p).iloc[-1])
            levels.append((f"EMA{p}_1H", (val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT))))
    if len(df1d) > max(EMA_PERIODS_1D):
        closes = df1d["close"].astype(float)
        for p in EMA_PERIODS_1D:
            val = float(ema(closes, p).iloc[-1])
            levels.append((f"EMA{p}_1D", (val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT))))
    return levels


def fib_levels(df1h: pd.DataFrame) -> List[Tuple[str, Tuple[float,float]]]:
    levels = []
    if len(df1h) < 48:
        return levels
    lo = df1h["low"].tail(48).min(); hi = df1h["high"].tail(48).max()
    levels.append(("FIB_TINY", fib_golden_zone(lo, hi, LEVEL_TOL_PCT)))
    lo = df1h["low"].tail(24*7).min(); hi = df1h["high"].tail(24*7).max()
    levels.append(("FIB_SHORT", fib_golden_zone(lo, hi, LEVEL_TOL_PCT)))
    return levels


def node_levels(vpoc: float, hvns: List[Tuple[float,float]]) -> List[Tuple[str, Tuple[float,float]]]:
    out = []
    if not math.isnan(vpoc):
        out.append(("VPOC", (vpoc*(1-LEVEL_TOL_PCT), vpoc*(1+LEVEL_TOL_PCT))))
    for i, (p, _v) in enumerate(hvns, start=1):
        out.append((f"HVN{i}", (p*(1-LEVEL_TOL_PCT), p*(1+LEVEL_TOL_PCT))))
    return out


def price_in_band(px: float, band: Tuple[float,float]) -> bool:
    lo, hi = band
    return (lo <= px <= hi)


def delta_threshold_usd(symbol: str) -> float:
    for tier in TIER.values():
        if symbol in tier["set"]:
            return float(tier["delta_usd"])
    return float(DEFAULT_DELTA_USD)

# ======================== Trade State & Logging ========================

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class VirtualTrade:
    symbol: str
    side: str
    level: str
    entry_ts: pd.Timestamp
    entry_px: float
    atr_1h: float
    deltas: Dict[int, float]
    levels_at_entry: Dict[str, float]
    meta_at_entry: Dict[str, object]          # if you’re using this
    track_until: pd.Timestamp

    # NEW: add these three
    best_lb: int
    best_delta: float
    threshold_usd: float

    samples: List[Dict] = field(default_factory=list)
    mae_pct: float = 0.0
    mfe_pct: float = 0.0


OPEN: Dict[str, VirtualTrade] = {}
LAST_TRADE_TS: Dict[str, float] = {}
LOCK = threading.Lock()


# at top
import csv

def ensure_csv_header():
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp_utc","symbol","side","level","entry","atr_1h",
                "best_lb","best_delta","threshold_usd",              # <— add these
                "lookbacks_json","levels_json","meta_json",
                "sample_ts","sample_px","ret_pct","mae_pct","mfe_pct"
            ])

def append_sample(tr: VirtualTrade, sample_px: float, ts: pd.Timestamp):
    ret_pct = (sample_px - tr.entry_px)/tr.entry_px*100.0 if tr.side == "LONG" else (tr.entry_px - sample_px)/tr.entry_px*100.0
    tr.mfe_pct = max(tr.mfe_pct, ret_pct)
    tr.mae_pct = min(tr.mae_pct, ret_pct)
    row = [
        tr.entry_ts.strftime('%Y-%m-%d %H:%M:%S'), tr.symbol, tr.side, tr.level, f"{tr.entry_px:.8f}", f"{tr.atr_1h:.8f}",
        tr.best_lb, f"{tr.best_delta:.2f}", f"{getattr(tr, 'threshold_usd', float('nan')):.2f}",
        json.dumps(tr.deltas), json.dumps(tr.levels_at_entry), json.dumps(tr.meta_at_entry),
        ts.strftime('%Y-%m-%d %H:%M:%S'), f"{sample_px:.8f}", f"{ret_pct:.5f}", f"{tr.mae_pct:.5f}", f"{tr.mfe_pct:.5f}"
    ]
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(row)


# ======================== Core Signal Logic ========================

def detect_and_open(symbol: str):
    # Cooldown
    now_ts = time.time()
    if symbol in LAST_TRADE_TS and (now_ts - LAST_TRADE_TS[symbol]) < COOLDOWN_MIN*60:
        return

    # Levels context & current price
    df1h = fetch_candles(symbol, 3600)
    df1d = fetch_candles(symbol, 86400)
    if len(df1h) < 30:
        return
    last_px = float(df1h["close"].iloc[-1])

    # Trades for flow + profile
    trades = fetch_trades(symbol)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=PROFILE_LOOKBACK_H)
    prof_trades = trades[trades["time"] >= cutoff]
    deltas = flow_deltas(trades, LOOKBACKS_S)

    vpoc, hvns = build_volume_profile(prof_trades, PROFILE_BIN_PCT)

    # Levels
    fibs = fib_levels(df1h)
    emas = ema_levels(df1h, df1d)
    nodes = node_levels(vpoc, hvns)
    levels = fibs + emas + nodes

    hits = [(name, band) for name, band in levels if price_in_band(last_px, band)]
    if not hits:
        return

    # Flow gate using strongest lookback
    # pick strongest lookback by absolute delta (unchanged)
    best_lb, best_delta = max(deltas.items(), key=lambda kv: abs(kv[1]["delta"]))
    best_delta_usd = float(best_delta["delta"])

# ⬇️ NEW: per-symbol, per-lookback threshold from JSON
    thr = threshold_for(symbol, best_lb, key="delta_usd")   # from thresholds_multi_lookback.json
    if abs(best_delta_usd) < thr:
        return  # not strong enough

# side follows the sign of the strongest lookback delta
    side = "LONG" if best_delta_usd > 0 else "SHORT"


    priority = {"VPOC":100, "HVN1":99, "HVN2":98, "HVN3":97,
                "EMA200_1D":90, "EMA100_1D":89, "EMA50_1D":88, "EMA20_1D":87,
                "EMA200_1H":80,  "EMA50_1H":79,
                "FIB_SHORT":60,  "FIB_TINY":50}
    hits.sort(key=lambda t: priority.get(t[0], 0), reverse=True)
    level_name, _band = hits[0]

    # ATR
    atr1h = atr_wilder(df1h, ATR_LEN)

    # Build level snapshot
    levels_at_entry: Dict[str, float] = {}
    for name, (lo_b, hi_b) in fibs:
        levels_at_entry[f"{name}_low"] = float(lo_b)
        levels_at_entry[f"{name}_high"] = float(hi_b)
        levels_at_entry[f"{name}_mid"] = float((lo_b + hi_b)/2.0)
    for name, (lo_b, hi_b) in emas:
        levels_at_entry[name] = float((lo_b + hi_b)/2.0)
    if not math.isnan(vpoc):
        levels_at_entry["VPOC"] = float(vpoc)
    for i, (p, _v) in enumerate(hvns, start=1):
        levels_at_entry[f"HVN{i}"] = float(p)

    # Meta snapshot: flow breakdown, best lb, regime flags, distances
    bbw = bb_width(df1h, length=20, mult=2.0)
    if len(bbw.dropna()) >= 20:
        last_bbw = float(bbw.iloc[-1])
        pctile = float((bbw.rank(pct=True).iloc[-1]))
    else:
        last_bbw, pctile = float('nan'), float('nan')

    now = pd.Timestamp.utcnow()
    dow = int(now.dayofweek)
    hour = int(now.hour)
    is_weekend = dow >= 5

    dist_pct = {}
    dist_atr = {}
    for k, v in levels_at_entry.items():
        d = (last_px - v)/v if v else float('nan')
        dist_pct[k] = float(d*100.0)
        dist_atr[k] = float(((last_px - v)/atr1h) if atr1h else float('nan'))

    meta_at_entry = {
        "best_lb_s": int(best_lb),
        "best_delta_usd": float(best_delta["delta"]),
        "flow_breakdown": {int(lb): {"buys": d["buys"], "sells": d["sells"], "delta": d["delta"], "n": d["n"]} for lb, d in deltas.items()},
        "bb_width": last_bbw,
        "bb_width_pctile": pctile,
        "day_of_week": dow,
        "hour_utc": hour,
        "is_weekend": is_weekend,
        "dist_pct": dist_pct,
        "dist_atr": dist_atr,
        "threshold_usd": float(thr),

    }

    vt = VirtualTrade(
        symbol=symbol,
        side=side,
        level=level_name,
        entry_ts=pd.Timestamp.utcnow(),
        entry_px=last_px,
        atr_1h=atr1h,
        deltas={lb: v["delta"] for lb, v in deltas.items()},
        levels_at_entry=levels_at_entry,
        meta_at_entry=meta_at_entry,
        track_until=pd.Timestamp.utcnow() + pd.Timedelta(minutes=TRACK_WINDOW_MIN),
        best_lb=int(best_lb),
        best_delta=best_delta_usd,
        threshold_usd=float(thr),
    )
    with LOCK:
        OPEN[symbol] = vt
        LAST_TRADE_TS[symbol] = time.time()
    append_sample(vt, last_px, pd.Timestamp.utcnow())
    print(f"[OPEN] {symbol} {side} @ {fmt_px(last_px)} | trigger Δ{best_lb}s={best_delta['delta']:.0f} USD | level={level_name}")

# ======================== Sampler & Main ========================

class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
    def run(self):
        while True:
            try:
                self.tick()
            except Exception as e:
                print("[Sampler] error:", e)
            time.sleep(SAMPLE_EVERY_MIN * 60)
    def tick(self):
        with LOCK:
            items = list(OPEN.items())
        if not items:
            return
        for sym, tr in items:
            df1m = fetch_candles(sym, 60)
            if df1m.empty:
                continue
            px = float(df1m["close"].iloc[-1])
            append_sample(tr, px, pd.Timestamp.utcnow())
            if pd.Timestamp.utcnow() >= tr.track_until:
                with LOCK:
                    OPEN.pop(sym, None)
                print(f"[CLOSE] {sym} tracked {TRACK_WINDOW_MIN}m | MFE={tr.mfe_pct:.2f}% MAE={tr.mae_pct:.2f}%")


def main():
    ensure_csv_header()
    sampler = Sampler()
    sampler.start()
    last_health = 0.0
    while True:
        t0 = time.time()
        for sym in SYMBOLS:
            try:
                detect_and_open(sym)
            except Exception as e:
                print("[Loop]", sym, "err:", e)
        if time.time() - last_health > 120:
            print("[health] ok")
            last_health = time.time()
        dt = time.time() - t0
        time.sleep(max(0.0, 10 - dt))

if __name__ == "__main__":
    main()
