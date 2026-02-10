#!/usr/bin/env python3
# probe_logger.py
# Data-only probe (no alerts, no TP/SL):
# - Logs candidate setups with rich context (EMAs, Stoch RSI 1h/15m, full Fib retrace/extension confluence,
#   optional volume-profile nodes)
# - Adds order-flow telemetry:
#     • Order book snapshot near mid: near_bids_usd, near_asks_usd, book_ratio
#     • Rolling ΔCVD (signed USD taker flow) over a short window, normalized: delta_usd_window, burst_window_sec,
#       delta_ratio, abs_delta_q_global (qG), abs_delta_q_coin (qC), flow_ok, book_ok
# - Records future prices/returns at 15m/30m/1h/2h/3h/4h after entry

import os, time, math, random
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pandas.api.types import is_integer_dtype, is_float_dtype

# =========================
# CONFIG (edit as you like)
# =========================
CSV_PATH           = os.getenv("PROBE_CSV", "probe_log.csv")
EXCHANGE_ID        = os.getenv("PROBE_EXCHANGE", "coinbase")   # "coinbase", "kraken", etc.
COINS              = os.getenv("PROBE_COINS",
                     "BTC/USD,ETH/USD,XRP/USD,SOL/USD,HBAR/USD,LINK/USD,ADA/USD,AVAX/USD,LTC/USD,XLM/USD").split(",")
SCAN_EVERY_SEC     = int(os.getenv("PROBE_SCAN_SEC", "60"))    # scan frequency
COOLDOWN_MIN_SEC   = int(os.getenv("PROBE_COOLDOWN_SEC", "600"))  # per-coin min spacing between entries

# Volume profile (liquidity nodes)
USE_NODE_SCORE     = os.getenv("USE_NODE_SCORE", "true").lower() == "true"
VP_LOOKBACK_DAYS   = int(os.getenv("VP_LOOKBACK_DAYS", "5"))
VP_BIN_PCT         = float(os.getenv("VP_BIN_PCT", "0.002"))   # ~0.2%

# Fib confluence band
FIB_PCT_BAND       = float(os.getenv("FIB_PCT_BAND", "0.0018")) # ~0.18%
FIB_ATR_K          = float(os.getenv("FIB_ATR_K", "0.25"))       # 0.25 * ATR(1h)
EMA_BAND_PCT       = float(os.getenv("EMA_BAND_PCT", "0.004"))   # ±0.4% EMA band (wider samples)

# Order book snapshot (±pct around mid to sum)
BOOK_NEAR_PCT      = float(os.getenv("BOOK_NEAR_PCT", "0.0075"))  # 0.75% around mid

# Rolling ΔCVD / trades
BURST_WINDOW_SEC   = int(os.getenv("BURST_WINDOW_SEC", "90"))      # flow window
BASE_BURST_WINDOW_SEC = int(os.getenv("BASE_BURST_WINDOW_SEC", "45"))  # baseline for delta_ratio normalization
TRADES_LIMIT       = int(os.getenv("TRADES_LIMIT", "1000"))        # per fetchTrades call
FLOW_LOOKBACK_H    = int(os.getenv("FLOW_LOOKBACK_H", "168"))      # qG/qC lookback over log (hours)

# Optional perp/spot tilt (basis/funding). Leave blank to skip.
PERP_EXCHANGE      = os.getenv("PERP_EXCHANGE", "")  # e.g., "binanceusdm", "krakenfutures"
PERP_SUFFIX        = os.getenv("PERP_SUFFIX", "USDT") # used to map "BTC/USDT:USDT" on some venues

# Price horizons to record (minutes)
HORIZONS_MIN = [15, 30, 60, 120, 180, 240]

# Libs
try:
    import ccxt  # type: ignore
except Exception:
    raise SystemExit("Please: pip install ccxt pandas numpy")

# =================
# Exchange clients
# =================
def ccxt_client(exid: str):
    ex = getattr(ccxt, exid)({"enableRateLimit": True})
    try: ex.load_markets()
    except Exception: pass
    return ex

EX = ccxt_client(EXCHANGE_ID)
PERP = ccxt_client(PERP_EXCHANGE) if PERP_EXCHANGE else None

def _norm_symbol(s: str, ex=None) -> str:
    s = s.replace("-", "/").upper().strip()
    base, quote = (s.split("/") + ["USD"])[:2]
    alias = {"XBT":"BTC","XDG":"DOGE"}
    base = alias.get(base, base); quote = alias.get(quote, quote)
    norm = f"{base}/{quote}"
    ex = ex or EX
    if hasattr(ex, "markets") and ex.markets:
        if norm not in ex.markets and quote == "USD" and f"{base}/USDT" in ex.markets:
            norm = f"{base}/USDT"
    return norm

# =================
# OHLCV utilities
# =================
def fetch_ohlcv(ex, sym: str, tf: str, since_ms: int, limit: int=1500) -> pd.DataFrame:
    for _ in range(7):
        try:
            ohlcv = ex.fetch_ohlcv(sym, timeframe=tf, since=since_ms, limit=limit)
            break
        except (ccxt.DDoSProtection, ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.ExchangeNotAvailable):
            time.sleep(0.8 + random.uniform(0, 0.2))
        except Exception:
            ohlcv = []
            break
    if not ohlcv:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])
    return pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])

def resample(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if df.empty: return df
    rule = {"1m":"1min","5m":"5min","15m":"15min","30m":"30min","1h":"1h","6h":"6h","1d":"1d"}[target]
    d = df.copy()
    # auto-detect time unit
    if is_integer_dtype(d["time"]) or is_float_dtype(d["time"]):
        d["time"] = pd.to_datetime(d["time"], unit="ms", utc=True)
    else:
        d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.set_index("time")
    o = d["open"].resample(rule, label="right", closed="right").first()
    h = d["high"].resample(rule, label="right", closed="right").max()
    l = d["low"].resample(rule, label="right", closed="right").min()
    c = d["close"].resample(rule, label="right", closed="right").last()
    v = d["volume"].resample(rule, label="right", closed="right").sum()
    out = pd.DataFrame({"time":o.index, "open":o.values,"high":h.values,"low":l.values,"close":c.values,"volume":v.values})
    return out.dropna().reset_index()

def get_candles(symbol: str, tf: str, lookback_bars: int) -> pd.DataFrame:
    tf_ms = {"1m":60000,"5m":300000,"15m":900000,"30m":1800000,"1h":3600000,"6h":21600000,"1d":86400000}[tf]
    since = int((pd.Timestamp.utcnow() - pd.Timedelta(milliseconds=tf_ms*(lookback_bars+5))).timestamp()*1000)
    ex = EX
    mkt = _norm_symbol(symbol, ex)
    supported = getattr(ex, "timeframes", {})
    if tf in supported:
        df = fetch_ohlcv(ex, mkt, tf, since)
    else:
        base = "1h" if tf=="6h" else "5m" if tf=="15m" else "1m"
        df = fetch_ohlcv(ex, mkt, base, since*(3 if base=="1m" else 1))
        df = resample(df, tf)
    return df.tail(lookback_bars)

# ==================
# Indicator helpers
# ==================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).ewm(span=period, adjust=False).mean()

def atr_ema(df: pd.DataFrame, n: int = 14) -> float:
    if df.empty or len(df) < n+1: return np.nan
    h,l,c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1/n, adjust=False).mean().iloc[-1])

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    s = close.astype(float)
    d = s.diff()
    up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1/length, adjust=False).mean()
    rd = dn.ewm(alpha=1/length, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    return 100 - (100/(1+rs))

def stoch_rsi_kd(close: pd.Series, rsi_len=14, stoch_len=14, k_len=3, d_len=3):
    r = _rsi(close, rsi_len)
    rmin = r.rolling(stoch_len, min_periods=stoch_len).min()
    rmax = r.rolling(stoch_len, min_periods=stoch_len).max()
    denom = (rmax - rmin).replace(0, np.nan)
    k = 100*(r - rmin)/denom
    k = k.rolling(k_len, min_periods=k_len).mean()
    d = k.rolling(d_len, min_periods=d_len).mean()
    return k, d

def stoch_last(df: pd.DataFrame) -> Tuple[Optional[float],Optional[float],Optional[float],Optional[float]]:
    if df is None or df.empty or "close" not in df: return (None,None,None,None)
    k,d = stoch_rsi_kd(df["close"])
    if k.dropna().shape[0]<2 or d.dropna().shape[0]<2: return (None,None,None,None)
    return float(k.iloc[-1]), float(d.iloc[-1]), float(k.iloc[-2]), float(d.iloc[-2])

# ===================
# Fibonacci helpers
# ===================
FIB_RETR = [0.236, 0.382, 0.500, 0.618, 0.660, 0.786, 0.886]
FIB_EXT  = [1.272, 1.414, 1.618]

def fib_levels(lo: float, hi: float) -> Dict[str, float]:
    if lo > hi: lo,hi = hi,lo
    base = hi - lo
    retr = {f"R_{r:.3f}": lo + base * r for r in FIB_RETR}
    ext  = {f"X_{x:.3f}": hi + base * (x - 1.0) for x in FIB_EXT}
    return {**retr, **ext}

def fib_confluence(entry: float, levels: List[Tuple[str,float,str]], band_abs: float):
    if not levels:
        return "", float("nan"), float("nan"), 0, [], {}, {}
    nearest = min(levels, key=lambda t: abs(entry - t[1]))
    nearest_name, nearest_px, nearest_win = nearest
    dist_pct = abs(entry - nearest_px)/entry*100.0
    active = [f"{nm}_{w}" for (nm,px,w) in levels if abs(entry - px) <= band_abs]
    conf = len(active)
    whits: Dict[str,int] = {}
    for (nm,px,w) in levels:
        if abs(entry - px) <= band_abs: whits[w] = whits.get(w,0) + 1
    buckets = {"R_0.236":0,"R_0.382":0,"R_0.500":0,"R_0.618_0.660":0,"R_0.786":0,"R_0.886":0,
               "X_1.272":0,"X_1.414":0,"X_1.618":0}
    for (nm,px,w) in levels:
        if abs(entry - px) <= band_abs:
            if nm in buckets: buckets[nm] = 1
            if nm in ("R_0.618","R_0.660"): buckets["R_0.618_0.660"] = 1
    return nearest_name, nearest_px, dist_pct, conf, active, whits, buckets

# ===================
# Volume profile nodes
# ===================
def build_volume_profile(df_1m: pd.DataFrame, bin_pct: float) -> pd.DataFrame:
    if df_1m.empty: return pd.DataFrame(columns=["price","vol"])
    _low = df_1m["low"].min(); _high = df_1m["high"].max()
    if _low<=0 or _high<=0: return pd.DataFrame(columns=["price","vol"])
    bin_w = max(bin_pct * float(df_1m["close"].iloc[-1]), 1e-9)
    bins = np.arange(_low, _high + bin_w, bin_w)
    px_mid = (df_1m["high"] + df_1m["low"]) / 2
    cats = pd.cut(px_mid, bins, right=True); cats.name = "bin"
    vp = df_1m.groupby(cats, observed=True)["volume"].sum().reset_index()
    vp["price"] = vp["bin"].apply(lambda iv: float(iv.right) if hasattr(iv, "right") else np.nan)
    vp = vp.rename(columns={"volume":"vol"})[["price","vol"]].dropna()
    vp["price"] = pd.to_numeric(vp["price"], errors="coerce")
    vp["vol"]   = pd.to_numeric(vp["vol"],   errors="coerce")
    vp = vp.dropna().reset_index(drop=True)
    return vp.sort_values("price").reset_index(drop=True)

def nearest_node(vp: pd.DataFrame, price: float, side: str) -> Optional[float]:
    if vp.empty: return None
    if not np.issubdtype(vp["price"].dtype, np.number):
        vp = vp.copy(); vp["price"] = pd.to_numeric(vp["price"], errors="coerce"); vp = vp.dropna(subset=["price"])
        if vp.empty: return None
    df = vp[vp["price"] < price] if side == "SHORT" else vp[vp["price"] > price]
    if df.empty: return None
    return float(df.loc[df["vol"].idxmax(), "price"])

# ===================
# Order book & trades
# ===================
def fetch_order_book_near(ex, symbol: str, near_pct: float) -> Tuple[Optional[float],Optional[float],Optional[float]]:
    """Return (near_bids_usd, near_asks_usd, mid)."""
    mkt = _norm_symbol(symbol, ex)
    try:
        ob = ex.fetch_order_book(mkt, limit=200)
    except Exception:
        return None, None, None
    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    if not bids or not asks: return None, None, None
    best_bid = float(bids[0][0]); best_ask = float(asks[0][0]); mid = (best_bid + best_ask)/2.0
    lo = mid * (1 - near_pct); hi = mid * (1 + near_pct)
    near_b = sum(float(p)*float(a) for p,a in bids if p >= lo)
    near_a = sum(float(p)*float(a) for p,a in asks if p <= hi)
    return near_b, near_a, mid

def fetch_trades_window(ex, symbol: str, since_ms: int, limit: int=TRADES_LIMIT) -> List[Dict]:
    mkt = _norm_symbol(symbol, ex)
    trades = []
    for _ in range(5):
        try:
            trades = ex.fetch_trades(mkt, since=since_ms, limit=limit)
            break
        except (ccxt.DDoSProtection, ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.ExchangeNotAvailable):
            time.sleep(0.6 + random.uniform(0,0.2))
        except Exception:
            trades = []
            break
    return trades or []

def signed_usd_from_trade(t: Dict, mid: Optional[float]) -> float:
    # Prefer ccxt 'side' if present; fallback to tick-rule vs mid
    px = float(t.get("price") or 0.0); amt = float(t.get("amount") or 0.0)
    usd = px * amt
    side = (t.get("side") or "").lower()
    if side in ("buy","sell"):
        return usd if side=="buy" else -usd
    if mid is None:
        return 0.0
    # tick rule: trade above/equal mid -> buy aggression; below -> sell
    return usd if px >= mid else -usd

# ===================
# CSV & outcomes
# ===================
def ensure_csv(path: str):
    if os.path.exists(path): return
    cols = [
        "timestamp_utc","symbol","side","mode","entry",
        # EMA context
        "ema50_1h","ema200_1h","ema200_1d",
        # Stoch RSI
        "stoch1h_k","stoch1h_d","stoch15m_k","stoch15m_d","stoch15m_cross_up","stoch15m_cross_dn",
        # Fib nearest + confluence
        "fib_nearest_ratio","fib_nearest_price","fib_dist_pct","fib_confluence_count","fib_active_list",
        "fib_window_hits_json",
        "fib_R_0.236","fib_R_0.382","fib_R_0.500","fib_R_0.618_0.660","fib_R_0.786","fib_R_0.886",
        "fib_X_1.272","fib_X_1.414","fib_X_1.618",
        # Volume profile node
        "vp_node_price","vp_node_score","vp_dist_entry_pct","vp_dist_tp_pct",
        # Order book & flow
        "near_bids_usd","near_asks_usd","book_ratio",
        "burst_window_sec","delta_usd_window","delta_ratio","abs_delta_q_global","abs_delta_q_coin",
        "flow_ok","book_ok",
        # Optional perp tilt
        "basis_pct","funding_8h","risk_tilt",
        # Outcomes (future prices and returns)
        "p_15m","p_30m","p_1h","p_2h","p_3h","p_4h",
        "r_15m_pct","r_30m_pct","r_1h_pct","r_2h_pct","r_3h_pct","r_4h_pct"
    ]
    pd.DataFrame(columns=cols).to_csv(path, index=False, encoding="utf-8-sig")

def append_row(path: str, row: Dict):
    df = pd.DataFrame([row])
    hdr = not os.path.exists(path) or os.path.getsize(path)==0
    df.to_csv(path, index=False, mode="a", header=hdr, encoding="utf-8-sig")

def fill_outcomes(path: str):
    try: df = pd.read_csv(path)
    except Exception: return
    if df.empty or "timestamp_utc" not in df.columns: return
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    now = pd.Timestamp.utcnow()
    changed = False
    for idx, r in df.iterrows():
        ts = r["timestamp_utc"]; 
        if pd.isna(ts): continue
        entry = float(r["entry"]); sym = _norm_symbol(str(r["symbol"]), EX); side = str(r["side"]).upper()
        def set_h(hmin: int, col_p: str, col_r: str):
            nonlocal changed
            if not pd.isna(r.get(col_p, np.nan)): return
            tgt = ts + pd.Timedelta(minutes=hmin)
            if now < tgt: return
            since = int((ts - pd.Timedelta(minutes=1)).timestamp()*1000)
            d1m = fetch_ohlcv(EX, sym, "1m", since, limit=5000)
            if d1m.empty: return
            d1m["time"] = pd.to_datetime(d1m["time"], unit="ms", utc=True)
            d1m = d1m[d1m["time"] <= tgt]
            if d1m.empty: return
            px = float(d1m["close"].iloc[-1])
            df.at[idx, col_p] = px
            df.at[idx, col_r] = (px - entry)/entry*100 if side=="LONG" else (entry - px)/entry*100
            changed = True
        for mins, pc, rc in [(15,"p_15m","r_15m_pct"),(30,"p_30m","r_30m_pct"),(60,"p_1h","r_1h_pct"),
                              (120,"p_2h","r_2h_pct"),(180,"p_3h","r_3h_pct"),(240,"p_4h","r_4h_pct")]:
            set_h(mins, pc, rc)
    if changed: df.to_csv(path, index=False, encoding="utf-8-sig")

# ===================
# Quintile helpers for flow (qG, qC)
# ===================
def load_flow_edges(log_path: str, lookback_h: int = FLOW_LOOKBACK_H):
    try:
        df = pd.read_csv(log_path, usecols=["timestamp_utc","symbol","delta_usd_window"])
    except Exception:
        return None, {}
    if df.empty or "delta_usd_window" not in df: return None, {}
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=lookback_h)
    df = df[(df["timestamp_utc"] >= cutoff)]
    df["abs_delta"] = pd.to_numeric(df["delta_usd_window"], errors="coerce").abs()
    df = df.dropna(subset=["abs_delta","symbol"])
    if df.empty: return None, {}
    # Global edges (q20,q40,q60,q80)
    g_edges = np.quantile(df["abs_delta"], [0.2,0.4,0.6,0.8]).tolist() if df["abs_delta"].notna().sum() >= 10 else None
    # Per-coin edges
    per_coin = {}
    for sym, g in df.groupby("symbol"):
        x = pd.to_numeric(g["abs_delta"], errors="coerce").dropna()
        if len(x) >= 10:
            per_coin[sym] = np.quantile(x, [0.2,0.4,0.6,0.8]).tolist()
    return g_edges, per_coin

def assign_quintile(x: Optional[float], edges: Optional[List[float]]) -> Optional[int]:
    if x is None or edges is None: return None
    q20,q40,q60,q80 = edges
    if x <= q20: return 1
    if x <= q40: return 2
    if x <= q60: return 3
    if x <= q80: return 4
    return 5

# =================
# Entry generation
# =================
LAST_LOGGED: Dict[str, float] = {}

FIB_RETR_LIST = [0.236, 0.382, 0.500, 0.618, 0.660, 0.786, 0.886]
FIB_EXT_LIST  = [1.272, 1.414, 1.618]

def propose_entry(symbol: str, flow_edges) -> Optional[Dict]:
    # Candles
    d15 = get_candles(symbol, "15m", 200)
    d1h = get_candles(symbol, "1h",  400)
    d6h = resample(d1h, "6h")
    d1d = get_candles(symbol, "1d",  400)
    if d1h.empty or d1d.empty: return None

    curr = float(d1h["close"].iloc[-1])
    prev = float(d1h["close"].iloc[-2] if len(d1h)>1 else curr)

    # EMAs
    ema50_1h  = float(ema(d1h["close"], 50).iloc[-1])
    ema200_1h = float(ema(d1h["close"], 200).iloc[-1])
    ema200_1d = float(ema(d1d["close"], 200).iloc[-1])

    # ATR band for Fib-bucket hit tolerance
    atr1h = atr_ema(d1h, 14)
    fib_band_abs = max(curr * FIB_PCT_BAND, (atr1h if not math.isnan(atr1h) else 0.0) * FIB_ATR_K)

    # Fib levels across windows
    def swing_levels(df: pd.DataFrame, bars: int, tag: str) -> List[Tuple[str,float,str]]:
        if df.empty or len(df) < bars: return []
        lo = float(df["low"].tail(bars).min()); hi = float(df["high"].tail(bars).max())
        lev = fib_levels(lo, hi)
        return [(nm,px,tag) for nm,px in lev.items()]

    levels: List[Tuple[str,float,str]] = []
    levels += swing_levels(d1h, 2*24,   "TINY")
    levels += swing_levels(d1h, 7*24,   "SHORT")
    levels += swing_levels(d6h, 14,     "MEDIUM")
    levels += swing_levels(d6h, 30,     "LONG")

    nearest_name, nearest_px, dist_pct, conf_cnt, active_list, window_hits, bucket_flags = fib_confluence(
        curr, levels, fib_band_abs
    )

    # “Zones”: any Fib hits within band; plus EMA bands
    zones_active = [f"{nm}_{w}" for (nm,px,w) in levels if abs(curr - px) <= fib_band_abs]
    for nm,val in [("EMA50_1H", ema50_1h), ("EMA200_1H", ema200_1h)]:
        if abs(curr - val)/curr <= EMA_BAND_PCT: zones_active.append(nm)

    # Pick an active zone (metadata only)
    prio = {"EMA200_1D":12,"EMA200_1H":8,"EMA50_1H":7,
            "R_0.886":6,"R_0.786":6,"R_0.660":6,"R_0.618":6,"R_0.500":5,"R_0.382":4,"R_0.236":3,
            "X_1.618":5,"X_1.414":4,"X_1.272":3}
    def zone_score(z: str) -> int:
        parts = z.split("_")
        key = z if parts[0].startswith("EMA") else "_".join(parts[:2])
        return prio.get(key, 1)
    active_zone = max(zones_active, key=zone_score) if zones_active else None
    if not active_zone:
        # widen to EMA proximity only
        if abs(curr-ema50_1h)/curr <= EMA_BAND_PCT*1.2: active_zone = "EMA50_1H"
        elif abs(curr-ema200_1h)/curr <= EMA_BAND_PCT*1.2: active_zone = "EMA200_1H"
        else: return None

    # Mode & side (light rules)
    def infer_mode(zone_name: str) -> str:
        return "CONTINUATION" if zone_name.startswith("EMA") else "REVERSAL"
    mode = infer_mode(active_zone)
    side = "SHORT" if (mode=="CONTINUATION" and curr < ema200_1d) or (mode=="REVERSAL" and curr > prev) else "LONG"

    # Stoch RSI (1h/15m)
    k1h,d1h_,k1h_p,d1h_p = stoch_last(d1h)
    k15,d15,k15_p,d15_p  = stoch_last(d15)
    cross15_up = int(k15_p is not None and d15_p is not None and k15 is not None and d15 is not None and k15_p <= d15_p and k15 > d15)
    cross15_dn = int(k15_p is not None and d15_p is not None and k15 is not None and d15 is not None and k15_p >= d15_p and k15 < d15)

    # Volume profile node (optional)
    vp_node_price = None; vp_node_score = None; dist_entry_pct=None; dist_tp_pct=None
    if USE_NODE_SCORE:
        since_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=VP_LOOKBACK_DAYS)).timestamp()*1000)
        d1m = fetch_ohlcv(EX, _norm_symbol(symbol, EX), "1m", since_ms, limit=1500*VP_LOOKBACK_DAYS)
        vp = build_volume_profile(d1m, VP_BIN_PCT)
        node = nearest_node(vp, curr, side)
        vp_node_price = node
        if node:
            dist_entry_pct = abs(curr-node)/curr*100
            stride = 0.7 if mode=="REVERSAL" else 1.5
            tp_guess = curr*(1 - stride/100) if side=="SHORT" else curr*(1 + stride/100)
            dist_tp_pct = abs(tp_guess - node)/curr*100
            vp_node_score = float(np.clip(1.0 - (dist_tp_pct/0.8), 0, 1)) * float(np.clip(dist_entry_pct/0.4, 0, 1))

    # ============== Order flow telemetry ============
    # Order book snapshot
    near_b, near_a, mid = fetch_order_book_near(EX, symbol, BOOK_NEAR_PCT)
    book_ratio = None
    if near_b is not None and near_a is not None and (near_b + near_a) > 0:
        book_ratio = (near_b - near_a) / (near_b + near_a)

    # Rolling ΔCVD over BURST_WINDOW_SEC (signed USD by aggressor)
    now_ms = int(pd.Timestamp.utcnow().timestamp()*1000)
    since_ms = now_ms - BURST_WINDOW_SEC*1000
    trades = fetch_trades_window(EX, symbol, since_ms, TRADES_LIMIT)
    delta_usd_window = 0.0
    if trades:
        # If mid is None, try to infer from last book or last trade prices
        if mid is None:
            pxs = [float(t.get("price") or 0.0) for t in trades if t.get("price") is not None]
            if pxs: mid = float(pxs[-1])
        for t in trades:
            delta_usd_window += signed_usd_from_trade(t, mid)

    # Normalize delta per second vs baseline window
    thr_per_sec = 1.0  # will rescale by dynamic threshold from log if available
    delta_per_sec = abs(delta_usd_window) / max(1.0, BURST_WINDOW_SEC)
    # edges from log (global & per-coin) -> compute delta_ratio and quintiles
    g_edges, per_coin_edges = flow_edges or (None, {})
    # derive a threshold proxy from edges: use q60 as "1×"
    def per_sec_from_edges(edges):
        if not edges: return None
        q20,q40,q60,q80 = edges
        return q60 / max(1.0, BASE_BURST_WINDOW_SEC)

    thr_ps_global = per_sec_from_edges(g_edges)
    thr_ps_coin   = per_sec_from_edges(per_coin_edges.get(symbol))
    # choose coin if exists else global, else fallback
    thr_per_sec = thr_ps_coin or thr_ps_global or (delta_per_sec if delta_per_sec>0 else 1.0)
    delta_ratio = delta_per_sec / max(1e-9, thr_per_sec)

    # assign quintiles on abs(delta_usd_window)
    abs_delta = abs(delta_usd_window)
    qG = assign_quintile(abs_delta, g_edges) if g_edges else None
    qC = assign_quintile(abs_delta, per_coin_edges.get(symbol)) if per_coin_edges else None

    # flow/book alignment flags
    flow_ok = None
    if delta_usd_window != 0:
        flow_ok = (delta_usd_window > 0 and side=="LONG") or (delta_usd_window < 0 and side=="SHORT")
    book_ok = None
    if book_ratio is not None:
        book_ok = (book_ratio > 0 and side=="LONG") or (book_ratio < 0 and side=="SHORT")

    # Optional perp tilt
    basis_pct = None; funding_8h = None; risk_tilt = None
    if PERP:
        try:
            base = symbol.split("/")[0].replace("-", "/").upper()
            perp_sym = f"{base}/USDT" if "USDT" in PERP_SUFFIX else f"{base}/USD"
            perp_norm = _norm_symbol(perp_sym, PERP)
            spot_norm = _norm_symbol(symbol, EX)
            # perp mid (ticker) and spot mid (book)
            perf_t = PERP.fetch_ticker(perp_norm)
            perp_mid = (float(perf_t.get("bid") or 0.0) + float(perf_t.get("ask") or 0.0)) / 2.0
            if mid is None:
                nb, na, mid = fetch_order_book_near(EX, symbol, BOOK_NEAR_PCT)
            if mid and perp_mid:
                basis_pct = (perp_mid - mid) / mid * 100.0
            # funding (if supported)
            try:
                fr = PERP.fetch_funding_rate(perp_norm)
                funding_8h = float(fr.get("fundingRate", None))
            except Exception:
                funding_8h = None
            # simple tilt proxy: tanh of basis/0.5% plus funding weight
            if basis_pct is not None:
                tilt_raw = (basis_pct/0.5) + ( (funding_8h*100 if funding_8h is not None else 0.0) * 0.6 )
                risk_tilt = float(np.tanh(tilt_raw/2.0))
        except Exception:
            pass
    # ===============================================

    # Build row
    row = {
        "timestamp_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol, "side": side, "mode": mode, "entry": curr,
        # EMA, Stoch
        "ema50_1h": ema50_1h, "ema200_1h": ema200_1h, "ema200_1d": ema200_1d,
        "stoch1h_k": k1h, "stoch1h_d": d1h_,
        "stoch15m_k": k15, "stoch15m_d": d15,
        "stoch15m_cross_up": cross15_up if k15 is not None else None,
        "stoch15m_cross_dn": cross15_dn if k15 is not None else None,
        # Fib context
        "fib_nearest_ratio": nearest_name, "fib_nearest_price": nearest_px, "fib_dist_pct": dist_pct,
        "fib_confluence_count": conf_cnt, "fib_active_list": ",".join(active_list) if active_list else "",
        "fib_window_hits_json": "" if not window_hits else str(window_hits),
        "fib_R_0.236": bucket_flags.get("R_0.236",0),
        "fib_R_0.382": bucket_flags.get("R_0.382",0),
        "fib_R_0.500": bucket_flags.get("R_0.500",0),
        "fib_R_0.618_0.660": bucket_flags.get("R_0.618_0.660",0),
        "fib_R_0.786": bucket_flags.get("R_0.786",0),
        "fib_R_0.886": bucket_flags.get("R_0.886",0),
        "fib_X_1.272": bucket_flags.get("X_1.272",0),
        "fib_X_1.414": bucket_flags.get("X_1.414",0),
        "fib_X_1.618": bucket_flags.get("X_1.618",0),
        # Node
        "vp_node_price": vp_node_price, "vp_node_score": vp_node_score,
        "vp_dist_entry_pct": dist_entry_pct, "vp_dist_tp_pct": dist_tp_pct,
        # Order book & flow
        "near_bids_usd": near_b, "near_asks_usd": near_a, "book_ratio": book_ratio,
        "burst_window_sec": BURST_WINDOW_SEC,
        "delta_usd_window": delta_usd_window,
        "delta_ratio": delta_ratio,
        "abs_delta_q_global": qG, "abs_delta_q_coin": qC,
        "flow_ok": int(flow_ok) if flow_ok is not None else None,
        "book_ok": int(book_ok) if book_ok is not None else None,
        # Perp tilt
        "basis_pct": basis_pct, "funding_8h": funding_8h, "risk_tilt": risk_tilt,
        # Outcomes placeholders
        "p_15m": np.nan, "p_30m": np.nan, "p_1h": np.nan, "p_2h": np.nan, "p_3h": np.nan, "p_4h": np.nan,
        "r_15m_pct": np.nan, "r_30m_pct": np.nan, "r_1h_pct": np.nan, "r_2h_pct": np.nan, "r_3h_pct": np.nan, "r_4h_pct": np.nan,
    }
    return row

# =================
# Main loop
# =================
LAST_LOGGED: Dict[str, float] = {}

def main_loop():
    ensure_csv(CSV_PATH)
    print(f"[probe] writing to {CSV_PATH} • exchange={EXCHANGE_ID} • {len(COINS)} symbols")
    while True:
        t0 = time.time()
        # Refresh flow edges (quintile edges) every pass (cheap)
        flow_edges = load_flow_edges(CSV_PATH, FLOW_LOOKBACK_H)
        for sym in COINS:
            sym = sym.strip()
            try:
                last = LAST_LOGGED.get(sym, 0.0)
                if time.time() - last < COOLDOWN_MIN_SEC:
                    continue
                row = propose_entry(sym, flow_edges)
                if not row: continue
                append_row(CSV_PATH, row)
                LAST_LOGGED[sym] = time.time()
                print(f"[logged] {row['timestamp_utc']}  {sym}  {row['mode']} {row['side']}  entry={row['entry']:.6g}  fib={row['fib_nearest_ratio']} conf={row['fib_confluence_count']}  flow|qG={row['abs_delta_q_global']} qC={row['abs_delta_q_coin']}")
            except Exception as e:
                print(f"[ERR] {sym}: {e}")
            time.sleep(0.05)
        try:
            fill_outcomes(CSV_PATH)
        except Exception as e:
            print("[fill] error:", e)
        dt = time.time() - t0
        time.sleep(max(1.0, SCAN_EVERY_SEC - dt))

if __name__ == "__main__":
    main_loop()
