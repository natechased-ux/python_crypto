#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ATR-Scalp Backtest (Chunked Fetch, No New Filters)

Entries:
  - Same as your current system (Fib zones, EMA zones, Liquidity sweeps, Session VWAP)
  - Stoch RSI strict (40/60) REQUIRED (as before)
  - No new filters added

Exit:
  - TP1 = ATR(14, 1H) * ATR_MULT (default: 1.0) -- close 100% at TP1
  - SL = zone boundary (± SL_BUFFER)
  - Timeout after MAX_HOLD_HOURS (close at market)

Outputs:
  - trades_atr_scalp.csv
  - summary_atr_scalp.csv
"""

import csv, time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# ===================== CONFIG =====================
COINS = [
    "BTC-USD","ETH-USD","XRP-USD","SOL-USD","ADA-USD","AVAX-USD","DOGE-USD","DOT-USD",
    "LINK-USD","ATOM-USD","NEAR-USD","ARB-USD","OP-USD","MATIC-USD","SUI-USD",
    "INJ-USD","AAVE-USD","LTC-USD","BCH-USD","ETC-USD","ALGO-USD","FIL-USD","ICP-USD",
    "RNDR-USD","STX-USD","JTO-USD","PYTH-USD","GRT-USD","SEI-USD",
    "ENS-USD","FLOW-USD","KSM-USD","KAVA-USD",
    "WLD-USD","HBAR-USD","JUP-USD","STRK-USD",
    "ONDO-USD","SUPER-USD","LDO-USD","POL-USD",
    "ZETA-USD","ZRO-USD","TIA-USD",
    "WIF-USD","MAGIC-USD","APE-USD","JASMY-USD","SYRUP-USD","FARTCOIN-USD",
    "AERO-USD","FET-USD","CRV-USD","TAO-USD","XCN-USD","UNI-USD","MKR-USD",
    "TOSHI-USD","TRUMP-USD","PEPE-USD","XLM-USD","MOODENG-USD","BONK-USD",
    "POPCAT-USD","QNT-USD","IP-USD","PNUT-USD","APT-USD","ENA-USD","TURBO-USD",
    "BERA-USD","MASK-USD","SAND-USD","MORPHO-USD","MANA-USD","C98-USD","AXS-USD"
]

# Backtest window (UTC)
START_DATE = "2025-02-01"
END_DATE   = "2025-08-25"

# Timeframe Fib windows
TINY_LOOKBACK_HOURS  = 2 * 24   # 2d on 1H
SHORT_LOOKBACK_HOURS = 7 * 24   # 7d on 1H
MEDIUM_LOOKBACK_DAYS = 14       # 14d on 1D
LONG_LOOKBACK_DAYS   = 30       # 30d on 1D

# EMA band width (±)
EMA_BAND_PCT = 0.005  # 0.5%

# Liquidity sweeps (1H)
SWEEP_LOOKBACK_HOURS = 48
SWEEP_BAND_PCT       = 0.0015  # ±0.15%

# VWAP (UTC session)
VWAP_BAND_PCT = 0.003

# Stoch RSI strict thresholds (same as before)
STOCH_RSI_PERIOD = 14
STOCH_SMOOTH_K   = 3
STOCH_SMOOTH_D   = 3
LONG_K_MAX  = 40.0  # LONG: K>D & K<40
SHORT_K_MIN = 60.0  # SHORT: K<D & K>60

# Fibonacci golden pocket
GOLDEN_MIN = 0.618
GOLDEN_MAX = 0.66
GOLDEN_TOL = 0.0025

# ATR scalp exits
SL_BUFFER     = 0.01     # SL at zone boundary (±1%)
ATR_MULT_TP1  = 1.0      # TP1 = 1.0 x ATR(14,1H); set 0.6~1.2 to tune
MAX_HOLD_HOURS= 7 * 24
SLIPPAGE_PCT  = 0.0
FEES_PCT      = 0.0

# I/O
TRADES_CSV  = "trades_atr_scalp.csv"
SUMMARY_CSV = "summary_atr_scalp.csv"

# Coinbase
CB_BASE = "https://api.exchange.coinbase.com"
MAX_BARS_PER_CALL = 300

# Warmup bars for indicators
WARMUP_BARS = { 3600: 250, 21600: 80, 86400: 220 }

# ================== TA / UTILS ==================
def rsi(series: pd.Series, period=14):
    d = series.diff()
    up = d.clip(lower=0.0); down = -d.clip(upper=0.0)
    ru = up.ewm(alpha=1/period, adjust=False).mean()
    rd = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd.replace(0, 1e-10))
    return 100 - (100/(1+rs))

def stoch_rsi(close: pd.Series, period: int, k: int, d_: int):
    r = rsi(close, period)
    lo = r.rolling(period).min(); hi = r.rolling(period).max()
    st = (r - lo)/(hi - lo + 1e-10)*100.0
    K = st.rolling(k).mean(); D = K.rolling(d_).mean()
    return K, D

def ema(series: pd.Series, period: int): return series.ewm(span=period, adjust=False).mean()

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([(high-low),(high-prev_close).abs(),(low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up = high.diff(); down = -low.diff()
    plus_dm  = np.where((up>down)&(up>0), up, 0.0)
    minus_dm = np.where((down>up)&(down>0), down, 0.0)
    atr_val  = atr(high, low, close, period=period)
    plus_di  = 100*pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean()/atr_val.replace(0,np.nan)
    minus_di = 100*pd.Series(minus_dm,index=high.index).ewm(alpha=1/period,adjust=False).mean()/atr_val.replace(0,np.nan)
    dx = (plus_di - minus_di).abs()/ (plus_di + minus_di).replace(0,np.nan) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)

def fib_golden_zone(lo: float, hi: float, tol: float = GOLDEN_TOL):
    span = hi - lo; f618 = lo+span*GOLDEN_MIN; f66 = lo+span*GOLDEN_MAX
    return f618*(1-tol), f66*(1+tol)

def recent_swing_levels(df_1h: pd.DataFrame, hours: int):
    w = df_1h.tail(hours)
    return float(w["high"].max()), float(w["low"].min())

def detect_reversal(prev: float, curr: float, zmin: float, zmax: float):
    if zmin is None or zmax is None: return None
    if prev < zmin and zmin <= curr <= zmax: return "SHORT"
    if prev > zmax and zmin <= curr <= zmax: return "LONG"
    return None

# --------- Chunked fetching + warmup ---------
def fetch_candles_range(product_id: str, granularity: int, start_dt: pd.Timestamp, end_dt: pd.Timestamp, session=None):
    sess = session or requests.Session()
    url = f"{CB_BASE}/products/{product_id}/candles"
    out = []; step_secs = granularity*MAX_BARS_PER_CALL; curr_end = end_dt
    while curr_end > start_dt:
        curr_start = max(start_dt, curr_end - pd.Timedelta(seconds=step_secs))
        params = {"granularity": granularity, "start": curr_start.isoformat(), "end": curr_end.isoformat()}
        r = sess.get(url, params=params, timeout=20); r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data: break
        df_chunk = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
        df_chunk["time"] = pd.to_datetime(df_chunk["time"], unit="s", utc=True)
        out.append(df_chunk)
        earliest = df_chunk["time"].min()
        if pd.isna(earliest): break
        curr_end = earliest - pd.Timedelta(seconds=1)
        time.sleep(0.05)
    if not out: raise ValueError(f"No data for {product_id}@{granularity}")
    df = pd.concat(out, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    df = df[(df["time"] >= start_dt) & (df["time"] < end_dt)].reset_index(drop=True)
    return df

def fetch_with_warmup(product_id: str, granularity: int, start_date: str, end_date: str) -> pd.DataFrame:
    s = pd.Timestamp(start_date, tz="UTC"); e = pd.Timestamp(end_date, tz="UTC")
    warm_bars = WARMUP_BARS.get(granularity, 0); s_warm = s - pd.Timedelta(seconds=warm_bars*granularity)
    return fetch_candles_range(product_id, granularity, s_warm, e)

# --------- Build zones for an index i ---------
def build_zones(df_1h, df_1d, i):
    t = df_1h["time"].iloc[i]; curr = float(df_1h["close"].iloc[i]); prev = float(df_1h["close"].iloc[i-1])
    entries = []; flags = {}
    # FIB 1H tiny/short
    for name, look in [("FIB_TINY", 2*24), ("FIB_SHORT", 7*24)]:
        w = df_1h.iloc[:i+1].tail(look)
        sl, sh = float(w["low"].min()), float(w["high"].max())
        zmin, zmax = fib_golden_zone(sl, sh)
        flags[f"in_{name.lower()}"] = (zmin <= curr <= zmax)
        side = detect_reversal(prev, curr, zmin, zmax)
        if side: entries.append((name, sl, sh, zmin, zmax, side))
    # FIB 1D medium/long
    df_1d_hist = df_1d[df_1d["time"] <= t]
    if len(df_1d_hist) >= MEDIUM_LOOKBACK_DAYS:
        w = df_1d_hist.tail(MEDIUM_LOOKBACK_DAYS); sl, sh = float(w["low"].min()), float(w["high"].max())
        zmin, zmax = fib_golden_zone(sl, sh)
        flags["in_fib_medium"] = (zmin <= curr <= zmax); side = detect_reversal(prev, curr, zmin, zmax)
        if side: entries.append(("FIB_MEDIUM", sl, sh, zmin, zmax, side))
    else: flags["in_fib_medium"]=False
    if len(df_1d_hist) >= LONG_LOOKBACK_DAYS:
        w = df_1d_hist.tail(LONG_LOOKBACK_DAYS); sl, sh = float(w["low"].min()), float(w["high"].max())
        zmin, zmax = fib_golden_zone(sl, sh)
        flags["in_fib_long"] = (zmin <= curr <= zmax); side = detect_reversal(prev, curr, zmin, zmax)
        if side: entries.append(("FIB_LONG", sl, sh, zmin, zmax, side))
    else: flags["in_fib_long"]=False

    # EMA 1H
    def ema_zone_1h(period, nm):
        if i+1 < period: 
            flags[f"in_{nm.lower()}"] = False
            return
        val = float(ema(df_1h["close"].iloc[:i+1], period).iloc[-1])
        zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
        flags[f"in_{nm.lower()}"] = (zmin <= curr <= zmax)
        side = detect_reversal(prev, curr, zmin, zmax)
        if side: entries.append((nm, val, val, zmin, zmax, side))
    ema_zone_1h(50,  "EMA50_1H")
    ema_zone_1h(200, "EMA200_1H")

    # EMA 1D
    def add_ema_1d(period, nm):
        if len(df_1d_hist) >= period:
            val = float(ema(df_1d_hist["close"], period).iloc[-1])
            zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
            flags[f"in_{nm.lower()}"] = (zmin <= curr <= zmax)
            side = detect_reversal(prev, curr, zmin, zmax)
            if side: entries.append((nm, val, val, zmin, zmax, side))
        else:
            flags[f"in_{nm.lower()}"]=False
    add_ema_1d(20,  "EMA20_1D")
    add_ema_1d(50,  "EMA50_1D")
    add_ema_1d(100, "EMA100_1D")
    add_ema_1d(200, "EMA200_1D")

    # Liquidity sweeps (1H)
    hi, lo = recent_swing_levels(df_1h.iloc[:i+1], SWEEP_LOOKBACK_HOURS)
    hi_min,hi_max = hi*(1-SWEEP_BAND_PCT), hi*(1+SWEEP_BAND_PCT)
    lo_min,lo_max = lo*(1-SWEEP_BAND_PCT), lo*(1+SWEEP_BAND_PCT)
    flags["in_liq_sweep_high_1h"] = (hi_min <= curr <= hi_max)
    flags["in_liq_sweep_low_1h"]  = (lo_min <= curr <= lo_max)
    side_s_hi = detect_reversal(prev, curr, hi_min, hi_max)
    side_s_lo = detect_reversal(prev, curr, lo_min, lo_max)
    if side_s_hi: entries.append(("LIQ_SWEEP_HIGH_1H", hi, hi, hi_min, hi_max, side_s_hi))
    if side_s_lo: entries.append(("LIQ_SWEEP_LOW_1H",  lo, lo, lo_min, lo_max, side_s_lo))

    # Session VWAP band (UTC)
    vmin, vmax, vwap_val = session_vwap_band(df_1h, i, VWAP_BAND_PCT)
    flags["in_vwap_session_1h"] = (vmin is not None and vmin <= curr <= vmax)
    side_vwap = detect_reversal(prev, curr, vmin, vmax) if vmin is not None else None
    if side_vwap: entries.append(("VWAP_SESSION_1H", vwap_val, vwap_val, vmin, vmax, side_vwap))

    return entries, flags

def session_vwap_band(df_1h: pd.DataFrame, idx: int, band_pct: float):
    ts = df_1h["time"].iloc[idx]; start_day = ts.normalize()
    mask = (df_1h["time"] >= start_day) & (df_1h["time"] <= ts)
    d = df_1h.loc[mask]
    if d.empty or d["volume"].sum() <= 0: return (None, None, None)
    pv = (d["close"] * d["volume"]).sum(); v = d["volume"].sum()
    vwap = float(pv / v)
    return (vwap*(1-band_pct), vwap*(1+band_pct), vwap)

def bollinger_position(close: pd.Series, period=20, mult=2.0):
    m = close.rolling(period).mean()
    s = close.rolling(period).std(ddof=0)
    pos = (close - m)/(mult*s + 1e-10)
    return m, m + mult*s, m - mult*s, pos

# ------------- Backtest -------------
def run_backtest_for_coin(coin: str):
    df_1h = fetch_with_warmup(coin, 3600,  START_DATE, END_DATE)
    df_1d = fetch_with_warmup(coin, 86400, START_DATE, END_DATE)
    if len(df_1h) < SHORT_LOOKBACK_HOURS + 250:
        return []

    # Indicators
    K, D = stoch_rsi(df_1h["close"], STOCH_RSI_PERIOD, STOCH_SMOOTH_K, STOCH_SMOOTH_D)
    atr_1h_series = atr(df_1h["high"], df_1h["low"], df_1h["close"], 14)

    trades = []
    warmup = max(SHORT_LOOKBACK_HOURS, 250)

    # zone priority (unchanged)
    priority = {
        "EMA200_1D":11,"EMA100_1D":10,"EMA20_1D":9,"EMA50_1D":8,
        "FIB_LONG":7,"FIB_MEDIUM":6,
        "LIQ_SWEEP_HIGH_1H":5,"LIQ_SWEEP_LOW_1H":5,
        "VWAP_SESSION_1H":4,"EMA200_1H":3,"EMA50_1H":2,
        "FIB_SHORT":2,"FIB_TINY":1
    }

    for i in range(warmup, len(df_1h)-1):
        # Build zones & choose active
        entries, flags = build_zones(df_1h, df_1d, i)
        entries = [e for e in entries if e[5] is not None]
        if not entries: continue
        entries.sort(key=lambda x: priority.get(x[0],1), reverse=True)
        zone_source, swing_low, swing_high, zmin, zmax, side = entries[0]

        # Stoch strict (no new filters)
        kv = float(K.iloc[i]) if not pd.isna(K.iloc[i]) else None
        dv = float(D.iloc[i]) if not pd.isna(D.iloc[i]) else None
        if kv is None or dv is None: continue
        stoch_ok = (kv>dv and kv<LONG_K_MAX) if side=="LONG" else (kv<dv and kv>SHORT_K_MIN)
        if not stoch_ok: continue

        # Risk + ATR TP1
        curr = float(df_1h["close"].iloc[i])
        if side=="LONG":
            sl = zmin*(1-SL_BUFFER); risk = curr - sl
            tp1 = curr + ATR_MULT_TP1 * float(atr_1h_series.iloc[i])
        else:
            sl = zmax*(1+SL_BUFFER); risk = sl - curr
            tp1 = curr - ATR_MULT_TP1 * float(atr_1h_series.iloc[i])
        if risk <= 0: continue

        # Manage forward (TP1 only; 100% close at TP1)
        exit_price = curr; outcome = "Timeout"; hold_hours = 0
        highest, lowest = curr, curr
        for j in range(i+1, len(df_1h)):
            bar = df_1h.iloc[j]; bh=float(bar["high"]); bl=float(bar["low"])
            if bh>highest: highest=bh
            if bl<lowest:  lowest=bl

            # TP1 (close all)
            if (side=="LONG" and bh>=tp1) or (side=="SHORT" and bl<=tp1):
                exit_price = tp1; outcome = "TP1"; hold_hours = j - i; break

            # SL
            if (side=="LONG" and bl<=sl) or (side=="SHORT" and bh>=sl):
                exit_price = sl; outcome = "SL"; hold_hours = j - i; break

            # Timeout
            if (j - i) >= MAX_HOLD_HOURS:
                exit_price = float(df_1h["close"].iloc[j]); outcome = "Timeout"; hold_hours = j - i; break

        # Compute R
        netR = (exit_price - curr) / risk if side=="LONG" else (curr - exit_price) / risk
        netR -= FEES_PCT

        trades.append({
            "coin": coin, "time": df_1h["time"].iloc[i].strftime("%Y-%m-%d %H:%M"),
            "side": side, "zone_source": zone_source,
            "entry": curr, "sl": sl, "tp1": tp1,
            "outcome": outcome, "R_multiple": round(float(netR),4),
            "hold_hours": hold_hours
        })

    return trades

# ------------- Save + Summary -------------
def save_trades_and_summary(trades, trades_path, summary_path):
    if not trades:
        print("No trades"); return
    df = pd.DataFrame(trades)
    df.to_csv(trades_path, index=False)

    def win_rate(s): return (s>0).mean() if len(s) else 0.0
    overall = pd.DataFrame([{
        "trades": len(df),
        "win_rate": win_rate(df["R_multiple"]),
        "TP1_rate": (df["outcome"]=="TP1").mean(),
        "avg_R": df["R_multiple"].mean(),
        "median_R": df["R_multiple"].median(),
        "avg_hold_hours": df["hold_hours"].mean()
    }])
    by_zone = df.groupby("zone_source").agg(
        trades=("zone_source","count"),
        win_rate=("R_multiple", win_rate),
        TP1_rate=("outcome", lambda s:(s=="TP1").mean()),
        avg_R=("R_multiple","mean"),
        median_R=("R_multiple","median"),
        avg_hold_hours=("hold_hours","mean")
    ).reset_index()

    overall["section"]="overall"; by_zone["section"]="by_zone"
    summary = pd.concat([overall, by_zone], ignore_index=True)
    summary.to_csv(summary_path, index=False)

    print(f"Saved {len(df)} trades to {trades_path}")
    print(f"Saved summary to {summary_path}")
    print("\n=== Overall ==="); print(overall.to_string(index=False))
    print("\n=== By zone ==="); print(by_zone.sort_values("avg_R", ascending=False).to_string(index=False))

# ------------- Main -------------
def main():
    all_trades=[]
    for coin in COINS:
        try:
            print(f"Backtesting {coin} ...")
            df_1h = fetch_with_warmup(coin, 3600,  START_DATE, END_DATE)
            df_1d = fetch_with_warmup(coin, 86400, START_DATE, END_DATE)
            all_trades.extend(run_backtest_for_coin(coin))
        except Exception as e:
            print(f"[ERR] {coin}: {e}")
    save_trades_and_summary(all_trades, TRADES_CSV, SUMMARY_CSV)

# VWAP helper (used above)
def session_vwap_band(df_1h: pd.DataFrame, idx: int, band_pct: float):
    ts = df_1h["time"].iloc[idx]
    s = ts.normalize()
    mask = (df_1h["time"] >= s) & (df_1h["time"] <= ts)
    d = df_1h.loc[mask]
    if d.empty or d["volume"].sum() <= 0: return (None, None, None)
    pv = (d["close"]*d["volume"]).sum(); v = d["volume"].sum()
    vwap = float(pv/v)
    return (vwap*(1-band_pct), vwap*(1+band_pct), vwap)

if __name__ == "__main__":
    main()
