#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtest: Timeframe Fib + EMA + Liquidity Sweeps + Session VWAP Reversal
(Stoch-only gate, Multi-TP + BE)
+ BTC Stoch RSI confluence (optional)
+ Momentum-fade confirmation (optional): MACD histogram slope & RSI slope

Outputs:
  - reversal_timeframe_plus_ema_liq_vwap_trades.csv
  - reversal_timeframe_plus_ema_liq_vwap_summary.csv
"""

import time
import pandas as pd
import requests
import csv
import numpy as np
from datetime import datetime, timedelta, timezone

# ============ USER CONFIG ============
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
START_DATE = "2025-05-01"
END_DATE   = "2025-08-25"

# Timeframe Fib windows
TINY_LOOKBACK_HOURS  = 2 * 24   # 2d on 1H
SHORT_LOOKBACK_HOURS = 7 * 24   # 7d on 1H
MEDIUM_LOOKBACK_DAYS = 14       # 14d on 1D
LONG_LOOKBACK_DAYS   = 30       # 30d on 1D

# EMA band width (±%)
EMA_BAND_PCT = 0.0025  # 0.25%
# Liquidity sweep params (1H)
SWEEP_LOOKBACK_HOURS = 48
SWEEP_BAND_PCT       = 0.0015
# VWAP band (intraday UTC session; reset at 00:00 UTC)
VWAP_BAND_PCT = 0.003  # 0.3%

# Stoch RSI strict thresholds
STOCH_RSI_PERIOD = 14
STOCH_SMOOTH_K   = 3
STOCH_SMOOTH_D   = 3
LONG_K_MAX  = 40.0   # LONG: K > D and K < 40
SHORT_K_MIN = 60.0   # SHORT: K < D and K > 60

# Fibonacci pocket
GOLDEN_MIN = 0.618
GOLDEN_MAX = 0.66
GOLDEN_TOL = 0.0025

# Risk/Targets
SL_BUFFER       = 0.01   # 1% beyond zone boundary
TP1_MULTIPLIER  = 1.0
TP2_MULTIPLIER  = 1.5
TP1_PCT         = 0.5
TP2_PCT         = 0.5

# Execution
ONE_TRADE_PER_DAY = False
MAX_HOLD_HOURS    = 7 * 24
SLIPPAGE_PCT      = 0.0
FEES_PCT          = 0.0

# BTC confluence configuration (optional)
REQUIRE_BTC_CONFLUENCE = False
BTC_PRODUCT_ID = "BTC-USD"

# Momentum-fade configuration (optional)
REQUIRE_MOMENTUM_FADE = False  # keep False to log only; set True to enforce
MFADE_WINDOW = 3               # bars for slope (MACD hist & RSI)
MFADE_USE_RSI = True           # include RSI slope in the fade check
MFADE_USE_MACD = True          # include MACD histogram slope in the fade check

# Outputs
TRADES_CSV  = "reversal_timeframe_plus_ema_liq_vwap_trades.csv"
SUMMARY_CSV = "reversal_timeframe_plus_ema_liq_vwap_summary.csv"

CB_BASE = "https://api.exchange.coinbase.com"
MAX_BARS_PER_CALL = 300  # Coinbase typical cap
# ============ END CONFIG ============

# Warmup cushions so indicators are valid before START_DATE
WARMUP_BARS = {
    3600:  250,   # ~250 hours
    21600: 80,    # ~20 days of 6H bars
    86400: 220,   # ~220 days (EMA200_1D + margin)
}

# ---------- Helpers ----------
def fetch_candles_range(product_id: str, granularity: int,
                        start_dt: pd.Timestamp, end_dt: pd.Timestamp,
                        session: requests.Session | None = None) -> pd.DataFrame:
    """Paged fetch using Coinbase 'start'/'end' params to stitch >300 bars."""
    sess = session or requests.Session()
    url = f"{CB_BASE}/products/{product_id}/candles"
    step_secs = granularity * MAX_BARS_PER_CALL
    out = []
    curr_end = end_dt
    while curr_end > start_dt:
        curr_start = max(start_dt, curr_end - pd.Timedelta(seconds=step_secs))
        params = {
            "granularity": granularity,
            "start": curr_start.isoformat(),
            "end": curr_end.isoformat(),
        }
        r = sess.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            break
        df_chunk = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
        df_chunk["time"] = pd.to_datetime(df_chunk["time"], unit="s", utc=True)
        out.append(df_chunk)
        earliest = df_chunk["time"].min()
        if pd.isna(earliest):
            break
        curr_end = earliest - pd.Timedelta(seconds=1)
        time.sleep(0.1)
    if not out:
        raise ValueError(f"No candle data returned for {product_id} @ {granularity}")
    df = pd.concat(out, ignore_index=True)
    df.drop_duplicates(subset=["time"], inplace=True)
    df.sort_values("time", inplace=True)
    df = df[(df["time"] >= start_dt) & (df["time"] < end_dt)].reset_index(drop=True)
    return df

def fetch_with_warmup(product_id: str, granularity: int,
                      start_date: str, end_date: str) -> pd.DataFrame:
    s = pd.Timestamp(start_date, tz="UTC")
    e = pd.Timestamp(end_date,   tz="UTC")
    warm_bars = WARMUP_BARS.get(granularity, 0)
    warm_secs = warm_bars * granularity
    s_warm = s - pd.Timedelta(seconds=warm_secs)
    return fetch_candles_range(product_id, granularity, s_warm, e)

def slice_last(df: pd.DataFrame, bars: int) -> pd.DataFrame:
    return df.iloc[-bars:] if bars < len(df) else df

def fib_golden_zone(low: float, high: float, tol: float = GOLDEN_TOL):
    span = high - low
    f618 = low + span * GOLDEN_MIN
    f66  = low + span * GOLDEN_MAX
    return f618*(1 - tol), f66*(1 + tol)

def rsi(series: pd.Series, period=14):
    d = series.diff()
    up   = d.clip(lower=0.0)
    down = -d.clip(upper=0.0)
    ru = up.ewm(alpha=1/period, adjust=False).mean()
    rd = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd.replace(0, 1e-10))
    return 100 - (100 / (1 + rs))

def stoch_rsi(close: pd.Series, period: int, k: int, d: int):
    r = rsi(close, period)
    lo = r.rolling(period).min()
    hi = r.rolling(period).max()
    st = (r - lo) / (hi - lo + 1e-10) * 100.0
    K = st.rolling(k).mean()
    D = K.rolling(d).mean()
    return K, D

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def macd_hist(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def session_vwap_band(df_1h: pd.DataFrame, idx: int, band_pct: float):
    if idx < 0: return (None, None, None)
    ts = df_1h["time"].iloc[idx]
    start_day = ts.normalize()  # UTC 00:00
    mask = (df_1h["time"] >= start_day) & (df_1h["time"] <= ts)
    df = df_1h.loc[mask]
    if df.empty or df["volume"].sum() <= 0:
        return (None, None, None)
    pv = (df["close"] * df["volume"]).sum()
    v = df["volume"].sum()
    vwap = float(pv / v)
    return (vwap*(1-band_pct), vwap*(1+band_pct), vwap)

def detect_reversal_side(prev_price: float, curr_price: float, zmin: float, zmax: float):
    if zmin is None or zmax is None:
        return None
    if (prev_price < zmin) and (zmin <= curr_price <= zmax):
        return "SHORT"
    if (prev_price > zmax) and (zmin <= curr_price <= zmax):
        return "LONG"
    return None

def in_zone(price: float, zmin: float, zmax: float) -> bool:
    return (zmin is not None) and (zmax is not None) and (zmin <= price <= zmax)

def recent_swing_levels(df_1h: pd.DataFrame, hours: int):
    w = slice_last(df_1h, hours)
    return float(w["high"].max()), float(w["low"].min())

# ---------- BTC Stoch RSI prefetch & map (NEW) ----------
def precompute_btc_stoch_map():
    df_btc = fetch_with_warmup(BTC_PRODUCT_ID, 3600, START_DATE, END_DATE)
    Kb, Db = stoch_rsi(df_btc["close"], STOCH_RSI_PERIOD, STOCH_SMOOTH_K, STOCH_SMOOTH_D)
    btc_map = {df_btc["time"].iloc[i]: (float(Kb.iloc[i]) if not pd.isna(Kb.iloc[i]) else None,
                                        float(Db.iloc[i]) if not pd.isna(Db.iloc[i]) else None)
               for i in range(len(df_btc))}
    return btc_map

def btc_aligns(side: str, kv: float | None, dv: float | None) -> bool | None:
    if kv is None or dv is None:
        return None
    if side == "LONG":
        return (kv > dv) and (kv < LONG_K_MAX)
    else:
        return (kv < dv) and (kv > SHORT_K_MIN)

# ---------- Momentum fade check (MACD hist slope + RSI slope) ----------
def momentum_fade_ok(side: str, rsi_series: pd.Series, hist_series: pd.Series, idx: int, window: int):
    """Returns (ok, rsi_slope, hist_slope) at index idx over lookback 'window'."""
    if idx - window < 0:
        return (False, float('nan'), float('nan'))
    rsi_slope = float(rsi_series.iloc[idx] - rsi_series.iloc[idx - window])
    hist_slope = float(hist_series.iloc[idx] - hist_series.iloc[idx - window])

    if side == "LONG":
        # bearish momentum fading: want RSI rising and MACD hist rising toward zero
        ok = ((not np.isnan(rsi_slope) and rsi_slope > 0) if MFADE_USE_RSI else True) and \
             ((not np.isnan(hist_slope) and hist_slope > 0) if MFADE_USE_MACD else True)
    else:
        # bullish momentum fading: want RSI falling and MACD hist falling toward zero
        ok = ((not np.isnan(rsi_slope) and rsi_slope < 0) if MFADE_USE_RSI else True) and \
             ((not np.isnan(hist_slope) and hist_slope < 0) if MFADE_USE_MACD else True)

    return (ok, rsi_slope, hist_slope)


# ---------- Core backtest ----------
def run_backtest_for_coin(coin: str, btc_kd_map=None):
    df_1h = fetch_with_warmup(coin, 3600,  START_DATE, END_DATE)
    df_1d = fetch_with_warmup(coin, 86400, START_DATE, END_DATE)
    if len(df_1h) < SHORT_LOOKBACK_HOURS + 50:
        return []

    # Indicators on 1H
    K, D = stoch_rsi(df_1h["close"], STOCH_RSI_PERIOD, STOCH_SMOOTH_K, STOCH_SMOOTH_D)
    rsi_ = rsi(df_1h["close"], 14)
    _, _, macd_hist_ = macd_hist(df_1h["close"])

    trades = []
    last_trade_day = None
    warmup = max(SHORT_LOOKBACK_HOURS, 200, 50)

    for i in range(warmup, len(df_1h) - 1):
        t = df_1h.iloc[i]["time"]
        if ONE_TRADE_PER_DAY and last_trade_day == t.date():
            continue

        curr = float(df_1h["close"].iloc[i])
        prev = float(df_1h["close"].iloc[i-1])

        entries = []
        flags   = {}  # presence snapshot

        # ---- Fib timeframe zones ----
        w_t = slice_last(df_1h.iloc[:i+1], TINY_LOOKBACK_HOURS)
        sl_t, sh_t = float(w_t["low"].min()), float(w_t["high"].max())
        zmin_t, zmax_t = fib_golden_zone(sl_t, sh_t)
        flags["in_fib_tiny"]  = in_zone(curr, zmin_t, zmax_t)
        side_fib_tiny = detect_reversal_side(prev, curr, zmin_t, zmax_t)
        if side_fib_tiny: entries.append(("FIB_TINY", sl_t, sh_t, zmin_t, zmax_t, side_fib_tiny))

        w_s = slice_last(df_1h.iloc[:i+1], SHORT_LOOKBACK_HOURS)
        sl_s, sh_s = float(w_s["low"].min()), float(w_s["high"].max())
        zmin_s, zmax_s = fib_golden_zone(sl_s, sh_s)
        flags["in_fib_short"] = in_zone(curr, zmin_s, zmax_s)
        side_fib_short = detect_reversal_side(prev, curr, zmin_s, zmax_s)
        if side_fib_short: entries.append(("FIB_SHORT", sl_s, sh_s, zmin_s, zmax_s, side_fib_short))

        # 1D zones
        df_1d_hist = df_1d[df_1d["time"] <= t]
        flags["in_fib_medium"] = False; flags["in_fib_long"] = False
        side_fib_medium = side_fib_long = None
        zmin_m = zmax_m = sl_m = sh_m = None
        zmin_l = zmax_l = sl_l = sh_l = None

        if len(df_1d_hist) >= MEDIUM_LOOKBACK_DAYS:
            w_m = slice_last(df_1d_hist, MEDIUM_LOOKBACK_DAYS)
            sl_m, sh_m = float(w_m["low"].min()), float(w_m["high"].max())
            zmin_m, zmax_m = fib_golden_zone(sl_m, sh_m)
            flags["in_fib_medium"] = in_zone(curr, zmin_m, zmax_m)
            side_fib_medium = detect_reversal_side(prev, curr, zmin_m, zmax_m)
            if side_fib_medium: entries.append(("FIB_MEDIUM", sl_m, sh_m, zmin_m, zmax_m, side_fib_medium))

        if len(df_1d_hist) >= LONG_LOOKBACK_DAYS:
            w_l = slice_last(df_1d_hist, LONG_LOOKBACK_DAYS)
            sl_l, sh_l = float(w_l["low"].min()), float(w_l["high"].max())
            zmin_l, zmax_l = fib_golden_zone(sl_l, sh_l)
            flags["in_fib_long"] = in_zone(curr, zmin_l, zmax_l)
            side_fib_long = detect_reversal_side(prev, curr, zmin_l, zmax_l)
            if side_fib_long: entries.append(("FIB_LONG", sl_l, sh_l, zmin_l, zmax_l, side_fib_long))

        # ---- 1H EMAs ----
        def ema_zone(series, period):
            if len(series) < period: return (None, None, None, None, False)
            val = float(ema(series, period).iloc[-1])
            zmin = val * (1 - EMA_BAND_PCT)
            zmax = val * (1 + EMA_BAND_PCT)
            inside = in_zone(curr, zmin, zmax)
            side = detect_reversal_side(prev, curr, zmin, zmax)
            return val, zmin, zmax, side, inside

        ema50_val,  e50min,  e50max,  side_e50,  in_e50  = ema_zone(df_1h["close"].iloc[:i+1], 50)
        ema200_val, e200min, e200max, side_e200, in_e200 = ema_zone(df_1h["close"].iloc[:i+1], 200)

        flags["in_ema50_1h"]  = in_e50
        flags["in_ema200_1h"] = in_e200
        if side_e50:  entries.append(("EMA50_1H",  ema50_val,  ema50_val,  e50min,  e50max,  side_e50))
        if side_e200: entries.append(("EMA200_1H", ema200_val, ema200_val, e200min, e200max, side_e200))

        # ---- 1D EMAs ----
        def last_ema_1d(period):
            if len(df_1d_hist) >= max(period, 5):
                return float(ema(df_1d_hist["close"], period).iloc[-1])
            return None

        ema20d = last_ema_1d(20)
        ema50d = last_ema_1d(50)
        ema100d= last_ema_1d(100)
        ema200d= last_ema_1d(200)

        def add_ema_1d_zone(name, val):
            if val is None or val <= 0:
                flags[f"in_{name.lower()}"] = False
                return
            zmin = val * (1 - EMA_BAND_PCT)
            zmax = val * (1 + EMA_BAND_PCT)
            inside = in_zone(curr, zmin, zmax)
            side   = detect_reversal_side(prev, curr, zmin, zmax)
            flags[f"in_{name.lower()}"] = inside
            if side: entries.append((name, val, val, zmin, zmax, side))

        add_ema_1d_zone("EMA20_1D",  ema20d)
        add_ema_1d_zone("EMA50_1D",  ema50d)
        add_ema_1d_zone("EMA100_1D", ema100d)
        add_ema_1d_zone("EMA200_1D", ema200d)

        # ---- Liquidity sweeps (1H) ----
        recent_hi, recent_lo = recent_swing_levels(df_1h.iloc[:i+1], SWEEP_LOOKBACK_HOURS)
        hi_min = recent_hi * (1 - SWEEP_BAND_PCT)
        hi_max = recent_hi * (1 + SWEEP_BAND_PCT)
        in_sweep_hi = in_zone(curr, hi_min, hi_max)
        side_sweep_hi = detect_reversal_side(prev, curr, hi_min, hi_max)  # ascending => SHORT
        flags["in_liq_sweep_high_1h"] = in_sweep_hi
        if side_sweep_hi:
            entries.append(("LIQ_SWEEP_HIGH_1H", recent_hi, recent_hi, hi_min, hi_max, side_sweep_hi))
        lo_min = recent_lo * (1 - SWEEP_BAND_PCT)
        lo_max = recent_lo * (1 + SWEEP_BAND_PCT)
        in_sweep_lo = in_zone(curr, lo_min, lo_max)
        side_sweep_lo = detect_reversal_side(prev, curr, lo_min, lo_max)  # descending => LONG
        flags["in_liq_sweep_low_1h"] = in_sweep_lo
        if side_sweep_lo:
            entries.append(("LIQ_SWEEP_LOW_1H", recent_lo, recent_lo, lo_min, lo_max, side_sweep_lo))

        # ---- Session VWAP (1H) ----
        vmin, vmax, vwap_val = session_vwap_band(df_1h.iloc[:i+1], i, VWAP_BAND_PCT)
        in_vwap = in_zone(curr, vmin, vmax) if vmin is not None else False
        side_vwap = detect_reversal_side(prev, curr, vmin, vmax) if vmin is not None else None
        flags["in_vwap_session_1h"] = in_vwap
        if side_vwap:
            entries.append(("VWAP_SESSION_1H", vwap_val, vwap_val, vmin, vmax, side_vwap))

        # If nothing directional fired, continue
        entries = [e for e in entries if e[5] is not None]
        if not entries:
            continue

        # ---- Priority ----
        priority = {
            "EMA200_1D":11,"EMA100_1D":10,"EMA20_1D":9,"EMA50_1D":8,
            "FIB_LONG":7,"FIB_MEDIUM":6,
            "LIQ_SWEEP_HIGH_1H":5,"LIQ_SWEEP_LOW_1H":5,
            "VWAP_SESSION_1H":4,"EMA200_1H":3,"EMA50_1H":2,
            "FIB_SHORT":2,"FIB_TINY":1
        }
        entries.sort(key=lambda x: priority.get(x[0], 1), reverse=True)
        zone_source, swing_low, swing_high, zmin, zmax, side = entries[0]

        # ---- Stoch RSI REQUIRED (per-coin) ----
        kv = float(K.iloc[i]) if not pd.isna(K.iloc[i]) else None
        dv = float(D.iloc[i]) if not pd.isna(D.iloc[i]) else None
        if kv is None or dv is None:
            continue
        stoch_ok = (kv > dv and kv < LONG_K_MAX) if side == "LONG" else (kv < dv and kv > SHORT_K_MIN)
        if not stoch_ok:
            continue

        # ---- BTC Stoch RSI confluence (optional) ----
        btc_k = btc_d = None
        btc_ok = None
        if btc_kd_map is not None:
            btc_k, btc_d = btc_kd_map.get(t, (None, None))
            btc_ok = btc_aligns(side, btc_k, btc_d)
            if REQUIRE_BTC_CONFLUENCE and (btc_ok is not True):
                continue

        # ---- Momentum fade (optional): MACD hist slope and/or RSI slope ----
        mf_ok = True
        rsi_slope = float('nan')
        hist_slope = float('nan')
        if MFADE_USE_RSI or MFADE_USE_MACD:
            mf_ok, rsi_slope, hist_slope = momentum_fade_ok(side, rsi_, macd_hist_, i, MFADE_WINDOW)
            if REQUIRE_MOMENTUM_FADE and not mf_ok:
                continue

        # Entry & risk
        entry = curr
        if side == "LONG":
            sl = zmin * (1 - SL_BUFFER)
            risk = entry - sl
            tp1 = entry + TP1_MULTIPLIER * risk
            tp2 = entry + TP2_MULTIPLIER * risk
        else:
            sl = zmax * (1 + SL_BUFFER)
            risk = sl - entry
            tp1 = entry - TP1_MULTIPLIER * risk
            tp2 = entry - TP2_MULTIPLIER * risk
        if risk <= 0:
            continue

        # Manage forward (BE after TP1)
        highest_before_exit = entry
        lowest_before_exit  = entry
        sl_work = sl
        tp1_hit = False
        tp2_hit = False
        exit_price = entry
        outcome = "Timeout"
        hold_hours = 0

        for j in range(i+1, len(df_1h)):
            bar = df_1h.iloc[j]
            bh, bl = float(bar["high"]), float(bar["low"])

            if bh > highest_before_exit: highest_before_exit = bh
            if bl < lowest_before_exit:  lowest_before_exit  = bl

            if not tp1_hit:
                if side == "LONG" and bh >= tp1:
                    tp1_hit = True; sl_work = max(sl_work, entry)
                elif side == "SHORT" and bl <= tp1:
                    tp1_hit = True; sl_work = min(sl_work, entry)

            if tp1_hit and not tp2_hit:
                if side == "LONG" and bh >= tp2:
                    tp2_hit = True; exit_price = tp2; outcome = "TP1_TP2"; hold_hours = j - i; break
                elif side == "SHORT" and bl <= tp2:
                    tp2_hit = True; exit_price = tp2; outcome = "TP1_TP2"; hold_hours = j - i; break

            if side == "LONG" and bl <= sl_work:
                exit_price = sl_work; outcome = "TP1_BE" if tp1_hit else "SL"; hold_hours = j - i; break
            if side == "SHORT" and bh >= sl_work:
                exit_price = sl_work; outcome = "TP1_BE" if tp1_hit else "SL"; hold_hours = j - i; break

            if (j - i) >= MAX_HOLD_HOURS:
                exit_price = float(df_1h["close"].iloc[j])
                outcome = "Timeout" if not tp1_hit else "Timeout_after_TP1"
                hold_hours = j - i; break

        netR = (exit_price - entry) / risk if side == "LONG" else (entry - exit_price) / risk
        netR -= FEES_PCT

        trades.append({
            "coin": coin,
            "entry_time_utc": df_1h.iloc[i]["time"].strftime("%Y-%m-%d %H:%M"),
            "side": side,
            "zone_source": zone_source,
            "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
            "outcome": outcome, "R_multiple": round(netR,4), "hold_hours": hold_hours,
            "tp1_hit": bool(tp1_hit), "tp2_hit": bool(tp2_hit),
            "highest_before_exit": highest_before_exit, "lowest_before_exit": lowest_before_exit,
            "zmin": zmin, "zmax": zmax, "swing_low": swing_low, "swing_high": swing_high,
            "stoch_k": kv, "stoch_d": dv,
            # BTC confluence logging
            "btc_stoch_k": btc_k, "btc_stoch_d": btc_d, "btc_stoch_align": (btc_ok if btc_ok is not None else False),
            "btc_confluence_enforced": REQUIRE_BTC_CONFLUENCE,
            # Momentum fade logging
            "mf_ok": bool(mf_ok), "rsi_slope": rsi_slope, "macd_hist_slope": hist_slope,
            "mf_enforced": REQUIRE_MOMENTUM_FADE,
            # Presence snapshot
            "in_fib_tiny":  bool(flags.get("in_fib_tiny", False)),
            "in_fib_short": bool(flags.get("in_fib_short", False)),
            "in_fib_medium":bool(flags.get("in_fib_medium", False)),
            "in_fib_long":  bool(flags.get("in_fib_long", False)),
            "in_ema50_1h":  bool(flags.get("in_ema50_1h", False)),
            "in_ema200_1h": bool(flags.get("in_ema200_1h", False)),
            "in_ema20_1d":  bool(flags.get("in_ema20_1d", False)),
            "in_ema50_1d":  bool(flags.get("in_ema50_1d", False)),
            "in_ema100_1d": bool(flags.get("in_ema100_1d", False)),
            "in_ema200_1d": bool(flags.get("in_ema200_1d", False)),
            "in_liq_sweep_high_1h": bool(flags.get("in_liq_sweep_high_1h", False)),
            "in_liq_sweep_low_1h":  bool(flags.get("in_liq_sweep_low_1h", False)),
            "in_vwap_session_1h":   bool(flags.get("in_vwap_session_1h", False)),
        })

        last_trade_day = t.date()

    return trades

# ---------- Summary ----------
def summarize_and_save(trades: list):
    if not trades:
        print("No trades produced.")
        return

    # Save trades CSV
    fieldnames = list(trades[0].keys())
    with open(TRADES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for row in trades: w.writerow(row)

    df = pd.DataFrame(trades)

    def win_rate_rpos(s: pd.Series) -> float:
        return (s > 0).mean() if len(s) else 0.0

    # Overall
    overall = pd.DataFrame([{
        "trades": len(df),
        "win_rate": win_rate_rpos(df["R_multiple"]),
        "TP1_win_rate": df["tp1_hit"].mean(),
        "avg_R": df["R_multiple"].mean(),
        "median_R": df["R_multiple"].median(),
        "avg_hold_hours": df["hold_hours"].mean(),
        "TP1_rate": df["tp1_hit"].mean(),
        "TP2_rate": df["tp2_hit"].mean(),
        "section": "overall"
    }])

    # By zone
    by_zone = df.groupby("zone_source").agg(
        trades=("zone_source","count"),
        win_rate=("R_multiple", win_rate_rpos),
        TP1_win_rate=("tp1_hit","mean"),
        TP1_rate=("tp1_hit","mean"),
        TP2_rate=("tp2_hit","mean"),
        avg_R=("R_multiple","mean"),
        median_R=("R_multiple","median"),
        avg_hold_hours=("hold_hours","mean")
    ).reset_index()
    by_zone["section"] = "by_zone"

    # BTC confluence slice
    tables = [overall, by_zone]
    if "btc_stoch_align" in df.columns:
        by_btc = df.groupby("btc_stoch_align").agg(
            trades=("btc_stoch_align","count"),
            win_rate=("R_multiple", win_rate_rpos),
            avg_R=("R_multiple","mean"),
            TP1_rate=("tp1_hit","mean"),
            TP2_rate=("tp2_hit","mean"),
        ).reset_index().rename(columns={"btc_stoch_align":"btc_align"})
        by_btc["section"] = "by_btc_confluence"
        tables.append(by_btc)

    # Momentum fade slice
    if "mf_ok" in df.columns:
        by_mf = df.groupby("mf_ok").agg(
            trades=("mf_ok","count"),
            win_rate=("R_multiple", win_rate_rpos),
            avg_R=("R_multiple","mean"),
            TP1_rate=("tp1_hit","mean"),
            TP2_rate=("tp2_hit","mean"),
        ).reset_index().rename(columns={"mf_ok":"momentum_fade_ok"})
        by_mf["section"] = "by_momentum_fade"
        tables.append(by_mf)

    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"Saved {len(df)} trades to {TRADES_CSV}")
    print(f"Saved summary to {SUMMARY_CSV}")
    print("\n=== Overall ===");               print(overall.to_string(index=False))
    print("\n=== By zone ===");              print(by_zone.to_string(index=False))
    if 'by_btc' in locals():    print("\n=== By BTC confluence ==="); print(by_btc.to_string(index=False))
    if 'by_mf' in locals():     print("\n=== By Momentum Fade ==="); print(by_mf.to_string(index=False))

# ---------- Main ----------
def main():
    print("Prefetching BTC Stoch RSI map for confluence logging ...")
    btc_kd_map = precompute_btc_stoch_map()
    all_trades = []
    for coin in COINS:
        try:
            print(f"Backtesting {coin} ...")
            all_trades.extend(run_backtest_for_coin(coin, btc_kd_map=btc_kd_map))
        except Exception as e:
            print(f"[ERR] {coin}: {e}")
            continue
    summarize_and_save(all_trades)

if __name__ == "__main__":
    main()
