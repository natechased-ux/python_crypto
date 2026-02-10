#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest: MFE/MAE at 1..12 hours after entry (Zones + 15m Volume Spike + 15m Stoch RSI)
- Entries: 1H/1D structure zones (same as before), BUT confirmation uses 15m timeframe
           with a volume spike AND strict 15m Stoch RSI.
- No exits (no SL/TP). We only measure excursions after entry.
- Zones:
    FIB_SHORT (7d/1H), FIB_MEDIUM (14d/1D), FIB_LONG (30d/1D)
    EMA50_1H, EMA200_1H, EMA20_1D, EMA50_1D, EMA100_1D, EMA200_1D  (± band)
- New:
    • 15m volume spike filter (percentile-based OR multiple-of-median)
    • 15m Stoch RSI strict (40/60) with 2-bar persistence
    • Entry promoted to first qualifying 15m bar after a zone touch (Option B)
- Chunked candle fetching + warmup so indicators hold over long spans

Output:
  trades_mfe_mae_1to12h_15m_spike.csv

R definition:
  Risk is measured from entry to the nearest zone boundary plus SL_BUFFER (same as before),
  so outcomes remain comparable in R-units with the previous study.

NOTE: This script intentionally removes ANY 1H Stoch RSI usage. All oscillator confirmation is at 15m.
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

# Backtest window (UTC) — adjust freely
START_DATE = "2025-05-01"
END_DATE   = "2025-08-25"

# Fib windows (bars)
SHORT_LOOKBACK_HOURS = 7 * 24    # 7d on 1H
MEDIUM_LOOKBACK_DAYS = 14        # 14d on 1D
LONG_LOOKBACK_DAYS   = 30        # 30d on 1D

# EMA band width (±)
EMA_BAND_PCT = 0.0025  # ±0.25%

# 15m Stoch RSI params (strict)
STOCH_RSI_PERIOD = 14
STOCH_SMOOTH_K   = 3
STOCH_SMOOTH_D   = 3
LONG_K_MAX  = 40.0               # LONG: K>D & K<40
SHORT_K_MIN = 60.0               # SHORT: K<D & K>60
STOCH_PERSIST_BARS = 2           # require 2 closed bars of agreement

# Fib pocket
GOLDEN_MIN = 0.618
GOLDEN_MAX = 0.66
GOLDEN_TOL = 0.0025

# Risk normalization buffer (for R)
SL_BUFFER = 0.01                 # zone boundary ±1%

# How far to look after entry (in HOURS, measured on 1H series)
HORIZON_HOURS_MAX = 12

# 15m volume spike settings
V_LOOKBACK_DAYS       = 7
V_METHOD              = "percentile"   # "percentile" or "x_median"
V_PCTL                = 95.0           # used if method="percentile"
V_MULTIPLE_OF_MEDIAN  = 1.8            # used if method="x_median"
RANGE_MEDIAN_LOOKBACK = 7              # days for median true range baseline

# Entry timing window after a zone touch (number of 15m bars to scan forward)
ENTRY_WINDOW_15M_BARS = 2  # 0..2 bars (0–30 minutes)

# Cooldown between trades per coin (in minutes) to avoid spam
PER_COIN_COOLDOWN_MIN = 45

# I/O
TRADES_CSV = "trades_mfe_mae_1to12h_15m_spike.csv"

# Coinbase API & chunking
CB_BASE = "https://api.exchange.coinbase.com"
MAX_BARS_PER_CALL = 300
WARMUP_BARS = { 900: 3000, 3600: 250, 21600: 80, 86400: 220 }  # add large 15m warmup for rolling stats

# ================== TA / UTILS ==================
def rsi(series: pd.Series, period=14):
    d = series.diff()
    up = d.clip(lower=0.0); down = -d.clip(upper=0.0)
    ru = up.ewm(alpha=1/period, adjust=False).mean()
    rd = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd.replace(0,1e-10))
    return 100 - (100 / (1 + rs))

def stoch_rsi(close: pd.Series, period: int, k: int, d_: int):
    r = rsi(close, period)
    lo = r.rolling(period).min(); hi = r.rolling(period).max()
    st = (r - lo)/(hi - lo + 1e-10)*100.0
    K = st.rolling(k).mean(); D = K.rolling(d_).mean()
    return K, D

def ema(series: pd.Series, period: int): return series.ewm(span=period, adjust=False).mean()

def fib_golden_zone(lo: float, hi: float, tol: float = GOLDEN_TOL):
    span = hi - lo; f618 = lo+span*GOLDEN_MIN; f66 = lo+span*GOLDEN_MAX
    return f618*(1-tol), f66*(1+tol)

def detect_reversal(prev: float, curr: float, zmin: float, zmax: float):
    if zmin is None or zmax is None: return None
    if (prev < zmin) and (zmin <= curr <= zmax): return "SHORT"
    if (prev > zmax) and (zmin <= curr <= zmax): return "LONG"
    return None

def true_range(row):
    return float(row["high"]) - float(row["low"])

# --------- Chunked fetching + warmup ---------
def fetch_candles_range(product_id: str, granularity: int,
                        start_dt: pd.Timestamp, end_dt: pd.Timestamp,
                        session=None) -> pd.DataFrame:
    sess = session or requests.Session()
    url = f"{CB_BASE}/products/{product_id}/candles"
    out=[]; step_secs=granularity*MAX_BARS_PER_CALL; curr_end=end_dt
    while curr_end > start_dt:
        curr_start = max(start_dt, curr_end - pd.Timedelta(seconds=step_secs))
        params={"granularity":granularity,"start":curr_start.isoformat(),"end":curr_end.isoformat()}
        r=sess.get(url, params=params, timeout=20); r.raise_for_status()
        data=r.json()
        if not isinstance(data,list) or not data: break
        df_chunk=pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
        df_chunk["time"]=pd.to_datetime(df_chunk["time"], unit="s", utc=True)
        out.append(df_chunk)
        earliest=df_chunk["time"].min()
        if pd.isna(earliest): break
        curr_end=earliest - pd.Timedelta(seconds=1)
        time.sleep(0.05)
    if not out: raise ValueError(f"No data for {product_id}@{granularity}")
    df=pd.concat(out, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    return df[(df["time"]>=start_dt) & (df["time"]<end_dt)].reset_index(drop=True)

def fetch_with_warmup(product_id: str, granularity: int, start_date: str, end_date: str) -> pd.DataFrame:
    s=pd.Timestamp(start_date, tz="UTC"); e=pd.Timestamp(end_date, tz="UTC")
    warm_bars=WARMUP_BARS.get(granularity,0); s_warm=s-pd.Timedelta(seconds=warm_bars*granularity)
    return fetch_candles_range(product_id, granularity, s_warm, e)

# ---------- ZONES (EMA + FIB SHORT/MED/LONG) ----------
def build_zones(df_1h, df_1d, i):
    """Return directional zone hits for bar i and candidates list (zone, swing_low, swing_high, zmin, zmax, side)."""
    t = df_1h["time"].iloc[i]
    curr = float(df_1h["close"].iloc[i])
    prev = float(df_1h["close"].iloc[i-1])
    entries = []

    # FIB_SHORT (7d/1H)
    w = df_1h.iloc[:i+1].tail(SHORT_LOOKBACK_HOURS)
    sl_s, sh_s = float(w["low"].min()), float(w["high"].max())
    zmin_s, zmax_s = fib_golden_zone(sl_s, sh_s)
    side_s = detect_reversal(prev, curr, zmin_s, zmax_s)
    if side_s: entries.append(("FIB_SHORT", sl_s, sh_s, zmin_s, zmax_s, side_s))

    # FIB_MEDIUM (14d/1D) & FIB_LONG (30d/1D)
    df_1d_hist = df_1d[df_1d["time"] <= t]
    if len(df_1d_hist) >= MEDIUM_LOOKBACK_DAYS:
        wm = df_1d_hist.tail(MEDIUM_LOOKBACK_DAYS)
        sl_m, sh_m = float(wm["low"].min()), float(wm["high"].max())
        zmin_m, zmax_m = fib_golden_zone(sl_m, sh_m)
        side_m = detect_reversal(prev, curr, zmin_m, zmax_m)
        if side_m: entries.append(("FIB_MEDIUM", sl_m, sh_m, zmin_m, zmax_m, side_m))
    if len(df_1d_hist) >= LONG_LOOKBACK_DAYS:
        wl = df_1d_hist.tail(LONG_LOOKBACK_DAYS)
        sl_l, sh_l = float(wl["low"].min()), float(wl["high"].max())
        zmin_l, zmax_l = fib_golden_zone(sl_l, sh_l)
        side_l = detect_reversal(prev, curr, zmin_l, zmax_l)
        if side_l: entries.append(("FIB_LONG", sl_l, sh_l, zmin_l, zmax_l, side_l))

    # EMA 1H
    def add_ema_1h(period, nm):
        val = float(ema(df_1h["close"].iloc[:i+1], period).iloc[-1])
        zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
        side = detect_reversal(prev, curr, zmin, zmax)
        if side: entries.append((nm, val, val, zmin, zmax, side))
    add_ema_1h(50,"EMA50_1H"); add_ema_1h(200,"EMA200_1H")

    # EMA 1D
    def add_ema_1d(period, nm):
        if len(df_1d_hist) >= period+10:
            val = float(ema(df_1d_hist["close"], period).iloc[-1])
            zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
            side = detect_reversal(prev, curr, zmin, zmax)
            if side: entries.append((nm, val, val, zmin, zmax, side))
    add_ema_1d(20,"EMA20_1D"); add_ema_1d(50,"EMA50_1D"); add_ema_1d(100,"EMA100_1D"); add_ema_1d(200,"EMA200_1D")

    return entries

# ---------- 15m helpers (volume spike + stoch) ----------
def add_rolling_volume_stats(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Add rolling stats needed for spike detection: pct95 or x_median envelope + median TR baseline."""
    df = df_15m.copy()
    win = V_LOOKBACK_DAYS * 24 * 4  # 15m bars in V_LOOKBACK_DAYS
    base = df["volume"].rolling(win)
    if V_METHOD == "percentile":
        df["v_thresh"] = base.quantile(V_PCTL/100.0)
    else:
        df["v_thresh"] = base.median() * V_MULTIPLE_OF_MEDIAN
    # median true range baseline for safety
    df["tr"] = (df["high"] - df["low"]).abs()
    df["tr_med"] = df["tr"].rolling(RANGE_MEDIAN_LOOKBACK*24*4).median()
    return df

def add_15m_stoch(df_15m: pd.DataFrame) -> pd.DataFrame:
    df = df_15m.copy()
    K, D = stoch_rsi(df["close"], STOCH_RSI_PERIOD, STOCH_SMOOTH_K, STOCH_SMOOTH_D)
    df["K"], df["D"] = K, D
    # persistence flags
    df["stoch_long_ok"] = (df["K"] > df["D"]) & (df["K"] < LONG_K_MAX)
    df["stoch_short_ok"]= (df["K"] < df["D"]) & (df["K"] > SHORT_K_MIN)
    df["stoch_long_ok_persist"]  = df["stoch_long_ok"].rolling(STOCH_PERSIST_BARS).apply(lambda x: 1.0 if x.all() else 0.0, raw=False).astype(bool)
    df["stoch_short_ok_persist"] = df["stoch_short_ok"].rolling(STOCH_PERSIST_BARS).apply(lambda x: 1.0 if x.all() else 0.0, raw=False).astype(bool)
    return df

def within_zone(px: float, zmin: float, zmax: float) -> bool:
    return (zmin is not None) and (zmax is not None) and (zmin <= px <= zmax)

# ---------- BACKTEST RUN ----------
def run_backtest_for_coin(coin: str):
    # Fetch candles
    df_1h  = fetch_with_warmup(coin, 3600,  START_DATE, END_DATE)
    df_1d  = fetch_with_warmup(coin, 86400, START_DATE, END_DATE)
    df_15m = fetch_with_warmup(coin, 900,   START_DATE, END_DATE)

    # Add 15m features
    df_15m = add_rolling_volume_stats(df_15m)
    df_15m = add_15m_stoch(df_15m)

    # priority: 1D EMAs > FIB long/med > 1H EMAs > FIB short
    priority = {
        "EMA200_1D":11,"EMA100_1D":10,"EMA50_1D":9,"EMA20_1D":8,
        "FIB_LONG":7,"FIB_MEDIUM":6,
        "EMA200_1H":4,"EMA50_1H":3,
        "FIB_SHORT":2
    }

    trades = []
    last_entry_time_by_coin = None

    # iterate 1H bars for structure/zone touches
    warmup = max(SHORT_LOOKBACK_HOURS, 200)
    for i in range(warmup, len(df_1h)-HORIZON_HOURS_MAX-1):
        t1h  = df_1h["time"].iloc[i]
        entries = build_zones(df_1h, df_1d, i)
        if not entries: continue
        # directional candidates only
        entries = [e for e in entries if e[5] is not None]
        if not entries: continue
        # choose highest priority
        entries = sorted(entries, key=lambda e: priority.get(e[0],0), reverse=True)
        zone, sl, sh, zmin, zmax, side = entries[0]

        # cooldown (per coin)
        if last_entry_time_by_coin is not None:
            if (t1h - last_entry_time_by_coin) < pd.Timedelta(minutes=PER_COIN_COOLDOWN_MIN):
                continue

        # SCAN 15m window: same bar and up to ENTRY_WINDOW_15M_BARS after
        # Find 15m bars whose time is in [t1h, t1h + window*15m]
        t_end = t1h + pd.Timedelta(minutes=15*ENTRY_WINDOW_15M_BARS)
        mask = (df_15m["time"]>=t1h) & (df_15m["time"]<=t_end)
        cand_15m = df_15m[mask].copy()
        if cand_15m.empty: 
            continue

        # Restrict to bars inside the chosen zone band
        cand_15m["in_zone"] = cand_15m["close"].apply(lambda px: within_zone(px, zmin, zmax))
        cand_15m = cand_15m[cand_15m["in_zone"]]
        if cand_15m.empty:
            continue

        # Volume spike + Stoch RSI
        if V_METHOD == "percentile":
            spike_flag = cand_15m["volume"] >= cand_15m["v_thresh"]
        else:
            spike_flag = cand_15m["volume"] >= cand_15m["v_thresh"]
        range_ok = cand_15m["tr"] >= cand_15m["tr_med"]
        if side == "LONG":
            stoch_ok = cand_15m["stoch_long_ok_persist"]
        else:
            stoch_ok = cand_15m["stoch_short_ok_persist"]

        cand_15m["qualify"] = spike_flag & range_ok & stoch_ok

        if not cand_15m["qualify"].any():
            continue

        # first qualifying 15m bar is our entry
        entry_row = cand_15m[cand_15m["qualify"]].iloc[0]
        entry_time = entry_row["time"]
        entry = float(entry_row["close"])

        # Compute risk vs nearest boundary same as before
        if side=="LONG":
            risk = max(1e-8, entry - (zmin*(1-SL_BUFFER)))
        else:
            risk = max(1e-8, (zmax*(1+SL_BUFFER)) - entry)

        # MFE/MAE measured forward on 1H bars from the first 1H bar >= entry_time
        j = int(df_1h.index[df_1h["time"] >= entry_time][0]) if (df_1h["time"] >= entry_time).any() else None
        if j is None or (j+HORIZON_HOURS_MAX+1)>=len(df_1h):
            continue

        out = {
            "coin": coin, 
            "entry_time_utc": entry_time.strftime("%Y-%m-%d %H:%M"),
            "arm_time_1h_utc": t1h.strftime("%Y-%m-%d %H:%M"),
            "side": side, "zone_source": zone,
            "entry": round(entry, 8), "risk_R": round(risk, 8), 
            "zmin": round(zmin, 8) if zmin is not None else None, 
            "zmax": round(zmax, 8) if zmax is not None else None,
            # diagnostics
            "k_15m": round(float(entry_row["K"]), 6) if not pd.isna(entry_row["K"]) else None,
            "d_15m": round(float(entry_row["D"]), 6) if not pd.isna(entry_row["D"]) else None,
            "v_15m": round(float(entry_row["volume"]), 6),
            "v_thresh": round(float(entry_row["v_thresh"]), 6) if not pd.isna(entry_row["v_thresh"]) else None,
            "tr_15m": round(float(entry_row["tr"]), 8) if not pd.isna(entry_row["tr"]) else None,
            "tr_med": round(float(entry_row["tr_med"]), 8) if not pd.isna(entry_row["tr_med"]) else None,
            "stoch_persist": STOCH_PERSIST_BARS,
            "v_method": V_METHOD,
        }

        running_high = entry
        running_low  = entry
        for h in range(1, HORIZON_HOURS_MAX+1):
            bar = df_1h.iloc[j+h]
            bh, bl = float(bar["high"]), float(bar["low"])
            if bh > running_high: running_high = bh
            if bl < running_low:  running_low  = bl

            if side == "LONG":
                mfe_px = running_high - entry
                mae_px = entry - running_low
            else:
                mfe_px = entry - running_low
                mae_px = running_high - entry

            mfe_R = mfe_px / risk
            mae_R = mae_px / risk

            out[f"mfe_h{h}_px"] = round(mfe_px, 8)
            out[f"mae_h{h}_px"] = round(mae_px, 8)
            out[f"mfe_h{h}_R"]  = round(mfe_R, 6)
            out[f"mae_h{h}_R"]  = round(mae_R, 6)

        trades.append(out)
        last_entry_time_by_coin = entry_time  # cooldown anchor

    return trades

# ---------- CSV ----------
def save_trades(trades, path):
    if not trades:
        print("No trades — nothing to write.")
        return
    keys = sorted(set().union(*[t.keys() for t in trades]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(trades)
    print(f"Wrote {len(trades)} rows to {path}")

# ------------- Main -------------
def main():
    all_trades = []
    for coin in COINS:
        try:
            print(f"Backtesting {coin} ...")
            trades = run_backtest_for_coin(coin)
            all_trades.extend(trades)
        except Exception as e:
            print(f"[ERR] {coin}: {e}")
    save_trades(all_trades, TRADES_CSV)

if __name__ == "__main__":
    main()
