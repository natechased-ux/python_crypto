#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtest: MFE/MAE at 1..12 hours after entry (Zones + Stoch-only entries)
- Entries: zone touch + reversal direction + Stoch RSI strict (40/60)
- No exits (no SL/TP). We only measure excursions after entry.
- Zones:
    FIB_SHORT (7d/1H), FIB_MEDIUM (14d/1D), FIB_LONG (30d/1D)
    EMA50_1H, EMA200_1H, EMA20_1D, EMA50_1D, EMA100_1D, EMA200_1D  (± band)
- Chunked candle fetching + warmup so indicators hold over long spans

Output:
  trades_mfe_mae_1to12h.csv

R definition:
  1R = |entry - (zone boundary ± SL_BUFFER)|  (same normalization you’ve used)
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

# Fib windows (bars)
SHORT_LOOKBACK_HOURS = 7 * 24    # 7d on 1H
MEDIUM_LOOKBACK_DAYS = 14        # 14d on 1D
LONG_LOOKBACK_DAYS   = 30        # 30d on 1D

# EMA band width (±)
EMA_BAND_PCT = 0.005             # 0.5%

# Stoch RSI strict (only filter)
STOCH_RSI_PERIOD = 14
STOCH_SMOOTH_K   = 3
STOCH_SMOOTH_D   = 3
LONG_K_MAX  = 40.0               # LONG: K>D & K<40
SHORT_K_MIN = 60.0               # SHORT: K<D & K>60

# Fib pocket
GOLDEN_MIN = 0.618
GOLDEN_MAX = 0.66
GOLDEN_TOL = 0.0025

# Risk normalization buffer (for R)
SL_BUFFER = 0.01                 # zone boundary ±1%

# How far to look after entry
HORIZON_HOURS_MAX = 12

# I/O
TRADES_CSV = "trades_mfe_mae_1to12h.csv"

# Coinbase API & chunking
CB_BASE = "https://api.exchange.coinbase.com"
MAX_BARS_PER_CALL = 300
WARMUP_BARS = { 3600: 250, 21600: 80, 86400: 220 }  # warmup for indicators

# ================== TA / UTILS ==================
def rsi(series: pd.Series, period=14):
    d = series.diff()
    up = d.clip(lower=0.0); down = -d.clip(upper=0.0)
    ru = up.ewm(alpha=1/period, adjust=False).mean()
    rd = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd.replace(0,1e-10))
    return 100 - (100/(1+rs))

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
    for period, name in [(50, "EMA50_1H"), (200, "EMA200_1H")]:
        if i+1 >= period:
            val = float(ema(df_1h["close"].iloc[:i+1], period).iloc[-1])
            zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
            side = detect_reversal(prev, curr, zmin, zmax)
            if side: entries.append((name, val, val, zmin, zmax, side))

    # EMA 1D
    def add_ema_1d(period, nm):
        if len(df_1d_hist) >= period:
            val = float(ema(df_1d_hist["close"], period).iloc[-1])
            zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
            side = detect_reversal(prev, curr, zmin, zmax)
            if side: entries.append((nm, val, val, zmin, zmax, side))
    add_ema_1d(20,"EMA20_1D"); add_ema_1d(50,"EMA50_1D"); add_ema_1d(100,"EMA100_1D"); add_ema_1d(200,"EMA200_1D")

    return entries

# ---------- BACKTEST RUN ----------
def run_backtest_for_coin(coin: str):
    df_1h = fetch_with_warmup(coin, 3600,  START_DATE, END_DATE)
    df_1d = fetch_with_warmup(coin, 86400, START_DATE, END_DATE)
    # Indicators
    K, D = stoch_rsi(df_1h["close"], STOCH_RSI_PERIOD, STOCH_SMOOTH_K, STOCH_SMOOTH_D)
    warmup = max(SHORT_LOOKBACK_HOURS, 200)

    trades = []
    # priority: 1D EMAs > FIB long/med > 1H EMAs > FIB short
    priority = {
        "EMA200_1D":11,"EMA100_1D":10,"EMA50_1D":9,"EMA20_1D":8,
        "FIB_LONG":7,"FIB_MEDIUM":6,
        "EMA200_1H":4,"EMA50_1H":3,
        "FIB_SHORT":2
    }

    for i in range(warmup, len(df_1h)-HORIZON_HOURS_MAX-1):
        entries = build_zones(df_1h, df_1d, i)
        if not entries: continue
        # directional candidates
        entries = [e for e in entries if e[5] is not None]
        if not entries: continue
        entries.sort(key=lambda x: priority.get(x[0],1), reverse=True)
        zone, swing_low, swing_high, zmin, zmax, side = entries[0]

        # Stoch strict (only filter)
        kv = float(K.iloc[i]) if not pd.isna(K.iloc[i]) else None
        dv = float(D.iloc[i]) if not pd.isna(D.iloc[i]) else None
        if kv is None or dv is None: continue
        stoch_ok = (kv>dv and kv<LONG_K_MAX) if side=="LONG" else (kv<dv and kv>SHORT_K_MIN)
        if not stoch_ok: continue

        entry = float(df_1h["close"].iloc[i])

        # Risk (for R-normalization) based on zone boundary buffer
        if side == "LONG":
            sl_level = zmin*(1-SL_BUFFER)
            risk = entry - sl_level
        else:
            sl_level = zmax*(1+SL_BUFFER)
            risk = sl_level - entry
        if risk <= 0: continue

        # Compute MFE/MAE at horizons 1..12h
        out = {
            "coin": coin, "entry_time_utc": df_1h["time"].iloc[i].strftime("%Y-%m-%d %H:%M"),
            "side": side, "zone_source": zone,
            "entry": entry, "risk_R": risk, "zmin": zmin, "zmax": zmax,
        }

        # track rolling min/max as we step forward bar-by-bar
        running_high = entry
        running_low  = entry
        for h in range(1, HORIZON_HOURS_MAX+1):
            bar = df_1h.iloc[i+h]
            bh, bl = float(bar["high"]), float(bar["low"])
            if bh > running_high: running_high = bh
            if bl < running_low:  running_low = bl

            # Price excursions since entry up to this horizon
            if side == "LONG":
                mfe_px = running_high - entry
                mae_px = entry - running_low
            else:
                mfe_px = entry - running_low
                mae_px = running_high - entry

            # In R units
            mfe_R = mfe_px / risk
            mae_R = mae_px / risk

            out[f"mfe_h{h}_px"] = round(mfe_px, 8)
            out[f"mae_h{h}_px"] = round(mae_px, 8)
            out[f"mfe_h{h}_R"]  = round(mfe_R, 4)
            out[f"mae_h{h}_R"]  = round(mae_R, 4)

        trades.append(out)

    return trades

# ------------- Save -------------
def save_trades(trades, path):
    if not trades:
        print("No trades produced."); return
    # Collect all keys
    cols = sorted({k for t in trades for k in t.keys()}, key=lambda x: (x.startswith("mfe"), x))
    # Ensure stable column order (meta first, then horizons)
    meta = ["coin","entry_time_utc","side","zone_source","entry","risk_R","zmin","zmax"]
    hcols = []
    for h in range(1, HORIZON_HOURS_MAX+1):
        hcols += [f"mfe_h{h}_px", f"mae_h{h}_px", f"mfe_h{h}_R", f"mae_h{h}_R"]
    cols = meta + hcols

    df = pd.DataFrame(trades)
    # If any column missing (edge cases), fill with NaN
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    df = df[cols]
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} trades to {path}")

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
