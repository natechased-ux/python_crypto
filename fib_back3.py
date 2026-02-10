#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtest (Filtered + Hybrid TP, Chunked Fetch, Two-Pass per-coin TP2)
Pass 1: Trend/ADX/Stoch-slope + Pinbar filters; track MFE_R per coin -> compute TP2_R = P75(MFE_R)
Pass 2: Same filters + exits:
        - TP1 = 1 x ATR(14, 1H), take 50% (no instant BE)
        - Runner exits at TP2 = per-coin P75(MFE_R)
Outputs:
  - trades_pass1.csv, summary_pass1.csv
  - trades_pass2.csv, summary_pass2.csv
"""

import csv, time, re
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# ===================== CONFIG =====================
COINS = [
    # merged list (edit freely)
    "BTC-USD","ETH-USD","XRP-USD","SOL-USD","ADA-USD","AVAX-USD","DOGE-USD","DOT-USD",
    "LINK-USD","ATOM-USD","NEAR-USD","ARB-USD","OP-USD","MATIC-USD","SUI-USD",
    "INJ-USD","AAVE-USD","LTC-USD","BCH-USD","ETC-USD","ALGO-USD","FIL-USD","ICP-USD",
    "RNDR-USD","STX-USD","JTO-USD","PYTH-USD","GRT-USD","FTM-USD","SEI-USD",
    "AR-USD","RUNE-USD","ENS-USD","FLOW-USD","KSM-USD","KAVA-USD",
    "DYDX-USD","WLD-USD","HBAR-USD","JUP-USD","ORDI-USD","STRK-USD",
    "ONDO-USD","SUPER-USD","SAGA-USD","LDO-USD","BEAM-USD","POL-USD",
    "ZETA-USD","ZRO-USD","NOT-USD","TIA-USD",
    "WIF-USD","MAGIC-USD","APE-USD","JASMY-USD","SYRUP-USD","FARTCOIN-USD",
    "AERO-USD","FET-USD","CRV-USD","TAO-USD","XCN-USD","UNI-USD","MKR-USD",
    "TOSHI-USD","TRUMP-USD","PEPE-USD","XLM-USD","MOODENG-USD","BONK-USD",
    "POPCAT-USD","QNT-USD","IP-USD","PNUT-USD","APT-USD","ENA-USD","TURBO-USD",
    "BERA-USD","MASK-USD","SAND-USD","MORPHO-USD","MANA-USD","C98-USD","AXS-USD"
]

# Backtest window (UTC)
START_DATE = "2025-02-01"
END_DATE   = "2025-08-25"

# Fib windows
TINY_LOOKBACK_HOURS  = 2 * 24   # 2d on 1H
SHORT_LOOKBACK_HOURS = 7 * 24   # 7d on 1H
MEDIUM_LOOKBACK_DAYS = 14       # 14d on 1D
LONG_LOOKBACK_DAYS   = 30       # 30d on 1D

# EMA bands (±)
EMA_BAND_PCT = 0.005

# Liquidity sweeps (1H)
SWEEP_LOOKBACK_HOURS = 48
SWEEP_BAND_PCT       = 0.0015

# VWAP (UTC session)
VWAP_BAND_PCT = 0.003

# Stoch RSI strict thresholds
STOCH_RSI_PERIOD = 14
STOCH_SMOOTH_K   = 3
STOCH_SMOOTH_D   = 3
LONG_K_MAX  = 40.0  # longs: K>D & K<40
SHORT_K_MIN = 60.0  # shorts: K<D & K>60

# Fibonacci pocket
GOLDEN_MIN = 0.618
GOLDEN_MAX = 0.66
GOLDEN_TOL = 0.0025

# Risk/targets (Hybrid)
SL_BUFFER       = 0.01      # SL at zone boundary (±1%)
TP1_ATR_MULT    = 1.0       # TP1 = 1 x ATR(14,1H)
TP2_QTL         = 0.75      # per-coin TP2_R = P75(MFE_R) from pass 1
TP1_PCT         = 0.5       # 50% at TP1 (no BE)
USE_ATR_TRAIL   = False     # keep OFF for this run (use fixed coin TP2_R)

# Execution
ONE_TRADE_PER_DAY = False
MAX_HOLD_HOURS    = 7 * 24
SLIPPAGE_PCT      = 0.0
FEES_PCT          = 0.0

# I/O
TRADES_P1  = "trades_pass1.csv"
SUMMARY_P1 = "summary_pass1.csv"
TRADES_P2  = "trades_pass2.csv"
SUMMARY_P2 = "summary_pass2.csv"

# Coinbase
CB_BASE = "https://api.exchange.coinbase.com"
MAX_BARS_PER_CALL = 300

# Warmup bars
WARMUP_BARS = { 3600: 250, 21600: 80, 86400: 220 }

# ============= TA / UTILS =============
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
def build_zones(df_1h, df_1d, i, EMA_BAND_PCT, SWEEP_LOOKBACK_HOURS, SWEEP_BAND_PCT, VWAP_BAND_PCT, GOLDEN_TOL):
    t = df_1h["time"].iloc[i]; curr = float(df_1h["close"].iloc[i]); prev = float(df_1h["close"].iloc[i-1])
    entries = []; flags = {}
    # FIB 1H tiny/short
    for name, look in [("FIB_TINY", 2*24), ("FIB_SHORT", 7*24)]:
        w = df_1h.iloc[:i+1].tail(look)
        sl, sh = float(w["low"].min()), float(w["high"].max())
        zmin, zmax = fib_golden_zone(sl, sh, GOLDEN_TOL)
        flags[f"in_{name.lower()}"] = (zmin is not None and zmax is not None and zmin <= curr <= zmax)
        side = detect_reversal(prev, curr, zmin, zmax)
        if side: entries.append((name, sl, sh, zmin, zmax, side))
    # FIB 1D medium/long
    df_1d_hist = df_1d[df_1d["time"] <= t]
    if len(df_1d_hist) >= MEDIUM_LOOKBACK_DAYS:
        w = df_1d_hist.tail(MEDIUM_LOOKBACK_DAYS); sl, sh = float(w["low"].min()), float(w["high"].max())
        zmin, zmax = fib_golden_zone(sl, sh, GOLDEN_TOL)
        flags["in_fib_medium"] = (zmin <= curr <= zmax); side = detect_reversal(prev, curr, zmin, zmax)
        if side: entries.append(("FIB_MEDIUM", sl, sh, zmin, zmax, side))
    else: flags["in_fib_medium"]=False
    if len(df_1d_hist) >= LONG_LOOKBACK_DAYS:
        w = df_1d_hist.tail(LONG_LOOKBACK_DAYS); sl, sh = float(w["low"].min()), float(w["high"].max())
        zmin, zmax = fib_golden_zone(sl, sh, GOLDEN_TOL)
        flags["in_fib_long"] = (zmin <= curr <= zmax); side = detect_reversal(prev, curr, zmin, zmax)
        if side: entries.append(("FIB_LONG", sl, sh, zmin, zmax, side))
    else: flags["in_fib_long"]=False
    # EMA 1H
    for period, name in [(50,"EMA50_1H"),(200,"EMA200_1H")]:
        if i+1 >= period:
            val = float(ema(df_1h["close"].iloc[:i+1], period).iloc[-1])
            zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
            flags[f"in_{name.lower()}"] = (zmin <= curr <= zmax)
            side = detect_reversal(prev, curr, zmin, zmax)
            if side: entries.append((name, val, val, zmin, zmax, side))
        else: flags[f"in_{name.lower()}"]=False
    # EMA 1D
    def add_ema_1d(period, nm):
        if len(df_1d_hist) >= period:
            val = float(ema(df_1d_hist["close"], period).iloc[-1])
            zmin,zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
            flags[f"in_{nm.lower()}"] = (zmin <= curr <= zmax)
            side = detect_reversal(prev, curr, zmin, zmax)
            if side: entries.append((nm, val, val, zmin, zmax, side))
        else: flags[f"in_{nm.lower()}"]=False
    add_ema_1d(20,"EMA20_1D"); add_ema_1d(50,"EMA50_1D"); add_ema_1d(100,"EMA100_1D"); add_ema_1d(200,"EMA200_1D")
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
    # VWAP band (UTC session)
    vmin, vmax, vwap_val = session_vwap_band(df_1h, i, VWAP_BAND_PCT)
    flags["in_vwap_session_1h"] = (vmin is not None and vmin <= curr <= vmax)
    side_vwap = detect_reversal(prev, curr, vmin, vmax) if vmin is not None else None
    if side_vwap: entries.append(("VWAP_SESSION_1H", vwap_val, vwap_val, vmin, vmax, side_vwap))

    return entries, flags, t

def session_vwap_band(df_1h: pd.DataFrame, idx: int, band_pct: float):
    ts = df_1h["time"].iloc[idx]
    s = ts.normalize()
    mask = (df_1h["time"] >= s) & (df_1h["time"] <= ts)
    df = df_1h.loc[mask]
    if df.empty or df["volume"].sum() <= 0: return (None,None,None)
    pv = (df["close"]*df["volume"]).sum(); v = df["volume"].sum()
    vwap = float(pv/v)
    return (vwap*(1-band_pct), vwap*(1+band_pct), vwap)

def bollinger_position(close: pd.Series, period=20, mult=2.0):
    m = close.rolling(period).mean()
    s = close.rolling(period).std(ddof=0)
    pos = (close - m)/(mult*s + 1e-10)
    return m, m + mult*s, m - mult*s, pos

# ------------- Single pass runner -------------
def run_pass(coin, df_1h, df_1d, pass_mode, percoin_tp2R=None):
    """
    pass_mode: "pass1" or "pass2"
    percoin_tp2R: dict coin->TP2_R (only used in pass2)
    Returns list of trade dicts.
    """
    # Indicators
    K, D = stoch_rsi(df_1h["close"], STOCH_RSI_PERIOD, STOCH_SMOOTH_K, STOCH_SMOOTH_D)
    stoch_slope = K.diff()
    ema200_1d_series = ema(df_1d["close"], 200) if len(df_1d) >= 200 else pd.Series(index=df_1d.index, dtype=float)
    adx_1d_series = adx(df_1d["high"], df_1d["low"], df_1d["close"], 14)
    atr_1h_series = atr(df_1h["high"], df_1h["low"], df_1h["close"], 14)

    trades=[]; warmup=max(SHORT_LOOKBACK_HOURS, 250)
    for i in range(warmup, len(df_1h)-1):
        entries, flags, t = build_zones(df_1h, df_1d, i, EMA_BAND_PCT, SWEEP_LOOKBACK_HOURS, SWEEP_BAND_PCT, VWAP_BAND_PCT, GOLDEN_TOL)
        entries = [e for e in entries if e[5] is not None]
        if not entries: continue

        # Priority by strength
        priority = {
            "EMA200_1D":11,"EMA100_1D":10,"EMA20_1D":9,"EMA50_1D":8,
            "FIB_LONG":7,"FIB_MEDIUM":6,
            "LIQ_SWEEP_HIGH_1H":5,"LIQ_SWEEP_LOW_1H":5,
            "VWAP_SESSION_1H":4,"EMA200_1H":3,"EMA50_1H":2,
            "FIB_SHORT":2,"FIB_TINY":1
        }
        entries.sort(key=lambda x:priority.get(x[0],1), reverse=True)
        zone_source, swing_low, swing_high, zmin, zmax, side = entries[0]
        curr=float(df_1h["close"].iloc[i]); prev=float(df_1h["close"].iloc[i-1])

        # ===== Filters: Trend + ADX + Stoch slope + Pinbar + Stoch strict =====
        # Daily context at entry
        dmask = (df_1d["time"] <= t); d_idx = dmask.sum()-1
        if d_idx < 0: continue
        ema200_1d_val = float(ema200_1d_series.iloc[d_idx]) if len(ema200_1d_series)>d_idx else np.nan
        daily_close = float(df_1d["close"].iloc[d_idx])
        daily_trend_up = (daily_close > ema200_1d_val)
        # Trend alignment
        if side=="LONG" and not daily_trend_up: continue
        if side=="SHORT" and daily_trend_up: continue
        # ADX
        adx_val = float(adx_1d_series.iloc[d_idx]) if len(adx_1d_series)>d_idx else 0.0
        if adx_val <= 20: continue
        # Stoch slope direction
        kv=float(K.iloc[i]); dv=float(D.iloc[i]); slope=float(stoch_slope.iloc[i]) if not pd.isna(stoch_slope.iloc[i]) else 0.0
        if side=="LONG" and slope <= 0: continue
        if side=="SHORT" and slope >= 0: continue
        # Pinbar (simple wick heuristic on entry candle)
        entry_row=df_1h.iloc[i]; body=abs(float(entry_row["close"]-entry_row["open"]))
        rng=max(float(entry_row["high"]-entry_row["low"]),1e-10)
        up_w=float(entry_row["high"]-max(entry_row["close"],entry_row["open"]))
        dn_w=float(min(entry_row["close"],entry_row["open"])-entry_row["low"])
        pinbar = ((up_w/rng>0.6) or (dn_w/rng>0.6)) and (body/rng<0.3)
        if not pinbar: continue
        # Stoch strict
        stoch_ok = (kv>dv and kv<LONG_K_MAX) if side=="LONG" else (kv<dv and kv>SHORT_K_MIN)
        if not stoch_ok: continue

        # Confluence count (for analysis only)
        zone_count = sum(int(v) for v in flags.values())

        # ========== Risk/targets ==========
        entry = curr
        if side=="LONG":
            sl = zmin*(1-SL_BUFFER); risk = entry - sl
        else:
            sl = zmax*(1+SL_BUFFER); risk = sl - entry
        if risk<=0: continue

        # TP1 = ATR-based
        atr_now = float(atr_1h_series.iloc[i]) if not pd.isna(atr_1h_series.iloc[i]) else 0.0
        if side=="LONG":
            tp1 = entry + TP1_ATR_MULT*atr_now
        else:
            tp1 = entry - TP1_ATR_MULT*atr_now

        # Provisional TP2 for pass1 (to compute MFE we don't need TP2, but keep symmetrical)
        if pass_mode=="pass1":
            tp2_R = 1.5  # placeholder (won't affect MFE tracking)
        else:
            tp2_R = percoin_tp2R.get(coin, 1.5)

        if pass_mode=="pass2":
            # TP2 distance in price from R units
            if side=="LONG":
                tp2 = entry + tp2_R * risk
            else:
                tp2 = entry - tp2_R * risk
        else:
            tp2 = None

        # ====== Manage forward (TP1 partial, runner to TP2 in pass2) ======
        highest=entry; lowest=entry
        sl_work=sl; tp1_hit=False; tp2_hit=False
        tp1_bar=-1; tp2_bar=-1; hold_hours=0
        exit_price=entry; outcome="Timeout"

        def R_at(px): return (px-entry)/risk if side=="LONG" else (entry-px)/risk

        for j in range(i+1, len(df_1h)):
            bar=df_1h.iloc[j]; bh=float(bar["high"]); bl=float(bar["low"])
            if bh>highest: highest=bh
            if bl<lowest:  lowest=bl

            # TP1 check
            if not tp1_hit:
                if (side=="LONG" and bh>=tp1) or (side=="SHORT" and bl<=tp1):
                    tp1_hit=True; tp1_bar=j-i
                    # No instant BE; partial exit simulated later

            # TP2 (pass2 only)
            if pass_mode=="pass2" and tp1_hit and not tp2_hit and tp2 is not None:
                if (side=="LONG" and bh>=tp2) or (side=="SHORT" and bl<=tp2):
                    tp2_hit=True; tp2_bar=j-i; exit_price=tp2; outcome="TP1_TP2"; hold_hours=j-i; break

            # SL check
            if side=="LONG" and bl<=sl_work:
                exit_price=sl_work; outcome="TP1_BE" if tp1_hit else "SL"; hold_hours=j-i; break
            if side=="SHORT" and bh>=sl_work:
                exit_price=sl_work; outcome="TP1_BE" if tp1_hit else "SL"; hold_hours=j-i; break

            # Timeout
            if (j-i)>=MAX_HOLD_HOURS:
                exit_price=float(df_1h["close"].iloc[j]); outcome="Timeout" if not tp1_hit else "Timeout_after_TP1"; hold_hours=j-i; break

        # Compute R
        netR = R_at(exit_price)

        # Partial exit simulation:
        if tp1_hit:
            if pass_mode=="pass2":
                # 50% at TP1, runner=exit or TP2
                tp1_R = R_at(tp1)
                if tp2_hit:
                    partial_R = 0.5*tp1_R + 0.5*tp2_R
                else:
                    partial_R = 0.5*tp1_R + 0.5*netR
            else:
                partial_R = netR  # pass1: not used as final metric
        else:
            partial_R = netR

        # MFE/MAE in R up to exit
        mfe_R = R_at(highest) if side=="LONG" else R_at(lowest)
        mae_R = R_at(lowest) if side=="LONG" else R_at(highest)

        trade = {
            "coin": coin, "time": t.strftime("%Y-%m-%d %H:%M"), "side": side, "zone_source": zone_source,
            "zone_count": zone_count, "entry": entry, "sl": sl, "tp1": tp1, "tp2_R_used": (tp2_R if pass_mode=="pass2" else np.nan),
            "outcome": outcome, "R_multiple": round(float(partial_R if pass_mode=="pass2" else netR),4),
            "tp1_hit": bool(tp1_hit), "tp2_hit": bool(tp2_hit), "tp1_bar": tp1_bar, "tp2_bar": tp2_bar,
            "hold_hours": hold_hours, "mfe_R": round(float(mfe_R),4), "mae_R": round(float(mae_R),4),
        }
        trades.append(trade)

    return trades

# ------------ Summary save ------------
def save_trades_and_summary(trades, trades_path, summary_path):
    if not trades:
        print("No trades")
        return
    fieldnames = list(trades[0].keys())
    df = pd.DataFrame(trades)
    df.to_csv(trades_path, index=False)

    def win_rate(s): return (s>0).mean() if len(s) else 0.0
    overall = pd.DataFrame([{
        "trades": len(df),
        "win_rate": win_rate(df["R_multiple"]),
        "TP1_rate": df["tp1_hit"].mean(),
        "TP2_rate": df["tp2_hit"].mean(),
        "avg_R": df["R_multiple"].mean(),
        "median_R": df["R_multiple"].median(),
        "avg_hold_hours": df["hold_hours"].mean()
    }])
    by_zone = df.groupby("zone_source").agg(
        trades=("zone_source","count"),
        win_rate=("R_multiple", win_rate),
        TP1_rate=("tp1_hit","mean"),
        TP2_rate=("tp2_hit","mean"),
        avg_R=("R_multiple","mean"),
        median_R=("R_multiple","median"),
        avg_hold_hours=("hold_hours","mean")
    ).reset_index()

    # save stacked summary
    overall["section"]="overall"; by_zone["section"]="by_zone"
    summary = pd.concat([overall, by_zone], ignore_index=True)
    summary.to_csv(summary_path, index=False)

    print(f"Saved {len(df)} trades to {trades_path}")
    print(f"Saved summary to {summary_path}")
    print("\n=== Overall ==="); print(overall.to_string(index=False))
    print("\n=== By zone ==="); print(by_zone.sort_values("avg_R", ascending=False).to_string(index=False))

# ---------- Main (two-pass) ----------
def main():
    # Pass 1: run with filters, compute per-coin P75(MFE_R)
    all_p1=[]
    for coin in COINS:
        try:
            print(f"[Pass1] {coin} ...")
            df_1h = fetch_with_warmup(coin, 3600,  START_DATE, END_DATE)
            df_1d = fetch_with_warmup(coin, 86400, START_DATE, END_DATE)
            all_p1.extend(run_pass(coin, df_1h, df_1d, pass_mode="pass1"))
        except Exception as e:
            print(f"[Pass1 ERR] {coin}: {e}")
    save_trades_and_summary(all_p1, TRADES_P1, SUMMARY_P1)

    # Compute per-coin TP2_R from P75(MFE_R) of pass1 trades
    tp2_map={}
    if all_p1:
        df_p1 = pd.DataFrame(all_p1)
        for coin, g in df_p1.groupby("coin"):
            if len(g)>=10:
                tp2_map[coin] = float(g["mfe_R"].quantile(TP2_QTL))
            else:
                tp2_map[coin] = 1.5  # fallback
    print("Per-coin TP2_R (P75 MFE):", {k:round(v,3) for k,v in tp2_map.items() if pd.notna(v)})

    # Pass 2: rerun with per-coin TP2 and hybrid exits
    all_p2=[]
    for coin in COINS:
        try:
            print(f"[Pass2] {coin} ...")
            df_1h = fetch_with_warmup(coin, 3600,  START_DATE, END_DATE)
            df_1d = fetch_with_warmup(coin, 86400, START_DATE, END_DATE)
            all_p2.extend(run_pass(coin, df_1h, df_1d, pass_mode="pass2", percoin_tp2R=tp2_map))
        except Exception as e:
            print(f"[Pass2 ERR] {coin}: {e}")
    save_trades_and_summary(all_p2, TRADES_P2, SUMMARY_P2)

# VWAP helper used above
def session_vwap_band(df_1h: pd.DataFrame, idx: int, band_pct: float):
    ts = df_1h["time"].iloc[idx]
    s = ts.normalize()
    mask = (df_1h["time"] >= s) & (df_1h["time"] <= ts)
    df = df_1h.loc[mask]
    if df.empty or df["volume"].sum() <= 0:
        return (None, None, None)
    pv = (df["close"] * df["volume"]).sum()
    v = df["volume"].sum()
    vwap = float(pv / v)
    return (vwap*(1-band_pct), vwap*(1+band_pct), vwap)

if __name__ == "__main__":
    main()
