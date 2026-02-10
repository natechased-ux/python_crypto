#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mean-Reversion Backtest Using Zone Overshoot + Re-Entry
------------------------------------------------------
Core idea:
  1) Price *overshoots* a reversal zone (Fib/EMA/VWAP/sweep) on the prior bar.
  2) Current bar *re-enters* the zone (close back inside) → this is our mean-reversion trigger.
  3) Only take the trade if deviation from a mean (Session VWAP or EMA50_1H) is large enough.
  4) TP = nearest valid mean in trade direction (prefer Session VWAP, then EMA50_1H, then EMA200_1H, then EMA20/50/100/200_1D).
  5) SL = beyond the zone boundary with a small buffer; move to BE after 50% of the path to TP.

Outputs:
  - mr_zones_trades.csv
  - mr_zones_summary.csv
"""

import time
import pandas as pd
import requests
import csv
import numpy as np

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
EMA_BAND_PCT = 0.003  # 0.3%
# Liquidity sweep params (1H)
SWEEP_LOOKBACK_HOURS = 48
SWEEP_BAND_PCT       = 0.0015
# VWAP band (intraday UTC session; reset at 00:00 UTC)
VWAP_BAND_PCT = 0.003  # 0.3%

# Mean-reversion deviation threshold (minimum distance to mean to accept trade)
MIN_DEV_TO_MEAN_PCT = 0.005  # 0.5% away from chosen mean at entry

# Risk/Management
SL_BUFFER       = 0.01    # 1% beyond zone boundary
MOVE_BE_AT_FRAC = 0.50    # move SL to entry after 50% of the path to TP
MAX_HOLD_HOURS  = 48
ONE_TRADE_PER_DAY = False
SLIPPAGE_PCT    = 0.0
FEES_PCT        = 0.0

# Outputs
TRADES_CSV  = "mr_zones_trades.csv"
SUMMARY_CSV = "mr_zones_summary.csv"

CB_BASE = "https://api.exchange.coinbase.com"
MAX_BARS_PER_CALL = 300  # Coinbase typical cap
# ============ END CONFIG ============

# Warmup cushions so indicators are valid before START_DATE
WARMUP_BARS = {
    3600:  250,   # ~250 hours
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

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def fib_golden_zone(low: float, high: float, tol: float = 0.0025):
    GOLDEN_MIN, GOLDEN_MAX = 0.618, 0.66
    span = high - low
    f618 = low + span * GOLDEN_MIN
    f66  = low + span * GOLDEN_MAX
    return f618*(1 - tol), f66*(1 + tol)

def in_zone(price: float, zmin: float, zmax: float) -> bool:
    return (zmin is not None) and (zmax is not None) and (zmin <= price <= zmax)

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

def recent_swing_levels(df_1h: pd.DataFrame, hours: int):
    w = slice_last(df_1h, hours)
    return float(w["high"].max()), float(w["low"].min())

def detect_reentry(prev_close: float, prev_high: float, prev_low: float,
                   curr_close: float, zmin: float, zmax: float, side: str) -> bool:
    """Overshoot previous bar, re-enter on current close."""
    if side == "LONG":
        # Overshoot below, then close back inside
        overshoot = (prev_low < zmin)
        reenter   = in_zone(curr_close, zmin, zmax)
        return overshoot and reenter
    else:
        overshoot = (prev_high > zmax)
        reenter   = in_zone(curr_close, zmin, zmax)
        return overshoot and reenter

def choose_mean_target(side: str, entry: float, vwap_val: float | None,
                       ema50_1h: float | None, ema200_1h: float | None,
                       ema20d: float | None, ema50d: float | None, ema100d: float | None, ema200d: float | None):
    """Pick nearest mean in trade direction: VWAP > EMA50_1H > EMA200_1H > EMA20/50/100/200_1D."""
    candidates = []
    def add(name, price):
        if price is None: return
        if side == "LONG" and price > entry: candidates.append((name, float(price)))
        if side == "SHORT" and price < entry: candidates.append((name, float(price)))
    add("VWAP", vwap_val)
    add("EMA50_1H", ema50_1h)
    add("EMA200_1H", ema200_1h)
    add("EMA20_1D", ema20d); add("EMA50_1D", ema50d); add("EMA100_1D", ema100d); add("EMA200_1D", ema200d)
    if not candidates: return None, None
    # nearest
    if side == "LONG":
        name, price = min(candidates, key=lambda x: x[1] - entry)
    else:
        name, price = max(candidates, key=lambda x: x[1] - entry)
    return name, price

# ---------- Core backtest ----------
def run_backtest_for_coin(coin: str):
    df_1h = fetch_with_warmup(coin, 3600,  START_DATE, END_DATE)
    df_1d = fetch_with_warmup(coin, 86400, START_DATE, END_DATE)
    if len(df_1h) < SHORT_LOOKBACK_HOURS + 50:
        return []

    trades = []
    last_trade_day = None
    warmup = max(SHORT_LOOKBACK_HOURS, 200, 50)  # ensure enough history

    # Precompute 1H EMAs
    ema50_series  = ema(df_1h["close"], 50)
    ema200_series = ema(df_1h["close"], 200)

    for i in range(warmup, len(df_1h) - 1):
        t = df_1h.iloc[i]["time"].to_pydatetime()
        if ONE_TRADE_PER_DAY and last_trade_day == t.date():
            continue

        curr_close = float(df_1h["close"].iloc[i])
        prev_close = float(df_1h["close"].iloc[i-1])
        prev_high  = float(df_1h["high"].iloc[i-1])
        prev_low   = float(df_1h["low"].iloc[i-1])

        entries = []  # (name, sl_level, sh_level, zmin, zmax, side)
        flags   = {}

        # ---- Fib timeframe zones ----
        w_t = slice_last(df_1h.iloc[:i+1], TINY_LOOKBACK_HOURS)
        sl_t, sh_t = float(w_t["low"].min()), float(w_t["high"].max())
        zmin_t, zmax_t = fib_golden_zone(sl_t, sh_t)
        # LONG if we re-enter from below; SHORT if we re-enter from above
        if detect_reentry(prev_close, prev_high, prev_low, curr_close, zmin_t, zmax_t, "LONG"):
            entries.append(("FIB_TINY", sl_t, sh_t, zmin_t, zmax_t, "LONG"))
        if detect_reentry(prev_close, prev_high, prev_low, curr_close, zmin_t, zmax_t, "SHORT"):
            entries.append(("FIB_TINY", sl_t, sh_t, zmin_t, zmax_t, "SHORT"))

        w_s = slice_last(df_1h.iloc[:i+1], SHORT_LOOKBACK_HOURS)
        sl_s, sh_s = float(w_s["low"].min()), float(w_s["high"].max())
        zmin_s, zmax_s = fib_golden_zone(sl_s, sh_s)
        if detect_reentry(prev_close, prev_high, prev_low, curr_close, zmin_s, zmax_s, "LONG"):  entries.append(("FIB_SHORT", sl_s, sh_s, zmin_s, zmax_s, "LONG"))
        if detect_reentry(prev_close, prev_high, prev_low, curr_close, zmin_s, zmax_s, "SHORT"): entries.append(("FIB_SHORT", sl_s, sh_s, zmin_s, zmax_s, "SHORT"))

        # ---- 1D Fibs ----
        df_1d_hist = df_1d[df_1d["time"] <= df_1h.iloc[i]["time"]]
        fib_m = fib_l = None
        if len(df_1d_hist) >= MEDIUM_LOOKBACK_DAYS:
            w_m = slice_last(df_1d_hist, MEDIUM_LOOKBACK_DAYS)
            sl_m, sh_m = float(w_m["low"].min()), float(w_m["high"].max())
            zmin_m, zmax_m = fib_golden_zone(sl_m, sh_m)
            if detect_reentry(prev_close, prev_high, prev_low, curr_close, zmin_m, zmax_m, "LONG"):  entries.append(("FIB_MEDIUM", sl_m, sh_m, zmin_m, zmax_m, "LONG"))
            if detect_reentry(prev_close, prev_high, prev_low, curr_close, zmin_m, zmax_m, "SHORT"): entries.append(("FIB_MEDIUM", sl_m, sh_m, zmin_m, zmax_m, "SHORT"))
            fib_m = (sl_m, sh_m, zmin_m, zmax_m)
        if len(df_1d_hist) >= LONG_LOOKBACK_DAYS:
            w_l = slice_last(df_1d_hist, LONG_LOOKBACK_DAYS)
            sl_l, sh_l = float(w_l["low"].min()), float(w_l["high"].max())
            zmin_l, zmax_l = fib_golden_zone(sl_l, sh_l)
            if detect_reentry(prev_close, prev_high, prev_low, curr_close, zmin_l, zmax_l, "LONG"):  entries.append(("FIB_LONG", sl_l, sh_l, zmin_l, zmax_l, "LONG"))
            if detect_reentry(prev_close, prev_high, prev_low, curr_close, zmin_l, zmax_l, "SHORT"): entries.append(("FIB_LONG", sl_l, sh_l, zmin_l, zmax_l, "SHORT"))
            fib_l = (sl_l, sh_l, zmin_l, zmax_l)

        # ---- 1H EMAs as zones ----
        ema50_val  = float(ema50_series.iloc[i])
        ema200_val = float(ema200_series.iloc[i])
        e50min, e50max   = ema50_val  * (1 - EMA_BAND_PCT),  ema50_val  * (1 + EMA_BAND_PCT)
        e200min, e200max = ema200_val * (1 - EMA_BAND_PCT),  ema200_val * (1 + EMA_BAND_PCT)
        if detect_reentry(prev_close, prev_high, prev_low, curr_close, e50min,  e50max,  "LONG"):  entries.append(("EMA50_1H",  ema50_val,  ema50_val,  e50min,  e50max,  "LONG"))
        if detect_reentry(prev_close, prev_high, prev_low, curr_close, e50min,  e50max,  "SHORT"): entries.append(("EMA50_1H",  ema50_val,  ema50_val,  e50min,  e50max,  "SHORT"))
        if detect_reentry(prev_close, prev_high, prev_low, curr_close, e200min, e200max, "LONG"):  entries.append(("EMA200_1H", ema200_val, ema200_val, e200min, e200max, "LONG"))
        if detect_reentry(prev_close, prev_high, prev_low, curr_close, e200min, e200max, "SHORT"): entries.append(("EMA200_1H", ema200_val, ema200_val, e200min, e200max, "SHORT"))

        # ---- 1D EMAs for target picking later ----
        def last_ema_1d(period):
            if len(df_1d_hist) >= max(period, 5):
                return float(ema(df_1d_hist["close"], period).iloc[-1])
            return None
        ema20d = last_ema_1d(20); ema50d = last_ema_1d(50); ema100d = last_ema_1d(100); ema200d = last_ema_1d(200)

        # ---- Liquidity sweeps ----
        recent_hi, recent_lo = recent_swing_levels(df_1h.iloc[:i+1], SWEEP_LOOKBACK_HOURS)
        hi_min = recent_hi * (1 - SWEEP_BAND_PCT)
        hi_max = recent_hi * (1 + SWEEP_BAND_PCT)
        lo_min = recent_lo * (1 - SWEEP_BAND_PCT)
        lo_max = recent_lo * (1 + SWEEP_BAND_PCT)
        if detect_reentry(prev_close, prev_high, prev_low, curr_close, lo_min, lo_max, "LONG"):   entries.append(("LIQ_SWEEP_LOW_1H",  recent_lo, recent_lo, lo_min, lo_max, "LONG"))
        if detect_reentry(prev_close, prev_high, prev_low, curr_close, hi_min, hi_max, "SHORT"):  entries.append(("LIQ_SWEEP_HIGH_1H", recent_hi, recent_hi, hi_min, hi_max, "SHORT"))

        # ---- Session VWAP band (acts as zone too) ----
        vmin, vmax, vwap_val = session_vwap_band(df_1h.iloc[:i+1], i, VWAP_BAND_PCT)
        if vmin is not None:
            if detect_reentry(prev_close, prev_high, prev_low, curr_close, vmin, vmax, "LONG"):  entries.append(("VWAP_SESSION_1H", vwap_val, vwap_val, vmin, vmax, "LONG"))
            if detect_reentry(prev_close, prev_high, prev_low, curr_close, vmin, vmax, "SHORT"): entries.append(("VWAP_SESSION_1H", vwap_val, vwap_val, vmin, vmax, "SHORT"))

        if not entries:
            continue

        # Priority: prefer higher-TF first, then sweeps, then VWAP/EMAs/FIB short/tiny
        priority = {
            "EMA200_1D":11,"EMA100_1D":10,"EMA20_1D":9,"EMA50_1D":8,
            "FIB_LONG":7,"FIB_MEDIUM":6,
            "LIQ_SWEEP_HIGH_1H":5,"LIQ_SWEEP_LOW_1H":5,
            "VWAP_SESSION_1H":4,"EMA200_1H":3,"EMA50_1H":2,
            "FIB_SHORT":2,"FIB_TINY":1
        }
        entries.sort(key=lambda x: priority.get(x[0], 1), reverse=True)
        zone_source, swing_low, swing_high, zmin, zmax, side = entries[0]

        # ---- Mean distance filter ----
        mean_name, mean_price = choose_mean_target(side, curr_close, vwap_val, ema50_val, ema200_val, ema20d, ema50d, ema100d, ema200d)
        if mean_name is None:
            continue
        dev_pct = abs(mean_price - curr_close) / curr_close
        if dev_pct < MIN_DEV_TO_MEAN_PCT:
            continue

        # ---- Entry, SL and TP (single target at chosen mean) ----
        entry = curr_close
        if side == "LONG":
            sl = zmin * (1 - SL_BUFFER)
            tp = mean_price
            risk = entry - sl
            be_trigger = entry + MOVE_BE_AT_FRAC * (tp - entry)
        else:
            sl = zmax * (1 + SL_BUFFER)
            tp = mean_price
            risk = sl - entry
            be_trigger = entry - MOVE_BE_AT_FRAC * (entry - tp)

        if risk <= 0:
            continue

        # Manage forward: move to BE once price reaches be_trigger; exit at TP or SL or timeout
        highest_before_exit = entry
        lowest_before_exit  = entry
        sl_work = sl
        exit_price = entry
        outcome = "Timeout"
        hold_hours = 0

        for j in range(i+1, len(df_1h)):
            bar = df_1h.iloc[j]
            bh, bl, bc = float(bar["high"]), float(bar["low"]), float(bar["close"])

            if bh > highest_before_exit: highest_before_exit = bh
            if bl < lowest_before_exit:  lowest_before_exit  = bl

            # Move to BE after halfway to target
            if side == "LONG" and bc >= be_trigger and sl_work < entry:
                sl_work = entry
            if side == "SHORT" and bc <= be_trigger and sl_work > entry:
                sl_work = entry

            # Check exits
            if side == "LONG" and bh >= tp:
                exit_price = tp; outcome = "TP"; hold_hours = j - i; break
            if side == "SHORT" and bl <= tp:
                exit_price = tp; outcome = "TP"; hold_hours = j - i; break

            if side == "LONG" and bl <= sl_work:
                exit_price = sl_work; outcome = "BE" if sl_work == entry else "SL"; hold_hours = j - i; break
            if side == "SHORT" and bh >= sl_work:
                exit_price = sl_work; outcome = "BE" if sl_work == entry else "SL"; hold_hours = j - i; break

            if (j - i) >= MAX_HOLD_HOURS:
                exit_price = bc; outcome = "Timeout"; hold_hours = j - i; break

        netR = (exit_price - entry) / risk if side == "LONG" else (entry - exit_price) / risk
        netR -= FEES_PCT

        trades.append({
            "coin": coin,
            "entry_time_utc": df_1h.iloc[i]["time"].strftime("%Y-%m-%d %H:%M"),
            "side": side,
            "zone_source": zone_source,
            "entry": entry, "sl": sl, "tp": tp,
            "outcome": outcome, "R_multiple": round(netR,4), "hold_hours": hold_hours,
            "highest_before_exit": highest_before_exit, "lowest_before_exit": lowest_before_exit,
            "zmin": zmin, "zmax": zmax, "mean_name": mean_name, "mean_price": mean_price,
            "dev_to_mean_pct": round(dev_pct,5)
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

    overall = pd.DataFrame([{
        "trades": len(df),
        "win_rate": win_rate_rpos(df["R_multiple"]),
        "avg_R": df["R_multiple"].mean(),
        "median_R": df["R_multiple"].median(),
        "avg_hold_hours": df["hold_hours"].mean(),
        "tp_rate": (df["outcome"] == "TP").mean(),
        "be_rate": (df["outcome"] == "BE").mean(),
        "sl_rate": (df["outcome"] == "SL").mean(),
        "timeout_rate": (df["outcome"] == "Timeout").mean(),
        "section": "overall"
    }])

    by_zone = df.groupby("zone_source").agg(
        trades=("zone_source","count"),
        win_rate=("R_multiple", win_rate_rpos),
        tp_rate=("outcome", lambda s: (s == "TP").mean()),
        avg_R=("R_multiple","mean"),
        median_R=("R_multiple","median"),
        avg_hold_hours=("hold_hours","mean")
    ).reset_index()
    by_zone["section"] = "by_zone"

    by_mean = df.groupby("mean_name").agg(
        trades=("mean_name","count"),
        win_rate=("R_multiple", win_rate_rpos),
        tp_rate=("outcome", lambda s: (s == "TP").mean()),
        avg_R=("R_multiple","mean"),
        avg_dev_to_mean=("dev_to_mean_pct","mean")
    ).reset_index()
    by_mean["section"] = "by_mean_target"

    summary = pd.concat([overall, by_zone, by_mean], ignore_index=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"Saved {len(df)} trades to {TRADES_CSV}")
    print(f"Saved summary to {SUMMARY_CSV}")
    print("\n=== Overall ===");          print(overall.to_string(index=False))
    print("\n=== By zone ===");         print(by_zone.to_string(index=False))
    print("\n=== By mean target ===");  print(by_mean.to_string(index=False))

# ---------- Main ----------
def main():
    all_trades = []
    for coin in COINS:
        try:
            print(f"Backtesting {coin} ...")
            all_trades.extend(run_backtest_for_coin(coin))
        except Exception as e:
            print(f"[ERR] {coin}: {e}")
            continue
    summarize_and_save(all_trades)

if __name__ == "__main__":
    import pandas as pd
    main()
