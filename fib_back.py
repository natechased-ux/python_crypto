#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Timeframe Fib + EMA Reversal Backtest (Stoch-only gate, Multi-TP + BE)

Zones:
  FIB_TINY   = last 2d on 1H (golden pocket)
  FIB_SHORT  = last 7d on 1H
  FIB_MEDIUM = last 14d on 1D
  FIB_LONG   = last 30d on 1D
  EMA50_1H   = 50 EMA on 1H with ±band      (NEW)
  EMA200_1H  = 200 EMA on 1H with ±band
  EMA20_1D   = 20 EMA on 1D with ±band      (NEW)
  EMA50_1D   = 50 EMA on 1D with ±band
  EMA100_1D  = 100 EMA on 1D with ±band
  EMA200_1D  = 200 EMA on 1D with ±band

Entry (reversal):
  - Ascend into zone => SHORT
  - Descend into zone => LONG
  - REQUIRED filter: Stoch RSI strict (40/60)

Risk:
  - TP1 = 1.0R (50%); move SL to BE after TP1
  - TP2 = 1.5R (50%)
"""

import csv
import requests
import numpy as np
import pandas as pd
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
START_DATE = "2025-01-01"
END_DATE   = "2025-08-25"

# Timeframe Fib windows
TINY_LOOKBACK_HOURS  = 2 * 24   # 2d on 1H
SHORT_LOOKBACK_HOURS = 7 * 24   # 7d on 1H
MEDIUM_LOOKBACK_DAYS = 14       # 14d on 1D
LONG_LOOKBACK_DAYS   = 30       # 30d on 1D

# EMA band width (±%)
EMA_BAND_PCT = 0.004  # 0.5%

# Stoch RSI strict thresholds (use 50/50 for midline if you want)
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

# Outputs
TRADES_CSV  = "reversal_timeframe_plus_ema_trades.csv"
SUMMARY_CSV = "reversal_timeframe_plus_ema_summary.csv"

CB_BASE = "https://api.exchange.coinbase.com"
# ============ END CONFIG ============


# ---------- Helpers ----------
def fetch_candles(product_id: str, granularity: int) -> pd.DataFrame:
    r = requests.get(f"{CB_BASE}/products/{product_id}/candles",
                     params={"granularity": granularity}, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not data:
        raise ValueError(f"No candle data for {product_id}@{granularity}")
    df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.sort_values("time", inplace=True)
    s, e = pd.Timestamp(START_DATE, tz="UTC"), pd.Timestamp(END_DATE, tz="UTC")
    return df[(df["time"] >= s) & (df["time"] < e)].reset_index(drop=True)

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


# ---------- Core backtest ----------
def run_backtest_for_coin(coin: str):
    df_1h = fetch_candles(coin, 3600)
    df_1d = fetch_candles(coin, 86400)
    if len(df_1h) < SHORT_LOOKBACK_HOURS + 50:
        return []

    # Stoch RSI on 1H
    K, D = stoch_rsi(df_1h["close"], STOCH_RSI_PERIOD, STOCH_SMOOTH_K, STOCH_SMOOTH_D)

    trades = []
    last_trade_day = None
    warmup = max(SHORT_LOOKBACK_HOURS, 50)

    for i in range(warmup, len(df_1h) - 1):
        t = df_1h.iloc[i]["time"]
        if ONE_TRADE_PER_DAY and last_trade_day == t.date():
            continue

        curr = float(df_1h["close"].iloc[i])
        prev = float(df_1h["close"].iloc[i-1])

        # ---- Fib timeframe zones ----
        # FIB_TINY (2d/1H)
        w_t = slice_last(df_1h.iloc[:i+1], TINY_LOOKBACK_HOURS)
        sl_t, sh_t = float(w_t["low"].min()), float(w_t["high"].max())
        zmin_t, zmax_t = fib_golden_zone(sl_t, sh_t)
        in_fib_tiny  = in_zone(curr, zmin_t, zmax_t)
        side_fib_tiny = detect_reversal_side(prev, curr, zmin_t, zmax_t)

        # FIB_SHORT (7d/1H)
        w_s = slice_last(df_1h.iloc[:i+1], SHORT_LOOKBACK_HOURS)
        sl_s, sh_s = float(w_s["low"].min()), float(w_s["high"].max())
        zmin_s, zmax_s = fib_golden_zone(sl_s, sh_s)
        in_fib_short = in_zone(curr, zmin_s, zmax_s)
        side_fib_short = detect_reversal_side(prev, curr, zmin_s, zmax_s)

        # FIB_MEDIUM (14d/1D)
        df_1d_hist = df_1d[df_1d["time"] <= t]
        in_fib_medium = False; side_fib_medium = None; zmin_m = zmax_m = sl_m = sh_m = None
        if len(df_1d_hist) >= MEDIUM_LOOKBACK_DAYS:
            w_m = slice_last(df_1d_hist, MEDIUM_LOOKBACK_DAYS)
            sl_m, sh_m = float(w_m["low"].min()), float(w_m["high"].max())
            zmin_m, zmax_m = fib_golden_zone(sl_m, sh_m)
            in_fib_medium  = in_zone(curr, zmin_m, zmax_m)
            side_fib_medium = detect_reversal_side(prev, curr, zmin_m, zmax_m)

        # FIB_LONG (30d/1D)
        in_fib_long = False; side_fib_long = None; zmin_l = zmax_l = sl_l = sh_l = None
        if len(df_1d_hist) >= LONG_LOOKBACK_DAYS:
            w_l = slice_last(df_1d_hist, LONG_LOOKBACK_DAYS)
            sl_l, sh_l = float(w_l["low"].min()), float(w_l["high"].max())
            zmin_l, zmax_l = fib_golden_zone(sl_l, sh_l)
            in_fib_long  = in_zone(curr, zmin_l, zmax_l)
            side_fib_long = detect_reversal_side(prev, curr, zmin_l, zmax_l)

        # ---- EMA zones ----
        # 1H EMAs
        ema50_1h_val  = float(ema(df_1h["close"].iloc[:i+1], 50).iloc[-1])  if i+1 >= 50 else None
        ema200_1h_val = float(ema(df_1h["close"].iloc[:i+1], 200).iloc[-1]) if i+1 >= 200 else None

        def ema_zone_from_val(val):
            if val is None or val <= 0: return (None, None, False, None)
            zmin = val * (1 - EMA_BAND_PCT)
            zmax = val * (1 + EMA_BAND_PCT)
            return zmin, zmax, in_zone(curr, zmin, zmax), detect_reversal_side(prev, curr, zmin, zmax)

        zmin_e50h,  zmax_e50h,  in_ema50_1h,  side_ema50_1h  = ema_zone_from_val(ema50_1h_val)
        zmin_e200h, zmax_e200h, in_ema200_1h, side_ema200_1h = ema_zone_from_val(ema200_1h_val)

        # 1D EMAs
        def last_ema_1d(period):
            if len(df_1d_hist) >= max(period, 5):
                return float(ema(df_1d_hist["close"], period).iloc[-1])
            return None

        ema20_1d_val  = last_ema_1d(20)
        ema50_1d_val  = last_ema_1d(50)
        ema100_1d_val = last_ema_1d(100)
        ema200_1d_val = last_ema_1d(200)

        zmin_e20d,  zmax_e20d,  in_ema20_1d,  side_ema20_1d  = ema_zone_from_val(ema20_1d_val)
        zmin_e50d,  zmax_e50d,  in_ema50_1d,  side_ema50_1d  = ema_zone_from_val(ema50_1d_val)
        zmin_e100d, zmax_e100d, in_ema100_1d, side_ema100_1d = ema_zone_from_val(ema100_1d_val)
        zmin_e200d, zmax_e200d, in_ema200_1d, side_ema200_1d = ema_zone_from_val(ema200_1d_val)

        # ---- Candidate entries (priority compares TF strength first) ----
        entries = []
        # Highest TF EMAs first
        if side_ema200_1d: entries.append(("EMA200_1D", ema200_1d_val, ema200_1d_val, zmin_e200d, zmax_e200d, side_ema200_1d))
        if side_ema100_1d: entries.append(("EMA100_1D", ema100_1d_val, ema100_1d_val, zmin_e100d, zmax_e100d, side_ema100_1d))
        if side_ema50_1d:  entries.append(("EMA50_1D",  ema50_1d_val,  ema50_1d_val,  zmin_e50d,  zmax_e50d,  side_ema50_1d))
        if side_ema20_1d:  entries.append(("EMA20_1D",  ema20_1d_val,  ema20_1d_val,  zmin_e20d,  zmax_e20d,  side_ema20_1d))  # NEW
        # High TF fibs
        if side_fib_long:  entries.append(("FIB_LONG",   sl_l, sh_l, zmin_l, zmax_l, side_fib_long))
        if side_fib_medium:entries.append(("FIB_MEDIUM", sl_m, sh_m, zmin_m, zmax_m, side_fib_medium))
        # 1H EMA then short fibs
        if side_ema200_1h: entries.append(("EMA200_1H",  ema200_1h_val, ema200_1h_val, zmin_e200h, zmax_e200h, side_ema200_1h))
        if side_ema50_1h:  entries.append(("EMA50_1H",   ema50_1h_val,  ema50_1h_val,  zmin_e50h,  zmax_e50h,  side_ema50_1h))   # NEW
        if side_fib_short: entries.append(("FIB_SHORT",  sl_s, sh_s, zmin_s, zmax_s, side_fib_short))
        if side_fib_tiny:  entries.append(("FIB_TINY",   sl_t, sh_t, zmin_t, zmax_t, side_fib_tiny))

        if not entries:
            continue

        # Priority table
        priority = {
            "EMA200_1D":10,"EMA100_1D":9,"EMA50_1D":8,"EMA20_1D":7,
            "FIB_LONG":6,"FIB_MEDIUM":5,
            "EMA200_1H":4,"EMA50_1H":3,
            "FIB_SHORT":2,"FIB_TINY":1
        }
        entries.sort(key=lambda x: priority[x[0]], reverse=True)
        zone_source, swing_low, swing_high, zmin, zmax, side = entries[0]

        # ---- Stoch RSI REQUIRED ----
        kv = float(K.iloc[i]) if not pd.isna(K.iloc[i]) else None
        dv = float(D.iloc[i]) if not pd.isna(D.iloc[i]) else None
        if kv is None or dv is None:
            continue
        stoch_ok = (kv > dv and kv < LONG_K_MAX) if side == "LONG" else (kv < dv and kv > SHORT_K_MIN)
        if not stoch_ok:
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

        # Manage forward (BE after TP1); track excursions
        highest_before_exit = entry
        lowest_before_exit  = entry
        sl_work = sl
        tp1_hit = False
        tp2_hit = False
        exit_price = entry
        outcome = "Timeout"
        hold_hours = 0

        for j in range(i+1, len(df_1h)):
            bh, bl = float(df_1h["high"].iloc[j]), float(df_1h["low"].iloc[j])

            if bh > highest_before_exit: highest_before_exit = bh
            if bl < lowest_before_exit:  lowest_before_exit  = bl

            if not tp1_hit:
                if side == "LONG" and bh >= tp1:
                    tp1_hit = True
                    sl_work = max(sl_work, entry)
                elif side == "SHORT" and bl <= tp1:
                    tp1_hit = True
                    sl_work = min(sl_work, entry)

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

        # R-multiple
        netR = (exit_price - entry) / risk if side == "LONG" else (entry - exit_price) / risk
        netR -= FEES_PCT

        trades.append({
            "coin": coin,
            "entry_time_utc": df_1h.iloc[i]["time"].strftime("%Y-%m-%d %H:%M"),
            "side": side,
            "zone_source": zone_source,  # FIB_* or EMA*_*D/H
            "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
            "outcome": outcome, "R_multiple": round(netR,4), "hold_hours": hold_hours,
            "tp1_hit": bool(tp1_hit), "tp2_hit": bool(tp2_hit),
            "highest_before_exit": highest_before_exit, "lowest_before_exit": lowest_before_exit,
            "zmin": zmin, "zmax": zmax, "swing_low": swing_low, "swing_high": swing_high,
            "stoch_k": kv, "stoch_d": dv,
            # presence snapshot (optional)
            "in_fib_tiny": bool(in_fib_tiny), "in_fib_short": bool(in_fib_short),
            "in_fib_medium": bool(in_fib_medium), "in_fib_long": bool(in_fib_long),
            "in_ema50_1h":  bool(in_ema50_1h),  "in_ema200_1h": bool(in_ema200_1h),
            "in_ema20_1d":  bool(in_ema20_1d),  "in_ema50_1d":  bool(in_ema50_1d),
            "in_ema100_1d": bool(in_ema100_1d), "in_ema200_1d": bool(in_ema200_1d),
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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in trades:
            w.writerow(row)

    df = pd.DataFrame(trades)

    def win_rate_rpos(s: pd.Series) -> float:
        return (s > 0).mean() if len(s) else 0.0

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

    summary = pd.concat([overall, by_zone], ignore_index=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"Saved {len(df)} trades to {TRADES_CSV}")
    print(f"Saved summary to {SUMMARY_CSV}")
    print("\n=== Overall ===");               print(overall.to_string(index=False))
    print("\n=== By zone ===");              print(by_zone.to_string(index=False))


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
    main()
