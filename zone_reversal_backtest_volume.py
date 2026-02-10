
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zone Reversal Backtester (Coinbase) — EMA + Fib Zone + 15m Volume Spike + 15m Stoch RSI (intrabar)
==================================================================================================

Changes in this version
-----------------------
- **15m Stoch RSI confirmation** (instead of 1H): confirm in the last N completed 15m bars before each 1H entry.
- **Pre-fetched 15m data** reused for BOTH volume spike and Stoch RSI checks (no repeated HTTP calls).
- Tunable 15m confirmation window (e.g., last 1–2 bars).

Usage example
-------------
py zone_reversal_backtest_volume.py --symbols XRP-USD --lookback_days 90 --swing_days 7 \
  --use_ema200 --vol_spike_mult 1.8 --stoch_confirm --stoch_15m_window 2 \
  --csv trades.csv --summary summary.csv
"""

import argparse
import datetime as dt
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from dateutil import parser as dateparser

# ------------------------------------------------------------------------------------
# Coinbase data helpers (chunked)
# ------------------------------------------------------------------------------------

GRAN_MAP = {'1m':60,'5m':300,'15m':900,'1h':3600,'6h':21600,'1d':86400}
MAX_PER_CALL = 300  # Coinbase per-request cap

def _cb_call(product_id: str, granularity: int, start=None, end=None) -> pd.DataFrame:
    base = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {"granularity": granularity}
    if start:
        params["start"] = dateparser.parse(str(start)).isoformat()
    if end:
        params["end"] = dateparser.parse(str(end)).isoformat()
    r = requests.get(base, params=params, timeout=20, headers={"User-Agent":"zone-reversal-bt"})
    r.raise_for_status()
    data = r.json()
    rows = []
    for arr in data:
        # [time, low, high, open, close, volume], newest-first
        rows.append({
            "time": int(arr[0]) * 1000,
            "open": float(arr[3]),
            "high": float(arr[2]),
            "low": float(arr[1]),
            "close": float(arr[4]),
            "volume": float(arr[5]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df

def fetch_chunked(product_id: str, timeframe: str, start=None, end=None, lookback_days: int = None, max_candles: int = 20000) -> pd.DataFrame:
    if timeframe not in GRAN_MAP:
        raise ValueError("Unsupported timeframe")
    gran = GRAN_MAP[timeframe]
    if end is None:
        end_dt = dt.datetime.utcnow()
    else:
        end_dt = dateparser.parse(str(end)).replace(tzinfo=None)
    if start is None and lookback_days is not None:
        start_dt = end_dt - dt.timedelta(days=int(lookback_days))
    elif start is not None:
        start_dt = dateparser.parse(str(start)).replace(tzinfo=None)
    else:
        start_dt = end_dt - dt.timedelta(days=120)

    out = []
    curr_end = end_dt
    candles = 0
    while candles < max_candles:
        curr_start = max(start_dt, curr_end - dt.timedelta(seconds=gran*MAX_PER_CALL))
        df = _cb_call(product_id, gran, start=curr_start, end=curr_end)
        if df.empty:
            break
        out.append(df)
        candles += len(df)
        earliest_ms = int(df["time"].iloc[0])
        curr_end = dt.datetime.utcfromtimestamp((earliest_ms // 1000) - gran)
        if cur_end := curr_end <= start_dt:
            break
    if not out:
        return pd.DataFrame(columns=["time","open","high","low","close","volume","date"])
    full = pd.concat(out, ignore_index=True).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    mask = (full["date"] >= pd.Timestamp(start_dt, tz="UTC")) & (full["date"] <= pd.Timestamp(end_dt, tz="UTC"))
    full = full.loc[mask].reset_index(drop=True)
    if len(full) > max_candles:
        full = full.iloc[-max_candles:].reset_index(drop=True)
    return full

# ------------------------------------------------------------------------------------
# Indicators
# ------------------------------------------------------------------------------------

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def stoch_rsi(close: pd.Series, rsi_len=14, stoch_len=14, k_len=3, d_len=3):
    r = rsi(close, rsi_len)
    ll = r.rolling(stoch_len, min_periods=stoch_len).min()
    hh = r.rolling(stoch_len, min_periods=stoch_len).max()
    raw = (r - ll) / (hh - ll).replace(0, np.nan) * 100.0
    k = raw.rolling(k_len, min_periods=k_len).mean()
    d = k.rolling(d_len, min_periods=d_len).mean()
    return k, d

# ------------------------------------------------------------------------------------
# Fib zone + Trend filter
# ------------------------------------------------------------------------------------

def compute_fib_zone_from_window(df_1h: pd.DataFrame, window_days: int) -> Tuple[float, float]:
    if len(df_1h) == 0:
        return np.nan, np.nan
    cutoff = df_1h["date"].iloc[-1] - pd.Timedelta(days=window_days)
    win = df_1h[df_1h["date"] >= cutoff]
    if len(win) < 10:
        win = df_1h
    swing_low = float(win["low"].min())
    swing_high = float(win["high"].max())
    if swing_high <= swing_low:
        return np.nan, np.nan
    fib_618 = swing_low + 0.618 * (swing_high - swing_low)
    fib_660 = swing_low + 0.660 * (swing_high - swing_low)
    return min(fib_618, fib_660), max(fib_618, fib_660)

def in_band(price: float, lo: float, hi: float, tol_pct: float = 0.0025) -> bool:
    pad = (hi - lo) * tol_pct if hi > lo else 0.0
    return (price >= lo - pad) and (price <= hi + pad)

# ------------------------------------------------------------------------------------
# 15m data prep (volume + Stoch RSI)
# ------------------------------------------------------------------------------------

def prepare_15m_data(symbol: str, lookback_days: int, stoch_params: Tuple[int,int,int,int], vol_window: int):
    df_15 = fetch_chunked(symbol, "15m", lookback_days=lookback_days, max_candles=50000)
    if df_15.empty or len(df_15) < 200:
        return df_15
    # Rolling avg volume (exclude current by shifting later in check)
    df_15["vol_roll"] = df_15["volume"].rolling(vol_window, min_periods=max(5, vol_window//2)).mean()
    # Stoch RSI on 15m
    k_len, d_len, rsi_len, stoch_len = stoch_params
    k15, d15 = stoch_rsi(df_15["close"], rsi_len=rsi_len, stoch_len=stoch_len, k_len=k_len, d_len=d_len)
    df_15["k15"], df_15["d15"] = k15, d15
    return df_15.dropna().reset_index(drop=True)

def check_15m_volume_spike(df_15: pd.DataFrame, ts_1h: pd.Timestamp, mult: float) -> Tuple[bool,float,float]:
    # Find the last completed 15m bar <= ts_1h
    sel = df_15[df_15["date"] <= ts_1h]
    if sel.empty:
        return False, np.nan, np.nan
    last = sel.iloc[-1]
    last_vol = float(last["volume"])
    # Use rolling mean that excludes current -> compute mean on prior window via shift
    prior_window_mean = float(sel["volume"].iloc[-min(len(sel)-1, max(1, int(sel["vol_roll"].notna().sum()))): -1].mean()) if len(sel)>1 else np.nan
    # If the above is messy due to window alignment, fall back to precomputed vol_roll at previous row
    if not np.isfinite(prior_window_mean):
        prior_window_mean = float(sel["vol_roll"].iloc[-2]) if len(sel) > 1 else np.nan
    if not np.isfinite(prior_window_mean) or prior_window_mean <= 0:
        return False, last_vol, prior_window_mean
    return (last_vol > mult * prior_window_mean), last_vol, prior_window_mean

def check_15m_stoch_confirm(df_15: pd.DataFrame, ts_1h: pd.Timestamp, window_bars: int) -> Tuple[bool,str,float,float]:
    sel = df_15[df_15["date"] <= ts_1h]
    if len(sel) < max(window_bars, 1):
        return False, "", np.nan, np.nan
    last_window = sel.iloc[-window_bars:]
    # Confirmation rule: any of the last N 15m bars shows K>D & K<40 (long) OR K<D & K>60 (short)
    long_ok = ((last_window["k15"] > last_window["d15"]) & (last_window["k15"] < 40)).any()
    short_ok = ((last_window["k15"] < last_window["d15"]) & (last_window["k15"] > 60)).any()
    if long_ok and not short_ok:
        # return last values just for logging
        return True, "LONG", float(last_window["k15"].iloc[-1]), float(last_window["d15"].iloc[-1])
    if short_ok and not long_ok:
        return True, "SHORT", float(last_window["k15"].iloc[-1]), float(last_window["d15"].iloc[-1])
    # Ambiguous (both seen): accept as True but neutral side; caller will decide via EMA/mid-zone
    if long_ok and short_ok:
        return True, "BOTH", float(last_window["k15"].iloc[-1]), float(last_window["d15"].iloc[-1])
    return False, "", float(last_window["k15"].iloc[-1]), float(last_window["d15"].iloc[-1])

# ------------------------------------------------------------------------------------
# Backtest core
# ------------------------------------------------------------------------------------

def run_backtest_for_symbol(symbol: str,
                            lookback_days: int,
                            swing_days: int,
                            use_ema200: bool,
                            stoch_confirm: bool,
                            stoch_kd: Tuple[int,int,int,int],
                            stoch_15m_window: int,
                            vol_spike_mult: float,
                            vol_spike_window: int,
                            tol_pct: float,
                            tp_atr_mult: float,
                            sl_atr_mult: float,
                            fixed_tp_pct: Optional[float],
                            fixed_sl_pct: Optional[float]) -> Tuple[pd.DataFrame, dict]:
    # 1H data for entries
    df_1h = fetch_chunked(symbol, "1h", lookback_days=lookback_days)
    if len(df_1h) < 200:
        return pd.DataFrame(), {"symbol": symbol, "trades": 0}
    df_1h["atr"] = atr(df_1h, length=14)

    # Daily EMA 200
    if use_ema200:
        df_1d = fetch_chunked(symbol, "1d", lookback_days=max(lookback_days, 240))
        if df_1d.empty:
            use_ema200 = False
            df_1h["ema200"] = np.nan
        else:
            df_1d["ema200"] = ema(df_1d["close"], 200)
            daily = df_1d[["date","ema200"]].copy()
            df_1h = pd.merge_asof(df_1h.sort_values("date"), daily.sort_values("date"),
                                  on="date", direction="backward")
    else:
        df_1h["ema200"] = np.nan

    # Pre-fetch 15m data for volume + Stoch
    df_15 = prepare_15m_data(symbol, lookback_days=lookback_days, stoch_params=stoch_kd, vol_window=vol_spike_window)

    # Fib zone from swing window
    fib_lo, fib_hi = compute_fib_zone_from_window(df_1h, swing_days)

    trades = []
    for i in range(200, len(df_1h)):
        ts = df_1h["date"].iloc[i]
        price = float(df_1h["close"].iloc[i])
        ema200 = float(df_1h["ema200"].iloc[i]) if use_ema200 else np.nan
        if not np.isfinite(price) or (use_ema200 and not np.isfinite(ema200)):
            continue

        # Trend regime
        long_ok = (not use_ema200) or (price >= ema200)
        short_ok = (not use_ema200) or (price <= ema200)

        # Must be inside Fib zone
        if not in_band(price, fib_lo, fib_hi, tol_pct=tol_pct):
            continue

        # 15m volume spike
        spike_ok, last_vol, avg_vol = (False, np.nan, np.nan)
        if df_15 is not None and not df_15.empty:
            spike_ok, last_vol, avg_vol = check_15m_volume_spike(df_15, ts, vol_spike_mult)
        if not spike_ok:
            continue

        # 15m Stoch RSI confirmation (if enabled)
        stoch_pass = True
        stoch_side = ""
        k_log = d_log = np.nan
        if stoch_confirm and df_15 is not None and not df_15.empty:
            stoch_pass, stoch_side, k_log, d_log = check_15m_stoch_confirm(df_15, ts, window_bars=stoch_15m_window)
            if not stoch_pass:
                continue

        # Decide side
        side = None
        if long_ok and not short_ok:
            side = "LONG"
        elif short_ok and not long_ok:
            side = "SHORT"
        else:
            if stoch_side == "LONG":
                side = "LONG"
            elif stoch_side == "SHORT":
                side = "SHORT"
            else:
                mid = 0.5*(fib_lo+fib_hi)
                side = "LONG" if price <= mid else "SHORT"

        entry = price
        atr_val = float(df_1h["atr"].iloc[i])
        # TP/SL
        if fixed_tp_pct is not None and fixed_sl_pct is not None:
            tp = entry * (1 + fixed_tp_pct/100.0) if side=="LONG" else entry * (1 - fixed_tp_pct/100.0)
            sl = entry * (1 - fixed_sl_pct/100.0) if side=="LONG" else entry * (1 + fixed_sl_pct/100.0)
        else:
            tp = entry + tp_atr_mult*atr_val if side=="LONG" else entry - tp_atr_mult*atr_val
            sl = entry - sl_atr_mult*atr_val if side=="LONG" else entry + sl_atr_mult*atr_val

        # Forward walk
        outcome = None
        exit_price = None
        exit_time = None
        for j in range(i+1, min(i+200, len(df_1h))):
            hi = float(df_1h["high"].iloc[j])
            lo = float(df_1h["low"].iloc[j])
            t2 = df_1h["date"].iloc[j]
            if side=="LONG":
                if lo <= sl:
                    outcome, exit_price, exit_time = "SL", sl, t2; break
                if hi >= tp:
                    outcome, exit_price, exit_time = "TP", tp, t2; break
            else:
                if hi >= sl:
                    outcome, exit_price, exit_time = "SL", sl, t2; break
                if lo <= tp:
                    outcome, exit_price, exit_time = "TP", tp, t2; break

        if outcome is None:
            exit_idx = min(i+199, len(df_1h)-1)
            exit_price = float(df_1h["close"].iloc[exit_idx])
            exit_time = df_1h["date"].iloc[exit_idx]
            outcome = "EXP"

        ret_pct = (exit_price - entry)/entry*100.0 if side=="LONG" else (entry - exit_price)/entry*100.0

        trades.append({
            "symbol": symbol,
            "time": ts,
            "side": side,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "outcome": outcome,
            "ret_pct": ret_pct,
            "ema200": ema200 if use_ema200 else np.nan,
            "fib_lo": fib_lo, "fib_hi": fib_hi,
            "in_zone": True,
            "stoch_used": stoch_confirm,
            "stoch15m_last_k": k_log, "stoch15m_last_d": d_log,
            "stoch15m_window": stoch_15m_window,
            "vol_spike_ok": bool(spike_ok),
            "vol_last_15m": last_vol, "vol_avg_15m": avg_vol,
            "atr_1h": atr_val
        })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        summary = {"symbol": symbol, "trades": 0, "win_rate": np.nan, "avg_ret_pct": np.nan, "median_ret_pct": np.nan}
    else:
        wins = (trades_df["outcome"] == "TP").sum()
        summary = {
            "symbol": symbol,
            "trades": len(trades_df),
            "win_rate": wins / len(trades_df) * 100.0,
            "avg_ret_pct": float(trades_df["ret_pct"].mean()),
            "median_ret_pct": float(trades_df["ret_pct"].median()),
        }
    return trades_df, summary

# ------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Zone reversal backtest with EMA + Fib zone + 15m volume spike + 15m Stoch RSI confirmation.")
    ap.add_argument("--symbols", type=str, required=True, help="Comma-separated Coinbase product IDs (e.g., BTC-USD,ETH-USD,XRP-USD)")
    ap.add_argument("--lookback_days", type=int, default=120, help="History window")
    ap.add_argument("--swing_days", type=int, default=7, help="Window for Fib swing calculation on 1H")
    ap.add_argument("--use_ema200", action="store_true", help="Apply daily 200 EMA trend filter")
    ap.add_argument("--stoch_confirm", action="store_true", help="Require 15m Stoch RSI confirmation")
    ap.add_argument("--stoch_k", type=int, default=3)
    ap.add_argument("--stoch_d", type=int, default=3)
    ap.add_argument("--stoch_rsi_len", type=int, default=14)
    ap.add_argument("--stoch_stoch_len", type=int, default=14)
    ap.add_argument("--stoch_15m_window", type=int, default=2, help="Number of last 15m bars to scan for confirmation")
    ap.add_argument("--vol_spike_mult", type=float, default=2.0, help="15m spike multiple vs rolling average")
    ap.add_argument("--vol_spike_window", type=int, default=10, help="15m rolling average window (bars)")
    ap.add_argument("--tol_pct", type=float, default=0.0025, help="Fib zone tolerance as a fraction of zone width")
    ap.add_argument("--tp_atr_mult", type=float, default=2.0, help="TP in ATR(14) multiples")
    ap.add_argument("--sl_atr_mult", type=float, default=1.5, help="SL in ATR(14) multiples")
    ap.add_argument("--fixed_tp_pct", type=float, default=None, help="Fixed TP percent (e.g., 1.5 for +1.5%%)")
    ap.add_argument("--fixed_sl_pct", type=float, default=None, help="Fixed SL percent (e.g., 1.0 for -1.0%%)")
    ap.add_argument("--csv", type=str, default="trades.csv", help="Output trades CSV")
    ap.add_argument("--summary", type=str, default="summary.csv", help="Output summary CSV")

    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    stoch_params = (args.stoch_k, args.stoch_d, args.stoch_rsi_len, args.stoch_stoch_len)

    trades_all = []
    summaries = []
    for sym in symbols:
        tdf, summ = run_backtest_for_symbol(
            symbol=sym,
            lookback_days=args.lookback_days,
            swing_days=args.swing_days,
            use_ema200=args.use_ema200,
            stoch_confirm=args.stoch_confirm,
            stoch_kd=stoch_params,
            stoch_15m_window=args.stoch_15m_window,
            vol_spike_mult=args.vol_spike_mult,
            vol_spike_window=args.vol_spike_window,
            tol_pct=args.tol_pct,
            tp_atr_mult=args.tp_atr_mult,
            sl_atr_mult=args.sl_atr_mult,
            fixed_tp_pct=args.fixed_tp_pct,
            fixed_sl_pct=args.fixed_sl_pct
        )
        if not tdf.empty:
            trades_all.append(tdf)
        summaries.append(summ)

    if trades_all:
        out_trades = pd.concat(trades_all, ignore_index=True)
        out_trades.sort_values(["symbol","time"], inplace=True)
        out_trades.to_csv(args.csv, index=False)
        print(f"Saved trades → {args.csv} ({len(out_trades)} rows)")
    else:
        print("No trades generated.")
        out_trades = pd.DataFrame()

    out_summary = pd.DataFrame(summaries)
    out_summary.to_csv(args.summary, index=False)
    print(f"Saved summary → {args.summary}")

if __name__ == "__main__":
    main()
