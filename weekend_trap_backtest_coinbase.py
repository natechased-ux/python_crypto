#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weekend Liquidity Trap Backtest (Coinbase, 6H candles)

Rules (default, all configurable):
- Define Friday range in UTC using Friday 00:00→23:59 (6H candles).
- During the weekend (Sat+Sun UTC), if a 6H candle CLOSES outside Friday high/low,
  and the very next 6H candle CLOSES back inside the Friday range,
  then on the next 6H candle OPEN we take a trade toward Friday MID:
    - Upper false breakout (close > FriHigh then re-entry close < FriHigh): SHORT toward FriMid
    - Lower false breakout (close < FriLow  then re-entry close > FriLow ): LONG  toward FriMid
- Stop Loss distance (from entry): 0.5 × weekend extension beyond Friday boundary
  (extension measured as max deviation between the first outside-close candle and the re-entry candle)
- Take Profit: Friday MID
- Time stop: If neither TP nor SL hit by end of Monday UTC (last Monday 18:00—>24:00 candle close),
  exit at the close of Monday 18:00 candle.
- Entry executes at the OPEN of the candle AFTER the re-entry close to avoid look-ahead.
- Intrabar exit resolution: conservative — if a candle touches both TP and SL,
  assume SL hits first (you can switch to 'tp_first' via CLI arg).

Outputs:
- CSV with per-trade details
- CSV with per-symbol and overall summary
- Console printouts of summary

Usage examples:
  python weekend_trap_backtest_coinbase.py
  python weekend_trap_backtest_coinbase.py --symbols BTC-USD,ETH-USD,XRP-USD --days 180
  python weekend_trap_backtest_coinbase.py --start 2024-01-01 --end 2025-08-25 --tp_first

Requires: pandas, numpy, requests, python-dateutil
"""

import argparse
import time
import math
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import requests
from dateutil import parser as dtparser

CB_BASE = "https://api.exchange.coinbase.com"  # Coinbase Exchange (formerly Pro) public candles

def isoformat(dt: datetime) -> str:
    # Coinbase accepts ISO8601; ensure UTC 'Z' form
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat()

def fetch_candles(product_id: str, start: datetime, end: datetime, granularity: int = 21600, max_per_req: int = 300, pause_sec: float = 0.35, max_retries: int = 5) -> pd.DataFrame:
    """
    Fetch OHLCV candles for [start, end) in UTC, chunked by limit (300 candles/request).
    Coinbase returns arrays: [ time, low, high, open, close, volume ] with time as bucket start (epoch seconds).
    """
    assert granularity in (60, 300, 900, 3600, 21600, 86400), "Granularity must be one of Coinbase supported values"
    rows = []
    chunk_seconds = granularity * max_per_req
    t0 = start.astimezone(timezone.utc)
    t1 = end.astimezone(timezone.utc)
    if t1 <= t0:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])

    while t0 < t1:
        t2 = min(t0 + timedelta(seconds=chunk_seconds), t1)
        params = {
            "granularity": granularity,
            "start": isoformat(t0),
            "end": isoformat(t2)
        }
        url = f"{CB_BASE}/products/{product_id}/candles"
        # retry/backoff
        for attempt in range(max_retries):
            try:
                r = requests.get(url, params=params, headers={"User-Agent":"weekend-trap-backtest/1.0"} , timeout=20)
                if r.status_code == 429:
                    # rate limited; backoff
                    sleep_s = pause_sec * (2 ** attempt)
                    time.sleep(sleep_s)
                    continue
                r.raise_for_status()
                data = r.json()
                # Coinbase returns in reverse chronological order (most recent first)
                for arr in data:
                    # [ time, low, high, open, close, volume ]
                    rows.append(arr)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[{product_id}] ERROR fetching {t0}→{t2}: {e}", file=sys.stderr)
                else:
                    time.sleep(pause_sec * (2 ** attempt))
        time.sleep(pause_sec)
        t0 = t2

    if not rows:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])

    df = pd.DataFrame(rows, columns=["epoch","low","high","open","close","volume"])
    # Reverse to chronological order
    df = df.sort_values("epoch").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df[["time","open","high","low","close","volume"]].astype({
        "open": float, "high": float, "low": float, "close": float, "volume": float
    })
    return df

def build_friday_range(df6h: pd.DataFrame) -> pd.DataFrame:
    """
    Add helper columns and identify Friday ranges per week.
    """
    d = df6h.copy()
    d["dow"] = d["time"].dt.weekday  # Monday=0, Sunday=6
    d["date"] = d["time"].dt.date
    return d

def weekend_trap_trades_from_df(df6h: pd.DataFrame, symbol: str, tp_first: bool = False, reentry_max_candles: int = 1) -> List[Dict]:
    """
    Core signal generation and trade simulation on a single symbol's 6H dataframe.
    Returns list of per-trade dicts.
    """
    d = build_friday_range(df6h)

    trades = []
    # iterate over unique Fridays present
    friday_dates = sorted(set(d.loc[d["dow"] == 4, "date"]))  # UTC Fridays
    for fri_date in friday_dates:
        # Friday 6H candles
        fri_mask = (d["date"] == fri_date)
        fri_df = d.loc[fri_mask]
        if fri_df.empty:
            continue
        # Check that Friday covers full 24h in 6H candles (ideally 4 candles)
        # Some data gaps may exist; still compute range of what's present
        fri_high = fri_df["high"].max()
        fri_low = fri_df["low"].min()
        if not np.isfinite(fri_high) or not np.isfinite(fri_low):
            continue
        fri_mid = (fri_high + fri_low) / 2.0

        # Weekend (Sat and Sun)
        sat_date = fri_date + timedelta(days=1)
        sun_date = fri_date + timedelta(days=2)
        weekend_mask = (d["date"].isin([sat_date, sun_date]))
        wknd_df = d.loc[weekend_mask].reset_index(drop=True)

        if wknd_df.empty:
            continue

        # Scan weekend for a close outside Friday range -> immediate re-entry on next close
        # We'll only take the first valid trap per weekend.
        took_trade = False

        for i in range(len(wknd_df) - 1):  # ensure a "next candle" exists
            row = wknd_df.iloc[i]
            nxt = wknd_df.iloc[i+1]

            # Upper fakeout: close above Friday high
            if row["close"] > fri_high:
                # Next candle must close back inside (below or equal FriHigh)
                if nxt["close"] <= fri_high:
                    # Measure extension between 'row' (outside close) and 'nxt' (re-entry close):
                    # We consider highs in inclusive range [i, i+1]
                    ext_high = max(row["high"], nxt["high"])
                    ext = max(0.0, ext_high - fri_high)
                    if ext <= 0:
                        # no meaningful extension — skip
                        continue

                    # Entry at OPEN of the candle after re-entry
                    # Find the global index for nxt in d to get the next candle
                    nxt_time = nxt["time"]
                    pos = d.index[d["time"] == nxt_time]
                    if len(pos) == 0:
                        continue
                    idx = pos[0]
                    if idx + 1 >= len(d):
                        continue  # no next candle to enter
                    entry_row = d.iloc[idx + 1]
                    entry_time = entry_row["time"]
                    entry = float(entry_row["open"])

                    # Short toward mid; ensure entry > mid to make sense
                    if entry <= fri_mid:
                        # already below/at target; skip
                        continue

                    sl = entry + 0.5 * ext
                    tp = fri_mid
                    sl_dist = max(1e-12, sl - entry)
                    tp_dist = entry - tp

                    # Only evaluate until end of Monday
                    mon_date = fri_date + timedelta(days=3)
                    monday_mask = (d["date"] == mon_date)
                    monday_df = d.loc[(d["time"] >= entry_time) & ((d["date"] == sat_date) | (d["date"] == sun_date) | monday_mask)].reset_index(drop=True)
                    if monday_df.empty:
                        continue

                    outcome, exit_price, exit_time, r_mult, ret_pct = simulate_path(monday_df, entry, sl, tp, side="short", tp_first=tp_first)

                    trades.append({
                        "symbol": symbol,
                        "friday_date": str(fri_date),
                        "side": "SHORT",
                        "entry_time": entry_time.isoformat(),
                        "entry": entry,
                        "tp": tp,
                        "sl": sl,
                        "friday_high": fri_high,
                        "friday_low": fri_low,
                        "friday_mid": fri_mid,
                        "extension": ext,
                        "outcome": outcome,
                        "exit_time": exit_time.isoformat() if exit_time else None,
                        "exit": exit_price,
                        "R": r_mult,
                        "return_pct": ret_pct
                    })
                    took_trade = True
                    break  # one trade per weekend

            # Lower fakeout: close below Friday low
            if row["close"] < fri_low:
                if nxt["close"] >= fri_low:
                    ext_low = min(row["low"], nxt["low"])
                    ext = max(0.0, fri_low - ext_low)
                    if ext <= 0:
                        continue

                    nxt_time = nxt["time"]
                    pos = d.index[d["time"] == nxt_time]
                    if len(pos) == 0:
                        continue
                    idx = pos[0]
                    if idx + 1 >= len(d):
                        continue
                    entry_row = d.iloc[idx + 1]
                    entry_time = entry_row["time"]
                    entry = float(entry_row["open"])

                    # Long toward mid; ensure entry < mid
                    if entry >= fri_mid:
                        continue

                    sl = entry - 0.5 * ext
                    tp = fri_mid
                    sl_dist = max(1e-12, entry - sl)
                    tp_dist = tp - entry

                    mon_date = fri_date + timedelta(days=3)
                    monday_mask = (d["date"] == mon_date)
                    monday_df = d.loc[(d["time"] >= entry_time) & ((d["date"] == sat_date) | (d["date"] == sun_date) | monday_mask)].reset_index(drop=True)
                    if monday_df.empty:
                        continue

                    outcome, exit_price, exit_time, r_mult, ret_pct = simulate_path(monday_df, entry, sl, tp, side="long", tp_first=tp_first)

                    trades.append({
                        "symbol": symbol,
                        "friday_date": str(fri_date),
                        "side": "LONG",
                        "entry_time": entry_time.isoformat(),
                        "entry": entry,
                        "tp": tp,
                        "sl": sl,
                        "friday_high": fri_high,
                        "friday_low": fri_low,
                        "friday_mid": fri_mid,
                        "extension": ext,
                        "outcome": outcome,
                        "exit_time": exit_time.isoformat() if exit_time else None,
                        "exit": exit_price,
                        "R": r_mult,
                        "return_pct": ret_pct
                    })
                    took_trade = True
                    break

        # done weekend scan
    return trades

def simulate_path(path_df: pd.DataFrame, entry: float, sl: float, tp: float, side: str, tp_first: bool = False) -> Tuple[str, float, Optional[pd.Timestamp], float, float]:
    """
    Given a path of 6H candles starting from entry_time onward (inclusive),
    simulate which level is hit first with conservative intrabar assumptions.
    Returns (outcome, exit_price, exit_time, R_multiple, return_pct).
    """
    # Determine order-of-touch priority
    # If tp_first=False, SL is assumed to be hit first if both touched in the same candle (conservative).
    # If tp_first=True, assume TP touched first.
    for i in range(len(path_df)):
        row = path_df.iloc[i]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        t = row["time"]

        if side == "short":
            hit_sl = high >= sl
            hit_tp = low <= tp
            if hit_sl and hit_tp:
                if tp_first:
                    hit_sl = False
                else:
                    hit_tp = False
            if hit_sl:
                r = -1.0
                ret_pct = ( (sl/entry) - 1.0 ) * -100.0  # negative return for short
                return ("SL", sl, t, r, ret_pct)
            if hit_tp:
                sl_dist = sl - entry
                tp_dist = entry - tp
                r = (tp_dist / sl_dist) if sl_dist > 0 else 0.0
                ret_pct = ( (tp/entry) - 1.0 ) * -100.0  # positive return for short
                return ("TP", tp, t, r, ret_pct)

        elif side == "long":
            hit_sl = low <= sl
            hit_tp = high >= tp
            if hit_sl and hit_tp:
                if tp_first:
                    hit_sl = False
                else:
                    hit_tp = False
            if hit_sl:
                r = -1.0
                ret_pct = ( (sl/entry) - 1.0 ) * 100.0  # negative return for long
                return ("SL", sl, t, r, ret_pct)
            if hit_tp:
                sl_dist = entry - sl
                tp_dist = tp - entry
                r = (tp_dist / sl_dist) if sl_dist > 0 else 0.0
                ret_pct = ( (tp/entry) - 1.0 ) * 100.0  # positive return for long
                return ("TP", tp, t, r, ret_pct)

    # If not hit by end of Monday window, exit at last candle close (time stop)
    last = path_df.iloc[-1]
    exit_price = float(last["close"])
    t = last["time"]
    if side == "short":
        sl_dist = sl - entry
        r = ( (entry - exit_price) / sl_dist ) if sl_dist > 0 else 0.0
        ret_pct = ( (exit_price/entry) - 1.0 ) * -100.0
    else:  # long
        sl_dist = entry - sl
        r = ( (exit_price - entry) / sl_dist ) if sl_dist > 0 else 0.0
        ret_pct = ( (exit_price/entry) - 1.0 ) * 100.0
    return ("TimeExit", exit_price, t, r, ret_pct)

def summarize_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["symbol","trades","win_rate","avg_R","pf","avg_return_pct","expectancy_R"])
    grp = trades_df.groupby("symbol")
    rows = []
    for sym, g in grp:
        n = len(g)
        wins = (g["outcome"] == "TP").sum()
        win_rate = wins / n if n else 0.0
        avg_R = g["R"].mean() if n else 0.0
        gross_win = g.loc[g["R"] > 0, "R"].sum()
        gross_loss = -g.loc[g["R"] < 0, "R"].sum()
        pf = (gross_win / gross_loss) if gross_loss > 0 else np.nan
        avg_ret = g["return_pct"].mean() if n else 0.0
        expectancy_R = g["R"].mean()
        rows.append({
            "symbol": sym,
            "trades": int(n),
            "win_rate": round(win_rate, 3),
            "avg_R": round(avg_R, 3),
            "pf": round(pf, 3) if not np.isnan(pf) else None,
            "avg_return_pct": round(avg_ret, 3),
            "expectancy_R": round(expectancy_R, 3)
        })
    # overall
    n = len(trades_df)
    wins = (trades_df["outcome"] == "TP").sum()
    win_rate = wins / n if n else 0.0
    avg_R = trades_df["R"].mean() if n else 0.0
    gross_win = trades_df.loc[trades_df["R"] > 0, "R"].sum()
    gross_loss = -trades_df.loc[trades_df["R"] < 0, "R"].sum()
    pf = (gross_win / gross_loss) if gross_loss > 0 else np.nan
    avg_ret = trades_df["return_pct"].mean() if n else 0.0
    expectancy_R = trades_df["R"].mean()
    rows.append({
        "symbol": "ALL",
        "trades": int(n),
        "win_rate": round(win_rate, 3),
        "avg_R": round(avg_R, 3),
        "pf": round(pf, 3) if not np.isnan(pf) else None,
        "avg_return_pct": round(avg_ret, 3),
        "expectancy_R": round(expectancy_R, 3)
    })
    return pd.DataFrame(rows)

def ensure_6h_alignment(df: pd.DataFrame, granularity: int = 21600) -> pd.DataFrame:
    """
    Coinbase 6H buckets are aligned at 00:00, 06:00, 12:00, 18:00 UTC.
    This ensures index uniqueness and sorts properly.
    """
    d = df.copy()
    d = d.sort_values("time").reset_index(drop=True)
    # Drop duplicates by time (keep first)
    d = d[~d["time"].duplicated()].reset_index(drop=True)
    return d

def backtest_symbols(symbols: List[str], start: Optional[str], end: Optional[str], days: Optional[int], tp_first: bool, granularity: int = 21600) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if days is not None and (start or end):
        print("NOTE: --days provided; ignoring --start/--end", file=sys.stderr)
        end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=days)
    else:
        end_dt = dtparser.parse(end).replace(tzinfo=timezone.utc) if end else datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_dt = dtparser.parse(start).replace(tzinfo=timezone.utc) if start else (end_dt - timedelta(days=180))

    all_trades = []
    for sym in symbols:
        print(f"Fetching {sym} {start_dt} → {end_dt} ...")
        df = fetch_candles(sym, start_dt, end_dt, granularity=granularity)
        if df.empty:
            print(f"[{sym}] No data returned.", file=sys.stderr)
            continue
        df = ensure_6h_alignment(df, granularity=granularity)
        trades = weekend_trap_trades_from_df(df, symbol=sym, tp_first=tp_first)
        print(f"[{sym}] Trades found: {len(trades)}")
        all_trades.extend(trades)

    trades_df = pd.DataFrame(all_trades)
    summary_df = summarize_trades(trades_df)

    return trades_df, summary_df

def main():
    parser = argparse.ArgumentParser(description="Weekend Liquidity Trap Backtest (Coinbase, 6H)")
    parser.add_argument("--symbols", type=str, default="BTC-USD,ETH-USD,XRP-USD,SOL-USD,ADA-USD", help="Comma-separated Coinbase product IDs")
    parser.add_argument("--start", type=str, default=None, help="Start date (UTC) e.g. 2024-01-01")
    parser.add_argument("--end", type=str, default=None, help="End date (UTC) e.g. 2025-08-25")
    parser.add_argument("--days", type=int, default=None, help="Lookback days (overrides start/end)")
    parser.add_argument("--tp_first", action="store_true", help="Assume TP hits before SL if both touched within the same candle")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    trades_df, summary_df = backtest_symbols(symbols, args.start, args.end, args.days, tp_first=args.tp_first)

    # Save outputs
    ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_path = f"weekend_trap_trades_{ts_label}.csv"
    summary_path = f"weekend_trap_summary_{ts_label}.csv"

    if not trades_df.empty:
        trades_df.to_csv(trades_path, index=False)
        print("\nPer-trade output ->", trades_path)
    else:
        print("\nNo trades found in the selected period.")

    if not summary_df.empty:
        summary_df.to_csv(summary_path, index=False)
        print("Summary output   ->", summary_path)

    # Pretty print summary
    if not summary_df.empty:
        print("\n=== Weekend Trap Summary ===")
        print(summary_df.to_string(index=False))
        print("\nTip: Re-run with --tp_first to see sensitivity to intrabar fill assumption.")
        print("     Adjust symbol list / dates to expand coverage.")

if __name__ == "__main__":
    main()
