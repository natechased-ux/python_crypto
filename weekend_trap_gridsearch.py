#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weekend Liquidity Trap - Parameter Sweep (Coinbase, 6H)
- Sweeps over SL multipliers, TP mid coefficients, re-entry windows, and intrabar priority.
- Reuses Coinbase public candles (no API key).

Parameters to sweep (examples):
--sl_mult 0.25,0.5,0.75,1.0
--tp_mid 0.4,0.5,0.6
--reentry_window 1,2
--tp_first false,true

Output:
- One CSV per parameter combo (per-trade logs) in ./grid_results/
- One master CSV "grid_summary_<timestamp>.csv" with per-symbol and overall metrics per combo.

Usage example:
py weekend_trap_gridsearch.py ^
  --symbols BTC-USD,ETH-USD,XRP-USD,SOL-USD,ADA-USD ^
  --days 365 ^
  --sl_mult 0.25,0.5,0.75,1.0 ^
  --tp_mid 0.4,0.5,0.6 ^
  --reentry_window 1,2 ^
  --tp_first false,true

Dependencies: pandas, numpy, requests, python-dateutil
"""

import argparse
import time
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import requests
from dateutil import parser as dtparser

CB_BASE = "https://api.exchange.coinbase.com"

def isoformat(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat()

def fetch_candles(product_id: str, start: datetime, end: datetime, granularity: int = 21600, max_per_req: int = 300, pause_sec: float = 0.35, max_retries: int = 5) -> pd.DataFrame:
    rows = []
    chunk_seconds = granularity * max_per_req
    t0 = start.astimezone(timezone.utc)
    t1 = end.astimezone(timezone.utc)
    if t1 <= t0:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])

    while t0 < t1:
        t2 = min(t0 + timedelta(seconds=chunk_seconds), t1)
        params = {"granularity": granularity, "start": isoformat(t0), "end": isoformat(t2)}
        url = f"{CB_BASE}/products/{product_id}/candles"
        for attempt in range(max_retries):
            try:
                r = requests.get(url, params=params, headers={"User-Agent":"weekend-trap-grid/1.0"}, timeout=20)
                if r.status_code == 429:
                    time.sleep(pause_sec * (2 ** attempt)); continue
                r.raise_for_status()
                data = r.json()
                rows.extend(data)
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
    df = df.sort_values("epoch").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df[["time","open","high","low","close","volume"]].astype({
        "open": float, "high": float, "low": float, "close": float, "volume": float
    })
    # drop duplicate buckets
    df = df[~df["time"].duplicated()].reset_index(drop=True)
    return df

def weekend_trap_trades(df6h: pd.DataFrame, symbol: str, sl_mult: float, tp_mid_coeff: float, reentry_window: int, tp_first: bool) -> List[Dict]:
    """
    sl_mult: multiplier on the weekend extension for SL (default 0.5 previously)
    tp_mid_coeff: where to target between Friday low and high (0..1). 0.5 = exact midpoint.
    reentry_window: number of consecutive 6H candles allowed for re-entry after the outside close (>=1)
    tp_first: intrabar resolution (if both hit in same candle)
    """
    if df6h.empty:
        return []
    d = df6h.copy()
    d["dow"] = d["time"].dt.weekday
    d["date"] = d["time"].dt.date

    trades = []
    friday_dates = sorted(set(d.loc[d["dow"] == 4, "date"]))  # Fridays

    for fri_date in friday_dates:
        fri_df = d.loc[d["date"] == fri_date]
        if fri_df.empty: 
            continue
        fri_high = fri_df["high"].max()
        fri_low  = fri_df["low"].min()
        if not np.isfinite(fri_high) or not np.isfinite(fri_low): 
            continue
        # generalized "mid"
        fri_target = fri_low + tp_mid_coeff * (fri_high - fri_low)

        sat_date = fri_date + timedelta(days=1)
        sun_date = fri_date + timedelta(days=2)
        wknd_df = d.loc[d["date"].isin([sat_date, sun_date])].reset_index(drop=True)
        if wknd_df.empty: 
            continue

        took_trade = False

        # scan for first outside close
        for i in range(len(wknd_df) - 1):
            row = wknd_df.iloc[i]

            # upper outside close
            if row["close"] > fri_high:
                # allow re-entry within "reentry_window" candles
                for w in range(1, reentry_window + 1):
                    if i + w >= len(wknd_df):
                        break
                    nxt = wknd_df.iloc[i + w]
                    if nxt["close"] <= fri_high:
                        ext_high = max(wknd_df.loc[i:i+w, "high"])
                        ext = max(0.0, ext_high - fri_high)
                        if ext <= 0: 
                            break
                        # entry next candle after 'nxt'
                        nxt_time = nxt["time"]
                        pos = d.index[d["time"] == nxt_time]
                        if len(pos) == 0: break
                        idx = pos[0]
                        if idx + 1 >= len(d): break
                        entry_row = d.iloc[idx + 1]
                        entry_time = entry_row["time"]
                        entry = float(entry_row["open"])
                        if entry <= fri_target: 
                            break

                        sl = entry + sl_mult * ext
                        tp = fri_target

                        mon_date = fri_date + timedelta(days=3)
                        window_df = d.loc[(d["time"] >= entry_time) & (d["date"].isin([sat_date, sun_date, mon_date]))].reset_index(drop=True)
                        if window_df.empty: 
                            break

                        outcome, exit_price, exit_time, r_mult, ret_pct = simulate_path(window_df, entry, sl, tp, side="short", tp_first=tp_first)
                        trades.append({
                            "symbol": symbol, "friday_date": str(fri_date), "side": "SHORT",
                            "entry_time": entry_time.isoformat(), "entry": entry, "tp": tp, "sl": sl,
                            "friday_high": fri_high, "friday_low": fri_low, "target_coeff": tp_mid_coeff,
                            "extension": ext, "reentry_wait": w, "outcome": outcome,
                            "exit_time": exit_time.isoformat() if exit_time else None,
                            "exit": exit_price, "R": r_mult, "return_pct": ret_pct,
                            "sl_mult": sl_mult, "tp_mid_coeff": tp_mid_coeff, "reentry_window": reentry_window,
                            "tp_first": tp_first
                        })
                        took_trade = True
                        break
                if took_trade: break

            # lower outside close
            if row["close"] < fri_low:
                for w in range(1, reentry_window + 1):
                    if i + w >= len(wknd_df):
                        break
                    nxt = wknd_df.iloc[i + w]
                    if nxt["close"] >= fri_low:
                        ext_low = min(wknd_df.loc[i:i+w, "low"])
                        ext = max(0.0, fri_low - ext_low)
                        if ext <= 0: 
                            break
                        nxt_time = nxt["time"]
                        pos = d.index[d["time"] == nxt_time]
                        if len(pos) == 0: break
                        idx = pos[0]
                        if idx + 1 >= len(d): break
                        entry_row = d.iloc[idx + 1]
                        entry_time = entry_row["time"]
                        entry = float(entry_row["open"])
                        if entry >= fri_target: 
                            break

                        sl = entry - sl_mult * ext
                        tp = fri_target

                        mon_date = fri_date + timedelta(days=3)
                        window_df = d.loc[(d["time"] >= entry_time) & (d["date"].isin([sat_date, sun_date, mon_date]))].reset_index(drop=True)
                        if window_df.empty:
                            break

                        outcome, exit_price, exit_time, r_mult, ret_pct = simulate_path(window_df, entry, sl, tp, side="long", tp_first=tp_first)
                        trades.append({
                            "symbol": symbol, "friday_date": str(fri_date), "side": "LONG",
                            "entry_time": entry_time.isoformat(), "entry": entry, "tp": tp, "sl": sl,
                            "friday_high": fri_high, "friday_low": fri_low, "target_coeff": tp_mid_coeff,
                            "extension": ext, "reentry_wait": w, "outcome": outcome,
                            "exit_time": exit_time.isoformat() if exit_time else None,
                            "exit": exit_price, "R": r_mult, "return_pct": ret_pct,
                            "sl_mult": sl_mult, "tp_mid_coeff": tp_mid_coeff, "reentry_window": reentry_window,
                            "tp_first": tp_first
                        })
                        took_trade = True
                        break
                if took_trade: break

    return trades

def simulate_path(path_df: pd.DataFrame, entry: float, sl: float, tp: float, side: str, tp_first: bool = False):
    for i in range(len(path_df)):
        row = path_df.iloc[i]
        high = float(row["high"]); low = float(row["low"]); t = row["time"]

        if side == "short":
            hit_sl = high >= sl; hit_tp = low <= tp
            if hit_sl and hit_tp:
                if tp_first: hit_sl = False
                else: hit_tp = False
            if hit_sl:
                r = -1.0
                ret_pct = ((sl/entry) - 1.0) * -100.0
                return ("SL", sl, t, r, ret_pct)
            if hit_tp:
                sl_dist = sl - entry; tp_dist = entry - tp
                r = (tp_dist / sl_dist) if sl_dist > 0 else 0.0
                ret_pct = ((tp/entry) - 1.0) * -100.0
                return ("TP", tp, t, r, ret_pct)

        else:  # long
            hit_sl = low <= sl; hit_tp = high >= tp
            if hit_sl and hit_tp:
                if tp_first: hit_sl = False
                else: hit_tp = False
            if hit_sl:
                r = -1.0
                ret_pct = ((sl/entry) - 1.0) * 100.0
                return ("SL", sl, t, r, ret_pct)
            if hit_tp:
                sl_dist = entry - sl; tp_dist = tp - entry
                r = (tp_dist / sl_dist) if sl_dist > 0 else 0.0
                ret_pct = ((tp/entry) - 1.0) * 100.0
                return ("TP", tp, t, r, ret_pct)

    # Time exit at last candle (Mon end window)
    last = path_df.iloc[-1]
    exit_price = float(last["close"]); t = last["time"]
    if side == "short":
        sl_dist = sl - entry; r = ((entry - exit_price) / sl_dist) if sl_dist > 0 else 0.0
        ret_pct = ((exit_price/entry) - 1.0) * -100.0
    else:
        sl_dist = entry - sl; r = ((exit_price - entry) / sl_dist) if sl_dist > 0 else 0.0
        ret_pct = ((exit_price/entry) - 1.0) * 100.0
    return ("TimeExit", exit_price, t, r, ret_pct)

def summarize(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["symbol","trades","win_rate","avg_R","pf","avg_return_pct","expectancy_R"])
    grp = trades_df.groupby("symbol")
    rows = []
    for sym, g in grp:
        n = len(g); wins = (g["outcome"] == "TP").sum()
        win_rate = wins/n if n else 0.0
        avg_R = g["R"].mean() if n else 0.0
        gross_win = g.loc[g["R"] > 0, "R"].sum()
        gross_loss = -g.loc[g["R"] < 0, "R"].sum()
        pf = (gross_win/gross_loss) if gross_loss > 0 else np.nan
        avg_ret = g["return_pct"].mean() if n else 0.0
        rows.append({
            "symbol": sym, "trades": int(n),
            "win_rate": round(win_rate, 3),
            "avg_R": round(avg_R, 3),
            "pf": round(pf, 3) if not np.isnan(pf) else None,
            "avg_return_pct": round(avg_ret, 3),
            "expectancy_R": round(avg_R, 3)
        })
    # overall
    n = len(trades_df); wins = (trades_df["outcome"] == "TP").sum()
    win_rate = wins/n if n else 0.0
    avg_R = trades_df["R"].mean() if n else 0.0
    gross_win = trades_df.loc[trades_df["R"] > 0, "R"].sum()
    gross_loss = -trades_df.loc[trades_df["R"] < 0, "R"].sum()
    pf = (gross_win/gross_loss) if gross_loss > 0 else np.nan
    avg_ret = trades_df["return_pct"].mean() if n else 0.0
    rows.append({
        "symbol": "ALL", "trades": int(n),
        "win_rate": round(win_rate, 3),
        "avg_R": round(avg_R, 3),
        "pf": round(pf, 3) if not np.isnan(pf) else None,
        "avg_return_pct": round(avg_ret, 3),
        "expectancy_R": round(avg_R, 3)
    })
    return pd.DataFrame(rows)

def parse_bool_list(s: str) -> List[bool]:
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if tok in ("true","1","yes","y"): out.append(True)
        elif tok in ("false","0","no","n"): out.append(False)
    return out

def main():
    parser = argparse.ArgumentParser(description="Weekend Trap Grid Search (Coinbase 6H)")
    parser.add_argument("--symbols", type=str, default="BTC-USD,ETH-USD,XRP-USD,SOL-USD,ADA-USD")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--days", type=int, default=365)

    parser.add_argument("--sl_mult", type=str, default="0.5", help="Comma list of SL multipliers on extension (e.g., 0.25,0.5,0.75,1.0)")
    parser.add_argument("--tp_mid", type=str, default="0.5", help="Comma list of target coefficients in [0..1] (0.5=midpoint)")
    parser.add_argument("--reentry_window", type=str, default="1", help="Comma list of allowed re-entry windows (candles)")
    parser.add_argument("--tp_first", type=str, default="false,true", help="Comma list of booleans")

    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    # date bounds
    if args.days is not None and (args.start or args.end):
        print("NOTE: --days provided; ignoring --start/--end", file=sys.stderr)
        end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=args.days)
    else:
        end_dt = dtparser.parse(args.end).replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_dt = dtparser.parse(args.start).replace(tzinfo=timezone.utc) if args.start else (end_dt - timedelta(days=365))

    sl_mults = [float(x) for x in args.sl_mult.split(",")]
    tp_coeffs = [float(x) for x in args.tp_mid.split(",")]
    re_windows = [int(x) for x in args.reentry_window.split(",")]
    tp_firsts = parse_bool_list(args.tp_first)

    # fetch data once per symbol
    print(f"Fetching data {start_dt} → {end_dt}")
    sym2df = {}
    for sym in symbols:
        print(f"  {sym} ...")
        df = fetch_candles(sym, start_dt, end_dt, granularity=21600)
        if df.empty:
            print(f"  [WARN] No data for {sym}")
        sym2df[sym] = df

    # ensure output directory
    outdir = "grid_results"
    os.makedirs(outdir, exist_ok=True)

    # run grid
    rows_summary = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_path = os.path.join(outdir, f"grid_summary_{ts}.csv")

    total_runs = len(sl_mults)*len(tp_coeffs)*len(re_windows)*len(tp_firsts)
    run_idx = 0

    for slm in sl_mults:
        for tpc in tp_coeffs:
            for rew in re_windows:
                for tpf in tp_firsts:
                    run_idx += 1
                    print(f"\n[{run_idx}/{total_runs}] sl_mult={slm}, tp_mid={tpc}, reentry_window={rew}, tp_first={tpf}")
                    all_trades = []
                    for sym, df in sym2df.items():
                        if df is None or df.empty: 
                            continue
                        trades = weekend_trap_trades(df, sym, sl_mult=slm, tp_mid_coeff=tpc, reentry_window=rew, tp_first=tpf)
                        all_trades.extend(trades)
                    trades_df = pd.DataFrame(all_trades)
                    summary_df = summarize(trades_df)

                    # annotate with params
                    summary_df["sl_mult"] = slm
                    summary_df["tp_mid"] = tpc
                    summary_df["reentry_window"] = rew
                    summary_df["tp_first"] = tpf

                    # save per-run trade log
                    run_tag = f"sl{slm}_tp{tpc}_rw{rew}_tf{int(tpf)}"
                    trades_path = os.path.join(outdir, f"trades_{run_tag}_{ts}.csv")
                    if not trades_df.empty:
                        trades_df.to_csv(trades_path, index=False)

                    # append to master
                    rows_summary.append(summary_df)

    if rows_summary:
        master_df = pd.concat(rows_summary, ignore_index=True)
        master_df.to_csv(master_path, index=False)
        print(f"\nMaster summary saved to: {master_path}")
        print("Tip: Filter rows where symbol == 'ALL' to compare global performance across parameter sets.")
    else:
        print("\nNo results produced. Check your symbols/time range.")
    
if __name__ == "__main__":
    main()
