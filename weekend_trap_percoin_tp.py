#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weekend Liquidity Trap — Per-Coin TP (R-Multiples) + MFE/MAE
Coinbase 6H, no API key required.

Key differences vs midpoint TP:
- TP is set per-coin as: entry ± (TP_R[symbol] * risk)
  where risk = |entry - SL| and TP_R is a per-coin multiple (e.g., 2.0 for 2R).
- SL is still entry ± (sl_mult * weekend_extension).

Inputs:
- --symbols: comma list of Coinbase product IDs (BTC-USD, ...)
- --tp_rules_csv: optional CSV with per-coin TP rules. Supports either:
    * column 'tp_r' (numeric), or
    * column 'tp_recommendation' with values like {'2R+','1R','≤1R (tight exits)','≤1R / Skip'}
      which will be mapped to {2.0, 1.0, 0.5, 0.5}
- --default_tp_r: fallback TP multiple if a symbol is not found in the CSV (default 1.0)
- --skip: optional comma list of symbols to exclude

Other controls:
- --sl_mult (default 0.25), --reentry_window (default 2), --tp_first (TP priority), --days or --start/--end

Outputs:
- grid_results/trades_percoinTP_<ts>.csv (per trade with MFE/MAE)
- grid_results/summary_percoinTP_<ts>.csv (per-coin and ALL aggregates)
"""

import argparse
import time
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

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
                r = requests.get(url, params=params, headers={"User-Agent":"weekend-trap-percoin-tp/1.0"}, timeout=20)
                if r.status_code == 429:
                    time.sleep(pause_sec * (2 ** attempt)); continue
                r.raise_for_status()
                rows.extend(r.json())
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
    df = df[~df["time"].duplicated()].reset_index(drop=True)
    return df

def load_tp_rules(csv_path: Optional[str], default_tp_r: float) -> Dict[str, float]:
    rules: Dict[str, float] = {}
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Normalize symbol case to uppercase
        if "symbol" not in df.columns:
            raise ValueError("tp_rules_csv must include a 'symbol' column.")
        if "tp_r" in df.columns:
            for _, row in df.iterrows():
                sym = str(row["symbol"]).upper()
                try:
                    rules[sym] = float(row["tp_r"])
                except:
                    continue
        elif "tp_recommendation" in df.columns:
            def map_label(lbl: str) -> float:
                s = str(lbl).strip().lower()
                if "2r" in s: return 2.0
                if "1r" in s: return 1.0
                if "≤" in s or "<=" in s or "tight" in s or "skip" in s: return 0.5
                return default_tp_r
            for _, row in df.iterrows():
                sym = str(row["symbol"]).upper()
                rules[sym] = map_label(row["tp_recommendation"])
        else:
            print("[WARN] tp_rules_csv has no 'tp_r' or 'tp_recommendation' column; using defaults.", file=sys.stderr)
    return rules

def compute_mfe_mae(path_df: pd.DataFrame, entry: float, sl: float, side: str, end_idx: int) -> Tuple[float, float]:
    if side == "long":
        risk = max(1e-12, entry - sl)
        mfe = (path_df.loc[:end_idx, "high"].max() - entry) / risk
        mae = (path_df.loc[:end_idx, "low"].min()  - entry) / risk
    else:
        risk = max(1e-12, sl - entry)
        mfe = (entry - path_df.loc[:end_idx, "low"].min()) / risk
        mae = (entry - path_df.loc[:end_idx, "high"].max()) / risk
    return float(mfe), float(mae)

def simulate_path_with_mfe(path_df: pd.DataFrame, entry: float, sl: float, tp: float, side: str, tp_first: bool = False):
    exit_idx = len(path_df) - 1
    for i in range(len(path_df)):
        row = path_df.iloc[i]
        high = float(row["high"]); low = float(row["low"]); t = row["time"]

        if side == "short":
            hit_sl = high >= sl; hit_tp = low <= tp
            if hit_sl and hit_tp:
                if tp_first: hit_sl = False
                else: hit_tp = False
            if hit_sl:
                mfe, mae = compute_mfe_mae(path_df, entry, sl, side, i)
                return ("SL", sl, t, -1.0, ((sl/entry)-1.0)*-100.0, mfe, mae)
            if hit_tp:
                sl_dist = sl - entry; tp_dist = entry - tp
                r = (tp_dist / sl_dist) if sl_dist > 0 else 0.0
                mfe, mae = compute_mfe_mae(path_df, entry, sl, side, i)
                return ("TP", tp, t, r, ((tp/entry)-1.0)*-100.0, mfe, mae)

        else:
            hit_sl = low <= sl; hit_tp = high >= tp
            if hit_sl and hit_tp:
                if tp_first: hit_sl = False
                else: hit_tp = False
            if hit_sl:
                mfe, mae = compute_mfe_mae(path_df, entry, sl, side, i)
                return ("SL", sl, t, -1.0, ((sl/entry)-1.0)*100.0, mfe, mae)
            if hit_tp:
                sl_dist = entry - sl; tp_dist = tp - entry
                r = (tp_dist / sl_dist) if sl_dist > 0 else 0.0
                mfe, mae = compute_mfe_mae(path_df, entry, sl, side, i)
                return ("TP", tp, t, r, ((tp/entry)-1.0)*100.0, mfe, mae)

    # time exit
    last = path_df.iloc[-1]; t = last["time"]; exit_price = float(last["close"])
    if side == "short":
        sl_dist = sl - entry; r = ((entry - exit_price) / sl_dist) if sl_dist > 0 else 0.0
        ret_pct = ((exit_price/entry) - 1.0) * -100.0
    else:
        sl_dist = entry - sl; r = ((exit_price - entry) / sl_dist) if sl_dist > 0 else 0.0
        ret_pct = ((exit_price/entry) - 1.0) * 100.0
    mfe, mae = compute_mfe_mae(path_df, entry, sl, side, len(path_df)-1)
    return ("TimeExit", exit_price, t, r, ret_pct, mfe, mae)

def weekend_trap_trades(df6h: pd.DataFrame, symbol: str, sl_mult: float, reentry_window: int, tp_first: bool, tp_r_multiple: float) -> List[Dict]:
    if df6h.empty:
        return []
    d = df6h.copy()
    d["dow"] = d["time"].dt.weekday
    d["date"] = d["time"].dt.date

    trades = []
    friday_dates = sorted(set(d.loc[d["dow"] == 4, "date"]))

    for fri_date in friday_dates:
        fri_df = d.loc[d["date"] == fri_date]
        if fri_df.empty:
            continue
        fri_high = fri_df["high"].max()
        fri_low  = fri_df["low"].min()
        if not np.isfinite(fri_high) or not np.isfinite(fri_low):
            continue

        sat_date = fri_date + timedelta(days=1)
        sun_date = fri_date + timedelta(days=2)
        wknd_df = d.loc[d["date"].isin([sat_date, sun_date])].reset_index(drop=True)
        if wknd_df.empty:
            continue

        took_trade = False
        for i in range(len(wknd_df) - 1):
            row = wknd_df.iloc[i]

            # upper outside -> potential short
            if row["close"] > fri_high:
                for w in range(1, reentry_window + 1):
                    if i + w >= len(wknd_df):
                        break
                    nxt = wknd_df.iloc[i + w]
                    if nxt["close"] <= fri_high:
                        ext_high = max(wknd_df.loc[i:i+w, "high"])
                        ext = max(0.0, ext_high - fri_high)
                        if ext <= 0: break
                        nxt_time = nxt["time"]
                        pos = d.index[d["time"] == nxt_time]
                        if len(pos) == 0: break
                        idx = pos[0]
                        if idx + 1 >= len(d): break
                        entry_row = d.iloc[idx + 1]
                        entry_time = entry_row["time"]
                        entry = float(entry_row["open"])

                        sl = entry + sl_mult * ext
                        risk = max(1e-12, sl - entry)
                        tp = entry - tp_r_multiple * risk  # short target below entry

                        mon_date = fri_date + timedelta(days=3)
                        window_df = d.loc[(d["time"] >= entry_time) & (d["date"].isin([sat_date, sun_date, mon_date]))].reset_index(drop=True)
                        if window_df.empty: break

                        outcome, exit_price, exit_time, r_mult, ret_pct, mfe_r, mae_r = simulate_path_with_mfe(window_df, entry, sl, tp, side="short", tp_first=tp_first)
                        trades.append({
                            "symbol": symbol, "friday_date": str(fri_date), "side": "SHORT",
                            "entry_time": entry_time.isoformat(), "entry": entry, "tp": tp, "sl": sl,
                            "friday_high": fri_high, "friday_low": fri_low, "extension": ext, "reentry_wait": w,
                            "outcome": outcome, "exit_time": exit_time.isoformat() if exit_time else None,
                            "exit": exit_price, "R": r_mult, "return_pct": ret_pct, "MFE_R": mfe_r, "MAE_R": mae_r,
                            "sl_mult": sl_mult, "tp_r": tp_r_multiple, "reentry_window": reentry_window, "tp_first": tp_first
                        })
                        took_trade = True
                        break
                if took_trade: break

            # lower outside -> potential long
            if row["close"] < fri_low:
                for w in range(1, reentry_window + 1):
                    if i + w >= len(wknd_df):
                        break
                    nxt = wknd_df.iloc[i + w]
                    if nxt["close"] >= fri_low:
                        ext_low = min(wknd_df.loc[i:i+w, "low"])
                        ext = max(0.0, fri_low - ext_low)
                        if ext <= 0: break
                        nxt_time = nxt["time"]
                        pos = d.index[d["time"] == nxt_time]
                        if len(pos) == 0: break
                        idx = pos[0]
                        if idx + 1 >= len(d): break
                        entry_row = d.iloc[idx + 1]
                        entry_time = entry_row["time"]
                        entry = float(entry_row["open"])

                        sl = entry - sl_mult * ext
                        risk = max(1e-12, entry - sl)
                        tp = entry + tp_r_multiple * risk  # long target above entry

                        mon_date = fri_date + timedelta(days=3)
                        window_df = d.loc[(d["time"] >= entry_time) & (d["date"].isin([sat_date, sun_date, mon_date]))].reset_index(drop=True)
                        if window_df.empty: break

                        outcome, exit_price, exit_time, r_mult, ret_pct, mfe_r, mae_r = simulate_path_with_mfe(window_df, entry, sl, tp, side="long", tp_first=tp_first)
                        trades.append({
                            "symbol": symbol, "friday_date": str(fri_date), "side": "LONG",
                            "entry_time": entry_time.isoformat(), "entry": entry, "tp": tp, "sl": sl,
                            "friday_high": fri_high, "friday_low": fri_low, "extension": ext, "reentry_wait": w,
                            "outcome": outcome, "exit_time": exit_time.isoformat() if exit_time else None,
                            "exit": exit_price, "R": r_mult, "return_pct": ret_pct, "MFE_R": mfe_r, "MAE_R": mae_r,
                            "sl_mult": sl_mult, "tp_r": tp_r_multiple, "reentry_window": reentry_window, "tp_first": tp_first
                        })
                        took_trade = True
                        break
                if took_trade: break

    return trades

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

def main():
    parser = argparse.ArgumentParser(description="Weekend Trap — Per-Coin TP (R-Multiples) + MFE/MAE")
    parser.add_argument("--symbols", type=str, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--sl_mult", type=float, default=0.25)
    parser.add_argument("--reentry_window", type=int, default=2)
    parser.add_argument("--tp_first", action="store_true")
    parser.add_argument("--tp_rules_csv", type=str, default=None)
    parser.add_argument("--default_tp_r", type=float, default=1.0)
    parser.add_argument("--skip", type=str, default="")

    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    skip_set = set([s.strip().upper() for s in args.skip.split(",") if s.strip()])

    # date bounds
    if args.days is not None and (args.start or args.end):
        print("NOTE: --days provided; ignoring --start/--end", file=sys.stderr)
        end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=args.days)
    else:
        end_dt = dtparser.parse(args.end).replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_dt = dtparser.parse(args.start).replace(tzinfo=timezone.utc) if args.start else (end_dt - timedelta(days=365))

    # load TP rules
    tp_rules = load_tp_rules(args.tp_rules_csv, args.default_tp_r)

    # fetch once per symbol
    print(f"Fetching data {start_dt} → {end_dt}")
    sym2df = {}
    for sym in symbols:
        if sym in skip_set:
            print(f"  [SKIP] {sym} (in skip list)")
            continue
        print(f"  {sym} ...")
        df = fetch_candles(sym, start_dt, end_dt, granularity=21600)
        if df.empty:
            print(f"  [WARN] No data for {sym}")
        sym2df[sym] = df

    outdir = "grid_results"
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_trades = []
    for sym, df in sym2df.items():
        if df is None or df.empty:
            continue
        tp_r = tp_rules.get(sym, args.default_tp_r)
        print(f"Running {sym} with TP_R={tp_r}")
        trades = weekend_trap_trades(df, sym, sl_mult=args.sl_mult, reentry_window=args.reentry_window, tp_first=args.tp_first, tp_r_multiple=tp_r)
        all_trades.extend(trades)

    trades_df = pd.DataFrame(all_trades)
    trades_path = os.path.join(outdir, f"trades_percoinTP_{ts}.csv")
    if not trades_df.empty:
        trades_df.to_csv(trades_path, index=False)
        print("Per-trade output ->", trades_path)
    else:
        print("No trades generated.")

    summary_df = summarize(trades_df)
    summary_path = os.path.join(outdir, f"summary_percoinTP_{ts}.csv")
    summary_df.to_csv(summary_path, index=False)
    print("Summary output   ->", summary_path)

    if not summary_df.empty:
        print("\n=== Per-Coin TP Summary ===")
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
