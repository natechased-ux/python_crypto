# session_cycle_backtest.py
# --------------------------------------------------------------
# Time-of-Day & Liquidity Cycle Strategy Backtester (Crypto)
#
# What it does
# ------------
# - Splits each day into sessions (ASIA, EUROPE, US) by UTC hours.
# - Walk-forward learns each coin's recent (rolling) session edge.
# - Pre-positions at session open in the learned direction.
# - Exits on TP/SL (based on avg session move or ATR) or session end.
# - Reports per-coin, per-session, and overall performance.
#
# Data expected
# -------------
# - OHLCV candles per symbol in a single CSV or multiple CSVs.
# - Columns: timestamp, open, high, low, close, volume, [symbol]
#   - timestamp must be ISO8601 in UTC (e.g., 2025-07-01T12:00:00Z)
#   - symbol is optional if you load one symbol at a time
#
# Quick start
# -----------
# python session_cycle_backtest.py --csv data.csv --symbols BTC-USD ETH-USD XRP-USD #   --freq 15min --lookback_days 45 --risk basis=range tp_mult=1.0 sl_mult=0.75
#
# Or ATR-based sizing:
# python session_cycle_backtest.py --csv data.csv --symbols BTC-USD ETH-USD #   --risk basis=atr atr_len=14 tp_mult=1.2 sl_mult=0.8
#
# You can also point to a folder of CSVs (one per symbol) via --csv_dir.
#
# Notes
# -----
# - Choose 15m candles for more realistic TP/SL intraday fills.
# - Sessions (UTC): ASIA 23:00-07:00, EUROPE 07:00-12:00, US 12:00-20:00.
# - Modify SESSION_DEFS below if you prefer different cuts.
# --------------------------------------------------------------

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ----------------------- Session definitions -----------------------
SESSION_DEFS = {
    "ASIA":   {"start_hour": 23, "end_hour": 7},   # wraps midnight
    "EUROPE": {"start_hour": 7,  "end_hour": 12},
    "US":     {"start_hour": 12, "end_hour": 20},
}

@dataclass
class RiskConfig:
    basis: str = "range"   # "range" or "atr"
    tp_mult: float = 1.0
    sl_mult: float = 0.75
    atr_len: int = 14

def parse_args():
    p = argparse.ArgumentParser(description="Time-of-Day & Liquidity Cycle Backtester")
    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--csv", type=str, help="Path to CSV with all symbols")
    gsrc.add_argument("--csv_dir", type=str, help="Path to folder with one CSV per symbol")

    p.add_argument("--symbols", nargs="+", help="Symbols to include (e.g., BTC-USD ETH-USD XRP-USD)")
    p.add_argument("--freq", type=str, default="15min", help="Candle frequency label (e.g., 1h, 15min). For reference only.")
    p.add_argument("--lookback_days", type=int, default=45, help="Rolling lookback for learning session edges")
    p.add_argument("--start_date", type=str, default=None, help="ISO date filter start (UTC)")
    p.add_argument("--end_date", type=str, default=None, help="ISO date filter end (UTC)")
    p.add_argument("--min_days_for_trade", type=int, default=15, help="Require this many prior session-days before trading")
    p.add_argument("--one_trade_per_coin_per_day", action="store_true", help="Limit to max 1 trade per coin per UTC day")

    # Risk config
    p.add_argument("--risk", nargs="+", default=["basis=range", "tp_mult=1.0", "sl_mult=0.75", "atr_len=14"],
                   help="Risk spec like basis=range tp_mult=1.0 sl_mult=0.75 atr_len=14 (atr_len used if basis=atr)")

    # Output
    p.add_argument("--out_prefix", type=str, default="session_cycle_results", help="Output file prefix (.csv)")

    return p.parse_args()

# ----------------------- Data loading helpers -----------------------
def load_csv_all(path: str, symbols: Optional[List[str]]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must include 'timestamp' column in UTC ISO format.")
    if "symbol" not in df.columns and (symbols is None or len(symbols) != 1):
        raise ValueError("CSV without 'symbol' column requires exactly one --symbols value.")
    if "symbol" not in df.columns:
        df["symbol"] = symbols[0]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df

def load_csv_dir(path: str, symbols: Optional[List[str]]) -> pd.DataFrame:
    rows = []
    files = os.listdir(path)
    for f in files:
        if not f.lower().endswith(".csv"):
            continue
        sym = f.rsplit(".", 1)[0]
        if symbols is not None and sym not in symbols:
            continue
        tmp = pd.read_csv(os.path.join(path, f))
        if "timestamp" not in tmp.columns:
            raise ValueError(f"{f} missing 'timestamp' column.")
        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True)
        tmp["symbol"] = sym
        rows.append(tmp)
    if not rows:
        raise ValueError("No matching CSVs found in folder.")
    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df

def clip_date_range(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    return df

# ----------------------- Feature calculations -----------------------
def compute_atr(df: pd.DataFrame, atr_len: int) -> pd.Series:
    # Expect per-symbol chunking before calling
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_len, min_periods=atr_len).mean()
    return atr

def label_sessions(ts: pd.Series) -> pd.Series:
    # Returns session name per timestamp, based on UTC hour
    hours = ts.dt.hour
    # handle wrap for ASIA 23-07
    asia = (hours >= SESSION_DEFS["ASIA"]["start_hour"]) | (hours < SESSION_DEFS["ASIA"]["end_hour"])
    europe = (hours >= SESSION_DEFS["EUROPE"]["start_hour"]) & (hours < SESSION_DEFS["EUROPE"]["end_hour"])
    us = (hours >= SESSION_DEFS["US"]["start_hour"]) & (hours < SESSION_DEFS["US"]["end_hour"])
    out = pd.Series(index=ts.index, dtype="object")
    out[asia] = "ASIA"
    out[europe] = "EUROPE"
    out[us] = "US"
    # Others (gaps) get NaN; we won't trade those periods
    return out

# ----------------------- Core backtest -----------------------
def backtest(df: pd.DataFrame,
             symbols: List[str],
             lookback_days: int,
             min_days_for_trade: int,
             risk_cfg: RiskConfig,
             one_trade_per_coin_per_day: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (trades, session_stats)
    """
    all_trades = []
    session_stats_rows = []

    for sym in symbols:
        d = df[df["symbol"] == sym].copy()
        if d.empty:
            continue
        d["session"] = label_sessions(d["timestamp"])
        d = d.dropna(subset=["session"])

        # Per-session segmentation
        for sname in SESSION_DEFS.keys():
            ds = d[d["session"] == sname].copy()
            if ds.empty: 
                continue

            # Attach day key for grouping by session-day
            ds["day"] = ds["timestamp"].dt.floor("D")

            # Pre-compute ATR% if needed
            if risk_cfg.basis == "atr":
                ds["atr"] = compute_atr(ds, risk_cfg.atr_len)
                ds["atr_pct"] = ds["atr"] / ds["close"]

            # Aggregate to session-day open/close + ranges for learning stats
            agg = ds.groupby("day").agg(
                open=("open", "first"),
                close=("close", "last"),
                high=("high", "max"),
                low=("low", "min")
            ).dropna()

            if agg.empty:
                continue

            agg["ret_pct"] = (agg["close"] - agg["open"]) / agg["open"]
            agg["abs_move_pct"] = agg["ret_pct"].abs()
            if risk_cfg.basis == "atr":
                # Session-day ATR% = mean of intraperiod atr_pct
                day_atr = ds.groupby("day")["atr_pct"].mean().reindex(agg.index)
                agg["atr_pct"] = day_atr

            # Walk-forward
            days = sorted(agg.index.unique())
            last_trade_day = None

            for i, day in enumerate(days):
                # rolling lookback window ends yesterday
                look_end_idx = i - 1
                if look_end_idx < 0:
                    continue
                look_start_day = days[max(0, i - lookback_days):look_end_idx + 1][0]
                lookback_slice = agg.loc[look_start_day:days[look_end_idx]]

                # Skip until we have enough history
                if len(lookback_slice) < min_days_for_trade:
                    continue

                # Optionally limit to one trade per coin per UTC day
                if one_trade_per_coin_per_day and last_trade_day == day:
                    continue

                # Learn edge: average session return
                mean_ret = lookback_slice["ret_pct"].mean()
                avg_abs = lookback_slice["abs_move_pct"].mean()
                if risk_cfg.basis == "atr":
                    avg_risk_unit = lookback_slice["atr_pct"].dropna().mean()
                else:
                    avg_risk_unit = avg_abs  # use average absolute session move

                if pd.isna(avg_risk_unit) or avg_risk_unit == 0:
                    continue

                direction = 1 if mean_ret > 0 else -1

                # Build trade at session open
                session_intrabar = ds[ds["day"] == day]
                if session_intrabar.empty:
                    continue
                entry_time = session_intrabar["timestamp"].iloc[0]
                entry = float(session_intrabar["open"].iloc[0])

                tp_pct = risk_cfg.tp_mult * avg_risk_unit
                sl_pct = risk_cfg.sl_mult * avg_risk_unit

                if direction > 0:
                    tp = entry * (1 + tp_pct)
                    sl = entry * (1 - sl_pct)
                else:
                    tp = entry * (1 - tp_pct)
                    sl = entry * (1 + sl_pct)

                exit_price = None
                exit_time = None
                outcome = None

                # Simulate intrabar: first touch wins (path-dependent)
                for _, r in session_intrabar.iterrows():
                    hi = float(r["high"]); lo = float(r["low"])
                    t = r["timestamp"]
                    if direction > 0:
                        hit_tp = hi >= tp
                        hit_sl = lo <= sl
                    else:
                        hit_tp = lo <= tp
                        hit_sl = hi >= sl

                    if hit_tp and hit_sl:
                        # Ambiguity; conservative: take SL first
                        exit_price = sl
                        exit_time = t
                        outcome = "SL"
                        break
                    elif hit_tp:
                        exit_price = tp
                        exit_time = t
                        outcome = "TP"
                        break
                    elif hit_sl:
                        exit_price = sl
                        exit_time = t
                        outcome = "SL"
                        break

                if exit_price is None:
                    # Exit at session close
                    exit_price = float(session_intrabar["close"].iloc[-1])
                    exit_time = session_intrabar["timestamp"].iloc[-1]
                    outcome = "EOD"

                # PnL in percent relative to entry (no fees)
                ret_pct = (exit_price - entry) / entry * direction

                all_trades.append({
                    "symbol": sym,
                    "session": sname,
                    "day": str(day.date()),
                    "entry_time": entry_time.isoformat(),
                    "exit_time": exit_time.isoformat(),
                    "direction": "LONG" if direction > 0 else "SHORT",
                    "entry": entry,
                    "tp": tp,
                    "sl": sl,
                    "exit": exit_price,
                    "outcome": outcome,
                    "ret_pct": ret_pct,
                    "mean_ret_lookback": mean_ret,
                    "risk_unit": avg_risk_unit,
                })

            # Per-session summary stats (for info)
            session_stats_rows.append({
                "symbol": sym,
                "session": sname,
                "days": len(agg),
                "mean_ret_pct": agg["ret_pct"].mean(),
                "median_ret_pct": agg["ret_pct"].median(),
                "win_rate_gt0": (agg["ret_pct"] > 0).mean(),
                "avg_abs_move_pct": agg["abs_move_pct"].mean(),
                "std_ret_pct": agg["ret_pct"].std(),
            })

    trades = pd.DataFrame(all_trades)
    session_stats = pd.DataFrame(session_stats_rows)

    if not trades.empty:
        trades["ret_pct"] = trades["ret_pct"].astype(float)
    return trades, session_stats

# ----------------------- Reporting -----------------------
def summarize(trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Overall
    overall = pd.DataFrame({
        "trades": [len(trades)],
        "win_rate": [(trades["ret_pct"] > 0).mean()],
        "avg_ret_pct": [trades["ret_pct"].mean()],
        "median_ret_pct": [trades["ret_pct"].median()],
        "cum_ret_pct": [(trades["ret_pct"] + 1).prod() - 1],
        "tp_rate": [(trades["outcome"] == "TP").mean()],
        "sl_rate": [(trades["outcome"] == "SL").mean()],
    })

    by_symbol_session = trades.groupby(["symbol", "session"]).agg(
        trades=("ret_pct", "count"),
        win_rate=("ret_pct", lambda x: (x > 0).mean()),
        avg_ret_pct=("ret_pct", "mean"),
        median_ret_pct=("ret_pct", "median"),
        cum_ret_pct=("ret_pct", lambda x: (x + 1).prod() - 1),
        tp_rate=("outcome", lambda x: (x == "TP").mean()),
        sl_rate=("outcome", lambda x: (x == "SL").mean()),
    ).reset_index().sort_values(["symbol", "session"])

    by_day = trades.groupby("day").agg(
        trades=("ret_pct", "count"),
        day_ret_pct=("ret_pct", lambda x: (x + 1).prod() - 1)
    ).reset_index()

    return overall, by_symbol_session, by_day

# ----------------------- Main -----------------------
def main():
    args = parse_args()

    # Risk config parse
    rc_kwargs = {}
    for tok in args.risk:
        if "=" in tok:
            k, v = tok.split("=", 1)
            if k == "basis":
                rc_kwargs[k] = v
            elif k in ("tp_mult", "sl_mult"):
                rc_kwargs[k] = float(v)
            elif k == "atr_len":
                rc_kwargs[k] = int(v)
    risk_cfg = RiskConfig(**rc_kwargs)

    # Load data
    if args.csv:
        df = load_csv_all(args.csv, args.symbols)
    else:
        df = load_csv_dir(args.csv_dir, args.symbols)

    # Filter symbols if provided
    if args.symbols:
        df = df[df["symbol"].isin(args.symbols)].copy()

    # Basic column checks
    needed = {"timestamp", "open", "high", "low", "close", "volume", "symbol"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Date filtering
    df = clip_date_range(df, args.start_date, args.end_date)

    # Ensure UTC and sorted
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Run backtest
    symbols = sorted(df["symbol"].unique().tolist())
    trades, session_stats = backtest(
        df, symbols,
        lookback_days=args.lookback_days,
        min_days_for_trade=args.min_days_for_trade,
        risk_cfg=risk_cfg,
        one_trade_per_coin_per_day=args.one_trade_per_coin_per_day
    )

    # Summaries
    overall, by_symbol_session, by_day = summarize(trades)

    # Save outputs
    out_trades = f"{args.out_prefix}_trades.csv"
    out_stats = f"{args.out_prefix}_session_stats.csv"
    out_sym = f"{args.out_prefix}_by_symbol_session.csv"
    out_day = f"{args.out_prefix}_by_day.csv"

    trades.to_csv(out_trades, index=False)
    session_stats.to_csv(out_stats, index=False)
    by_symbol_session.to_csv(out_sym, index=False)
    by_day.to_csv(out_day, index=False)

    print("Saved:")
    print(out_trades)
    print(out_stats)
    print(out_sym)
    print(out_day)

if __name__ == "__main__":
    main()
