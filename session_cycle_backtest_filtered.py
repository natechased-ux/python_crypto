# session_cycle_backtest_filtered.py
# Same as session_cycle_backtest.py but adds --whitelist to restrict traded sessions.
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set

import pandas as pd
import numpy as np

SESSION_DEFS = {
    "ASIA":   {"start_hour": 23, "end_hour": 7},
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
    p = argparse.ArgumentParser(description="Time-of-Day & Liquidity Cycle Backtester (Filtered)")
    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--csv", type=str, help="Path to CSV with all symbols")
    gsrc.add_argument("--csv_dir", type=str, help="Path to folder with one CSV per symbol")

    p.add_argument("--symbols", nargs="+", help="Symbols to include")
    p.add_argument("--lookback_days", type=int, default=45)
    p.add_argument("--start_date", type=str, default=None)
    p.add_argument("--end_date", type=str, default=None)
    p.add_argument("--min_days_for_trade", type=int, default=15)
    p.add_argument("--one_trade_per_coin_per_day", action="store_true")

    p.add_argument("--risk", nargs="+", default=["basis=range","tp_mult=1.0","sl_mult=0.75","atr_len=14"])
    p.add_argument("--out_prefix", type=str, default="session_cycle_filtered")

    # NEW: whitelist CSV (columns: symbol,session). If absent, trades all sessions
    p.add_argument("--whitelist", type=str, default=None, help="CSV with columns symbol,session to allow")

    return p.parse_args()

def load_csv_all(path: str, symbols: Optional[List[str]]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must include 'timestamp' column")
    if "symbol" not in df.columns and (symbols is None or len(symbols) != 1):
        raise ValueError("CSV without 'symbol' column requires exactly one --symbols value.")
    if "symbol" not in df.columns:
        df["symbol"] = symbols[0]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["symbol","timestamp"]).reset_index(drop=True)

def load_csv_dir(path: str, symbols: Optional[List[str]]) -> pd.DataFrame:
    rows = []
    for f in os.listdir(path):
        if not f.lower().endswith(".csv"): continue
        sym = f.rsplit(".",1)[0]
        if symbols is not None and sym not in symbols: continue
        tmp = pd.read_csv(os.path.join(path,f))
        if "timestamp" not in tmp.columns: raise ValueError(f"{f} missing 'timestamp'")
        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True)
        tmp["symbol"] = sym
        rows.append(tmp)
    if not rows: raise ValueError("No matching CSVs found")
    df = pd.concat(rows, ignore_index=True)
    return df.sort_values(["symbol","timestamp"]).reset_index(drop=True)

def clip_date_range(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start: df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end: df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    return df

def compute_atr(df: pd.DataFrame, n: int) -> pd.Series:
    high = df["high"].astype(float); low = df["low"].astype(float); close = df["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def label_sessions(ts: pd.Series) -> pd.Series:
    h = ts.dt.hour
    asia = (h >= SESSION_DEFS["ASIA"]["start_hour"]) | (h < SESSION_DEFS["ASIA"]["end_hour"])
    eu = (h >= SESSION_DEFS["EUROPE"]["start_hour"]) & (h < SESSION_DEFS["EUROPE"]["end_hour"])
    us = (h >= SESSION_DEFS["US"]["start_hour"]) & (h < SESSION_DEFS["US"]["end_hour"])
    out = pd.Series(index=ts.index, dtype="object")
    out[asia] = "ASIA"; out[eu] = "EUROPE"; out[us] = "US"
    return out

def backtest(df: pd.DataFrame, symbols: List[str], lookback_days: int, min_days_for_trade: int,
             risk_cfg: RiskConfig, one_trade_per_coin_per_day: bool, allow: Optional[Set[Tuple[str,str]]]):
    all_trades = []; session_stats_rows = []
    for sym in symbols:
        d = df[df["symbol"] == sym].copy()
        if d.empty: continue
        d["session"] = label_sessions(d["timestamp"]); d = d.dropna(subset=["session"])

        for sname in SESSION_DEFS.keys():
            if allow is not None and (sym, sname) not in allow:
                continue  # skip non-whitelisted sessions

            ds = d[d["session"] == sname].copy()
            if ds.empty: continue
            ds["day"] = ds["timestamp"].dt.floor("D")

            if risk_cfg.basis == "atr":
                ds["atr"] = compute_atr(ds, risk_cfg.atr_len)
                ds["atr_pct"] = ds["atr"] / ds["close"]

            agg = ds.groupby("day").agg(
                open=("open","first"), close=("close","last"),
                high=("high","max"), low=("low","min")
            ).dropna()
            if agg.empty: continue
            agg["ret_pct"] = (agg["close"] - agg["open"])/agg["open"]
            agg["abs_move_pct"] = agg["ret_pct"].abs()
            if risk_cfg.basis == "atr":
                day_atr = ds.groupby("day")["atr_pct"].mean().reindex(agg.index)
                agg["atr_pct"] = day_atr

            days = sorted(agg.index.unique())
            last_trade_day = None

            for i, day in enumerate(days):
                look_end_idx = i - 1
                if look_end_idx < 0: continue
                look_start_day = days[max(0, i - lookback_days):look_end_idx+1][0]
                look = agg.loc[look_start_day:days[look_end_idx]]
                if len(look) < min_days_for_trade: continue
                if one_trade_per_coin_per_day and last_trade_day == day: continue

                mean_ret = look["ret_pct"].mean()
                avg_abs = look["abs_move_pct"].mean()
                avg_risk_unit = (look["atr_pct"].dropna().mean() if risk_cfg.basis=="atr" else avg_abs)
                if pd.isna(avg_risk_unit) or avg_risk_unit == 0: continue

                direction = 1 if mean_ret > 0 else -1

                intrabar = ds[ds["day"] == day]
                if intrabar.empty: continue
                entry_time = intrabar["timestamp"].iloc[0]
                entry = float(intrabar["open"].iloc[0])
                tp_pct = risk_cfg.tp_mult * avg_risk_unit
                sl_pct = risk_cfg.sl_mult * avg_risk_unit

                if direction > 0:
                    tp = entry * (1 + tp_pct); sl = entry * (1 - sl_pct)
                else:
                    tp = entry * (1 - tp_pct); sl = entry * (1 + sl_pct)

                exit_price = None; exit_time = None; outcome = None
                for _, r in intrabar.iterrows():
                    hi = float(r["high"]); lo = float(r["low"]); t = r["timestamp"]
                    if direction > 0:
                        hit_tp = hi >= tp; hit_sl = lo <= sl
                    else:
                        hit_tp = lo <= tp; hit_sl = hi >= sl
                    if hit_tp and hit_sl:
                        exit_price = sl; exit_time = t; outcome = "SL"; break
                    elif hit_tp:
                        exit_price = tp; exit_time = t; outcome = "TP"; break
                    elif hit_sl:
                        exit_price = sl; exit_time = t; outcome = "SL"; break
                if exit_price is None:
                    exit_price = float(intrabar["close"].iloc[-1])
                    exit_time = intrabar["timestamp"].iloc[-1]; outcome = "EOD"

                ret_pct = (exit_price - entry) / entry * direction
                all_trades.append({
                    "symbol": sym, "session": sname, "day": str(day.date()),
                    "entry_time": entry_time.isoformat(), "exit_time": exit_time.isoformat(),
                    "direction": "LONG" if direction>0 else "SHORT",
                    "entry": entry, "tp": tp, "sl": sl, "exit": exit_price,
                    "outcome": outcome, "ret_pct": ret_pct,
                    "mean_ret_lookback": mean_ret, "risk_unit": avg_risk_unit,
                })

            session_stats_rows.append({
                "symbol": sym, "session": sname, "days": len(agg),
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

def summarize(trades: pd.DataFrame):
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    overall = pd.DataFrame({
        "trades":[len(trades)],
        "win_rate":[(trades["ret_pct"]>0).mean()],
        "avg_ret_pct":[trades["ret_pct"].mean()],
        "median_ret_pct":[trades["ret_pct"].median()],
        "cum_ret_pct":[(trades["ret_pct"]+1).prod()-1],
        "tp_rate":[(trades["outcome"]=="TP").mean()],
        "sl_rate":[(trades["outcome"]=="SL").mean()],
    })
    by_symbol_session = trades.groupby(["symbol","session"]).agg(
        trades=("ret_pct","count"),
        win_rate=("ret_pct", lambda x:(x>0).mean()),
        avg_ret_pct=("ret_pct","mean"),
        median_ret_pct=("ret_pct","median"),
        cum_ret_pct=("ret_pct", lambda x:(x+1).prod()-1),
        tp_rate=("outcome", lambda x:(x=="TP").mean()),
        sl_rate=("outcome", lambda x:(x=="SL").mean()),
    ).reset_index().sort_values(["symbol","session"])
    by_day = trades.groupby("day").agg(
        trades=("ret_pct","count"),
        day_ret_pct=("ret_pct", lambda x:(x+1).prod()-1)
    ).reset_index()
    return overall, by_symbol_session, by_day

def main():
    args = parse_args()
    rc_kwargs = {}
    for tok in args.risk:
        if "=" in tok:
            k, v = tok.split("=",1)
            if k=="basis": rc_kwargs[k]=v
            elif k in ("tp_mult","sl_mult"): rc_kwargs[k]=float(v)
            elif k=="atr_len": rc_kwargs[k]=int(v)
    risk_cfg = RiskConfig(**rc_kwargs)

    if args.csv:
        df = load_csv_all(args.csv, args.symbols)
    else:
        df = load_csv_dir(args.csv_dir, args.symbols)

    if args.symbols:
        df = df[df["symbol"].isin(args.symbols)].copy()

    needed = {"timestamp","open","high","low","close","volume","symbol"}
    missing = needed - set(df.columns)
    if missing: raise ValueError(f"Missing columns: {missing}")

    df = clip_date_range(df, args.start_date, args.end_date)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)

    allow = None
    if args.whitelist:
        wl = pd.read_csv(args.whitelist)
        need = {"symbol","session"}
        if not need.issubset(wl.columns):
            raise ValueError("whitelist CSV must have columns: symbol,session")
        allow = {(row["symbol"], row["session"]) for _, row in wl.iterrows()}

    symbols = sorted(df["symbol"].unique().tolist())
    trades, session_stats = backtest(df, symbols, args.lookback_days, args.min_days_for_trade,
                                     risk_cfg, args.one_trade_per_coin_per_day, allow)

    overall, by_symbol_session, by_day = summarize(trades)

    trades.to_csv(f"{args.out_prefix}_trades.csv", index=False)
    session_stats.to_csv(f"{args.out_prefix}_session_stats.csv", index=False)
    by_symbol_session.to_csv(f"{args.out_prefix}_by_symbol_session.csv", index=False)
    by_day.to_csv(f"{args.out_prefix}_by_day.csv", index=False)
    print("Saved:")
    print(f"{args.out_prefix}_trades.csv")
    print(f"{args.out_prefix}_session_stats.csv")
    print(f"{args.out_prefix}_by_symbol_session.csv")
    print(f"{args.out_prefix}_by_day.csv")

if __name__ == "__main__":
    main()
