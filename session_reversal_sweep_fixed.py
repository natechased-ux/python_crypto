
"""
Session Reversal Fade — Parameter Sweep

What it does
------------
Runs multiple parameter combinations for the Time-of-Day "Session Reversal Fade"
strategy and saves a summary CSV of results per (symbol, params).

Inputs
------
- Either --symbols (fetches Coinbase OHLCV) with --since/--until
- Or --data_dir with CSVs (timestamp, open, high, low, close, volume)

Output
------
- sweep_results.csv with columns:
  symbol, timeframe, min_move_atr, confirm, tp_mult, sl_mult, atr_len,
  stoch_len, k, d, trades, winrate, avg_R, median_R, TP_rate, SL_rate, TIME_rate

Usage (Windows)
---------------
py session_reversal_sweep.py --symbols BTC-USD ETH-USD --since "2024-10-01" --until "2025-08-25" --timeframe 1h
"""

from __future__ import annotations
import argparse, os, itertools, time
from datetime import datetime, timezone
from dateutil import parser as dateparser
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

try:
    import ccxt
except ImportError:
    ccxt = None

# --- import helpers from the main strategy (we'll embed minimal duplicates to avoid import issues)

def to_dt(series: pd.Series) -> pd.Series:
    # Robustly handle tz-aware datetimes, naive datetimes, epoch ms/s, and strings
    try:
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.to_datetime(series, utc=True)
    except Exception:
        pass
    if np.issubdtype(series.dtype, np.number):
        s = series.copy()
        if s.dropna().median() < 1e12:
            return pd.to_datetime(s, unit='s', utc=True)
        return pd.to_datetime(s, unit='ms', utc=True)
    return pd.to_datetime(series, utc=True)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().fillna(method='bfill')

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def stoch_rsi(series: pd.Series, period: int = 14, k: int = 3, d: int = 3):
    r = rsi(series, period)
    rmin = r.rolling(period).min()
    rmax = r.rolling(period).max()
    stoch = (r - rmin) / (rmax - rmin)
    k_line = stoch.rolling(k).mean() * 100.0
    d_line = k_line.rolling(d).mean()
    return k_line.fillna(50), d_line.fillna(50)

def label_session(ts: pd.Timestamp) -> str:
    h = ts.hour
    if 0 <= h < 8: return 'ASIA'
    if 8 <= h < 13: return 'LONDON'
    if 13 <= h < 21: return 'NY'
    return 'OFF'

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for want in ['timestamp','open','high','low','close','volume']:
        for c in df.columns:
            if c.lower() == want:
                rename[c] = want
                break
    df = df.rename(columns=rename)
    df['timestamp'] = to_dt(df['timestamp'])
    df = df.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['open','high','low','close'])

def compute_sessions(df: pd.DataFrame, atr_len: int, stoch_len: int, k: int, d: int) -> pd.DataFrame:
    out = df.copy()
    out['session'] = [label_session(ts) for ts in out.index]
    out['ATR'] = atr(out, atr_len)
    out['K'], out['D'] = stoch_rsi(out['close'], stoch_len, k, d)
    return out

def backtest_symbol(df: pd.DataFrame, timeframe: str, min_move_atr: float, confirm: bool,
                    atr_len: int, stoch_len: int, k: int, d: int,
                    tp_mult: float, sl_mult: float, max_hold: int) -> pd.DataFrame:
    sig = compute_sessions(df, atr_len, stoch_len, k, d)
    idx = sig.index
    sessions = sig['session'].values
    trades = []
    highs = sig['high'].values; lows = sig['low'].values; closes = sig['close'].values

    for i in range(1, len(sig)):
        if sessions[i] != sessions[i-1]:
            new_sess = sessions[i]; prev_sess = sessions[i-1]
            if new_sess == 'OFF' or prev_sess == 'OFF': continue
            j = i-1
            while j > 0 and sessions[j-1] == prev_sess:
                j -= 1
            prior_open = sig['open'].iloc[j]
            prior_close = sig['close'].iloc[i-1]
            prior_ret = (prior_close - prior_open) / prior_open
            atr_here = sig['ATR'].iloc[i-1]
            price_here = sig['close'].iloc[i-1]
            if atr_here <= 0 or np.isnan(atr_here): 
                continue
            prior_ret_atr_mult = abs(prior_ret) / (atr_here / price_here)

            k_now = sig['K'].iloc[i]; d_now = sig['D'].iloc[i]
            long_conf = (k_now > d_now and k_now < 40)
            short_conf = (k_now < d_now and k_now > 60)
            conf_ok = (long_conf or short_conf) if confirm else True

            if prior_ret_atr_mult >= min_move_atr and conf_ok:
                side = 'short' if prior_ret > 0 else 'long'
                entry_idx = i
                entry = closes[entry_idx]
                a = sig['ATR'].iloc[entry_idx]
                if np.isnan(a) or a <= 0: continue
                sl = entry - sl_mult*a if side=='long' else entry + sl_mult*a
                tp = entry + tp_mult*a if side=='long' else entry - tp_mult*a

                exit_idx=None; exit_price=closes[entry_idx]; result='TIME'
                for j2 in range(entry_idx+1, min(entry_idx+1+max_hold, len(sig))):
                    hi = highs[j2]; lo = lows[j2]; cl = closes[j2]
                    if side=='long':
                        if lo <= sl: exit_idx=j2; exit_price=sl; result='SL'; break
                        if hi >= tp: exit_idx=j2; exit_price=tp; result='TP'; break
                    else:
                        if hi >= sl: exit_idx=j2; exit_price=sl; result='SL'; break
                        if lo <= tp: exit_idx=j2; exit_price=tp; result='TP'; break
                    exit_price = cl
                if exit_idx is None:
                    exit_idx = min(entry_idx + max_hold, len(sig)-1)
                    result = 'TIME'

                risk = abs(entry - sl)
                pnl = (exit_price - entry) if side=='long' else (entry - exit_price)
                r_mult = 0.0 if risk<=0 else pnl/risk

                trades.append({
                    'entry_time': idx[entry_idx], 'result': result, 'r_multiple': r_mult
                })
    return pd.DataFrame(trades)

def fetch_coinbase(symbol: str, timeframe: str, since: str, until: str) -> pd.DataFrame:
    if ccxt is None:
        raise RuntimeError("ccxt not installed. pip install ccxt")
    ex = ccxt.coinbase({'enableRateLimit': True})
    def pdate(s):
        dt = dateparser.parse(s)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        else: dt = dt.astimezone(timezone.utc)
        return int(dt.timestamp()*1000)
    since_ms = pdate(since); until_ms = pdate(until)
    limit = 300
    out=[]; cursor=since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch: break
        out.extend(batch)
        last_ts = batch[-1][0]; cursor = last_ts + 1
        if until_ms and last_ts >= until_ms: break
        time.sleep(ex.rateLimit/1000.0)
        if len(batch) < 2: break
    if not out: return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])
    df = pd.DataFrame(out, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", help="Coinbase symbols like BTC-USD ETH-USD")
    ap.add_argument("--since", type=str, help="Start date for fetch")
    ap.add_argument("--until", type=str, help="End date for fetch")
    ap.add_argument("--data_dir", type=str, help="CSV folder (if not fetching)")
    ap.add_argument("--timeframe", type=str, default="1h", choices=["15m","1h"])
    ap.add_argument("--out", type=str, default="sweep_results.csv")
    # Sweep ranges
    ap.add_argument("--min_move_atr_vals", nargs="+", type=float, default=[1.0,1.2,1.5])
    ap.add_argument("--confirm_vals", nargs="+", type=int, default=[0,1])
    ap.add_argument("--tp_vals", nargs="+", type=float, default=[1.0,1.5,2.0])
    ap.add_argument("--sl_vals", nargs="+", type=float, default=[0.8,1.0,1.2])
    ap.add_argument("--atr_vals", nargs="+", type=int, default=[10,14,20])
    ap.add_argument("--stoch_vals", nargs="+", type=int, default=[14])
    ap.add_argument("--k_vals", nargs="+", type=int, default=[3])
    ap.add_argument("--d_vals", nargs="+", type=int, default=[3])
    args = ap.parse_args()

    # Load series
    series: Dict[str, pd.DataFrame] = {}
    if args.data_dir:
        for fn in os.listdir(args.data_dir):
            if not fn.lower().endswith(".csv"): continue
            sym = os.path.splitext(fn)[0].lower()
            df = pd.read_csv(os.path.join(args.data_dir, fn))
            # normalize
            cols = {c.lower(): c for c in df.columns}
            rename = {}
            for want in ['timestamp','open','high','low','close','volume']:
                for c in df.columns:
                    if c.lower() == want:
                        rename[c] = want
                        break
            df = df.rename(columns=rename)
            df['timestamp'] = to_dt(df['timestamp'])
            df = df.sort_values('timestamp').drop_duplicates('timestamp')
            df = df.set_index('timestamp')
            for c in ['open','high','low','close','volume']:
                df[c]=pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['open','high','low','close'])
            series[sym] = df
    elif args.symbols:
        if not args.since or not args.until:
            raise ValueError("Provide --since and --until when fetching data.")
        for sym in args.symbols:
            df = fetch_coinbase(sym, args.timeframe, args.since, args.until)
            if df.empty:
                print(f"[WARN] No data for {sym}")
                continue
            df = df.sort_values('timestamp').set_index('timestamp')
            series[sym.lower().replace('-','')] = df
    else:
        raise ValueError("Provide either --data_dir or --symbols")

    rows = []
    for sym, df in series.items():
        for (min_move_atr, confirm, tp, sl, atr_len, stoch_len, k, d) in itertools.product(
            args.min_move_atr_vals, args.confirm_vals, args.tp_vals, args.sl_vals,
            args.atr_vals, args.stoch_vals, args.k_vals, args.d_vals
        ):
            tr = backtest_symbol(df.copy(), args.timeframe, min_move_atr, bool(confirm),
                                 atr_len, stoch_len, k, d, tp, sl, max_hold=16)
            if tr.empty:
                rows.append([sym, args.timeframe, min_move_atr, confirm, tp, sl, atr_len, stoch_len, k, d,
                             0, 0, 0, 0, 0, 0, 0])
                continue
            trades = tr
            stats = {
                "trades": len(trades),
                "winrate": (trades['r_multiple']>0).mean(),
                "avg_R": trades['r_multiple'].mean(),
                "median_R": trades['r_multiple'].median(),
                "TP_rate": (trades['result']=='TP').mean(),
                "SL_rate": (trades['result']=='SL').mean(),
                "TIME_rate": (trades['result']=='TIME').mean(),
            }
            rows.append([sym, args.timeframe, min_move_atr, confirm, tp, sl, atr_len, stoch_len, k, d,
                        stats['trades'], stats['winrate'], stats['avg_R'], stats['median_R'], stats['TP_rate'], stats['SL_rate'], stats['TIME_rate']])

    out_df = pd.DataFrame(rows, columns=[
        "symbol","timeframe","min_move_atr","confirm","tp_mult","sl_mult","atr_len","stoch_len","k","d",
        "trades","winrate","avg_R","median_R","TP_rate","SL_rate","TIME_rate"
    ])
    out_df = out_df.sort_values(["symbol","avg_R","winrate"], ascending=[True, False, False])
    out_df.to_csv(args.out, index=False)
    print(f"[OK] Saved sweep results to {args.out}")
    print(out_df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
