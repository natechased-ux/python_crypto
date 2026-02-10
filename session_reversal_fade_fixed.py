
"""
Time-of-Day Flows — Session Reversal Fade (Spot-Only, Coinbase OHLCV)

Edge
----
Crypto often trends during one session (Asia/London) and mean-reverts at the next session's open.
This strategy fades extended moves at session transitions using only spot data.

Sessions (UTC, configurable)
----------------------------
- Asia:   00:00–08:00
- London: 08:00–13:00
- NewYork:13:00–21:00
- Off:    21:00–24:00 (ignored)

Signal Logic (at session open bar)
----------------------------------
1) Compute prior session return and ATR.
2) If |prior session return| >= `min_session_move` (e.g., 1.2× ATR) → market considered "extended".
3) Fade the move at the *next* session open:
   - If prior session up big → enter SHORT at new session open.
   - If prior session down big → enter LONG at new session open.
4) Optional confirmation: Stoch RSI turn on the entry bar.
5) Risk: ATR-based SL/TP (default TP=1.5×ATR, SL=1.0×ATR).
6) Max hold (in bars) to avoid drift.

Data
----
- Works directly from CSVs (timestamp, open, high, low, close, volume)
- OR fetches via ccxt.coinbase if you pass --symbols.

Usage Examples
--------------
# From CSV directory (one file per symbol):
python session_reversal_fade.py --data_dir ./data --timeframe 1h

# Fetch spot data from Coinbase then backtest:
python session_reversal_fade.py --symbols BTC-USD ETH-USD --since "2024-10-01" --until "2025-08-25" --timeframe 1h

Key Params
----------
--min_move_atr 1.2     # prior session absolute return in ATR multiples
--atr 14               # ATR length
--tp 1.5 --sl 1.0      # ATR multiples
--stoch 14 --k 3 --d 3 # confirmation
--confirm 1            # require Stoch RSI confirmation (1=yes, 0=no)
--max_hold 16          # e.g., for 1h bars ≈ two sessions
--preview 10
"""

from __future__ import annotations
import argparse, math, os, time
from datetime import datetime, timezone
from dateutil import parser as dateparser
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

try:
    import ccxt
except ImportError:
    ccxt = None


def to_dt(series: pd.Series) -> pd.Series:
    # Robustly handle tz-aware datetimes, naive datetimes, epoch ms/s, and strings
    try:
        # If already datetime (incl. datetime64[ns, UTC]), just ensure UTC
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

def stoch_rsi(series: pd.Series, period: int = 14, k: int = 3, d: int = 3) -> Tuple[pd.Series, pd.Series]:
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

@dataclass
class Config:
    timeframe: str = '1h'
    atr_len: int = 14
    stoch_len: int = 14
    stoch_k: int = 3
    stoch_d: int = 3
    tp_mult: float = 1.5
    sl_mult: float = 1.0
    min_move_atr: float = 1.2
    require_confirmation: bool = True
    max_hold_bars: int = 16

@dataclass
class Trade:
    symbol: str
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    tp: float
    sl: float
    exit_time: pd.Timestamp
    exit_price: float
    result: str
    r_multiple: float
    session_from: str
    session_to: str
    prior_ret: float
    prior_atr: float
    k: float
    d: float

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize columns
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

def compute_sessions(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out['session'] = [label_session(ts) for ts in out.index]
    out['ATR'] = atr(out, cfg.atr_len)
    out['K'], out['D'] = stoch_rsi(out['close'], cfg.stoch_len, cfg.stoch_k, cfg.stoch_d)
    return out

def session_ranges(sig: pd.DataFrame) -> pd.DataFrame:
    # Aggregate by session blocks (continuous runs of same session label)
    blocks = []
    current = None
    for ts, row in sig.iterrows():
        s = row['session']
        if current is None or current['session'] != s:
            # start new
            current = {'start': ts, 'end': ts, 'session': s, 'open': row['open'], 'close': row['close'], 'high': row['high'], 'low': row['low']}
            blocks.append(current)
        else:
            current['end'] = ts
            current['close'] = row['close']
            current['high'] = max(current['high'], row['high'])
            current['low']  = min(current['low'],  row['low'])
    return pd.DataFrame(blocks)

def backtest_symbol(symbol: str, df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sig = compute_sessions(df, cfg)
    idx = sig.index
    trades: List[Trade] = []

    # Build list of session change indices
    sessions = sig['session'].values
    for i in range(1, len(sig)):
        if sessions[i] != sessions[i-1]:
            # Transition at index i (this bar is the first of new session)
            new_sess = sessions[i]
            prev_sess = sessions[i-1]
            if new_sess == 'OFF' or prev_sess == 'OFF':
                continue
            # Measure prior session move from its first close to final close
            # Find the start index of prior session
            j = i-1
            while j > 0 and sessions[j-1] == prev_sess:
                j -= 1
            prior_open = sig['open'].iloc[j]
            prior_close = sig['close'].iloc[i-1]
            prior_ret = (prior_close - prior_open) / prior_open
            atr_here = sig['ATR'].iloc[i-1]
            price_here = sig['close'].iloc[i-1]
            # Convert threshold to absolute price move for comparison
            # We use ATR multiple relative to price percent via atr/price
            if atr_here <= 0 or np.isnan(atr_here):
                continue
            prior_ret_atr_mult = abs(prior_ret) / (atr_here / price_here)

            # Confirmation (Stoch RSI turn on entry bar i)
            k_now = sig['K'].iloc[i]; d_now = sig['D'].iloc[i]
            long_conf = (k_now > d_now and k_now < 40)
            short_conf = (k_now < d_now and k_now > 60)
            conf_ok = (long_conf or short_conf) if cfg.require_confirmation else True

            if prior_ret_atr_mult >= cfg.min_move_atr and conf_ok:
                side = 'short' if prior_ret > 0 else 'long'
                entry_idx = i
                entry_time = idx[entry_idx]
                entry = sig['close'].iloc[entry_idx]
                a = sig['ATR'].iloc[entry_idx]
                if np.isnan(a) or a <= 0: 
                    continue
                if side == 'long':
                    sl = entry - cfg.sl_mult * a
                    tp = entry + cfg.tp_mult * a
                else:
                    sl = entry + cfg.sl_mult * a
                    tp = entry - cfg.tp_mult * a

                # Walk forward to exit
                highs = sig['high'].values; lows = sig['low'].values; closes = sig['close'].values
                exit_idx = None; exit_price = closes[entry_idx]; result = 'TIME'
                for j2 in range(entry_idx+1, min(entry_idx+1+cfg.max_hold_bars, len(sig))):
                    hi = highs[j2]; lo = lows[j2]; cl = closes[j2]
                    if side == 'long':
                        if lo <= sl:
                            exit_idx = j2; exit_price = sl; result = 'SL'; break
                        if hi >= tp:
                            exit_idx = j2; exit_price = tp; result = 'TP'; break
                    else:
                        if hi >= sl:
                            exit_idx = j2; exit_price = sl; result = 'SL'; break
                        if lo <= tp:
                            exit_idx = j2; exit_price = tp; result = 'TP'; break
                    exit_price = cl
                if exit_idx is None:
                    exit_idx = min(entry_idx + cfg.max_hold_bars, len(sig)-1)
                    result = 'TIME'

                risk = abs(entry - sl)
                pnl = (exit_price - entry) if side == 'long' else (entry - exit_price)
                r_mult = 0.0 if risk <= 0 else pnl / risk

                trades.append(Trade(
                    symbol=symbol, side=side,
                    entry_time=entry_time, entry_price=float(entry),
                    tp=float(tp), sl=float(sl),
                    exit_time=idx[exit_idx], exit_price=float(exit_price),
                    result=result, r_multiple=float(r_mult),
                    session_from=prev_sess, session_to=new_sess,
                    prior_ret=float(prior_ret), prior_atr=float(atr_here),
                    k=float(k_now), d=float(d_now)
                ))
    if not trades:
        return pd.DataFrame()
    df_tr = pd.DataFrame([t.__dict__ for t in trades]).sort_values('entry_time')
    return df_tr

def summarize(trades: pd.DataFrame) -> Dict[str, float]:
    if trades is None or len(trades) == 0: return {}
    wins = (trades['r_multiple'] > 0).sum()
    total = len(trades)
    return {
        "trades": int(total),
        "winrate": float(wins/total),
        "avg_R": float(trades['r_multiple'].mean()),
        "median_R": float(trades['r_multiple'].median()),
        "TP_rate": float((trades['result']=='TP').mean()),
        "SL_rate": float((trades['result']=='SL').mean()),
        "TIME_rate": float((trades['result']=='TIME').mean()),
    }

def fetch_coinbase(symbol: str, timeframe: str, since: str, until: str) -> pd.DataFrame:
    if ccxt is None:
        raise RuntimeError("ccxt not installed. pip install ccxt")
    ex = ccxt.coinbase({'enableRateLimit': True})
    # Parse dates
    def pdate(s): 
        dt = dateparser.parse(s)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        else: dt = dt.astimezone(timezone.utc)
        return int(dt.timestamp()*1000)
    since_ms = pdate(since); until_ms = pdate(until)
    limit = 300
    rows = []; cursor = since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch: break
        rows.extend(batch)
        last_ts = batch[-1][0]; cursor = last_ts + 1
        if until_ms and last_ts >= until_ms: break
        time.sleep(ex.rateLimit/1000.0)
        if len(batch) < 2: break
    if not rows: return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])
    df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, help="Folder with CSVs per symbol")
    ap.add_argument("--symbols", nargs="+", help="Fetch from Coinbase, e.g., BTC-USD ETH-USD")
    ap.add_argument("--since", type=str, help="Start date (for fetch)")
    ap.add_argument("--until", type=str, help="End date (for fetch)")
    ap.add_argument("--timeframe", type=str, default="1h", choices=["15m","1h"])
    ap.add_argument("--atr", dest="atr_len", type=int, default=14)
    ap.add_argument("--stoch", dest="stoch_len", type=int, default=14)
    ap.add_argument("--k", dest="stoch_k", type=int, default=3)
    ap.add_argument("--d", dest="stoch_d", type=int, default=3)
    ap.add_argument("--tp", dest="tp_mult", type=float, default=1.5)
    ap.add_argument("--sl", dest="sl_mult", type=float, default=1.0)
    ap.add_argument("--min_move_atr", type=float, default=1.2)
    ap.add_argument("--confirm", type=int, default=1)
    ap.add_argument("--max_hold", type=int, default=16)
    ap.add_argument("--out", type=str, default="./session_trades.csv")
    ap.add_argument("--preview", type=int, default=0)
    args = ap.parse_args()

    cfg = Config(
        timeframe=args.timeframe,
        atr_len=args.atr_len, stoch_len=args.stoch_len, stoch_k=args.stoch_k, stoch_d=args.stoch_d,
        tp_mult=args.tp_mult, sl_mult=args.sl_mult,
        min_move_atr=args.min_move_atr, require_confirmation=bool(args.confirm),
        max_hold_bars=args.max_hold
    )

    series: Dict[str, pd.DataFrame] = {}
    if args.data_dir:
        for fn in os.listdir(args.data_dir):
            if not fn.lower().endswith(".csv"): continue
            sym = os.path.splitext(fn)[0].lower()
            df = pd.read_csv(os.path.join(args.data_dir, fn))
            series[sym] = preprocess(df)
    elif args.symbols:
        if args.since is None or args.until is None:
            raise ValueError("Provide --since and --until when fetching from Coinbase.")
        for sym in args.symbols:
            df = fetch_coinbase(sym, cfg.timeframe, args.since, args.until)
            if df.empty:
                print(f"[WARN] No data for {sym}")
                continue
            series[sym.lower().replace('-','')] = preprocess(df)
    else:
        raise ValueError("Provide either --data_dir or --symbols")

    all_trades = []
    for sym, df in series.items():
        tr = backtest_symbol(sym, df, cfg)
        if not tr.empty:
            all_trades.append(tr)

    if not all_trades:
        print("No trades generated. Try lowering --min_move_atr or disabling confirmation (--confirm 0).")
        return

    trades = pd.concat(all_trades).sort_values('entry_time')
    trades.to_csv(args.out, index=False)
    print(f"[OK] Saved trades to {args.out}")
    # Print summary per symbol and overall
    overall = trades.copy()
    print("=== Overall Summary ===")
    print(pd.Series({
        "trades": len(overall),
        "winrate": (overall['r_multiple']>0).mean(),
        "avg_R": overall['r_multiple'].mean(),
        "median_R": overall['r_multiple'].median(),
        "TP_rate": (overall['result']=='TP').mean(),
        "SL_rate": (overall['result']=='SL').mean(),
        "TIME_rate": (overall['result']=='TIME').mean(),
    }))
    if args.preview > 0:
        print(trades.head(args.preview).to_string(index=False))

if __name__ == "__main__":
    main()
