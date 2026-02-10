
"""
Funding Rate Whiplash — Research Backtester (1H candles)

Overview
--------
Fade extreme funding imbalances with momentum-confirmation and ATR-based risk.
- Enter SHORT when funding is extremely positive and momentum turns down.
- Enter LONG when funding is extremely negative and momentum turns up.

Data Requirements
-----------------
Provide one CSV per symbol with at least these columns:
    timestamp, open, high, low, close, volume, funding_rate

Notes:
- timestamp should be in ISO 8601 or epoch ms; we'll parse automatically.
- funding_rate should be decimal (e.g., 0.001 = 0.1% per funding interval).
- If your funding prints every 8 hours, forward-fill to 1H by pre-processing,
  or leave it sparse; this script forward-fills within the backtest anyway.

Usage
-----
python funding_whiplash_backtest.py --data_dir ./data --out ./whiplash_trades.csv

Optional flags:
  --z 2.0                   # funding z-score threshold
  --abs 0.0008              # absolute funding threshold (e.g., 0.08% = 0.0008)
  --z_lookback 720          # rolling bars for z-score (720 1H bars ≈ 30 days)
  --atr 14                  # ATR length
  --stoch 14 --k 3 --d 3    # Stochastic RSI settings
  --tp 2.0 --sl 1.5         # ATR multiples
  --max_hold 48             # max holding bars (2 days on 1H)
  --cooldown 24             # bars to wait after any exit
  --min_bb_width 0.0        # optional min BB width filter (0 to disable)
  --preview 0               # print first N trades as preview

You can also import this module and call run_backtest() programmatically.
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# -------------------- Indicators --------------------

def _to_datetime(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        # epoch ms or s — try ms first, fall back to s
        s = series.copy()
        # heuristic: if values are too small to be ms, assume seconds
        if s.dropna().median() < 1e12:
            s = pd.to_datetime(s, unit='s', utc=True)
        else:
            s = pd.to_datetime(s, unit='ms', utc=True)
        return s
    else:
        return pd.to_datetime(series, utc=True)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

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

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().fillna(method='bfill')

def bollinger_band_width(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std(ddof=0)
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    width = (upper - lower) / ma
    return width.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def rolling_zscore(x: pd.Series, lookback: int) -> pd.Series:
    mean = x.rolling(lookback).mean()
    std = x.rolling(lookback).std(ddof=0)
    z = (x - mean) / std.replace(0, np.nan)
    return z.fillna(0.0)

# -------------------- Strategy --------------------

@dataclass
class Config:
    funding_z: float = 2.0
    funding_abs: float = 0.0008  # 0.08% (per interval) as decimal
    z_lookback: int = 720  # 30 days of 1H bars
    atr_len: int = 14
    stoch_len: int = 14
    stoch_k: int = 3
    stoch_d: int = 3
    tp_mult: float = 2.0
    sl_mult: float = 1.5
    max_hold_bars: int = 48
    cooldown_bars: int = 24
    min_bb_width: float = 0.0  # require some vol regime if >0

@dataclass
class Trade:
    symbol: str
    side: str  # 'long' or 'short'
    entry_time: pd.Timestamp
    entry_price: float
    tp: float
    sl: float
    exit_time: pd.Timestamp
    exit_price: float
    result: str  # 'TP' | 'SL' | 'TIME' 
    r_multiple: float
    hold_bars: int
    funding_rate: float
    funding_z: float
    stoch_k: float
    stoch_d: float
    atr_at_entry: float

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for want in ['timestamp','open','high','low','close','volume','funding_rate']:
        for c in df.columns:
            if c.lower() == want:
                rename[c] = want
                break
    df = df.rename(columns=rename)
    # Parse time
    if 'timestamp' in df.columns:
        df['timestamp'] = _to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').drop_duplicates('timestamp')
        df = df.set_index('timestamp')
    else:
        raise ValueError("CSV must contain 'timestamp' column")
    # Ensure numeric
    for col in ['open','high','low','close','volume','funding_rate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            raise ValueError(f"CSV missing required column: {col}")
    # Forward-fill funding_rate if sparse (e.g., 8H prints)
    df['funding_rate'] = df['funding_rate'].ffill()
    return df

def generate_signals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out['ATR'] = atr(out, cfg.atr_len)
    out['K'], out['D'] = stoch_rsi(out['close'], cfg.stoch_len, cfg.stoch_k, cfg.stoch_d)
    out['funding_z'] = rolling_zscore(out['funding_rate'], cfg.z_lookback)
    out['bb_width'] = bollinger_band_width(out['close'])
    # Funding extremes
    out['excess_long']  = (out['funding_z'] >= cfg.funding_z) | (out['funding_rate'] >= cfg.funding_abs)
    out['excess_short'] = (out['funding_z'] <= -cfg.funding_z) | (out['funding_rate'] <= -cfg.funding_abs)
    # Momentum confirmation (simple + robust)
    out['short_confirm'] = (out['K'] < out['D']) & (out['K'] > 60)
    out['long_confirm']  = (out['K'] > out['D']) & (out['K'] < 40)
    if cfg.min_bb_width > 0:
        out['short_confirm'] &= (out['bb_width'] >= cfg.min_bb_width)
        out['long_confirm']  &= (out['bb_width'] >= cfg.min_bb_width)
    # Entry signals at bar close
    out['short_signal'] = out['excess_long']  & out['short_confirm']
    out['long_signal']  = out['excess_short'] & out['long_confirm']
    return out

def backtest_symbol(symbol: str, df: pd.DataFrame, cfg: Config) -> List[Trade]:
    sig = generate_signals(df, cfg)
    trades: List[Trade] = []
    in_pos = False
    next_eligible_idx = 0  # cooldown control using integer bar index
    idx_list = list(range(len(sig)))
    closes = sig['close'].values
    highs = sig['high'].values
    lows  = sig['low'].values
    atrs  = sig['ATR'].values
    kline = sig['K'].values
    dline = sig['D'].values
    fund  = sig['funding_rate'].values
    fundz = sig['funding_z'].values
    times = sig.index.to_list()

    for i in idx_list[:-1]:  # enter on next bar open (approx using next close/open proxy)
        if i < next_eligible_idx or in_pos:
            continue
        row = sig.iloc[i]
        if row['short_signal'] or row['long_signal']:
            side = 'short' if row['short_signal'] else 'long'
            # Enter at next bar's open proxy = current close (conservative for 1H)
            entry_idx = i + 1
            if entry_idx >= len(sig):
                break
            entry_time = times[entry_idx]
            entry_price = closes[entry_idx]
            a = atrs[entry_idx]
            if math.isnan(a) or a <= 0:
                continue
            if side == 'long':
                sl = entry_price - cfg.sl_mult * a
                tp = entry_price + cfg.tp_mult * a
            else:
                sl = entry_price + cfg.sl_mult * a
                tp = entry_price - cfg.tp_mult * a

            # Walk forward to exit
            exit_idx = None
            result = 'TIME'
            exit_price = closes[entry_idx]
            for j in range(entry_idx + 1, min(entry_idx + 1 + cfg.max_hold_bars, len(sig))):
                hi = highs[j]; lo = lows[j]; cl = closes[j]
                if side == 'long':
                    # TP/SL intrabar checks
                    if lo <= sl:
                        exit_idx = j
                        exit_price = sl
                        result = 'SL'
                        break
                    if hi >= tp:
                        exit_idx = j
                        exit_price = tp
                        result = 'TP'
                        break
                else:
                    if hi >= sl:
                        exit_idx = j
                        exit_price = sl
                        result = 'SL'
                        break
                    if lo <= tp:
                        exit_idx = j
                        exit_price = tp
                        result = 'TP'
                        break
                exit_price = cl  # trailing for TIME exit if no TP/SL
            if exit_idx is None:
                exit_idx = min(entry_idx + cfg.max_hold_bars, len(sig)-1)
                result = 'TIME'

            # Compute R-multiple
            risk = abs(entry_price - sl)
            reward = abs(tp - entry_price)
            if risk <= 0:
                r_mult = 0.0
            else:
                pnl = (exit_price - entry_price) if side == 'long' else (entry_price - exit_price)
                r_mult = pnl / risk

            t = Trade(
                symbol=symbol,
                side=side,
                entry_time=entry_time,
                entry_price=float(entry_price),
                tp=float(tp),
                sl=float(sl),
                exit_time=times[exit_idx],
                exit_price=float(exit_price),
                result=result,
                r_multiple=float(r_mult),
                hold_bars=int(exit_idx - entry_idx),
                funding_rate=float(fund[entry_idx]),
                funding_z=float(fundz[entry_idx]),
                stoch_k=float(kline[entry_idx]),
                stoch_d=float(dline[entry_idx]),
                atr_at_entry=float(a),
            )
            trades.append(t)

            in_pos = False
            next_eligible_idx = exit_idx + cfg.cooldown_bars

    return trades

def run_backtest(files: Dict[str, str], cfg: Config) -> pd.DataFrame:
    all_trades: List[Trade] = []
    for symbol, path in files.items():
        df = pd.read_csv(path)
        df = preprocess(df)
        ts = backtest_symbol(symbol, df, cfg)
        all_trades.extend(ts)
    # Assemble to DataFrame
    rows = [t.__dict__ for t in all_trades]
    trades_df = pd.DataFrame(rows)
    if len(trades_df) == 0:
        return trades_df
    # Summary stats
    return trades_df.sort_values('entry_time')

def summarize(trades: pd.DataFrame) -> Dict[str, float]:
    if trades is None or len(trades) == 0:
        return {}
    wins = (trades['r_multiple'] > 0).sum()
    total = len(trades)
    winrate = wins / total if total else 0.0
    avg_r = trades['r_multiple'].mean()
    med_r = trades['r_multiple'].median()
    exp_r = avg_r  # per-trade expectancy in R
    return {
        "trades": int(total),
        "winrate": float(winrate),
        "avg_R": float(avg_r),
        "median_R": float(med_r),
        "expectancy_R": float(exp_r),
        "TP_rate": float((trades['result'] == 'TP').mean() if total else 0.0),
        "SL_rate": float((trades['result'] == 'SL').mean() if total else 0.0),
        "TIME_rate": float((trades['result'] == 'TIME').mean() if total else 0.0),
    }

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Folder with CSV files per symbol")
    ap.add_argument("--out", type=str, default="./whiplash_trades.csv", help="Output CSV for trade logs")
    ap.add_argument("--z", type=float, default=2.0, help="Funding z-score threshold")
    ap.add_argument("--abs", dest="abs_th", type=float, default=0.0008, help="Absolute funding threshold (decimal)")
    ap.add_argument("--z_lookback", type=int, default=720, help="Z-score lookback in bars")
    ap.add_argument("--atr", dest="atr_len", type=int, default=14)
    ap.add_argument("--stoch", dest="stoch_len", type=int, default=14)
    ap.add_argument("--k", dest="stoch_k", type=int, default=3)
    ap.add_argument("--d", dest="stoch_d", type=int, default=3)
    ap.add_argument("--tp", dest="tp_mult", type=float, default=2.0)
    ap.add_argument("--sl", dest="sl_mult", type=float, default=1.5)
    ap.add_argument("--max_hold", type=int, default=48)
    ap.add_argument("--cooldown", type=int, default=24)
    ap.add_argument("--min_bb_width", type=float, default=0.0)
    ap.add_argument("--preview", type=int, default=0)
    args = ap.parse_args()

    cfg = Config(
        funding_z=args.z,
        funding_abs=args.abs_th,
        z_lookback=args.z_lookback,
        atr_len=args.atr_len,
        stoch_len=args.stoch_len,
        stoch_k=args.stoch_k,
        stoch_d=args.stoch_d,
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
        max_hold_bars=args.max_hold,
        cooldown_bars=args.cooldown,
        min_bb_width=args.min_bb_width,
    )

    # Build symbol -> path mapping (all CSVs in data_dir)
    import os, glob
    files = {}
    for fp in glob.glob(os.path.join(args.data_dir, "*.csv")):
        sym = os.path.splitext(os.path.basename(fp))[0].lower()
        files[sym] = fp

    trades = run_backtest(files, cfg)
    if len(trades) == 0:
        print("No trades generated. Consider loosening thresholds or verifying data columns.")
        return

    # Save logs
    trades.to_csv(args.out, index=False)
    print(f"Saved trades: {args.out}")
    # Summary
    summary = summarize(trades)
    print("Summary:", summary)
    if args.preview > 0:
        print(trades.head(args.preview).to_string(index=False))

if __name__ == "__main__":
    main()
