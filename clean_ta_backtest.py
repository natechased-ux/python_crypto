#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean TA Backtest (DonAlt/CryptoCred-style)
-------------------------------------------
- Daily trend filter: 200 EMA on HTF (default: 1D)
- Entry TF (default: 1H): Reclaim of resistance / loss of support with retest
- Momentum confirm (strict): Stoch RSI (K>D & K<40 for longs; K<D & K>60 for shorts)
- Optional ADX filter (default: ADX(14) > 20)
- Risk management: ATR-based SL/TP (default: SL = 1.5x ATR, TP = 2.0x ATR)
- One trade per coin per day option
- CSV trade log and summary metrics

Data expectations:
- OHLCV with columns: ["timestamp","open","high","low","close","volume"]
- timestamp in UTC (ISO 8601 or epoch seconds). The loader will parse automatically.

Usage examples:
1) Backtest CSVs in a folder:
   python clean_ta_backtest.py --data_dir ./data --symbols "BTC-USD,ETH-USD" --entry_tf 1H --htf 1D

2) Backtest a single CSV:
   python clean_ta_backtest.py --csv ./data/BTC-USD_1h.csv --symbol BTC-USD --entry_tf 1H --htf 1D

3) Customize parameters:
   python clean_ta_backtest.py --data_dir ./data --symbols "BTC-USD" --sr_lookback 120 --retest_bars 5 --atr_period 14 --tp_mult 2.0 --sl_mult 1.5 --adx_min 20 --one_trade_per_day

Author: ChatGPT
"""
import argparse
import os
import sys
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# -----------------------------
# Utility: Indicators (no TA-Lib)
# -----------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilders smoothing (RMA), commonly used for ATR and RSI components."""
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, use_rma: bool = True) -> pd.Series:
    tr = true_range(high, low, close)
    if use_rma:
        return rma(tr, period)
    else:
        return tr.rolling(period).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    # Compute directional movement
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = true_range(high, low, close)
    tr_rma = rma(tr, period)
    plus_di = 100 * rma(plus_dm, period) / tr_rma
    minus_di = 100 * rma(minus_dm, period) / tr_rma
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = rma(dx, period)
    return adx_val

def stoch_rsi(close: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0))
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Stochastic RSI
    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()
    stoch = 100 * (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d

# -----------------------------
# Strategy Components
# -----------------------------

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV to a new timeframe using pandas rule (e.g., '1H','1D').
    Assumes df has columns: timestamp, open, high, low, close, volume and timestamp is datetime index.
    """
    o = df['open'].resample(rule).first()
    h = df['high'].resample(rule).max()
    l = df['low'].resample(rule).min()
    c = df['close'].resample(rule).last()
    v = df['volume'].resample(rule).sum()
    out = pd.concat([o,h,l,c,v], axis=1)
    out.columns = ['open','high','low','close','volume']
    out.dropna(inplace=True)
    return out

def find_swing_levels(df: pd.DataFrame, lookback: int = 120, swing_window: int = 5) -> pd.DataFrame:
    """
    Identify recent swing highs/lows on the entry timeframe.
    - lookback: number of bars to scan for the most recent swing high/low
    - swing_window: pivots defined as high greater than its neighbors within +/- N bars
    Returns new columns: swing_high, swing_low, last_resistance, last_support
    """
    highs = df['high']
    lows  = df['low']

    # Pivot logic
    def is_swing_high(i):
        left = highs.iloc[max(0, i - swing_window):i]
        right = highs.iloc[i+1:i+1+swing_window]
        if len(left)==0 or len(right)==0: 
            return False
        return highs.iloc[i] == max([highs.iloc[i], left.max(), right.max()])

    def is_swing_low(i):
        left = lows.iloc[max(0, i - swing_window):i]
        right = lows.iloc[i+1:i+1+swing_window]
        if len(left)==0 or len(right)==0: 
            return False
        return lows.iloc[i] == min([lows.iloc[i], left.min(), right.min()])

    swing_high = pd.Series(False, index=df.index)
    swing_low  = pd.Series(False, index=df.index)
    for i in range(len(df)):
        if is_swing_high(i):
            swing_high.iloc[i] = True
        if is_swing_low(i):
            swing_low.iloc[i] = True

    df['swing_high'] = swing_high
    df['swing_low']  = swing_low

    # Track last resistance (recent swing high) and support (recent swing low)
    last_res = []
    last_sup = []
    recent_highs = []
    recent_lows = []
    for i in range(len(df)):
        idx = df.index[i]
        if swing_high.iloc[i]:
            recent_highs.append(df.loc[idx, 'high'])
        if swing_low.iloc[i]:
            recent_lows.append(df.loc[idx, 'low'])

        # Limit history to lookback recent pivots
        if len(recent_highs) > lookback:
            recent_highs = recent_highs[-lookback:]
        if len(recent_lows) > lookback:
            recent_lows = recent_lows[-lookback:]

        last_res.append(recent_highs[-1] if recent_highs else np.nan)
        last_sup.append(recent_lows[-1] if recent_lows else np.nan)

    df['last_resistance'] = last_res
    df['last_support'] = last_sup
    return df

@dataclass
class Params:
    entry_tf: str = "1H"        # Entry timeframe
    htf: str = "1D"             # Higher timeframe for trend
    ema_period_trend: int = 200 # Trend EMA on HTF
    stoch_len: int = 14
    stoch_k: int = 3
    stoch_d: int = 3
    adx_period: int = 14
    adx_min: float = 20.0
    atr_period: int = 14
    tp_mult: float = 2.0
    sl_mult: float = 1.5
    sr_lookback: int = 120
    swing_window: int = 5
    retest_bars: int = 4        # # of bars allowed for retest after reclaim/loss
    retest_tolerance: float = 0.001  # 0.1% tolerance around level
    one_trade_per_day: bool = False
    timezone: Optional[str] = None   # e.g., "America/Los_Angeles"

@dataclass
class Trade:
    symbol: str
    side: str                # "long" or "short"
    entry_time: pd.Timestamp
    entry_price: float
    sl: float
    tp: float
    exit_time: Optional[pd.Timestamp]
    exit_price: Optional[float]
    outcome: Optional[str]   # "TP","SL","timeout","NA"
    r_multiple: Optional[float]

def derive_trend_filter(htf_df: pd.DataFrame, ema_period: int) -> pd.Series:
    ema_htf = ema(htf_df['close'], ema_period)
    return (htf_df['close'] > ema_htf).astype(int)  # 1 = bullish, 0 = bearish

def align_htf_to_entry(htf_signal: pd.Series, entry_index: pd.DatetimeIndex) -> pd.Series:
    # Forward-fill the HTF trend to entry timeframe index
    htf_signal = htf_signal.reindex(entry_index, method='ffill')
    return htf_signal

def momentum_confirms(k: pd.Series, d: pd.Series, side: str) -> pd.Series:
    if side == "long":
        return ((k > d) & (k < 40)).astype(int)
    else:
        return ((k < d) & (k > 60)).astype(int)

def reclaim_retest_long(df: pd.DataFrame, retest_bars: int, tol: float) -> pd.Series:
    """
    True when:
    1) Close crosses above last_resistance (reclaim)
    2) Within next N bars, price retests near that level (low <= level*(1+tol)) and closes back above it
    We mark the bar of the successful retest close as the signal bar.
    """
    idx = df.index
    signal = pd.Series(0, index=idx)
    for i in range(1, len(df)-1):
        level = df.loc[idx[i], 'last_resistance']
        if math.isnan(level):
            continue
        # Reclaim event on bar i: close crossed from below to above level
        prev_close = df.loc[idx[i-1], 'close']
        close_i = df.loc[idx[i], 'close']
        if (prev_close <= level) and (close_i > level):
            # Search retest within next retest_bars
            for j in range(i+1, min(i+1+retest_bars, len(df))):
                low_j = df.loc[idx[j], 'low']
                close_j = df.loc[idx[j], 'close']
                if low_j <= level*(1+tol) and close_j > level:
                    signal.iloc[j] = 1
                    break
    return signal

def loss_retest_short(df: pd.DataFrame, retest_bars: int, tol: float) -> pd.Series:
    """
    Mirror logic for shorts against last_support.
    True when:
    1) Close crosses below last_support (loss)
    2) Within next N bars, retest near that level (high >= level*(1 - tol)) and closes back below it
    """
    idx = df.index
    signal = pd.Series(0, index=idx)
    for i in range(1, len(df)-1):
        level = df.loc[idx[i], 'last_support']
        if math.isnan(level):
            continue
        prev_close = df.loc[idx[i-1], 'close']
        close_i = df.loc[idx[i], 'close']
        if (prev_close >= level) and (close_i < level):
            for j in range(i+1, min(i+1+retest_bars, len(df))):
                high_j = df.loc[idx[j], 'high']
                close_j = df.loc[idx[j], 'close']
                if high_j >= level*(1 - tol) and close_j < level:
                    signal.iloc[j] = 1
                    break
    return signal

def intrabar_exit_price(row_high: float, row_low: float, entry: float, tp: float, sl: float, side: str) -> Tuple[str, float]:
    """
    Decide outcome within a bar given H/L range. Use conservative assumption:
    - For longs: if low <= SL before high >= TP, assume SL hits first if both are inside the bar.
    - For shorts: if high >= SL before low <= TP, assume SL hits first.
    Returns outcome ("TP" or "SL") and exit_price.
    """
    if side == "long":
        hit_sl = row_low <= sl
        hit_tp = row_high >= tp
        if hit_sl and hit_tp:
            # Assume SL first (conservative)
            return "SL", sl
        elif hit_sl:
            return "SL", sl
        elif hit_tp:
            return "TP", tp
        else:
            return "NA", entry
    else:
        hit_sl = row_high >= sl
        hit_tp = row_low <= tp
        if hit_sl and hit_tp:
            return "SL", sl
        elif hit_sl:
            return "SL", sl
        elif hit_tp:
            return "TP", tp
        else:
            return "NA", entry

def backtest_symbol(symbol: str, df_entry: pd.DataFrame, df_htf: pd.DataFrame, params: Params) -> Tuple[List[Trade], Dict]:
    # Indicators on entry TF
    k, d = stoch_rsi(df_entry['close'], period=params.stoch_len, smooth_k=params.stoch_k, smooth_d=params.stoch_d)
    adx_val = adx(df_entry['high'], df_entry['low'], df_entry['close'], period=params.adx_period)
    atr_val = atr(df_entry['high'], df_entry['low'], df_entry['close'], period=params.atr_period)

    df_entry = df_entry.copy()
    df_entry['k'] = k
    df_entry['d'] = d
    df_entry['adx'] = adx_val
    df_entry['atr'] = atr_val

    df_entry = find_swing_levels(df_entry, lookback=params.sr_lookback, swing_window=params.swing_window)
    long_rr = reclaim_retest_long(df_entry, params.retest_bars, params.retest_tolerance)
    short_rr = loss_retest_short(df_entry, params.retest_bars, params.retest_tolerance)

    # Trend filter on HTF
    htf_trend = derive_trend_filter(df_htf, params.ema_period_trend)  # 1 bull, 0 bear
    trend_on_entry = align_htf_to_entry(htf_trend, df_entry.index)
    df_entry['bull_trend'] = trend_on_entry  # 1 or 0

    trades: List[Trade] = []
    last_trade_day: Optional[pd.Timestamp] = None

    for i in range(1, len(df_entry)):
        ts = df_entry.index[i]

        if params.one_trade_per_day:
            if last_trade_day is not None and ts.date() == last_trade_day.date():
                continue

        row = df_entry.iloc[i]

        # LONG setup: bull trend + reclaim/retest + momentum + ADX
        long_setup = (row['bull_trend'] == 1) and (long_rr.iloc[i] == 1)
        long_momo  = momentum_confirms(df_entry['k'], df_entry['d'], "long").iloc[i] == 1
        long_adx   = (row['adx'] >= params.adx_min)

        if long_setup and long_momo and long_adx and not np.isnan(row['atr']):
            entry = row['close']  # next-bar close entry assumption implemented below
            sl = entry - params.sl_mult * row['atr']
            tp = entry + params.tp_mult * row['atr']
            # Execute from next bar
            if i+1 < len(df_entry):
                next_row = df_entry.iloc[i+1]
                outcome, exit_price = intrabar_exit_price(next_row['high'], next_row['low'], entry, tp, sl, "long")
                exit_time = df_entry.index[i+1]
                if outcome in ("TP","SL"):
                    r = (exit_price - entry) / (entry - sl)
                else:
                    r = np.nan
                trades.append(Trade(symbol, "long", ts, float(entry), float(sl), float(tp),
                                    exit_time, float(exit_price), outcome, float(r) if not np.isnan(r) else None))
                last_trade_day = ts
                continue

        # SHORT setup: bear trend + loss/retest + momentum + ADX
        short_setup = (row['bull_trend'] == 0) and (short_rr.iloc[i] == 1)
        short_momo  = momentum_confirms(df_entry['k'], df_entry['d'], "short").iloc[i] == 1
        short_adx   = (row['adx'] >= params.adx_min)

        if short_setup and short_momo and short_adx and not np.isnan(row['atr']):
            entry = row['close']
            sl = entry + params.sl_mult * row['atr']
            tp = entry - params.tp_mult * row['atr']
            if i+1 < len(df_entry):
                next_row = df_entry.iloc[i+1]
                outcome, exit_price = intrabar_exit_price(next_row['high'], next_row['low'], entry, tp, sl, "short")
                exit_time = df_entry.index[i+1]
                if outcome in ("TP","SL"):
                    r = (entry - exit_price) / (sl - entry)
                else:
                    r = np.nan
                trades.append(Trade(symbol, "short", ts, float(entry), float(sl), float(tp),
                                    exit_time, float(exit_price), outcome, float(r) if not np.isnan(r) else None))
                last_trade_day = ts
                continue

    # Summary
    df_trades = pd.DataFrame([asdict(t) for t in trades]) if trades else pd.DataFrame(columns=[
        "symbol","side","entry_time","entry_price","sl","tp","exit_time","exit_price","outcome","r_multiple"
    ])
    wins = (df_trades['outcome'] == 'TP').sum() if not df_trades.empty else 0
    losses = (df_trades['outcome'] == 'SL').sum() if not df_trades.empty else 0
    total = len(df_trades)
    win_rate = (wins / total * 100.0) if total > 0 else 0.0
    avg_r = df_trades['r_multiple'].dropna().mean() if not df_trades.empty else 0.0
    summary = {
        "symbol": symbol,
        "trades": total,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate_%": round(win_rate, 2),
        "avg_R": round(float(avg_r) if not math.isnan(avg_r) else 0.0, 3)
    }
    return trades, summary

# -----------------------------
# Data Loading
# -----------------------------

def parse_timestamp(ts):
    if isinstance(ts, (int, float, np.integer, np.floating)):
        # epoch seconds
        return pd.to_datetime(ts, unit='s', utc=True)
    return pd.to_datetime(ts, utc=True)

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = [c.lower() for c in df.columns]
    rename_map = {}
    for need in ['timestamp','open','high','low','close','volume']:
        # find matching case-insensitive col
        matches = [c for c in df.columns if c.lower() == need]
        if matches:
            rename_map[matches[0]] = need
    df = df.rename(columns=rename_map)
    missing = set(['timestamp','open','high','low','close','volume']) - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df['timestamp'] = df['timestamp'].apply(parse_timestamp)
    df = df.sort_values('timestamp')
    df = df.set_index('timestamp')
    # Ensure numeric
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df[['open','high','low','close','volume']]

def fetch_ohlcv(symbol: str, tf: str, data_dir: Optional[str]=None, csv_path: Optional[str]=None) -> pd.DataFrame:
    """
    Default loader looks for CSV files:
    - If csv_path is provided, load that directly (single symbol backtest).
    - Else expects files like: {symbol}_{tf}.csv in data_dir.
    CSV must contain timestamp, open, high, low, close, volume.
    """
    if csv_path:
        df = load_csv(csv_path)
        return df
    if not data_dir:
        raise ValueError("Either csv_path or data_dir must be provided.")
    filename = f"{symbol.replace('/','-')}_{tf.lower()}.csv"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find CSV at {path}. Expected naming: SYMBOL_{tf}.csv (e.g., BTC-USD_1h.csv)")
    return load_csv(path)

# -----------------------------
# Runner
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean TA Backtest")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory of CSV files (SYMBOL_TF.csv)")
    parser.add_argument("--csv", type=str, default=None, help="Path to a single CSV file for one symbol")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol for single CSV mode (e.g., BTC-USD)")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols for data_dir mode")
    parser.add_argument("--entry_tf", type=str, default="1H", help="Entry timeframe (e.g., 15T,1H,4H)")
    parser.add_argument("--htf", type=str, default="1D", help="Higher timeframe for trend filter")
    parser.add_argument("--ema_period_trend", type=int, default=200)
    parser.add_argument("--stoch_len", type=int, default=14)
    parser.add_argument("--stoch_k", type=int, default=3)
    parser.add_argument("--stoch_d", type=int, default=3)
    parser.add_argument("--adx_period", type=int, default=14)
    parser.add_argument("--adx_min", type=float, default=20.0)
    parser.add_argument("--atr_period", type=int, default=14)
    parser.add_argument("--tp_mult", type=float, default=2.0)
    parser.add_argument("--sl_mult", type=float, default=1.5)
    parser.add_argument("--sr_lookback", type=int, default=120)
    parser.add_argument("--swing_window", type=int, default=5)
    parser.add_argument("--retest_bars", type=int, default=4)
    parser.add_argument("--retest_tolerance", type=float, default=0.001)
    parser.add_argument("--one_trade_per_day", action="store_true")
    args = parser.parse_args()

    params = Params(
        entry_tf=args.entry_tf,
        htf=args.htf,
        ema_period_trend=args.ema_period_trend,
        stoch_len=args.stoch_len,
        stoch_k=args.stoch_k,
        stoch_d=args.stoch_d,
        adx_period=args.adx_period,
        adx_min=args.adx_min,
        atr_period=args.atr_period,
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
        sr_lookback=args.sr_lookback,
        swing_window=args.swing_window,
        retest_bars=args.retest_bars,
        retest_tolerance=args.retest_tolerance,
        one_trade_per_day=args.one_trade_per_day
    )

    if args.csv and args.symbol:
        symbols = [args.symbol]
    elif args.data_dir and args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        print("You must either:\n  (A) provide --csv and --symbol\n  or\n  (B) provide --data_dir and --symbols (comma-separated)\n", file=sys.stderr)
        sys.exit(1)

    all_trades: List[Trade] = []
    summaries: List[Dict] = []

    for sym in symbols:
        # Load entry TF data
        df_entry = fetch_ohlcv(sym, params.entry_tf, data_dir=args.data_dir, csv_path=(args.csv if args.symbol == sym else None))
        # Ensure datetime index
        if not isinstance(df_entry.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be datetime.")

        # Build HTF data from entry TF via resampling if no separate file is provided.
        # For simplicity we resample entry data up to HTF.
        df_htf = resample_ohlcv(df_entry, params.htf)

        # Run backtest
        trades, summary = backtest_symbol(sym, df_entry, df_htf, params)
        all_trades.extend(trades)
        summaries.append(summary)

    # Save outputs
    trades_df = pd.DataFrame([asdict(t) for t in all_trades]) if all_trades else pd.DataFrame(columns=[
        "symbol","side","entry_time","entry_price","sl","tp","exit_time","exit_price","outcome","r_multiple"
    ])
    summary_df = pd.DataFrame(summaries)

    out_dir = "./backtest_output"
    os.makedirs(out_dir, exist_ok=True)
    trades_path = os.path.join(out_dir, "clean_ta_trades.csv")
    summary_path = os.path.join(out_dir, "clean_ta_summary.csv")

    trades_df.to_csv(trades_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n===== Clean TA Backtest Results =====")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
        total_trades = int(summary_df["trades"].sum())
        total_wins = int(summary_df["wins"].sum())
        total_losses = int(summary_df["losses"].sum())
        win_rate = (100.0 * total_wins / total_trades) if total_trades > 0 else 0.0
        avg_R_all = trades_df['r_multiple'].dropna().mean() if not trades_df.empty else 0.0
        print(f"\nOverall trades: {total_trades} | Wins: {total_wins} | Losses: {total_losses} | Win rate: {win_rate:.2f}% | Avg R: {avg_R_all:.3f}")
    else:
        print("No trades found. Try loosening parameters or increasing data length.")

    print(f"\nTrade log saved to: {trades_path}")
    print(f"Summary saved to:   {summary_path}")

if __name__ == "__main__":
    main()
