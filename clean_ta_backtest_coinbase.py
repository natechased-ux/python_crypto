
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean TA Backtest (Coinbase Live Fetch)
--------------------------------------
- Pulls OHLCV candlesticks from Coinbase Exchange public API.
- Handles multiple symbols and a start/end range.
- Runs the Clean TA strategy:
  * HTF 200 EMA trend filter (default 1D)
  * Reclaim+retest (long) / loss+retest (short) of swing S/R
  * Momentum confirm: Stoch RSI (strict) + ADX floor
  * ATR-based SL/TP
- Next-bar execution with intrabar TP/SL check (conservative ordering)
- Outputs CSV logs and console summary

Usage (Windows PowerShell/CMD with `py`):
  py clean_ta_backtest_coinbase.py ^
    --symbols "BTC-USD,ETH-USD,XRP-USD" ^
    --entry_tf 1H --htf 1D ^
    --start "2024-01-01 00:00:00" ^
    --end   "2025-09-01 00:00:00"

Author: ChatGPT
"""
import argparse
import os
import sys
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta

# --------------- Indicators (no TA-Lib) ---------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def rma(series: pd.Series, period: int) -> pd.Series:
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
    return rma(tr, period) if use_rma else tr.rolling(period).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
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
    return rma(dx, period)

def stoch_rsi(close: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0))
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()
    stoch = 100 * (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d

# --------------- Strategy helpers ---------------

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
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
    highs = df['high']
    lows  = df['low']

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
    entry_tf: str = "1H"
    htf: str = "1D"
    ema_period_trend: int = 200
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
    retest_bars: int = 4
    retest_tolerance: float = 0.001
    one_trade_per_day: bool = False

@dataclass
class Trade:
    symbol: str
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    sl: float
    tp: float
    exit_time: Optional[pd.Timestamp]
    exit_price: Optional[float]
    outcome: Optional[str]
    r_multiple: Optional[float]

def derive_trend_filter(htf_df: pd.DataFrame, ema_period: int) -> pd.Series:
    ema_htf = ema(htf_df['close'], ema_period)
    return (htf_df['close'] > ema_htf).astype(int)  # 1 bull, 0 bear

def align_htf_to_entry(htf_signal: pd.Series, entry_index: pd.DatetimeIndex) -> pd.Series:
    return htf_signal.reindex(entry_index, method='ffill')

def momentum_confirms(k: pd.Series, d: pd.Series, side: str) -> pd.Series:
    if side == "long":
        return ((k > d) & (k < 40)).astype(int)
    else:
        return ((k < d) & (k > 60)).astype(int)

def reclaim_retest_long(df: pd.DataFrame, retest_bars: int, tol: float) -> pd.Series:
    idx = df.index
    signal = pd.Series(0, index=idx)
    for i in range(1, len(df)-1):
        level = df.loc[idx[i], 'last_resistance']
        if math.isnan(level):
            continue
        prev_close = df.loc[idx[i-1], 'close']
        close_i = df.loc[idx[i], 'close']
        if (prev_close <= level) and (close_i > level):
            for j in range(i+1, min(i+1+retest_bars, len(df))):
                low_j = df.loc[idx[j], 'low']
                close_j = df.loc[idx[j], 'close']
                if low_j <= level*(1+tol) and close_j > level:
                    signal.iloc[j] = 1
                    break
    return signal

def loss_retest_short(df: pd.DataFrame, retest_bars: int, tol: float) -> pd.Series:
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
    if side == "long":
        hit_sl = row_low <= sl
        hit_tp = row_high >= tp
        if hit_sl and hit_tp:
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

# --------------- Coinbase fetch ---------------

CB_BASE = "https://api.exchange.coinbase.com"
# Coinbase supports granularity (seconds): 60, 300, 900, 3600, 21600, 86400
SUPPORTED = [60, 300, 900, 3600, 21600, 86400]

TF_TO_SECONDS = {
    "1M": 60,  # 1 minute
    "5M": 300,
    "15T": 900,  # 15 minutes
    "1H": 3600,
    "2H": 7200,   # not native; fetch 1H and resample
    "4H": 14400,  # not native; fetch 1H and resample
    "6H": 21600,
    "12H": 43200, # not native; fetch 1H and resample
    "1D": 86400
}

def parse_iso(s: str) -> datetime:
    # Accept "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
    try:
        dt = datetime.fromisoformat(s.replace("Z","").strip())
    except Exception as e:
        raise ValueError(f"Couldn't parse datetime: {s}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

def fetch_coinbase_candles(product_id: str, granularity: int, start_dt: datetime, end_dt: datetime, pause: float=0.35) -> pd.DataFrame:
    """Fetch candles in chunks of up to 300 bars per request."""
    rows = []
    start = start_dt
    headers = {"User-Agent": "CleanTA-Backtest/1.0"}
    # Coinbase expects ISO8601; returns arrays [time, low, high, open, close, volume]
    while start < end_dt:
        chunk = granularity * 300
        end = min(start + timedelta(seconds=chunk), end_dt)
        params = {
            "start": start.astimezone(timezone.utc).isoformat(),
            "end": end.astimezone(timezone.utc).isoformat(),
            "granularity": granularity
        }
        url = f"{CB_BASE}/products/{product_id}/candles"
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"Coinbase API error {r.status_code}: {r.text}")
        data = r.json()
        if not isinstance(data, list):
            break
        for arr in data:
            ts = pd.to_datetime(int(arr[0]), unit="s", utc=True)
            low, high, open_, close, vol = float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4]), float(arr[5])
            rows.append([ts, open_, high, low, close, vol])
        start = end
        time.sleep(pause)
    if not rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    return df

def get_entry_dataframe(product_id: str, tf_label: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    secs = TF_TO_SECONDS.get(tf_label)
    if secs is None:
        raise ValueError(f"Unsupported entry_tf: {tf_label}. Use one of {list(TF_TO_SECONDS.keys())}.")
    # If native granularity, fetch directly; else fetch 1H and resample
    fetch_secs = secs if secs in SUPPORTED else 3600
    df = fetch_coinbase_candles(product_id, fetch_secs, start_dt, end_dt)
    if df.empty:
        return df
    if fetch_secs != secs:
        # Resample to desired tf_label
        rule = tf_label
        df = resample_ohlcv(df, rule)
    return df

def get_htf_dataframe_from_entry(df_entry: pd.DataFrame, htf_label: str) -> pd.DataFrame:
    return resample_ohlcv(df_entry, htf_label)

# --------------- Backtest ---------------

def backtest_symbol(symbol: str, df_entry: pd.DataFrame, df_htf: pd.DataFrame, params: Params):
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

    htf_trend = derive_trend_filter(df_htf, params.ema_period_trend)  # 1 bull, 0 bear
    trend_on_entry = align_htf_to_entry(htf_trend, df_entry.index)
    df_entry['bull_trend'] = trend_on_entry

    trades: List[Trade] = []
    last_trade_day: Optional[pd.Timestamp] = None

    for i in range(1, len(df_entry)):
        ts = df_entry.index[i]
        if params.one_trade_per_day:
            if last_trade_day is not None and ts.date() == last_trade_day.date():
                continue
        row = df_entry.iloc[i]

        # LONG
        long_setup = (row['bull_trend'] == 1) and (long_rr.iloc[i] == 1)
        long_momo  = momentum_confirms(df_entry['k'], df_entry['d'], "long").iloc[i] == 1
        long_adx   = (row['adx'] >= params.adx_min)
        if long_setup and long_momo and long_adx and not np.isnan(row['atr']):
            entry = row['close']
            sl = entry - params.sl_mult * row['atr']
            tp = entry + params.tp_mult * row['atr']
            if i+1 < len(df_entry):
                next_row = df_entry.iloc[i+1]
                outcome, exit_price = intrabar_exit_price(next_row['high'], next_row['low'], entry, tp, sl, "long")
                exit_time = df_entry.index[i+1]
                r = (exit_price - entry) / (entry - sl) if outcome in ("TP","SL") else np.nan
                trades.append(Trade(symbol, "long", ts, float(entry), float(sl), float(tp),
                                    exit_time, float(exit_price), outcome, float(r) if not np.isnan(r) else None))
                last_trade_day = ts
                continue

        # SHORT
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
                r = (entry - exit_price) / (sl - entry) if outcome in ("TP","SL") else np.nan
                trades.append(Trade(symbol, "short", ts, float(entry), float(sl), float(tp),
                                    exit_time, float(exit_price), outcome, float(r) if not np.isnan(r) else None))
                last_trade_day = ts
                continue

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

# --------------- Runner ---------------

def main():
    parser = argparse.ArgumentParser(description="Clean TA Backtest (Coinbase Live Fetch)")
    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated product IDs, e.g., 'BTC-USD,ETH-USD,XRP-USD'")
    parser.add_argument("--entry_tf", type=str, default="1H", help="Entry timeframe (1M,5M,15T,1H,2H,4H,6H,12H,1D)")
    parser.add_argument("--htf", type=str, default="1D", help="Higher timeframe for trend filter (same labels)")
    parser.add_argument("--start", type=str, required=True, help="Start datetime UTC, e.g., '2024-01-01 00:00:00'")
    parser.add_argument("--end", type=str, required=True, help="End datetime UTC, e.g., '2025-09-01 00:00:00'")

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

    start_dt = parse_iso(args.start)
    end_dt = parse_iso(args.end)
    if end_dt <= start_dt:
        print("Error: --end must be after --start", file=sys.stderr)
        sys.exit(1)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("No symbols provided.", file=sys.stderr)
        sys.exit(1)

    all_trades: List[Trade] = []
    summaries: List[Dict] = []

    for sym in symbols:
        # Entry TF fetch (fetch native or fetch 1H then resample if needed)
        try:
            df_entry = get_entry_dataframe(sym, params.entry_tf, start_dt, end_dt)
        except Exception as e:
            print(f"[{sym}] fetch failed: {e}", file=sys.stderr)
            continue
        if df_entry.empty or len(df_entry) < 300:
            print(f"[{sym}] Not enough entry data ({params.entry_tf}). Skipping.", file=sys.stderr)
            continue

        # HTF from entry via resample (ensures exact index align)
        df_htf = get_htf_dataframe_from_entry(df_entry, params.htf)
        if df_htf.empty or len(df_htf) < 250:
            print(f"[{sym}] Not enough HTF data ({params.htf}). Skipping.", file=sys.stderr)
            continue

        trades, summary = backtest_symbol(sym, df_entry, df_htf, params)
        all_trades.extend(trades)
        summaries.append(summary)

    trades_df = pd.DataFrame([asdict(t) for t in all_trades]) if all_trades else pd.DataFrame(columns=[
        "symbol","side","entry_time","entry_price","sl","tp","exit_time","exit_price","outcome","r_multiple"
    ])
    summary_df = pd.DataFrame(summaries)

    out_dir = "./backtest_output"
    os.makedirs(out_dir, exist_ok=True)
    trades_path = os.path.join(out_dir, "clean_ta_trades_coinbase.csv")
    summary_path = os.path.join(out_dir, "clean_ta_summary_coinbase.csv")

    trades_df.to_csv(trades_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n===== Clean TA Backtest (Coinbase) =====")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
        total_trades = int(summary_df["trades"].sum())
        total_wins = int(summary_df["wins"].sum())
        total_losses = int(summary_df["losses"].sum())
        win_rate = (100.0 * total_wins / total_trades) if total_trades > 0 else 0.0
        avg_R_all = trades_df['r_multiple'].dropna().mean() if not trades_df.empty else 0.0
        print(f"\nOverall trades: {total_trades} | Wins: {total_wins} | Losses: {total_losses} | Win rate: {win_rate:.2f}% | Avg R: {avg_R_all:.3f}")
    else:
        print("No trades found. Try loosening filters or expanding the date range.")

    print(f"\nTrade log saved to: {trades_path}")
    print(f"Summary saved to:   {summary_path}")

if __name__ == "__main__":
    main()
