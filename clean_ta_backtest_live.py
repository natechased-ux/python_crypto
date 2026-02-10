
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean TA Backtest (Live Fetch Version)
--------------------------------------
- Fetches OHLCV from Bybit v5 public API (default).
- Multiple symbols supported via --symbols "BTC-USD,ETH-USD,..."
- Date range via --start/--end (UTC ISO, e.g., 2024-01-01 00:00:00).

Strategy:
- Daily trend filter: 200 EMA on HTF (default 1D)
- Entry TF (default 1H): Reclaim of resistance / loss of support with retest
- Momentum confirm (strict): Stoch RSI (K>D & K<40 for longs; K<D & K>60 for shorts)
- ADX filter (default >=20)
- ATR-based TP/SL
- Optional one-trade-per-day per symbol
- Outputs CSV logs + console summary

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
from datetime import datetime, timezone

# -----------------------------
# Indicators (no TA-Lib)
# -----------------------------

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
    if use_rma:
        return rma(tr, period)
    else:
        return tr.rolling(period).mean()

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
    adx_val = rma(dx, period)
    return adx_val

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

# -----------------------------
# Strategy Helpers
# -----------------------------

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

# -----------------------------
# Data: Bybit v5 fetch
# -----------------------------

BYBIT_BASE = "https://api.bybit.com"

INTERVAL_MAP = {
    "15T": "15",  # 15 minutes
    "30T": "30",
    "1H": "60",
    "2H": "120",
    "4H": "240",
    "6H": "360",
    "12H":"720",
    "1D": "D",
    "1W": "W"
}

def normalize_symbol_to_bybit(symbol: str) -> str:
    # "BTC-USD" -> "BTCUSDT"; "ETH-USD" -> "ETHUSDT"
    s = symbol.replace("/", "-").upper()
    s = s.replace("-USD", "USDT").replace("-USDT", "USDT")
    s = s.replace("-", "")
    return s

def to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def fetch_bybit_klines(symbol: str, interval_code: str, start_dt: datetime, end_dt: datetime, category: str="spot", limit: int=1000, pause: float=0.2) -> pd.DataFrame:
    """
    Pulls historical klines from Bybit v5 (spot by default).
    Pagination via 'cursor'. Returns DataFrame with UTC datetime index.
    """
    sym = normalize_symbol_to_bybit(symbol)
    start_ms = to_ms(start_dt)
    end_ms = to_ms(end_dt)
    rows = []
    cursor = None
    while True:
        params = {
            "category": category,
            "symbol": sym,
            "interval": interval_code,
            "start": start_ms,
            "end": end_ms,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{BYBIT_BASE}/v5/market/kline", params=params, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"Bybit API error {r.status_code}: {r.text}")
        data = r.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit API retCode {data.get('retCode')}: {data.get('retMsg')}")
        result = data.get("result", {})
        klines = result.get("list", [])
        if not klines:
            break
        # Each item: [startTime, open, high, low, close, volume, turnover]
        for k in klines:
            ts = pd.to_datetime(int(k[0]), unit="ms", utc=True)
            o,h,l,c,v = map(float, k[1:6])
            rows.append([ts, o,h,l,c,v])
        cursor = result.get("nextPageCursor")
        if not cursor:
            break
        time.sleep(pause)  # polite
    if not rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    return df

# -----------------------------
# Backtest core
# -----------------------------

def backtest_symbol(symbol: str, df_entry: pd.DataFrame, df_htf: pd.DataFrame, params: Params) -> Tuple[List[Trade], Dict]:
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

    htf_trend = derive_trend_filter(df_htf, params.ema_period_trend)
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

# -----------------------------
# Runner
# -----------------------------

def parse_iso(s: str) -> datetime:
    # Accept "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
    try:
        dt = datetime.fromisoformat(s.replace("Z","").strip())
    except Exception:
        raise ValueError(f"Couldn't parse datetime: {s}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

def main():
    parser = argparse.ArgumentParser(description="Clean TA Backtest (Live Fetch)")
    parser.add_argument("--exchange", type=str, default="bybit", choices=["bybit"], help="Exchange for data")
    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols, e.g., 'BTC-USD,ETH-USD,XRP-USD'")
    parser.add_argument("--entry_tf", type=str, default="1H", help="Entry timeframe (15T,30T,1H,2H,4H,6H,12H,1D)")
    parser.add_argument("--htf", type=str, default="1D", help="Higher timeframe for trend filter")
    parser.add_argument("--start", type=str, required=True, help="Start datetime UTC (e.g., '2024-01-01 00:00:00')")
    parser.add_argument("--end", type=str, required=True, help="End datetime UTC (e.g., '2025-09-01 00:00:00')")

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
    parser.add_argument("--category", type=str, default="spot", choices=["spot","linear"], help="Bybit category: spot or linear (USDT perps)")

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

    # Validate intervals
    if args.entry_tf not in INTERVAL_MAP or args.htf not in INTERVAL_MAP:
        print(f"Unsupported timeframe. Supported: {list(INTERVAL_MAP.keys())}", file=sys.stderr)
        sys.exit(1)

    all_trades: List[Trade] = []
    summaries: List[Dict] = []

    for sym in symbols:
        # Fetch entry TF
        interval_entry = INTERVAL_MAP[params.entry_tf]
        df_entry = fetch_bybit_klines(sym, interval_entry, start_dt, end_dt, category=args.category)
        if df_entry.empty or len(df_entry) < 300:
            print(f"[{sym}] Not enough data on entry TF ({params.entry_tf}). Skipping.", file=sys.stderr)
            continue

        # Build HTF via resampling (consistent with the entry feed)
        df_htf = resample_ohlcv(df_entry, params.htf)
        if df_htf.empty or len(df_htf) < 250:
            print(f"[{sym}] Not enough data on HTF ({params.htf}). Skipping.", file=sys.stderr)
            continue

        # Run backtest
        trades, summary = backtest_symbol(sym, df_entry, df_htf, params)
        all_trades.extend(trades)
        summaries.append(summary)

    trades_df = pd.DataFrame([asdict(t) for t in all_trades]) if all_trades else pd.DataFrame(columns=[
        "symbol","side","entry_time","entry_price","sl","tp","exit_time","exit_price","outcome","r_multiple"
    ])
    summary_df = pd.DataFrame(summaries)

    out_dir = "./backtest_output"
    os.makedirs(out_dir, exist_ok=True)
    trades_path = os.path.join(out_dir, "clean_ta_trades_live.csv")
    summary_path = os.path.join(out_dir, "clean_ta_summary_live.csv")

    trades_df.to_csv(trades_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n===== Clean TA Backtest (Live Fetch) =====")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
        total_trades = int(summary_df["trades"].sum())
        total_wins = int(summary_df["wins"].sum())
        total_losses = int(summary_df["losses"].sum())
        win_rate = (100.0 * total_wins / total_trades) if total_trades > 0 else 0.0
        avg_R_all = trades_df['r_multiple'].dropna().mean() if not trades_df.empty else 0.0
        print(f"\nOverall trades: {total_trades} | Wins: {total_wins} | Losses: {total_losses} | Win rate: {win_rate:.2f}% | Avg R: {avg_R_all:.3f}")
    else:
        print("No trades found. Try loosening parameters or expanding the date range.")

    print(f"\nTrade log saved to: {trades_path}")
    print(f"Summary saved to:   {summary_path}")

if __name__ == "__main__":
    main()
