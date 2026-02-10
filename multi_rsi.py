
#!/usr/bin/env python3
"""
Multi-Coin Multi-Timeframe Stoch RSI Backtest (1D / 6H / 1H)
------------------------------------------------------------
Run a Stoch RSI strategy across MANY symbols at once.

INPUT: One 1-hour OHLCV CSV per symbol (UTC) with columns:
    timestamp,open,high,low,close,volume

USAGE EXAMPLES:

1) Explicit pairs:
   python multi_tf_stoch_rsi_backtest_multi.py \
       --pair BTC-USD=/path/btc_1h.csv \
       --pair ETH-USD=/path/eth_1h.csv

2) JSON mapping file (symbol->csv_path):
   python multi_tf_stoch_rsi_backtest_multi.py --map_json symbols.json
   # symbols.json: {"BTC-USD": "btc.csv", "ETH-USD": "eth.csv"}

OPTIONS:
  --enable_shorts        Enable symmetric short entries/exits
  --no_daily_filter      Disable 1D K>D (or K<D) trend filter
  --config_json PATH     JSON of config overrides (keys must match Config fields)

OUTPUT:
  - trades_<SYMBOL>.csv for each coin
  - trades_all.csv with all coins combined
  - metrics_summary.csv with per-coin metrics
  - Console prints a compact metrics table

Notes:
  - This is a research backtester; it ignores fees/slippage.
  - CSV timestamps may be ISO8601 or epoch seconds/ms (UTC assumed).
"""

import argparse
import json
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

# ----------------------------
# Core from single-coin version
# ----------------------------

def parse_timestamp(ts):
    try:
        return pd.to_datetime(ts, utc=True)
    except Exception:
        ts = float(ts)
        if ts > 1e12:
            return pd.to_datetime(int(ts), unit='ms', utc=True)
        else:
            return pd.to_datetime(int(ts), unit='s', utc=True)

def resample_ohlcv(df, rule):
    o = df['open'].resample(rule).first()
    h = df['high'].resample(rule).max()
    l = df['low'].resample(rule).min()
    c = df['close'].resample(rule).last()
    v = df['volume'].resample(rule).sum()
    out = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
    return out.dropna()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def stoch_rsi(close: pd.Series, rsi_len=14, stoch_len=14, k=3, d=3) -> pd.DataFrame:
    r = rsi(close, rsi_len)
    min_r = r.rolling(stoch_len, min_periods=1).min()
    max_r = r.rolling(stoch_len, min_periods=1).max()
    denom = (max_r - min_r).replace(0, np.nan)
    stoch = ((r - min_r) / denom) * 100.0
    K = stoch.rolling(k, min_periods=1).mean()
    D = K.rolling(d, min_periods=1).mean()
    out = pd.DataFrame({'K': K.fillna(50.0), 'D': D.fillna(50.0)})
    return out

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def cross_over(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))

def cross_under(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))

@dataclass
class Config:
    rsi_len: int = 14
    stoch_len: int = 14
    k_smooth: int = 3
    d_smooth: int = 3
    enable_shorts: bool = False

    lookback_6h_cross: int = 2
    require_daily_bullish: bool = True

    atr_len: int = 14
    atr_mult_tp: float = 2.0
    atr_mult_sl: float = 1.5
    use_tp_sl: bool = True

    start_equity: float = 10000.0
    risk_per_trade_pct: float = 1.0

def build_mtf_indicators(h1: pd.DataFrame, cfg: Config):
    h6 = resample_ohlcv(h1, '6H')
    d1 = resample_ohlcv(h1, '1D')

    h1_srsi = stoch_rsi(h1['close'], cfg.rsi_len, cfg.stoch_len, cfg.k_smooth, cfg.d_smooth).rename(columns=lambda x: f'h1_{x}')
    h6_srsi = stoch_rsi(h6['close'], cfg.rsi_len, cfg.stoch_len, cfg.k_smooth, cfg.d_smooth).rename(columns=lambda x: f'h6_{x}')
    d1_srsi = stoch_rsi(d1['close'], cfg.rsi_len, cfg.stoch_len, cfg.k_smooth, cfg.d_smooth).rename(columns=lambda x: f'd1_{x}')

    h6_to_1h = h6_srsi.reindex(h1.index, method='ffill')
    d1_to_1h = d1_srsi.reindex(h1.index, method='ffill')

    h1_atr = atr(h1, cfg.atr_len).rename('atr1h')

    out = pd.concat([h1, h1_srsi, h6_to_1h, d1_to_1h, h1_atr], axis=1).dropna()
    return out

def backtest(df: pd.DataFrame, cfg: Config, symbol: str):
    h1_cross_up = cross_over(df['h1_K'], df['h1_D']) & (df['h1_K'] < 20)
    h1_cross_down = cross_under(df['h1_K'], df['h1_D']) & (df['h1_K'] > 80)

    h6_bull_now = df['h6_K'] > df['h6_D']
    h6_cross_up_recent = h6_bull_now | (cross_over(df['h6_K'], df['h6_D']).rolling(cfg.lookback_6h_cross, min_periods=1).max().astype(bool))

    d1_bull = df['d1_K'] > df['d1_D'] if cfg.require_daily_bullish else pd.Series(True, index=df.index)

    position = None
    trades = []

    for ts, row in df.iterrows():
        price = row['close']

        if position is not None:
            side, entry, t_entry, atr_entry = position
            tp = sl = None
            if cfg.use_tp_sl:
                tp = entry * (1 + cfg.atr_mult_tp * (atr_entry / entry)) if side == 'long' else entry * (1 - cfg.atr_mult_tp * (atr_entry / entry))
                sl = entry * (1 - cfg.atr_mult_sl * (atr_entry / entry)) if side == 'long' else entry * (1 + cfg.atr_mult_sl * (atr_entry / entry))

            exit_reason = None
            exit_price = None

            if cfg.use_tp_sl and tp is not None and sl is not None:
                bar_high = row['high']
                bar_low = row['low']
                if side == 'long':
                    if bar_low <= sl:
                        exit_price = sl; exit_reason = 'SL'
                    elif bar_high >= tp:
                        exit_price = tp; exit_reason = 'TP'
                else:
                    if bar_high >= sl:
                        exit_price = sl; exit_reason = 'SL'
                    elif bar_low <= tp:
                        exit_price = tp; exit_reason = 'TP'

            if exit_price is None:
                if side == 'long' and h1_cross_down.loc[ts]:
                    exit_price = price; exit_reason = 'K↓ >80'
                elif side == 'short' and cfg.enable_shorts and cross_over(df['h1_K'], df['h1_D']).loc[ts] and df['h1_K'].loc[ts] < 20:
                    exit_price = price; exit_reason = 'K↑ <20 (short exit)'

            if exit_price is not None:
                ret_pct = (exit_price - entry) / entry if side == 'long' else (entry - exit_price) / entry
                trades.append({
                    'symbol': symbol,
                    'side': side,
                    'entry_time': t_entry,
                    'exit_time': ts,
                    'entry': entry,
                    'exit': exit_price,
                    'return_pct': ret_pct * 100.0,
                    'exit_reason': exit_reason
                })
                position = None

        if position is None:
            if h1_cross_up.loc[ts] and h6_cross_up_recent.loc[ts] and d1_bull.loc[ts]:
                position = ('long', price, ts, row['atr1h'])
                continue

            if cfg.enable_shorts:
                h1_cross_down_entry = cross_under(df['h1_K'], df['h1_D']) & (df['h1_K'] > 80)
                d1_bear = (df['d1_K'] < df['d1_D']) if cfg.require_daily_bullish else pd.Series(True, index=df.index)
                h6_bear_now = df['h6_K'] < df['h6_D']
                h6_cross_down_recent = h6_bear_now | (cross_under(df['h6_K'], df['h6_D']).rolling(cfg.lookback_6h_cross, min_periods=1).max().astype(bool))
                if h1_cross_down_entry.loc[ts] and h6_cross_down_recent.loc[ts] and d1_bear.loc[ts]:
                    position = ('short', price, ts, row['atr1h'])
                    continue

    trades_df = pd.DataFrame(trades)
    metrics = {}
    if not trades_df.empty:
        total_ret = (trades_df['return_pct'] / 100 + 1).prod() - 1
        wins = (trades_df['return_pct'] > 0).sum()
        losses = (trades_df['return_pct'] <= 0).sum()
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0.0
        avg_return = trades_df['return_pct'].mean()
        metrics = {
            'symbol': symbol,
            'trades': len(trades_df),
            'win_rate_pct': win_rate,
            'avg_return_pct': avg_return,
            'total_return_pct': total_ret * 100.0,
        }
    else:
        metrics = {'symbol': symbol, 'trades': 0, 'win_rate_pct': 0.0, 'avg_return_pct': 0.0, 'total_return_pct': 0.0}
    return trades_df, metrics

def load_1h_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    needed = ['timestamp','open','high','low','close','volume']
    rename = {}
    for need in needed:
        if need in lower:
            rename[lower[need]] = need
    df = df.rename(columns=rename)
    if 'timestamp' not in df.columns:
        raise ValueError(f"CSV must include 'timestamp'. File: {path}")
    df['timestamp'] = df['timestamp'].apply(parse_timestamp)
    df = df.set_index('timestamp').sort_index()
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', action='append', help='Repeatable: SYMBOL=PATH (e.g., BTC-USD=/data/btc.csv)')
    ap.add_argument('--map_json', help='Path to JSON file: {"SYMBOL": "path.csv", ...}')
    ap.add_argument('--enable_shorts', action='store_true')
    ap.add_argument('--no_daily_filter', action='store_true')
    ap.add_argument('--config_json', help='Config override JSON')
    args = ap.parse_args()

    # Build symbol->path mapping
    sym2path: Dict[str, str] = {}
    if args.pair:
        for item in args.pair:
            if '=' not in item:
                raise ValueError(f"--pair must be SYMBOL=PATH, got: {item}")
            sym, path = item.split('=', 1)
            sym2path[sym.strip()] = path.strip()
    if args.map_json:
        with open(args.map_json, 'r') as f:
            mapping = json.load(f)
        for k, v in mapping.items():
            sym2path[k] = v

    if not sym2path:
        raise SystemExit("Provide at least one --pair SYMBOL=PATH or a --map_json.")

    cfg = Config()
    if args.enable_shorts:
        cfg.enable_shorts = True
    if args.no_daily_filter:
        cfg.require_daily_bullish = False
    if args.config_json:
        with open(args.config_json, 'r') as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    all_trades: List[pd.DataFrame] = []
    metrics_rows: List[dict] = []

    for symbol, path in sym2path.items():
        h1 = load_1h_csv(path)
        df = build_mtf_indicators(h1, cfg)
        trades_df, metrics = backtest(df, cfg, symbol)
        metrics_rows.append(metrics)

        # Save per-coin trades
        per_path = f"trades_{symbol.replace('/','_')}.csv"
        trades_df.to_csv(per_path, index=False)

        if not trades_df.empty:
            all_trades.append(trades_df)

    # Aggregate
    if all_trades:
        trades_all = pd.concat(all_trades, ignore_index=True)
    else:
        trades_all = pd.DataFrame(columns=['symbol','side','entry_time','exit_time','entry','exit','return_pct','exit_reason'])

    trades_all.to_csv("trades_all.csv", index=False)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("metrics_summary.csv", index=False)

    # Pretty print
    if not metrics_df.empty:
        cols = ['symbol','trades','win_rate_pct','avg_return_pct','total_return_pct']
        show = metrics_df[cols].copy()
        # Round for display
        for c in ['win_rate_pct','avg_return_pct','total_return_pct']:
            show[c] = show[c].astype(float).round(2)
        print("\nPer-coin metrics:")
        print(show.to_string(index=False))
        print("\nSaved: trades_all.csv, metrics_summary.csv, trades_<SYMBOL>.csv")
    else:
        print("No trades generated. Check your data and rules.")
        print("Saved: trades_all.csv (empty), metrics_summary.csv (empty).")

if __name__ == "__main__":
    main()
