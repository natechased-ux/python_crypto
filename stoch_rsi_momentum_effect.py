
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Momentum Effect Analyzer (Coinbase, chunked + multi-symbol)
===========================================================

Modes (choose with --mode):
- **stochrsi**: Median % move while Stoch RSI momentum runs (K>D until K<D, and vice versa)
- **adx**:      Median % move while ADX trend holds ((+DI>-DI & ADX rising > thresh) until cross, and vice versa)

Examples
--------
py stoch_rsi_momentum_effect.py --mode adx --exchange coinbase --symbols BTC-USD,ETH-USD,XRP-USD,SOL-USD --timeframe 6h --lookback_days 365 --adx_len 14 --adx_thresh 20 --csv adx_results.csv

py stoch_rsi_momentum_effect.py --mode stochrsi --exchange coinbase --symbols XRP-USD --timeframe 1h --lookback_days 180 --zone --csv stoch_results.csv
"""

import argparse
import datetime as dt
import numpy as np
import pandas as pd
import requests
from dateutil import parser as dateparser

############################
# Coinbase data (chunked)
############################

GRAN_MAP = {'1m':60,'5m':300,'15m':900,'1h':3600,'6h':21600,'1d':86400}
MAX_PER_CALL = 300  # Coinbase's approximate limit

def _coinbase_call(product_id: str, granularity: int, start=None, end=None):
    base = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {"granularity": granularity}
    if start:
        params["start"] = dateparser.parse(str(start)).isoformat()
    if end:
        params["end"] = dateparser.parse(str(end)).isoformat()
    r = requests.get(base, params=params, timeout=20, headers={"User-Agent":"momentum-effect-test"})
    r.raise_for_status()
    data = r.json()
    rows = []
    for arr in data:
        # [time, low, high, open, close, volume] newest-first
        rows.append({
            "time": int(arr[0]) * 1000,
            "open": float(arr[3]),
            "high": float(arr[2]),
            "low": float(arr[1]),
            "close": float(arr[4]),
            "volume": float(arr[5]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df

def fetch_coinbase_chunked(product_id: str, timeframe: str, start=None, end=None, lookback_days: int = None, max_candles: int = 10000):
    if timeframe not in GRAN_MAP:
        raise ValueError("Coinbase: unsupported timeframe")
    gran = GRAN_MAP[timeframe]

    # Determine end/start
    if end is None:
        end_dt = dt.datetime.utcnow()
    else:
        end_dt = dateparser.parse(str(end)).replace(tzinfo=None)
    if start is None and lookback_days is not None:
        start_dt = end_dt - dt.timedelta(days=int(lookback_days))
    elif start is not None:
        start_dt = dateparser.parse(str(start)).replace(tzinfo=None)
    else:
        # default to 120-day lookback
        start_dt = end_dt - dt.timedelta(days=120)

    # Iterate backwards, appending chunks until we pass start_dt or hit max_candles
    all_rows = []
    cur_end = end_dt
    candles = 0
    while candles < max_candles:
        cur_start = max(start_dt, cur_end - dt.timedelta(seconds=gran*MAX_PER_CALL))
        df = _coinbase_call(product_id, gran, start=cur_start, end=cur_end)
        if df.empty:
            break
        all_rows.append(df)
        candles += len(df)
        earliest_ms = int(df["time"].iloc[0])
        cur_end = dt.datetime.utcfromtimestamp((earliest_ms // 1000) - gran)
        if cur_end <= start_dt:
            break

    if not all_rows:
        return pd.DataFrame(columns=["time","open","high","low","close","volume","date"])

    full = pd.concat(all_rows, ignore_index=True).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    # trim to [start_dt, end_dt]
    mask = (full["date"] >= pd.Timestamp(start_dt, tz="UTC")) & (full["date"] <= pd.Timestamp(end_dt, tz="UTC"))
    full = full.loc[mask].reset_index(drop=True)
    # enforce max_candles
    if len(full) > max_candles:
        full = full.iloc[-max_candles:].reset_index(drop=True)
    return full

############################
# Indicators
############################

# ===========================
# MACD
# ===========================
def ema(series: pd.Series, length: int):
    return series.ewm(span=length, adjust=False).mean()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal_len: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal = macd_line.ewm(span=signal_len, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

def find_segments_macd(df: pd.DataFrame, zero_filter=False):
    macd_line = df["macd_line"]
    signal = df["macd_signal"]
    # Cross conditions
    crosses_up = (macd_line.shift(1) <= signal.shift(1)) & (macd_line > signal)
    crosses_dn = (macd_line.shift(1) >= signal.shift(1)) & (macd_line < signal)

    up_segments = []
    down_segments = []

    i = 1
    n = len(df)
    while i < n:
        if crosses_up.iloc[i] and (not zero_filter or (macd_line.iloc[i] > 0)):
            start = i
            j = i + 1
            while j < n and not crosses_dn.iloc[j]:
                j += 1
            if j < n:
                pct = (df["close"].iloc[j] - df["close"].iloc[start]) / df["close"].iloc[start] * 100.0
                up_segments.append({"start": start, "end": j, "pct": pct})
                i = j + 1
                continue
        if crosses_dn.iloc[i] and (not zero_filter or (macd_line.iloc[i] < 0)):
            start = i
            j = i + 1
            while j < n and not crosses_up.iloc[j]:
                j += 1
            if j < n:
                pct = (df["close"].iloc[j] - df["close"].iloc[start]) / df["close"].iloc[start] * 100.0
                down_segments.append({"start": start, "end": j, "pct": pct})
                i = j + 1
                continue
        i += 1

    return up_segments, down_segments

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def stoch_rsi(close: pd.Series, rsi_len=14, stoch_len=14, k_len=3, d_len=3):
    r = rsi(close, rsi_len)
    ll = r.rolling(stoch_len, min_periods=stoch_len).min()
    hh = r.rolling(stoch_len, min_periods=stoch_len).max()
    stoch = (r - ll) / (hh - ll).replace(0, np.nan) * 100.0
    k = stoch.rolling(k_len, min_periods=k_len).mean()
    d = k.rolling(d_len, min_periods=d_len).mean()
    return k, d

def adx_di(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # Directional movement
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    # Wilder's smoothing (EMA with alpha=1/length)
    tr_s = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_dm_s = plus_dm.ewm(alpha=1/length, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(alpha=1/length, adjust=False).mean()

    plus_di = 100 * (plus_dm_s / tr_s.replace(0, np.nan))
    minus_di = 100 * (minus_dm_s / tr_s.replace(0, np.nan))

    dx = 100 * ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) )
    adx = dx.ewm(alpha=1/length, adjust=False).mean()

    return plus_di, minus_di, adx

############################
# Segment logic
############################

def find_segments_stochrsi(df: pd.DataFrame, zone=False):
    k, d = df["k"], df["d"]
    crosses_up = (k.shift(1) <= d.shift(1)) & (k > d)
    crosses_dn = (k.shift(1) >= d.shift(1)) & (k < d)
    up_segments = []
    down_segments = []
    i = 1
    n = len(df)
    while i < n:
        if crosses_up.iloc[i] and (not zone or (k.iloc[i] < 20)):
            start = i
            j = i + 1
            while j < n and not crosses_dn.iloc[j]:
                j += 1
            if j < n:
                pct = (df["close"].iloc[j] - df["close"].iloc[start]) / df["close"].iloc[start] * 100.0
                up_segments.append({"start": start, "end": j, "pct": pct})
                i = j + 1
                continue
        if crosses_dn.iloc[i] and (not zone or (k.iloc[i] > 80)):
            start = i
            j = i + 1
            while j < n and not crosses_up.iloc[j]:
                j += 1
            if j < n:
                pct = (df["close"].iloc[j] - df["close"].iloc[start]) / df["close"].iloc[start] * 100.0
                down_segments.append({"start": start, "end": j, "pct": pct})
                i = j + 1
                continue
        i += 1
    return up_segments, down_segments

def find_segments_adx(df: pd.DataFrame, adx_thresh=20.0, require_rising=True):
    plus_di = df["plus_di"]
    minus_di = df["minus_di"]
    adx = df["adx"]

    cross_up = (plus_di.shift(1) <= minus_di.shift(1)) & (plus_di > minus_di)   # +DI crossing above -DI
    cross_dn = (minus_di.shift(1) <= plus_di.shift(1)) & (minus_di > plus_di)   # -DI crossing above +DI

    up_segments = []
    down_segments = []
    i = 1
    n = len(df)

    while i < n:
        # Start up trend when +DI > -DI & ADX > thresh (and optionally rising)
        cond_up = (plus_di.iloc[i] > minus_di.iloc[i]) and (adx.iloc[i] > adx_thresh)
        if require_rising:
            cond_up = cond_up and (adx.iloc[i] > adx.shift(1).iloc[i])

        if cond_up:
            start = i
            j = i + 1
            # end when -DI crosses above +DI
            while j < n and not ((minus_di.shift(1).iloc[j] <= plus_di.shift(1).iloc[j]) and (minus_di.iloc[j] > plus_di.iloc[j])):
                j += 1
            if j < n:
                pct = (df["close"].iloc[j] - df["close"].iloc[start]) / df["close"].iloc[start] * 100.0
                up_segments.append({"start": start, "end": j, "pct": pct})
                i = j + 1
                continue

        # Start down trend when -DI > +DI & ADX > thresh (and optionally rising)
        cond_dn = (minus_di.iloc[i] > plus_di.iloc[i]) and (adx.iloc[i] > adx_thresh)
        if require_rising:
            cond_dn = cond_dn and (adx.iloc[i] > adx.shift(1).iloc[i])

        if cond_dn:
            start = i
            j = i + 1
            # end when +DI crosses above -DI
            while j < n and not ((plus_di.shift(1).iloc[j] <= minus_di.shift(1).iloc[j]) and (plus_di.iloc[j] > minus_di.iloc[j])):
                j += 1
            if j < n:
                pct = (df["close"].iloc[j] - df["close"].iloc[start]) / df["close"].iloc[start] * 100.0
                down_segments.append({"start": start, "end": j, "pct": pct})
                i = j + 1
                continue

        i += 1

    return up_segments, down_segments

############################
# Stats helpers
############################

def describe_changes(changes):
    if len(changes) == 0:
        return {"count": 0, "median": np.nan, "mean": np.nan, "iqr": (np.nan, np.nan)}
    arr = np.array(changes, dtype=float)
    q25, q50, q75 = np.nanpercentile(arr, [25, 50, 75])
    return {"count": len(arr), "median": q50, "mean": float(np.nanmean(arr)), "iqr": (q25, q75)}

############################
# Runners
############################

def analyze_symbol(mode, symbol, timeframe, start, end, lookback_days, max_candles,
                   rsi_len, stoch_len, k_len, d_len, zone,
                   adx_len, adx_thresh, adx_require_rising,
                   macd_fast, macd_slow, macd_signal_len, macd_zero_filter):
    df = fetch_coinbase_chunked(symbol, timeframe, start=start, end=end, lookback_days=lookback_days, max_candles=max_candles)
    if df.empty:
        return symbol, None, None

    if mode == "stochrsi":
        if len(df) < max(50, rsi_len + stoch_len + k_len + d_len + 5):
            return symbol, None, None
        k, d = stoch_rsi(df["close"], rsi_len=rsi_len, stoch_len=stoch_len, k_len=k_len, d_len=d_len)
        df["k"], df["d"] = k, d
        df = df.dropna().reset_index(drop=True)
        up_segments, down_segments = find_segments_stochrsi(df, zone=zone)

    elif mode == "adx":
        if len(df) < max(50, adx_len * 3):
            return symbol, None, None
        plus_di, minus_di, adx = adx_di(df["high"], df["low"], df["close"], length=adx_len)
        df["plus_di"], df["minus_di"], df["adx"] = plus_di, minus_di, adx
        df = df.dropna().reset_index(drop=True)
        up_segments, down_segments = find_segments_adx(df, adx_thresh=adx_thresh, require_rising=adx_require_rising)

    elif mode == "macd":
        if len(df) < 100:
            return symbol, None, None
        macd_line, macd_signal, macd_hist = macd(df['close'], fast=macd_fast, slow=macd_slow, signal_len=macd_signal_len)
        df['macd_line'], df['macd_signal'], df['macd_hist'] = macd_line, macd_signal, macd_hist
        df = df.dropna().reset_index(drop=True)
        up_segments, down_segments = find_segments_macd(df, zero_filter=macd_zero_filter)
    else:
        raise ValueError("Unknown mode")

    up_changes = [s["pct"] for s in up_segments]
    down_changes = [s["pct"] for s in down_segments]
    return symbol, describe_changes(up_changes), describe_changes(down_changes)

def run_multi(mode, symbols, timeframe, start, end, lookback_days, max_candles,
              rsi_len, stoch_len, k_len, d_len, zone,
              adx_len, adx_thresh, adx_require_rising,
              macd_fast, macd_slow, macd_signal_len, macd_zero_filter,
              csv_path=None):
    results = []
    for sym in symbols:
        sym = sym.strip()
        s, up_stats, dn_stats = analyze_symbol(mode, sym, timeframe, start, end, lookback_days, max_candles,
                                               rsi_len, stoch_len, k_len, d_len, zone,
                                               adx_len, adx_thresh, adx_require_rising,
                                               macd_fast, macd_slow, macd_signal_len, macd_zero_filter)
        results.append((s, up_stats, dn_stats))

    # Print table
    print("="*108)
    print(f"Mode={mode.upper()} | Coinbase | TF={timeframe} | Symbols={', '.join(symbols)}")
    if start or end:
        print(f"Window: {start or '...'} → {end or '...'} (UTC)")
    elif lookback_days:
        print(f"Lookback: last {lookback_days} day(s)")
    if mode == "stochrsi":
        print(f"StochRSI params: rsi_len={rsi_len}, stoch_len={stoch_len}, k_len={k_len}, d_len={d_len} | Zone={zone}")
    elif mode == "adx":
        print(f"ADX params: adx_len={adx_len}, adx_thresh={adx_thresh}, require_rising={adx_require_rising}")
    else:
        print(f"MACD params: fast={macd_fast}, slow={macd_slow}, signal={macd_signal_len}, zero_filter={macd_zero_filter}")
    print("="*108)
    header = f"{'Symbol':10} | {'UP Cnt':6} {'UP Median%':10} {'UP Mean%':10} {'UP IQR%':26} || {'DN Cnt':6} {'DN Median%':10} {'DN Mean%':10} {'DN IQR%':26}"
    print(header)
    print("-"*108)

    rows_for_csv = []
    all_up = []
    all_dn = []

    for s, up, dn in results:
        if up is None or dn is None:
            line = f"{s:10} | {'NA':6} {'NA':10} {'NA':10} {'NA':26} || {'NA':6} {'NA':10} {'NA':10} {'NA':26}"
            print(line)
            continue
        ucnt, umed, umean, u_iqr = up["count"], up["median"], up["mean"], up["iqr"]
        dcnt, dmed, dmean, d_iqr = dn["count"], dn["median"], dn["mean"], dn["iqr"]
        line = f"{s:10} | {ucnt:6d} {umed:10.4f} {umean:10.4f} {str((round(u_iqr[0],4), round(u_iqr[1],4))):26} || {dcnt:6d} {dmed:10.4f} {dmean:10.4f} {str((round(d_iqr[0],4), round(d_iqr[1],4))):26}"
        print(line)

        rows_for_csv.append({
            "mode": mode,
            "symbol": s,
            "up_count": ucnt, "up_median_pct": umed, "up_mean_pct": umean, "up_iqr_low_pct": u_iqr[0], "up_iqr_high_pct": u_iqr[1],
            "down_count": dcnt, "down_median_pct": dmed, "down_mean_pct": dmean, "down_iqr_low_pct": d_iqr[0], "down_iqr_high_pct": d_iqr[1],
            "timeframe": timeframe,
            "zone": zone if mode=="stochrsi" else None,
            "adx_len": adx_len if mode=="adx" else None,
            "adx_thresh": adx_thresh if mode=="adx" else None,
            "adx_require_rising": adx_require_rising if mode=="adx" else None,
            "start": start, "end": end, "lookback_days": lookback_days
        })

        if not np.isnan(umed): all_up.append(umed)
        if not np.isnan(dmed): all_dn.append(dmed)

    # Overall median of medians
    if all_up:
        print("-"*108)
        print(f"OVERALL  | UP median-of-medians: {np.median(all_up):.4f}%  (across {len(all_up)} symbols)")
    if all_dn:
        print(f"OVERALL  | DN median-of-medians: {np.median(all_dn):.4f}%  (across {len(all_dn)} symbols)")
    print("="*108)

    if csv_path:
        df_csv = pd.DataFrame(rows_for_csv)
        df_csv.to_csv(csv_path, index=False)
        print(f"Saved CSV → {csv_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Median price change while an indicator's condition holds (Coinbase, chunked + multi-symbol).")
    p.add_argument("--mode", type=str, choices=["stochrsi","adx","macd"], required=True, help="Which indicator mode to run")
    p.add_argument("--exchange", type=str, choices=["coinbase"], required=True, help="Only coinbase is supported")
    p.add_argument("--symbols", type=str, required=True, help="Comma-separated Coinbase product IDs, e.g., BTC-USD,ETH-USD,XRP-USD")
    p.add_argument("--timeframe", type=str, choices=list(GRAN_MAP.keys()), required=True, help="1m,5m,15m,1h,6h,1d")
    p.add_argument("--start", type=str, default=None, help="e.g., 2024-01-01 (UTC)")
    p.add_argument("--end", type=str, default=None, help="e.g., 2024-12-31 (UTC)")
    p.add_argument("--lookback_days", type=int, default=180, help="Used if start not provided")
    p.add_argument("--max_candles", type=int, default=10000, help="Max candles per symbol")

    # StochRSI params
    p.add_argument("--rsi_len", type=int, default=14)
    p.add_argument("--stoch_len", type=int, default=14)
    p.add_argument("--k_len", type=int, default=3)
    p.add_argument("--d_len", type=int, default=3)
    p.add_argument("--zone", action="store_true", help="(stochrsi) Require K<20 for up-segment start, K>80 for down-segment start")

    # ADX params
    p.add_argument("--adx_len", type=int, default=14, help="ADX/DI smoothing length")
    p.add_argument("--adx_thresh", type=float, default=20.0, help="Minimum ADX to qualify a trend segment")
    p.add_argument("--adx_require_rising", action="store_true", help="Require ADX to be rising at segment start")

    # MACD params
    p.add_argument("--macd_fast", type=int, default=12)
    p.add_argument("--macd_slow", type=int, default=26)
    p.add_argument("--macd_signal", dest="macd_signal_len", type=int, default=9)
    p.add_argument("--macd_zero_filter", action="store_true", help="Require MACD > 0 for up starts and < 0 for down starts")

    p.add_argument("--csv", type=str, default=None, help="Optional CSV path to save results")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    run_multi(
        mode=args.mode,
        symbols=symbols,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        lookback_days=args.lookback_days,
        max_candles=args.max_candles,
        rsi_len=args.rsi_len,
        stoch_len=args.stoch_len,
        k_len=args.k_len,
        d_len=args.d_len,
        zone=args.zone,
        adx_len=args.adx_len,
        adx_thresh=args.adx_thresh,
        adx_require_rising=args.adx_require_rising,
        macd_fast=args.macd_fast,
        macd_slow=args.macd_slow,
        macd_signal_len=args.macd_signal_len,
        macd_zero_filter=args.macd_zero_filter,
        csv_path=args.csv
    )
