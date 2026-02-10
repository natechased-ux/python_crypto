#!/usr/bin/env python3

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# ============================================================
# CONFIG
# ============================================================

COINBASE_API = "https://api.exchange.coinbase.com"

DAYS_BACK = 90
GRANULARITY = 3600   # 1-hour candles; we will resample to 4H

# Exclude these as BASE currencies for USD pairs
EXCLUDED_BASES = {"USDT", "USDC", "DAI", "EUROC", "PYUSD"}

# EMA / MA periods must match Nih.py
EMA_PERIODS = [25, 50, 75, 150, 200]
MA7 = 7
MA25 = 25
MA99 = 99

EMA_NEAR_PCT = 0.05
PRICE_ABOVE_EMA_REQUIRED = True

# Simple rate limiting
REQUEST_SLEEP = 0.15

# ============================================================
# FETCH PRODUCT LIST
# ============================================================

def get_all_symbols():
    """
    All USD-quoted symbols, excluding stablecoin bases.
    Modeled after cross_back_all.get_all_symbols, but for 1h/4h strat.
    """
    products = requests.get(f"{COINBASE_API}/products", timeout=10).json()
    symbols = [
        p["id"]
        for p in products
        if p.get("quote_currency") == "USD"
        and p["base_currency"].upper() not in EXCLUDED_BASES
    ]
    return sorted(symbols)


# ============================================================
# FETCH HISTORICAL DATA (1H)
# ============================================================

def fetch_candles(symbol, granularity, start, end):
    url = f"{COINBASE_API}/products/{symbol}/candles"
    params = {
        "granularity": granularity,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return []
    return r.json()


def download_history(symbol, granularity, days_back=90):
    """
    Same paging pattern as cross_back_all.download_history,
    but for 1h candles.
    """
    all_rows = []
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days_back)

    current_end = end_time
    while current_end > start_time:
        current_start = current_end - timedelta(seconds=300 * granularity)
        if current_start < start_time:
            current_start = start_time

        rows = fetch_candles(symbol, granularity, current_start, current_end)
        if not rows:
            break

        all_rows.extend(rows)
        current_end = current_start
        time.sleep(REQUEST_SLEEP)

    if not all_rows:
        return pd.DataFrame(columns=["time", "low", "high", "open", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ============================================================
# RESAMPLE 1H -> 4H
# ============================================================

def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Same resampling logic as in Nih.py (resample_to_4h).
    """
    if df_1h.empty:
        return df_1h.copy()

    df = df_1h.set_index("time")
    df4 = pd.DataFrame()
    df4["open"]   = df["open"].resample("4H").first()
    df4["high"]   = df["high"].resample("4H").max()
    df4["low"]    = df["low"].resample("4H").min()
    df4["close"]  = df["close"].resample("4H").last()
    df4["volume"] = df["volume"].resample("4H").sum()
    df4.dropna(inplace=True)
    df4 = df4.reset_index()
    return df4


# ============================================================
# INDICATORS
# ============================================================

def EMA(series, n):
    return series.ewm(span=n, adjust=False).mean()


def add_ema_ma_columns(df: pd.DataFrame) -> pd.DataFrame:
    for p in EMA_PERIODS:
        df[f"EMA{p}"] = EMA(df["close"], p)

    df[f"MA{MA7}"]  = df["close"].rolling(MA7,  min_periods=1).mean()
    df[f"MA{MA25}"] = df["close"].rolling(MA25, min_periods=1).mean()
    df[f"MA{MA99}"] = df["close"].rolling(MA99, min_periods=1).mean()
    return df


# ============================================================
# SIGNAL LOGIC (ported from Nih.py + mirrored shorts)
# ============================================================

def ema_trend_support(df: pd.DataFrame):
    """
    Direct port of check_ema_trend_support from Nih.py, but returns just bool.
    In Nih.py it checks:
      - EMAs 25,50,75,150,200 in bullish order: EMA25 > EMA50 > EMA75 > EMA150 > EMA200
      - It *constructs* near_support / between_25_50 but the final condition
        only requires the EMA stack to be bullish.
    """
    if df.empty:
        return False

    last = df.iloc[-1]
    keys = [f"EMA{p}" for p in EMA_PERIODS]
    if any(k not in df.columns for k in keys):
        return False

    emavalues = [last[k] for k in keys]
    bullish = all(emavalues[i] > emavalues[i+1] for i in range(len(emavalues)-1))
    if not bullish:
        return False

    price = last["close"]
    ema25 = last["EMA25"]
    if ema25 and ema25 > 0:
        pct_from_ema25 = (price - ema25) / ema25
    else:
        pct_from_ema25 = 0.0

    if PRICE_ABOVE_EMA_REQUIRED and price < ema25:
        return False

    if abs(pct_from_ema25) > EMA_NEAR_PCT:
        return False

    return True


def ema_trend_resistance(df: pd.DataFrame):
    """
    Mirrored bearish version of ema_trend_support:
      - EMA25 < EMA50 < EMA75 < EMA150 < EMA200
      - price near EMA25 (within EMA_NEAR_PCT), optionally below EMA25.
    """
    if df.empty:
        return False

    last = df.iloc[-1]
    keys = [f"EMA{p}" for p in EMA_PERIODS]
    if any(k not in df.columns for k in keys):
        return False

    emavalues = [last[k] for k in keys]
    bearish = all(emavalues[i] < emavalues[i+1] for i in range(len(emavalues)-1))
    if not bearish:
        return False

    price = last["close"]
    ema25 = last["EMA25"]
    if ema25 and ema25 > 0:
        pct_from_ema25 = (price - ema25) / ema25
    else:
        pct_from_ema25 = 0.0

    # For shorts, we like price near/under EMA25
    if price > ema25:
        return False
    if abs(pct_from_ema25) > EMA_NEAR_PCT:
        return False

    return True


def ma_cross_up(df: pd.DataFrame):
    """
    Port of check_ma_cross from Nih.py:
      - MA7 crosses UP above MA99.
    """
    if f"MA{MA7}" not in df.columns or f"MA{MA99}" not in df.columns:
        return False
    if len(df) < 2:
        return False

    prev = df.iloc[-2]
    last = df.iloc[-1]
    prev7 = prev[f"MA{MA7}"]; prev99 = prev[f"MA{MA99}"]
    last7 = last[f"MA{MA7}"]; last99 = last[f"MA{MA99}"]

    crossed_up = (prev7 <= prev99) and (last7 > last99)
    return bool(crossed_up)


def ma_cross_down(df: pd.DataFrame):
    """
    Mirrored short version:
      - MA7 crosses DOWN below MA99.
    """
    if f"MA{MA7}" not in df.columns or f"MA{MA99}" not in df.columns:
        return False
    if len(df) < 2:
        return False

    prev = df.iloc[-2]
    last = df.iloc[-1]
    prev7 = prev[f"MA{MA7}"]; prev99 = prev[f"MA{MA99}"]
    last7 = last[f"MA{MA7}"]; last99 = last[f"MA{MA99}"]

    crossed_down = (prev7 >= prev99) and (last7 < last99)
    return bool(crossed_down)


def long_signal(df: pd.DataFrame) -> bool:
    """
    Live LONG logic = EMA trend support + MA7 cross up over MA99.
    """
    return ema_trend_support(df) and ma_cross_up(df)


def short_signal(df: pd.DataFrame) -> bool:
    """
    Backtest SHORT as full mirror: EMA trend resistance + MA7 cross down.
    """
    return ema_trend_resistance(df) and ma_cross_down(df)


# ============================================================
# BACKTEST
# ============================================================

def run_backtest_4h(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    For each 4h candle, if long/short signal fires, record:
      - entry_time, entry_price
      - forward returns every 4h up to 24h => 6 columns:
          chg_4h, chg_8h, ..., chg_24h
    """
    trades = []

    # Need enough data for EMAs and MA99 & 6 future candles
    for i in range(200, len(df_4h) - 6):
        slice_df = df_4h.iloc[:i+1]

        is_long = long_signal(slice_df)
        is_short = short_signal(slice_df)

        if not (is_long or is_short):
            continue

        entry_time = df_4h["time"].iloc[i]
        entry_price = df_4h["close"].iloc[i]
        direction = 1 if is_long else -1

        record = {
            "type": "long" if is_long else "short",
            "entry_time": entry_time,
            "entry_price": entry_price,
        }

        # 6 forward 4h steps: +4h, +8h, ..., +24h
        for step in range(1, 7):
            idx = i + step
            key = f"chg_{step*4}h"

            future_price = df_4h["close"].iloc[idx]
            pct = (future_price - entry_price) / entry_price * direction
            record[key] = pct

        trades.append(record)

    return pd.DataFrame(trades)


# ============================================================
# MASTER EXECUTION
# ============================================================

if __name__ == "__main__":

    all_symbols = get_all_symbols()
    print(f"\n=== Testing {len(all_symbols)} coins (USD, non-stable bases) ===\n")
    print(all_symbols)

    summary_rows = []

    for symbol in all_symbols:
        print(f"\n=== Processing {symbol} ===")

        # 1h history
        df_1h = download_history(symbol, GRANULARITY, DAYS_BACK)

        if len(df_1h) < 400:
            print("Too little data, skipping.")
            continue

        # 1h -> 4h
        df_4h = resample_to_4h(df_1h)

        if len(df_4h) < 250:
            print("Too few 4h candles, skipping.")
            continue

        df_4h = add_ema_ma_columns(df_4h)

        results = run_backtest_4h(df_4h)
        results.to_csv(f"{symbol}_4h_msb_cross_backtest.csv", index=False)

        if len(results) == 0:
            print("No signals for this symbol.")
            continue

        longs  = results[results["type"] == "long"]
        shorts = results[results["type"] == "short"]

        def expectancy_24h(df):
            if "chg_24h" in df.columns and len(df) > 0:
                return df["chg_24h"].mean()
            return np.nan

        summary_rows.append({
            "symbol": symbol,
            "long_trades": len(longs),
            "long_expect_24h": expectancy_24h(longs),
            "short_trades": len(shorts),
            "short_expect_24h": expectancy_24h(shorts),
        })

    # Global summary like cross_back_all
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.to_csv("ALL_COIN_SUMMARY_4H_MSB_CROSS.csv", index=False)

        print("\n\n=== COMPLETE ===")
        print("\nTop 20 by SHORT 24h expectancy:")
        print(summary.sort_values("short_expect_24h", ascending=False).head(20))

        print("\nTop 20 by LONG 24h expectancy:")
        print(summary.sort_values("long_expect_24h", ascending=False).head(20))
    else:
        print("\n\n=== COMPLETE: No trades generated for any symbol ===")
