#!/usr/bin/env python3

import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import time

# ============================================================
# CONFIG
# ============================================================

COINBASE_API = "https://api.exchange.coinbase.com"

COIN_SYMBOL = "IP-USD"     # <--- CHANGE ME
DAYS_BACK = 90
RSI_PERIOD = 14


# ============================================================
# DATA FETCHING
# ============================================================

def fetch_candles(symbol, granularity, start, end):
    url = f"{COINBASE_API}/products/{symbol}/candles"
    params = {
        "granularity": granularity,
        "start": start.isoformat(),
        "end": end.isoformat()
    }
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        print(f"Fetch error for {symbol}: {r.text}")
        return []
    return r.json()


def download_history(symbol, granularity, days_back=90):
    all_rows = []
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days_back)

    print(f"Downloading {symbol} ({granularity}s) from {start_time} to {end_time}")

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
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame(columns=["time","low","high","open","close","volume"])

    df = pd.DataFrame(all_rows, columns=["time","low","high","open","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)
    return df


def resample_4h(df):
    if df.empty:
        return df
    df = df.set_index("time")
    out = pd.DataFrame()
    out["open"] = df["open"].resample("4H").first()
    out["high"] = df["high"].resample("4H").max()
    out["low"] = df["low"].resample("4H").min()
    out["close"] = df["close"].resample("4H").last()
    out["volume"] = df["volume"].resample("4H").sum()
    out.dropna(inplace=True)
    return out.reset_index()


# ============================================================
# INDICATORS
# ============================================================

def RSI(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def EMA(series, n):
    return series.ewm(span=n, adjust=False).mean()


# ============================================================
# LONG STRATEGY (FINAL VERSION)
# ============================================================

def long_signal(df_coin, df_btc_4h):
    """
    FINAL LONG ENTRY RULES:

    1. RSI(15m) crosses UP through 30–40
    2. Price reclaims MA25
    3. Volume > SMA(volume, 20)
    4. Trend filter: EMA25 > EMA50 > EMA75
    5. BTC 4h RSI > 50
    """

    if len(df_coin) < 80:
        return False

    prev = df_coin.iloc[-2]
    curr = df_coin.iloc[-1]
    recent = df_coin["rsi"].iloc[-8:]     # last ~5 candles
    dipped_oversold = (recent < 30).any()

    rsi_recovery = (curr["rsi"] > 30) and (curr["rsi"] < 55)

    if not (dipped_oversold and rsi_recovery):
        return False


    # 2. Price reclaims MA25
    if curr["close"] <= curr["ma25"]:
        return False

    # 3. Volume confirmation
    vol_sma20 = df_coin["volume"].rolling(20).mean().iloc[-1]
    if curr["volume"] <= vol_sma20 or np.isnan(vol_sma20):
        return False

    # 4. Trend filter: EMA stack
    if not (curr["ema25"] > curr["ema50"] > curr["ema75"]):
        return False

    # 5. BTC macro filter
    btc_rsi = df_btc_4h["rsi"].iloc[-1]
    if np.isnan(btc_rsi) or btc_rsi <= 50:
        return False

    return True


# ============================================================
# SHORT STRATEGY (unchanged)
# ============================================================

def short_signal(df_coin, df_btc_15m, df_btc_4h):

    if len(df_coin) < 100:
        return False

    prev = df_coin.iloc[-2]
    curr = df_coin.iloc[-1]

    cross_down = (prev["ma7"] >= prev["ma99"]) and (curr["ma7"] < curr["ma99"])
    if not cross_down:
        return False

    if curr["rsi"] >= 50:
        return False
    if df_btc_15m["rsi"].iloc[-1] >= 50:
        return False
    if df_btc_4h["rsi"].iloc[-1] >= 50:
        return False

    return True


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(df_coin, df_btc_15m, df_btc_4h):

    trades = []

    for i in range(100, len(df_coin) - 1):

        current_time = df_coin["time"].iloc[i]
        coin_slice = df_coin.iloc[:i+1]

        btc15_slice = df_btc_15m[df_btc_15m["time"] <= current_time]
        btc4h_slice = df_btc_4h[df_btc_4h["time"] <= current_time]

        if len(btc15_slice) < 50 or len(btc4h_slice) < 10:
            continue

        is_long = long_signal(coin_slice, btc4h_slice)
        is_short = short_signal(coin_slice, btc15_slice, btc4h_slice)

        if not (is_long or is_short):
            continue

        entry_price = df_coin["close"].iloc[i]
        direction = 1 if is_long else -1

        record = {
            "type": "long" if is_long else "short",
            "entry_time": current_time,
            "entry_price": entry_price,
        }

        # Capture 16 future steps
        for step in range(1, 17):
            idx = i + step
            minutes = step * 15
            key = f"chg_{minutes}m"

            if idx >= len(df_coin):
                record[key] = np.nan
            else:
                future_price = df_coin["close"].iloc[idx]
                pct = (future_price - entry_price) / entry_price * direction
                record[key] = pct

        trades.append(record)

    return pd.DataFrame(trades)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n=== DOWNLOADING DATA ===\n")

    df_coin = download_history(COIN_SYMBOL, 900, DAYS_BACK)
    df_btc_15m = download_history("BTC-USD", 900, DAYS_BACK)
    df_btc_1h  = download_history("BTC-USD", 3600, DAYS_BACK)

    # Coin indicators
    df_coin["ma7"]  = df_coin["close"].rolling(7).mean()
    df_coin["ma25"] = df_coin["close"].rolling(25).mean()
    df_coin["ma99"] = df_coin["close"].rolling(99).mean()
    df_coin["ema25"] = EMA(df_coin["close"], 25)
    df_coin["ema50"] = EMA(df_coin["close"], 50)
    df_coin["ema75"] = EMA(df_coin["close"], 75)
    df_coin["rsi"]  = RSI(df_coin, RSI_PERIOD)

    # BTC indicators
    df_btc_15m["rsi"] = RSI(df_btc_15m, RSI_PERIOD)
    df_btc_1h["rsi"]  = RSI(df_btc_1h, RSI_PERIOD)
    df_btc_4h = resample_4h(df_btc_1h)
    df_btc_4h["rsi"] = RSI(df_btc_4h, RSI_PERIOD)

    print("\n=== RUNNING BACKTEST ===\n")

    results = run_backtest(df_coin, df_btc_15m, df_btc_4h)

    print(f"Total trades found: {len(results)}\n")
    print(results.head())

    results.to_csv(f"{COIN_SYMBOL}_15m_path_backtest.csv", index=False)
    print(f"\nSaved to {COIN_SYMBOL}_15m_path_backtest.csv")
