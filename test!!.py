import pandas as pd
import numpy as np
import os
import requests
import time
from datetime import datetime, timedelta
from ta.momentum import StochRSIIndicator
from ta.trend import MACD
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator

# ------------------------
# CONFIGURATION
# ------------------------

COINS = ["btc-usd", "eth-usd", "xrp-usd"]  # Add more
START_DATE = "2024-06-01"
END_DATE = "2024-07-20"
RISK_PER_TRADE = 100  # Dollars per trade

# ------------------------
# DATA FETCHING
# ------------------------

import time

def fetch_candles(symbol, granularity_seconds, start, end, max_retries=5):
    url = f"https://api.pro.coinbase.com/products/{symbol}/candles"
    df_all = []

    max_candles = 300
    delta = timedelta(seconds=granularity_seconds * max_candles)

    while start < end:
        end_segment = min(start + delta, end)
        params = {
            "start": start.isoformat(),
            "end": end_segment.isoformat(),
            "granularity": granularity_seconds
        }

        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    if not data:
                        break
                    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    df_all.append(df)
                    break
                else:
                    print(f"⚠️ {symbol} fetch failed ({resp.status_code}), retrying in 5s...")
                    time.sleep(5)
            except Exception as e:
                print(f"⚠️ {symbol} exception: {e}, retrying in 5s...")
                time.sleep(5)

        start = end_segment

    if not df_all:
        print(f"❌ No data returned for {symbol}")
        return None
    df = pd.concat(df_all).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df



# ------------------------
# ENTRY LOGIC
# ------------------------

def check_entry_signal(df_6h):
    macd = MACD(df_6h["close"])
    adx = ADXIndicator(df_6h["high"], df_6h["low"], df_6h["close"])
    rsi = RSIIndicator(df_6h["close"])

    df_6h["macd_diff"] = macd.macd_diff()
    df_6h["adx"] = adx.adx()
    df_6h["rsi"] = rsi.rsi()

    signals = []
    for i in range(2, len(df_6h)):
        row = df_6h.iloc[i]
        if row["macd_diff"] > 0 and row["adx"] > 20 and row["rsi"] > 55:
            signals.append(("long", row["time"], row["close"]))
        elif row["macd_diff"] < 0 and row["adx"] > 20 and row["rsi"] < 45:
            signals.append(("short", row["time"], row["close"]))
    return signals

# ------------------------
# STOCH RSI CONFIRMATION
# ------------------------

def stoch_rsi_confirm(df_1h, signal_time, direction):
    df = df_1h[df_1h["time"] > signal_time].head(6)  # Check next 6 candles (6 hours)
    if len(df) < 2:
        return None
    stoch = StochRSIIndicator(df["close"])
    k = stoch.stochrsi_k().dropna()
    d = stoch.stochrsi_d().dropna()
    if len(k) < 2 or len(d) < 2:
        return None
    for i in range(1, len(k)):
        if direction == "long" and k.iloc[i] > d.iloc[i] and k.iloc[i] < 40:
            return df["close"].iloc[i], df["time"].iloc[i]
        elif direction == "short" and k.iloc[i] < d.iloc[i] and k.iloc[i] > 60:
            return df["close"].iloc[i], df["time"].iloc[i]
    return None

# ------------------------
# TRADE LOGGER
# ------------------------

def log_trade(trades, coin, side, entry_time, entry_price, confirm_time, confirm_price, sl, tp, df_prices):
    exit_price = None
    outcome = "open"
    mfe = 0
    high_price = df_prices["high"].max()
    low_price = df_prices["low"].min()

    if side == "long":
        if low_price <= sl:
            exit_price = sl
            outcome = "loss"
        elif high_price >= tp:
            exit_price = tp
            outcome = "win"
        mfe = (high_price - entry_price) / (tp - sl)
    elif side == "short":
        if high_price >= sl:
            exit_price = sl
            outcome = "loss"
        elif low_price <= tp:
            exit_price = tp
            outcome = "win"
        mfe = (entry_price - low_price) / (sl - tp)

    # Weekend logic
    is_weekend = (
        confirm_time.weekday() == 5 or confirm_time.weekday() == 6 or
        (confirm_time.weekday() == 4 and confirm_time.hour >= 17)
    )

    trades.append({
        "symbol": coin,
        "side": side,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "confirm_time": confirm_time,
        "confirm_price": confirm_price,
        "tp": tp,
        "sl": sl,
        "exit_price": exit_price,
        "outcome": outcome,
        "mfe": round(mfe, 2),
        "weekend": is_weekend,
        "high_between": high_price,
        "low_between": low_price
    })

# ------------------------
# MAIN BACKTEST LOOP
# ------------------------

def run_backtest():
    trades = []
    for coin in COINS:
        print(f"🔄 Backtesting {coin.upper()}...")
        df_6h = fetch_candles(coin, 21600, datetime.fromisoformat(START_DATE), datetime.fromisoformat(END_DATE))
        df_1h = fetch_candles(coin, 3600, datetime.fromisoformat(START_DATE), datetime.fromisoformat(END_DATE))
        if df_6h is None or df_1h is None:
            continue

        signals = check_entry_signal(df_6h)

        for side, signal_time, signal_price in signals:
            conf = stoch_rsi_confirm(df_1h, signal_time, side)
            if conf is None:
                continue
            confirmed_price, confirm_time = conf

            # Set SL/TP: risk 1x below swing, TP 0.75x reward
            risk = 0.02 * confirmed_price
            if side == "long":
                sl = confirmed_price - risk
                tp = confirmed_price + 0.75 * risk
            else:
                sl = confirmed_price + risk
                tp = confirmed_price - 0.75 * risk

            future_data = df_1h[(df_1h["time"] > confirm_time) & (df_1h["time"] <= confirm_time + timedelta(hours=24))]
            if len(future_data) == 0:
                continue

            log_trade(trades, coin, side, signal_time, signal_price, confirm_time, confirmed_price, sl, tp, future_data)

    df = pd.DataFrame(trades)
    df.to_csv("backtest_results.csv", index=False)
    print(f"✅ Backtest complete. {len(df)} trades saved to backtest_results.csv")

# ------------------------
# RUN
# ------------------------

if __name__ == "__main__":
    run_backtest()
