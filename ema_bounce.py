import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# Settings
coins = [
    "btc-usd", "eth-usd", "xrp-usd", "doge-usd", "sol-usd", "ada-usd", "link-usd", "ltc-usd",
    "uni-usd", "aave-usd", "snx-usd", "rndr-usd", "arb-usd", "op-usd", "sui-usd", "fxs-usd",
    "lrc-usd", "ens-usd", "rune-usd", "mina-usd"
]
ema_periods = [10, 20, 50, 100, 200]
proximity_pct = 0.0025  # Within 0.25% of EMA
tp_pct = 0.015
sl_pct = 0.01
lookback_days = 30
granularities = {'1H': 3600, '6H': 21600}

def fetch_candles(symbol, granularity, days):
    end = int(time.time())
    start = end - days * 86400
    df_all = []

    while start < end:
        params = {
            'start': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start)),
            'end': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(min(start + 300 * granularity, end))),
            'granularity': granularity
        }
        url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                break
            data = response.json()
            if not data:
                break
            df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
            df_all.append(df)
            start += 300 * granularity
            time.sleep(0.2)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break

    if not df_all:
        return None

    df = pd.concat(df_all, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def evaluate_ema_bounces(df, ema_period):
    df[f'EMA_{ema_period}'] = df['close'].ewm(span=ema_period).mean()
    signals = []

    for i in range(ema_period + 1, len(df) - 10):
        price = df.loc[i, 'close']
        ema = df.loc[i, f'EMA_{ema_period}']
        if abs(price - ema) / ema <= proximity_pct:
            entry = price
            tp_level = entry * (1 + tp_pct)
            sl_level = entry * (1 - sl_pct)
            for j in range(i + 1, i + 10):
                high = df.loc[j, 'high']
                low = df.loc[j, 'low']
                if high >= tp_level:
                    signals.append("win")
                    break
                elif low <= sl_level:
                    signals.append("loss")
                    break
            else:
                signals.append("no_result")
    return signals

# Run test
results = []
for tf_name, granularity in granularities.items():
    for coin in coins:
        print(f"Testing {coin} @ {tf_name}")
        df = fetch_candles(coin, granularity, lookback_days)
        if df is None or len(df) < max(ema_periods) + 10:
            continue
        for period in ema_periods:
            signals = evaluate_ema_bounces(df.copy(), period)
            if signals:
                total = len(signals)
                wins = signals.count("win")
                losses = signals.count("loss")
                results.append({
                    "Coin": coin,
                    "Timeframe": tf_name,
                    "EMA": period,
                    "Tests": total,
                    "Win %": round(wins / total * 100, 2),
                    "Loss %": round(losses / total * 100, 2),
                    "No Result": signals.count("no_result")
                })

# Output
df_results = pd.DataFrame(results)
df_results.sort_values(["Timeframe", "EMA", "Win %"], ascending=[True, True, False], inplace=True)
print(df_results.to_string(index=False))
df_results.to_csv("ema_bounce_results.csv", index=False)
