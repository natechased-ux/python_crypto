import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# Parameters
coins = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
         "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
         "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd", "avax-usd", "xcn-usd", "uni-usd",
         "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
         "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd", "qnt-usd", "tia-usd", "ip-usd",
         "pnut-usd", "apt-usd", "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "pyth-usd", "sand-usd",
         "morpho-usd", "mana-usd", "coti-usd", "c98-usd", "axs-usd"]
ema_periods = [10,20, 50,100, 200]
granularity = 86400  # 1 day in seconds
proximity_pct = 0.0025  # 0.25%
tp_pct = 0.015          # 1.5%
sl_pct = 0.01           # 1.0%
lookback_days = 300     # Need ~100 to properly test EMA 200

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

# Run the test
results = []
for coin in coins:
    print(f"Testing {coin} on 1D candles")
    df = fetch_candles(coin, granularity, lookback_days)
    if df is None or len(df) < max(ema_periods) + 10:
        print(f"Not enough data for {coin}")
        continue
    for period in ema_periods:
        signals = evaluate_ema_bounces(df.copy(), period)
        if signals:
            total = len(signals)
            wins = signals.count("win")
            losses = signals.count("loss")
            results.append({
                "Coin": coin,
                "EMA": period,
                "Tests": total,
                "Win %": round(wins / total * 100, 2),
                "Loss %": round(losses / total * 100, 2),
                "No Result": signals.count("no_result")
            })

# Output
df_results = pd.DataFrame(results)
if df_results.empty:
    print("No EMA bounce setups detected.")
else:
    print(df_results.to_string(index=False))
    df_results.to_csv("ema_bounce_1d_results.csv", index=False)
