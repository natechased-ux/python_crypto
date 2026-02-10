
import pandas as pd
import requests
import numpy as np
from datetime import datetime

COINS = [
    "btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd", "wif-usd", "ondo-usd"
]
INTERVALS = {'1H': 3600}
BASE_URL = "https://api.exchange.coinbase.com/products"

def get_candles(symbol, granularity, limit=300):
    url = f"{BASE_URL}/{symbol}/candles?granularity={granularity}"
    response = requests.get(url)
    if response.status_code != 200:
        return pd.DataFrame()
    data = response.json()
    df = pd.DataFrame(data, columns=['time','low','high','open','close','volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df.sort_values('time')

def macd(df, fast=12, slow=26, signal=9):
    fast_ema = df['close'].ewm(span=fast, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def backtest():
    results = []

    for coin in COINS:
        try:
            df = get_candles(coin.upper(), INTERVALS['1H'])
            if df.empty or len(df) < 50:
                print(f"{coin.upper()} - Not enough data")
                continue

            df['macd'], df['signal'] = macd(df)
            df['atr'] = atr(df)

            trades, wins, losses, profit = 0, 0, 0, 0

            for i in range(35, len(df) - 13):  # 13 candles = about 13 hours
                price = df['close'].iloc[i]
                macd_prev = df['macd'].iloc[i-1]
                macd_now = df['macd'].iloc[i]
                signal_prev = df['signal'].iloc[i-1]
                signal_now = df['signal'].iloc[i]
                atr_val = df['atr'].iloc[i]

                if np.isnan(macd_now) or np.isnan(signal_now) or np.isnan(atr_val) or atr_val == 0:
                    continue

                # Long setup
                if macd_prev < signal_prev and macd_now > signal_now:
                    entry = price
                    sl = entry - 1.5 * atr_val
                    tp = entry + 3 * atr_val
                    trades += 1
                    future = df['close'].iloc[i+1:i+13]
                    hit_tp = any(p >= tp for p in future)
                    hit_sl = any(p <= sl for p in future)
                    if hit_tp and not hit_sl:
                        wins += 1
                        profit += 2500 * ((tp - entry) / entry)
                    elif hit_sl and not hit_tp:
                        losses += 1
                        profit += 2500 * ((sl - entry) / entry)
                    else:
                        trades -= 1

                # Short setup
                if macd_prev > signal_prev and macd_now < signal_now:
                    entry = price
                    sl = entry + 1.5 * atr_val
                    tp = entry - 3 * atr_val
                    trades += 1
                    future = df['close'].iloc[i+1:i+13]
                    hit_tp = any(p <= tp for p in future)
                    hit_sl = any(p >= sl for p in future)
                    if hit_tp and not hit_sl:
                        wins += 1
                        profit += 2500 * ((entry - tp) / entry)
                    elif hit_sl and not hit_tp:
                        losses += 1
                        profit += 2500 * ((entry - sl) / entry)
                    else:
                        trades -= 1

            if trades > 0:
                results.append({
                    "symbol": coin.upper(),
                    "trades": trades,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round(100 * wins / trades, 2),
                    "net_profit": round(profit, 2)
                })

        except Exception as e:
            print(f"Error with {coin.upper()}: {e}")

    df = pd.DataFrame(results)
    df.to_csv("swing_trade_backtest_1H.csv", index=False)
    print(df)

if __name__ == "__main__":
    backtest()
