
import pandas as pd
import requests
import numpy as np
from datetime import datetime

COINS = [
    "btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd", "wif-usd", "ondo-usd"
]
INTERVALS = {'4H': 14400, '1H': 3600, '1D': 86400}
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

def run_diagnostics():
    for coin in COINS:
        try:
            df = get_candles(coin.upper(), INTERVALS['1H'])
            if df.empty or len(df) < 50:
                print(f"{coin.upper()} - Not enough data")
                continue

            df['macd'], df['signal'] = macd(df)
            df['atr'] = atr(df)

            print(f"🧪 {coin.upper()} - Checking MACD crossovers...")

            for i in range(35, len(df) - 13):  # 35 for warm-up
                macd_prev = df['macd'].iloc[i-1]
                macd_now = df['macd'].iloc[i]
                signal_prev = df['signal'].iloc[i-1]
                signal_now = df['signal'].iloc[i]
                date = df['time'].iloc[i]

                # Long crossover
                if macd_prev < signal_prev and macd_now > signal_now:
                    print(f"{coin.upper()} | {date} | MACD Bullish Crossover")

                # Short crossover
                if macd_prev > signal_prev and macd_now < signal_now:
                    print(f"{coin.upper()} | {date} | MACD Bearish Crossover")

        except Exception as e:
            print(f"Error on {coin.upper()}: {e}")

if __name__ == "__main__":
    run_diagnostics()
