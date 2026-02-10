import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from telegram import Bot

# Telegram configuration
BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"

# List of cryptocurrencies to monitor (use Coinbase ticker symbols)
CRYPTO_LIST = ["BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD"]

# Stochastic RSI calculation

def calculate_stochastic_rsi(data, period=14):
    delta = data["close"].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi_min = rsi.rolling(window=period).min()
    rsi_max = rsi.rolling(window=period).max()

    stochastic_rsi = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)

    k = stochastic_rsi.rolling(window=3).mean()
    d = k.rolling(window=3).mean()

    return k, d

# Fetch daily data from Coinbase API
def fetch_data(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity=86400"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.sort_values("time")
        return df
    else:
        print(f"Failed to fetch data for {symbol}: {response.status_code}")
        return None

# Send Telegram alert
def send_telegram_message(message):
    bot = Bot(token=BOT_TOKEN)
    bot.send_message(chat_id=CHAT_ID, text=message)

# Main function
def monitor_signals():
    while True:
        for symbol in CRYPTO_LIST:
            data = fetch_data(symbol)
            if data is not None:
                k, d = calculate_stochastic_rsi(data)

                # Check conditions for buy and sell signals
                if k.iloc[-1] < 20 and d.iloc[-1] < 20 and d.iloc[-2] <= k.iloc[-2] and d.iloc[-1] > k.iloc[-1]:
                    send_telegram_message(f"BUY signal for {symbol}: %D crossed above %K below 20.")

                if k.iloc[-1] > 80 and d.iloc[-1] > 80 and k.iloc[-2] >= d.iloc[-2] and k.iloc[-1] < d.iloc[-1]:
                    send_telegram_message(f"SELL signal for {symbol}: %K crossed below %D above 80.")

        # Wait 1 hour before the next check
        time.sleep(3600)

if __name__ == "__main__":
    monitor_signals()
