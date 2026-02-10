import requests
import numpy as np
import pandas as pd
from ta.volume import AccDistIndexIndicator
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator, EMAIndicator
from telegram import Bot
from pathlib import Path
import time

# Telegram bot configuration
bot_token = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
chat_id = "7967738614"
bot = Bot(token=bot_token)

# Configuration parameters
atr_multiplier = 1.5
symbols = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd"]
granularity = 3600  # 5-minute candles

# Fetch historical data from local files
def fetch_historical_data_local(symbol):
    file_path = Path(f"historical_data/{symbol}.csv")
    if file_path.exists():
        data = pd.read_csv(file_path)
        data = data.sort_values(by="time")
        return data
    else:
        raise FileNotFoundError(f"Historical data for {symbol} not found.")

# Analyze data for signals
def analyze_data(data):
    # Indicators
    data['ema_short'] = EMAIndicator(close=data['close'], window=12).ema_indicator()
    data['ema_long'] = EMAIndicator(close=data['close'], window=26).ema_indicator()
    data['stochastic'] = StochasticOscillator(high=data['high'], low=data['low'], close=data['close'], window=14).stoch()
    data['bollinger_high'] = BollingerBands(close=data['close'], window=20, window_dev=2).bollinger_hband()
    data['bollinger_low'] = BollingerBands(close=data['close'], window=20, window_dev=2).bollinger_lband()
    data['adx'] = ADXIndicator(high=data['high'], low=data['low'], close=data['close'], window=14).adx()
    data['atr'] = data['high'] - data['low']  # Simplified ATR

    # Signals
    buy_signal = (
        (data['ema_short'].iloc[-1] > data['ema_long'].iloc[-1]) and
        (data['stochastic'].iloc[-1] < 20) and
        (data['adx'].iloc[-1] > 25) and
        (data['close'].iloc[-1] < data['bollinger_low'].iloc[-1])
    )
    sell_signal = (
        (data['ema_short'].iloc[-1] < data['ema_long'].iloc[-1]) and
        (data['stochastic'].iloc[-1] > 80) and
        (data['adx'].iloc[-1] > 25) and
        (data['close'].iloc[-1] > data['bollinger_high'].iloc[-1])
    )
    return buy_signal, sell_signal

# Simulate trades and calculate PnL
def simulate_trades(symbol):
    try:
        data = fetch_historical_data_local(symbol)
        buy_signal, sell_signal = analyze_data(data)

        if buy_signal:
            entry_price = data['close'].iloc[-1]
            tp = entry_price * (1 + atr_multiplier * data['atr'].iloc[-1] / entry_price)
            sl = entry_price * (1 - atr_multiplier * data['atr'].iloc[-1] / entry_price)
            print(f"Simulated Buy Signal for {symbol.upper()}! Entry: {entry_price}, TP: {tp}, SL: {sl}")

        elif sell_signal:
            entry_price = data['close'].iloc[-1]
            tp = entry_price * (1 - atr_multiplier * data['atr'].iloc[-1] / entry_price)
            sl = entry_price * (1 + atr_multiplier * data['atr'].iloc[-1] / entry_price)
            print(f"Simulated Sell Signal for {symbol.upper()}! Entry: {entry_price}, TP: {tp}, SL: {sl}")
        else:
            print(f"No signals for {symbol.upper()} at this time.")
    except Exception as e:
        print(f"Error processing {symbol}: {e}")

# Main testing loop
def main():
    for symbol in symbols:
        simulate_trades(symbol)

if __name__ == "__main__":
    main()
