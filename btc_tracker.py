import asyncio
import requests
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from telegram import Bot

# Telegram Bot Configurations
TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = Bot(token=TOKEN)

# Configuration Parameters
SYMBOL = "BTC-USD"
RSI_PERIOD = 14
ATR_PERIOD = 14
OVERSOLD = 25
OVERBOUGHT = 75
VOLUME_MULTIPLIER = 1.1  # Spike if volume > 1.5x average
TAKE_PROFIT_MULTIPLIER = 1.5  # TP: 2x ATR
STOP_LOSS_MULTIPLIER = .5  # SL: 1x ATR

# Fetch Historical Data
def fetch_historical_data():
    url = f"https://api.exchange.coinbase.com/products/{SYMBOL}/candles?granularity=3600"
    response = requests.get(url)
    if response.status_code == 200:
        columns = ["time", "low", "high", "open", "close", "volume"]
        data = pd.DataFrame(response.json(), columns=columns)
        data = data.sort_values(by="time").astype(float)
        return data
    else:
        raise Exception(f"Failed to fetch data for {SYMBOL}: {response.status_code}")

# Calculate Indicators
def calculate_indicators(data):
    data['rsi'] = RSIIndicator(close=data['close'], window=RSI_PERIOD).rsi()
    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=ATR_PERIOD)
    data['atr'] = atr_indicator.average_true_range()
    data['avg_volume'] = data['volume'].rolling(window=RSI_PERIOD).mean()
    return data

# Analyze Signals
def analyze_signals(data):
    latest = data.iloc[-1]
    close_price = latest['close']
    atr = latest['atr']
    rsi = latest['rsi']
    volume = latest['volume']
    avg_volume = latest['avg_volume']

    # Define Signals
    long_signal = rsi < OVERSOLD and volume > VOLUME_MULTIPLIER * avg_volume
    short_signal = rsi > OVERBOUGHT and volume > VOLUME_MULTIPLIER * avg_volume

    def format_price(price):
        return f"{price:.2f}"

    # Generate Alerts
    if long_signal:
        tp = close_price + (atr * TAKE_PROFIT_MULTIPLIER)
        sl = close_price - (atr * STOP_LOSS_MULTIPLIER)
        return (f"BTC Futures Alert 🚀: LONG Signal\n"
                f"Entry: {format_price(close_price)}\n"
                f"Take Profit: {format_price(tp)}\n"
                f"Stop Loss: {format_price(sl)}")
    elif short_signal:
        tp = close_price - (atr * TAKE_PROFIT_MULTIPLIER)
        sl = close_price + (atr * STOP_LOSS_MULTIPLIER)
        return (f"BTC Futures Alert 📉: SHORT Signal\n"
                f"Entry: {format_price(close_price)}\n"
                f"Take Profit: {format_price(tp)}\n"
                f"Stop Loss: {format_price(sl)}")
    return None

# Send Telegram Alert
async def send_to_telegram(message):
    await bot.send_message(chat_id=CHAT_ID, text=message)

# Hourly Updates
async def hourly_updates():
    while True:
        try:
            data = fetch_historical_data()
            data = calculate_indicators(data)
            alert = analyze_signals(data)
            if alert:
                await send_to_telegram(alert)
        except Exception as e:
            await send_to_telegram(f"Error: {str(e)}")
        await asyncio.sleep(60)  # Run every hour

# Main Function
if __name__ == "__main__":
    asyncio.run(hourly_updates())
