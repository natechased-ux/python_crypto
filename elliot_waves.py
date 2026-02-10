import asyncio
import requests
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from telegram import Bot

# Telegram Bot Configurations
TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = Bot(token=TOKEN)

# Configuration Parameters
SYMBOL = "BTC-USD"
GRANULARITY = 21600  # Daily data
LOOKBACK = 100  # Number of data points to analyze
MIN_DIST = 5  # Minimum distance between local extrema for Elliott Waves
WAVE_DEVIATION = 0.05  # Deviation to qualify Elliott Wave structure

# State to track last processed data
last_processed_time = None

# Fetch Historical Data
def fetch_historical_data(symbol, granularity):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    response = requests.get(url)
    if response.status_code == 200:
        columns = ["time", "low", "high", "open", "close", "volume"]
        data = pd.DataFrame(response.json(), columns=columns)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        numeric_columns = ["low", "high", "open", "close", "volume"]
        data[numeric_columns] = data[numeric_columns].astype(float)
        data = data.sort_values(by="time").reset_index(drop=True)
        return data
    else:
        raise Exception(f"Failed to fetch data for {symbol}: {response.status_code}")

# Identify Local Extrema
def identify_extrema(data):
    highs = argrelextrema(data['high'].values, np.greater, order=MIN_DIST)[0].astype(int)
    lows = argrelextrema(data['low'].values, np.less, order=MIN_DIST)[0].astype(int)
    return highs, lows

# Validate Elliott Wave
def validate_elliott_wave(data, highs, lows):
    waves = []
    all_points = sorted(list(highs) + list(lows))
    for i in range(len(all_points) - 4):
        wave_points = all_points[i:i+5]
        wave_prices = data['close'].iloc[wave_points].values

        if (
            wave_prices[1] > wave_prices[0] and
            wave_prices[2] < wave_prices[1] and
            wave_prices[3] > wave_prices[2] and
            wave_prices[4] < wave_prices[3] and
            abs((wave_prices[4] - wave_prices[0]) / wave_prices[0]) > WAVE_DEVIATION
        ):
            waves.append({
                "start": wave_points[0],
                "end": wave_points[4],
                "wave_prices": wave_prices
            })
    return waves

# Analyze Signals
def analyze_signals(data, waves):
    signals = []
    for wave in waves:
        start_price = data['close'].iloc[wave['start']]
        end_price = data['close'].iloc[wave['end']]
        if end_price < start_price:  # Downward Wave
            signals.append({
                "type": "SELL",
                "entry_price": end_price,
                "stop_loss": end_price + (end_price * 0.02),
                "take_profit": end_price - (end_price * 0.05)
            })
        else:  # Upward Wave
            signals.append({
                "type": "BUY",
                "entry_price": end_price,
                "stop_loss": end_price - (end_price * 0.02),
                "take_profit": end_price + (end_price * 0.05)
            })
    return signals

# Send Telegram Alerts
async def send_telegram_alerts(signals):
    for signal in signals:
        message = (
            f"Signal Type: {signal['type']}\n"
            f"Entry Price: {signal['entry_price']:.2f}\n"
            f"Take Profit: {signal['take_profit']:.2f}\n"
            f"Stop Loss: {signal['stop_loss']:.2f}"
        )
        await bot.send_message(chat_id=CHAT_ID, text=message)

# Monitor for Live Signals
async def monitor_signals():
    global last_processed_time
    while True:
        try:
            data = fetch_historical_data(SYMBOL, GRANULARITY)
            if last_processed_time:
                data = data[data['time'] > last_processed_time]

            if not data.empty:
                highs, lows = identify_extrema(data)
                waves = validate_elliott_wave(data, highs, lows)
                signals = analyze_signals(data, waves)

                if signals:
                    await send_telegram_alerts(signals)

                # Update last processed time
                last_processed_time = data['time'].max()
        except Exception as e:
            await bot.send_message(chat_id=CHAT_ID, text=f"Error: {str(e)}")

        await asyncio.sleep(3600)  # Check every 24 hours

# Run the Monitor
if __name__ == "__main__":
    asyncio.run(monitor_signals())
