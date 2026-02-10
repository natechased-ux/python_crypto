import asyncio
import requests
import pandas as pd
from ta.momentum import RSIIndicator
from datetime import datetime
from ta.volatility import AverageTrueRange
from telegram import Bot

# Telegram Bot Configurations
TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = Bot(token=TOKEN)

# Configuration Parameters
SYMBOLS = ["btc-usd", "xrp-usd", "ltc-usd", "doge-usd", "sol-usd", "magic-usd", "wld-usd",
           "link-usd", "hbar-usd", "crv-usd", "tao-usd", "avax-usd", "uni-usd",
           "mkr-usd", "bch-usd", "xlm-usd", "dot-usd", "arb-usd", "icp-usd",
           "qnt-usd",  "axs-usd"]
RSI_PERIOD = 14
ATR_PERIOD = 14
OVERSOLD = 20
OVERBOUGHT = 80
VOLUME_MULTIPLIER = 1.1
TAKE_PROFIT_MULTIPLIER = 1.5
STOP_LOSS_MULTIPLIER = 0.5

# Track last processed time and first run status
last_processed_time = {symbol: None for symbol in SYMBOLS}
first_run = {symbol: True for symbol in SYMBOLS}

# Fetch Historical Data
def fetch_historical_data(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity=3600"
    response = requests.get(url)
    if response.status_code == 200:
        columns = ["time", "low", "high", "open", "close", "volume"]
        data = pd.DataFrame(response.json(), columns=columns)
        data = data.sort_values(by="time").astype(float)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        return data
    else:
        raise Exception(f"Failed to fetch data for {symbol}: {response.status_code}")

# Calculate Indicators
def calculate_indicators(data):
    data['rsi'] = RSIIndicator(close=data['close'], window=RSI_PERIOD).rsi()
    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=ATR_PERIOD)
    data['atr'] = atr_indicator.average_true_range()
    data['avg_volume'] = data['volume'].rolling(window=RSI_PERIOD).mean()
    return data

# Format Price
def format_price(price):
    if price >= 1000:
        return f"{price:.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    else:
        return f"{price:.8f}"

# Analyze Signals
def analyze_signals(data, symbol):
    signals = []
    for i in range(len(data)):
        row = data.iloc[i]
        close_price = row['close']
        atr = row['atr']
        rsi = row['rsi']
        volume = row['volume']
        avg_volume = row['avg_volume']

        long_signal = rsi < OVERSOLD and volume > VOLUME_MULTIPLIER * avg_volume
        short_signal = rsi > OVERBOUGHT and volume > VOLUME_MULTIPLIER * avg_volume

        if long_signal:
            tp = close_price + (atr * TAKE_PROFIT_MULTIPLIER)
            sl = close_price - (atr * STOP_LOSS_MULTIPLIER)
            signals.append({
                'time': row['time'],
                'symbol': symbol,
                'signal': 'LONG',
                'entry': close_price,
                'take_profit': tp,
                'stop_loss': sl
            })
        elif short_signal:
            tp = close_price - (atr * TAKE_PROFIT_MULTIPLIER)
            sl = close_price + (atr * STOP_LOSS_MULTIPLIER)
            signals.append({
                'time': row['time'],
                'symbol': symbol,
                'signal': 'SHORT',
                'entry': close_price,
                'take_profit': tp,
                'stop_loss': sl
            })
    return signals

# Send Telegram Alerts
async def send_alerts(signals):
    for signal in signals:
        message = (
            f"Symbol: {signal['symbol']}\n"
            f"Time: {signal['time']}\n"
            f"Signal: {signal['signal']}\n"
            f"Entry: {format_price(signal['entry'])}\n"
            f"Take Profit: {format_price(signal['take_profit'])}\n"
            f"Stop Loss: {format_price(signal['stop_loss'])}"
        )
        await bot.send_message(chat_id=CHAT_ID, text=message)

# Initialize timestamps on startup
def initialize_last_processed_times():
    for symbol in SYMBOLS:
        try:
            data = fetch_historical_data(symbol)
            if not data.empty:
                last_processed_time[symbol] = data['time'].max()
                print(f"Initialized {symbol} with timestamp {last_processed_time[symbol]}")
        except Exception as e:
            print(f"Initialization error for {symbol}: {e}")

# Monitor for Signals
async def monitor_signals():
    while True:
        try:
            for symbol in SYMBOLS:
                try:
                    data = fetch_historical_data(symbol)
                    data = calculate_indicators(data)

                    # Filter out already processed rows
                    latest_time = last_processed_time[symbol]
                    if latest_time:
                        data = data[data['time'] > latest_time]

                    # Skip alerting on the first run
                    if first_run[symbol]:
                        first_run[symbol] = False
                        if not data.empty:
                            last_processed_time[symbol] = data['time'].max()
                        continue

                    if not data.empty:
                        signals = analyze_signals(data, symbol)
                        if signals:
                            await send_alerts(signals)
                            last_processed_time[symbol] = data['time'].max()
                except Exception as e:
                    print(f"Error with {symbol}: {str(e)}")

            print(f"⏰ money maker — {pd.Timestamp.now()}")
            await asyncio.sleep(60)
        except Exception as e:
            print(f"General Error: {str(e)}")

# Entry Point
if __name__ == "__main__":
    initialize_last_processed_times()
    asyncio.run(monitor_signals())
