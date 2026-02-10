import asyncio
import requests
import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange
from telegram import Bot

# Telegram Bot Configurations
TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = Bot(token=TOKEN)

# Configuration Parameters
symbols = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd"]
whale_threshold = 1.0
price_range_percentage = 10
aggregation_threshold = 0.001
proximity_weight = 0.6
quantity_weight = 0.4
confidence_threshold = 0.1
vwap_length = 14
imbalance_threshold = 10.0
smoothing_length = 3
atr_length = 14
risk_reward_ratio = 2.0

# Helper Functions
def group_close_prices(orders, threshold):
    if not orders:
        return []
    orders.sort(key=lambda x: x[0])
    grouped, current = [], [orders[0]]
    for price, qty in orders[1:]:
        prev_price = current[-1][0]
        if abs(price - prev_price) / prev_price <= threshold:
            current.append((price, qty))
        else:
            total_qty = sum(q for _, q in current)
            avg_price = sum(p * q for p, q in current) / total_qty
            grouped.append((avg_price, total_qty))
            current = [(price, qty)]
    total_qty = sum(q for _, q in current)
    avg_price = sum(p * q for p, q in current) / total_qty
    grouped.append((avg_price, total_qty))
    return grouped

def calculate_vwap(data):
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['volume']).rolling(vwap_length).sum() / data['volume'].rolling(vwap_length).sum()
    return data

def calculate_imbalance(data):
    data['bullish_volume'] = data['volume'] * (data['close'] > data['open'])
    data['bearish_volume'] = data['volume'] * (data['close'] <= data['open'])
    data['imbalance'] = 100 * (data['bullish_volume'] - data['bearish_volume']) / data['volume']
    data['smoothed_imbalance'] = data['imbalance'].rolling(smoothing_length).mean()
    return data

def calculate_atr(data):
    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=atr_length)
    data['atr'] = atr_indicator.average_true_range()
    return data

def analyze_order_book(symbol, data):
    try:
        # VWAP and Volume Imbalance Analysis
        data = calculate_vwap(data)
        data = calculate_imbalance(data)
        data = calculate_atr(data)

        latest = data.iloc[-1]
        vwap, smoothed_imbalance, atr = latest['vwap'], latest['smoothed_imbalance'], latest['atr']
        close_price = latest['close']

        # Buy and Sell Signal Logic
        buy_signal = smoothed_imbalance > imbalance_threshold and close_price > vwap
        sell_signal = smoothed_imbalance < -imbalance_threshold and close_price < vwap

        if buy_signal:
            tp = close_price + (atr * risk_reward_ratio)
            sl = close_price - atr
            return f"{symbol.upper()}: BUY Signal | Entry: {close_price:.2f}, TP: {tp:.2f}, SL: {sl:.2f}"
        elif sell_signal:
            tp = close_price - (atr * risk_reward_ratio)
            sl = close_price + atr
            return f"{symbol.upper()}: SELL Signal | Entry: {close_price:.2f}, TP: {tp:.2f}, SL: {sl:.2f}"
        else:
            return f"{symbol.upper()}: NO significant signal"

    except Exception as e:
        return f"{symbol.upper()}: Error - {str(e)}"

async def send_to_telegram(message):
    await bot.send_message(chat_id=CHAT_ID, text=message)

def fetch_historical_data(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity=3600"
    response = requests.get(url)
    if response.status_code == 200:
        columns = ["time", "low", "high", "open", "close", "volume"]
        data = pd.DataFrame(response.json(), columns=columns)
        data = data.sort_values(by="time")
        return data
    else:
        raise Exception(f"Failed to fetch data for {symbol}: {response.status_code}")

async def hourly_updates():
    while True:
        try:
            messages = []
            for symbol in symbols:
                data = fetch_historical_data(symbol)
                message = analyze_order_book(symbol, data)
                messages.append(message)

            full_message = "\n".join(messages)
            await send_to_telegram(full_message)
        except Exception as e:
            await send_to_telegram(f"Error: {str(e)}")

        await asyncio.sleep(3600)  # Run every hour

if __name__ == "__main__":
    asyncio.run(hourly_updates())
