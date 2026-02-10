import asyncio
import requests
import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange
from telegram import Bot
import time

# Telegram Bot Configurations
TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = Bot(token=TOKEN)

# Configuration Parameters
symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT',
           'LTCUSDT', 'ADAUSDT', 'DOGEUSDT', 'SOLUSDT', 'WIFUSDT', 'ONDOUSDT',
           'SEIUSDT', 'MAGICUSDT', 'APEUSDT', 'JASMYUSDT', 'WLDUSDT', 'SYRUPUSDT',
           'LINKUSDT', 'HBARUSDT', 'AAVEUSDT', 'FETUSDT', 'CRVUSDT', 'TAOUSDT', 'AVAXUSDT',
           'UNIUSDT', 'MKRUSDT', 'NEARUSDT', 'ALGOUSDT', 'TRUMPUSDT', 'BCHUSDT', 'INJUSDT',
           'PEPEUSDT', 'XLMUSDT', 'BONKUSDT', 'DOTUSDT', 'ARBUSDT', 'ICPUSDT', 'QNTUSDT',
           'TIAUSDT', 'PNUTUSDT', 'APTUSDT', 'VETUSDT', 'ENAUSDT', 'TURBOUSDT', 'BERAUSDT',
           'POLUSDT', 'MASKUSDT', 'ACHUSDT', 'PYTHUSDT', 'SANDUSDT', 'MANAUSDT', 'COTIUSDT', 'AXSUSDT']

whale_threshold = 1.0
price_range_percentage = 10
aggregation_threshold = 0.001
vwap_length = 14
atr_length = 14
risk_reward_ratio = 2.0

# Rate Limit Parameters
REQUEST_DELAY = 0.1  # Add a 100ms delay between requests to avoid hitting the limit

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

def calculate_atr(data):
    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=atr_length)
    data['atr'] = atr_indicator.average_true_range()
    return data

def fetch_order_book(symbol):
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=5000"
    for _ in range(3):  # Retry up to 3 times
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print("Rate limit hit. Sleeping for 1 minute...")
            time.sleep(60)  # Sleep for a minute if rate limit is exceeded
        else:
            print(f"Error: {response.status_code} for {symbol}")
    raise Exception(f"Failed to fetch order book for {symbol}")

def analyze_whale_orders(symbol, mid_price):
    order_book = fetch_order_book(symbol)
    bids = [(float(bid[0]), float(bid[1])) for bid in order_book['bids'] if float(bid[1]) >= whale_threshold]
    asks = [(float(ask[0]), float(ask[1])) for ask in order_book['asks'] if float(ask[1]) >= whale_threshold]

    grouped_bids = group_close_prices(bids, aggregation_threshold)
    grouped_asks = group_close_prices(asks, aggregation_threshold)

    filtered_bids = [bid for bid in grouped_bids if abs(mid_price - bid[0]) / mid_price >= 0.01]
    filtered_asks = [ask for ask in grouped_asks if abs(mid_price - ask[0]) / mid_price >= 0.01]

    nearest_bid = min(filtered_bids, key=lambda x: abs(mid_price - x[0]), default=None)
    nearest_ask = min(filtered_asks, key=lambda x: abs(mid_price - x[0]), default=None)

    return nearest_bid, nearest_ask

def fetch_historical_data(symbol):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=100"
    for _ in range(3):  # Retry up to 3 times
        response = requests.get(url)
        if response.status_code == 200:
            columns = [
                "time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ]
            data = pd.DataFrame(response.json(), columns=columns)
            data = data[["time", "open", "high", "low", "close", "volume"]]
            data = data.astype(float)
            data = data.sort_values(by="time")
            return data
        elif response.status_code == 429:
            print("Rate limit hit. Sleeping for 1 minute...")
            time.sleep(60)
        else:
            print(f"Error: {response.status_code} for {symbol}")
    raise Exception(f"Failed to fetch historical data for {symbol}")

def analyze_order_book(symbol, data):
    try:
        data = calculate_vwap(data)
        data = calculate_atr(data)

        latest = data.iloc[-1]
        vwap, atr = latest['vwap'], latest['atr']
        close_price = latest['close']

        nearest_bid, nearest_ask = analyze_whale_orders(symbol, close_price)

        def format_price(price):
            return f"{price:.4f}" if price < 10 else f"{price:.2f}"

        if nearest_bid and close_price <= nearest_bid[0]:
            tp = close_price + (atr * risk_reward_ratio)
            sl = close_price - atr
            return (f"BINANCE_1hr{symbol}: LONG Signal | Entry: {format_price(close_price)}, TP: {format_price(tp)}, SL: {format_price(sl)}\n"
                    f"Support (Nearest Bid): {format_price(nearest_bid[0])}")
        elif nearest_ask and close_price >= nearest_ask[0]:
            tp = close_price - (atr * risk_reward_ratio)
            sl = close_price + atr
            return (f"BINANCE_1hr{symbol}: SHORT Signal | Entry: {format_price(close_price)}, TP: {format_price(tp)}, SL: {format_price(sl)}\n"
                    f"Resistance (Nearest Ask): {format_price(nearest_ask[0])}")
        else:
            return None
    except Exception as e:
        return f"{symbol}: Error - {str(e)}"

async def hourly_updates():
    while True:
        try:
            messages = []
            for symbol in symbols:
                data = fetch_historical_data(symbol)
                message = analyze_order_book(symbol, data)
                if message:
                    messages.append(message)
                time.sleep(REQUEST_DELAY)  # Add delay to avoid rate limit

            if messages:
                full_message = "\n".join(messages)
                await bot.send_message(chat_id=CHAT_ID, text=full_message)
        except Exception as e:
            await bot.send_message(chat_id=CHAT_ID, text=f"Error: {str(e)}")

        await asyncio.sleep(60)  # Run every hour

if __name__ == "__main__":
    asyncio.run(hourly_updates())
