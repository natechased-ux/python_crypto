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
           "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd",
           "syrup-usd","fartcoin-usd", "aero-usd", "link-usd","hbar-usd",
           "aave-usd", "fet-usd", "crv-usd","tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd",
           "bch-usd", "inj-usd", "pepe-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
           "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
           "qnt-usd", "tia-usd", "ip-usd" , "pnut-usd", "apt-usd", "vet-usd",
           "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "coti-usd",
           "axs-usd"]

price_range_percentage = 10
aggregation_threshold = 0.001
vwap_length = 14
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

# Dynamic Whale Threshold Calculation
def get_dynamic_whale_threshold(order_book):
    """
    Calculates a dynamic whale threshold based on the average order size in the order book.
    """
    all_sizes = [float(bid[1]) for bid in order_book['bids']] + [float(ask[1]) for ask in order_book['asks']]
    average_size = sum(all_sizes) / len(all_sizes) if all_sizes else 1
    whale_threshold = average_size * 20  # Define a whale as 5x the average order size
    return whale_threshold

def calculate_vwap(data):
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['volume']).rolling(vwap_length).sum() / data['volume'].rolling(vwap_length).sum()
    return data

def calculate_atr(data):
    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=atr_length)
    data['atr'] = atr_indicator.average_true_range()
    return data

def analyze_whale_orders(symbol, mid_price):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
    response = requests.get(url)
    if response.status_code == 200:
        order_book = response.json()
        whale_threshold = get_dynamic_whale_threshold(order_book)
        bids = [(float(bid[0]), float(bid[1])) for bid in order_book['bids'] if float(bid[1]) >= whale_threshold]
        asks = [(float(ask[0]), float(ask[1])) for ask in order_book['asks'] if float(ask[1]) >= whale_threshold]

        grouped_bids = group_close_prices(bids, aggregation_threshold)
        grouped_asks = group_close_prices(asks, aggregation_threshold)

        # Filter bids and asks to be at least 1% away from the mid-price
        filtered_bids = [bid for bid in grouped_bids if abs(mid_price - bid[0]) / mid_price >= 0.01]
        filtered_asks = [ask for ask in grouped_asks if abs(mid_price - ask[0]) / mid_price >= 0.01]

        nearest_bid = min(filtered_bids, key=lambda x: abs(mid_price - x[0]), default=None)
        nearest_ask = min(filtered_asks, key=lambda x: abs(mid_price - x[0]), default=None)

        return nearest_bid, nearest_ask
    else:
        return None, None

def analyze_order_book(symbol, data):
    try:
        # VWAP and ATR Analysis
        data = calculate_vwap(data)
        data = calculate_atr(data)

        latest = data.iloc[-1]
        vwap, atr = latest['vwap'], latest['atr']
        close_price = latest['close']

        # Fetch Whale Orders
        nearest_bid, nearest_ask = analyze_whale_orders(symbol, close_price)

        # Long and Short Signal Logic
        long_signal = nearest_bid and close_price <= nearest_bid[0]
        short_signal = nearest_ask and close_price >= nearest_ask[0]

        def format_price(price):
            return f"{price:.4f}" if price < 10 else f"{price:.2f}"

        if long_signal:
            tp1 = close_price + (atr * (risk_reward_ratio-.5))
            tp2 = close_price + (atr * (risk_reward_ratio))                    
            sl1 = close_price - (atr*.5)
            sl2 = close_price - (atr)
            return (f"24hr{symbol.upper()}: LONG Signal | Entry: {format_price(close_price)}, TP1: {format_price(tp1)}, TP2: {format_price(tp2)}\n"
                    f"SL1: {format_price(sl1)}, SL2: {format_price(sl2)}\n"
                    f"Support (Nearest Bid): {format_price(nearest_bid[0])}")
        elif short_signal:
            tp1 = close_price - (atr * (risk_reward_ratio-.5))
            tp2 = close_price - (atr * (risk_reward_ratio))
            sl1 = close_price + (atr*.5)
            sl2 = close_price + atr
            return (f"24hr{symbol.upper()}: SHORT Signal | Entry: {format_price(close_price)}, TP1: {format_price(tp1)}, TP2: {format_price(tp2)}\n"
                    f"SL1: {format_price(sl1)}, SL2: {format_price(sl2)}\n"
                    f"Resistance (Nearest Ask): {format_price(nearest_ask[0])}")
        else:
            return None

    except Exception as e:
        return f"{symbol.upper()}: Error - {str(e)}"

async def send_to_telegram(message):
    await bot.send_message(chat_id=CHAT_ID, text=message)

def fetch_historical_data(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity=86400"
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
                if message:  # Only add messages for significant signals
                    messages.append(message)

            if messages:
                full_message = "\n".join(messages)
                await send_to_telegram(full_message)
        except Exception as e:
            await send_to_telegram(f"Error: {str(e)}")

        await asyncio.sleep(60)  # Run every hour

if __name__ == "__main__":
    asyncio.run(hourly_updates())
