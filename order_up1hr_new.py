import asyncio
import requests
import numpy as np
import pandas as pd
import time
from datetime import datetime
from ta.volatility import AverageTrueRange
from telegram import Bot
from pytz import timezone

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

min_distance_pct = 0.0015  # Reduced to 0.15% to loosen filtering

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

def get_dynamic_whale_threshold(order_book):
    all_sizes = [float(bid[1]) for bid in order_book['bids']] + [float(ask[1]) for ask in order_book['asks']]
    average_size = sum(all_sizes) / len(all_sizes) if all_sizes else 1
    whale_threshold = average_size * 20  # Loosened threshold
    return whale_threshold

def calculate_vwap(data):
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['volume']).rolling(vwap_length).sum() / data['volume'].rolling(vwap_length).sum()
    return data

def calculate_atr(data):
    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=atr_length)
    data['atr'] = atr_indicator.average_true_range()
    return data

def get_live_price(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/ticker"
    response = requests.get(url)
    if response.status_code == 200:
        return float(response.json()['price'])
    else:
        raise Exception(f"Failed to fetch live price for {symbol}: {response.status_code}")

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

        filtered_bids = [bid for bid in grouped_bids if abs(bid[0] - mid_price) / mid_price >= min_distance_pct]
        filtered_asks = [ask for ask in grouped_asks if abs(ask[0] - mid_price) / mid_price >= min_distance_pct]

        nearest_bid = min(filtered_bids, key=lambda x: abs(mid_price - x[0]), default=None)
        nearest_ask = min(filtered_asks, key=lambda x: abs(mid_price - x[0]), default=None)

        return nearest_bid, nearest_ask
    else:
        return None, None

def analyze_order_book(symbol, data):
    try:
        data = calculate_vwap(data)
        data = calculate_atr(data)
        latest = data.iloc[-1]
        vwap, atr = latest['vwap'], latest['atr']

        close_price = get_live_price(symbol)
        nearest_bid, nearest_ask = analyze_whale_orders(symbol, close_price)

        long_signal = nearest_bid and close_price <= nearest_bid[0] and close_price >= 0.98 * vwap
        short_signal = nearest_ask and close_price >= nearest_ask[0] and close_price <= 1.02 * vwap

        def format_price(price):
            return f"{price:.6f}" if price < 10 else f"{price:.2f}"

        pst_now = datetime.now(timezone("US/Pacific")).strftime("%Y-%m-%d %I:%M:%S %p %Z")

        if long_signal:
            tp1 = close_price + (atr * (risk_reward_ratio - 0.5))
            tp2 = close_price + (atr * risk_reward_ratio)
            sl1 = close_price - (atr * 0.5)
            sl2 = close_price - atr
            return (f"\n📈 1hr new {symbol.upper()}: LONG Signal\nTime: {pst_now}\n"
                    f"Entry: {format_price(close_price)}, TP1: {format_price(tp1)}, TP2: {format_price(tp2)}\n"
                    f"SL1: {format_price(sl1)}, SL2: {format_price(sl2)}\n"
                    f"Support (Nearest Bid): {format_price(nearest_bid[0])}")
        elif short_signal:
            tp1 = close_price - (atr * (risk_reward_ratio - 0.5))
            tp2 = close_price - (atr * risk_reward_ratio)
            sl1 = close_price + (atr * 0.5)
            sl2 = close_price + atr
            return (f"\n📉 1hr new {symbol.upper()}: SHORT Signal\nTime: {pst_now}\n"
                    f"Entry: {format_price(close_price)}, TP1: {format_price(tp1)}, TP2: {format_price(tp2)}\n"
                    f"SL1: {format_price(sl1)}, SL2: {format_price(sl2)}\n"
                    f"Resistance (Nearest Ask): {format_price(nearest_ask[0])}")
        else:
            return None

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
        start_time = time.time()
        try:
            tasks = []
            for symbol in symbols:
                data = fetch_historical_data(symbol)
                tasks.append(asyncio.to_thread(analyze_order_book, symbol, data))

            results = await asyncio.gather(*tasks)
            messages = [result for result in results if result and "Error" not in result]

            if messages:
                full_message = "\n".join(messages)
                await send_to_telegram(full_message)
        except Exception as e:
            print(f"Error: {str(e)}")

        end_time = time.time()
        print(f"⏰ NEW Cycle completed in {end_time - start_time:.2f} seconds — {pd.Timestamp.now()}")

        await asyncio.sleep(60)  # Sleep 60s to keep it running hourly

if __name__ == "__main__":
    asyncio.run(hourly_updates())
