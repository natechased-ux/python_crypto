import asyncio
import requests
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
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "velo-usd", "coti-usd",
           "axs-usd"]
price_range_percentage = 1  # Filter for bids/asks within .25% of the mid-price
aggregation_threshold = 0.001
vwap_length = 14
atr_length = 14
risk_reward_ratio = 1.5
volume_spike_multiplier = 1.1
granularity = 21600

# Dynamic Whale Threshold Calculation
def get_dynamic_whale_threshold(order_book):
    """
    Calculates a dynamic whale threshold based on the average order size in the order book.
    """
    all_sizes = [float(bid[1]) for bid in order_book['bids']] + [float(ask[1]) for ask in order_book['asks']]
    average_size = sum(all_sizes) / len(all_sizes) if all_sizes else 1
    whale_threshold = average_size * 100  # Define a whale as 5x the average order size
    return whale_threshold

# Group Close Prices
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

# Calculate Indicators
def calculate_indicators(data):
    data['vwap'] = ((data['high'] + data['low'] + data['close']) / 3).rolling(vwap_length).sum() / data['volume'].rolling(vwap_length).sum()
    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=atr_length)
    data['atr'] = atr_indicator.average_true_range()
    data['avg_volume'] = data['volume'].rolling(window=14).mean()
    data['volume_spike'] = data['volume'] > volume_spike_multiplier * data['avg_volume']
    return data

# Analyze Whale Orders
def analyze_whale_orders(symbol, mid_price):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
    response = requests.get(url)
    if response.status_code == 200:
        order_book = response.json()
        whale_threshold = get_dynamic_whale_threshold(order_book)

        def within_price_range(price):
            return abs(price - mid_price) / mid_price <= price_range_percentage / 100

        bids = [(float(bid[0]), float(bid[1])) for bid in order_book['bids'] if float(bid[1]) >= whale_threshold and within_price_range(float(bid[0]))]
        asks = [(float(ask[0]), float(ask[1])) for ask in order_book['asks'] if float(ask[1]) >= whale_threshold and within_price_range(float(ask[0]))]

        grouped_bids = group_close_prices(bids, aggregation_threshold)
        grouped_asks = group_close_prices(asks, aggregation_threshold)

        nearest_bid = min((bid for bid in grouped_bids if bid[0] < mid_price), key=lambda x: abs(mid_price - x[0]), default=None)
        nearest_ask = min((ask for ask in grouped_asks if ask[0] > mid_price), key=lambda x: abs(mid_price - x[0]), default=None)

        next_ask = min((ask for ask in grouped_asks if ask[0] > mid_price and ask != nearest_ask), key=lambda x: x[0], default=None)
        next_bid = max((bid for bid in grouped_bids if bid[0] < mid_price and bid != nearest_bid), key=lambda x: x[0], default=None)

        return nearest_bid, nearest_ask, next_ask, next_bid
    else:
        return None, None, None, None

# Analyze Order Book
def analyze_order_book(symbol, data):
    try:
        data = calculate_indicators(data)
        latest = data.iloc[-1]
        close_price = latest['close']
        atr = latest['atr']
        volume_spike = latest['volume_spike']

        nearest_bid, nearest_ask, next_ask, next_bid = analyze_whale_orders(symbol, close_price)

        def format_price(price):
            return f"{price:.4f}" if price < 10 else f"{price:.2f}"

        if nearest_bid and nearest_bid[0] < close_price:
            tp = close_price + (atr * risk_reward_ratio)
            sl = close_price - atr
            resistance = format_price(next_ask[0]) if next_ask else "N/A"
            return f"{symbol.upper()}: LONG Signal | Entry: {format_price(close_price)}, TP: {format_price(tp)}, SL: {format_price(sl)}, Resistance: {resistance}, Support: {format_price(nearest_bid[0])}"
        elif nearest_ask and nearest_ask[0] > close_price:
            tp = close_price - (atr * risk_reward_ratio)
            sl = close_price + atr
            resistance = format_price(next_bid[0]) if next_bid else "N/A"
            return f"{symbol.upper()}: SHORT Signal | Entry: {format_price(close_price)}, TP: {format_price(tp)}, SL: {format_price(sl)}, Resistance: {resistance}, Support: {format_price(nearest_ask[0])}"
        else:
            return None

    except Exception as e:
        return f"{symbol.upper()}: Error - {str(e)}"

# Fetch Historical Data
def fetch_historical_data(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    response = requests.get(url)
    if response.status_code == 200:
        columns = ["time", "low", "high", "open", "close", "volume"]
        data = pd.DataFrame(response.json(), columns=columns)
        data = data.sort_values(by="time").astype(float)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        return data
    else:
        raise Exception(f"Failed to fetch data for {symbol}: {response.status_code}")

# Monitor Signals
async def monitor_signals():
    while True:
        try:
            messages = []
            for symbol in symbols:
                data = fetch_historical_data(symbol)
                message = analyze_order_book(symbol, data)
                if message:
                    messages.append(message)

            if messages:
                full_message = "\n".join(messages)
                await bot.send_message(chat_id=CHAT_ID, text=full_message)
        except Exception as e:
            await bot.send_message(chat_id=CHAT_ID, text=f"General Error: {str(e)}")

        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(monitor_signals())
