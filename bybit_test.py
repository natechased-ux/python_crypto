import asyncio
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from ta.volatility import AverageTrueRange
from telegram import Bot

# Telegram Bot Configurations
TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = Bot(token=TOKEN)

# Bybit-compatible symbols
symbols = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT", "ADAUSDT",
    "DOGEUSDT", "SOLUSDT", "WIFUSDT", "ONDOUSDT", "SEIUSDT",
    "MAGICUSDT", "APEUSDT", "JASMYUSDT", "WLDUSDT", "SYRUPUSDT",
    "AEROUSDT", "LINKUSDT", "HBARUSDT", "AAVEUSDT", 
    "CRVUSDT", "AVAXUSDT", "XCNUSDT", "UNIUSDT", "MKRUSDT",
     "NEARUSDT", "ALGOUSDT", "TRUMPUSDT", "BCHUSDT",
    "INJUSDT",  "XLMUSDT", "MOODENGUSDT",
    "DOTUSDT", "POPCATUSDT", "ARBUSDT", "ICPUSDT", "QNTUSDT",
    "TIAUSDT", "IPUSDT", "PNUTUSDT", "APTUSDT", "VETUSDT",
    "ENAUSDT",  "BERAUSDT", "POLUSDT", "MASKUSDT",
    "ACHUSDT", "PYTHUSDT", "SANDUSDT", "MORPHOUSDT", "MANAUSDT",
    "COTIUSDT", "AXSUSDT"
]

price_range_percentage = 10
aggregation_threshold = 0.001
vwap_length = 14
atr_length = 14
risk_reward_ratio = 2.0

def is_valid_symbol(symbol):
    url = "https://api.bybit.com/v5/market/instruments-info?category=spot"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return False
        instruments = response.json()['result']['list']
        return any(item['symbol'].upper() == symbol.upper() for item in instruments)
    except:
        return False

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
    all_sizes = [float(bid[1]) for bid in order_book['b']] + [float(ask[1]) for ask in order_book['a']]
    avg_size = sum(all_sizes) / len(all_sizes) if all_sizes else 1
    return avg_size * 2

def calculate_vwap(data):
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['volume']).rolling(vwap_length).sum() / data['volume'].rolling(vwap_length).sum()
    return data

def calculate_atr(data):
    atr = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=atr_length)
    data['atr'] = atr.average_true_range()
    return data

def analyze_whale_orders(symbol, mid_price):
    url = f"https://api.bybit.com/v5/market/orderbook?category=spot&symbol={symbol}"
    response = requests.get(url)
    if response.status_code != 200:
        return None, None
    data = response.json()
    if 'result' not in data:
        return None, None
    order_book = data['result']
    whale_threshold = get_dynamic_whale_threshold(order_book)
    bids = [(float(b[0]), float(b[1])) for b in order_book['b'] if float(b[1]) >= whale_threshold]
    asks = [(float(a[0]), float(a[1])) for a in order_book['a'] if float(a[1]) >= whale_threshold]

    grouped_bids = group_close_prices(bids, aggregation_threshold)
    grouped_asks = group_close_prices(asks, aggregation_threshold)

    filtered_bids = [b for b in grouped_bids if abs(mid_price - b[0]) / mid_price >= 0.01]
    filtered_asks = [a for a in grouped_asks if abs(mid_price - a[0]) / mid_price >= 0.01]

    nearest_bid = min(filtered_bids, key=lambda x: abs(mid_price - x[0]), default=None)
    nearest_ask = min(filtered_asks, key=lambda x: abs(mid_price - x[0]), default=None)

    return nearest_bid, nearest_ask

def analyze_order_book(symbol, data):
    try:
        data = calculate_vwap(data)
        data = calculate_atr(data)

        latest = data.iloc[-1]
        vwap, atr = latest['vwap'], latest['atr']
        close_price = latest['close']

        nearest_bid, nearest_ask = analyze_whale_orders(symbol, close_price)

        long_signal = nearest_bid and close_price <= nearest_bid[0]
        short_signal = nearest_ask and close_price >= nearest_ask[0]

        def fmt(price): return f"{price:.4f}" if price < 10 else f"{price:.2f}"

        if long_signal:
            tp1 = close_price + (atr * (risk_reward_ratio - .5))
            tp2 = close_price + (atr * risk_reward_ratio)
            sl1 = close_price - (atr * .5)
            sl2 = close_price - atr
            return (f"bybit1hr {symbol}: 🟢 LONG\nEntry: {fmt(close_price)}\n"
                    f"TP1: {fmt(tp1)}, TP2: {fmt(tp2)}\n"
                    f"SL1: {fmt(sl1)}, SL2: {fmt(sl2)}\n"
                    f"Support: {fmt(nearest_bid[0])}")
        elif short_signal:
            tp1 = close_price - (atr * (risk_reward_ratio - .5))
            tp2 = close_price - (atr * risk_reward_ratio)
            sl1 = close_price + (atr * .5)
            sl2 = close_price + atr
            return (f"bybit 1hr {symbol}: 🔴 SHORT\nEntry: {fmt(close_price)}\n"
                    f"TP1: {fmt(tp1)}, TP2: {fmt(tp2)}\n"
                    f"SL1: {fmt(sl1)}, SL2: {fmt(sl2)}\n"
                    f"Resistance: {fmt(nearest_ask[0])}")
        return None

    except Exception as e:
        return f"{symbol}: Error - {str(e)}"

async def send_to_telegram(message):
    await bot.send_message(chat_id=CHAT_ID, text=message)

def fetch_historical_data(symbol):
    url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval=60&limit=200"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch {symbol}: {response.status_code}")
    json_data = response.json()
    if not isinstance(json_data, dict) or 'result' not in json_data or 'list' not in json_data['result']:
        raise Exception(f"Unexpected format for {symbol}: {json_data}")
    rows = json_data['result']['list']
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
    df = df.astype({"timestamp": "int64", "open": "float", "high": "float",
                    "low": "float", "close": "float", "volume": "float"})
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values("time")
    return df[["time", "low", "high", "open", "close", "volume"]]

async def hourly_updates():
    valid_symbols = [s for s in symbols if is_valid_symbol(s)]
    while True:
        try:
            messages = []
            for symbol in valid_symbols:
                try:
                    data = fetch_historical_data(symbol)
                    message = analyze_order_book(symbol, data)
                    if message:
                        messages.append(message)
                except Exception as e:
                    print(f"⚠️ {symbol}: {e}")
            if messages:
                await send_to_telegram("\n\n".join(messages))
        except Exception as e:
            await send_to_telegram(f"Error: {str(e)}")
        print(f"⏰ bybit Checked at {datetime.now()}")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(hourly_updates())
