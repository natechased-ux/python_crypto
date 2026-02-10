import websocket
import json
import threading
import requests
import pandas as pd
from datetime import datetime
from collections import deque

# === CONFIG ===
SYMBOL = 'ethusd'
RSI_PERIOD = 14
SMA_PERIOD = 50
IMBALANCE_BUY = 60
IMBALANCE_SELL = 40
TELEGRAM_TOKEN = '8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw'
TELEGRAM_CHAT_ID = '7967738614'

minute_data = []  # Stores raw data for the current 5-minute period
price_data = deque(maxlen=RSI_PERIOD + SMA_PERIOD)  # Stores 5-minute close prices

# === INDICATORS ===
def compute_rsi(prices, period=14):
    df = pd.Series(prices)
    delta = df.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_sma(prices, period):
    return pd.Series(prices).rolling(period).mean()

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Error: {e}")

# === AGGREGATION ===
def aggregate_5min_data():
    global minute_data
    if not minute_data:
        return None

    # Aggregate to create a 5-minute "candle"
    df = pd.DataFrame(minute_data, columns=["timestamp", "price"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    candle = {
        'open': df['price'].iloc[0],
        'high': df['price'].max(),
        'low': df['price'].min(),
        'close': df['price'].iloc[-1]
    }
    minute_data = []  # Reset for the next 5-minute period
    return candle

# === ORDER BOOK PROCESSING ===
def calculate_imbalance(data):
    bids = data['bids'][:10]
    asks = data['asks'][:10]
    bid_vol = sum(float(qty) for price, qty in bids)
    ask_vol = sum(float(qty) for price, qty in asks)
    if bid_vol + ask_vol == 0:
        return 50
    return (bid_vol / (bid_vol + ask_vol)) * 100

# === CALLBACKS ===
def on_message(ws, message):
    global price_data, minute_data

    msg = json.loads(message)
    if msg['event'] == 'data':
        data = msg['data']
        last_price = float(data['bids'][0][0])
        timestamp = datetime.utcnow()

        # Add raw data to 5-minute aggregation
        minute_data.append((timestamp, last_price))

        # Check if a new 5-minute period has started
        if len(minute_data) > 1 and (minute_data[-1][0].minute // 5 != minute_data[-2][0].minute // 5):
            candle = aggregate_5min_data()
            if candle:
                price_data.append(candle['close'])

        # Ensure enough data is available for indicators
        if len(price_data) < max(RSI_PERIOD, SMA_PERIOD):
            return

        # Calculate indicators based on 5-minute close prices
        rsi = compute_rsi(list(price_data), RSI_PERIOD).iloc[-1]
        sma = compute_sma(list(price_data), SMA_PERIOD).iloc[-1]

        print(f"[{timestamp}] Price: {last_price:.4f}, RSI: {rsi:.2f}, SMA: {sma:.4f}")

        # === BUY / SELL LOGIC ===
        if last_price > sma and rsi < 40:
            msg = f"📈 BUY SIGNAL\nPrice: {last_price:.4f}\nRSI: {rsi:.2f}"
            print(msg)
            send_telegram_alert(msg)

        elif last_price < sma and rsi > 60:
            msg = f"📉 SELL SIGNAL\nPrice: {last_price:.4f}\nRSI: {rsi:.2f}"
            print(msg)
            send_telegram_alert(msg)

def on_open(ws):
    print("✅ Connected to Bitstamp WebSocket")
    subscribe_msg = {
        "event": "bts:subscribe",
        "data": {
            "channel": f"order_book_{SYMBOL}"
        }
    }
    ws.send(json.dumps(subscribe_msg))

def on_close(ws):
    print("❌ Disconnected from Bitstamp WebSocket")

def start_ws():
    url = "wss://ws.bitstamp.net"
    ws = websocket.WebSocketApp(url,
                                 on_open=on_open,
                                 on_message=on_message,
                                 on_close=on_close)
    ws.run_forever()

# === MAIN ENTRY ===
if __name__ == "__main__":
    ws_thread = threading.Thread(target=start_ws)
    ws_thread.start()
