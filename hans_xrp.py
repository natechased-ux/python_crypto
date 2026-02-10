import websocket
import json
import threading
import requests
import pandas as pd
from datetime import datetime

# === CONFIG ===
SYMBOL = 'xrpusd'
RSI_PERIOD = 14
SMA_PERIOD = 50
PRICE_CHANGE_THRESHOLD = 0.1  # Percentage threshold for alerts (adjusted)
IMBALANCE_BUY = 60  # Buy imbalance threshold (adjusted)
IMBALANCE_SELL = 40  # Sell imbalance threshold (adjusted)
RSI_BUY_LEVEL = 35  # RSI threshold for buy (adjusted)
RSI_SELL_LEVEL = 65  # RSI threshold for sell (adjusted)
TELEGRAM_TOKEN = '8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw'
TELEGRAM_CHAT_ID = '7967738614'

price_data = []

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
    global price_data

    msg = json.loads(message)
    if msg['event'] == 'data':
        data = msg['data']
        imbalance = calculate_imbalance(data)
        last_price = float(data['bids'][0][0])
        timestamp = datetime.utcnow()

        # Append price for indicators
        price_data.append(last_price)
        if len(price_data) < max(RSI_PERIOD, SMA_PERIOD) + 1:
            return  # Wait until we have enough data

        # Calculate indicators
        rsi = compute_rsi(price_data, RSI_PERIOD).iloc[-1]
        sma = compute_sma(price_data, SMA_PERIOD).iloc[-1]
        price_change_from_sma = abs((last_price - sma) / sma) * 100 if sma != 0 else 0

        print(f"[{timestamp}] Price: {last_price:.4f}, RSI: {rsi:.2f}, SMA: {sma:.4f}, Imbalance: {imbalance:.2f}%, Price Change: {price_change_from_sma:.2f}%")

        # === BUY / SELL LOGIC ===
        if price_change_from_sma > PRICE_CHANGE_THRESHOLD:
            if imbalance > IMBALANCE_BUY and rsi < RSI_BUY_LEVEL and last_price > sma:
                msg = f"📈 BUY SIGNAL\nPrice: {last_price:.4f}\nRSI: {rsi:.2f}\nImbalance: {imbalance:.2f}%\nPrice Change: {price_change_from_sma:.2f}%"
                print(msg)
                send_telegram_alert(msg)

            elif imbalance < IMBALANCE_SELL and rsi > RSI_SELL_LEVEL and last_price < sma:
                msg = f"📉 SELL SIGNAL\nPrice: {last_price:.4f}\nRSI: {rsi:.2f}\nImbalance: {imbalance:.2f}%\nPrice Change: {price_change_from_sma:.2f}%"
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
