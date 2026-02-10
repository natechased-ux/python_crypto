import websocket
import json
import threading
import requests
import pandas as pd
from datetime import datetime
from collections import deque

# === CONFIG ===
SYMBOL = 'trumpusd'
RSI_PERIOD = 14
SMA_PERIOD = 50
IMBALANCE_BUY = 60
IMBALANCE_SELL = 40
PRICE_CHANGE_THRESHOLD = 0.1  # Percentage threshold for significant trend
TELEGRAM_TOKEN = '8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw'
TELEGRAM_CHAT_ID = '7967738614'

price_data = deque(maxlen=RSI_PERIOD + SMA_PERIOD)  # Stores recent prices

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

# === CALLBACKS ===
def on_message(ws, message):
    global price_data

    msg = json.loads(message)
    if msg['event'] == 'data':
        data = msg['data']
        last_price = float(data['bids'][0][0])
        price_data.append(last_price)

        # Ensure enough data is available for indicators
        if len(price_data) < max(RSI_PERIOD, SMA_PERIOD):
            return

        # Calculate indicators
        rsi = compute_rsi(list(price_data), RSI_PERIOD).iloc[-1]
        sma = compute_sma(list(price_data), SMA_PERIOD).iloc[-1]

        # Calculate percentage difference from SMA
        price_change = abs((last_price - sma) / sma) * 100

        print(f"[{datetime.utcnow()}] Price: {last_price:.4f}, RSI: {rsi:.2f}, SMA: {sma:.4f}, Change: {price_change:.2f}%")

        # === TREND REVERSAL LOGIC ===
        if price_change > PRICE_CHANGE_THRESHOLD:
            if rsi < 30 and last_price > sma:  # Reversal to upward trend
                msg = f"📈 POTENTIAL REVERSAL TO UPWARD TREND\nPrice: {last_price:.4f}\nRSI: {rsi:.2f}\nSMA: {sma:.4f}\nChange: {price_change:.2f}%"
                print(msg)
                send_telegram_alert(msg)

            elif rsi > 70 and last_price < sma:  # Reversal to downward trend
                msg = f"📉 POTENTIAL REVERSAL TO DOWNWARD TREND\nPrice: {last_price:.4f}\nRSI: {rsi:.2f}\nSMA: {sma:.4f}\nChange: {price_change:.2f}%"
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
