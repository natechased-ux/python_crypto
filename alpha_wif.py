import websocket
import json
import threading
import requests
import pandas as pd
from datetime import datetime

# === CONFIG ===
SYMBOL = 'wifusd'
RSI_PERIOD = 14
EMA_FAST = 12
EMA_SLOW = 26
IMBALANCE_BUY = 60
IMBALANCE_SELL = 40
RSI_BUY_LEVEL = 30
RSI_SELL_LEVEL = 70
PRICE_CHANGE_THRESHOLD = 0.1  # Minimum percentage price change for alerts (e.g., 1%)
TELEGRAM_TOKEN = '8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw'
TELEGRAM_CHAT_ID = '7967738614'

price_data = []

# === TELEGRAM ALERT FUNCTION ===
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Error: {e}")

# === INDICATORS ===
def compute_rsi(prices, period):
    df = pd.Series(prices)
    delta = df.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_ema(prices, period):
    return pd.Series(prices).ewm(span=period).mean()

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
        if len(price_data) < max(RSI_PERIOD, EMA_SLOW):
            return  # Wait until enough data is collected

        # Calculate indicators
        rsi = compute_rsi(price_data, RSI_PERIOD).iloc[-1]
        ema_fast = compute_ema(price_data, EMA_FAST).iloc[-1]
        ema_slow = compute_ema(price_data, EMA_SLOW).iloc[-1]
        price_change = abs((last_price - ema_slow) / ema_slow) * 100 if ema_slow != 0 else 0

        # Print data for debugging
        print(f"[{timestamp}] Price: {last_price:.4f}, RSI: {rsi:.2f}, EMA Fast: {ema_fast:.4f}, EMA Slow: {ema_slow:.4f}, Imbalance: {imbalance:.2f}%, Price Change: {price_change:.2f}%")

        # === BUY / SELL LOGIC ===
        if price_change >= PRICE_CHANGE_THRESHOLD:  # Only consider significant price changes
            if ema_fast > ema_slow and rsi < RSI_BUY_LEVEL and imbalance > IMBALANCE_BUY:
                msg = f"📈 BUY WIF\nPrice: {last_price:.4f}\nRSI: {rsi:.2f}\nEMA Fast: {ema_fast:.4f}\nEMA Slow: {ema_slow:.4f}\nImbalance: {imbalance:.2f}%\nPrice Change: {price_change:.2f}%"
                print(msg)
                send_telegram_alert(msg)

            elif ema_fast < ema_slow and rsi > RSI_SELL_LEVEL and imbalance < IMBALANCE_SELL:
                msg = f"📉 SELL WIF\nPrice: {last_price:.4f}\nRSI: {rsi:.2f}\nEMA Fast: {ema_fast:.4f}\nEMA Slow: {ema_slow:.4f}\nImbalance: {imbalance:.2f}%\nPrice Change: {price_change:.2f}%"
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

# === MAIN FUNCTION ===
def start_ws():
    url = "wss://ws.bitstamp.net"
    ws = websocket.WebSocketApp(url,
                                 on_open=on_open,
                                 on_message=on_message,
                                 on_close=on_close)
    ws.run_forever()

if __name__ == "__main__":
    ws_thread = threading.Thread(target=start_ws)
    ws_thread.start()
