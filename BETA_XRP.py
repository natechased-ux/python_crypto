import websocket
import json
import requests
import pandas as pd
from datetime import datetime
from collections import deque

# === CONFIG ===
SYMBOL = "xrpusd"
VWAP_PERIOD = 20
RSI_PERIOD = 14
ORDER_IMBALANCE_THRESHOLD = 60  # Percent imbalance for alerts
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"

price_data = deque(maxlen=VWAP_PERIOD)
volume_data = deque(maxlen=VWAP_PERIOD)

# === TELEGRAM ALERT ===
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Error: {e}")

# === INDICATORS ===
def compute_rsi(prices, period=14):
    df = pd.Series(prices)
    delta = df.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_vwap(prices, volumes):
    return sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)

def calculate_imbalance(order_book):
    bids = order_book["bids"][:10]
    asks = order_book["asks"][:10]
    bid_vol = sum(float(qty) for price, qty in bids)
    ask_vol = sum(float(qty) for price, qty in asks)
    if bid_vol + ask_vol == 0:
        return 50  # Neutral
    return (bid_vol / (bid_vol + ask_vol)) * 100

# === CALLBACK FUNCTIONS ===
def on_message(ws, message):
    global price_data, volume_data

    msg = json.loads(message)
    if msg["event"] == "data":
        data = msg["data"]
        bids = data["bids"]
        asks = data["asks"]

        # Calculate mid-price and volume
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        current_price = (best_bid + best_ask) / 2
        current_volume = sum(float(qty) for price, qty in bids[:10])

        # Append to deque
        price_data.append(current_price)
        volume_data.append(current_volume)

        if len(price_data) >= VWAP_PERIOD:
            vwap = compute_vwap(price_data, volume_data)
            rsi = compute_rsi(list(price_data), RSI_PERIOD).iloc[-1]
            imbalance = calculate_imbalance(data)

            print(f"Price: {current_price:.4f}, VWAP: {vwap:.4f}, RSI: {rsi:.2f}, Imbalance: {imbalance:.2f}%")

            # === BUY SIGNAL ===
            if current_price < vwap and rsi < 30 and imbalance > ORDER_IMBALANCE_THRESHOLD:
                message = (
                    f"📈 BUY Signal\nPrice: {current_price:.4f}\nVWAP: {vwap:.4f}\nRSI: {rsi:.2f}\nImbalance: {imbalance:.2f}%"
                )
                print(message)
                send_telegram_alert(message)

            # === SELL SIGNAL ===
            elif current_price > vwap and rsi > 70 and imbalance < (100 - ORDER_IMBALANCE_THRESHOLD):
                message = (
                    f"📉 SELL Signal\nPrice: {current_price:.4f}\nVWAP: {vwap:.4f}\nRSI: {rsi:.2f}\nImbalance: {imbalance:.2f}%"
                )
                print(message)
                send_telegram_alert(message)

def on_open(ws):
    print("✅ Connected to Bitstamp WebSocket")
    subscribe_msg = {
        "event": "bts:subscribe",
        "data": {"channel": f"order_book_{SYMBOL}"},
    }
    ws.send(json.dumps(subscribe_msg))

def on_close(ws):
    print("❌ Disconnected from Bitstamp WebSocket")

# === MAIN FUNCTION ===
def main():
    ws_url = "wss://ws.bitstamp.net"
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_close=on_close,
    )
    ws.run_forever()

if __name__ == "__main__":
    main()
