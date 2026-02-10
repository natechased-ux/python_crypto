import asyncio
import json
import websockets
import requests
import csv
import os
from datetime import datetime
from collections import defaultdict

# ---------------------
# Settings
# ---------------------
SYMBOLS = {
    "BTC-USD": 250_000,
    "ETH-USD": 100_000,
    "SOL-USD": 50_000
}

CSV_FILE = "whale_walls_log.csv"

# Telegram settings
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"


#personal chat id:"7967738614"
# Order book storage
order_books = {symbol: {"bids": defaultdict(float), "asks": defaultdict(float)} for symbol in SYMBOLS}
last_alerted_walls = set()  # To avoid duplicate alerts


# ---------------------
# CSV Logging
# ---------------------
def log_to_csv(symbol, side, price, size, usd_value):
    """Log whale wall to CSV file."""
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "side", "price", "size", "usd_value"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            symbol,
            side,
            f"{price:.2f}",
            f"{size:.4f}",
            f"{usd_value:.0f}"
        ])


# ---------------------
# Telegram alert
# ---------------------
def send_telegram_message(text):
    """Send message to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        print("⚠️ Telegram send error:", e)


# ---------------------
# Wall detection
# ---------------------
def detect_liquidity_walls(symbol):
    global last_alerted_walls
    bids = order_books[symbol]["bids"]
    asks = order_books[symbol]["asks"]
    threshold = SYMBOLS[symbol]

    buy_walls = [(p, s) for p, s in bids.items() if p * s >= threshold]
    sell_walls = [(p, s) for p, s in asks.items() if p * s >= threshold]

    new_alerts = []
    for price, size in sorted(buy_walls, reverse=True):
        wall_id = f"{symbol}-BUY-{price}"
        usd_value = price * size
        if wall_id not in last_alerted_walls:
            msg = f"🐋 [BUY WALL] {symbol} ${price:,.2f} x {size:,.4f} = ${usd_value:,.0f}"
            print(msg)
            send_telegram_message(msg)
            log_to_csv(symbol, "BUY", price, size, usd_value)
            new_alerts.append(wall_id)

    for price, size in sorted(sell_walls):
        wall_id = f"{symbol}-SELL-{price}"
        usd_value = price * size
        if wall_id not in last_alerted_walls:
            msg = f"🐋 [SELL WALL] {symbol} ${price:,.2f} x {size:,.4f} = ${usd_value:,.0f}"
            print(msg)
            send_telegram_message(msg)
            log_to_csv(symbol, "SELL", price, size, usd_value)
            new_alerts.append(wall_id)

    last_alerted_walls.update(new_alerts)


# ---------------------
# Process WebSocket messages
# ---------------------
async def process_message(msg):
    if msg.get("type") == "snapshot":
        symbol = msg["product_id"]
        order_books[symbol]["bids"] = defaultdict(float, {float(p): float(s) for p, s in msg["bids"]})
        order_books[symbol]["asks"] = defaultdict(float, {float(p): float(s) for p, s in msg["asks"]})
        detect_liquidity_walls(symbol)

    elif msg.get("type") == "l2update":
        symbol = msg["product_id"]
        bids = order_books[symbol]["bids"]
        asks = order_books[symbol]["asks"]

        for side, price, size in msg["changes"]:
            price = float(price)
            size = float(size)
            if side == "buy":
                if size == 0.0:
                    bids.pop(price, None)
                else:
                    bids[price] = size
            else:
                if size == 0.0:
                    asks.pop(price, None)
                else:
                    asks[price] = size

        detect_liquidity_walls(symbol)


# ---------------------
# WebSocket connection
# ---------------------
async def orderbook_listener():
    uri = "wss://ws-feed.exchange.coinbase.com"
    async with websockets.connect(uri) as ws:
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": list(SYMBOLS.keys()),
            "channels": ["level2"]
        }
        await ws.send(json.dumps(subscribe_msg))

        print(f"📡 Connected to Coinbase: {', '.join(SYMBOLS.keys())}")
        print("🔍 Monitoring large liquidity walls...\n")

        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                await process_message(data)
            except Exception as e:
                print("⚠️ Error:", e)
                break


if __name__ == "__main__":
    asyncio.run(orderbook_listener())
