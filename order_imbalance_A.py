import json
import time
import threading
import websocket
import pandas as pd
from datetime import datetime, timedelta
from collections import deque, defaultdict
import requests
import os

# === Config ===
COINS = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LINK-USD']
ROLLING_WINDOW_SECONDS = 30
IMBALANCE_THRESHOLD = 0.99
VOLUME_WINDOW = 240
VOLUME_PERCENTILE = 99
COOLDOWN_SECONDS = 300
AGGREGATION_INTERVAL_SECONDS = 60  # aggregate every 1 minute
TELEGRAM_BOT_TOKEN = '8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw'
TELEGRAM_CHAT_ID = '7967738614'
LOG_FILE = "volume_log.csv"

# === In-memory storage ===
trade_buffer = {coin: deque() for coin in COINS}
volume_history = {coin: {'buy': deque(maxlen=VOLUME_WINDOW), 'sell': deque(maxlen=VOLUME_WINDOW)} for coin in COINS}
last_alert_time = defaultdict(lambda: 0)
history_ready = {coin: False for coin in COINS}
aggregated_volume = {coin: {'buy': 0, 'sell': 0, 'start_time': datetime.utcnow()} for coin in COINS}

# === Telegram ===
def send_telegram_message(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print(f"Telegram error: {e}")

# === CSV Logging ===
def ensure_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,coin,buy,sell\n")

def log_volume_to_csv(timestamp, coin, buy_vol, sell_vol):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"{timestamp},{coin},{buy_vol:.2f},{sell_vol:.2f}\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"❌ CSV write error: {e}")

# === Load history ===
def load_volume_history():
    if not os.path.exists(LOG_FILE):
        return
    try:
        df = pd.read_csv(LOG_FILE, names=["timestamp", "coin", "buy", "sell"], skiprows=1)
        for coin in COINS:
            recent = df[df["coin"] == coin].tail(VOLUME_WINDOW)
            volume_history[coin]['buy'].extend(recent["buy"].tolist())
            volume_history[coin]['sell'].extend(recent["sell"].tolist())
    except Exception as e:
        print(f"⚠️ Failed to load volume history: {e}")

# === WebSocket ===
def on_open(ws):
    subscribe_msg = {
        "type": "subscribe",
        "channels": [{"name": "matches", "product_ids": COINS}]
    }
    ws.send(json.dumps(subscribe_msg))

def on_message(ws, message):
    data = json.loads(message)
    if data.get('type') != 'match':
        return

    coin = data['product_id']
    side = data['side']
    price = float(data['price'])
    size = float(data['size'])
    timestamp = datetime.strptime(data['time'], '%Y-%m-%dT%H:%M:%S.%fZ')

    notional = price * size
    trade = {'timestamp': timestamp, 'side': side, 'notional': notional, 'price': price}
    trade_buffer[coin].append(trade)

    if side == 'buy':
        aggregated_volume[coin]['buy'] += notional
    else:
        aggregated_volume[coin]['sell'] += notional

def start_websocket():
    ws = websocket.WebSocketApp("wss://ws-feed.exchange.coinbase.com",
                                on_open=on_open,
                                on_message=on_message)
    ws.run_forever()

# === Check loop ===
def start_check_loop():
    while True:
        for coin in COINS:
            now = datetime.utcnow()
            agg = aggregated_volume[coin]
            elapsed = (now - agg['start_time']).total_seconds()
            if elapsed >= AGGREGATION_INTERVAL_SECONDS:
                volume_history[coin]['buy'].append(agg['buy'])
                volume_history[coin]['sell'].append(agg['sell'])
                log_volume_to_csv(now, coin, agg['buy'], agg['sell'])
                aggregated_volume[coin] = {'buy': 0, 'sell': 0, 'start_time': now}
            check_signal(coin)
        time.sleep(5)

# === Signal logic ===
def check_signal(coin):
    now = time.time()
    cutoff = datetime.utcnow() - timedelta(seconds=ROLLING_WINDOW_SECONDS)
    while trade_buffer[coin] and trade_buffer[coin][0]['timestamp'] < cutoff:
        trade_buffer[coin].popleft()

    trades = trade_buffer[coin]
    if not trades:
        return

    buys = sum(t['notional'] for t in trades if t['side'] == 'buy')
    sells = sum(t['notional'] for t in trades if t['side'] == 'sell')
    total = buys + sells
    if total == 0:
        return

    if not history_ready[coin]:
        if len(volume_history[coin]['buy']) >= VOLUME_WINDOW:
            history_ready[coin] = True
        else:
            return

    if now - last_alert_time[coin] < COOLDOWN_SECONDS:
        return

    imbalance = buys / total
    latest_price = trades[-1]['price']

    if imbalance > IMBALANCE_THRESHOLD:
        threshold = pd.Series(volume_history[coin]['buy']).quantile(VOLUME_PERCENTILE / 100)
        if buys < threshold:
            return
        signal = "🔵 BUY"
    elif imbalance < (1 - IMBALANCE_THRESHOLD):
        threshold = pd.Series(volume_history[coin]['sell']).quantile(VOLUME_PERCENTILE / 100)
        if sells < threshold:
            return
        signal = "🔴 SELL"
    else:
        return

    msg = (
        f"{signal} signal for {coin}"
        f"Price: ${latest_price:.4f}"
        f"Imbalance: {imbalance:.2%}"
        f"Buy Vol: ${buys:,.2f} | Sell Vol: ${sells:,.2f}"
        f"Time: {datetime.utcnow()} UTC"
    )
    send_telegram_message(msg)
    last_alert_time[coin] = now

# === Start ===
if __name__ == "__main__":
    ensure_log_file()
    load_volume_history()

    ws_thread = threading.Thread(target=start_websocket)
    ws_thread.start()

    check_thread = threading.Thread(target=start_check_loop)
    check_thread.start()

    ws_thread.join()
    check_thread.join()
