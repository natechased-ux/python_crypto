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
IMBALANCE_THRESHOLD = 0.9
VOLUME_WINDOW = 60
VOLUME_PERCENTILE = 90
COOLDOWN_SECONDS = 300
AGGREGATION_INTERVAL_SECONDS = 300  # aggregate every 5 minutes
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

# === Format price based on size ===
def price_format(price):
    if price >= 100:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:,.4f}"
    elif price >= 0.01:
        return f"{price:,.6f}"
    else:
        return f"{price:,.8f}"

# === CSV logging ===
def ensure_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,coin,buy,sell\n")
        print(f"📄 Created new log file: {LOG_FILE}")

def log_volume_to_csv(timestamp, coin, buy_vol, sell_vol):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"{timestamp},{coin},{buy_vol:.2f},{sell_vol:.2f}\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"📈 CSV logged: {timestamp} {coin} ${buy_vol:.2f} / ${sell_vol:.2f}")
    except Exception as e:
        print(f"❌ CSV write error: {e}")

# === Load history on restart ===
def load_volume_history():
    if not os.path.exists(LOG_FILE):
        print("No existing volume log found.")
        return
    try:
        df = pd.read_csv(LOG_FILE, names=["timestamp", "coin", "buy", "sell"], skiprows=1)
        for coin in COINS:
            recent = df[df["coin"] == coin].tail(VOLUME_WINDOW)
            volume_history[coin]['buy'].extend(recent["buy"].tolist())
            volume_history[coin]['sell'].extend(recent["sell"].tolist())
        print("✅ Volume history restored from CSV.")
    except Exception as e:
        print(f"⚠️ Failed to load volume history: {e}")

# === WebSocket callbacks ===
def on_open(ws):
    print("WebSocket connected.")
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
    trade = {'timestamp': timestamp, 'side': side, 'notional': notional}
    trade_buffer[coin].append(trade)

    if side == 'buy':
        aggregated_volume[coin]['buy'] += notional
    else:
        aggregated_volume[coin]['sell'] += notional

    

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, *_):
    print("WebSocket closed. Reconnecting in 5s...")
    time.sleep(5)
    start_websocket()

def start_websocket():
    ws = websocket.WebSocketApp("wss://ws-feed.exchange.coinbase.com",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()

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

# === Main logic ===
def check_signal(coin):
    now = time.time()

    # Trim old trades
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
        if len(volume_history[coin]['buy']) >= VOLUME_WINDOW and len(volume_history[coin]['sell']) >= VOLUME_WINDOW:
            print(f"✅ History ready for {coin}")
            history_ready[coin] = True
        else:
            return

    if now - last_alert_time[coin] < COOLDOWN_SECONDS:
        return

    imbalance = buys / total

    if imbalance > IMBALANCE_THRESHOLD:
        side = "buy"
        threshold = pd.Series(volume_history[coin]['buy']).quantile(VOLUME_PERCENTILE / 100)
        if buys < threshold:
            return
        signal = "🔵 BUY"
    elif imbalance < (1 - IMBALANCE_THRESHOLD):
        side = "sell"
        threshold = pd.Series(volume_history[coin]['sell']).quantile(VOLUME_PERCENTILE / 100)
        if sells < threshold:
            return
        signal = "🔴 SELL"
    else:
        return

    msg = (
        f"{signal} signal for {coin}\n"
        f"Buy Vol: ${buys:,.2f} | Sell Vol: ${sells:,.2f}\n"
        f"Imbalance: {imbalance:.2%}\n"
        f"{side.capitalize()} volume > {VOLUME_PERCENTILE}th percentile\n"
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
