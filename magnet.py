import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta, timezone
from ta.momentum import StochRSIIndicator

# === CONFIGURATION ===
COINS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
         "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
         "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd", "avax-usd", "xcn-usd", "uni-usd",
         "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
         "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd", "qnt-usd", "tia-usd", "ip-usd",
         "pnut-usd", "apt-usd", "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "pyth-usd", "sand-usd",
         "morpho-usd", "mana-usd", "velo-usd", "coti-usd", "c98-usd", "axs-usd"]

BASE_URL = "https://api.exchange.coinbase.com"
CANDLE_GRANULARITY_5M = 300
CANDLE_GRANULARITY_1H = 3600
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

BIN_WIDTH_PCT = 0.002
CLUSTER_VOLUME_PERCENTILE = 99.9
CLUSTER_MIN_DISTANCE = 0.75
CLUSTER_MAX_DISTANCE = 3.0
SL_BUFFER_PCT = 0.003
COOLDOWN_MINUTES = 30

last_alert_time = {}

# === UTILITY FUNCTIONS ===

def fetch_candles(product_id, granularity, lookback=300):
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=granularity * lookback)
    url = f"{BASE_URL}/products/{product_id}/candles?granularity={granularity}&start={start.isoformat()}&end={end.isoformat()}"
    time.sleep(0.2)
    response = requests.get(url).json()
    df = pd.DataFrame(response, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.sort_values("time")

def fetch_order_book(product_id, level=2):
    url = f"{BASE_URL}/products/{product_id}/book?level={level}"
    time.sleep(0.2)
    return requests.get(url).json()

def fetch_live_price(product_id):
    url = f"{BASE_URL}/products/{product_id}/ticker"
    time.sleep(0.2)
    return float(requests.get(url).json()["price"])

def calculate_stoch_rsi(df, length=14, smooth_k=3, smooth_d=3):
    indicator = StochRSIIndicator(close=df["close"], window=length, smooth1=smooth_k, smooth2=smooth_d)
    df["stoch_k"] = indicator.stochrsi_k()
    df["stoch_d"] = indicator.stochrsi_d()
    return df

def get_ema_200(df):
    if len(df) < 200:
        return None
    closes = df["close"].sort_index()
    return closes.ewm(span=200, adjust=False).mean().iloc[-1]

def detect_clusters_binned(orders, side, current_price, product_id=""):
    df = pd.DataFrame(orders, columns=["price", "size", "num_orders"])
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    df = df[(df["price"] >= current_price * 0.95) & (df["price"] <= current_price * 1.05)]
    if df.empty:
        return []

    bin_width = current_price * BIN_WIDTH_PCT
    df["bin"] = df["price"].apply(lambda p: round(p / bin_width) * bin_width)
    clusters = df.groupby("bin")

    cluster_dfs = [group for _, group in clusters]
    cluster_volumes = [c["size"].sum() for c in cluster_dfs]

    if not cluster_volumes:
        return []

    threshold = np.percentile(cluster_volumes, CLUSTER_VOLUME_PERCENTILE)
    whale_clusters = [c for c, vol in zip(cluster_dfs, cluster_volumes) if vol >= threshold]
    return whale_clusters

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=payload)

def format_price(p):
    return f"{p:.8f}" if p < 0.1 else f"{p:.4f}"

# === MAIN LOGIC ===

def process_coin(coin):
    global last_alert_time
    now = datetime.now(timezone.utc)
    if coin in last_alert_time and (now - last_alert_time[coin]).total_seconds() < COOLDOWN_MINUTES * 60:
        return

    try:
        price = fetch_live_price(coin)
        book = fetch_order_book(coin)
        df_5m = fetch_candles(coin, CANDLE_GRANULARITY_5M)
        df_1h = fetch_candles(coin, CANDLE_GRANULARITY_1H, lookback=250)
        df_1h = calculate_stoch_rsi(df_1h)

        if len(df_1h) < 2 or df_1h["stoch_k"].isna().iloc[-1] or df_1h["stoch_k"].isna().iloc[-2]:
            return

        k_now = df_1h["stoch_k"].iloc[-1]
        k_prev = df_1h["stoch_k"].iloc[-2]
        d_now = df_1h["stoch_d"].iloc[-1]

        ema_200 = get_ema_200(df_1h)
        if ema_200 is None:
            return

        above_ema = price > ema_200

        bids = detect_clusters_binned(book["bids"], "bids", price, coin)
        asks = detect_clusters_binned(book["asks"], "asks", price, coin)

        # === LONG SCENARIO ===
        if above_ema:
            for cluster in asks:
                cluster_price = cluster["price"].median()
                distance = abs(price - cluster_price) / price * 100
                if cluster_price > price and CLUSTER_MIN_DISTANCE <= distance <= CLUSTER_MAX_DISTANCE and k_now > k_prev and k_now > 5 and k_now < 90:
                    tp = cluster_price
                    sl = price - SL_BUFFER_PCT * price
                    msg = (f"🚀 LONG {coin.upper()} | EMA200: Above\n"
                           f"Price: {format_price(price)}\nTP: {format_price(tp)}\nSL: {format_price(sl)}\n"
                           f"Stoch RSI rising (K: {k_now:.2f}, D: {d_now:.2f})")
                    send_telegram_message(msg)
                    last_alert_time[coin] = now
                    return

        # === SHORT SCENARIO ===
        if not above_ema:
            for cluster in bids:
                cluster_price = cluster["price"].median()
                distance = abs(price - cluster_price) / price * 100
                if cluster_price < price and CLUSTER_MIN_DISTANCE <= distance <= CLUSTER_MAX_DISTANCE and k_now < k_prev and k_now < 95 and k_now > 10:
                    tp = cluster_price
                    sl = price + SL_BUFFER_PCT * price
                    msg = (f"🔻 SHORT {coin.upper()} | EMA200: Below\n"
                           f"Price: {format_price(price)}\nTP: {format_price(tp)}\nSL: {format_price(sl)}\n"
                           f"Stoch RSI falling (K: {k_now:.2f}, D: {d_now:.2f})")
                    send_telegram_message(msg)
                    last_alert_time[coin] = now
                    return

    except Exception as e:
        print(f"Error processing {coin}: {e}")

# === LOOP ===
if __name__ == "__main__":
    while True:
        print(f"\n[{datetime.now(timezone.utc).isoformat()}] Scanning for magnet scalps...")
        for coin in COINS:
            process_coin(coin)
        time.sleep(60)
