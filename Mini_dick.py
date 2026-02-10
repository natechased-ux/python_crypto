import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
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
CANDLE_GRANULARITY = 300  # 5 minutes
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"

BIN_WIDTH_PCT = 0.002
CLUSTER_VOLUME_PERCENTILE = 99.9
CLUSTER_MIN_DISTANCE = 1
CLUSTER_MAX_DISTANCE = 2.0
SL_BUFFER_PCT = 0.003
COOLDOWN_MINUTES = 0

last_alert_time = {}

def fetch_order_book(product_id, level=2):
    url = f"{BASE_URL}/products/{product_id}/book?level={level}"
    time.sleep(0.2)
    return requests.get(url).json()

def fetch_live_price(product_id):
    url = f"{BASE_URL}/products/{product_id}/ticker"
    time.sleep(0.2)
    response = requests.get(url).json()
    return float(response["price"])

def fetch_candles(product_id, granularity):
    end = datetime.utcnow()
    start = end - timedelta(seconds=granularity * 300)
    url = f"{BASE_URL}/products/{product_id}/candles?granularity={granularity}&start={start.isoformat()}&end={end.isoformat()}"
    time.sleep(0.2)
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.sort_values("time")

def calculate_stoch_rsi(df, length=14, smooth_k=3, smooth_d=3):
    indicator = StochRSIIndicator(close=df["close"], window=length, smooth1=smooth_k, smooth2=smooth_d)
    df["stoch_k"] = indicator.stochrsi_k()
    return df

def detect_clusters_binned(orders, side, current_price, product_id=""):
    df = pd.DataFrame(orders, columns=["price", "size", "num_orders"])
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    df = df[df["price"] > 0]
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

    volume_threshold = np.percentile(cluster_volumes, CLUSTER_VOLUME_PERCENTILE)
    print(f"[{product_id.upper()}] {side.upper()} cluster volumes: {cluster_volumes}, threshold: {volume_threshold:.2f}")

    whale_clusters = [c for c, vol in zip(cluster_dfs, cluster_volumes) if vol >= volume_threshold]
    return whale_clusters

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=payload)

def process_coin(coin):
    global last_alert_time
    now = datetime.utcnow()
    if coin in last_alert_time and (now - last_alert_time[coin]).total_seconds() < COOLDOWN_MINUTES * 60:
        return

    try:
        book = fetch_order_book(coin)
        current_price = fetch_live_price(coin)
        candles = fetch_candles(coin, CANDLE_GRANULARITY)
        candles = calculate_stoch_rsi(candles)

        if len(candles) < 2 or candles["stoch_k"].isna().iloc[-1] or candles["stoch_k"].isna().iloc[-2]:
            print(f"{coin.upper()} — insufficient or invalid Stoch RSI data")
            return

        k_now = candles["stoch_k"].iloc[-1]
        k_prev = candles["stoch_k"].iloc[-2]

        bids = detect_clusters_binned(book["bids"], "bids", current_price, coin)
        asks = detect_clusters_binned(book["asks"], "asks", current_price, coin)

        print(f"{coin.upper()} — k: {k_now:.2f}, prev k: {k_prev:.2f}, Bid clusters: {len(bids)}, Ask clusters: {len(asks)}")

        for cluster in asks:
            cluster_price = cluster["price"].median()
            distance = abs(current_price - cluster_price) / current_price * 100
            print(f"Checking LONG for {coin.upper()} ➜ Current: {current_price:.4f}, Cluster: {cluster_price:.4f}, Distance: {distance:.4f}%")
            print(f"Stoch RSI K: now={k_now:.2f}, prev={k_prev:.2f}")
            print(f"Passes distance? {CLUSTER_MIN_DISTANCE <= distance <= CLUSTER_MAX_DISTANCE}")
            print(f"Passes RSI? {k_now > k_prev}")

            if cluster_price > current_price:
                print(f"Cluster > Current? {cluster_price > current_price}")
                print(f"✅ ALL CONDITIONS PASSED — sending alert for {coin.upper()}")

                
                if CLUSTER_MIN_DISTANCE <= distance <= CLUSTER_MAX_DISTANCE and k_now > k_prev and k_now > 20:
                    tp = cluster_price
                    sl = current_price - SL_BUFFER_PCT * current_price
                    message = f"🚀 LONG (toward ask cluster) for {coin.upper()}\nPrice: {current_price:.4f}\nTarget cluster @ {tp:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\nStoch RSI rising (k: {k_now:.2f})"
                    send_telegram_message(message)
                    last_alert_time[coin] = now
                    return

        for cluster in bids:
            cluster_price = cluster["price"].median()
            distance = abs(current_price - cluster_price) / current_price * 100
            print(f"Checking SHORT for {coin.upper()} ➜ Current: {current_price:.4f}, Cluster: {cluster_price:.4f}, Distance: {distance:.4f}%")
            print(f"Stoch RSI K: now={k_now:.2f}, prev={k_prev:.2f}")
            print(f"Passes distance? {CLUSTER_MIN_DISTANCE <= distance <= CLUSTER_MAX_DISTANCE}")
            print(f"Passes RSI? {k_now < k_prev}")

            if cluster_price < current_price:
                print(f"Cluster < Current? {cluster_price < current_price}")
                print(f"✅ ALL CONDITIONS PASSED — sending alert for {coin.upper()}")

                
                if CLUSTER_MIN_DISTANCE <= distance <= CLUSTER_MAX_DISTANCE and k_now < k_prev and k_now < 80:
                    tp = cluster_price
                    sl = current_price + SL_BUFFER_PCT * current_price
                    message = f"🔻 SHORT (toward ask cluster) for {coin.upper()}\nPrice: {current_price:.4f}\nTarget cluster @ {tp:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\nStoch RSI falling (k: {k_now:.2f})"
                    send_telegram_message(message)
                    last_alert_time[coin] = now
                    return

    except Exception as e:
        print(f"Error processing {coin}: {e}")

# === LOOP: run every 5 minutes ===
if __name__ == "__main__":
    while True:
        print(f"\n[{datetime.utcnow().isoformat()}] Scanning for magnet scalps...")
        for coin in COINS:
            process_coin(coin)
        time.sleep(60)
