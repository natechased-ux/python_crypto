import requests
import pandas as pd
from ta.volatility import AverageTrueRange
from telegram import Bot
import time
import matplotlib.pyplot as plt

# Telegram Config
TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = Bot(token=TOKEN)

# Configuration
symbols = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd",
           "syrup-usd", "fartcoin-usd", "aero-usd", "link-usd", "hbar-usd",
           "aave-usd", "fet-usd", "crv-usd", "tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd",
           "bch-usd", "inj-usd", "pepe-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
           "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
           "qnt-usd", "tia-usd", "ip-usd", "pnut-usd", "apt-usd", "vet-usd",
           "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "velo-usd", "coti-usd",
           "axs-usd"]
granularity = 3600
atr_length = 14
vwap_length = 14
aggregation_threshold = 0.001
price_range_percentage = 0.25
volume_spike_multiplier = .5
risk_reward_ratio = 1.5
whale_cluster_percentile = 95
max_price_distance_percent = 10  # Max distance from price to consider (±%)

# Helper
def group_orders(orders, threshold):
    if not orders:
        return []
    orders.sort(key=lambda x: x[0])
    groups, current = [], [orders[0]]
    for price, qty in orders[1:]:
        if abs(price - current[-1][0]) / price <= threshold:
            current.append((price, qty))
        else:
            total_qty = sum(q for _, q in current)
            avg_price = sum(p * q for p, q in current) / total_qty
            groups.append((avg_price, total_qty))
            current = [(price, qty)]
    total_qty = sum(q for _, q in current)
    avg_price = sum(p * q for p, q in current) / total_qty
    groups.append((avg_price, total_qty))
    return groups

# Fetch candles
def fetch_data(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    res = requests.get(url)
    if res.status_code != 200:
        raise Exception(f"Error fetching candles for {symbol}")
    df = pd.DataFrame(res.json(), columns=["time", "low", "high", "open", "close", "volume"])
    df = df.sort_values(by="time")
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

# Indicators
def calculate_indicators(df):
    df['vwap'] = ((df['high'] + df['low'] + df['close']) / 3 * df['volume']).rolling(vwap_length).sum() / df['volume'].rolling(vwap_length).sum()
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=atr_length)
    df['atr'] = atr.average_true_range()
    df['avg_volume'] = df['volume'].rolling(atr_length).mean()
    df['volume_spike'] = df['volume'] > volume_spike_multiplier * df['avg_volume']
    return df

# Whale cluster analysis
def analyze_whale_clusters(symbol, price):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
    res = requests.get(url)
    if res.status_code != 200:
        return None, None, None
    book = res.json()

    # Filter raw orders within ±10%
    bids = [(float(x[0]), float(x[1])) for x in book['bids'] if (price - float(x[0])) / price <= max_price_distance_percent / 100]
    asks = [(float(x[0]), float(x[1])) for x in book['asks'] if (float(x[0]) - price) / price <= max_price_distance_percent / 100]

    grouped_bids = group_orders(bids, aggregation_threshold)
    grouped_asks = group_orders(asks, aggregation_threshold)

    bid_volumes = [q for _, q in grouped_bids]
    ask_volumes = [q for _, q in grouped_asks]
    bid_cutoff = pd.Series(bid_volumes).quantile(whale_cluster_percentile / 100) if bid_volumes else 0
    ask_cutoff = pd.Series(ask_volumes).quantile(whale_cluster_percentile / 100) if ask_volumes else 0

    whale_bids = [b for b in grouped_bids if b[1] >= bid_cutoff]
    whale_asks = [a for a in grouped_asks if a[1] >= ask_cutoff]

    bid = min(whale_bids, key=lambda x: abs(x[0] - price), default=None)
    ask = min(whale_asks, key=lambda x: abs(x[0] - price), default=None)

    return bid, ask


# Analyzer
def analyze(symbol, df):
    df = calculate_indicators(df)
    row = df.iloc[-1]
    price = row['close']
    atr = row['atr']
    if pd.isna(atr) or not row['volume_spike']:
        return None

    bid, ask = analyze_whale_clusters(symbol, price)

    if bid:
        tp = price + atr * risk_reward_ratio
        sl = price - atr
        return f"{symbol.upper()} LONG\nEntry: {price:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\nSupport: {bid[0]:.4f}"
    elif ask:
        tp = price - atr * risk_reward_ratio
        sl = price + atr
        return f"{symbol.upper()} SHORT\nEntry: {price:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\nResistance: {ask[0]:.4f}"
    return None

# Send to Telegram
def notify(message):
    bot.send_message(chat_id=CHAT_ID, text=message)

# Main loop
def monitor():
    while True:
        try:
            messages = []
            for symbol in symbols:
                df = fetch_data(symbol)
                signal = analyze(symbol, df)
                if signal:
                    messages.append(signal)
            if messages:
                notify("\n\n".join(messages))
        except Exception as e:
            notify(f"Error: {str(e)}")
        time.sleep(60)

# Whale cluster visualization
def visualize_clusters(symbol):
    df = fetch_data(symbol)
    last_price = df['close'].iloc[-1]

    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
    res = requests.get(url)
    book = res.json()

    # Filter raw orders within ±10%
    bids = [(float(x[0]), float(x[1])) for x in book['bids'] if (last_price - float(x[0])) / last_price <= max_price_distance_percent / 100]
    asks = [(float(x[0]), float(x[1])) for x in book['asks'] if (float(x[0]) - last_price) / last_price <= max_price_distance_percent / 100]

    grouped_bids = group_orders(bids, aggregation_threshold)
    grouped_asks = group_orders(asks, aggregation_threshold)

    bid_volumes = [q for _, q in grouped_bids]
    ask_volumes = [q for _, q in grouped_asks]
    bid_cutoff = pd.Series(bid_volumes).quantile(whale_cluster_percentile / 100) if bid_volumes else 0
    ask_cutoff = pd.Series(ask_volumes).quantile(whale_cluster_percentile / 100) if ask_volumes else 0

    whale_bids = [(p, q) for p, q in grouped_bids if q >= bid_cutoff]
    whale_asks = [(p, q) for p, q in grouped_asks if q >= ask_cutoff]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(last_price, color='black', linestyle='--')
    ax.text(0.98, last_price, f'Price: {last_price:.2f}', va='center', ha='right',
            transform=ax.get_yaxis_transform(), fontsize=10, color='black')

    for p, q in whale_bids:
        ax.scatter(q, p, color='green', s=60)
    for p, q in whale_asks:
        ax.scatter(q, p, color='red', s=60)

    ax.set_xlabel("Size (Qty)")
    ax.set_ylabel("Price")
    ax.set_title(f"{symbol.upper()} Whale Clusters (Top {whale_cluster_percentile}%, ±{max_price_distance_percent}%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Uncomment to test visualization
visualize_clusters("sui-usd")

if __name__ == "__main__":
    monitor()
