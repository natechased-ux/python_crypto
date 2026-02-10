# whale_visualizer.py
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
symbol = "plume-usd"
granularity = 3600
aggregation_threshold = 0.001
whale_cluster_percentile = 90
max_price_distance_percent = 10

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

def fetch_data(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    res = requests.get(url)
    df = pd.DataFrame(res.json(), columns=["time", "low", "high", "open", "close", "volume"])
    df = df.sort_values(by="time")
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def visualize_clusters(symbol):
    df = fetch_data(symbol)
    last_price = df['close'].iloc[-1]

    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
    res = requests.get(url)
    book = res.json()

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

if __name__ == "__main__":
    visualize_clusters(symbol)
