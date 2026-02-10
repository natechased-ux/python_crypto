import requests
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
symbol = "PI_XBTUSD"  # Kraken BTC perpetual futures
bin_width_percent = 0.2
price_range_percent = 5
whale_cluster_percentile = 90  # Top X% of volume is considered 'whale cluster'

# === FETCH ORDER BOOK ===
def fetch_order_book(symbol):
    url = f"https://futures.kraken.com/derivatives/api/v3/orderbook?symbol={symbol}"
    res = requests.get(url)
    if res.status_code != 200:
        raise Exception("Failed to fetch Kraken Futures order book")
    data = res.json()["orderBook"]
    bids = [(float(p), float(q)) for p, q in data["bids"]]
    asks = [(float(p), float(q)) for p, q in data["asks"]]
    return bids, asks

# === FETCH CURRENT PRICE ===
def fetch_current_price(symbol):
    url = "https://futures.kraken.com/derivatives/api/v3/tickers"
    res = requests.get(url)
    data = res.json()["tickers"]
    for t in data:
        if t["symbol"] == symbol:
            return float(t["last"])
    raise Exception("Symbol not found in tickers")

# === BIN ORDERS ===
def bin_orders(orders, current_price, bin_width_percent):
    bin_width = current_price * bin_width_percent / 100
    binned = {}
    for price, size in orders:
        if abs(price - current_price) / current_price * 100 > price_range_percent:
            continue  # Skip orders too far from price
        bin_price = round(price / bin_width) * bin_width
        binned[bin_price] = binned.get(bin_price, 0) + size
    return [(p, q) for p, q in binned.items()]

# === MAIN VISUALIZER ===
def visualize_clusters(symbol):
    price = fetch_current_price(symbol)
    bids, asks = fetch_order_book(symbol)

    binned_bids = bin_orders(bids, price, bin_width_percent)
    binned_asks = bin_orders(asks, price, bin_width_percent)

    bid_sizes = [q for _, q in binned_bids]
    ask_sizes = [q for _, q in binned_asks]

    bid_cutoff = pd.Series(bid_sizes).quantile(whale_cluster_percentile / 100) if bid_sizes else 0
    ask_cutoff = pd.Series(ask_sizes).quantile(whale_cluster_percentile / 100) if ask_sizes else 0

    whale_bids = [(p, q) for p, q in binned_bids if q >= bid_cutoff]
    whale_asks = [(p, q) for p, q in binned_asks if q >= ask_cutoff]

    # === PLOT ===
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(price, color='black', linestyle='--', linewidth=1)
    ax.text(0.98, price, f'Price: {price:.2f}', va='center', ha='right',
            transform=ax.get_yaxis_transform(), fontsize=10, color='black')

    for p, q in whale_bids:
        ax.scatter(q, p, color='green', s=60, label='Bid Cluster')
    for p, q in whale_asks:
        ax.scatter(q, p, color='red', s=60, label='Ask Cluster')

    ax.set_xlabel("Order Size (Contracts)")
    ax.set_ylabel("Price")
    ax.set_title(f"{symbol} Whale Clusters (Top {whale_cluster_percentile}% ±{price_range_percent}%)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# === RUN ===
if __name__ == "__main__":
    visualize_clusters(symbol)
