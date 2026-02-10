import requests
import time

# Configuration parameters
whale_threshold = 1.0
price_range_percentage = 10
aggregation_threshold = 0.001
proximity_weight = 0.6
quantity_weight = 0.4
confidence_threshold = 0.1
price_change_threshold = 0.01  # Minimum 1% price difference to consider
refresh_interval = 60  # Check all symbols every minute

# Helper functions (same as before)
def group_close_prices(orders, threshold):
    if not orders:
        return []
    orders.sort(key=lambda x: x[0])
    grouped_orders = []
    current_group = [orders[0]]

    for i in range(1, len(orders)):
        current_price = orders[i][0]
        current_qty = orders[i][1]
        prev_price = current_group[-1][0]

        if abs(current_price - prev_price) / prev_price <= threshold:
            current_group.append((current_price, current_qty))
        else:
            avg_price = sum(p * q for p, q in current_group) / sum(q for _, q in current_group)
            total_qty = sum(q for _, q in current_group)
            grouped_orders.append((avg_price, total_qty))
            current_group = [(current_price, current_qty)]

    if current_group:
        avg_price = sum(p * q for p, q in current_group) / sum(q for _, q in current_group)
        total_qty = sum(q for _, q in current_group)
        grouped_orders.append((avg_price, total_qty))

    return grouped_orders

def format_price(price, mid_price):
    return f"{price:.4f}" if mid_price < 10 else f"{price:.2f}"

def analyze_symbol(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=1"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    order_book = response.json()
    best_bid = float(order_book['bids'][0][0])
    best_ask = float(order_book['asks'][0][0])
    mid_price = (best_bid + best_ask) / 2

    lower_bound = mid_price * (1 - price_range_percentage / 100)
    upper_bound = mid_price * (1 + price_range_percentage / 100)

    url_full = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
    response_full = requests.get(url_full)
    if response_full.status_code != 200:
        return None

    full_order_book = response_full.json()
    bids = [
        (float(bid[0]), float(bid[1]))
        for bid in full_order_book['bids']
        if lower_bound <= float(bid[0]) <= upper_bound and float(bid[1]) >= whale_threshold
    ]
    asks = [
        (float(ask[0]), float(ask[1]))
        for ask in full_order_book['asks']
        if lower_bound <= float(ask[0]) <= upper_bound and float(ask[1]) >= whale_threshold
    ]

    grouped_bids = group_close_prices(bids, aggregation_threshold)
    grouped_asks = group_close_prices(asks, aggregation_threshold)

    sorted_bids = sorted(grouped_bids, key=lambda x: abs(mid_price - x[0]))
    sorted_asks = sorted(grouped_asks, key=lambda x: abs(mid_price - x[0]))

    bid_score_total = sum(
        proximity_weight * (1 / abs(mid_price - bid[0])) + quantity_weight * bid[1]
        for bid in sorted_bids
    )
    ask_score_total = sum(
        proximity_weight * (1 / abs(mid_price - ask[0])) + quantity_weight * ask[1]
        for ask in sorted_asks
    )

    score_difference = bid_score_total - ask_score_total
    decision = "NO POSITION"
    execution_price = None

    if abs(score_difference) >= confidence_threshold:
        if score_difference > 0 and sorted_bids:
            closest_bid = sorted_bids[0]
            if abs(closest_bid[0] - mid_price) / mid_price > price_change_threshold:
                decision = "SHORT"
                execution_price = closest_bid[0]
        elif sorted_asks:
            closest_ask = sorted_asks[0]
            if abs(closest_ask[0] - mid_price) / mid_price > price_change_threshold:
                decision = "LONG"
                execution_price = closest_ask[0]

    return {
        "symbol": symbol,
        "decision": decision,
        "execution_price": execution_price,
        "mid_price": mid_price,
        "score_difference": score_difference,
    }

def scan_cryptocurrencies(symbols):
    results = []
    for symbol in symbols:
        result = analyze_symbol(symbol)
        if result:
            results.append(result)

    # Filter and rank results
    long_candidates = sorted(
        [res for res in results if res["decision"] == "LONG"],
        key=lambda x: x["score_difference"],
        reverse=True,
    )
    short_candidates = sorted(
        [res for res in results if res["decision"] == "SHORT"],
        key=lambda x: x["score_difference"],
        reverse=True,
    )

    print("\nTop Long Candidates:")
    for candidate in long_candidates[:3]:
        print(
            f"{candidate['symbol']} - LONG at {format_price(candidate['execution_price'], candidate['mid_price'])} (Score: {candidate['score_difference']:.2f})"
        )

    print("\nTop Short Candidates:")
    for candidate in short_candidates[:3]:
        print(
            f"{candidate['symbol']} - SHORT at {format_price(candidate['execution_price'], candidate['mid_price'])} (Score: {candidate['score_difference']:.2f})"
        )

if __name__ == "__main__":
    symbols = [
        "btc-usd",
        "eth-usd",
        "xrp-usd",
        "ltc-usd",
        "ada-usd",
        "doge-usd",
        "sol-usd",
        "wif-usd",
        "ondo-usd",
        "sei-usd",
        "magic-usd",
        "ape-usd",
        "jasmy-usd",
    ]
    while True:
        print("\nScanning for best opportunities...")
        scan_cryptocurrencies(symbols)
        time.sleep(refresh_interval)
