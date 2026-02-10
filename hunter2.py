import requests
import time

# Configuration parameters
whale_threshold = 1.0  # Minimum order quantity for whale orders
price_range_percentage = 10  # Restrict orders to ±10% of the mid-price
aggregation_threshold = 0.001  # Threshold for grouping close prices (0.1% of price)
proximity_weight = 0.6  # Weight of proximity in scoring
quantity_weight = 0.4  # Weight of order quantity in scoring
confidence_threshold = 0.1  # Minimum score difference to make a decision
price_change_threshold = 0.01  # Minimum percentage change to trigger output (1%)
refresh_interval = 60  # Time in seconds between data refreshes

# Helper function: Group close prices
def group_close_prices(orders, threshold):
    """Aggregate orders with prices within a threshold of each other."""
    if not orders:
        return []

    orders.sort(key=lambda x: x[0])  # Sort by price
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

# Formatting function for price
def format_price(price, mid_price):
    return f"{price:.4f}" if mid_price < 10 else f"{price:.2f}"

# Fetch top trading pairs by 24-hour volume
def fetch_top_pairs():
    url = "https://api.exchange.coinbase.com/products"
    response = requests.get(url)
    if response.status_code == 200:
        products = response.json()
        usd_pairs = [
            product['id'] for product in products
            if product['quote_currency'] == 'USD'
        ]

        # Fetch 24-hour stats for each pair
        stats = []
        for pair in usd_pairs:
            stats_url = f"https://api.exchange.coinbase.com/products/{pair}/stats"
            stats_response = requests.get(stats_url)
            if stats_response.status_code == 200:
                stats_data = stats_response.json()
                volume = float(stats_data['volume'])
                stats.append((pair, volume))
            time.sleep(1)  # Avoid rate-limiting

        # Sort by volume and select top 50 pairs
        stats.sort(key=lambda x: x[1], reverse=True)
        top_pairs = [pair for pair, _ in stats[:50]]
        return top_pairs
    else:
        print("Failed to fetch trading pairs.")
        return []

# Analyze the order book of a specific pair
def analyze_order_book(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=1"
    response = requests.get(url)
    if response.status_code == 200:
        order_book = response.json()

        best_bid = float(order_book['bids'][0][0])
        best_ask = float(order_book['asks'][0][0])
        mid_price = (best_bid + best_ask) / 2

        lower_bound = mid_price * (1 - price_range_percentage / 100)
        upper_bound = mid_price * (1 + price_range_percentage / 100)

        # Fetch full order book
        url_full = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
        response_full = requests.get(url_full)
        if response_full.status_code == 200:
            full_order_book = response_full.json()

            # Filter and aggregate whale bids and asks
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

            # Sort orders by proximity to mid-price
            sorted_bids = sorted(grouped_bids, key=lambda x: abs(mid_price - x[0]))
            sorted_asks = sorted(grouped_asks, key=lambda x: abs(mid_price - x[0]))

            # Decision logic
            def make_decision(sorted_bids, sorted_asks, mid_price):
                bid_score_total = sum(
                    proximity_weight * (1 / abs(mid_price - bid[0])) + quantity_weight * bid[1]
                    for bid in sorted_bids
                )
                ask_score_total = sum(
                    proximity_weight * (1 / abs(mid_price - ask[0])) + quantity_weight * ask[1]
                    for ask in sorted_asks
                )

                score_difference = bid_score_total - ask_score_total
                if abs(score_difference) >= confidence_threshold:
                    if score_difference > 0 and sorted_bids:
                        return "SHORT", sorted_bids[:3]
                    elif sorted_asks:
                        return "LONG", sorted_asks[:3]
                return "NO POSITION", []

            decision, top_candidates = make_decision(sorted_bids, sorted_asks, mid_price)
            formatted_candidates = [
                (format_price(candidate[0], mid_price), candidate[1]) for candidate in top_candidates
            ]
            return decision, formatted_candidates, mid_price

        else:
            print(f"Failed to fetch full order book for {symbol}: {response_full.status_code}")
            return "NO POSITION", [], None
    else:
        print(f"Failed to fetch mid-price for {symbol}: {response.status_code}")
        return "NO POSITION", [], None

# Analyze top trading pairs
def analyze_top_coins():
    top_pairs = fetch_top_pairs()

    while True:
        for symbol in top_pairs:
            decision, candidates, mid_price = analyze_order_book(symbol)
            if mid_price:
                print(f"\nSymbol: {symbol}")
                print(f"Mid-Price: {format_price(mid_price, mid_price)}")
                print(f"Decision: {decision}")
                if candidates:
                    print("Top Candidates:")
                    for price, qty in candidates:
                        print(f"  Price: {price}, Quantity: {qty:.2f}")
                else:
                    print("No significant candidates.")
            time.sleep(1)  # Short delay between symbols
        time.sleep(refresh_interval - len(top_pairs))  # Refresh interval

# Run the analysis
analyze_top_coins()
