import requests

# Configuration parameters
symbol = "mana-usd"
whale_threshold = 1  # Minimum order quantity for whale orders
price_range_percentage = 10  # Restrict orders to ±10% of the mid-price
aggregation_threshold = 0.001  # Threshold for grouping close prices (0.1% of price)
proximity_weight = 0.6  # Weight of proximity in scoring
quantity_weight = 0.4  # Weight of order quantity in scoring
confidence_threshold = 0.1  # Minimum score difference to make a decision

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

# Fetch order book data and process
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
                if score_difference > 0:
                    return "SHORT", sorted_bids[0][0]
                else:
                    return "LONG", sorted_asks[0][0]
            else:
                return "NO POSITION", None

        # First decision
        first_decision, first_execution_price = make_decision(sorted_bids, sorted_asks, mid_price)
        print("\nFirst Decision:")
        print(f"{first_decision} to {first_execution_price:.2f}" if first_execution_price else "NO POSITION")

        # Remove executed order
        if first_decision == "SHORT" and first_execution_price:
            sorted_bids.pop(0)
        elif first_decision == "LONG" and first_execution_price:
            sorted_asks.pop(0)

        # Next two decisions
        decisions = [(first_decision, first_execution_price)]
        for _ in range(2):
            if decisions[-1][0] == "LONG" and sorted_bids:
                next_decision, next_price = "SHORT", sorted_bids.pop(0)[0]
            elif decisions[-1][0] == "SHORT" and sorted_asks:
                next_decision, next_price = "LONG", sorted_asks.pop(0)[0]
            else:
                next_decision, next_price = "NO POSITION", None
            decisions.append((next_decision, next_price))

        print("\nNext Decisions:")
        for decision, price in decisions[1:]:
            print(f"{decision} to {price:.2f}" if price else "NO POSITION")

    else:
        print(f"Failed to fetch full order book: {response_full.status_code}")
else:
    print(f"Failed to fetch mid-price: {response.status_code}")
