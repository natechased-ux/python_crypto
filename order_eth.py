import requests

# Define the product symbol and thresholds
symbol = "eth-USD"
price_range_percentage = 10  # Restrict orders to within ±10% of the mid-price
top_whale_count = 7  # Number of top whale orders to consider
vwap_threshold = 0.5  # Threshold percentage for VWAP proximity

# API endpoint for Coinbase Pro order book
url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=1"

# Fetch the best bid and ask to calculate mid-price
response = requests.get(url)
if response.status_code == 200:
    order_book = response.json()

    # Calculate mid-price
    best_bid = float(order_book['bids'][0][0])
    best_ask = float(order_book['asks'][0][0])
    mid_price = (best_bid + best_ask) / 2

    # Define price range
    lower_bound = mid_price * (1 - price_range_percentage / 100)
    upper_bound = mid_price * (1 + price_range_percentage / 100)

    # Fetch full order book (Level 2) for filtering
    url_full = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
    response_full = requests.get(url_full)
    if response_full.status_code == 200:
        full_order_book = response_full.json()

        # Filter bids and asks within the price range
        bids = [
            (float(bid[0]), float(bid[1]))
            for bid in full_order_book['bids']
            if lower_bound <= float(bid[0]) <= upper_bound
        ]
        asks = [
            (float(ask[0]), float(ask[1]))
            for ask in full_order_book['asks']
            if lower_bound <= float(ask[0]) <= upper_bound
        ]

        # Sort by price in ascending order
        sorted_bids = sorted(bids, key=lambda x: x[0])[:top_whale_count]
        sorted_asks = sorted(asks, key=lambda x: x[0])[:top_whale_count]

        # Calculate VWAP
        total_volume = sum(order[1] for order in sorted_bids + sorted_asks)
        if total_volume > 0:
            vwap = sum(order[0] * order[1] for order in sorted_bids + sorted_asks) / total_volume
        else:
            vwap = mid_price  # Default to mid-price if no orders

        # Decision Analysis
        closest_bid = min(sorted_bids, key=lambda x: abs(mid_price - x[0]), default=None)
        closest_ask = min(sorted_asks, key=lambda x: abs(mid_price - x[0]), default=None)

        # Determine position signals
        position_signal = None
        if closest_bid and closest_ask:
            if abs(mid_price - closest_bid[0]) < abs(mid_price - closest_ask[0]):
                if mid_price < vwap * (1 - vwap_threshold / 100):
                    position_signal = "LONG"
            elif abs(mid_price - closest_ask[0]) < abs(mid_price - closest_bid[0]):
                if mid_price > vwap * (1 + vwap_threshold / 100):
                    position_signal = "SHORT"

        # Display results
        print(f"Mid-Price: {mid_price:.2f}")
        print(f"VWAP: {vwap:.2f}")
        print("\nTop Whale Bids (Sorted by Price):")
        for price, qty in sorted_bids:
            print(f" - Price: {price:.2f}, Quantity: {qty:.2f}")

        print("\nTop Whale Asks (Sorted by Price):")
        for price, qty in sorted_asks:
            print(f" - Price: {price:.2f}, Quantity: {qty:.2f}")

        print("\nDecision Signals:")
        if position_signal:
            print(f"Recommended Position: {position_signal}")
            if position_signal == "LONG":
                print(f"Take profit area: Closest Ask @ {closest_ask[0]:.2f}")
            elif position_signal == "SHORT":
                print(f"Take profit area: Closest Bid @ {closest_bid[0]:.2f}")
        else:
            print("No clear position signal at this time.")
    else:
        print(f"Failed to fetch full order book: {response_full.status_code}")
else:
    print(f"Failed to fetch mid-price: {response.status_code}")
