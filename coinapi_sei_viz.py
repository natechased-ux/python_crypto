import requests

# Define the product symbol and thresholds
symbol = "sei-USD"
price_range_percentage = 10  # Restrict orders to within ±10% of the mid-price

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

        # Sort by volume and find the top 3 orders
        top_3_bids = sorted(bids, key=lambda x: -x[1])[:3]
        top_3_asks = sorted(asks, key=lambda x: -x[1])[:3]

        # Determine the most likely next execution
        likely_next_bid = max(top_3_bids, key=lambda x: x[1]) if top_3_bids else None
        likely_next_ask = max(top_3_asks, key=lambda x: x[1]) if top_3_asks else None

        # Display results
        print(f"Mid-Price: {mid_price:.2f}")
        print("\nTop 3 Whale Bids (within 10% of mid-price):")
        for price, qty in top_3_bids:
            print(f" - Price: {price:.2f}, Quantity: {qty:.2f}")

        print("\nTop 3 Whale Asks (within 10% of mid-price):")
        for price, qty in top_3_asks:
            print(f" - Price: {price:.2f}, Quantity: {qty:.2f}")

        print("\nMost Likely Next Execution:")
        if likely_next_bid and likely_next_ask:
            if likely_next_bid[1] >= likely_next_ask[1]:
                print(f"Likely Next: Bid @ {likely_next_bid[0]:.2f} (Quantity: {likely_next_bid[1]:.2f})")
            else:
                print(f"Likely Next: Ask @ {likely_next_ask[0]:.2f} (Quantity: {likely_next_ask[1]:.2f})")
        elif likely_next_bid:
            print(f"Likely Next: Bid @ {likely_next_bid[0]:.2f} (Quantity: {likely_next_bid[1]:.2f})")
        elif likely_next_ask:
            print(f"Likely Next: Ask @ {likely_next_ask[0]:.2f} (Quantity: {likely_next_ask[1]:.2f})")
        else:
            print("No likely execution found.")
    else:
        print(f"Failed to fetch full order book: {response_full.status_code}")
else:
    print(f"Failed to fetch mid-price: {response.status_code}")
