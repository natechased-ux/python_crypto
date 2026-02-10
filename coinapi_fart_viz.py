import requests
import matplotlib.pyplot as plt

# Define the product symbol and thresholds
symbol = "fartcoin-USD"
whale_threshold = 10000.0  # Minimum order quantity to qualify as a whale
price_range_percentage = 20  # Restrict orders to within ±25% of the mid-price

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

        # Filter whale bids and asks within the price range
        whale_bids = [
            (float(bid[0]), float(bid[1]))
            for bid in full_order_book['bids']
            if lower_bound <= float(bid[0]) <= upper_bound and float(bid[1]) >= whale_threshold
        ]
        whale_asks = [
            (float(ask[0]), float(ask[1]))
            for ask in full_order_book['asks']
            if lower_bound <= float(ask[0]) <= upper_bound and float(ask[1]) >= whale_threshold
        ]

        # Visualization
        plt.figure(figsize=(12, 8))

        # Offsets for labels
        label_offset_y = 0.5  # Offset for y-axis labels to prevent overlap
        label_offset_x = 0.002  # Offset for x-axis labels to avoid crowding

        # Bids
        if whale_bids:
            bid_prices, bid_quantities = zip(*whale_bids)
            plt.scatter(bid_prices, bid_quantities, color='green', label='Whale Bids', alpha=0.6)
            for price, qty in whale_bids:
                plt.text(
                    price + label_offset_x, qty + label_offset_y,
                    f"{price:.2f}", fontsize=8, color='green', ha='center', va='bottom'
                )

        # Asks
        if whale_asks:
            ask_prices, ask_quantities = zip(*whale_asks)
            plt.scatter(ask_prices, [-qty for qty in ask_quantities], color='red', label='Whale Asks', alpha=0.6)
            for price, qty in whale_asks:
                plt.text(
                    price + label_offset_x, -qty - label_offset_y,
                    f"{price:.2f}", fontsize=8, color='red', ha='center', va='top'
                )

        # Add mid-price line
        plt.axvline(x=mid_price, color='blue', linestyle='--', label=f'Mid Price: {mid_price:.2f}')
        plt.axhline(0, color='black', linewidth=0.8)

        # Labels and title
        plt.title(f"Whale Orders for {symbol}")
        plt.xlabel("Price")
        plt.ylabel("Order Size")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    else:
        print(f"Failed to fetch full order book: {response_full.status_code}")
else:
    print(f"Failed to fetch mid-price: {response.status_code}")
