import requests
import matplotlib.pyplot as plt
import numpy as np

# Fetch the order book from Coinbase
def fetch_order_book(product_id="BTC-USD", level=2):
    url = f"https://api.exchange.coinbase.com/products/{product_id}/book?level={level}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch order book: {response.text}")

# Process and visualize the order book
def visualize_order_book(order_book):
    # Extract bids and asks
    bids = order_book["bids"]  # [price, size, num_orders]
    asks = order_book["asks"]  # [price, size, num_orders]

    # Convert to numpy arrays for easier processing
    bid_prices = np.array([float(bid[0]) for bid in bids])
    bid_sizes = np.array([float(bid[1]) for bid in bids])
    ask_prices = np.array([float(ask[0]) for ask in asks])
    ask_sizes = np.array([float(ask[1]) for ask in asks])

    # Create the depth chart
    plt.figure(figsize=(12, 6))
    
    # Cumulative sum for depth visualization
    cumulative_bid_sizes = np.cumsum(bid_sizes[::-1])[::-1]  # Reverse for correct cumulative plot
    cumulative_ask_sizes = np.cumsum(ask_sizes)

    # Plot bids
    plt.plot(bid_prices, cumulative_bid_sizes, label="Bids (Buy Orders)", color="green")
    # Plot asks
    plt.plot(ask_prices, cumulative_ask_sizes, label="Asks (Sell Orders)", color="red")

    # Add labels and title
    plt.title("Order Book Depth Chart", fontsize=16)
    plt.xlabel("Price", fontsize=14)
    plt.ylabel("Cumulative Size", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
try:
    product_id = "BTC-USD"  # Change to desired product
    order_book = fetch_order_book(product_id)
    visualize_order_book(order_book)
except Exception as e:
    print(f"Error: {e}")
