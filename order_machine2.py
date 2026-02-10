import requests

# Configuration Parameters
price_range_percentage = 10  # ±10% range around the mid-price
top_n_whale_orders = 10  # Number of top whale orders to display

def fetch_order_book(symbol):
    """Fetch the order book and mid-price for the given symbol."""
    try:
        # Fetch mid-price
        url_level1 = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=1"
        response_level1 = requests.get(url_level1)
        response_level1.raise_for_status()
        order_book_level1 = response_level1.json()

        best_bid = float(order_book_level1['bids'][0][0])
        best_ask = float(order_book_level1['asks'][0][0])
        mid_price = (best_bid + best_ask) / 2

        # Fetch full order book
        url_level2 = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
        response_level2 = requests.get(url_level2)
        response_level2.raise_for_status()
        order_book_level2 = response_level2.json()

        return order_book_level2, mid_price
    except requests.exceptions.RequestException as e:
        print(f"Error fetching order book for {symbol}: {e}")
        return None, None

def find_top_whale_orders(order_book, mid_price):
    """Identify the top whale orders within ±10% of the mid-price."""
    if not order_book or not mid_price:
        return []

    lower_bound = mid_price * (1 - price_range_percentage / 100)
    upper_bound = mid_price * (1 + price_range_percentage / 100)

    # Extract and sort bids and asks
    whale_orders = []
    for order_type, orders in [('Bid', order_book['bids']), ('Ask', order_book['asks'])]:
        for price, qty, *_ in orders:
            price, qty = float(price), float(qty)
            if lower_bound <= price <= upper_bound:
                whale_orders.append({'Type': order_type, 'Price': price, 'Quantity': qty, 'Value': price * qty})

    # Sort by order size and limit to top N orders
    whale_orders.sort(key=lambda x: x['Value'], reverse=True)
    return whale_orders[:top_n_whale_orders]

def display_whale_orders(whale_orders, mid_price):
    """Display the top whale orders in a user-friendly format."""
    if not whale_orders:
        print("No whale orders found within the specified range.")
        return

    print(f"\nMid-Price: {mid_price:.4f}")
    print(f"{'Type':<6} {'Price':<10} {'Quantity':<10} {'Value':<12}")
    print("-" * 40)
    for order in whale_orders:
        price = f"{order['Price']:.4f}" if order['Price'] < 10 else f"{order['Price']:.2f}"
        print(f"{order['Type']:<6} {price:<10} {order['Quantity']:<10.4f} {order['Value']:<12.2f}")

if __name__ == "__main__":
    # Input coin symbol
    symbol = input("Enter the cryptocurrency symbol (e.g., btc-usd, eth-usd): ").strip().lower()
    order_book, mid_price = fetch_order_book(symbol)
    whale_orders = find_top_whale_orders(order_book, mid_price)
    display_whale_orders(whale_orders, mid_price)
