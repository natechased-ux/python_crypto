import requests
import pandas as pd

# Configuration
price_range_percentage = 10
usd_threshold = 10000  # Minimum USD value for whale orders
vwap_length = 14

# Function to calculate dynamic aggregation threshold
def get_dynamic_aggregation_threshold(symbol):
    """Determine the aggregation threshold based on the coin's characteristics."""
    try:
        # Fetch recent historical data for price range analysis
        url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity=3600"
        response = requests.get(url)
        if response.status_code == 200:
            data = pd.DataFrame(response.json(), columns=["time", "low", "high", "open", "close", "volume"])
            avg_range = (data["high"] - data["low"]).mean()
            dynamic_threshold = avg_range / data["close"].mean() * 0.1  # Scale factor for threshold
            return max(dynamic_threshold, 0.0001)  # Ensure a minimum threshold
        else:
            raise Exception(f"Failed to fetch data for {symbol}")
    except Exception as e:
        print(f"Error calculating dynamic threshold for {symbol}: {e}")
        return 0.001  # Default fallback threshold

# Helper Function: Group close prices
def group_close_prices(orders, threshold):
    if not orders:
        return []
    orders.sort(key=lambda x: x[0])
    grouped, current = [], [orders[0]]
    for price, qty in orders[1:]:
        prev_price = current[-1][0]
        if abs(price - prev_price) / prev_price <= threshold:
            current.append((price, qty))
        else:
            total_qty = sum(q for _, q in current)
            avg_price = sum(p * q for p, q in current) / total_qty
            grouped.append((avg_price, total_qty))
            current = [(price, qty)]
    total_qty = sum(q for _, q in current)
    avg_price = sum(p * q for p, q in current) / total_qty
    grouped.append((avg_price, total_qty))
    return grouped

# Analyze Whale Orders
def analyze_whale_orders(symbol):
    try:
        # Fetch mid-price
        url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=1"
        response = requests.get(url)
        response.raise_for_status()
        order_book = response.json()
        best_bid = float(order_book['bids'][0][0])
        best_ask = float(order_book['asks'][0][0])
        mid_price = (best_bid + best_ask) / 2

        # Define price range
        lower_bound = mid_price * (1 - price_range_percentage / 100)
        upper_bound = mid_price * (1 + price_range_percentage / 100)

        # Fetch full order book
        url_full = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
        response_full = requests.get(url_full)
        response_full.raise_for_status()
        full_order_book = response_full.json()

        # Calculate dynamic aggregation threshold
        aggregation_threshold = get_dynamic_aggregation_threshold(symbol)

        # Filter and group whale orders
        bids = [(float(bid[0]), float(bid[1])) for bid in full_order_book['bids']
                if lower_bound <= float(bid[0]) <= upper_bound and float(bid[0]) * float(bid[1]) >= usd_threshold]
        asks = [(float(ask[0]), float(ask[1])) for ask in full_order_book['asks']
                if lower_bound <= float(ask[0]) <= upper_bound and float(ask[0]) * float(ask[1]) >= usd_threshold]

        grouped_bids = group_close_prices(bids, aggregation_threshold)
        grouped_asks = group_close_prices(asks, aggregation_threshold)

        return {
            "symbol": symbol.upper(),
            "mid_price": mid_price,
            "whale_bids": grouped_bids,
            "whale_asks": grouped_asks
        }
    except Exception as e:
        return {"error": str(e)}

# Main Function
def main():
    symbol = input("Enter the cryptocurrency symbol (e.g., btc-usd, eth-usd): ").strip().lower()
    result = analyze_whale_orders(symbol)

    if "error" in result:
        print(f"Error fetching data for {symbol.upper()}: {result['error']}")
    else:
        print(f"\nSymbol: {result['symbol']}")
        print(f"Mid Price: {result['mid_price']:.2f}")
        print("\nWhale Bids (Grouped):")
        for price, qty in result["whale_bids"]:
            print(f"  Price: {price:.4f}, Quantity: {qty:.2f}, USD Value: {price * qty:.2f}")
        print("\nWhale Asks (Grouped):")
        for price, qty in result["whale_asks"]:
            print(f"  Price: {price:.4f}, Quantity: {qty:.2f}, USD Value: {price * qty:.2f}")

if __name__ == "__main__":
    main()
