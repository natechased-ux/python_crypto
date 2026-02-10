import asyncio
import requests
from collections import defaultdict

# Telegram Bot Configurations
TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"

# Configuration Parameters
symbols = ["btc-usd", "eth-usd", "xrp-usd"]
price_range_percentage = 2  # 2% range around mid-price
aggregation_range = 0.01  # Aggregation range as a percentage of mid-price

def fetch_order_book(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch order book for {symbol}: {response.status_code}")

def get_dynamic_whale_threshold(order_book):
    """
    Calculates a dynamic whale threshold based on the average order size in the order book.
    """
    all_sizes = [float(bid[1]) for bid in order_book['bids']] + [float(ask[1]) for ask in order_book['asks']]
    average_size = sum(all_sizes) / len(all_sizes) if all_sizes else 1
    whale_threshold = average_size * 20  # Define a whale as 100x the average order size
    return whale_threshold

def aggregate_orders(orders, aggregation_range):
    """
    Aggregates orders that are close in price within the specified aggregation range.
    """
    aggregated = defaultdict(float)
    for price, size in orders:
        aggregated_price = round(price / aggregation_range) * aggregation_range
        aggregated[aggregated_price] += size
    return sorted(aggregated.items())

def get_top_whale_orders(order_book, mid_price, price_range_percentage, whale_threshold):
    def filter_and_sort_orders(orders):
        filtered_orders = [
            (float(order[0]), float(order[1]))
            for order in orders
            if abs(float(order[0]) - mid_price) / mid_price <= price_range_percentage / 100 and float(order[1]) >= whale_threshold
        ]
        aggregated_orders = aggregate_orders(filtered_orders, mid_price * aggregation_range / 100)
        return sorted(aggregated_orders, key=lambda x: abs(x[0] - mid_price))[:1]

    top_bids = filter_and_sort_orders(order_book['bids'])
    top_asks = filter_and_sort_orders(order_book['asks'])

    return top_bids, top_asks

def determine_next_fill_order(top_bids, top_asks, mid_price):
    if not top_bids or not top_asks:
        return "Not enough data to determine next order fill."

    nearest_bid = min(top_bids, key=lambda x: abs(mid_price - x[0]))
    nearest_ask = min(top_asks, key=lambda x: abs(mid_price - x[0]))

    if abs(mid_price - nearest_bid[0]) < abs(mid_price - nearest_ask[0]):
        return f"Next likely order to fill: BID at {nearest_bid[0]:.8f} for {nearest_bid[1]:.8f}"
    else:
        return f"Next likely order to fill: ASK at {nearest_ask[0]:.8f} for {nearest_ask[1]:.8f}"

async def monitor_coin(symbol):
    try:
        order_book = fetch_order_book(symbol)
        mid_price = (float(order_book['bids'][0][0]) + float(order_book['asks'][0][0])) / 2

        whale_threshold = get_dynamic_whale_threshold(order_book)

        top_bids, top_asks = get_top_whale_orders(
            order_book, mid_price, price_range_percentage, whale_threshold
        )

        print(f"\nTop 4 Whale Bids for {symbol}:")
        for bid in top_bids:
            print(f"Price: {bid[0]:.8f}, Size: {bid[1]:.8f}")

        print(f"\nTop 4 Whale Asks for {symbol}:")
        for ask in top_asks:
            print(f"Price: {ask[0]:.8f}, Size: {ask[1]:.8f}")

        next_fill_order = determine_next_fill_order(top_bids, top_asks, mid_price)
        print(f"\n{next_fill_order}")

    except Exception as e:
        print(f"Error monitoring {symbol}: {str(e)}")

async def main():
    tasks = [monitor_coin(symbol) for symbol in symbols]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
