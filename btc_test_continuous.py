import asyncio
import requests
from telegram import Bot

# Telegram Bot Configurations
TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = Bot(token=TOKEN)

# Configuration Parameters
symbol = "btc-usd"  # Focus on Bitcoin
whale_threshold = 10.0  # Minimum size for whale orders
price_range_percentage = 5  # 0.25% range around mid-price

def fetch_order_book():
    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch order book for {symbol}: {response.status_code}")

def get_top_whale_orders(order_book, mid_price):
    def filter_and_sort_orders(orders):
        filtered_orders = [
            (float(order[0]), float(order[1]))
            for order in orders
            if abs(float(order[0]) - mid_price) / mid_price <= price_range_percentage / 100 and float(order[1]) >= whale_threshold
        ]
        return sorted(filtered_orders, key=lambda x: abs(x[0] - mid_price))[:4]

    mid_price = (float(order_book['bids'][0][0]) + float(order_book['asks'][0][0])) / 2
    top_bids = filter_and_sort_orders(order_book['bids'])
    top_asks = filter_and_sort_orders(order_book['asks'])

    return top_bids, top_asks

def determine_next_fill_order(top_bids, top_asks, mid_price):
    if not top_bids or not top_asks:
        return None  # No valid orders to compare

    nearest_bid = min(top_bids, key=lambda x: abs(mid_price - x[0]))
    nearest_ask = min(top_asks, key=lambda x: abs(mid_price - x[0]))

    if abs(mid_price - nearest_bid[0]) < abs(mid_price - nearest_ask[0]):
        return f"Next likely order to fill: BID at {nearest_bid[0]:.2f} for {nearest_bid[1]:.2f} BTC"
    else:
        return f"Next likely order to fill: ASK at {nearest_ask[0]:.2f} for {nearest_ask[1]:.2f} BTC"

async def monitor_whale_orders():
    while True:
        try:
            order_book = fetch_order_book()
            mid_price = (float(order_book['bids'][0][0]) + float(order_book['asks'][0][0])) / 2

            top_bids, top_asks = get_top_whale_orders(order_book, mid_price)
            next_fill_order = determine_next_fill_order(top_bids, top_asks, mid_price)

            if next_fill_order:
                await bot.send_message(chat_id=CHAT_ID, text=next_fill_order)
        except Exception as e:
            await bot.send_message(chat_id=CHAT_ID, text=f"Error: {str(e)}")

        await asyncio.sleep(60)  # Wait 60 seconds before checking again

if __name__ == "__main__":
    asyncio.run(monitor_whale_orders())
