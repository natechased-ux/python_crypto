import asyncio
import requests
from telegram.ext import Application

# Telegram Bot Configurations
TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"  # Replace with your bot's token
CHAT_ID = "7967738614"  # Replace with your Telegram chat ID
app = Application.builder().token(TOKEN).build()

# Configuration parameters
symbols = [
    "btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
    "doge-usd", "sol-usd", "wif-usd", "ondo-usd", "sei-usd",
    "magic-usd", "ape-usd", "jasmy-usd", "wld-usd"
]
whale_threshold = 1.0
price_range_percentage = 10
aggregation_threshold = 0.001
proximity_weight = 0.6
quantity_weight = 0.4

# Helper Function: Group close prices
def group_close_prices(orders, threshold):
    if not orders:
        return []
    orders.sort(key=lambda x: x[0])
    grouped_orders = []
    current_group = [orders[0]]
    for i in range(1, len(orders)):
        current_price, current_qty = orders[i]
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

# Analyze Order Book for a Symbol
def analyze_order_book(symbol):
    try:
        # Fetch mid-price
        url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=1"
        response = requests.get(url)
        if response.status_code != 200:
            return f"{symbol.upper()}: Failed to fetch mid-price"
        order_book = response.json()
        best_bid = float(order_book['bids'][0][0])
        best_ask = float(order_book['asks'][0][0])
        mid_price = (best_bid + best_ask) / 2
        lower_bound = mid_price * (1 - price_range_percentage / 100)
        upper_bound = mid_price * (1 + price_range_percentage / 100)

        # Fetch full order book
        url_full = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
        response_full = requests.get(url_full)
        if response_full.status_code != 200:
            return f"{symbol.upper()}: Failed to fetch full order book"
        full_order_book = response_full.json()
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

        # Get nearest whale bid and ask
        nearest_bid = grouped_bids[0] if grouped_bids else None
        nearest_ask = grouped_asks[0] if grouped_asks else None

        # Calculate scores
        bid_score = (
            proximity_weight * (1 / abs(mid_price - nearest_bid[0])) + quantity_weight * nearest_bid[1]
            if nearest_bid else 0
        )
        ask_score = (
            proximity_weight * (1 / abs(mid_price - nearest_ask[0])) + quantity_weight * nearest_ask[1]
            if nearest_ask else 0
        )

        # Determine which is likely to be hit first
        if bid_score > ask_score:
            likely_hit = "BID likely to be hit first"
        elif ask_score > bid_score:
            likely_hit = "ASK likely to be hit first"
        else:
            likely_hit = "Neither BID nor ASK is dominant"

        # Format message
        bid_info = f"BID at {nearest_bid[0]:.4f} ({nearest_bid[1]:.2f})" if nearest_bid else "No BIDs"
        ask_info = f"ASK at {nearest_ask[0]:.4f} ({nearest_ask[1]:.2f})" if nearest_ask else "No ASKs"
        return f"{symbol.upper()}:\n  {bid_info}\n  {ask_info}\n  {likely_hit} (Mid: {mid_price:.4f})"
    except Exception as e:
        return f"{symbol.upper()}: Error - {str(e)}"

# Fetch and Analyze All Symbols
async def fetch_all_results():
    messages = [analyze_order_book(symbol) for symbol in symbols]
    return "\n".join(messages)

# Send Results to Telegram
async def send_to_telegram(message):
    await app.bot.send_message(chat_id=CHAT_ID, text=message)

# Hourly Updates
async def hourly_updates():
    while True:
        try:
            message = await fetch_all_results()
            await send_to_telegram(message)
        except Exception as e:
            await send_to_telegram(f"Error: {str(e)}")
        await asyncio.sleep(3600)

# Main Entry Point
if __name__ == '__main__':
    asyncio.run(hourly_updates())
