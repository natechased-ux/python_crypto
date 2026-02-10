import asyncio
import requests
from telegram import Bot

# Telegram Bot Configurations
TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = Bot(token=TOKEN)

# Configuration parameters
symbols = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd"]
whale_threshold = 1.0
price_range_percentage = 10
aggregation_threshold = 0.001
proximity_weight = 0.6
quantity_weight = 0.4
confidence_threshold = 0.1

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

def analyze_order_book(symbol):
    try:
        r1 = requests.get(f"https://api.exchange.coinbase.com/products/{symbol}/book?level=1").json()
        best_bid, best_ask = float(r1['bids'][0][0]), float(r1['asks'][0][0])
        mid = (best_bid + best_ask) / 2
        lb, ub = mid * (1 - price_range_percentage / 100), mid * (1 + price_range_percentage / 100)
        r2 = requests.get(f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2").json()
        bids = [(float(p), float(q)) for p, q, *_ in r2['bids'] if lb <= float(p) <= ub and float(q) >= whale_threshold]
        asks = [(float(p), float(q)) for p, q, *_ in r2['asks'] if lb <= float(p) <= ub and float(q) >= whale_threshold]
        gb, ga = group_close_prices(bids, aggregation_threshold), group_close_prices(asks, aggregation_threshold)
        sb = sorted(gb, key=lambda x: abs(mid - x[0]))
        sa = sorted(ga, key=lambda x: abs(mid - x[0]))
        bid_score = sum(proximity_weight * (1 / abs(mid - p)) + quantity_weight * q for p, q in sb)
        ask_score = sum(proximity_weight * (1 / abs(mid - p)) + quantity_weight * q for p, q in sa)
        diff = bid_score - ask_score
        if abs(diff) >= confidence_threshold:
            if diff > 0 and sb:
                decision, price = 'SHORT', sb[0][0]
            elif diff < 0 and sa:
                decision, price = 'LONG', sa[0][0]
            else:
                decision, price = 'NO POSITION', None
        else:
            decision, price = 'NO POSITION', None
        pct = abs(price - mid) / mid * 100 if price else None
        return {'symbol': symbol, 'mid': mid, 'decision': decision, 'price': price, 'pct_diff': pct}
    except Exception:
        return {'symbol': symbol, 'mid': None, 'decision': 'ERROR', 'price': None, 'pct_diff': None}

async def send_to_telegram(message):
    """Send message to Telegram bot."""
    await bot.send_message(chat_id=CHAT_ID, text=message)

def get_order_book_results():
    """Fetch and format order book results."""
    results = [analyze_order_book(sym) for sym in symbols]
    results = [r for r in results if r['price'] is not None]
    results.sort(key=lambda x: x['pct_diff'], reverse=True)
    message_lines = []
    for r in results:
        s = r['symbol'].upper()
        d = r['decision']
        p = r['price']
        pct = r['pct_diff']
        price_str = f"{p:.4f}" if p < 10 else f"{p:.2f}"
        message_lines.append(f"{s}: {d} to {price_str} ({pct:.2f}% from mid)")
    return "\n".join(message_lines)

async def hourly_updates():
    """Continuously send updates to Telegram every hour."""
    while True:
        try:
            message = get_order_book_results()
            await send_to_telegram(message)
        except Exception as e:
            await send_to_telegram(f"Error: {str(e)}")
        await asyncio.sleep(3600)  # Wait for 1 hour

if __name__ == '__main__':
    asyncio.run(hourly_updates())
