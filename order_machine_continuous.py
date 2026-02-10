import requests
import numpy as np
import pandas as pd
from ta.volume import AccDistIndexIndicator
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator, EMAIndicator
from pathlib import Path

# Configuration parameters
atr_multiplier = 1.5
symbols = []
granularity = 300  # 5-minute candles
whale_threshold = 1.0  # Minimum order quantity for whale orders
price_range_percentage = 10  # Restrict orders to ±10% of the mid-price
aggregation_threshold = 0.001  # Threshold for grouping close prices (0.1% of price)
proximity_weight = 0.6  # Weight of proximity in scoring
quantity_weight = 0.4  # Weight of order quantity in scoring
confidence_threshold = 0.1  # Minimum score difference to make a decision

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

# Analyze order book and return result dict
def analyze_order_book(symbol):
    try:
        r1 = requests.get(f"https://api.exchange.coinbase.com/products/{symbol}/book?level=1").json()
        best_bid, best_ask = float(r1['bids'][0][0]), float(r1['asks'][0][0])
        mid = (best_bid + best_ask) / 2
        lb, ub = mid * (1 - price_range_percentage/100), mid * (1 + price_range_percentage/100)
        r2 = requests.get(f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2").json()
        bids = [(float(p), float(q)) for p, q, *_ in r2['bids'] if lb <= float(p) <= ub and float(q) >= whale_threshold]
        asks = [(float(p), float(q)) for p, q, *_ in r2['asks'] if lb <= float(p) <= ub and float(q) >= whale_threshold]
        gb, ga = group_close_prices(bids, aggregation_threshold), group_close_prices(asks, aggregation_threshold)
        sb = sorted(gb, key=lambda x: abs(mid - x[0]))
        sa = sorted(ga, key=lambda x: abs(mid - x[0]))
        bid_score = sum(proximity_weight*(1/abs(mid-p))+quantity_weight*q for p,q in sb)
        ask_score = sum(proximity_weight*(1/abs(mid-p))+quantity_weight*q for p,q in sa)
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
        pct = abs(price-mid)/mid*100 if price else None
        return {'symbol': symbol, 'mid': mid, 'decision': decision, 'price': price, 'pct_diff': pct}
    except Exception:
        return {'symbol': symbol, 'mid': None, 'decision': 'ERROR', 'price': None, 'pct_diff': None}

# Telegram configuration
from telegram import Bot

bot_token = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
chat_id = "7967738614"
bot = Bot(token=bot_token)

# Telegram configuration
from telegram import Bot

bot_token = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
chat_id = "7967738614"
bot = Bot(token=bot_token)

# Main loop: run every hour and send to Telegram
if __name__ == '__main__':
    import time
    while True:
        results = [analyze_order_book(sym) for sym in symbols]
        results = [r for r in results if r['price'] is not None]
        # Sort by distance from mid-price ascending for quickest hit
        results.sort(key=lambda x: x['pct_diff'])

        output_lines = []
        for r in results:
            s = r['symbol'].upper()
            d = r['decisio
