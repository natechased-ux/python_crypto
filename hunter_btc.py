import requests
import numpy as np
import time
import pandas as pd

# Telegram configuration
TELEGRAM_BOT_TOKEN = '8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw'
TELEGRAM_CHAT_ID = '7967738614'

# Configuration parameters
symbols = ["btc-usd", "eth-usd", "xrp-usd"]
granularity = 300  # Candle interval in seconds
lookback_periods = 100  # Number of candles to fetch
price_distance_threshold = 0.01
confidence_threshold = 0.1
short_window = 5
long_window = 20

# Helper functions
def fetch_candles(symbol, granularity=300, periods=100):
    """Fetch recent candlestick data and return closing prices."""
    end = int(time.time())
    start = end - granularity * periods
    url = (
        f"https://api.exchange.coinbase.com/products/{symbol}/candles"
        f"?granularity={granularity}&start={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start))}"
        f"&end={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(end))}"
    )
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        closes = [candle[4] for candle in sorted(data, key=lambda x: x[0])]
        return closes
    else:
        print(f"Failed to fetch data for {symbol}: {resp.status_code}\n{resp.json()}")
        return []

def compute_rsi(prices, length=14):
    """Calculate RSI from list of closing prices."""
    deltas = np.diff(prices)
    gains = deltas.clip(min=0)
    losses = -deltas.clip(max=0)
    avg_gain = np.mean(gains[-length:])
    avg_loss = np.mean(losses[-length:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_trend(prices, short_window, long_window):
    """Compute trend based on moving averages."""
    short_ma = pd.Series(prices).rolling(window=short_window).mean().iloc[-1]
    long_ma = pd.Series(prices).rolling(window=long_window).mean().iloc[-1]
    return short_ma, long_ma

def group_close_prices(price_levels, threshold=0.001):
    """
    Group price levels that are within a certain threshold and sum their volumes.
    """
    if not price_levels:
        return []

    # Sort the price levels by price
    price_levels.sort(key=lambda x: x[0])
    grouped_levels = []
    current_price, current_volume = price_levels[0]

    for price, volume in price_levels[1:]:
        if abs(price - current_price) <= threshold:
            # Group the levels
            current_volume += volume
        else:
            # Add the current group to the list
            grouped_levels.append((current_price, current_volume))
            current_price, current_volume = price, volume

    # Append the last group
    grouped_levels.append((current_price, current_volume))
    return grouped_levels

def analyze_order_book(symbol):
    """Analyze order book and compute signals."""
    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch order book for {symbol}")
        return None

    order_book = response.json()
    bids = [(float(bid[0]), float(bid[1])) for bid in order_book['bids']]
    asks = [(float(ask[0]), float(ask[1])) for ask in order_book['asks']]
    mid_price = (bids[0][0] + asks[0][0]) / 2

    grouped_bids = group_close_prices(bids)
    grouped_asks = group_close_prices(asks)

    # Compute scores
    bid_score = sum(vol / abs(mid_price - price) for price, vol in grouped_bids if price < mid_price)
    ask_score = sum(vol / abs(mid_price - price) for price, vol in grouped_asks if price > mid_price)

    # Decision logic
    score_diff = bid_score - ask_score
    decision = "NO POSITION"

    closes = fetch_candles(symbol)
    if len(closes) < 15:
        return None
    rsi = compute_rsi(closes)
    short_ma, long_ma = compute_trend(closes, short_window, long_window)

    if abs(score_diff) >= confidence_threshold:
        if score_diff > 0 and rsi > 70 and short_ma > long_ma:
            decision = "LONG"
        elif score_diff < 0 and rsi < 30 and short_ma < long_ma:
            decision = "SHORT"

    return {"symbol": symbol, "decision": decision, "rsi": rsi, "mid_price": mid_price}

def backtest():
    """Backtest strategy with historical data."""
    results = []
    for symbol in symbols:
        closes = fetch_candles(symbol, granularity, lookback_periods)
        if len(closes) < 2:
            continue

        trades = []
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                trades.append(1)  # Profit
            elif closes[i] < closes[i - 1]:
                trades.append(-1)  # Loss
            else:
                trades.append(0)  # No change

        total_profit = sum(trades) / len(trades) if trades else 0
        win_rate = sum(1 for t in trades if t > 0) / len(trades) if trades else 0

        results.append({"symbol": symbol, "profit": total_profit, "win_rate": win_rate})

    for result in results:
        print(f"{result['symbol']}: Total Profit: {result['profit']:.2f}, Win Rate: {result['win_rate']:.2f}")

backtest()
