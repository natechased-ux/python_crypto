import requests
import time
import numpy as np

# Configuration
symbols = ["btc-usd", "eth-usd", "xrp-usd"]
confidence_threshold = 0.2
volume_threshold = 10000  # Example value, customize as needed
short_ma_window = 10
long_ma_window = 50

# Helper functions remain unchanged (group_close_prices, fetch_candles, compute_rsi, send_telegram_message)

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
    if len(prices) < long_window:
        return "neutral"
    short_ma = np.mean(prices[-short_window:])
    long_ma = np.mean(prices[-long_window:])
    return "up" if short_ma > long_ma else "down"

def refined_decision_logic(bid_score, ask_score, rsi, trend):
    """Enhanced decision-making logic based on multiple factors."""
    if abs(bid_score - ask_score) < confidence_threshold:
        return "NO POSITION"

    if bid_score > ask_score:  # Short scenarios
        if rsi > 85 and trend == "down":
            return "SUPER SHORT"
        elif rsi > 70 and trend == "down":
            return "STRONG SHORT"
    elif bid_score < ask_score:  # Long scenarios
        if rsi < 15 and trend == "up":
            return "SUPER LONG"
        elif rsi < 30 and trend == "up":
            return "STRONG LONG"
    
    return "NO POSITION"

def analyze_order_book(symbol):
    """Analyze the order book and make a refined trading decision."""
    # Fetch mid-price and full order book
    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=1"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    
    order_book = response.json()
    best_bid = float(order_book['bids'][0][0])
    best_ask = float(order_book['asks'][0][0])
    mid_price = (best_bid + best_ask) / 2

    # Fetch candles for RSI and trend analysis
    closes = fetch_candles(symbol)
    if len(closes) < long_ma_window:
        return None
    rsi = compute_rsi(closes)
    trend = compute_trend(closes, short_ma_window, long_ma_window)

    # Decision-making logic
    lower_bound = mid_price * 0.99
    upper_bound = mid_price * 1.01

    grouped_bids = group_close_prices(
        [(float(bid[0]), float(bid[1])) for bid in order_book['bids'] if lower_bound <= float(bid[0]) <= upper_bound],
        threshold=0.001
    )
    grouped_asks = group_close_prices(
        [(float(ask[0]), float(ask[1])) for ask in order_book['asks'] if lower_bound <= float(ask[0]) <= upper_bound],
        threshold=0.001
    )

    bid_score_total = sum((0.6 / abs(mid_price - bid[0])) + 0.4 * bid[1] for bid in grouped_bids)
    ask_score_total = sum((0.6 / abs(mid_price - ask[0])) + 0.4 * ask[1] for ask in grouped_asks)
    
    decision = refined_decision_logic(bid_score_total, ask_score_total, rsi, trend)
    return {
        "symbol": symbol,
        "decision": decision,
        "rsi": rsi,
        "trend": trend,
        "targets": [f"{bid[0]:.2f}" for bid in grouped_bids[:3] if decision in ["SHORT", "STRONG SHORT"]] or
                   [f"{ask[0]:.2f}" for ask in grouped_asks[:3] if decision in ["LONG", "STRONG LONG"]],
    }

def analyze_top_coins():
    while True:
        results = []
        for symbol in symbols:
            analysis = analyze_order_book(symbol)
            if analysis:
                results.append(analysis)

                # Only send Telegram messages for "SUPER" signals
                if analysis["decision"] in ["SUPER LONG", "SUPER SHORT"]:
                    message = (
                        f"🔔 *Strong Signal: {analysis['decision']}*\n"
                        f"Symbol: {analysis['symbol']}\n"
                        f"RSI: {analysis['rsi']:.2f}\n"
                        f"Trend: {analysis['trend']}\n"
                        f"Targets: {', '.join(analysis['targets'])}"
                    )
                    send_telegram_message(message)
        
        if results:
            strongest_candidate = max(
                results, key=lambda x: abs(x["rsi"] - 50) * (1 if x["decision"] != "NO POSITION" else 0)
            )
            print(
                f"Strongest Candidate: {strongest_candidate['symbol']} with Decision: {strongest_candidate['decision']}, "
                f"RSI: {strongest_candidate['rsi']:.2f}, Trend: {strongest_candidate['trend']}"
            )
        time.sleep(30)

analyze_top_coins()
