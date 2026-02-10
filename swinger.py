import requests
import time
import datetime
import numpy as np

# Telegram config (your values)
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"

# Coins to monitor on Coinbase
COINS = ["BTC-USD", "ETH-USD", "XRP-USD"]

# Constants for indicators
EMA_PERIOD = 50
ATR_PERIOD = 14
RSI_PERIOD = 14

# Coinbase API endpoint for historic rates (granularity in seconds)
COINBASE_API = "https://api.exchange.coinbase.com/products/{}/candles"

# Telegram send message
def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Telegram send failed: {response.text}")
    except Exception as e:
        print(f"Telegram exception: {e}")

# Fetch candles from Coinbase with updated URL and params
def fetch_candles(product_id: str, granularity: int):
    params = {"granularity": granularity}
    url = COINBASE_API.format(product_id)
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch candles for {product_id} {response.text}")
        return None
    data = response.json()
    # Coinbase returns [time, low, high, open, close, volume] descending order
    candles = [{
        "time": c[0],
        "low": c[1],
        "high": c[2],
        "open": c[3],
        "close": c[4],
        "volume": c[5],
    } for c in reversed(data)]
    return candles

# Calculate EMA
def calculate_ema(prices, period):
    ema = []
    k = 2 / (period + 1)
    for i, price in enumerate(prices):
        if i == 0:
            ema.append(price)
        else:
            ema.append(price * k + ema[-1] * (1 - k))
    return ema

# Calculate ATR
def calculate_atr(candles, period):
    trs = []
    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i-1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    atr = []
    for i in range(len(trs)):
        if i < period:
            atr.append(np.nan)  # insufficient data
        elif i == period:
            atr.append(np.mean(trs[:period]))
        else:
            atr.append((atr[-1] * (period - 1) + trs[i]) / period)
    # Align length with candles: prepend nan for first candle
    return [np.nan] + atr

# Calculate RSI
def calculate_rsi(prices, period):
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = [np.nan] * (period)
    rsi.append(100 - 100 / (1 + rs))

    for i in range(period + 1, len(prices)):
        delta = deltas[i - 1]
        upval = max(delta, 0)
        downval = max(-delta, 0)
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi.append(100 - 100 / (1 + rs))
    return rsi

# Check bullish engulfing candle
def is_bullish_engulfing(candles, idx):
    if idx == 0:
        return False
    c0 = candles[idx]
    c1 = candles[idx-1]
    return (c0["open"] < c0["close"] and
            c1["open"] > c1["close"] and
            c0["open"] < c1["close"] and
            c0["close"] > c1["open"])

# Check bearish engulfing candle
def is_bearish_engulfing(candles, idx):
    if idx == 0:
        return False
    c0 = candles[idx]
    c1 = candles[idx-1]
    return (c0["open"] > c0["close"] and
            c1["open"] < c1["close"] and
            c0["open"] > c1["close"] and
            c0["close"] < c1["open"])

# Check hammer or shooting star pin bars (simple version)
def is_hammer(candle):
    body = abs(candle["close"] - candle["open"])
    lower_shadow = candle["open"] - candle["low"] if candle["close"] > candle["open"] else candle["close"] - candle["low"]
    upper_shadow = candle["high"] - candle["close"] if candle["close"] > candle["open"] else candle["high"] - candle["open"]
    return lower_shadow > 2 * body and upper_shadow < body

def is_shooting_star(candle):
    body = abs(candle["close"] - candle["open"])
    upper_shadow = candle["high"] - max(candle["close"], candle["open"])
    lower_shadow = min(candle["close"], candle["open"]) - candle["low"]
    return upper_shadow > 2 * body and lower_shadow < body

# Main signal logic
def check_signal(candles_1h, candles_6h):
    # We'll base signals on 1H candles but check 6H trend

    # Extract closes for indicators
    closes_1h = [c["close"] for c in candles_1h]
    closes_6h = [c["close"] for c in candles_6h]

    # Calculate indicators on 1H
    ema_50_1h = calculate_ema(closes_1h, EMA_PERIOD)
    atr_1h = calculate_atr(candles_1h, ATR_PERIOD)
    rsi_1h = calculate_rsi(closes_1h, RSI_PERIOD)

    # Calculate EMA 50 on 6H for trend confirmation
    ema_50_6h = calculate_ema(closes_6h, EMA_PERIOD)

    idx = -1  # most recent candle

    if np.isnan(ema_50_1h[idx]) or np.isnan(atr_1h[idx]) or np.isnan(rsi_1h[idx]) or np.isnan(ema_50_6h[-1]):
        return None  # Not enough data yet

    price = closes_1h[idx]
    ema_50 = ema_50_1h[idx]
    atr = atr_1h[idx]
    rsi = rsi_1h[idx]
    ema_50_6h_now = ema_50_6h[-1]

    # Volatility check: ATR vs 20-period ATR average (using 1H)
    atr_avg = np.nanmean(atr_1h[-(ATR_PERIOD*2):-ATR_PERIOD])  # past ATR period excluding current
    if np.isnan(atr_avg):
        return None
    if atr < atr_avg:
        # Low volatility, skip signals
        return None

    # Trend direction by 6H EMA
    trend = "bull" if closes_6h[-1] > ema_50_6h_now else "bear"

    # Price action candles index in 1H
    # We'll check the last candle for engulfing / pin bar patterns
    candle = candles_1h[idx]

    # For price action checks, we might need idx-1 candle
    if idx - 1 < 0:
        return None

    signal = None

    # Long signal conditions:
    if trend == "bull":
        if price > ema_50:
            if rsi < 40:
                if is_bullish_engulfing(candles_1h, idx) or is_hammer(candle):
                    signal = "LONG"

    # Short signal conditions:
    if trend == "bear":
        if price < ema_50:
            if rsi > 60:
                if is_bearish_engulfing(candles_1h, idx) or is_shooting_star(candle):
                    signal = "SHORT"

    if signal:
        sl = round(price - 1.5 * atr, 2) if signal == "LONG" else round(price + 1.5 * atr, 2)
        tp = round(price + 2.5 * atr, 2) if signal == "LONG" else round(price - 2.5 * atr, 2)
        return {
            "signal": signal,
            "price": price,
            "stop_loss": sl,
            "take_profit": tp,
            "atr": round(atr, 4),
            "rsi": round(rsi, 2),
            "trend": trend
        }
    return None

# Keep track of last signals to avoid repeats
last_signals = {coin: None for coin in COINS}

def run_bot():
    print(f"Starting day-swing alert bot - {datetime.datetime.now()}")
    while True:
        for coin in COINS:
            try:
                candles_1h = fetch_candles(coin, 3600)
                candles_6h = fetch_candles(coin, 21600)
                if not candles_1h or not candles_6h:
                    print(f"Skipping {coin} due to missing data")
                    continue
                signal_data = check_signal(candles_1h, candles_6h)
                if signal_data and signal_data != last_signals[coin]:
                    msg = (
                        f"*{coin} {signal_data['signal']} Signal*\n"
                        f"Price: {signal_data['price']}\n"
                        f"Stop Loss: {signal_data['stop_loss']}\n"
                        f"Take Profit: {signal_data['take_profit']}\n"
                        f"ATR: {signal_data['atr']}\n"
                        f"RSI: {signal_data['rsi']}\n"
                        f"Trend (6H EMA): {signal_data['trend']}\n"
                        f"_Time: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}_"
                    )
                    send_telegram(msg)
                    print(f"Sent alert for {coin}: {signal_data['signal']}")
                    last_signals[coin] = signal_data
                else:
                    print(f"No new signal for {coin}")
            except Exception as e:
                print(f"Error processing {coin}: {e}")
        # Wait 1 hour before next check (can be adjusted)
        time.sleep(3600)

if __name__ == "__main__":
    run_bot()
