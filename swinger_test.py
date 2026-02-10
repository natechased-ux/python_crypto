import requests
import datetime
import numpy as np

# Coinbase API endpoint
COINBASE_API = "https://api.exchange.coinbase.com/products/{}/candles"

COINS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd",
           "syrup-usd","fartcoin-usd", "aero-usd", "link-usd","hbar-usd",
           "aave-usd", "fet-usd", "crv-usd","tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd",
           "bch-usd", "inj-usd", "pepe-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
           "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
           "qnt-usd", "tia-usd", "ip-usd" , "pnut-usd", "apt-usd", "vet-usd",
           "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "coti-usd",
           "axs-usd"]
EMA_PERIOD = 50
ATR_PERIOD = 14
RSI_PERIOD = 14

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
            atr.append(np.nan)
        elif i == period:
            atr.append(np.mean(trs[:period]))
        else:
            atr.append((atr[-1] * (period - 1) + trs[i]) / period)
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

def fetch_candles(product_id: str, granularity: int, start_iso: str, end_iso: str):
    from datetime import datetime, timedelta
    max_candles = 300
    start_dt = datetime.fromisoformat(start_iso)
    end_dt = datetime.fromisoformat(end_iso)

    all_candles = []

    while start_dt < end_dt:
        chunk_end = min(start_dt + timedelta(seconds=granularity * max_candles), end_dt)
        params = {
            "granularity": granularity,
            "start": start_dt.isoformat(),
            "end": chunk_end.isoformat(),
        }
        url = COINBASE_API.format(product_id)
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Failed fetching {product_id}: {response.text}")
            return None
        data = response.json()
        chunk_candles = [{
            "time": c[0],
            "low": c[1],
            "high": c[2],
            "open": c[3],
            "close": c[4],
            "volume": c[5],
        } for c in reversed(data)]
        all_candles.extend(chunk_candles)
        start_dt = chunk_end

    # Remove duplicates if any (overlapping edges)
    unique_candles = []
    seen_times = set()
    for c in all_candles:
        if c["time"] not in seen_times:
            unique_candles.append(c)
            seen_times.add(c["time"])

    return unique_candles


def is_bullish_engulfing(candles, idx):
    if idx == 0:
        return False
    c0 = candles[idx]
    c1 = candles[idx-1]
    return (c0["open"] < c0["close"] and
            c1["open"] > c1["close"] and
            c0["open"] < c1["close"] and
            c0["close"] > c1["open"])

def is_bearish_engulfing(candles, idx):
    if idx == 0:
        return False
    c0 = candles[idx]
    c1 = candles[idx-1]
    return (c0["open"] > c0["close"] and
            c1["open"] < c1["close"] and
            c0["open"] > c1["close"] and
            c0["close"] < c1["open"])

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

def check_signal(candles_1h, candles_6h, idx):
    # idx is the current candle index in 1H data
    closes_1h = [c["close"] for c in candles_1h]
    closes_6h = [c["close"] for c in candles_6h]

    if idx < EMA_PERIOD or idx < ATR_PERIOD or idx < RSI_PERIOD:
        return None

    ema_50_1h = calculate_ema(closes_1h[:idx+1], EMA_PERIOD)
    atr_1h = calculate_atr(candles_1h[:idx+1], ATR_PERIOD)
    rsi_1h = calculate_rsi(closes_1h[:idx+1], RSI_PERIOD)

    ema_50_6h = calculate_ema(closes_6h, EMA_PERIOD)

    if (np.isnan(ema_50_1h[-1]) or np.isnan(atr_1h[-1]) or np.isnan(rsi_1h[-1]) or np.isnan(ema_50_6h[-1])):
        return None

    price = closes_1h[idx]
    ema_50 = ema_50_1h[-1]
    atr = atr_1h[-1]
    rsi = rsi_1h[-1]
    ema_50_6h_now = ema_50_6h[-1]

    atr_avg = np.nanmean(atr_1h[-(ATR_PERIOD*2):-ATR_PERIOD]) if len(atr_1h) >= ATR_PERIOD*2 else atr
    if np.isnan(atr_avg):
        atr_avg = atr

    if atr < atr_avg:
        return None  # low volatility filter

    trend = "bull" if closes_6h[-1] > ema_50_6h_now else "bear"

    candle = candles_1h[idx]

    signal = None

    if trend == "bull":
        if price > ema_50:
            if rsi < 40:
                if is_bullish_engulfing(candles_1h, idx) or is_hammer(candle):
                    signal = "LONG"
    if trend == "bear":
        if price < ema_50:
            if rsi > 60:
                if is_bearish_engulfing(candles_1h, idx) or is_shooting_star(candle):
                    signal = "SHORT"

    if signal:
        sl = price - 1.5 * atr if signal == "LONG" else price + 1.5 * atr
        tp = price + 2.5 * atr if signal == "LONG" else price - 2.5 * atr
        return {
            "signal": signal,
            "price": price,
            "stop_loss": sl,
            "take_profit": tp,
            "atr": atr,
            "rsi": rsi,
            "trend": trend
        }
    return None

def simulate_trade(candles, entry_idx, signal_data):
    entry_price = signal_data["price"]
    sl = signal_data["stop_loss"]
    tp = signal_data["take_profit"]
    signal = signal_data["signal"]

    # Simulate candles after entry candle
    for i in range(entry_idx + 1, len(candles)):
        candle = candles[i]
        low = candle["low"]
        high = candle["high"]

        if signal == "LONG":
            # Check if SL hit first
            if low <= sl:
                # Loss
                return (sl - entry_price) / entry_price, False
            elif high >= tp:
                # Win
                return (tp - entry_price) / entry_price, True
        else:
            # SHORT
            if high >= sl:
                return (entry_price - sl) / entry_price, False
            elif low <= tp:
                return (entry_price - tp) / entry_price, True
    # If neither hit, exit at last candle close, assume small loss or gain
    exit_price = candles[-1]["close"]
    if signal == "LONG":
        return (exit_price - entry_price) / entry_price, False
    else:
        return (entry_price - exit_price) / entry_price, False

def backtest_coin(coin, days=30):
    print(f"Backtesting {coin} for last {days} days...")
    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(days=days)

    # Convert to ISO 8601
    start_iso = start.isoformat()
    end_iso = end.isoformat()

    candles_1h = fetch_candles(coin, 3600, start_iso, end_iso)
    candles_6h = fetch_candles(coin, 21600, start_iso, end_iso)
    if not candles_1h or not candles_6h:
        print(f"Skipping {coin} due to data fetch error")
        return

    trades = []
    for idx in range(max(EMA_PERIOD, ATR_PERIOD, RSI_PERIOD), len(candles_1h)):
        signal = check_signal(candles_1h, candles_6h, idx)
        if signal:
            ret, win = simulate_trade(candles_1h, idx, signal)
            trades.append({"return": ret, "win": win, "signal": signal["signal"], "entry_time": candles_1h[idx]["time"]})

    wins = sum(t["win"] for t in trades)
    total = len(trades)
    total_return = sum(t["return"] for t in trades) * 100  # % profit/loss

    if total > 0:
        print(f"Trades: {total}, Wins: {wins}, Win rate: {wins/total*100:.2f}%, Total Return: {total_return:.2f}%")
    else:
        print("No trades triggered.")

    # Optional: detailed report
    # for t in trades:
    #     print(t)

def main():
    for coin in COINS:
        backtest_coin(coin, days=30)

if __name__ == "__main__":
    main()
