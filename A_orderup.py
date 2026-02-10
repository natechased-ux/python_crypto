import asyncio
import requests
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.trend import MACD
from telegram import Bot

# Telegram Config
TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = Bot(token=TOKEN)

# Configuration
symbols = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd",
           "syrup-usd","fartcoin-usd", "aero-usd", "link-usd","hbar-usd",
           "aave-usd", "fet-usd", "crv-usd","tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd",
           "bch-usd", "inj-usd", "pepe-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
           "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
           "qnt-usd", "tia-usd", "ip-usd" , "pnut-usd", "apt-usd", "vet-usd",
           "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "velo-usd", "coti-usd",
           "axs-usd"]
granularity = 3600
atr_length = 14
vwap_length = 14
aggregation_threshold = 0.001
price_range_percentage = 0.25
volume_spike_multiplier = 1.2
risk_reward_ratio = 1.5

macd_enabled = False
rsi_enabled = False
cvd_enabled = False

# Whale threshold using top percentile
def get_whale_threshold(order_book, percentile=99.5):
    sizes = [float(x[1]) for x in order_book['bids'] + order_book['asks']]
    if not sizes:
        return 1
    return pd.Series(sizes).quantile(percentile / 100.0)

# Helper
def group_orders(orders, threshold):
    if not orders:
        return []
    orders.sort(key=lambda x: x[0])
    groups, current = [], [orders[0]]
    for price, qty in orders[1:]:
        if abs(price - current[-1][0]) / price <= threshold:
            current.append((price, qty))
        else:
            total_qty = sum(q for _, q in current)
            avg_price = sum(p * q for p, q in current) / total_qty
            groups.append((avg_price, total_qty))
            current = [(price, qty)]
    total_qty = sum(q for _, q in current)
    avg_price = sum(p * q for p, q in current) / total_qty
    groups.append((avg_price, total_qty))
    return groups

# Fetch candles
def fetch_data(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    res = requests.get(url)
    if res.status_code != 200:
        raise Exception(f"Error fetching candles for {symbol}")
    df = pd.DataFrame(res.json(), columns=["time", "low", "high", "open", "close", "volume"])
    df = df.sort_values(by="time")
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

# Indicators
def calculate_indicators(df):
    df['vwap'] = ((df['high'] + df['low'] + df['close']) / 3 * df['volume']).rolling(vwap_length).sum() / df['volume'].rolling(vwap_length).sum()
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=atr_length)
    df['atr'] = atr.average_true_range()
    df['avg_volume'] = df['volume'].rolling(atr_length).mean()
    df['volume_spike'] = df['volume'] > volume_spike_multiplier * df['avg_volume']

    if macd_enabled:
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()

    if cvd_enabled:
        df["delta"] = df["volume"].where(df["close"] > df["open"], -df["volume"])
        df["cvd"] = df["delta"].cumsum()

    return df

# Whale analysis
def analyze_whale_book(symbol, price):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
    res = requests.get(url)
    if res.status_code != 200:
        return None, None, None
    book = res.json()
    threshold = get_whale_threshold(book)
    bids = [(float(x[0]), float(x[1])) for x in book['bids'] if float(x[1]) >= threshold]
    asks = [(float(x[0]), float(x[1])) for x in book['asks'] if float(x[1]) >= threshold]

    grouped_bids = group_orders(bids, aggregation_threshold)
    grouped_asks = group_orders(asks, aggregation_threshold)

    nearby_bids = [b for b in grouped_bids if 0 < (price - b[0]) / price * 100 < price_range_percentage]
    nearby_asks = [a for a in grouped_asks if 0 < (a[0] - price) / price * 100 < price_range_percentage]

    bid = min(nearby_bids, key=lambda x: abs(x[0] - price), default=None)
    ask = min(nearby_asks, key=lambda x: abs(x[0] - price), default=None)

    # Resistance after entry
    future_ask = next((a for a in grouped_asks if a[0] > price), None)
    future_bid = next((b for b in grouped_bids if b[0] < price), None)

    return bid, ask, future_ask if bid else future_bid

# Analyzer
def analyze(symbol, df):
    df = calculate_indicators(df)
    row = df.iloc[-1]
    price = row['close']
    atr = row['atr']

    if pd.isna(atr) or not row['volume_spike']:
        return None

    if macd_enabled and row['macd'] < row['macd_signal']:
        return None

    if cvd_enabled:
        if row['cvd'] - df['cvd'].iloc[-2] < 0:
            return None

    bid, ask, resistance = analyze_whale_book(symbol, price)

    if bid:
        tp = price + atr * risk_reward_ratio
        sl = price - atr
        return f"{symbol.upper()} LONG\nEntry: {price:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\nSupport: {bid[0]:.4f}\nResistance: {resistance[0]:.4f}" if resistance else None
    elif ask:
        tp = price - atr * risk_reward_ratio
        sl = price + atr
        return f"{symbol.upper()} SHORT\nEntry: {price:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\nResistance: {ask[0]:.4f}\nSupport: {resistance[0]:.4f}" if resistance else None
    return None

# Send to Telegram
def notify(message):
    bot.send_message(chat_id=CHAT_ID, text=message)

# Main loop
def monitor():
    while True:
        try:
            messages = []
            for symbol in symbols:
                df = fetch_data(symbol)
                signal = analyze(symbol, df)
                if signal:
                    messages.append(signal)
            if messages:
                notify("\n\n".join(messages))
        except Exception as e:
            notify(f"Error: {str(e)}")
        time.sleep(60)

if __name__ == "__main__":
    import time
    monitor()
