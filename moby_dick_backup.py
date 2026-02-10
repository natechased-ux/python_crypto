# === ALERT SYSTEM ===

import requests
import pandas as pd
import time
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from telegram import Bot

# Telegram config
TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = Bot(token=TOKEN)

# Configuration
symbols = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd",
           "syrup-usd", "fartcoin-usd", "aero-usd", "link-usd", "hbar-usd",
           "aave-usd", "fet-usd", "crv-usd", "tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd",
           "bch-usd", "inj-usd", "pepe-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
           "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
           "qnt-usd", "tia-usd", "ip-usd", "pnut-usd", "apt-usd",
           "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd",
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "velo-usd", "coti-usd", "c98-usd",
           "axs-usd"] # Truncated for example
granularity = 3600
atr_length = 14
vwap_length = 14
rsi_length = 14
rsi_long_threshold = 40
rsi_short_threshold = 60
bin_width_percent = 0.2
price_range_percent = 10
volume_spike_multiplier = 0.5
risk_reward_ratio = 1.5
whale_cluster_percentile = 95
conflict_check_range = 1.5
conflict_volume_ratio = 0.9
cooldown_minutes = 30

# State tracking
last_alert_times = {}
last_signals = {}  # Track last signal state

# Helpers
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

def get_live_price(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/ticker"
    res = requests.get(url)
    if res.status_code != 200:
        raise Exception(f"Error fetching live price for {symbol}")
    return float(res.json()["price"])

def calculate_indicators(df):
    df['vwap'] = ((df['high'] + df['low'] + df['close']) / 3 * df['volume']).rolling(vwap_length).sum() / df['volume'].rolling(vwap_length).sum()
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=atr_length)
    df['atr'] = atr.average_true_range()
    df['avg_volume'] = df['volume'].rolling(atr_length).mean()
    df['volume_spike'] = df['volume'] > volume_spike_multiplier * df['avg_volume']
    df['rsi'] = RSIIndicator(df['close'], window=rsi_length).rsi()
    return df

def notify(message):
    bot.send_message(chat_id=CHAT_ID, text=message)

def bin_orders(orders, price, bin_width_percent):
    bin_width = price * bin_width_percent / 100
    binned = {}
    for p, q in orders:
        if abs(p - price) / price * 100 > price_range_percent:
            continue
        bin_price = round(p / bin_width) * bin_width
        binned[bin_price] = binned.get(bin_price, 0) + q
    return [(k, v) for k, v in binned.items()]

def analyze(symbol, df):
    now = time.time()
    df = calculate_indicators(df)
    row = df.iloc[-1]
    price = get_live_price(symbol)
    atr = row['atr']
    rsi = row['rsi']
    if pd.isna(atr) or not row['volume_spike'] or pd.isna(rsi):
        return None

    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
    res = requests.get(url)
    if res.status_code != 200:
        return None
    book = res.json()

    bids = [(float(x[0]), float(x[1])) for x in book['bids']]
    asks = [(float(x[0]), float(x[1])) for x in book['asks']]

    binned_bids = bin_orders(bids, price, bin_width_percent)
    binned_asks = bin_orders(asks, price, bin_width_percent)

    bid_volumes = [q for _, q in binned_bids]
    ask_volumes = [q for _, q in binned_asks]

    bid_cutoff = pd.Series(bid_volumes).quantile(whale_cluster_percentile / 100) if bid_volumes else 0
    ask_cutoff = pd.Series(ask_volumes).quantile(whale_cluster_percentile / 100) if ask_volumes else 0

    whale_bids = [b for b in binned_bids if b[1] >= bid_cutoff and 0 < (price - b[0]) / price * 100 < conflict_check_range]
    whale_asks = [a for a in binned_asks if a[1] >= ask_cutoff and 0 < (a[0] - price) / price * 100 < conflict_check_range]

    support = min(whale_bids, key=lambda x: abs(x[0] - price), default=None)
    resistance = min(whale_asks, key=lambda x: abs(x[0] - price), default=None)

    nearby_bid = support
    nearby_ask = resistance

    def build_signal(direction, entry, tp, sl, cluster, opposing):
        return {
            'direction': direction,
            'entry': round(entry, 4),
            'tp': round(tp, 4),
            'sl': round(sl, 4),
            'cluster': cluster,
            'opposing': opposing
        }

    def signal_changed(old, new):
        if not old:
            return True
        delta = lambda a, b: abs(a - b) / ((a + b) / 2)
        return (
            old['direction'] != new['direction'] or
            delta(old['entry'], new['entry']) > 0.001 or
            delta(old['tp'], new['tp']) > 0.01 or
            delta(old['sl'], new['sl']) > 0.01
        )

    message = None
    new_signal = None
    if support and rsi < rsi_long_threshold and (not nearby_ask or nearby_ask[1] < conflict_volume_ratio * support[1]):
        tp = price + atr * risk_reward_ratio
        sl = support[0] - 0.001
        new_signal = build_signal("long", price, tp, sl, support, nearby_ask)
        if signal_changed(last_signals.get(symbol), new_signal):
            message = (f"{symbol.upper()} LONG\nEntry: {price:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\n"
                       f"Support: {support[0]:.4f} (vol: {support[1]:.0f})\n"
                       f"Nearby Resistance: {nearby_ask[0]:.4f} (vol: {nearby_ask[1]:.0f})" if nearby_ask else "")
    elif resistance and rsi > rsi_short_threshold and (not nearby_bid or nearby_bid[1] < conflict_volume_ratio * resistance[1]):
        tp = price - atr * risk_reward_ratio
        sl = resistance[0] + 0.001
        new_signal = build_signal("short", price, tp, sl, resistance, nearby_bid)
        if signal_changed(last_signals.get(symbol), new_signal):
            message = (f"{symbol.upper()} SHORT\nEntry: {price:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\n"
                       f"Resistance: {resistance[0]:.4f} (vol: {resistance[1]:.0f})\n"
                       f"Nearby Support: {nearby_bid[0]:.4f} (vol: {nearby_bid[1]:.0f})" if nearby_bid else "")

    if message:
        last_alert_times[symbol] = now
        last_signals[symbol] = new_signal
    return message

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
    monitor()
