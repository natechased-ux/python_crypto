import ccxt
import pandas as pd
import numpy as np
import time
import requests
from scipy.signal import find_peaks
from datetime import datetime

# ==== Telegram Settings ====
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"

#TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
#CHAT_ID = "7967738614"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=payload)

# ==== Profitable whitelist ====
WHITELIST = {
    ("ETC-USD", "Butterfly", "Bearish"),
    ("HBAR-USD", "Bat", "Bearish"),
    ("AAVE-USD", "Crab", "Bullish"),
    ("FET-USD", "Bat", "Bullish"),
    ("FET-USD", "Crab", "Bullish"),
    ("AAVE-USD", "Bat", "Bullish"),
    ("BCH-USD", "Bat", "Bullish"),
    ("POL-USD", "Cypher", "Bullish"),
    ("BTC-USD", "Bat", "Bullish"),
    ("AVAX-USD", "Bat", "Bullish"),
    ("FLR-USD", "Cypher", "Bullish"),
    ("ARB-USD", "Cypher", "Bearish"),
    ("SUI-USD", "Gartley", "Bearish"),
    ("DOGE-USD", "Butterfly", "Bearish"),
    ("CRV-USD", "Butterfly", "Bullish"),
    ("ALGO-USD", "Butterfly", "Bullish"),
    ("BTC-USD", "Cypher", "Bullish"),
    ("LINK-USD", "Bat", "Bearish"),
    ("HBAR-USD", "Cypher", "Bullish"),
    ("VET-USD", "Three Drives", "Bullish"),
}

# ==== Pattern definitions ====
PATTERNS = {
    "Gartley": {"AB": (0.618, 0.618), "BC": (0.382, 0.886), "CD": (1.27, 1.618)},
    "Bat": {"AB": (0.382, 0.5), "BC": (0.382, 0.886), "CD": (1.618, 2.618)},
    "Butterfly": {"AB": (0.786, 0.786), "BC": (0.382, 0.886), "CD": (1.618, 2.24)},
    "Crab": {"AB": (0.382, 0.618), "BC": (0.382, 0.886), "CD": (2.618, 3.618)},
    "Deep Crab": {"AB": (0.886, 0.886), "BC": (0.382, 0.886), "CD": (2.618, 3.618)},
    "Shark": {"AB": (0.886, 1.13), "BC": (1.13, 1.618), "CD": (1.618, 2.24), "XD": (0.886, 0.886)},
    "Cypher": {"AB": (0.382, 0.618), "BC": (1.13, 1.414), "CD": (0.618, 0.786)},
    "Alternate Bat": {"AB": (0.382, 0.5), "BC": (0.382, 0.886), "CD": (2.0, 3.618)},
    "Three Drives": {"AB": (1.272, 1.618), "BC": (0.618, 0.786), "CD": (1.272, 1.618)},
}

# ==== CCXT exchange ====
exchange = ccxt.coinbase()

# ==== Helper functions ====
def ratio(a, b):
    return abs(a / b) if b != 0 else np.nan

def in_range(val, low, high, tol=0.1):
    return (low - tol) <= val <= (high + tol)

def match_pattern(pattern_rules, prices, X, A, B, C, D):
    XA = prices[A] - prices[X]
    AB = prices[B] - prices[A]
    BC = prices[C] - prices[B]
    CD = prices[D] - prices[C]
    if not in_range(ratio(AB, XA), *pattern_rules["AB"]): return False
    if not in_range(ratio(BC, AB), *pattern_rules["BC"]): return False
    if not in_range(ratio(CD, BC), *pattern_rules["CD"]): return False
    if "XD" in pattern_rules:
        XD = prices[D] - prices[X]
        if not in_range(ratio(XD, XA), *pattern_rules["XD"]): return False
    return True

def get_prz_and_bias(prices, points):
    X, A, B, C, D = points
    bias = "Bullish" if prices[D] < prices[C] else "Bearish"
    prz_low = min(prices[D], prices[C])
    prz_high = max(prices[D], prices[C])
    return bias, prz_low, prz_high

def fetch_ohlcv(symbol, limit=200):
    df = pd.DataFrame(exchange.fetch_ohlcv(symbol, '1h', limit=limit), 
                      columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ==== Live scanner loop ====
alerted_today = set()
last_checked_candle = None

while True:
    now = datetime.utcnow()
    # Only run check when a new 1H candle is closed
    if now.minute == 1:  # Run right after the hour closes
        for coin in set(c for c,_,_ in WHITELIST):
            try:
                df = fetch_ohlcv(coin)
                prices = df['close'].values
                peaks, _ = find_peaks(prices, distance=5)
                troughs, _ = find_peaks(-prices, distance=5)
                swings = sorted(np.concatenate([peaks, troughs]))

                if len(swings) < 5:
                    continue

                last_points = swings[-5:]
                for pname, rules in PATTERNS.items():
                    if match_pattern(rules, prices, *last_points):
                        bias, prz_low, prz_high = get_prz_and_bias(prices, last_points)
                        combo = (coin, pname, bias)
                        if combo in WHITELIST and combo not in alerted_today:
                            message = (
                                f"*Harmonic Alert*\n"
                                f"Coin: `{coin}`\n"
                                f"Pattern: `{pname}`\n"
                                f"Bias: `{bias}`\n"
                                f"PRZ: {prz_low:.2f} - {prz_high:.2f}\n"
                                f"Detected at: {df['timestamp'].iloc[-1]}"
                            )
                            send_telegram_message(message)
                            alerted_today.add(combo)
            except Exception as e:
                print(f"Error scanning {coin}: {e}")
        time.sleep(60)  # Wait to avoid running multiple times within the same minute
    else:
        time.sleep(20)  # Check again soon
