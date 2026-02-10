import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone

# === CONFIGURATION ===
FIB_LEVELS = [0.382, 0.5, 0.618, 0.66, 0.786]
GOLDEN_ZONE = (0.618, 0.66)
PRICE_BUFFER = 0.0025  # 0.25%
COOLDOWN_MINUTES = 30
TRADE_AMOUNT = 2500
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"
SYMBOLS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd",
           "fartcoin-usd", "aero-usd", "link-usd","hbar-usd",
           "aave-usd", "fet-usd", "crv-usd","tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd",
           "bch-usd", "inj-usd", "pepe-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
           "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
           "qnt-usd", "tia-usd", "ip-usd" , "pnut-usd", "apt-usd", "vet-usd",
           "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "coti-usd",
           "axs-usd"]

cooldowns = {}

# === FETCH FUNCTIONS ===

def fetch_ohlcv(symbol, granularity=3600):
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=granularity * 300)  # Max 300 candles
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    params = {
        "granularity": granularity,
        "start": start.isoformat(),
        "end": end.isoformat()
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch OHLCV for {symbol}")
        return None
    df = pd.DataFrame(response.json(), columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)
    return df

def fetch_current_price(symbol):
    url = f"https://api.coinbase.com/v2/prices/{symbol.upper()}/spot"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return float(resp.json()["data"]["amount"])
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

# === TECHNICAL INDICATORS ===

def calculate_stoch_rsi(df, period=14, smooth_k=3, smooth_d=3):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    min_rsi = rsi.rolling(period).min()
    max_rsi = rsi.rolling(period).max()
    stoch_rsi = 100 * (rsi - min_rsi) / (max_rsi - min_rsi)

    k = stoch_rsi.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.iloc[-1]

# === UTILITIES ===

def is_cooldown(symbol):
    now = datetime.now(timezone.utc)
    if symbol in cooldowns and (now - cooldowns[symbol]).total_seconds() < COOLDOWN_MINUTES * 60:
        return True
    cooldowns[symbol] = now
    return False

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

# === CORE LOGIC ===

def analyze_symbol(symbol):
    if is_cooldown(symbol):
        return

    df = fetch_ohlcv(symbol)
    if df is None or len(df) < 50:
        return
    price = fetch_current_price(symbol)
    if price is None:
        return

    recent_high = df["high"].max()
    recent_low = df["low"].min()
    fib_range = recent_high - recent_low
    golden_high = recent_high - fib_range * GOLDEN_ZONE[0]
    golden_low = recent_high - fib_range * GOLDEN_ZONE[1]

    k, d = calculate_stoch_rsi(df)
    if k.isna().any() or d.isna().any():
        return
    k_val, d_val = k.iloc[-1], d.iloc[-1]
    k_prev, d_prev = k.iloc[-2], d.iloc[-2]
    crossed = (k_prev < d_prev and k_val > d_val) or (k_prev > d_prev and k_val < d_val)

    atr = calculate_atr(df)

    in_golden_zone = golden_low <= price <= golden_high
    slightly_below = golden_low * (1 - PRICE_BUFFER) <= price < golden_low
    slightly_above = golden_high < price <= golden_high * (1 + PRICE_BUFFER)

    if crossed:
        if (in_golden_zone or slightly_below) and k_val < 20 and d_val < 20:
            tp = price + atr
            sl = golden_low * 0.995
            send_telegram_alert(
                f"🔔 LONG Signal: {symbol.upper()}\n"
                f"Price: {price:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\n"
                f"Golden Zone: [{golden_low:.4f} - {golden_high:.4f}]"
            )
        elif (in_golden_zone or slightly_above) and k_val > 80 and d_val > 80:
            tp = price - atr
            sl = golden_high * 1.005
            send_telegram_alert(
                f"🔔 SHORT Signal: {symbol.upper()}\n"
                f"Price: {price:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\n"
                f"Golden Zone: [{golden_low:.4f} - {golden_high:.4f}]"
            )

def run():
    print(f"Running alert check at {datetime.now(timezone.utc).isoformat()}")
    for symbol in SYMBOLS:
        analyze_symbol(symbol)

# === UNCOMMENT THIS TO RUN CONTINUOUSLY EVERY 5 MINUTES ===

while True:
     run()
     time.sleep(300)
