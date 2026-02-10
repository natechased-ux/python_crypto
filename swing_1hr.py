import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator

# === CONFIGURATION ===
COINS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
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
           "axs-usd"]
BASE_URL = "https://api.exchange.coinbase.com"
CANDLE_GRANULARITY = 3600  # 1-hour
SWING_LOOKBACK = 20  # candles to find swing highs/lows
ADX_THRESHOLD = 20
MACD_CONFIRMATION = True
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
COOLDOWN_MINUTES = 30
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"

last_alert_time = {}

def fetch_candles(product_id, granularity):
    end = datetime.utcnow()
    start = end - timedelta(seconds=granularity * 300)
    url = f"{BASE_URL}/products/{product_id}/candles?granularity={granularity}&start={start.isoformat()}&end={end.isoformat()}"
    time.sleep(0.25)
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.sort_values("time")

def calculate_indicators(df):
    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"])
    macd = MACD(close=df["close"])
    rsi = RSIIndicator(close=df["close"])
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()
    df["macd_diff"] = macd.macd_diff()
    df["rsi"] = rsi.rsi()
    return df

def find_swing_highs_lows(df, lookback):
    highs = df["high"].rolling(lookback).max()
    lows = df["low"].rolling(lookback).min()
    return highs, lows

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=payload)

def process_coin(coin):
    global last_alert_time
    now = datetime.utcnow()

    if coin in last_alert_time and (now - last_alert_time[coin]).total_seconds() < COOLDOWN_MINUTES * 60:
        return

    try:
        df = fetch_candles(coin, CANDLE_GRANULARITY)
        df = calculate_indicators(df)
        highs, lows = find_swing_highs_lows(df, SWING_LOOKBACK)

        current_price = df["close"].iloc[-1]
        adx = df["adx"].iloc[-1]
        plus_di = df["plus_di"].iloc[-1]
        minus_di = df["minus_di"].iloc[-1]
        macd_delta = df["macd_diff"].iloc[-1]
        rsi = df["rsi"].iloc[-1]

        print(f"{coin.upper()} ➜ Price: {current_price:.2f}, ADX: {adx:.1f}, +DI: {plus_di:.1f}, -DI: {minus_di:.1f}, MACD Δ: {macd_delta:.4f}, RSI: {rsi:.1f}")

        # === Structure-based LONG
        if plus_di > minus_di and adx > ADX_THRESHOLD and (not MACD_CONFIRMATION or macd_delta > 0) and rsi < RSI_OVERBOUGHT:
            sl = lows.iloc[-SWING_LOOKBACK]
            tp = current_price + 2 * (current_price - sl)
            message = f"🚀 LONG {coin.upper()} (Structure + Momentum)\nEntry: {current_price:.4f}\nTarget: {tp:.4f}\nSL: {sl:.4f}\nADX: {adx:.2f}, MACD Δ: {macd_delta:.4f}, RSI: {rsi:.1f}"
            send_telegram_message(message)
            last_alert_time[coin] = now
            return

        # === Structure-based SHORT
        if minus_di > plus_di and adx > ADX_THRESHOLD and (not MACD_CONFIRMATION or macd_delta < 0) and rsi > RSI_OVERSOLD:
            sl = highs.iloc[-SWING_LOOKBACK]
            tp = current_price - 2 * (sl - current_price)
            message = f"🔻 SHORT {coin.upper()} (Structure + Momentum)\nEntry: {current_price:.4f}\nTarget: {tp:.4f}\nSL: {sl:.4f}\nADX: {adx:.2f}, MACD Δ: {macd_delta:.4f}, RSI: {rsi:.1f}"
            send_telegram_message(message)
            last_alert_time[coin] = now
            return

    except Exception as e:
        print(f"Error processing {coin}: {e}")

# === LOOP ===
if __name__ == "__main__":
    while True:
        print(f"\n[{datetime.utcnow().isoformat()}] Scanning structure-based signals...")
        for coin in COINS:
            process_coin(coin)
        time.sleep(60)

