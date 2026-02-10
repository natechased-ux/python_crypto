import schedule
import time
from datetime import datetime, timedelta
import pytz
import requests
import pandas as pd
import numpy as np
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator
import telegram

# === CONFIG ===
COINS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd", "ondo-usd", "sei-usd",
         "ape-usd", "jasmy-usd", "wld-usd", "aero-usd", "link-usd", "hbar-usd", "aave-usd", "avax-usd", "xcn-usd",
         "uni-usd", "mkr-usd", "near-usd", "algo-usd", "trump-usd", "bch-usd", "inj-usd", "xlm-usd", "bonk-usd",
         "dot-usd", "arb-usd", "icp-usd", "qnt-usd", "ip-usd", "ena-usd", "bera-usd", "pol-usd", "mask-usd",
         "pyth-usd", "mana-usd", "coti-usd", "c98-usd"]  # Removed poor performers
BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY = 21600  # 6H
TRADE_AMOUNT = 1000
TP_MULTIPLIER = 0.75
SL_MULTIPLIER = 0.5
LOOKBACK_DAYS = 14
BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"

# === FUNCTIONS ===
def fetch_candles(symbol):
    end = datetime.utcnow()
    start = end - timedelta(days=LOOKBACK_DAYS)
    url = f"{BASE_URL}/products/{symbol}/candles"
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "granularity": GRANULARITY
    }
    res = requests.get(url, params=params)
    try:
        data = res.json()
        df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except:
        return None

def calculate_indicators(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"])
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()
    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()
    return df

def find_swing_high(df, index, window=2):
    if index < window or index + window >= len(df):
        return None
    center = df["high"].iloc[index]
    return all(center > df["high"].iloc[index - i] and center > df["high"].iloc[index + i] for i in range(1, window + 1))

def find_swing_low(df, index, window=2):
    if index < window or index + window >= len(df):
        return None
    center = df["low"].iloc[index]
    return all(center < df["low"].iloc[index - i] and center < df["low"].iloc[index + i] for i in range(1, window + 1))

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg}
    requests.post(url, data=payload)

def check_signals():
    for coin in COINS:
        df = fetch_candles(coin)
        if df is None or len(df) < 50:
            continue
        df = calculate_indicators(df)
        last_row = df.iloc[-1]
        candle_time = df["time"].iloc[-1]
        pst_time = candle_time.tz_localize("UTC").astimezone(pytz.timezone("America/Los_Angeles"))

        # Find swings before last row
        swing_lows = [df["low"].iloc[i] for i in range(len(df)-1) if find_swing_low(df, i)]
        swing_highs = [df["high"].iloc[i] for i in range(len(df)-1) if find_swing_high(df, i)]

        # Long Signal
        if last_row["adx"] > 20 and last_row["plus_di"] > last_row["minus_di"] and last_row["macd_diff"] > 0 and last_row["rsi"] > 55:
            if swing_lows:
                sl = swing_lows[-1]
                entry = last_row["close"]
                risk = entry - sl
                tp = entry + TP_MULTIPLIER * risk
                msg = f"📈 LONG SIGNAL for {coin}\nEntry: {entry:.4f}\nSL: {sl:.4f}\nTP: {tp:.4f}\nTime: {pst_time.strftime('%Y-%m-%d %I:%M%p %Z')}"
                send_telegram(msg)

        # Short Signal
        elif last_row["adx"] > 20 and last_row["minus_di"] > last_row["plus_di"] and last_row["macd_diff"] < 0 and last_row["rsi"] < 45:
            if swing_highs:
                sl = swing_highs[-1]
                entry = last_row["close"]
                risk = sl - entry
                tp = entry - TP_MULTIPLIER * risk
                msg = f"📉 SHORT SIGNAL for {coin}\nEntry: {entry:.4f}\nSL: {sl:.4f}\nTP: {tp:.4f}\nTime: {pst_time.strftime('%Y-%m-%d %I:%M%p %Z')}"
                send_telegram(msg)

# === 6-HOUR SCHEDULING ===
def get_next_coinbase_6h_time():
    now = datetime.utcnow().replace(second=0, microsecond=0)
    hours_since_midnight = now.hour
    next_6h = ((hours_since_midnight // 6) + 1) * 6
    if next_6h >= 24:
        next_run = now.replace(hour=0, minute=5) + timedelta(days=1)
    else:
        next_run = now.replace(hour=next_6h, minute=5)
    return next_run

def schedule_6h_task():
    def run_and_reschedule():
        print(f"✅ Running check at {datetime.utcnow().isoformat()}")
        check_signals()
        schedule_6h_task()  # Reschedule next run

    next_run_time = get_next_coinbase_6h_time()
    delay = (next_run_time - datetime.utcnow()).total_seconds()
    print(f"🔔 Next signal check at {next_run_time.isoformat()} UTC")
    schedule.clear()
    schedule.every(delay).seconds.do(run_and_reschedule)


# === START SCHEDULER ===
check_signals()
schedule_6h_task()
while True:
    schedule.run_pending()
    time.sleep(1)
