#!/usr/bin/env python3
"""
crypto_telegram_alerts.py

Scans Coinbase products, computes EMAs/MAs and a simple MSB breakout rule on timeframes
(4h, 1d, 1w) and sends alerts to Telegram when conditions match the images you provided.

Requirements:
    pip install requests pandas numpy python-dateutil

Environment variables:
    TELEGRAM_BOT_TOKEN - your telegram bot token
    TELEGRAM_CHAT_ID  - chat id to send alerts to (or channel id)
Optional:
    POLL_INTERVAL_SECS - how often to rescan (default 300)
"""

import os
import time
import math
import requests
import pandas as pd
import numpy as np
from dateutil import parser

# -------------- CONFIG --------------
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECS", "3600"))  # seconds

COINBASE_API = "https://api.exchange.coinbase.com"
# which timeframes to check (we map to Coinbase granularity in seconds)
# which timeframes to check
# 4h candles must be created manually via resampling 1h candles
TIMEFRAMES = {
    "4h": 3600,           # fetch 1h, then resample to 4h
    "1d": 86400,          # valid
             # INVALID on Coinbase – remove or also resample
}


# EMA periods from image
EMA_PERIODS = [25, 50, 75, 150, 200]

# MA periods from image
MA7 = 7
MA25 = 25
MA99 = 99

# MSB lookback high window (how many candles to consider for previous structure)
MSB_LOOKBACK = {
    "4h": 100,  # ~ ~16 days of 4h candles
    "1d": 120,  # ~4 months daily
    "1w": 52    # 1 year of weekly candles
}

# thresholds
EMA_NEAR_PCT = 0.05  # price within 5% of EMA25/EMA50 counts as "sit on EMA support"
PRICE_ABOVE_EMA_REQUIRED = True  # require price above EMA25 to consider trend support

# filter products: only USD quote (can be changed to include USDC, USDT)
QUOTE_CURRENCY = "USD"

# optional: only alert for spot products (not margin/perp)
PRODUCT_TYPE_FILTER = None  # None or list like ["spot"] - Coinbase product object has 'quote_currency', etc.

# -------------- helper functions --------------

def resample_to_4h(df):
    df = df.set_index("time")
    df4 = pd.DataFrame()
    df4["open"]   = df["open"].resample("4H").first()
    df4["high"]   = df["high"].resample("4H").max()
    df4["low"]    = df["low"].resample("4H").min()
    df4["close"]  = df["close"].resample("4H").last()
    df4["volume"] = df["volume"].resample("4H").sum()
    df4.dropna(inplace=True)
    df4 = df4.reset_index()
    return df4


def send_telegram(text):
    token = TELEGRAM_BOT_TOKEN
    chat = TELEGRAM_CHAT_ID
    if "<YOUR_TOKEN>" in token or "<YOUR_CHAT_ID>" in chat:
        print("Telegram token/chat not set in environment. Would send:")
        print(text)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print("Failed to send telegram:", e, r.text if 'r' in locals() else "")

def coinbase_get(path, params=None):
    url = COINBASE_API + path
    headers = {"User-Agent": "crypto-alert-script/1.0"}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()

def get_products():
    # returns product dicts
    try:
        prods = coinbase_get("/products")
    except Exception as e:
        print("Error fetching products:", e)
        return []
    # filter by quote currency
    out = []
    for p in prods:
        if p.get("quote_currency", "").upper() == QUOTE_CURRENCY:
            out.append(p)
    return out

def fetch_candles(product_id, granularity, limit=500):
    # granularity in seconds
    # Coinbase returns [time, low, high, open, close, volume] lists
    params = {"granularity": granularity}
    try:
        raw = coinbase_get(f"/products/{product_id}/candles", params=params)
    except Exception as e:
        print("Error fetching candles for", product_id, e)
        return None
    if not raw:
        return None
    # convert to DataFrame
    df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    # Coinbase returns in reverse chronological
    df = df.sort_values("time").reset_index(drop=True)
    return df

# indicator helpers
def compute_emas(df, periods):
    for p in periods:
        df[f"EMA{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    return df

def compute_mas(df, periods):
    for p in periods:
        df[f"MA{p}"] = df["close"].rolling(window=p, min_periods=1).mean()
    return df

# -------------- signal rules --------------

def check_ema_trend_support(df):
    """
    Rule (from first image):
    - EMAs 25,50,75,150,200 should be in bullish order: EMA25 > EMA50 > EMA75 > EMA150 > EMA200
    - Price should be >= EMA25 (or within EMA_NEAR_PCT above/below)
    - If price is within EMA_NEAR_PCT above/below EMA25 or between EMA25 and EMA50, consider 'support' bounce
    """
    last = df.iloc[-1]
    # ensure EMAs exist
    keys = [f"EMA{p}" for p in EMA_PERIODS]
    if any(k not in df.columns for k in keys):
        return False, None

    emavalues = [last[k] for k in keys]
    # bullish order: EMA25 > EMA50 > EMA75 > ...
    bullish = all(emavalues[i] > emavalues[i+1] for i in range(len(emavalues)-1))
    price = last["close"]
    ema25 = last["EMA25"]

    pct_from_ema25 = (price - ema25) / ema25 if ema25 and ema25 > 0 else 0

    near_support = abs(pct_from_ema25) <= EMA_NEAR_PCT and (not PRICE_ABOVE_EMA_REQUIRED or price >= ema25)
    between_25_50 = last["EMA50"] < price <= ema25 or ema25 <= price <= last["EMA50"]

    if bullish:
        msg = {
            "price": price,
            "pct_from_ema25": pct_from_ema25,
            "ema_values": dict(zip(keys, emavalues))
        }
        return True, msg
    return False, None

def check_ma_cross(df):
    """
    Rule (from second image):
    - MA7 crossing MA99 upwards (previous candle MA7 <= MA99 and current MA7 > MA99)
    - Also optionally check MA25 alignment.
    """
    if f"MA{MA7}" not in df.columns or f"MA{MA99}" not in df.columns:
        return False, None
    # need at least 2 points
    if len(df) < 2:
        return False, None
    prev = df.iloc[-2]
    last = df.iloc[-1]
    prev7 = prev[f"MA{MA7}"]; prev99 = prev[f"MA{MA99}"]
    last7 = last[f"MA{MA7}"]; last99 = last[f"MA{MA99}"]

    crossed_up = (prev7 <= prev99) and (last7 > last99)
    if crossed_up:
        msg = {
            "prev7": prev7, "prev99": prev99, "last7": last7, "last99": last99, "price": last["close"]
        }
        return True, msg
    return False, None



# -------------- main scan --------------

def scan_and_alert():
    products = get_products()
    if not products:
        print("No products found.")
        return

    # filter out suspicious product ids (keep simple ones)
    targets = [p["id"] for p in products if "/" not in p["id"]]  # product ids like BTC-USD (contain dash) are fine
    # We'll iterate through them
    for pid in targets:
        # quick skip for weird ids (optional)
        # fetch candles per timeframe
        try:
            for tf_name, gran in TIMEFRAMES.items():
    
    # --- 4h timeframe (special handling) ---
                if tf_name == "4h":
        # fetch 1h candles
                    df1h = fetch_candles(pid, granularity=3600)
                    if df1h is None or len(df1h) < 50:
                        continue
        
        # convert 1h → 4h
                    df = resample_to_4h(df1h)
    
                else:
        # standard fetch
                    df = fetch_candles(pid, granularity=gran)
    
                if df is None or len(df) < 10:
                    continue

                # compute indicators
                df = compute_emas(df, EMA_PERIODS)
                df = compute_mas(df, [MA7, MA25, MA99])

                # EMA support
                ema_ok, ema_msg = check_ema_trend_support(df)
                ma_ok, ma_msg = check_ma_cross(df)

                if ema_ok and  ma_ok:
     

                # MA cross
            
                        last_time = df["time"].iloc[-1]
                        text = (f"🚀 <b>MA Cross Alert</b>\n"
                                f"<b>{pid} · {tf_name}</b>\n"
                                f"MA{MA7}: {ma_msg['last7']:.6f} crossed above MA{MA99}: {ma_msg['last99']:.6f}\n"
                                f"Price: {ma_msg['price']:.8f}\n"
                                f"Time: {last_time}\n"
                                f"Tip: 'Wait for 99's cross always' (per rule)")
                        send_telegram(text)


                # rate limiting between timeframes
                time.sleep(0.33)
        except Exception as e:
            print("Error scanning", pid, e)
        # small sleep between products to avoid API throttling
        time.sleep(0.2)


if __name__ == "__main__":
    print("Starting crypto Telegram alerts (Coinbase)...")
    # run in loop
    while True:
        try:
            scan_and_alert()
        except Exception as e:
            print("Top-level error:", e)
        time.sleep(POLL_INTERVAL)
