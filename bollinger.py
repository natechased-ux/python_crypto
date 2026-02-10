
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD

# === User Configuration ===
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"
COINS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
         "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
         "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd", "avax-usd", "xcn-usd", "uni-usd",
         "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
         "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd", "qnt-usd", "tia-usd", "ip-usd",
         "pnut-usd", "apt-usd", "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "pyth-usd", "sand-usd",
         "morpho-usd", "mana-usd", "coti-usd", "c98-usd", "axs-usd"]
VOLUME_SPIKE_MULTIPLIER = 1.5
BOLL_BAND_WIDTH_THRESHOLD = 0.015
cooldowns = {}

def format_price(value):
    if value >= 100:
        return f"${value:.2f}"
    elif value >= 1:
        return f"${value:.4f}"
    else:
        return f"${value:.6f}"

def fetch_data(symbol, granularity='900', limit=100):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    try:
        df = pd.DataFrame(requests.get(url).json(), columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.sort_values("time")
        return df.tail(limit)
    except:
        return None

def compute_indicators(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    bb = BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"]).adx()
    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()
    return df

def send_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Telegram error:", e)

def check_mean_reversion(symbol, df, live_price):
    latest = df.iloc[-1]
    if latest["close"] < latest["bb_lower"] and latest["rsi"] < 25:
        tp = latest["bb_middle"]
        sl = latest["close"] - 1.0 * latest["atr"]
        direction = "Long"
    elif latest["close"] > latest["bb_upper"] and latest["rsi"] > 75:
        tp = latest["bb_middle"]
        sl = latest["close"] + 1.0 * latest["atr"]
        direction = "Short"
    else:
        return
    msg =(f"🎯 *Mean Reversion Alert* — {symbol.upper()}\n"
    f"Side: {direction}\n"
    f"Live Price: {format_price(live_price)}\n"
    f"TP: {format_price(tp)} | SL: {format_price(sl)}")
    send_alert(msg)

def check_breakout(symbol, df, live_price):
    latest = df.iloc[-1]
    vol_mean = df["volume"].rolling(20).mean().iloc[-1]
    if (
        latest["adx"] > 30 and
        latest["bb_width"] < BOLL_BAND_WIDTH_THRESHOLD and
        latest["volume"] > VOLUME_SPIKE_MULTIPLIER * vol_mean
    ):
        atr = latest["atr"]
        direction = "Long" if live_price > latest["bb_upper"] else "Short"
        tp = live_price + 1.5 * atr if direction == "Long" else live_price - 1.5 * atr
        sl = live_price - 1.0 * atr if direction == "Long" else live_price + 1.0 * atr
        msg = (f"🚀 *Breakout Alert* — {symbol.upper()}\n"
        f"Side: {direction}\n"
        f"Live Price: {format_price(live_price)}\n"
        f"TP: {format_price(tp)} | SL: {format_price(sl)}")
        send_alert(msg)

def check_breakout_imminent(symbol, df, live_price):
    latest = df.iloc[-1]
    bb_width = latest["bb_width"]
    macd_diff_1 = df["macd_diff"].diff().iloc[-1]
    macd_diff_2 = df["macd_diff"].diff().iloc[-2]
    macd_up = macd_diff_1 > 0 and macd_diff_2 > 0
    macd_down = macd_diff_1 < 0 and macd_diff_2 < 0
    vol_avg = df["volume"].rolling(3).mean()
    vol_up = vol_avg.diff().iloc[-1] > 0 and vol_avg.diff().iloc[-2] > 0
    if bb_width < BOLL_BAND_WIDTH_THRESHOLD and vol_up:
        direction = None
        if macd_up and live_price > latest["bb_middle"]:
            direction = "Bullish"
        elif macd_down and live_price < latest["bb_middle"]:
            direction = "Bearish"
        if direction:
            atr = latest["atr"]
            tp = live_price + 1.5 * atr if direction == "Bullish" else live_price - 1.5 * atr
            sl = live_price - 1.0 * atr if direction == "Bullish" else live_price + 1.0 * atr
            msg = (
                f"⚠️ *Breakout Imminent* for *{symbol.upper()}*\n"

                f"Bias: {direction}\n"
                f"Live Price: {format_price(live_price)}\n"

                f"TP: {format_price(tp)} | SL: {format_price(sl)}\n"

                f"Pressing {'upper' if direction == 'Bullish' else 'lower'} band with squeeze and momentum."
            )
            send_alert(msg)

def run_alerts():
    now = datetime.utcnow()
    for symbol in COINS:
        if symbol in cooldowns and (now - cooldowns[symbol]).total_seconds() < 1800:
            continue
        df = fetch_data(symbol)
        if df is None or df.empty:
            continue
        df = compute_indicators(df)
        live_price = df["close"].iloc[-1]
        check_mean_reversion(symbol, df, live_price)
        #check_breakout(symbol, df, live_price)
        #check_breakout_imminent(symbol, df, live_price)
        cooldowns[symbol] = now

if __name__ == "__main__":
    while True:
        run_alerts()
        time.sleep(300)
