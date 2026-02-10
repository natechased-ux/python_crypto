import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from pytz import timezone
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator
import schedule
import threading
import telegram
import warnings
import csv
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# === CONFIGURATION ===
COINS = ["btc-usd", "eth-usd", "xrp-usd", "doge-usd"]
BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY_6H = 21600
GRANULARITY_1H = 3600
GRANULARITY_1D = 86400
TP_MULTIPLIER = 0.75
CONFIRMATION_TIMEOUT_HOURS = 6

open_confirmations = {}

# === Price formatting ===
def format_price(price):
    if price >= 100:
        return f"{price:.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.6f}"
    elif price >= 0.0001:
        return f"{price:.8f}"
    else:
        return f"{price:.10f}"

# === EMA text formatting ===
def get_all_daily_emas_text(daily_emas):
    lines = ["📊 Daily EMAs:"]
    for p in [10, 20, 50, 200]:
        ema_val = daily_emas.get(f"ema{p}")
        if ema_val is not None:
            lines.append(f"• EMA{p}: {format_price(ema_val)}")
    return "\n".join(lines)

def get_hourly_50_200_text(hourly_emas):
    lines = ["📊 Hourly EMAs:"]
    for p in [50, 200]:
        ema_val = hourly_emas.get(f"ema{p}")
        if ema_val is not None:
            lines.append(f"• EMA{p}: {format_price(ema_val)}")
    return "\n".join(lines)

# === Coinbase API ===
def fetch_candles(symbol, granularity, lookback_days=30):
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    df = pd.DataFrame()
    while start < end:
        chunk_end = min(start + timedelta(hours=300), end)
        params = {
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "granularity": granularity
        }
        url = f"{BASE_URL}/products/{symbol}/candles"
        res = requests.get(url, params=params)
        time.sleep(0.25)
        try:
            data = res.json()
            if isinstance(data, list) and data:
                chunk = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
                df = pd.concat([df, chunk], ignore_index=True)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        start = chunk_end
    if df.empty:
        return None
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# === Indicators ===
def calculate_indicators(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    adx = ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()
    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()
    for period in [10, 20, 50, 200]:
        df[f"ema{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df

# === Divergence detection ===
def detect_rsi_divergence(df, side="bullish", rsi_period=14, swing_lookback=5):
    if len(df) < swing_lookback * 2 + rsi_period + 1:
        return False
    df = df.copy()
    df["rsi"] = RSIIndicator(df["close"], window=rsi_period).rsi()
    df["swing_low"] = df["low"].rolling(window=swing_lookback*2+1, center=True).apply(
        lambda x: x[swing_lookback] == min(x), raw=True).fillna(0)
    df["swing_high"] = df["high"].rolling(window=swing_lookback*2+1, center=True).apply(
        lambda x: x[swing_lookback] == max(x), raw=True).fillna(0)
    lows = df[df["swing_low"] == 1]
    highs = df[df["swing_high"] == 1]
    if side in ["bullish", "hidden_bullish"] and len(lows) >= 2:
        p1, p2 = lows["low"].iloc[-2], lows["low"].iloc[-1]
        r1, r2 = lows["rsi"].iloc[-2], lows["rsi"].iloc[-1]
        if side == "bullish" and p2 < p1 and r2 > r1:
            return True
        if side == "hidden_bullish" and p2 > p1 and r2 < r1:
            return True
    if side in ["bearish", "hidden_bearish"] and len(highs) >= 2:
        p1, p2 = highs["high"].iloc[-2], highs["high"].iloc[-1]
        r1, r2 = highs["rsi"].iloc[-2], highs["rsi"].iloc[-1]
        if side == "bearish" and p2 > p1 and r2 < r1:
            return True
        if side == "hidden_bearish" and p2 < p1 and r2 > r1:
            return True
    return False

# === Telegram ===
def send_telegram_alert(message):
    try:
        bot.send_message(chat_id=CHAT_ID, text=message)
    except Exception as e:
        print(f"Telegram error: {e}")

# === Monitor Divergence ===
def monitor_divergence(symbol, side, entry, tp, sl, signal_time):
    timeout = datetime.utcnow() + timedelta(hours=CONFIRMATION_TIMEOUT_HOURS)
    print(f"🕒 Monitoring {symbol.upper()} for Stoch RSI entry confirmation...")
    while datetime.utcnow() < timeout:
        df_1h = fetch_candles(symbol, GRANULARITY_1H, lookback_days=12)
        df_1d = fetch_candles(symbol, GRANULARITY_1D, lookback_days=300)
        if df_1h is None or df_1d is None:
            time.sleep(300)
            continue
        for p in [10, 20, 50, 200]:
            df_1h[f"ema{p}"] = df_1h["close"].ewm(span=p, adjust=False).mean()
            df_1d[f"ema{p}"] = df_1d["close"].ewm(span=p, adjust=False).mean()
        hourly_emas = {f"ema{p}": df_1h[f"ema{p}"].iloc[-1] for p in [10, 20, 50, 200]}
        daily_emas = {f"ema{p}": df_1d[f"ema{p}"].iloc[-1] for p in [10, 20, 50, 200]}
        bullish = detect_rsi_divergence(df_1h, "bullish") or detect_rsi_divergence(df_1h, "hidden_bullish")
        bearish = detect_rsi_divergence(df_1h, "bearish") or detect_rsi_divergence(df_1h, "hidden_bearish")
        confirmed_price = df_1h["close"].iloc[-1]
        if side == "long" and bullish:
            alert = (
                f"✅ DIV *LONG Confirmed* on {symbol.upper()}\n"
                f"Entry: {format_price(confirmed_price)}\n"
                f"TP: {format_price(tp)}\n"
                f"SL: {format_price(sl)}\nSignal Time: {signal_time}\n\n"
                f"{get_all_daily_emas_text(daily_emas)}\n\n"
                f"{get_hourly_50_200_text(hourly_emas)}"
            )
            send_telegram_alert(alert)
            break
        elif side == "short" and bearish:
            alert = (
                f"✅ DIV *SHORT Confirmed* on {symbol.upper()}\n"
                f"Entry: {format_price(confirmed_price)}\n"
                f"TP: {format_price(tp)}\n"
                f"SL: {format_price(sl)}\nSignal Time: {signal_time}\n\n"
                f"{get_all_daily_emas_text(daily_emas)}\n\n"
                f"{get_hourly_50_200_text(hourly_emas)}"
            )
            send_telegram_alert(alert)
            break
        time.sleep(300)

# === Signal check ===
def check_signals():
    for symbol in COINS:
        df = fetch_candles(symbol, GRANULARITY_6H)
        if df is None:
            continue
        df = calculate_indicators(df)
        last = df.iloc[-1]
        entry_price = last["close"]
        signal_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        if last["adx"] > 20 and last["plus_di"] > last["minus_di"] and last["macd_diff"] > 0:
            sl = entry_price * 0.98
            tp = entry_price * 1.02
            threading.Thread(target=monitor_divergence, args=(symbol, "long", entry_price, tp, sl, signal_time)).start()
        elif last["adx"] > 20 and last["minus_di"] > last["plus_di"] and last["macd_diff"] < 0:
            sl = entry_price * 1.02
            tp = entry_price * 0.98
            threading.Thread(target=monitor_divergence, args=(symbol, "short", entry_price, tp, sl, signal_time)).start()

def get_next_coinbase_6h_time():
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    # Coinbase 6H candle times: 00:00, 06:00, 12:00, 18:00 UTC
    next_times = [0, 6, 12, 18]
    for t in next_times:
        candidate = now.replace(hour=t)
        if candidate > now:
            return candidate + timedelta(minutes=5)  # buffer
    # If none ahead today, go to the first of the next day
    return (now + timedelta(days=1)).replace(hour=0) + timedelta(minutes=5)


def schedule_6h_task():
    def run_and_reschedule():
        print(f"🕒 Running check at {datetime.utcnow().isoformat()}")
        check_signals()
        schedule_6h_task()

    next_run_time = get_next_coinbase_6h_time()
    delay = (next_run_time - datetime.utcnow()).total_seconds()
    print(f"🔔 Next signal check at {next_run_time.isoformat()} UTC")
    schedule.clear()
    schedule.every(delay).seconds.do(run_and_reschedule)

# === STARTUP ===
check_signals()
schedule_6h_task()


print("🚀 Smart Money Alert System Running")
while True:
    schedule.run_pending()
    time.sleep(5)
