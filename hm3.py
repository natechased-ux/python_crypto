# === FULL MASTER SCRIPT ===
# Smart Money Alert System with RSI Divergence Confirmation + Early Exit

import pandas as pd
import numpy as np
import requests
import time
import os
import csv
import threading
import schedule
import telegram
from datetime import datetime, timedelta
from pytz import timezone
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# === CONFIG ===
# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = telegram.Bot(token=TELEGRAM_TOKEN)

COINS = ["btc-usd", "eth-usd", "xrp-usd"]
BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY_6H = 21600
GRANULARITY_1H = 3600
GRANULARITY_1D = 86400
TP_MULTIPLIER = 0.75
CONFIRMATION_TIMEOUT_HOURS = 6

open_trades = {}
lock = threading.Lock()

# === UTILS ===
def fetch_candles(symbol, granularity, lookback_days=30):
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    url = f"{BASE_URL}/products/{symbol}/candles"
    df = pd.DataFrame()
    while start < end:
        chunk_end = min(start + timedelta(hours=300), end)
        params = {
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "granularity": granularity
        }
        r = requests.get(url, params=params)
        try:
            data = r.json()
            if isinstance(data, list):
                chunk = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
                df = pd.concat([df, chunk])
        except:
            pass
        start = chunk_end
        time.sleep(0.25)
    if df.empty: return None
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.sort_values("time", inplace=True)
    return df

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
    
def calculate_indicators(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    adx = ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()
    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()
    for p in [10, 20, 50, 200]:
        df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    return df

def detect_rsi_divergence(df, side="bullish", rsi_period=14, swing_lookback=5):
    if len(df) < swing_lookback * 2 + rsi_period: return False
    df = df.copy()
    df["rsi"] = RSIIndicator(df["close"], window=rsi_period).rsi()
    lows = df["low"].rolling(swing_lookback*2+1, center=True).apply(
        lambda x: x[swing_lookback]==min(x), raw=True).fillna(0)
    highs = df["high"].rolling(swing_lookback*2+1, center=True).apply(
        lambda x: x[swing_lookback]==max(x), raw=True).fillna(0)
    l_idx = lows[lows==1].index
    h_idx = highs[highs==1].index
    if side in ["bullish","hidden_bullish"] and len(l_idx)>=2:
        p1,p2 = df.loc[l_idx[-2],"low"], df.loc[l_idx[-1],"low"]
        r1,r2 = df.loc[l_idx[-2],"rsi"], df.loc[l_idx[-1],"rsi"]
        if side=="bullish" and p2<p1 and r2>r1: return True
        if side=="hidden_bullish" and p2>p1 and r2<r1: return True
    if side in ["bearish","hidden_bearish"] and len(h_idx)>=2:
        p1,p2 = df.loc[h_idx[-2],"high"], df.loc[h_idx[-1],"high"]
        r1,r2 = df.loc[h_idx[-2],"rsi"], df.loc[h_idx[-1],"rsi"]
        if side=="bearish" and p2>p1 and r2<r1: return True
        if side=="hidden_bearish" and p2<p1 and r2>r1: return True
    return False

def get_live_price(symbol):
    try:
        r = requests.get(f"{BASE_URL}/products/{symbol}/ticker")
        return float(r.json()["price"])
    except: return None

def format_price(price):
    return f"{price:.8f}" if price < 0.01 else (f"{price:.4f}" if price < 1 else f"{price:.2f}")

def log_trade(symbol, side, entry, tp, sl, signal_time, ema_daily, ema_hourly, outcome=None, weekend=False):
    file_exists = os.path.isfile("trade_log.csv")
    with open("trade_log.csv","a",newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["symbol","side","entry","tp","sl","signal_time","ema10d","ema20d","ema50d","ema200d","ema50h","ema200h","outcome","weekend"])
        w.writerow([symbol,side,entry,tp,sl,signal_time,
                    ema_daily["ema10"],ema_daily["ema20"],ema_daily["ema50"],ema_daily["ema200"],
                    ema_hourly["ema50"],ema_hourly["ema200"],outcome,weekend])

def send_alert(symbol, side, entry, tp, sl, ema_daily, ema_hourly, extra=""):
    msg = f"✅ {side.upper()} Confirmed {symbol.upper()}\n"
    msg += f"Entry: {format_price(entry)}\nTP: {format_price(tp)}\nSL: {format_price(sl)}\n"
    msg += "\n📊 Daily EMAs:\n"
    for k in [10,20,50,200]:
        msg += f"• EMA{k}: {format_price(ema_daily[f'ema{k}'])}\n"
    msg += "\n⏱ Hourly EMAs:\n"
    for k in [50,200]:
        msg += f"• EMA{k}: {format_price(ema_hourly[f'ema{k}'])}\n"
    if extra: msg += f"\n{extra}"
    bot.send_message(chat_id=CHAT_ID, text=msg)

# === MONITOR TRADE ===
def monitor_trade(symbol, side, tp, sl):
    while True:
        price = get_live_price(symbol)
        if price is None:
            time.sleep(60)
            continue
        if side=="long":
            if price >= tp: 
                bot.send_message(chat_id=CHAT_ID, text=f"🎯 TP Hit {symbol.upper()}")
                return
            if price <= sl:
                bot.send_message(chat_id=CHAT_ID, text=f"⛔ SL Hit {symbol.upper()}")
                return
            if detect_rsi_divergence(fetch_candles(symbol,GRANULARITY_1H),"bearish"):
                bot.send_message(chat_id=CHAT_ID, text=f"🚪 Early Exit {symbol.upper()} — Bearish Divergence")
                return
        else:
            if price <= tp:
                bot.send_message(chat_id=CHAT_ID, text=f"🎯 TP Hit {symbol.upper()}")
                return
            if price >= sl:
                bot.send_message(chat_id=CHAT_ID, text=f"⛔ SL Hit {symbol.upper()}")
                return
            if detect_rsi_divergence(fetch_candles(symbol,GRANULARITY_1H),"bullish"):
                bot.send_message(chat_id=CHAT_ID, text=f"🚪 Early Exit {symbol.upper()} — Bullish Divergence")
                return
        time.sleep(300)

# === CONFIRMATION ===
def monitor_confirmation(symbol, side, entry, tp, sl, signal_time):
    timeout = datetime.utcnow() + timedelta(hours=CONFIRMATION_TIMEOUT_HOURS)
    while datetime.utcnow() < timeout:
        df_1h = fetch_candles(symbol, GRANULARITY_1H, lookback_days=12)
        if df_1h is None: 
            time.sleep(300); continue
        if side=="long" and (detect_rsi_divergence(df_1h,"bullish") or detect_rsi_divergence(df_1h,"hidden_bullish")):
            ema_daily = {f"ema{k}": fetch_candles(symbol,GRANULARITY_1D)[f"close"].ewm(span=k).mean().iloc[-1] for k in [10,20,50,200]}
            ema_hourly = {f"ema{k}": df_1h[f"close"].ewm(span=k).mean().iloc[-1] for k in [50,200]}
            send_alert(symbol, side, entry, tp, sl, ema_daily, ema_hourly)
            threading.Thread(target=monitor_trade, args=(symbol, side, tp, sl)).start()
            return
        elif side=="short" and (detect_rsi_divergence(df_1h,"bearish") or detect_rsi_divergence(df_1h,"hidden_bearish")):
            ema_daily = {f"ema{k}": fetch_candles(symbol,GRANULARITY_1D)[f"close"].ewm(span=k).mean().iloc[-1] for k in [10,20,50,200]}
            ema_hourly = {f"ema{k}": df_1h[f"close"].ewm(span=k).mean().iloc[-1] for k in [50,200]}
            send_alert(symbol, side, entry, tp, sl, ema_daily, ema_hourly)
            threading.Thread(target=monitor_trade, args=(symbol, side, tp, sl)).start()
            return
        time.sleep(300)

# === SIGNAL CHECK ===
def check_signals():
    for symbol in COINS:
        df = fetch_candles(symbol, GRANULARITY_6H)
        if df is None or len(df) < 30: continue
        df = calculate_indicators(df)
        last = df.iloc[-1]
        entry = last["close"]
        if last["adx"]>23 and last["plus_di"]>last["minus_di"] and last["macd_diff"]>0 and last["rsi"]>55:
            sl = df["low"].iloc[-3]
            tp = entry + TP_MULTIPLIER*(entry-sl)
            threading.Thread(target=monitor_confirmation, args=(symbol,"long",entry,tp,sl,datetime.utcnow())).start()
        elif last["adx"]>23 and last["minus_di"]>last["plus_di"] and last["macd_diff"]<0 and last["rsi"]<45:
            sl = df["high"].iloc[-3]
            tp = entry - TP_MULTIPLIER*(sl-entry)
            threading.Thread(target=monitor_confirmation, args=(symbol,"short",entry,tp,sl,datetime.utcnow())).start()

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

