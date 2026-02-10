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

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# === CONFIG ===
BASE_URL = "https://api.exchange.coinbase.com"
COINS = ["btc-usd", "eth-usd", "xrp-usd"]
GRANULARITY_6H = 21600
GRANULARITY_1H = 3600
GRANULARITY_1D = 86400
TP_MULTIPLIER = 0.75
CONFIRMATION_TIMEOUT_HOURS = 6
open_confirmations = {}
active_trades = {}

# === UTILITIES ===
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
        try:
            res = requests.get(f"{BASE_URL}/products/{symbol}/candles", params=params)
            data = res.json()
            if isinstance(data, list) and data:
                chunk = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
                df = pd.concat([df, chunk])
        except:
            pass
        start = chunk_end
    if df.empty: return None
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.sort_values("time").reset_index(drop=True)

def get_live_price(symbol):
    try:
        r = requests.get(f"{BASE_URL}/products/{symbol}/ticker").json()
        return float(r["price"])
    except:
        return None

def send_telegram_alert(msg):
    try:
        bot.send_message(chat_id=CHAT_ID, text=msg)
    except:
        pass

def get_pst_time(ts):
    return ts.replace(tzinfo=timezone("UTC")).astimezone(timezone("US/Pacific")).strftime("%Y-%m-%d %I:%M %p %Z")

# === EMA & CLUSTER SCORING ===
def get_ema_cluster_score(entry, hourly_emas, daily_emas, side):
    score = 0
    above, below = [], []
    for tf, emas in [("1H", hourly_emas), ("1D", daily_emas)]:
        for k, v in emas.items():
            if v is None: continue
            pct = abs(entry-v)/entry
            if pct <= 0.025:
                if v > entry: above.append((f"{tf} EMA{k}", v))
                else: below.append((f"{tf} EMA{k}", v))
    if side=="long": score += len(below) - len(above)
    else: score += len(above) - len(below)
    msg = f"\n📊 Cluster Score: {score}"
    if above: msg += "\n⬆️ " + ", ".join([f"{l}:{v:.2f}" for l,v in above])
    if below: msg += "\n⬇️ " + ", ".join([f"{l}:{v:.2f}" for l,v in below])
    return msg

# === DIVERGENCE DETECTION ===
def detect_rsi_divergence(df, side="bullish", rsi_period=14, swing_lookback=5):
    """
    Detect RSI divergences:
    - bullish: price lower low, RSI higher low
    - hidden_bullish: price higher low, RSI lower low
    - bearish: price higher high, RSI lower high
    - hidden_bearish: price lower high, RSI higher high
    """
    if len(df) < swing_lookback * 2 + rsi_period + 1:
        return False

    df = df.copy()
    df["rsi"] = RSIIndicator(df["close"], window=rsi_period).rsi()

    def is_swing_low(prices):
        mid = swing_lookback
        if len(prices) < swing_lookback * 2 + 1:
            return False
        return prices[mid] == min(prices)

    def is_swing_high(prices):
        mid = swing_lookback
        if len(prices) < swing_lookback * 2 + 1:
            return False
        return prices[mid] == max(prices)

    df["swing_low"] = df["low"].rolling(window=swing_lookback * 2 + 1, center=True).apply(
        lambda x: is_swing_low(x), raw=True
    ).fillna(0)

    df["swing_high"] = df["high"].rolling(window=swing_lookback * 2 + 1, center=True).apply(
        lambda x: is_swing_high(x), raw=True
    ).fillna(0)

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


# === LOGGING ===
def log_trade(symbol, side, entry, tp, sl, signal_time, outcome=None, high=None, low=None):
    file_exists = os.path.isfile("live_trade_log.csv")
    with open("live_trade_log.csv","a",newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["symbol","side","entry","tp","sl","signal_time","outcome","high","low"])
        w.writerow([symbol,side,entry,tp,sl,signal_time,outcome,high,low])

# === EARLY EXIT ===
def monitor_trade_exit(symbol, side, tp, sl):
    high, low = -np.inf, np.inf
    while symbol in active_trades:
        price = get_live_price(symbol)
        if price is None: time.sleep(60); continue
        high, low = max(high, price), min(low, price)
        if (side=="long" and price>=tp) or (side=="short" and price<=tp):
            send_telegram_alert(f"🎯 TP Hit for {symbol.upper()} at {price}")
            log_trade(symbol,side,active_trades[symbol]["entry"],tp,sl,active_trades[symbol]["time"],"TP",high,low)
            del active_trades[symbol]
            break
        if (side=="long" and price<=sl) or (side=="short" and price>=sl):
            send_telegram_alert(f"⛔ SL Hit for {symbol.upper()} at {price}")
            log_trade(symbol,side,active_trades[symbol]["entry"],tp,sl,active_trades[symbol]["time"],"SL",high,low)
            del active_trades[symbol]
            break
        # Early exit on opposite divergence
        df = fetch_candles(symbol, GRANULARITY_1H, 5)
        if side=="long" and (detect_rsi_divergence(df,"bearish") or detect_rsi_divergence(df,"hidden_bearish")):
            send_telegram_alert(f"⚠️ Early Exit: Bearish Divergence on {symbol.upper()} at {price}")
            log_trade(symbol,side,active_trades[symbol]["entry"],tp,sl,active_trades[symbol]["time"],"EarlyExit",high,low)
            del active_trades[symbol]
            break
        if side=="short" and (detect_rsi_divergence(df,"bullish") or detect_rsi_divergence(df,"hidden_bullish")):
            send_telegram_alert(f"⚠️ Early Exit: Bullish Divergence on {symbol.upper()} at {price}")
            log_trade(symbol,side,active_trades[symbol]["entry"],tp,sl,active_trades[symbol]["time"],"EarlyExit",high,low)
            del active_trades[symbol]
            break
        time.sleep(300)

# === CONFIRMATION MONITOR ===
def monitor_divergence(symbol, side, entry, tp, sl, signal_time):
    timeout = datetime.utcnow()+timedelta(hours=CONFIRMATION_TIMEOUT_HOURS)
    while datetime.utcnow()<timeout:
        df = fetch_candles(symbol, GRANULARITY_1H, 12)
        if side=="long" and (detect_rsi_divergence(df,"bullish") or detect_rsi_divergence(df,"hidden_bullish")):
            price = get_live_price(symbol)
            hourly_emas = {p:df["close"].ewm(span=p).mean().iloc[-1] for p in [10,50,200]}
            daily_df = fetch_candles(symbol, GRANULARITY_1D, 300)
            daily_emas = {p:daily_df["close"].ewm(span=p).mean().iloc[-1] for p in [10,50,200]}
            msg = f"✅ DIV LONG Confirmed {symbol.upper()} at {price:.2f}\nTP:{tp:.2f} SL:{sl:.2f}\n{signal_time}"
            msg += get_ema_cluster_score(price,hourly_emas,daily_emas,"long")
            send_telegram_alert(msg)
            active_trades[symbol] = {"entry":price,"time":signal_time}
            threading.Thread(target=monitor_trade_exit,args=(symbol,"long",tp,sl)).start()
            break
        elif side=="short" and (detect_rsi_divergence(df,"bearish") or detect_rsi_divergence(df,"hidden_bearish")):
            price = get_live_price(symbol)
            hourly_emas = {p:df["close"].ewm(span=p).mean().iloc[-1] for p in [10,50,200]}
            daily_df = fetch_candles(symbol, GRANULARITY_1D, 300)
            daily_emas = {p:daily_df["close"].ewm(span=p).mean().iloc[-1] for p in [10,50,200]}
            msg = f"✅ DIV SHORT Confirmed {symbol.upper()} at {price:.2f}\nTP:{tp:.2f} SL:{sl:.2f}\n{signal_time}"
            msg += get_ema_cluster_score(price,hourly_emas,daily_emas,"short")
            send_telegram_alert(msg)
            active_trades[symbol] = {"entry":price,"time":signal_time}
            threading.Thread(target=monitor_trade_exit,args=(symbol,"short",tp,sl)).start()
            break
        time.sleep(300)

# === MAIN SIGNAL CHECK ===
def check_signals():
    for sym in COINS:
        df = fetch_candles(sym, GRANULARITY_6H)
        if df is None or len(df)<30: continue
        df["adx"] = ADXIndicator(df["high"],df["low"],df["close"]).adx()
        df["plus_di"] = ADXIndicator(df["high"],df["low"],df["close"]).adx_pos()
        df["minus_di"] = ADXIndicator(df["high"],df["low"],df["close"]).adx_neg()
        df["macd_diff"] = MACD(df["close"]).macd_diff()
        df["rsi"] = RSIIndicator(df["close"]).rsi()
        last = df.iloc[-1]
        entry = last["close"]
        sig_time = get_pst_time(last["time"])
        # Long setup
        if last["adx"]>23 and last["plus_di"]>last["minus_di"] and last["macd_diff"]>0 and last["rsi"]>55:
            sl = min(df["low"].iloc[-5:])
            tp = entry + TP_MULTIPLIER*(entry-sl)
            threading.Thread(target=monitor_divergence,args=(sym,"long",entry,tp,sl,sig_time)).start()
        # Short setup
        elif last["adx"]>23 and last["minus_di"]>last["plus_di"] and last["macd_diff"]<0 and last["rsi"]<45:
            sl = max(df["high"].iloc[-5:])
            tp = entry - TP_MULTIPLIER*(sl-entry)
            threading.Thread(target=monitor_divergence,args=(sym,"short",entry,tp,sl,sig_time)).start()


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

