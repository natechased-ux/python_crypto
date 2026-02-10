import pandas as pd
import numpy as np
import requests
import time
from pytz import timezone
from datetime import datetime, timedelta
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
CHAT_ID = "-4916911067"
bot = telegram.Bot(token=TELEGRAM_TOKEN)

#channel chat id:
#"-4916911067"

#personal for testing
#CHAT_ID = "7967738614"

# === CONFIGURATION ===
COINS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
         "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
         "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd", "avax-usd", "xcn-usd", "uni-usd",
         "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
         "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd", "qnt-usd", "tia-usd", "ip-usd",
         "pnut-usd", "apt-usd", "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "pyth-usd", "sand-usd",
         "morpho-usd", "mana-usd", "coti-usd", "c98-usd", "axs-usd"]
BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY_6H = 21600
GRANULARITY_1H = 3600
GRANULARITY_1D = 86400
TP_MULTIPLIER = 0.75
CONFIRMATION_TIMEOUT_HOURS = 6

# === HELPERS ===
def fetch_candles(symbol, granularity, lookback_days=30):
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    df = pd.DataFrame()
    while start < end:
        chunk_end = min(start + timedelta(hours=300), end)
        params = {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "granularity": granularity}
        try:
            r = requests.get(f"{BASE_URL}/products/{symbol}/candles", params=params, timeout=10)
            data = r.json()
            if isinstance(data, list) and data:
                chunk = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
                df = pd.concat([df, chunk], ignore_index=True)
        except Exception as e:
            print(f"⚠️ Error fetching {symbol}: {e}")
        start = chunk_end
        time.sleep(0.25)
    if df.empty: return None
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def fmt_price(p):
    if p >= 100: return f"{p:.2f}"
    elif p >= 1: return f"{p:.4f}"
    elif p >= 0.01: return f"{p:.6f}"
    else: return f"{p:.8f}"

def get_live_price(symbol):
    try:
        r = requests.get(f"{BASE_URL}/products/{symbol}/ticker", timeout=10)
        return float(r.json()["price"])
    except:
        return None

def get_pst_time(utc_dt):
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone("UTC"))
    return utc_dt.astimezone(timezone("US/Pacific")).strftime("%Y-%m-%d %I:%M %p %Z")

def send_telegram_alert(msg):
    try:
        bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
    except Exception as e:
        print(f"⚠️ Telegram error: {e}")

# === EMA CONTEXT ===
def get_ema_context(entry_price, hourly_emas, daily_emas):
    lines = ["📊 EMA Guidance:"]
    def add_line(label, val):
        lines.append(f"• {label}: {fmt_price(val)}")
    lines.append("⬆️ Daily EMAs Above:")
    for p in [200, 100, 50, 20, 10]:
        v = daily_emas.get(f"ema{p}")
        if v and v > entry_price: add_line(f"1D EMA{p}", v)
    lines.append("⬇️ Daily EMAs Below:")
    for p in [200, 100, 50, 20, 10]:
        v = daily_emas.get(f"ema{p}")
        if v and v < entry_price: add_line(f"1D EMA{p}", v)
    lines.append("⬆️ Hourly EMAs Above:")
    for p in [200, 50]:
        v = hourly_emas.get(f"ema{p}")
        if v and v > entry_price: add_line(f"1H EMA{p}", v)
    lines.append("⬇️ Hourly EMAs Below:")
    for p in [200, 50]:
        v = hourly_emas.get(f"ema{p}")
        if v and v < entry_price: add_line(f"1H EMA{p}", v)
    return "\n".join(lines)

# === RSI DIVERGENCE ===
def detect_rsi_divergence(df, side="bullish", rsi_period=14, swing_lookback=5,
                          min_rsi_diff=3, min_price_diff_pct=0.25):
    if df is None or len(df) < swing_lookback*2 + rsi_period + 1:
        return False
    df = df.copy()
    df["rsi"] = RSIIndicator(df["close"], window=rsi_period).rsi()
    df["swing_low"] = df["low"].rolling(window=swing_lookback*2+1, center=True).apply(
        lambda x: x.iloc[swing_lookback] == min(x), raw=False).fillna(0)
    df["swing_high"] = df["high"].rolling(window=swing_lookback*2+1, center=True).apply(
        lambda x: x.iloc[swing_lookback] == max(x), raw=False).fillna(0)
    lows = df[df["swing_low"] == 1]
    highs = df[df["swing_high"] == 1]
    if side in ["bullish", "hidden_bullish"] and len(lows) >= 2:
        p1, p2 = lows["low"].iloc[-2], lows["low"].iloc[-1]
        r1, r2 = lows["rsi"].iloc[-2], lows["rsi"].iloc[-1]
        if abs(r2 - r1) < min_rsi_diff: return False
        if abs((p2 - p1) / p1 * 100) < min_price_diff_pct: return False
        if side == "bullish" and p2 < p1 and r2 > r1: return True
        if side == "hidden_bullish" and p2 > p1 and r2 < r1: return True
    if side in ["bearish", "hidden_bearish"] and len(highs) >= 2:
        p1, p2 = highs["high"].iloc[-2], highs["high"].iloc[-1]
        r1, r2 = highs["rsi"].iloc[-2], highs["rsi"].iloc[-1]
        if abs(r2 - r1) < min_rsi_diff: return False
        if abs((p2 - p1) / p1 * 100) < min_price_diff_pct: return False
        if side == "bearish" and p2 > p1 and r2 < r1: return True
        if side == "hidden_bearish" and p2 < p1 and r2 > r1: return True
    return False

# === LOGGING ===
def log_trade(symbol, side, entry, tp, sl, signal_time, outcome=None, high=None, low=None, indicators=None, ema_vals=None):
    file = "bigpapi_trade_log.csv"
    exists = os.path.isfile(file)
    with open(file, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["symbol","side","entry","tp","sl","signal_time",
                        "adx","macd_diff","rsi",
                        "ema10_1d","ema20_1d","ema50_1d","ema100_1d","ema200_1d",
                        "ema50_1h","ema200_1h",
                        "outcome","high","low","weekend"])
        w.writerow([
            symbol, side, fmt_price(entry), fmt_price(tp), fmt_price(sl), signal_time,
            indicators.get("adx") if indicators else None,
            indicators.get("macd_diff") if indicators else None,
            indicators.get("rsi") if indicators else None,
            ema_vals.get("ema10_1d") if ema_vals else None,
            ema_vals.get("ema20_1d") if ema_vals else None,
            ema_vals.get("ema50_1d") if ema_vals else None,
            ema_vals.get("ema100_1d") if ema_vals else None,
            ema_vals.get("ema200_1d") if ema_vals else None,
            ema_vals.get("ema50_1h") if ema_vals else None,
            ema_vals.get("ema200_1h") if ema_vals else None,
            outcome, fmt_price(high) if high else None, fmt_price(low) if low else None,
            datetime.utcnow().weekday() >= 5
        ])

# === MONITOR CONFIRMATION ===
def monitor_confirmation(symbol, side, entry, tp, sl, signal_time, last_row):
    timeout = datetime.utcnow() + timedelta(hours=CONFIRMATION_TIMEOUT_HOURS)
    print(f"📈 Monitoring confirmation for {symbol.upper()}...")
    while datetime.utcnow() < timeout:
        df_1h = fetch_candles(symbol, GRANULARITY_1H, 10)
        df_1d = fetch_candles(symbol, GRANULARITY_1D, 200)
        if df_1h is None or df_1d is None:
            time.sleep(300)
            continue
        for p in [10,20,50,100,200]:
            df_1h[f"ema{p}_1h"] = df_1h["close"].ewm(span=p).mean()
            df_1d[f"ema{p}_1d"] = df_1d["close"].ewm(span=p).mean()
        hourly_emas = {f"ema{p}": df_1h[f"ema{p}_1h"].iloc[-1] for p in [10,20,50,100,200]}
        daily_emas = {f"ema{p}": df_1d[f"ema{p}_1d"].iloc[-1] for p in [10,20,50,100,200]}
        ema_msg = get_ema_context(entry, hourly_emas, daily_emas)
        if side=="long" and (detect_rsi_divergence(df_1h,"bullish") or detect_rsi_divergence(df_1h,"hidden_bullish")):
            confirmed = get_live_price(symbol)
            send_telegram_alert(f"✅ BigPApi*LONG Confirmed* on {symbol.upper()} @ {fmt_price(confirmed)}\nTP: {fmt_price(tp)} SL: {fmt_price(sl)}\n{signal_time}\n\n{ema_msg}")
            indicators = {"adx":last_row["adx"],"macd_diff":last_row["macd_diff"],"rsi":last_row["rsi"]}
            ema_vals = {f"ema{p}_1d":daily_emas[f"ema{p}"] for p in [10,20,50,100,200]}
            ema_vals.update({f"ema{p}_1h":hourly_emas[f"ema{p}"] for p in [50,200]})
            log_trade(symbol, side, confirmed, tp, sl, signal_time, None, None, None, indicators, ema_vals)
            break
        elif side=="short" and (detect_rsi_divergence(df_1h,"bearish") or detect_rsi_divergence(df_1h,"hidden_bearish")):
            confirmed = get_live_price(symbol)
            send_telegram_alert(f"✅ BigPApi*SHORT Confirmed* on {symbol.upper()} @ {fmt_price(confirmed)}\nTP: {fmt_price(tp)} SL: {fmt_price(sl)}\n{signal_time}\n\n{ema_msg}")
            indicators = {"adx":last_row["adx"],"macd_diff":last_row["macd_diff"],"rsi":last_row["rsi"]}
            ema_vals = {f"ema{p}_1d":daily_emas[f"ema{p}"] for p in [10,20,50,100,200]}
            ema_vals.update({f"ema{p}_1h":hourly_emas[f"ema{p}"] for p in [50,200]})
            log_trade(symbol, side, confirmed, tp, sl, signal_time, None, None, None, indicators, ema_vals)
            break
        time.sleep(300)

# === SIGNAL CHECK ===
def check_signals():
    for sym in COINS:
        df = fetch_candles(sym, GRANULARITY_6H)
        if df is None or len(df) < 30: continue
        df["rsi"] = RSIIndicator(df["close"]).rsi()
        adx = ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx.adx()
        df["plus_di"] = adx.adx_pos()
        df["minus_di"] = adx.adx_neg()
        macd = MACD(df["close"])
        df["macd_diff"] = macd.macd_diff()
        last = df.iloc[-1]
        entry = last["close"]
        signal_time = get_pst_time(last["time"])
        if last["adx"] > 20 and last["plus_di"] > last["minus_di"] and last["macd_diff"] > 0 and last["rsi"] > 55:
            swings = df["low"].iloc[-5:]
            sl = swings.min()
            risk = entry - sl
            tp = entry + TP_MULTIPLIER * risk
            threading.Thread(target=monitor_confirmation, args=(sym,"long",entry,tp,sl,signal_time,last)).start()
        elif last["adx"] > 20 and last["minus_di"] > last["plus_di"] and last["macd_diff"] < 0 and last["rsi"] < 45:
            swings = df["high"].iloc[-5:]
            sl = swings.max()
            risk = sl - entry
            tp = entry - TP_MULTIPLIER * risk
            threading.Thread(target=monitor_confirmation, args=(sym,"short",entry,tp,sl,signal_time,last)).start()

# === SCHEDULER ===
def get_next_6h():
    now = datetime.utcnow().replace(minute=0,second=0,microsecond=0)
    for h in [0,6,12,18]:
        if now.hour < h:
            return now.replace(hour=h) + timedelta(minutes=5)
    return (now+timedelta(days=1)).replace(hour=0) + timedelta(minutes=5)

def schedule_6h():
    def run():
        check_signals()
        schedule_6h()
    nxt = get_next_6h()
    delay = (nxt - datetime.utcnow()).total_seconds()
    schedule.clear()
    schedule.every(delay).seconds.do(run)
    print(f"📅 Next run {nxt}")

# === START ===
if __name__ == "__main__":
    print("🚀 BigPapi running with RSI divergence + EMA guidance")
    check_signals()
    schedule_6h()
    while True:
        schedule.run_pending()
        time.sleep(5)
