import pandas as pd
import numpy as np
import requests
import time
import pytz
from pytz import timezone
from datetime import datetime, timedelta
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator, StochRSIIndicator
import schedule
import threading
import pytz
import telegram

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# === CONFIGURATION ===
COINS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
         "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
         "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd", "avax-usd", "xcn-usd", "uni-usd",
         "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
         "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd", "qnt-usd", "tia-usd", "ip-usd",
         "pnut-usd", "apt-usd", "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "pyth-usd", "sand-usd",
         "morpho-usd", "mana-usd", "coti-usd", "c98-usd", "axs-usd"]

BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY_1H = 3600  # 6-hour candles
GRANULARITY_15m = 900   # 1-hour candles
TP_MULTIPLIER = 0.5
SL_BUFFER = 0.25
LOOKBACK_DAYS = 3
TRADE_AMOUNT = 1000
CONFIRMATION_TIMEOUT_HOURS = 1

open_confirmations = {}  # Track coins awaiting Stoch RSI confirmation

def fetch_candles(symbol, granularity, lookback_days=LOOKBACK_DAYS):
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    df = pd.DataFrame()
    while start < end:
        chunk_end = min(start + timedelta(hours=250), end)
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
                df = pd.concat([df, chunk])
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        start = chunk_end
    if df.empty:
        return None
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_ema_targets(df, current_price):
    ema_values = {
        "EMA_10": df["close"].ewm(span=10).mean().iloc[-1],
        "EMA_20": df["close"].ewm(span=20).mean().iloc[-1],
        "EMA_50": df["close"].ewm(span=50).mean().iloc[-1],
        "EMA_200": df["close"].ewm(span=200).mean().iloc[-1],
    }

    # Find closest EMA above and below price
    above = {k: v for k, v in ema_values.items() if v > current_price}
    below = {k: v for k, v in ema_values.items() if v < current_price}

    closest_above = min(above.items(), key=lambda x: x[1], default=(None, None))
    closest_below = max(below.items(), key=lambda x: x[1], default=(None, None))

    return {
        "tp_ema_label": closest_above[0],
        "tp_ema_value": closest_above[1],
        "sl_ema_label": closest_below[0],
        "sl_ema_value": closest_below[1]
    }


def calculate_indicators(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    adx = ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()
    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()
    
    # Add EMA calculations (1H)
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
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

def send_telegram_alert(message):
    bot.send_message(chat_id=CHAT_ID, text=message)

def get_pst_time(utc_timestamp):
    utc_dt = utc_timestamp if utc_timestamp.tzinfo else utc_timestamp.replace(tzinfo=timezone('UTC'))
    pst_dt = utc_dt.astimezone(timezone('US/Pacific'))
    return pst_dt.strftime("%Y-%m-%d %I:%M %p %Z")

def check_signals():
    print(f"🔍 Checking signals at {datetime.utcnow().isoformat()} UTC")
    for symbol in COINS:
        df = fetch_candles(symbol, GRANULARITY_1H)
        if df is None or len(df) < 30:
            continue
        df = calculate_indicators(df)
        swings_high = [(i, df["high"].iloc[i]) for i in range(len(df)) if find_swing_high(df, i)]
        swings_low = [(i, df["low"].iloc[i]) for i in range(len(df)) if find_swing_low(df, i)]
        last_row = df.iloc[-1]  # last completed 6h candle
        entry_price = last_row["close"]
                # EMA guidance for TP/SL
        emas = {
            "EMA 10": last_row.get("ema_10"),
            "EMA 20": last_row.get("ema_20"),
            "EMA 50": last_row.get("ema_50"),
            "EMA 200": last_row.get("ema_200")
        }
        above = {k: v for k, v in emas.items() if v and v > entry_price}
        below = {k: v for k, v in emas.items() if v and v < entry_price}
        closest_above = min(above.items(), key=lambda x: x[1], default=(None, None))
        closest_below = max(below.items(), key=lambda x: x[1], default=(None, None))
        ema_msg = ""
        
        if closest_above[1]:
            ema_msg += (f" \n EMA Above: {closest_above[0]} @ {closest_above[1]:.4f}")
        
        if closest_below[1]:
            ema_msg += (f" \n EMA Below: {closest_below[0]} @ {closest_below[1]:.4f}")
        

        
        signal_time = get_pst_time(last_row["time"])




        if np.isnan(last_row["adx"]) or np.isnan(last_row["rsi"]) or np.isnan(last_row["macd_diff"]):
            continue

        # === LONG SETUP ===
        if last_row["adx"] > 25 and last_row["plus_di"] > last_row["minus_di"]+5 and last_row["macd_diff"] > 0.1 and last_row["rsi"] > 58:
            swings = [p for j, p in swings_low if j < len(df) - 2]
            if swings:
                sl = swings[-1]
                risk = entry_price - sl
                tp = entry_price + TP_MULTIPLIER * risk
                open_confirmations[symbol] = {
                    "side": "long", "entry": entry_price, "tp": tp, "sl": sl,
                    "timestamp": datetime.utcnow(), "signal_time": signal_time
                }
                #send_telegram_alert(f"📈 *Potential LONG* on {symbol.upper()}\nEntry: {entry_price:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\nSignal Time (PST): {signal_time}")
                threading.Thread(target=monitor_stoch_confirmation, args=(symbol, "long", entry_price, tp, sl, signal_time)).start()
        
        # === SHORT SETUP ===
        elif last_row["adx"] > 25 and last_row["minus_di"] > last_row["plus_di"]+5 and last_row["macd_diff"] < -0.1 and last_row["rsi"] < 42:
            swings = [p for j, p in swings_high if j < len(df) - 2]
            if swings:
                sl = swings[-1]
                risk = sl - entry_price
                tp = entry_price - TP_MULTIPLIER * risk

                open_confirmations[symbol] = {
                    "side": "short", "entry": entry_price, "tp": tp, "sl": sl,
                    "timestamp": datetime.utcnow(), "signal_time": signal_time
                }
                #send_telegram_alert(f"📉 *Potential SHORT* on {symbol.upper()}\nEntry: {entry_price:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\nSignal Time (PST): {signal_time}")
                threading.Thread(target=monitor_stoch_confirmation, args=(symbol, "short", entry_price, tp, sl, ema_msg,  signal_time)).start()

def monitor_stoch_confirmation(symbol, side, entry, tp, sl, ema_msg, signal_time):
    timeout = datetime.utcnow() + timedelta(hours=CONFIRMATION_TIMEOUT_HOURS)
    print(f"🕒 Monitoring {symbol.upper()} for Stoch RSI entry confirmation...")

    while datetime.utcnow() < timeout:
        print(f"🕒")
        df = fetch_candles(symbol, GRANULARITY_15m)
        if df is None or len(df) < 5:
            print(f"trouble")
            time.sleep(300)
            continue

        stoch = StochRSIIndicator(df["close"]).stochrsi_k().dropna() * 100
        stoch_d = StochRSIIndicator(df["close"]).stochrsi_d().dropna() * 100
        

        if stoch.isna().all() or stoch_d.isna().all():
            time.sleep(300)
            continue

        k, d = stoch.iloc[-1], stoch_d.iloc[-1]
        prev_k = stoch.iloc[-2] if len(stoch) > 1 else k
        print(f"🕒 {symbol.upper()} k {k:.4f}")
        print(f"🕒 {symbol.upper()} d {d:.4f}")
        print(f"🕒 {symbol.upper()} kpre {prev_k:.4f}")

        
        
        if side == "long"  and k > d and k < 20 and k > prev_k:
           

            send_telegram_alert( f"✅ *LONG Confirmed* on {symbol.upper()}\nEntry: {entry:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\n{signal_time}{ema_msg}")



            if symbol in open_confirmations:
                del open_confirmations[symbol]
            break

        elif side == "short" and k < d and k > 80 and k < prev_k:
            send_telegram_alert(f"✅ *SHORT Confirmed* on {symbol.upper()}\nEntry: {entry:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\n{signal_time}{ema_msg}")


            if symbol in open_confirmations:
                del open_confirmations[symbol]
            break

        time.sleep(300)  # Retry every 5 minutes

def get_next_coinbase_1h_time():
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    # Coinbase 1H candle times: 00:00, 06:00, 12:00, 18:00 UTC
    next_times = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    for t in next_times:
        candidate = now.replace(hour=t)
        if candidate > now:
            return candidate + timedelta(minutes=5)  # buffer
    # If none ahead today, go to the first of the next day
    return (now + timedelta(days=1)).replace(hour=0) + timedelta(minutes=5)


def schedule_1h_task():
    def run_and_reschedule():
        print(f"🕒 Running check at {datetime.utcnow().isoformat()}")
        check_signals()
        schedule_1h_task()

    next_run_time = get_next_coinbase_1h_time()
    delay = (next_run_time - datetime.utcnow()).total_seconds()
    print(f"🔔 Next signal check at {next_run_time.isoformat()} UTC")
    schedule.clear()
    schedule.every(delay).seconds.do(run_and_reschedule)

# === STARTUP ===
check_signals()
schedule_1h_task()


print("🚀 Smart Money Alert System Running")
while True:
    schedule.run_pending()
    time.sleep(5)
