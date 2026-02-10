import pandas as pd
import numpy as np
import requests
import time
import pytz
from pytz import timezone
from datetime import datetime, timedelta
from ta.trend import ADXIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import schedule
import threading
import pytz
import telegram
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
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
GRANULARITY_15m = 900  # 6-hour candles
GRANULARITY_6H = 21600  # 6-hour candles
GRANULARITY_1H = 3600   # 1-hour candles
GRANULARITY_1D = 86400  # 1-day candles
TP_MULTIPLIER = 0.75
LOOKBACK_DAYS = 30
TRADE_AMOUNT = 1000
CONFIRMATION_TIMEOUT_HOURS = 1
CLUSTER_SPACING_THRESHOLD = 0.01  # 1%
EMAS_CLOSE_TO_PRICE_PCT = 0.025   # 2.5%

open_confirmations = {}  # Track coins awaiting Stoch RSI confirmation

def fetch_candles(symbol, granularity, lookback_days=LOOKBACK_DAYS):
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

    # 🧹 Clean up
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.drop_duplicates(subset="time", inplace=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def calculate_indicators(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    adx = ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()
    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()
    
    # ✅ Add EMA calculations
    for period in [10, 50, 100, 200]:
        df[f"ema{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    return df

def check_ema_stacking(ema_values, direction):
    """
    Check if the EMAs are correctly stacked in the trend direction.
    :param ema_values: List of EMA values in priority order
    :param direction: 'long' or 'short'
    :return: True if stacked properly
    """
    if None in ema_values or any(pd.isna(v) for v in ema_values):
        return False
    if direction == "long":
        return all(ema_values[i] > ema_values[i+1] for i in range(len(ema_values)-1))
    else:
        return all(ema_values[i] < ema_values[i+1] for i in range(len(ema_values)-1))


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
    for attempt in range(3):
        try:
            bot.send_message(chat_id=CHAT_ID, text=message)
            return
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            time.sleep(2)

def get_live_price(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/ticker"
    try:
        response = requests.get(url)
        data = response.json()
        return float(data["price"])
    except Exception as e:
        print(f"Error fetching live price for {symbol}: {e}")
        return None

def get_ema_cluster_score(entry_price, hourly_emas, daily_emas, side):
    def is_close(ema_val):
        return ema_val is not None and abs(entry_price - ema_val) / entry_price <= 0.025

    def stacking_score(ema_dict, tf_label):
        score = 0
        emas = [ema_dict.get(f"ema{p}") for p in [10, 20, 50, 200]]
        if all(v is not None for v in emas):
            if side == "long" and emas[0] > emas[1] > emas[2] > emas[3]:
                score += 2
            elif side == "short" and emas[0] < emas[1] < emas[2] < emas[3]:
                score += 2
        return score

    score = 0
    lines = ["\n"]

    # Clustered EMAs within 2.5%
    close_emas_above = []
    close_emas_below = []

    for label, value in hourly_emas.items():
        if is_close(value):
            if value > entry_price:
                close_emas_above.append((label, value))
            else:
                close_emas_below.append((label, value))

    for label, value in daily_emas.items():
        if is_close(value):
            if value > entry_price:
                close_emas_above.append((label, value))
            else:
                close_emas_below.append((label, value))

    if side == "long":
        score += len(close_emas_below)
        score -= len(close_emas_above)
    else:
        score += len(close_emas_above)
        score -= len(close_emas_below)

    # Add stacking alignment score
    score += stacking_score(daily_emas, "1D")
    score += stacking_score(hourly_emas, "1H")
        

    lines.append(f"📊 \n*Score*: {score:+d}")
    if stacking_score(daily_emas, "1D")>0:
        lines.append(f"📊Stacked")

    if close_emas_below:
        lines.append("⬇️")
        for label, value in sorted(close_emas_below, key=lambda x: x[1], reverse=True):
            lines.append(f"• {label.upper()}: {value:.4f}")

    if close_emas_above:
        lines.append("⬆️")
        for label, value in sorted(close_emas_above, key=lambda x: x[1]):
            lines.append(f"• {label.upper()}: {value:.4f}")

    return "\n".join(lines)


def get_live_price(symbol, retries=3, delay=10):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/ticker"
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            price = float(data.get("price", None))
            if price is None:
                print(f"⚠️ Attempt {attempt + 1}: No 'price' field in ticker data for {symbol}: {data}")
                time.sleep(delay)
                continue
            return price
        except Exception as e:
            print(f"❌ Attempt {attempt + 1}: Error fetching live price for {symbol}: {e}")
            time.sleep(delay)
    print(f"⛔ Failed to get live price for {symbol} after {retries} attempts.")
    return None
    


    

def get_nearby_emas(price, ema_dict, max_percent_diff=5.0):
    """
    Returns EMAs within ±5% of current price, sorted above and below.
    `ema_dict`: dict like {"EMA10_1D": 1234.5, "EMA20_1H": 1250.3}
    """
    above = []
    below = []
    for label, val in ema_dict.items():
        if pd.isna(val):
            continue
        pct_diff = abs((val - price) / price) * 100
        if pct_diff <= max_percent_diff:
            if val > price:
                above.append((label, val))
            elif val < price:
                below.append((label, val))
    # Sort closest first
    above.sort(key=lambda x: x[1])
    below.sort(key=lambda x: -x[1])
    return above, below

def is_nearby(ema_value, entry_price, direction, threshold=0.02):
    if ema_value is None:
        return False
    if direction == "long" and ema_value > entry_price and (ema_value - entry_price) / entry_price <= threshold:
        return True
    if direction == "short" and ema_value < entry_price and (entry_price - ema_value) / entry_price <= threshold:
        return True
    return False

def get_weighted_ema_guidance(entry_price, hourly_emas, daily_emas):
    """
    Returns a cleanly sorted message showing nearby EMAs within ±10% of entry,
    sorted by direction (above/below) and priority.
    """
    above = []
    below = []

    # Define priority EMAs (higher first)
    priority_emas = [
        ("1D", 200, daily_emas.get("ema200")),
        ("1D", 50, daily_emas.get("ema50")),
        ("1D", 20, daily_emas.get("ema20")),
        ("1D", 10, daily_emas.get("ema10")),
        ("1H", 200, hourly_emas.get("ema200")),
        ("1H", 50, hourly_emas.get("ema50")),
    ]

    for tf, period, value in priority_emas:
        if value is None or pd.isna(value):
            continue
        pct_diff = abs(entry_price - value) / entry_price
        if pct_diff <= 0.10:  # Only include EMAs within 10%
            label = f"{tf} EMA {period}: {value:.4f}"
            if value > entry_price:
                above.append(label)
            else:
                below.append(label)

    lines = ["📊 EMA Guidance (within 10% of entry):"]
    if above:
        lines.append("⬆️ *Resistance EMAs Above Entry:*")
        lines.extend([f"• {l}" for l in above])
    if below:
        lines.append("⬇️ *Support EMAs Below Entry:*")
        lines.extend([f"• {l}" for l in below])

    return "\n".join(lines) if above or below else ""


def detect_ema_confluence(ema_dict, entry_price, max_distance_pct=1.0):
    """
    Returns lists of confluence zones above and below entry price.
    A confluence zone = 2 or more EMAs within `max_distance_pct` of each other.
    """
    above = [val for val in ema_dict.values() if val > entry_price and not pd.isna(val)]
    below = [val for val in ema_dict.values() if val < entry_price and not pd.isna(val)]

    def find_clusters(values):
        values = sorted(values)
        clusters = []
        current_cluster = [values[0]]
        for v in values[1:]:
            if abs(v - current_cluster[-1]) / current_cluster[-1] * 100 <= max_distance_pct:
                current_cluster.append(v)
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [v]
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        return clusters

    return find_clusters(above), find_clusters(below)


    

def format_ema_guidance(emas_above, emas_below):
    lines = []
    if emas_above:
        lines.append("📈 *Nearby EMAs ABOVE:*")
        for label, val in sorted(emas_above, key=lambda x: x[1]):
            lines.append(f"• {label}: {val:.4f}")
    if emas_below:
        lines.append("📉 *Nearby EMAs BELOW:*")
        for label, val in sorted(emas_below, key=lambda x: -x[1]):
            lines.append(f"• {label}: {val:.4f}")
    return "\n".join(lines)

    

def get_pst_time(utc_timestamp):
    utc_dt = utc_timestamp if utc_timestamp.tzinfo else utc_timestamp.replace(tzinfo=timezone('UTC'))
    pst_dt = utc_dt.astimezone(timezone('US/Pacific'))
    return pst_dt.strftime("%Y-%m-%d %I:%M %p %Z")

def check_signals():
    print(f"🔍 Checking signals at {datetime.utcnow().isoformat()} UTC")
    for symbol in COINS:
        df = fetch_candles(symbol, GRANULARITY_1H)
        if df is None or len(df) < 12:
            continue


        df = calculate_indicators(df)
        swings_high = [(i, df["high"].iloc[i]) for i in range(len(df)) if find_swing_high(df, i)]
        swings_low = [(i, df["low"].iloc[i]) for i in range(len(df)) if find_swing_low(df, i)]
        last_row = df.iloc[-1]  # last completed 6h candle
        entry_price = last_row["close"]
        atr = last_row["atr"]
        #ema_msg = get_ema_guidance_all(last_row, entry_price)
        
        signal_time = get_pst_time(last_row["time"])

        if np.isnan(last_row["adx"]) or np.isnan(last_row["rsi"]) or np.isnan(last_row["macd_diff"]):
            continue

        # === LONG SETUP ===
        if last_row["adx"] > 25 and last_row["plus_di"] > last_row["minus_di"]+2 and last_row["macd_diff"] > 0.1 and last_row["rsi"] > 60:
            swings = [p for j, p in swings_low if j < len(df) - 2]
            if swings:
                sl = entry_price - 1.0 * atr
                risk = entry_price - sl
                tp = entry_price + 1.5 * atr
                open_confirmations[symbol] = {
                    "side": "long", "entry": entry_price, "tp": tp, "sl": sl,
                    "timestamp": datetime.utcnow(), "signal_time": signal_time
                }
                #send_telegram_alert(f"📈 *Potential LONG* on {symbol.upper()}\nEntry: {entry_price:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\nSignal Time (PST): {signal_time}")
                
                threading.Thread(target=monitor_stoch_confirmation, args=(symbol, "long", entry_price, tp, sl, signal_time)).start()
        
        # === SHORT SETUP ===
        elif last_row["adx"] > 25 and last_row["minus_di"] > last_row["plus_di"]+2 and last_row["macd_diff"] < -0.1 and last_row["rsi"] < 40:
            swings = [p for j, p in swings_high if j < len(df) - 2]
            if swings:
                sl = entry_price + 1.0 * atr
                risk = sl - entry_price
                tp = entry_price - 1.5 * atr
                open_confirmations[symbol] = {
                    "side": "short", "entry": entry_price, "tp": tp, "sl": sl,
                    "timestamp": datetime.utcnow(), "signal_time": signal_time
                }
                #send_telegram_alert(f"📉 *Potential SHORT* on {symbol.upper()}\nEntry: {entry_price:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\nSignal Time (PST): {signal_time}")
                threading.Thread(target=monitor_stoch_confirmation, args=(symbol, "short", entry_price, tp, sl, signal_time)).start()
                

def monitor_stoch_confirmation(symbol, side, entry, tp, sl, signal_time):
    timeout = datetime.utcnow() + timedelta(hours=CONFIRMATION_TIMEOUT_HOURS)
    print(f"🕒 Monitoring {symbol.upper()} for Stoch RSI entry confirmation...")

    while datetime.utcnow() < timeout:
        df_1h = fetch_candles(symbol, GRANULARITY_1H, lookback_days=12)
        df_1d = fetch_candles(symbol, GRANULARITY_1D, lookback_days=300)
        df_15m = fetch_candles(symbol, GRANULARITY_15m, lookback_days=3)

        if df_1h is None or df_1d is None or df_15m is None or len(df_1h) < 50 or len(df_15m) < 15 or len(df_1d) < 50:
            time.sleep(300)
            continue

        # Compute required EMAs
        for p in [10, 20,50, 200]:
            df_1h[f"ema{p}_1h"] = df_1h["close"].ewm(span=p, adjust=False).mean()
            df_1d[f"ema{p}_1d"] = df_1d["close"].ewm(span=p, adjust=False).mean()

        # Calculate Stoch RSI
        stoch = StochRSIIndicator(df_15m["close"]).stochrsi_k().dropna() * 100
        stoch_d = StochRSIIndicator(df_15m["close"]).stochrsi_d().dropna() * 100
        if stoch.isna().all() or stoch_d.isna().all():
            time.sleep(300)
            continue

        entry_price = df_15m["close"].iloc[-1]

        # Grab EMAs for SL/TP context
        hourly_emas = {}
        daily_emas = {}
        for p in [10, 20, 50, 200]:
            hourly_col = f"ema{p}_1h"
            daily_col = f"ema{p}_1d"
            hourly_emas[f"ema{p}"] = df_1h[hourly_col].iloc[-1] if hourly_col in df_1h.columns else None
            daily_emas[f"ema{p}"] = df_1d[daily_col].iloc[-1] if daily_col in df_1d.columns else None

        ema_dict1h = {}
        for key, val in hourly_emas.items():
            if key not in ["ema10", "ema20"]:  # exclude 1H ema10 and ema20
                ema_dict1h[f"{key}_1H"] = val

        ema_msg = get_weighted_ema_guidance(entry_price, hourly_emas, daily_emas)
        cluster_msg = get_ema_cluster_score(entry_price, ema_dict1h, daily_emas, side)

        full_msg = f"\n\n{ema_msg}{cluster_msg}" if ema_msg or cluster_msg else ""
    

        # Check if nearby EMA200 blocks the trade
        if is_nearby(daily_emas["ema200"], entry_price, side) or is_nearby(hourly_emas["ema200"], entry_price, side):
            print(f"❌ {symbol.upper()} skipped due to nearby EMA 200")
            if symbol in open_confirmations:
                del open_confirmations[symbol]
            return

        # Check for confirmation
        k, d = stoch.iloc[-1], stoch_d.iloc[-1]
        prev_k = stoch.iloc[-2] if len(stoch) > 1 else k
        prev_d = stoch_d.iloc[-2] if len(stoch_d) > 1 else d

        print(f"🕒 {symbol.upper()} k={k:.2f}, d={d:.2f}, prev_k={prev_k:.2f}, prev_d={prev_d:.2f}")

        if side == "long" and k > d and k < 30 and k > prev_k and d > prev_d:
            confirmed_entry_price = get_live_price(symbol)
            if confirmed_entry_price is None:
                print(f"❌ Skipping alert for {symbol.upper()} — live price not available.")
                if symbol in open_confirmations:
                    del open_confirmations[symbol]
                return


            send_telegram_alert(f"✅mini *LONG Confirmed* on {symbol.upper()}\nEntry: {confirmed_entry_price:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\n{signal_time}{full_msg}")
            if symbol in open_confirmations:
                del open_confirmations[symbol]
            break

        elif side == "short" and k < d and k > 70 and k < prev_k and d < prev_d:
            confirmed_entry_price = get_live_price(symbol)
            if confirmed_entry_price is None:
                print(f"❌ Skipping alert for {symbol.upper()} — live price not available.")
                if symbol in open_confirmations:
                    del open_confirmations[symbol]
                return

            send_telegram_alert(f"✅mini *SHORT Confirmed* on {symbol.upper()}\nEntry: {confirmed_entry_price:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\n{signal_time}{full_msg}")
            if symbol in open_confirmations:
                del open_confirmations[symbol]
            break

        time.sleep(300)  # Wait 5 minutes before checking again


def get_next_coinbase_1h_time():
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    # Coinbase 1H candle times: 00:00, 06:00, 12:00, 18:00 UTC
    next_times = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
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
