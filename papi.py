import pandas as pd
import numpy as np
import requests
import time
import pytz
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
CHAT_ID = "7967738614"
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# === CONFIGURATION ===
COINS = [
    "btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd",
    "sol-usd", "wif-usd", "ondo-usd", "sei-usd", "magic-usd", "ape-usd",
    "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
    "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd",
    "avax-usd", "xcn-usd", "uni-usd", "mkr-usd", "toshi-usd", "near-usd",
    "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
    "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
    "qnt-usd", "tia-usd", "ip-usd", "pnut-usd", "apt-usd", "ena-usd", "turbo-usd",
    "bera-usd", "pol-usd", "mask-usd", "pyth-usd", "sand-usd", "morpho-usd",
    "mana-usd", "coti-usd", "c98-usd", "axs-usd"
]

BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY_6H = 21600
GRANULARITY_1H = 3600
GRANULARITY_1D = 86400
TP_MULTIPLIER = 0.75
CONFIRMATION_TIMEOUT_HOURS = 6

open_trades = {}
open_confirmations = {}

# =========================
#   FETCH DATA FUNCTIONS
# =========================
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
                chunk = pd.DataFrame(
                    data, columns=["time", "low", "high", "open", "close", "volume"]
                )
                df = pd.concat([df, chunk], ignore_index=True)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        start = chunk_end

    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.drop_duplicates(subset="time", inplace=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# =========================
#   INDICATORS
# =========================
def calculate_indicators(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    adx = ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()
    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()

    for period in [10, 50, 100, 200]:
        df[f"ema{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df

# =========================
#   SWING POINT DETECTION
# =========================
def get_pst_time(utc_timestamp):
    utc_dt = utc_timestamp if utc_timestamp.tzinfo else utc_timestamp.replace(tzinfo=timezone('UTC'))
    pst_dt = utc_dt.astimezone(timezone('US/Pacific'))
    return pst_dt.strftime("%Y-%m-%d %I:%M %p %Z")

def find_swing_high(df, index, window=2):
    """
    Detects if a candle is a swing high: high is greater than 'window' candles before and after.
    """
    if index < window or index + window >= len(df):
        return False
    center = df["high"].iloc[index]
    return all(center > df["high"].iloc[index - i] and center > df["high"].iloc[index + i]
               for i in range(1, window + 1))

def find_swing_low(df, index, window=2):
    """
    Detects if a candle is a swing low: low is less than 'window' candles before and after.
    """
    if index < window or index + window >= len(df):
        return False
    center = df["low"].iloc[index]
    return all(center < df["low"].iloc[index - i] and center < df["low"].iloc[index + i]
               for i in range(1, window + 1))

def get_live_price(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/ticker"
    try:
        response = requests.get(url)
        data = response.json()
        return float(data["price"])
    except Exception as e:
        print(f"Error fetching live price for {symbol}: {e}")
        return None
# =========================
#   RSI DIVERGENCE
# =========================
def detect_rsi_divergence(df, side="bullish", rsi_period=14, swing_lookback=5):
    """
    Detect RSI divergences:
    - bullish: price lower low, RSI higher low
    - hidden_bullish: price higher low, RSI lower low
    - bearish: price higher high, RSI lower high
    - hidden_bearish: price lower high, RSI higher high
    """
    if df is None or len(df) < swing_lookback * 2 + rsi_period + 1:
        return False

    df = df.copy()
    df["rsi"] = RSIIndicator(df["close"], window=rsi_period).rsi()

    # Detect recent swing highs/lows
    df["swing_low"] = df["low"].rolling(window=swing_lookback*2+1, center=True).apply(
        lambda x: x.iloc[swing_lookback] == min(x), raw=False
    ).fillna(0)
    df["swing_high"] = df["high"].rolling(window=swing_lookback*2+1, center=True).apply(
        lambda x: x.iloc[swing_lookback] == max(x), raw=False
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

# =========================
#   PRICE FORMATTING
# =========================
def fmt_price(price):
    """Formats price with dynamic decimal places for small coins."""
    if price >= 100:
        return f"{price:.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.6f}"
    else:
        return f"{price:.8f}"

# =========================
#   EMA & CLUSTER INFO
# =========================
def get_weighted_ema_guidance(entry_price, hourly_emas, daily_emas):
    above, below = [], []
    priority_emas = [
        ("1D", 200, daily_emas.get("ema200")),
        ("1D", 100, daily_emas.get("ema100")),
        ("1D", 50, daily_emas.get("ema50")),
        ("1D", 20, daily_emas.get("ema20")),
        ("1D", 10, daily_emas.get("ema10")),
        ("1H", 200, hourly_emas.get("ema200")),
        ("1H", 50, hourly_emas.get("ema50"))
    ]
    for tf, period, value in priority_emas:
        if value is None or pd.isna(value):
            continue
        pct_diff = abs(entry_price - value) / entry_price
        if pct_diff <= 0.10:
            label = f"{tf} EMA {period}: {fmt_price(value)}"
            if value > entry_price:
                above.append(label)
            else:
                below.append(label)

    lines = ["📊 EMA Guidance:"]
    if above:
        lines.append("⬆️ Resistance EMAs Above Entry:")
        lines.extend([f"• {l}" for l in above])
    if below:
        lines.append("⬇️ Support EMAs Below Entry:")
        lines.extend([f"• {l}" for l in below])

    return "\n".join(lines) if above or below else ""

# =========================
#   TELEGRAM ALERT
# =========================
def send_telegram_alert(message):
    for attempt in range(3):
        try:
            bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")
            return
        except Exception as e:
            print(f"❌ Attempt {attempt+1} failed: {e}")
            time.sleep(2)
# =========================
#   MAIN SIGNAL LOGIC
# =========================
def check_signals():
    print(f"🔍 Checking signals at {datetime.utcnow().isoformat()} UTC")
    for symbol in COINS:
        df = fetch_candles(symbol, GRANULARITY_6H)
        if df is None or len(df) < 30:
            continue

        df = calculate_indicators(df)
        swings_high = [(i, df["high"].iloc[i]) for i in range(len(df)) if find_swing_high(df, i)]
        swings_low = [(i, df["low"].iloc[i]) for i in range(len(df)) if find_swing_low(df, i)]
        last_row = df.iloc[-1]
        entry_price = last_row["close"]
        signal_time = get_pst_time(last_row["time"])

        if np.isnan(last_row["adx"]) or np.isnan(last_row["rsi"]) or np.isnan(last_row["macd_diff"]):
            continue

        # === LONG SETUP ===
        if last_row["adx"] > 20 and last_row["plus_di"] > last_row["minus_di"] and last_row["macd_diff"] > 0 and last_row["rsi"] > 55:
            swings = [p for j, p in swings_low if j < len(df) - 2]
            if swings:
                sl = swings[-1]
                risk = entry_price - sl
                tp = entry_price + TP_MULTIPLIER * risk
                threading.Thread(target=monitor_divergence, args=(symbol, "long", entry_price, tp, sl, signal_time)).start()

        # === SHORT SETUP ===
        elif last_row["adx"] > 20 and last_row["minus_di"] > last_row["plus_di"] and last_row["macd_diff"] < 0 and last_row["rsi"] < 45:
            swings = [p for j, p in swings_high if j < len(df) - 2]
            if swings:
                sl = swings[-1]
                risk = sl - entry_price
                tp = entry_price - TP_MULTIPLIER * risk
                threading.Thread(target=monitor_divergence, args=(symbol, "short", entry_price, tp, sl, signal_time)).start()

# =========================
#   MONITOR FOR DIVERGENCE CONFIRMATION
# =========================
def monitor_divergence(symbol, side, entry, tp, sl, signal_time):
    timeout = datetime.utcnow() + timedelta(hours=CONFIRMATION_TIMEOUT_HOURS)
    print(f"🕒 Monitoring {symbol.upper()} for RSI divergence confirmation...")

    while datetime.utcnow() < timeout:
        df_1h = fetch_candles(symbol, GRANULARITY_1H, lookback_days=10)
        df_1d = fetch_candles(symbol, GRANULARITY_1D, lookback_days=200)

        if df_1h is None or df_1d is None or len(df_1h) < 50:
            time.sleep(300)
            continue

        # EMAs for alert context
        for p in [10, 20, 50, 200]:
            df_1h[f"ema{p}_1h"] = df_1h["close"].ewm(span=p).mean()
            df_1d[f"ema{p}_1d"] = df_1d["close"].ewm(span=p).mean()

        hourly_emas = {f"ema{p}": df_1h[f"ema{p}_1h"].iloc[-1] for p in [10, 20, 50, 200]}
        daily_emas = {f"ema{p}": df_1d[f"ema{p}_1d"].iloc[-1] for p in [10, 20, 50, 200]}

        ema_msg = get_weighted_ema_guidance(entry, hourly_emas, daily_emas)

        # Check divergence
        if side == "long" and (detect_rsi_divergence(df_1h, "bullish") or detect_rsi_divergence(df_1h, "hidden_bullish")):
            confirmed_entry = get_live_price(symbol)
            alert_msg = (
                f"✅DIV *LONG Confirmed* on {symbol.upper()}\n"
                f"Entry: {fmt_price(confirmed_entry)}\nTP: {fmt_price(tp)}\nSL: {fmt_price(sl)}\n"
                f"{signal_time}\n\n{ema_msg}"
            )
            send_telegram_alert(alert_msg)
            monitor_trade(symbol, side, confirmed_entry, tp, sl, signal_time)
            break

        elif side == "short" and (detect_rsi_divergence(df_1h, "bearish") or detect_rsi_divergence(df_1h, "hidden_bearish")):
            confirmed_entry = get_live_price(symbol)
            alert_msg = (
                f"✅DIV *SHORT Confirmed* on {symbol.upper()}\n"
                f"Entry: {fmt_price(confirmed_entry)}\nTP: {fmt_price(tp)}\nSL: {fmt_price(sl)}\n"
                f"{signal_time}\n\n{ema_msg}"
            )
            send_telegram_alert(alert_msg)
            monitor_trade(symbol, side, confirmed_entry, tp, sl, signal_time)
            break

        time.sleep(300)  # wait 5m

# =========================
#   EARLY EXIT LOGIC
# =========================
def monitor_trade(symbol, side, entry, tp, sl, signal_time):
    """Monitor trade for TP, SL, or opposite divergence to exit early."""
    print(f"📈 Monitoring active trade for {symbol.upper()}...")

    while True:
        price = get_live_price(symbol)
        if not price:
            time.sleep(60)
            continue

        # TP hit
        if side == "long" and price >= tp:
            send_telegram_alert(f"🎯 {symbol.upper()} TP Hit at {fmt_price(price)}")
            break
        elif side == "short" and price <= tp:
            send_telegram_alert(f"🎯 {symbol.upper()} TP Hit at {fmt_price(price)}")
            break

        # SL hit
        if side == "long" and price <= sl:
            send_telegram_alert(f"❌ {symbol.upper()} SL Hit at {fmt_price(price)}")
            break
        elif side == "short" and price >= sl:
            send_telegram_alert(f"❌ {symbol.upper()} SL Hit at {fmt_price(price)}")
            break

        # Early exit if opposite divergence appears
        df_1h = fetch_candles(symbol, GRANULARITY_1H, lookback_days=10)
        if df_1h is not None:
            if side == "long" and detect_rsi_divergence(df_1h, "bearish"):
                send_telegram_alert(f"⚠️ {symbol.upper()} Opposite Bearish Divergence — Early Exit at {fmt_price(price)}")
                break
            elif side == "short" and detect_rsi_divergence(df_1h, "bullish"):
                send_telegram_alert(f"⚠️ {symbol.upper()} Opposite Bullish Divergence — Early Exit at {fmt_price(price)}")
                break

        time.sleep(300)  # check every 5m

# =========================
#   PRICE FORMATTER
# =========================
def fmt_price(price):
    """Format prices with dynamic decimal places based on magnitude."""
    if price is None:
        return "N/A"
    if price >= 100:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:,.4f}"
    else:
        return f"{price:,.8f}"

# =========================
#   SCHEDULER HELPERS
# =========================
def get_next_coinbase_6h_time():
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    # Coinbase 6H candle times: 00:00, 06:00, 12:00, 18:00 UTC
    next_times = [0, 6, 12, 18]
    for t in next_times:
        candidate = now.replace(hour=t)
        if candidate > now:
            return candidate + timedelta(minutes=5)  # buffer
    return (now + timedelta(days=1)).replace(hour=0) + timedelta(minutes=5)

def schedule_6h_task():
    def run_and_reschedule():
        print(f"🕒 Running check at {datetime.utcnow().isoformat()}")
        check_signals()
        schedule_6h_task()

    next_run_time = get_next_coinbase_6h_time()
    delay = (next_run_time - datetime.utcnow()).total_seconds()
    print(f"📅 Next signal check at {next_run_time.isoformat()} UTC")
    schedule.clear()
    schedule.every(delay).seconds.do(run_and_reschedule)

# =========================
#   STARTUP
# =========================
if __name__ == "__main__":
    print("🚀 Big Papi Smart Money Alert System Running")
    check_signals()
    schedule_6h_task()

    while True:
        schedule.run_pending()
        time.sleep(5)
        
