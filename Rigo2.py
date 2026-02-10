import requests
import pandas as pd
import time
import datetime
import numpy as np
from ta.momentum import StochRSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# === CONFIG ===
BOT_TOKEN = '8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw'
CHAT_ID = '7967738614'
COINS = [
    "btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
    "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
    "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd", "avax-usd", "xcn-usd", "uni-usd",
    "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
    "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd", "qnt-usd", "tia-usd", "ip-usd",
    "pnut-usd", "apt-usd", "vet-usd", "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
    "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "velo-usd", "coti-usd", "axs-usd"
]
CHECK_INTERVAL = 300  # seconds (5 minutes)
COOLDOWN = 1800  # 30 minutes
FIB_LEVELS = [0.618, 0.66]
GOLDEN_ZONE_MARGIN = 0.0025  # ±0.25%

last_alert_time = {}

# === HELPERS ===
def send_telegram_message(message: str):
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    payload = {'chat_id': CHAT_ID, 'text': message}
    requests.post(url, data=payload)

def get_candles(symbol: str, granularity: int, limit: int = 300):
    url = f'https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}'
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df.sort_values('time')

def calc_fibs(high, low):
    diff = high - low
    return {
        '0.618': high - diff * 0.618,
        '0.66': high - diff * 0.66,
        'target': high  # target is full retrace
    }

def detect_engulfing(df):
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    if curr['close'] > curr['open'] and prev['close'] < prev['open']:
        return curr['close'] > prev['open'] and curr['open'] < prev['close']
    if curr['close'] < curr['open'] and prev['close'] > prev['open']:
        return curr['close'] < prev['open'] and curr['open'] > prev['close']
    return False

def detect_inside_bar(df):
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return curr['high'] < prev['high'] and curr['low'] > prev['low']

# === CORE LOGIC ===
def analyze_coin(symbol):
    global last_alert_time

    df_1h = get_candles(symbol, 3600)
    df_1d = get_candles(symbol, 86400)
    if df_1h is None or df_1d is None or len(df_1h) < 100 or len(df_1d) < 100:
        return

    df = df_1h.copy()
    current_price = df['close'].iloc[-1]

    # Daily trend
    ema200 = EMAIndicator(df_1d['close'], window=200).ema_indicator().iloc[-1]
    is_bullish = df_1d['close'].iloc[-1] > ema200
    is_bearish = not is_bullish

    # Fibs
    swing_high = df['high'][-168:].max()
    swing_low = df['low'][-168:].min()
    fibs = calc_fibs(swing_high, swing_low)
    golden_min = fibs['0.66'] * (1 - GOLDEN_ZONE_MARGIN)
    golden_max = fibs['0.618'] * (1 + GOLDEN_ZONE_MARGIN)

    # Indicators
    stoch_rsi = StochRSIIndicator(df['close']).stochrsi_k()
    k = stoch_rsi.iloc[-1]
    d = StochRSIIndicator(df['close']).stochrsi_d().iloc[-1]
    crossed_down = k < d and k > 80
    crossed_up = k > d and k < 20

    # Price action
    engulfing = detect_engulfing(df)
    inside_bar = detect_inside_bar(df)

    now = time.time()
    last_time = last_alert_time.get(symbol, 0)
    if now - last_time < COOLDOWN:
        return

    message = None

    if is_bullish and golden_min <= current_price <= golden_max and crossed_up and (engulfing or inside_bar):
        sl = fibs['0.66'] * 0.995
        tp = swing_high
        message = f"🟢 *LONG SIGNAL*\nSymbol: {symbol}\nPrice: {current_price:.2f}\nTP: {tp:.2f}\nSL: {sl:.2f}\nConfluence: Golden Zone + Bull Trend + StochRSI up + {'Engulfing' if engulfing else 'Inside Bar'}"
    elif is_bearish and golden_min <= current_price <= golden_max and crossed_down and (engulfing or inside_bar):
        sl = fibs['0.618'] * 1.005
        tp = swing_low
        message = f"🔴 *SHORT SIGNAL*\nSymbol: {symbol}\nPrice: {current_price:.2f}\nTP: {tp:.2f}\nSL: {sl:.2f}\nConfluence: Golden Zone + Bear Trend + StochRSI down + {'Engulfing' if engulfing else 'Inside Bar'}"

    if message:
        send_telegram_message(message)
        last_alert_time[symbol] = now

# === MAIN LOOP ===
while True:
    for coin in COINS:
        try:
            analyze_coin(coin)
        except Exception as e:
            print(f"[{coin}] Error: {e}")
    print(f"[{datetime.datetime.now()}] Cycle complete.")
    time.sleep(CHECK_INTERVAL)
