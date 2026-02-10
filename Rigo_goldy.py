import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from ta.momentum import StochRSIIndicator
from ta.trend import EMAIndicator, MACD

# === CONFIG ===
COINS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd",
           "syrup-usd","fartcoin-usd", "aero-usd", "link-usd","hbar-usd",
           "aave-usd", "fet-usd", "crv-usd","tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd",
           "bch-usd", "inj-usd", "pepe-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
           "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
           "qnt-usd", "tia-usd", "ip-usd" , "pnut-usd", "apt-usd", "vet-usd",
           "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "velo-usd", "coti-usd",
           "axs-usd"]
GOLDEN_ZONE_MARGIN = 0.1
LOOKBACK_HOURS = 168
SUP_RES_WINDOW = 20
COOLDOWN_MINUTES = 30
POSITION_SIZE = 2500
BOT_TOKEN = '8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw'
CHAT_ID = '7967738614'

last_alert_times = {}

def send_alert(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Telegram error:", e)

def get_candles(symbol, granularity):
    url = f'https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}'
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to fetch {symbol} candles.")
        return None
    df = pd.DataFrame(r.json(), columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df.sort_values('time')

def calc_fibs(high, low):
    diff = high - low
    return {
        '0.618': high - diff * 0.618,
        '0.66': high - diff * 0.66
    }

def find_zones(prices, window=SUP_RES_WINDOW):
    supports, resistances = [], []
    for i in range(window, len(prices)):
        segment = prices[i-window:i]
        high = segment['high'].max()
        low = segment['low'].min()
        close = segment.iloc[-1]['close']
        if abs(close - low) / close < 0.01:
            supports.append(low)
        if abs(close - high) / close < 0.01:
            resistances.append(high)
    return supports, resistances

def check_signal(symbol):
    df_1h = get_candles(symbol, 3600)
    df_1d = get_candles(symbol, 86400)

    if df_1h is None or df_1d is None or len(df_1h) < LOOKBACK_HOURS + 5:
        return

    df_1h['stoch_k'] = StochRSIIndicator(df_1h['close']).stochrsi_k()
    df_1h['stoch_d'] = StochRSIIndicator(df_1h['close']).stochrsi_d()
    macd = MACD(df_1h['close'])
    df_1h['macd_diff'] = macd.macd_diff()
    df_1d['ema200'] = EMAIndicator(df_1d['close'], window=200).ema_indicator()
    df_1d.set_index('time', inplace=True)

    row = df_1h.iloc[-1]
    time = row['time']
    price = row['close']

    if row['macd_diff'] < 0:
        return

    recent_df = df_1h.iloc[-LOOKBACK_HOURS:]
    zones_df = df_1h.iloc[-SUP_RES_WINDOW:]
    supports, resistances = find_zones(zones_df)

    daily_row = df_1d[df_1d.index <= time]
    if daily_row.empty:
        return

    ema200 = daily_row['ema200'].iloc[-1]
    is_bull = df_1d['close'].iloc[-1] > ema200
    is_bear = not is_bull

    swing_high = recent_df['high'].max()
    swing_low = recent_df['low'].min()
    fibs = calc_fibs(swing_high, swing_low)
    golden_min = fibs['0.66'] * (1 - GOLDEN_ZONE_MARGIN)
    golden_max = fibs['0.618'] * (1 + GOLDEN_ZONE_MARGIN)

    k = row['stoch_k']
    d = row['stoch_d']
    crossed_up = k > d and k < 40
    crossed_down = k < d and k > 60

    now = datetime.utcnow()
    if symbol in last_alert_times:
        if now - last_alert_times[symbol] < timedelta(minutes=COOLDOWN_MINUTES):
            return

    if is_bull and golden_min <= price <= golden_max and crossed_up:
        tp = max([r for r in resistances if r > price], default=swing_high)
        sl = min([s for s in supports if s < price], default=swing_low)
        message = f"LONG SIGNAL for {symbol}\nPrice: {price:.2f}\nTP: {tp:.2f}\nSL: {sl:.2f}"
        send_alert(message)
        last_alert_times[symbol] = now
    elif is_bear and golden_min <= price <= golden_max and crossed_down:
        tp = min([s for s in supports if s < price], default=swing_low)
        sl = max([r for r in resistances if r > price], default=swing_high)
        message = f"SHORT SIGNAL for {symbol}\nPrice: {price:.2f}\nTP: {tp:.2f}\nSL: {sl:.2f}"
        send_alert(message)
        last_alert_times[symbol] = now

# === CONTINUOUS LIVE SCAN ===
while True:
    for coin in COINS:
        check_signal(coin)
    time.sleep(300)  # wait 5 minutes between scans
