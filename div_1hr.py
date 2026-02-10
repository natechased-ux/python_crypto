import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from ta.volatility import AverageTrueRange
from telegram import Bot

# Telegram Bot Configurations
TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = Bot(token=TOKEN)

# Config
symbols = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd",
           "syrup-usd", "fartcoin-usd", "aero-usd", "link-usd", "hbar-usd",
           "aave-usd", "fet-usd", "crv-usd", "tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd",
           "bch-usd", "inj-usd", "pepe-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
           "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
           "qnt-usd", "tia-usd", "ip-usd", "pnut-usd", "apt-usd", "vet-usd",
           "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "velo-usd", "coti-usd",
           "axs-usd"]

granularity = 3600
atr_length = 14
hold_bars = 6
sl_atr_mult = 1.0
tp_atr_mult = 2.0

# Calculate ATR
def calculate_atr(data):
    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=atr_length)
    data['atr'] = atr_indicator.average_true_range()
    return data

# Calculate CVD
def calculate_cvd(data):
    data['vol_delta'] = data['volume'] * np.sign(data['close'] - data['open'])
    data['cvd'] = data['vol_delta'].cumsum()
    return data

# Find local peaks
def get_peaks(series, window=3):
    return [i for i in range(window, len(series) - window) if series[i] == max(series[i - window:i + window + 1])]

# Find local troughs
def get_troughs(series, window=3):
    return [i for i in range(window, len(series) - window) if series[i] == min(series[i - window:i + window + 1])]

# Detect divergences
def detect_divergences(data):
    price = data['close'].values
    cvd = data['cvd'].values
    price_highs = get_peaks(price)
    price_lows = get_troughs(price)
    cvd_highs = get_peaks(cvd)
    cvd_lows = get_troughs(cvd)

    divergences = []

    for i in range(len(price_highs) - 1):
        ph1, ph2 = price_highs[i], price_highs[i + 1]
        if ph2 <= ph1:
            continue
        ch1 = min(cvd_highs, key=lambda x: abs(x - ph1)) if cvd_highs else None
        ch2 = min(cvd_highs, key=lambda x: abs(x - ph2)) if cvd_highs else None
        if ch1 is None or ch2 is None:
            continue
        if cvd[ch2] < cvd[ch1]:
            divergences.append((ph2, 'bearish'))

    for i in range(len(price_lows) - 1):
        pl1, pl2 = price_lows[i], price_lows[i + 1]
        if price[pl2] >= price[pl1]:
            continue
        if pl1 in cvd_lows and pl2 in cvd_lows:
            if cvd[pl2] > cvd[pl1]:
                divergences.append((pl2, 'bullish'))

    return sorted(divergences, key=lambda x: x[0])

# Fetch data
def fetch_historical_data(symbol):
    for _ in range(3):
        try:
            url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
            response = requests.get(url)
            if response.status_code == 200:
                columns = ["time", "low", "high", "open", "close", "volume"]
                data = pd.DataFrame(response.json(), columns=columns)
                data = data.sort_values(by="time").reset_index(drop=True)
                return data
            elif response.status_code == 429:
                time.sleep(2)
            else:
                return None
        except:
            time.sleep(1)
    return None

# Format price
def format_price(p):
    return f"{p:.4f}" if p < 10 else f"{p:.2f}"

# Send Telegram alert (non-async)
def send_alert(symbol, idx, div_type, data):
    entry_price = data.loc[idx, 'close']
    atr = data.loc[idx, 'atr']

    if div_type == 'bullish':
        sl = entry_price - atr * sl_atr_mult
        tp = entry_price + atr * tp_atr_mult
    else:
        sl = entry_price + atr * sl_atr_mult
        tp = entry_price - atr * tp_atr_mult

    msg = (
        f"📈 {symbol.upper()} 1 hr CVD Divergence Alert!\n"
        f"Type: {div_type.upper()} Divergence\n"
        f"Entry Price: {format_price(entry_price)}\n"
        f"Suggested SL: {format_price(sl)} | TP: {format_price(tp)}\n"
        f"Time: {pd.to_datetime(data.loc[idx, 'time'], unit='s')}\n"
        f"Hold period: approx {hold_bars * (granularity // 3600)}h"
    )
    bot.send_message(chat_id=CHAT_ID, text=msg)

# Main monitoring loop
def monitor():
    last_alerts = {}
    for symbol in symbols:
        data = fetch_historical_data(symbol)
        if data is not None:
            last_alerts[(symbol, 'bullish')] = len(data) - 1
            last_alerts[(symbol, 'bearish')] = len(data) - 1

    while True:
        for symbol in symbols:
            data = fetch_historical_data(symbol)
            if data is None:
                continue

            data = calculate_atr(data)
            data = calculate_cvd(data)
            divergences = detect_divergences(data)

            for idx, div_type in divergences:
                key = (symbol, div_type)
                last_idx = last_alerts.get(key, -1)

                if idx > last_idx:
                    send_alert(symbol, idx, div_type, data)
                    last_alerts[key] = idx

        print(f"⏰ 1 hr DIV — {pd.Timestamp.now()}")
        time.sleep(60)

if __name__ == "__main__":
    monitor()
