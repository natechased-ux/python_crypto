import requests
import pandas as pd
import numpy as np
import time
from ta.volatility import AverageTrueRange
from datetime import datetime

# === CONFIG ===
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

granularity = 3600  # 1 hour
num_candles = 100
atr_length = 14
tp_mult = 2.0
sl_mult = 1.0
cooldown_secs = 1800  # 30 minutes

TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
last_alert_time = {}

# === HELPERS ===
def fetch_data(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    try:
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        df = pd.DataFrame(res.json(), columns=["time", "low", "high", "open", "close", "volume"])
        df = df.sort_values("time").reset_index(drop=True).head(num_candles)
        return df
    except Exception as e:
        print(f"[{symbol}] Fetch error: {e}")
        return None

def calculate_indicators(df):
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=atr_length).average_true_range()
    df['vol_delta'] = df['volume'] * np.sign(df['close'] - df['open'])
    df['cvd'] = df['vol_delta'].cumsum()
    return df

def get_peaks(series, window=3):
    return [i for i in range(window, len(series) - window)
            if series[i] == max(series[i - window:i + window + 1])]

def get_troughs(series, window=3):
    return [i for i in range(window, len(series) - window)
            if series[i] == min(series[i - window:i + window + 1])]

def detect_divergence(df):
    price = df['close'].values
    cvd = df['cvd'].values
    highs = get_peaks(price)
    lows = get_troughs(price)
    cvd_highs = get_peaks(cvd)
    cvd_lows = get_troughs(cvd)

    for i in reversed(range(len(highs) - 1)):
        ph1, ph2 = highs[i], highs[i + 1]
        if ph2 <= ph1: continue
        ch1 = min(cvd_highs, key=lambda x: abs(x - ph1), default=None)
        ch2 = min(cvd_highs, key=lambda x: abs(x - ph2), default=None)
        if ch1 and ch2 and cvd[ch2] < cvd[ch1]:
            return ph2, 'bearish'

    for i in reversed(range(len(lows) - 1)):
        pl1, pl2 = lows[i], lows[i + 1]
        if pl2 >= pl1: continue
        if pl1 in cvd_lows and pl2 in cvd_lows and cvd[pl2] > cvd[pl1]:
            return pl2, 'bullish'

    return None

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        r = requests.post(url, data=payload)
        if not r.ok:
            print("Telegram error:", r.text)
    except Exception as e:
        print("Telegram send failed:", e)

# === MAIN CHECK ===
def check_divergences():
    now = int(time.time())
    for symbol in symbols:
        df = fetch_data(symbol)
        if df is None or len(df) < atr_length + 10:
            continue

        df = calculate_indicators(df)
        result = detect_divergence(df)
        if result is None:
            continue

        idx, div_type = result
        latest_idx = len(df) - 2  # use second-to-last bar only
        if idx != latest_idx:
            continue

        if symbol in last_alert_time and now - last_alert_time[symbol] < cooldown_secs:
            continue

        price = df.loc[idx, 'close']
        atr = df.loc[idx, 'atr']
        if pd.isna(atr) or atr == 0:
            continue

        tp = price + atr * tp_mult if div_type == 'bullish' else price - atr * tp_mult
        sl = price - atr * sl_mult if div_type == 'bullish' else price + atr * sl_mult
        timestamp = datetime.utcfromtimestamp(df.loc[idx, 'time']).strftime('%Y-%m-%d %H:%M:%S UTC')

        message = (
            f"📉 Divergence Alert — {symbol.upper()}\n"
            f"Type: {div_type.capitalize()}\n"
            f"Time: {timestamp}\n"
            f"Price: ${price:.4f}\n"
            f"TP: ${tp:.4f}\n"
            f"SL: ${sl:.4f}"
        )

        send_telegram(message)
        last_alert_time[symbol] = now
        print(f"[{symbol.upper()}] Alert sent: {div_type} divergence")

# === ENTRYPOINT ===
if __name__ == "__main__":
    while True:
        print(f"\n⏱️ Checking at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        check_divergences()
        time.sleep(300)  # Run every hour
