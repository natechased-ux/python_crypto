import requests
import pandas as pd
import numpy as np
import datetime
import time
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD

# --- Telegram Config ---
BOT_TOKEN = '8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw'
CHAT_ID = '7967738614'

# --- Parameters ---
SYMBOLS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
         "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
         "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd", "avax-usd", "xcn-usd", "uni-usd",
         "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
         "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd", "qnt-usd", "tia-usd", "ip-usd",
         "pnut-usd", "apt-usd", "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "pyth-usd", "sand-usd",
         "morpho-usd", "mana-usd", "coti-usd", "c98-usd", "axs-usd"]
INTERVAL = '1h'
CANDLE_LIMIT = 100
BOLL_BAND_WIDTH_THRESHOLD = 0.02  # 2%
VOLUME_SPIKE_MULTIPLIER = 1.5

# --- Coinbase Candle Fetcher ---
def fetch_candles(symbol, interval='1h', limit=100):
    granularity_map = {'15m': 900, '1h': 3600}
    granularity = granularity_map[interval]
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    params = {'granularity': granularity, 'limit': limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# --- Indicator Calculator ---
def calculate_indicators(df):
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    rsi = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi.rsi()

    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()

    macd = MACD(close=df['close'])
    df['macd_diff'] = macd.macd_diff()

    df['avg_volume'] = df['volume'].rolling(window=20).mean()
    return df

# --- Market Mode Detection ---
def is_range(df):
    return df['adx'].iloc[-1] < 20

def is_breakout(df):
    return df['adx'].iloc[-1] > 25 and df['bb_width'].iloc[-2] < BOLL_BAND_WIDTH_THRESHOLD

# --- Mean Reversion Strategy ---
def check_mean_reversion_signal(df):
    latest = df.iloc[-1]
    if latest['close'] < latest['bb_lower'] and latest['rsi'] < 30:
        return "Mean Reversion Long"
    elif latest['close'] > latest['bb_upper'] and latest['rsi'] > 70:
        return "Mean Reversion Short"
    return None

# --- Breakout Strategy ---
def check_breakout_signal(df):
    latest = df.iloc[-1]
    volume_spike = latest['volume'] > VOLUME_SPIKE_MULTIPLIER * df['avg_volume'].iloc[-1]
    if latest['close'] > latest['bb_upper'] and latest['macd_diff'] > 0 and volume_spike:
        return "Breakout Long"
    elif latest['close'] < latest['bb_lower'] and latest['macd_diff'] < 0 and volume_spike:
        return "Breakout Short"
    return None

# --- Telegram Alert ---
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

# --- Main Execution ---
def fetch_live_price(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/ticker"
    response = requests.get(url).json()
    return float(response['price'])

def run_alerts():
    for symbol in SYMBOLS:
        try:
            df = fetch_candles(symbol, INTERVAL, CANDLE_LIMIT)
            df = calculate_indicators(df)

            strategy = None
            if is_range(df):
                strategy = check_mean_reversion_signal(df)
            elif is_breakout(df):
                strategy = check_breakout_signal(df)

            if strategy:
                live_price = fetch_live_price(symbol)
                latest = df.iloc[-1]

                # === Live price confirmation ===
                confirm = False
                if strategy == "Mean Reversion Long" and live_price <= latest['bb_lower']:
                    confirm = True
                elif strategy == "Mean Reversion Short" and live_price >= latest['bb_upper']:
                    confirm = True
                elif strategy == "Breakout Long" and live_price >= latest['bb_upper']:
                    confirm = True
                elif strategy == "Breakout Short" and live_price <= latest['bb_lower']:
                    confirm = True

                if confirm:
                    alert_msg = (
                        f"📊 *{strategy}* confirmed for *{symbol}*\n"
                        f"Live Price: ${live_price:.2f}\n"
                        f"1H Candle Time: {latest['time']}\n"
                        f"ADX: {latest['adx']:.2f} | RSI: {latest['rsi']:.2f}\n"
                        f"MACD Δ: {latest['macd_diff']:.4f} | BB Width: {latest['bb_width']:.4f}"
                    )
                    send_telegram_alert(alert_msg)
                else:
                    print(f"Live price invalidated {strategy} for {symbol}.")

        except Exception as e:
            print(f"Error processing {symbol}: {e}")


# --- Continuous Loop Every 5 Minutes ---
if __name__ == "__main__":
    while True:
        run_alerts()
        time.sleep(300)  # 5 minutes
