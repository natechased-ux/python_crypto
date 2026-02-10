import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from math import isnan

# --- CONFIG ---
BOT_TOKEN = '8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw'
CHAT_ID = '7967738614'
COINS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
         "doge-usd", "wif-usd", "ondo-usd", "magic-usd", "ape-usd",
         "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
         "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd",
         "avax-usd", "xcn-usd", "uni-usd", "toshi-usd", "near-usd", "algo-usd",
         "trump-usd", "bch-usd", "inj-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
         "dot-usd", "popcat-usd", "arb-usd", "icp-usd", "tia-usd", "ip-usd",
         "pnut-usd", "apt-usd", "vet-usd", "ena-usd", "turbo-usd", "bera-usd",
         "pol-usd", "mask-usd", "ach-usd", "pyth-usd", "sand-usd", "morpho-usd",
         "mana-usd", "velo-usd", "coti-usd", "axs-usd"]
TIMEFRAME = '3600'  # 1H candles
COOLDOWN_MINUTES = 30

HEADERS = {'User-Agent': 'BreakoutHunterBot'}
alerted = {}

# --- HELPERS ---
def get_candles(symbol, granularity=21600, limit=300):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values("time")
    return df

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=payload)

def bollinger_bands(df, period=20):
    df['ma'] = df['close'].rolling(period).mean()
    df['std'] = df['close'].rolling(period).std()
    df['upper'] = df['ma'] + 2 * df['std']
    df['lower'] = df['ma'] - 2 * df['std']
    df['bandwidth'] = df['upper'] - df['lower']
    return df

def calculate_atr(df, period=14):
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(period).mean()
    return df

def check_breakout(df, symbol):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    recent_bandwidths = df['bandwidth'].iloc[-28:]  # past 7 days of 1H candles
    min_bandwidth = recent_bandwidths.min()

    if latest['bandwidth'] > min_bandwidth * 1.05:
        return  # Only trigger when it breaks out from the lowest squeeze

    now = datetime.utcnow()
    if symbol in alerted and now < alerted[symbol] + timedelta(minutes=COOLDOWN_MINUTES):
        return

    direction = None
    if latest['close'] > latest['upper']:
        direction = "LONG"
    elif latest['close'] < latest['lower']:
        direction = "SHORT"

    if direction:
        atr = latest['ATR']
        if isnan(atr) or atr == 0:
            return

        if direction == "LONG":
            sl = latest['close'] - 1.5 * atr
            tp = latest['close'] + 3 * atr
        else:
            sl = latest['close'] + 1.5 * atr
            tp = latest['close'] - 3 * atr

        message = (
            f"🚨 Breakout Alert: ${symbol}\n"
            f"Direction: {direction}\n"
            f"Entry: {latest['close']:.6f}\n"
            f"SL: {sl:.6f} | TP: {tp:.6f}\n"
            f"Reason: Bollinger Band Squeeze Breakout\n"
            f"Time: {latest['time'].strftime('%Y-%m-%d %H:%M UTC')}"
        )
        send_telegram(message)
        alerted[symbol] = now

# --- MAIN LOOP ---
while True:
    for coin in COINS:
        try:
            df = get_candles(coin, granularity=int(TIMEFRAME))
            df = bollinger_bands(df)
            df = calculate_atr(df)
            check_breakout(df, coin)
        except Exception as e:
            print(f"Error with {coin}: {e}")
        time.sleep(1)
    time.sleep(60)
