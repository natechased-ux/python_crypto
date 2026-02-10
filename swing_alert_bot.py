
import pandas as pd
import requests
import numpy as np
import time
import os
from datetime import datetime
import telegram

# --- Configuration ---
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"
COINS = [
    "btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
    "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
    "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd", "avax-usd", "xcn-usd", "uni-usd",
    "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
    "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd", "qnt-usd", "tia-usd", "ip-usd",
    "pnut-usd", "apt-usd", "vet-usd", "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
    "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "velo-usd", "coti-usd", "axs-usd"
]
INTERVALS = {'1D': 86400, '4H': 14400, '1H': 3600}
BASE_URL = "https://api.exchange.coinbase.com/products"
LOG_FILE = "swing_trade_alert_log.csv"

bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

# --- Indicator Functions ---
def get_candles(symbol, granularity, limit=300):
    url = f"{BASE_URL}/{symbol}/candles?granularity={granularity}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return pd.DataFrame()
    data = resp.json()
    df = pd.DataFrame(data, columns=['time','low','high','open','close','volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df.sort_values('time')

def ema(series, period=200):
    return series.ewm(span=period, adjust=False).mean()

def macd(df, fast=12, slow=26, signal=9):
    fast_ema = df['close'].ewm(span=fast, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def send_alert(symbol, direction, entry, sl, tp, confidence):
    msg = (
        f"🚨 *{direction.upper()} SIGNAL* on {symbol}\n"
        f"*Entry:* {entry:.4f}\n"
        f"*Stop Loss:* {sl:.4f}\n"
        f"*Take Profit:* {tp:.4f}\n"
        f"*Confidence:* {confidence}/4\n"
        f"#swingtrade"
    )
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode=telegram.ParseMode.MARKDOWN)

def log_alert(symbol, direction, entry, sl, tp, confidence, timestamp):
    row = {
        "timestamp": timestamp,
        "symbol": symbol,
        "direction": direction,
        "entry": round(entry, 4),
        "stop_loss": round(sl, 4),
        "take_profit": round(tp, 4),
        "confidence": confidence
    }
    log_df = pd.DataFrame([row])
    if os.path.exists(LOG_FILE):
        log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        log_df.to_csv(LOG_FILE, index=False)

def evaluate_signals():
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for coin in COINS:
        try:
            df_d = get_candles(coin.upper(), INTERVALS['1D'])
            df_4h = get_candles(coin.upper(), INTERVALS['4H'])
            df_1h = get_candles(coin.upper(), INTERVALS['1H'])

            if df_d.empty or df_4h.empty or df_1h.empty:
                continue

            df_d['ema200'] = ema(df_d['close'], 200)
            trend_up = df_d['close'].iloc[-1] > df_d['ema200'].iloc[-1]
            trend_down = df_d['close'].iloc[-1] < df_d['ema200'].iloc[-1]

            macd_line, signal_line, hist = macd(df_4h)
            macd_cross_up = macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]
            macd_cross_down = macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]

            rsi_1h = rsi(df_1h)
            rsi_4h = rsi(df_4h)

            rsi_ok_long = rsi_1h.iloc[-1] < 70 and rsi_4h.iloc[-1] < 70
            rsi_ok_short = rsi_1h.iloc[-1] > 30 and rsi_4h.iloc[-1] > 30

            atr_val = atr(df_1h).iloc[-1]
            price = df_1h['close'].iloc[-1]

            if trend_up and macd_cross_up and rsi_ok_long:
                entry = price
                sl = entry - 1.5 * atr_val
                tp = entry + 3 * atr_val
                confidence = sum([trend_up, macd_cross_up, rsi_ok_long])
                send_alert(coin.upper(), "long", entry, sl, tp, confidence)
                log_alert(coin.upper(), "long", entry, sl, tp, confidence, now)

            if trend_down and macd_cross_down and rsi_ok_short:
                entry = price
                sl = entry + 1.5 * atr_val
                tp = entry - 3 * atr_val
                confidence = sum([trend_down, macd_cross_down, rsi_ok_short])
                send_alert(coin.upper(), "short", entry, sl, tp, confidence)
                log_alert(coin.upper(), "short", entry, sl, tp, confidence, now)

        except Exception as e:
            print(f"Error processing {coin.upper()}: {e}")

if __name__ == "__main__":
    evaluate_signals()
