import ccxt
import pandas as pd
import requests
import numpy as np
import ta
import time
from datetime import datetime

# ===== CONFIG =====
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"
CHECK_INTERVAL = 60 * 30  # 30 minutes
ALT_LIST = ["ETH/USD", "SOL/USD", "ADA/USD", "MATIC/USD", "AVAX/USD", "DOT/USD", "LINK/USD"]

# ===== TELEGRAM ALERT FUNCTION =====
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=payload)

# ===== FETCH DATA =====
exchange = ccxt.coinbase()

def fetch_ohlcv(symbol, timeframe='21600', limit=200):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['time','open','high','low','close','volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df
    except Exception as e:
        send_telegram(f"⚠ Error fetching {symbol}: {e}")
        return None

# ===== FIB FUNCTION =====
def fib_levels(df):
    swing_high = df['high'].max()
    swing_low = df['low'].min()
    diff = swing_high - swing_low
    return {
        '0.382': swing_high - 0.382 * diff,
        '0.5': swing_high - 0.5 * diff,
        '0.618': swing_high - 0.618 * diff
    }

# ===== CHECK ALT STRENGTH =====
def check_alt_strength():
    strong_count = 0
    strong_alts = []
    for alt in ALT_LIST:
        df = fetch_ohlcv(alt)
        if df is None:
            continue
        df['ema50'] = ta.trend.ema_indicator(df['close'], 50)
        rsi = ta.momentum.rsi(df['close'], 14).iloc[-1]
        if df['close'].iloc[-1] > df['ema50'].iloc[-1] and rsi > 55:
            strong_count += 1
            strong_alts.append(alt.split("/")[0])
    return strong_count, strong_alts

# ===== MAIN MARKET CHECK =====
def check_market():
    btc = fetch_ohlcv('BTC/USD')
    if btc is None:
        return
    
    btc['ema20'] = ta.trend.ema_indicator(btc['close'], 20)
    btc['ema50'] = ta.trend.ema_indicator(btc['close'], 50)
    btc_trend_up = btc['ema20'].iloc[-1] > btc['ema50'].iloc[-1]

    # BTC Fib zone check
    btc_fibs = fib_levels(btc)
    btc_fib_zone = btc_fibs['0.382'] <= btc['close'].iloc[-1] <= btc_fibs['0.618']

    # BTC Dominance from CoinGecko
    btc_d_data = requests.get("https://api.coingecko.com/api/v3/global").json()
    btc_d_val = btc_d_data['data']['market_cap_percentage']['btc']

    # Alt strength
    strong_count, strong_alts = check_alt_strength()

    # ===== DECISION LOGIC =====
    if btc_trend_up and btc_d_val < 46 and btc_fib_zone and strong_count >= 3:
        msg = (
            f"🚀 *ALT SEASON SIGNAL* 🚀\n"
            f"*Time:* {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"*BTC:* ${btc['close'].iloc[-1]:,.0f} (Sideways/Uptrend)\n"
            f"*BTC.D:* {btc_d_val:.2f}% (Dropping)\n"
            f"*Fib Zone:* {btc_fib_zone}\n"
            f"*Strong Alts:* {', '.join(strong_alts)}\n\n"
            f"_Suggested Action:_ Monitor for altcoin breakouts."
        )
    elif btc_d_val > 46 and strong_count < 3:
        msg = (
            f"⚠️ *ALT SEASON EXIT* ⚠️\n"
            f"*Time:* {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"*BTC:* ${btc['close'].iloc[-1]:,.0f} (Trend Shift)\n"
            f"*BTC.D:* {btc_d_val:.2f}% (Rising)\n"
            f"_Suggested Action:_ Reduce altcoin exposure."
        )
    else:
        msg = (
            f"📊 *No Clear Alt Season Signal*\n"
            f"*Time:* {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"*BTC:* ${btc['close'].iloc[-1]:,.0f}\n"
            f"*BTC.D:* {btc_d_val:.2f}%\n"
            f"Strong Alts: {', '.join(strong_alts) if strong_alts else 'None'}"
        )

    send_telegram(msg)

# ===== LOOP =====
if __name__ == "__main__":
    send_telegram("🤖 Alt Season Alert Bot Started ✅")
    while True:
        try:
            check_market()
        except Exception as e:
            send_telegram(f"❌ Bot Error: {e}")
        time.sleep(CHECK_INTERVAL)
