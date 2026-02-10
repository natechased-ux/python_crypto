import asyncio
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from ta.volatility import AverageTrueRange
from telegram import Bot

# --- CONFIG ---
TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = Bot(token=TOKEN)

symbols = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd",
           "syrup-usd","fartcoin-usd", "aero-usd", "link-usd","hbar-usd",
           "aave-usd", "fet-usd", "crv-usd","tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd",
           "bch-usd", "inj-usd", "pepe-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
           "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
           "qnt-usd", "tia-usd", "ip-usd" , "pnut-usd", "apt-usd", "vet-usd",
           "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "coti-usd",
           "axs-usd"]
fib_levels = [0.382, 0.5, 0.618, 0.66, 0.786]
volume_lookback = 30
atr_length = 14

# --- UTILITIES ---
def fetch_historical_data(symbol, granularity=3600):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    response = requests.get(url)
    if response.status_code == 200:
        columns = ["time", "low", "high", "open", "close", "volume"]
        data = pd.DataFrame(response.json(), columns=columns)
        data = data.sort_values(by="time").reset_index(drop=True)
        return data
    else:
        raise Exception(f"Failed to fetch data for {symbol}: {response.status_code}")


def is_volume_spike(data):
    recent_vol = data['volume'].iloc[-1]
    threshold = data['volume'].iloc[-volume_lookback:].quantile(0.9)
    return recent_vol >= threshold


def detect_reversal_candle(data):
    o, h, l, c = data['open'].iloc[-1], data['high'].iloc[-1], data['low'].iloc[-1], data['close'].iloc[-1]
    prev_o, prev_c = data['open'].iloc[-2], data['close'].iloc[-2]
    body = abs(c - o)
    range_ = h - l
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    # Bullish engulfing
    if c > o and prev_c < prev_o and c > prev_o and o < prev_c:
        return "bullish engulfing"
    # Hammer
    elif body < range_ * 0.3 and lower_wick > body * 2:
        return "hammer"
    # Doji
    elif body < range_ * 0.1:
        return "doji"
    # Bearish engulfing
    elif c < o and prev_c > prev_o and c < prev_o and o > prev_c:
        return "bearish engulfing"
    # Shooting star
    elif body < range_ * 0.3 and upper_wick > body * 2:
        return "shooting star"
    return None


def is_fib_retrace(data):
    recent = data.iloc[-30:]
    swing_high = recent['high'].max()
    swing_low = recent['low'].min()
    retracement = data['close'].iloc[-1]
    diff = swing_high - swing_low

    for level in fib_levels:
        target = swing_high - (diff * level)
        if abs(retracement - target) / target <= 0.003:  # within 0.3%
            return level
    return None


def calculate_atr(data):
    atr = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=atr_length)
    return atr.average_true_range().iloc[-1]


def format_price(p):
    return f"{p:,.2f}" if p > 1 else f"{p:.6f}"


def detect_rafaela_signal(symbol, data):
    candle = detect_reversal_candle(data)
    if not candle:
        return None

    fib = is_fib_retrace(data)
    if not fib:
        return None

    if not is_volume_spike(data):
        return None

    close = data['close'].iloc[-1]
    atr = calculate_atr(data)
    direction = "LONG" if "bullish" in candle or candle in ["hammer", "doji"] else "SHORT"
    entry_low = close * 0.9975
    entry_high = close * 1.0025
    sl = close - atr if direction == "LONG" else close + atr
    tps = [close + (atr * m) if direction == "LONG" else close - (atr * m) for m in [0.5, 1.0, 1.5, 2.0, 2.5]]

    coin = symbol.upper().replace("-USD", "/USDT")
    fib_note = f"Fib 0.{int(fib*1000):03d}" if fib else ""
    rationale = f"{fib_note} + {candle} + volume spike"

    msg = f"""
🚨 {coin} – {direction}
Leverage: 50×
Entry Zone: {format_price(entry_low)} – {format_price(entry_high)}
TPs: {' / '.join([format_price(tp) for tp in tps])}
SL: {format_price(sl)}
Trade Type: Swing (1–3 days)
Reason: {rationale}
""".strip()

    return msg


async def send_to_telegram(message):
    await bot.send_message(chat_id=CHAT_ID, text=message)


async def hourly_updates():
    while True:
        try:
            for symbol in symbols:
                data = fetch_historical_data(symbol, granularity=3600)
                message = detect_rafaela_signal(symbol, data)
                if message:
                    await send_to_telegram(message)
        except Exception as e:
            print(f"Error: {str(e)}")

        print(f"⏰ Rafaela scan complete – {datetime.now()}")
        await asyncio.sleep(300)


if __name__ == "__main__":
    asyncio.run(hourly_updates())
