import requests
import pandas as pd
import time
from ta.momentum import StochRSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
from datetime import datetime
import os

# === CONFIG ===
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
GOLDEN_ZONE_MARGIN = 0.1
LOOKBACK_HOURS = 168
ATR_MULT_TP = 2.0
ATR_MULT_SL = 1.5
POSITION_SIZE = 2500
COOLDOWN_MINUTES = 120
LOG_FILE = "rigo_live_trade_log.csv"

BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"

last_alert_time = {}

def get_candles(symbol, granularity):
    url = f'https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}'
    r = requests.get(url)
    if r.status_code != 200:
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

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, json=payload)

def should_alert(symbol):
    now = time.time()
    if symbol not in last_alert_time:
        last_alert_time[symbol] = 0
    if now - last_alert_time[symbol] >= COOLDOWN_MINUTES * 60:
        last_alert_time[symbol] = now
        return True
    return False

def log_trade(data):
    df = pd.DataFrame([data])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)

def check_signal(symbol):
    df_1h = get_candles(symbol, 3600)
    df_1d = get_candles(symbol, 86400)

    if df_1h is None or df_1d is None or len(df_1h) < LOOKBACK_HOURS + 20:
        return

    df_1h['stoch_k'] = StochRSIIndicator(df_1h['close']).stochrsi_k()
    df_1h['stoch_d'] = StochRSIIndicator(df_1h['close']).stochrsi_d()
    df_1h['atr'] = AverageTrueRange(df_1h['high'], df_1h['low'], df_1h['close'], window=14).average_true_range()

    df_1d['ema200'] = EMAIndicator(df_1d['close'], window=200).ema_indicator()
    df_1d.set_index('time', inplace=True)

    row = df_1h.iloc[-1]
    price = row['close']
    time_now = row['time']

    recent_df = df_1h.iloc[-LOOKBACK_HOURS:]
    swing_high = recent_df['high'].max()
    swing_low = recent_df['low'].min()
    fibs = calc_fibs(swing_high, swing_low)
    golden_min = fibs['0.66'] * (1 - GOLDEN_ZONE_MARGIN)
    golden_max = fibs['0.618'] * (1 + GOLDEN_ZONE_MARGIN)

    k = row['stoch_k']
    d = row['stoch_d']
    crossed_up = k > d and k < 40
    crossed_down = k < d and k > 60

    daily_row = df_1d[df_1d.index <= time_now]
    if daily_row.empty:
        return
    ema200 = daily_row['ema200'].iloc[-1]
    is_bull = df_1d['close'].iloc[-1] > ema200
    is_bear = not is_bull

    direction = None
    entry = price
    atr_val = row['atr']
    if pd.isna(atr_val):
        return

    if is_bull and golden_min <= price <= golden_max and crossed_up:
        direction = 'LONG'
        tp = entry + ATR_MULT_TP * atr_val
        sl = entry - ATR_MULT_SL * atr_val
    elif is_bear and golden_min <= price <= golden_max and crossed_down:
        direction = 'SHORT'
        tp = entry - ATR_MULT_TP * atr_val
        sl = entry + ATR_MULT_SL * atr_val
    else:
        return

    if should_alert(symbol):
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        message = (
            f"\n🚨 *{symbol.upper()}* {direction} Signal\n"
            f"Entry: {entry:.4f}\nTP: {tp:.4f} | SL: {sl:.4f}\n"
            f"ATR: {atr_val:.4f}\nStoch RSI: K={k:.2f}, D={d:.2f}\n"
            f"200 EMA Trend: {'Bullish' if is_bull else 'Bearish'}\n"
            f"Golden Zone: {golden_min:.2f}–{golden_max:.2f}\n"
            f"Size: ${POSITION_SIZE}\nTime: {timestamp} UTC"
        )
        send_telegram_message(message)

        log_trade({
            'symbol': symbol,
            'direction': direction,
            'entry_time': timestamp,
            'entry_price': entry,
            'tp': tp,
            'sl': sl,
            'atr': atr_val,
            'stoch_k': k,
            'stoch_d': d,
            'ema200': ema200,
            'resolved': False,
            'result': ''
        })

def monitor_open_trades():
    if not os.path.exists(LOG_FILE):
        return

    df = pd.read_csv(LOG_FILE)
    df = df[df['tp'].notna() & df['sl'].notna()]
    df['resolved'] = df.get('resolved', False)

    for index, row in df[df['resolved'] != True].iterrows():
        symbol = row['symbol']
        direction = row['direction']
        entry_price = row['entry_price']
        tp = row['tp']
        sl = row['sl']

        df_latest = get_candles(symbol, 3600)
        if df_latest is None or df_latest.empty:
            continue

        last_row = df_latest.iloc[-1]
        high = last_row['high']
        low = last_row['low']

        hit_tp = hit_sl = False
        if direction == 'LONG':
            hit_tp = high >= tp
            hit_sl = low <= sl
        elif direction == 'SHORT':
            hit_tp = low <= tp
            hit_sl = high >= sl

        if hit_tp or hit_sl:
            result = 'WIN' if hit_tp else 'LOSS'
            df.at[index, 'resolved'] = True
            df.at[index, 'result'] = result
            df.to_csv(LOG_FILE, index=False)

            send_telegram_message(f"✅ {symbol.upper()} {direction} closed: {result}\nEntry: {entry_price:.2f}\nTP: {tp:.2f}, SL: {sl:.2f}")

# === MAIN LOOP ===
if __name__ == "__main__":
    while True:
        for coin in COINS:
            try:
                check_signal(coin)
            except Exception as e:
                print(f"Error on {coin}: {e}")
        monitor_open_trades()
        print("Cycle complete. Sleeping 5 minutes.")
        time.sleep(300)
