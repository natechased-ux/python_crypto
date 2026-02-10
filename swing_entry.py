import pandas as pd
import numpy as np
import requests
import time
import torch
import torch.nn as nn
import telegram
from datetime import datetime, timedelta
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import schedule
import threading
import os

# === CONFIG ===
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
COINS = [
    "btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd",
    "sol-usd", "wif-usd", "ondo-usd", "sei-usd", "magic-usd", "ape-usd",
    "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
    "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd",
    "avax-usd", "xcn-usd", "uni-usd", "mkr-usd", "toshi-usd", "near-usd",
    "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
    "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
    "qnt-usd", "tia-usd", "ip-usd", "pnut-usd", "apt-usd", "ena-usd", "turbo-usd",
    "bera-usd", "pol-usd", "mask-usd", "pyth-usd", "sand-usd", "morpho-usd",
    "mana-usd", "coti-usd", "c98-usd", "axs-usd"
]

BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY = 21600  # 6-hour candles
SEQ_LEN = 60         # LSTM sequence length
CONF_THRESHOLD = 0.70 # Confidence threshold
TP_MULTIPLIER = 0.75 # Take-profit multiplier

# === LSTM MODEL CLASS ===
class LSTMTrader(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.fc(h[:, -1, :])
        return out

# === LOAD MODEL & SCALER ===
checkpoint = torch.load("lstm_crypto_model.pt", map_location=torch.device('cpu'))
scaler = checkpoint['scaler']
FEATURES = [
    'rsi', 'adx', 'plus_di', 'minus_di', 'macd_diff',
    'ema10', 'ema20', 'ema50', 'ema200',
    'ema10_slope', 'ema20_slope', 'ema50_slope', 'ema200_slope',
    'atr', 'vol_change', 'body_wick_ratio', 'above_ema200'
]
model = LSTMTrader(input_size=len(FEATURES))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# === FETCH CANDLES ===
def fetch_candles(symbol, lookback_days=60):
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    df = pd.DataFrame()

    while start < end:
        chunk_end = min(start + timedelta(hours=(GRANULARITY / 3600) * 300), end)
        params = {
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "granularity": GRANULARITY
        }
        url = f"{BASE_URL}/products/{symbol}/candles"
        res = requests.get(url, params=params)
        time.sleep(0.25)
        try:
            data = res.json()
            if isinstance(data, list) and data:
                chunk = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
                df = pd.concat([df, chunk], ignore_index=True)
        except:
            pass
        start = chunk_end

    if df.empty:
        return None
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.drop_duplicates(subset="time", inplace=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# === FEATURE CALCULATION ===
def calculate_features(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    adx = ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()
    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()

    for p in [10, 20, 50, 200]:
        df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()
        df[f"ema{p}_slope"] = df[f"ema{p}"].diff()

    atr = AverageTrueRange(df["high"], df["low"], df["close"])
    df["atr"] = atr.average_true_range()

    df["vol_change"] = df["volume"].pct_change()
    df["body_wick_ratio"] = np.where(
        (df["high"] - df["low"]) != 0,
        abs(df["close"] - df["open"]) / (df["high"] - df["low"]),
        0
    )
    df["above_ema200"] = (df["close"] > df["ema200"]).astype(int)
    return df

# === SWING HIGH/LOW DETECTION ===
def find_swing_high(df, index, window=2):
    if index < window or index + window >= len(df):
        return False
    center = df["high"].iloc[index]
    return all(center > df["high"].iloc[index - i] and center > df["high"].iloc[index + i] for i in range(1, window + 1))

def find_swing_low(df, index, window=2):
    if index < window or index + window >= len(df):
        return False
    center = df["low"].iloc[index]
    return all(center < df["low"].iloc[index - i] and center < df["low"].iloc[index + i] for i in range(1, window + 1))

# === AI SIGNAL CHECK ===
def check_ai_signals():
    print(f"🔍 Checking AI signals at {datetime.utcnow().isoformat()} UTC")

    for symbol in COINS:
        df = fetch_candles(symbol, lookback_days=60)
        if df is None or len(df) < SEQ_LEN:
            continue

        df = calculate_features(df)
        df.dropna(inplace=True)

        latest_seq = df[FEATURES].tail(SEQ_LEN).values
        latest_seq = scaler.transform(latest_seq)
        latest_seq = torch.tensor(latest_seq, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            outputs = model(latest_seq)
            probs = torch.softmax(outputs, dim=1).numpy()[0]
            short_prob, no_prob, long_prob = probs

        last_price = df["close"].iloc[-1]

        # === LONG ===
        if long_prob > CONF_THRESHOLD:
            # Find swing low for SL
            swings_low = [df["low"].iloc[i] for i in range(len(df)) if find_swing_low(df, i)]
            if swings_low:
                sl = swings_low[-1]
                risk = last_price - sl
                tp = last_price + TP_MULTIPLIER * risk
                send_alert(symbol, "LONG", long_prob, last_price, tp, sl)

        # === SHORT ===
        elif short_prob > CONF_THRESHOLD:
            swings_high = [df["high"].iloc[i] for i in range(len(df)) if find_swing_high(df, i)]
            if swings_high:
                sl = swings_high[-1]
                risk = sl - last_price
                tp = last_price - TP_MULTIPLIER * risk
                send_alert(symbol, "SHORT", short_prob, last_price, tp, sl)

# === ALERTING ===
def send_alert(symbol, side, conf, entry, tp, sl):
    msg = (
        f"🤖 AI SIGNAL: {side} on {symbol.upper()}\n"
        f"Confidence: {conf:.2%}\n"
        f"Entry: {entry:.4f} | TP: {tp:.4f} | SL: {sl:.4f}"
    )
    try:
        bot.send_message(chat_id=CHAT_ID, text=msg)
        print(f"✅ Sent alert: {msg}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

# === SCHEDULER ===
check_ai_signals()
schedule.every(6).hours.do(check_ai_signals)

print("🚀 AI Swing Trading Bot Running...")
while True:
    schedule.run_pending()
    time.sleep(5)
