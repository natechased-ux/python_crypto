import pandas as pd
import numpy as np
import requests
import time
import torch
import torch.nn as nn
import telegram
import pytz
from datetime import datetime, timedelta
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import schedule
from sklearn.preprocessing import StandardScaler
import warnings

# === CONFIG ===
COINS = ["BTC-USD", "ETH-USD", "XRP-USD"]  # Add your 1h trained coins
SEQ_LEN = 60
CONF_THRESHOLD = 0.70


CHAT_ID = "7967738614"
BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"

#TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
#CHAT_ID = "-4916911067"

#personal chat id:"7967738614"
#channel chat id:
#"-4916911067"


MODELS_PATH = "models_entry_1h"
TP_SL_MODELS_PATH = "models_tp_sl_1h"

bot = telegram.Bot(token=BOT_TOKEN)

# === FEATURE LIST (must match your training) ===
FEATURES = [
    'low', 'high', 'open', 'close', 'volume', 'rsi', 'adx', 'plus_di', 'minus_di',
    'macd_diff', 'ema10', 'ema10_slope', 'ema20', 'ema20_slope', 'ema50', 'ema50_slope',
    'ema200', 'ema200_slope', 'atr', 'vol_change', 'body_wick_ratio', 'above_ema200',
    'fib_long_0', 'fib_long_236', 'fib_long_382', 'fib_long_5', 'fib_long_618', 'fib_long_786', 'fib_long_1',
    'fib_med_0', 'fib_med_236', 'fib_med_382', 'fib_med_5', 'fib_med_618', 'fib_med_786', 'fib_med_1',
    'fib_short_0', 'fib_short_236', 'fib_short_382', 'fib_short_5', 'fib_short_618', 'fib_short_786', 'fib_short_1',
    'btc_close', 'btc_return', 'coin_return', 'rel_strength', 'btc_dominance'
]

# === MODEL CLASSES ===
class LSTMEntryModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, output_dim=3):
        super(LSTMEntryModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMTpSlModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, output_dim=2):
        super(LSTMTpSlModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
# === FETCH CANDLES FROM COINBASE ===
def fetch_candles(symbol, granularity=3600, limit=300):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    params = {"granularity": granularity}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print(f"❌ Failed to fetch {symbol}")
        return None
    data = r.json()
    if not data:
        return None
    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.sort_values("time", inplace=True)
    return df

def add_indicators(df):
    # ATR
    atr_indicator = AverageTrueRange(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=14
    )
    df['atr'] = atr_indicator.average_true_range()

    # RSI
    rsi_indicator = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi_indicator.rsi()

    # ADX
    adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx_indicator.adx()
    df['plus_di'] = adx_indicator.adx_pos()
    df['minus_di'] = adx_indicator.adx_neg()

    # MACD
    macd = MACD(close=df['close'])
    df['macd_diff'] = macd.macd_diff()

    return df

# === CALCULATE TECHNICAL FEATURES ===
def calculate_features(df):
    df["rsi"] = compute_rsi(df["close"], 14)
    df["adx"] = compute_adx(df, 14)
    df["plus_di"], df["minus_di"] = compute_di(df, 14)
    df["macd_diff"] = compute_macd(df)
    for period in [10, 20, 50, 200]:
        df[f"ema{period}"] = df["close"].ewm(span=period).mean()
        df[f"ema{period}_slope"] = df[f"ema{period}"].diff()
    df["atr"] = compute_atr(df, 14)
    df["vol_change"] = df["volume"].pct_change()
    df["body_wick_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    df["above_ema200"] = (df["close"] > df["ema200"]).astype(int)
    # Fibs — can use static or dynamic
    return df

# === SIMPLE INDICATOR FUNCTIONS ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift())
    lc = abs(df["low"] - df["close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["close"].ewm(span=fast).mean()
    ema_slow = df["close"].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd - signal_line

def compute_adx(df, period=14):
    plus_dm = df["high"].diff()
    minus_dm = df["low"].diff().abs()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / df["atr"])
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / df["atr"])
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1/period).mean()

def compute_di(df, period=14):
    plus_dm = df["high"].diff()
    minus_dm = df["low"].diff().abs()
    atr = compute_atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
    return plus_di, minus_di
# === LOAD ENTRY MODEL FOR COIN ===
def load_entry_model_for_coin(symbol):
    model_path = os.path.join(MODELS_PATH, f"{symbol.replace('-', '_')}_entry.pt")
    if not os.path.exists(model_path):
        print(f"❌ Entry model missing for {symbol}")
        return None
    checkpoint = torch.load(model_path, map_location="cpu")
    model = LSTMEntryModel(input_dim=len(FEATURES))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# === LOAD TP/SL MODEL FOR COIN ===
def load_tp_sl_model_for_coin(symbol):
    model_path = os.path.join(TP_SL_MODELS_PATH, f"{symbol.replace('-', '_')}_tp_sl.pt")
    if not os.path.exists(model_path):
        print(f"❌ TP/SL model missing for {symbol}")
        return None
    checkpoint = torch.load(model_path, map_location="cpu")
    model = LSTMTpSlModel(input_dim=len(FEATURES))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# === SEND ALERT TO TELEGRAM ===
def send_alert(symbol, direction, conf, entry, tp, sl, candle_time):
    try:
        msg = (
            f"🚨 *{symbol}* {direction}\n"
            f"📅 Candle: {candle_time}\n"
            f"📈 Entry: {entry:.4f}\n"
            f"🎯 TP: {tp:.4f}\n"
            f"🛑 SL: {sl:.4f}\n"
            f"🤖 Confidence: {conf:.1%}"
        )
        bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
        print(f"✅ Sent alert for {symbol}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")
def check_ai_signals():
    print(f"\n🔍 Checking AI signals at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    for symbol in COINS:
        print(f"\n📌 Processing {symbol} ...")

        df = fetch_candles(symbol)
        if df is None or len(df) < SEQ_LEN + 1:
            print(f"❌ Not enough data for {symbol}")
            continue
        df = add_indicators(df)

        df = calculate_features(df)
        df.dropna(inplace=True)

        # Last closed candle (avoid partial candle)
        last_candle_time = df["time"].iloc[-1]
        if last_candle_time > datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(hours=1):
            df = df.iloc[:-1]

        # Prepare sequence for model
        seq = df[FEATURES].tail(SEQ_LEN).values
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        # Predict entry direction
        entry_model = load_entry_model_for_coin(symbol)
        if entry_model is None:
            continue
        with torch.no_grad():
            probs = torch.softmax(entry_model(seq), dim=1).numpy()[0]
            short_prob, no_prob, long_prob = probs

        print(f"Probs → Short: {short_prob:.2%}, No Trade: {no_prob:.2%}, Long: {long_prob:.2%}")

        if long_prob > CONF_THRESHOLD:
            direction = "LONG"
            conf = long_prob
        elif short_prob > CONF_THRESHOLD:
            direction = "SHORT"
            conf = short_prob
        else:
            print("→ No high-confidence trade.")
            continue

        # Predict TP/SL
        tp_sl_model = load_tp_sl_model_for_coin(symbol)
        if tp_sl_model is None:
            continue
        with torch.no_grad():
            tp_pct, sl_pct = tp_sl_model(seq).numpy()[0]

        last_price = df["close"].iloc[-1]
        if direction == "LONG":
            tp_price = last_price * (1 + abs(tp_pct))
            sl_price = last_price * (1 - abs(sl_pct))
        else:
            tp_price = last_price * (1 - abs(tp_pct))
            sl_price = last_price * (1 + abs(sl_pct))

        candle_time_str = df["time"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")

        send_alert(symbol, direction, conf, last_price, tp_price, sl_price, candle_time_str)
if __name__ == "__main__":
    print("🚀 Super Swinger Charlie started... Running hourly checks.")

    while True:
        try:
            check_ai_signals()
        except Exception as e:
            print(f"❌ Error in check_ai_signals: {e}")

        # Wait until 3 minutes after next full hour
        now = datetime.utcnow()
        next_run = (now + timedelta(hours=1)).replace(minute=3, second=0, microsecond=0)
        sleep_seconds = (next_run - datetime.utcnow()).total_seconds()
        print(f"⏳ Sleeping {floor(sleep_seconds/60)} minutes until next run...")
        time.sleep(max(0, sleep_seconds))
