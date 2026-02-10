import requests
import pandas as pd
import numpy as np
import ta
import time
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import pytz
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ======================
# CONFIGURATION
# ======================
COINS = [
    "bitcoin", "ethereum", "solana", "cardano", "ripple", "dogecoin", "polkadot",
    "matic-network",  # Polygon ID in CoinGecko
    "avalanche-2", "tron", "chainlink", "litecoin", "stellar", "cosmos",
    "internet-computer", "vechain", "the-graph", "aptos", "near", "uniswap",
    "algorand", "optimism", "arbitrum"
]
VS_CURRENCY = "usd"
SEQ_LEN = 12  # last 12 hours sequence
FEATURES = ["price", "rsi", "ema_fast", "ema_slow", "macd", "return"]
MODEL_PATH = "crypto_lstm_model.h5"

# Telegram credentials
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"

model = None
scaler = StandardScaler()

# ======================
# FETCH DATA (FREE API)
# ======================
def fetch_coin_data(coin, days=90):  # Max 90 days for free API
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
        params = {"vs_currency": VS_CURRENCY, "days": str(days)}  # removed interval=hourly
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            print(f"⚠️ API error {r.status_code} for {coin}")
            return None

        data = r.json()
        if "prices" not in data or not data["prices"]:
            print(f"⚠️ No price data for {coin}")
            return None

        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')

        # Indicators
        df["rsi"] = ta.momentum.rsi(df["price"], window=14, fillna=True)
        df["ema_fast"] = ta.trend.ema_indicator(df["price"], window=12, fillna=True)
        df["ema_slow"] = ta.trend.ema_indicator(df["price"], window=26, fillna=True)
        df["macd"] = df["ema_fast"] - df["ema_slow"]
        df["return"] = df["price"].pct_change().fillna(0)

        # Target
        df["future_return"] = df["price"].shift(-4) / df["price"] - 1
        df["target"] = (df["future_return"] > 0.005).astype(int)

        df.fillna(0, inplace=True)
        return df
    except Exception as e:
        print(f"⚠️ Error fetching {coin}: {e}")
        return None

def create_sequences(df):
    df_scaled = df.copy()
    df_scaled[FEATURES] = scaler.fit_transform(df[FEATURES])
    X, y = [], []
    for i in range(len(df_scaled) - SEQ_LEN):
        seq_x = df_scaled[FEATURES].iloc[i:i+SEQ_LEN].values
        seq_y = df_scaled["target"].iloc[i+SEQ_LEN]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# ======================
# TRAIN MODEL
# ======================
def train_model():
    global model
    print(f"📈 Retraining model at {datetime.now()}...")
    X_all, y_all = [], []

    for coin in COINS:
        df = fetch_coin_data(coin, days=90)
        if df is None:
            continue
        Xc, yc = create_sequences(df)
        if Xc.size > 0:
            X_all.append(Xc)
            y_all.append(yc)
        else:
            print(f"⚠️ No sequences for {coin}")
        time.sleep(1.2)  # avoid rate limits

    if not X_all:
        print("❌ No valid training data found.")
        return

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    y_cat = to_categorical(y, num_classes=2)

    # LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(SEQ_LEN, len(FEATURES)), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X, y_cat, validation_split=0.2, epochs=15, batch_size=64, callbacks=[es])

    model.save(MODEL_PATH)
    print("✅ Model retrained and saved.")

# ======================
# TELEGRAM ALERT
# ======================
def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, data=payload)
    except:
        print("⚠️ Failed to send Telegram message.")

# ======================
# SCAN MARKET
# ======================
def ai_scan():
    global model
    if model is None:
        try:
            model = load_model(MODEL_PATH)
        except:
            print("❌ No model found. Run training first.")
            return

    results = []
    for coin in COINS:
        df = fetch_coin_data(coin, days=30)
        if df is None or len(df) < SEQ_LEN:
            continue
        seq = create_sequences(df)[0][-1:]
        prob_up = model.predict(seq, verbose=0)[0][1]
        price = df.iloc[-1]["price"]
        results.append((coin, prob_up, price))
        time.sleep(2.5) 

    if not results:
        print("⚠️ No predictions made.")
        return

    ranked = sorted(results, key=lambda x: x[1], reverse=True)
    top_pick = ranked[0]
    msg = f"🤖 AI Crypto Prediction Alert\nTop Pick: {top_pick[0].upper()} @ ${top_pick[2]:.2f}\n" \
          f"Probability UP: {top_pick[1]*100:.1f}%\n\nTop 3 Predictions:\n"
    for coin, prob, price in ranked[:3]:
        msg += f"{coin.upper()} | {prob*100:.1f}% UP | ${price:.2f}\n"

    print(msg)
    send_telegram_message(msg)

# ======================
# SCHEDULER
# ======================
if __name__ == "__main__":
    scheduler = BackgroundScheduler(timezone=pytz.UTC)
    scheduler.add_job(train_model, 'interval', weeks=1)  # retrain weekly
    scheduler.add_job(ai_scan, 'interval', hours=1)      # scan hourly
    scheduler.start()

    print("🚀 AI Crypto Scanner running...")
    train_model()  # Initial training
    ai_scan()      # Initial scan

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
