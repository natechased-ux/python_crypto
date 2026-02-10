import requests
import pandas as pd
import numpy as np
import time
import joblib
import subprocess
from datetime import datetime
from pathlib import Path
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight

# === CONFIG ===
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"
COINS = [
    "btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd",
    "wif-usd", "ondo-usd", "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd",
    "syrup-usd", "fartcoin-usd", "aero-usd", "link-usd", "hbar-usd", "aave-usd", "fet-usd",
    "crv-usd", "tao-usd", "avax-usd", "xcn-usd", "uni-usd", "mkr-usd", "toshi-usd",
    "near-usd", "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
    "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd", "qnt-usd",
    "tia-usd", "ip-usd", "pnut-usd", "apt-usd", "ena-usd", "turbo-usd", "bera-usd",
    "pol-usd", "mask-usd", "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "coti-usd",
    "c98-usd", "axs-usd"
]
LOOKAHEAD_CANDLES = 6
ATR_MULT_TARGET = 1.0
MODEL_PATH = Path("breakout_model.pkl")
LAST_TRAIN_FILE = Path("last_train.txt")
cooldowns = {}

# === HELPERS ===
def format_price(value):
    if value >= 100:
        return f"${value:.2f}"
    elif value >= 1:
        return f"${value:.4f}"
    else:
        return f"${value:.6f}"

def send_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Telegram error:", e)

def fetch_data(symbol, granularity='900', limit=100):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    try:
        df = pd.DataFrame(requests.get(url, timeout=10).json(), columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df.sort_values("time").tail(limit).reset_index(drop=True)
    except:
        return None

def compute_indicators(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    df["atr_norm"] = df["atr"] / df["close"]
    bb = BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"]).adx()
    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()
    return df

def multi_tf_confirmation(symbol):
    df_1h = fetch_data(symbol, granularity='3600', limit=60)
    if df_1h is None or df_1h.empty:
        return None
    df_1h = compute_indicators(df_1h)
    latest = df_1h.iloc[-1]
    if latest["adx"] > 20:
        return "Bullish" if latest["close"] > latest["bb_middle"] else "Bearish"
    return None

# === TRAINING ===
def label_breakouts(df):
    labels = []
    for i in range(len(df)):
        if i + LOOKAHEAD_CANDLES >= len(df):
            labels.append(np.nan)
            continue
        atr = df["atr"].iloc[i]
        price_now = df["close"].iloc[i]
        future_max = df["high"].iloc[i+1:i+LOOKAHEAD_CANDLES].max()
        future_min = df["low"].iloc[i+1:i+LOOKAHEAD_CANDLES].min()
        success = int((future_max - price_now) > ATR_MULT_TARGET * atr or (price_now - future_min) > ATR_MULT_TARGET * atr)
        labels.append(success)
    df["label"] = labels
    return df.dropna()

def extract_features(df):
    features = df[["atr_norm", "bb_width", "rsi", "macd_diff", "adx",
                   "volume", "open", "close", "high", "low"]].copy()
    features["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    features["price_above_mid"] = df["close"] - df["bb_middle"]
    features["price_body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    return features.fillna(0)

def train_breakout_model():
    all_data = []
    coins_used = 0
    coins_skipped = 0
    total_positive = 0
    total_negative = 0

    for coin in COINS:
        print(f"📊 Downloading {coin}...")
        df = fetch_data(coin, limit=300)
        if df is None or df.empty:
            print(f"⚠ Skipping {coin} — no data.")
            coins_skipped += 1
            continue

        df = compute_indicators(df)
        df = label_breakouts(df)
        if df.empty:
            print(f"⚠ Skipping {coin} — no valid labels.")
            coins_skipped += 1
            continue

        X = extract_features(df)
        y = df["label"].astype(int)

        pos_count = int((y == 1).sum())
        neg_count = int((y == 0).sum())
        total_positive += pos_count
        total_negative += neg_count

        all_data.append((X, y))
        coins_used += 1

    if not all_data:
        print("❌ No training data available.")
        return False

    X_all = pd.concat([x for x, _ in all_data], ignore_index=True)
    y_all = pd.concat([y for _, y in all_data], ignore_index=True)

    weights = compute_sample_weight(class_weight="balanced", y=y_all)

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_all, y_all, sample_weight=weights)
    joblib.dump(model, MODEL_PATH)

    print("\n✅ Model trained:")
    print(f"   Coins used: {coins_used} / {len(COINS)}")
    print(f"   Breakout examples: {len(X_all)}")
    print(f"   Positive cases: {total_positive}  | Negative cases: {total_negative}")
    print(f"   Skipped coins: {coins_skipped}")

    return True


def retrain_if_needed():
    now = datetime.utcnow()
    if LAST_TRAIN_FILE.exists():
        last_train = datetime.fromisoformat(LAST_TRAIN_FILE.read_text().strip())
        if (now - last_train).days < 7:
            return
    print("🔄 Retraining breakout model...")
    if train_breakout_model():
        LAST_TRAIN_FILE.write_text(now.isoformat())

# === ML PREDICTION ===
def predict_breakout_probability(df):
    if not MODEL_PATH.exists():
        return None
    model = joblib.load(MODEL_PATH)
    features = extract_features(df.iloc[[-1]])
    prob = model.predict_proba(features)[0][1]
    return prob

# === ALERTS ===
def check_breakout(symbol, df, live_price):
    latest = df.iloc[-1]
    vol_mean = df["volume"].rolling(20).mean().iloc[-1]
    atr = latest["atr"]
    dynamic_bb_threshold = max(0.01, df["bb_width"].rolling(50).mean().iloc[-1] * 0.8)
    squeeze = latest["bb_width"] < dynamic_bb_threshold
    price_break_high = latest["close"] > df["high"].rolling(20).max().iloc[-2]
    price_break_low = latest["close"] < df["low"].rolling(20).min().iloc[-2]
    volume_surge = latest["volume"] > 1.5 * vol_mean
    trend_strong = latest["adx"] > 25
    tf_bias = multi_tf_confirmation(symbol)
    if squeeze and volume_surge and trend_strong:
        if price_break_high and tf_bias == "Bullish":
            direction = "Long"
        elif price_break_low and tf_bias == "Bearish":
            direction = "Short"
        else:
            return
        prob = predict_breakout_probability(df)
        if prob is not None and prob < 0.65:
            return
        tp = live_price + 2.0 * atr if direction == "Long" else live_price - 2.0 * atr
        sl = live_price - 1.2 * atr if direction == "Long" else live_price + 1.2 * atr
        msg = (f"🚀 *Breakout Alert* — {symbol.upper()}\n"
               f"Side: {direction}\n"
               f"Live Price: {format_price(live_price)}\n"
               f"TP: {format_price(tp)} | SL: {format_price(sl)}\n"
               f"ML Confidence: {prob*100:.1f}%")
        send_alert(msg)

def check_breakout_imminent(symbol, df, live_price):
    latest = df.iloc[-1]
    atr = latest["atr"]
    dynamic_bb_threshold = max(0.01, df["bb_width"].rolling(50).mean().iloc[-1] * 0.85)
    band_press_upper = latest["close"] > latest["bb_middle"] and latest["close"] > (latest["bb_upper"] - 0.25 * atr)
    band_press_lower = latest["close"] < latest["bb_middle"] and latest["close"] < (latest["bb_lower"] + 0.25 * atr)
    vol_avg_short = df["volume"].rolling(3).mean().iloc[-1]
    vol_avg_med = df["volume"].rolling(10).mean().iloc[-1]
    accumulation = vol_avg_short > 0.9 * vol_avg_med
    macd_hist = latest["macd_diff"]
    momentum_up = macd_hist > 0 and band_press_upper
    momentum_down = macd_hist < 0 and band_press_lower
    tf_bias = multi_tf_confirmation(symbol)
    if latest["bb_width"] < dynamic_bb_threshold and accumulation:
        if momentum_up and tf_bias == "Bullish":
            direction = "Bullish"
        elif momentum_down and tf_bias == "Bearish":
            direction = "Bearish"
        else:
            return
        prob = predict_breakout_probability(df)
        if prob is not None and prob < 0.65:
            return
        tp = live_price + 1.8 * atr if direction == "Bullish" else live_price - 1.8 * atr
        sl = live_price - 1.0 * atr if direction == "Bullish" else live_price + 1.0 * atr
        msg = (f"⚠️ *Breakout Imminent* — {symbol.upper()}\n"
               f"Bias: {direction}\n"
               f"Live Price: {format_price(live_price)}\n"
               f"TP: {format_price(tp)} | SL: {format_price(sl)}\n"
               f"ML Confidence: {prob*100:.1f}%")
        send_alert(msg)

def run_alerts():
    now = datetime.utcnow()
    for symbol in COINS:
        if symbol in cooldowns and (now - cooldowns[symbol]).total_seconds() < 1800:
            continue
        df = fetch_data(symbol)
        if df is None or df.empty:
            continue
        df = compute_indicators(df)
        live_price = df["close"].iloc[-1]
        check_breakout(symbol, df, live_price)
        check_breakout_imminent(symbol, df, live_price)
        cooldowns[symbol] = now

# === MAIN ===
if __name__ == "__main__":
    retrain_if_needed()
    while True:
        run_alerts()
        time.sleep(300)
