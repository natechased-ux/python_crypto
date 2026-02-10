import requests, pandas as pd, numpy as np, time, joblib, subprocess
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
    "c98-usd", "axs-usd"]  # shortened for example
LOOKAHEAD_CANDLES = 6
ATR_MULT_TARGET = 1.0
MODEL_PATH_BREAKOUT = Path("breakout_model.pkl")
MODEL_PATH_IMMINENT = Path("imminent_breakout_model.pkl")
LAST_TRAIN_FILE = Path("last_train.txt")
cooldowns = {}

BREAKOUT_THRESHOLD = 0.65
IMMINENT_THRESHOLD = 0.55

def format_price(v):
    return f"${v:.2f}" if v >= 100 else f"${v:.4f}" if v >= 1 else f"${v:.6f}"

def fetch_data(symbol, granularity='900', limit=100):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    try:
        df = pd.DataFrame(requests.get(url, timeout=10).json(),
                          columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df.sort_values("time").tail(limit).reset_index(drop=True)
    except:
        return None

def compute_indicators(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    df["atr_norm"] = df["atr"] / df["close"]
    bb = BollingerBands(df["close"])
    df["bb_upper"], df["bb_lower"], df["bb_middle"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"]).adx()
    df["macd_diff"] = MACD(df["close"]).macd_diff()
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

def label_breakouts(df):
    labels = []
    for i in range(len(df)):
        if i + LOOKAHEAD_CANDLES >= len(df): labels.append(np.nan); continue
        atr = df["atr"].iloc[i]
        price_now = df["close"].iloc[i]
        future_max = df["high"].iloc[i+1:i+LOOKAHEAD_CANDLES].max()
        future_min = df["low"].iloc[i+1:i+LOOKAHEAD_CANDLES].min()
        labels.append(int((future_max - price_now) > ATR_MULT_TARGET * atr or
                          (price_now - future_min) > ATR_MULT_TARGET * atr))
    df["label"] = labels
    return df.dropna()

def label_imminent(df):
    labels = []
    for i in range(len(df)):
        if i + LOOKAHEAD_CANDLES >= len(df): labels.append(np.nan); continue
        if df["bb_width"].iloc[i] < df["bb_width"].rolling(50).mean().iloc[i] * 0.85:
            atr = df["atr"].iloc[i]
            price_now = df["close"].iloc[i]
            future_max = df["high"].iloc[i+1:i+LOOKAHEAD_CANDLES].max()
            future_min = df["low"].iloc[i+1:i+LOOKAHEAD_CANDLES].min()
            labels.append(int((future_max - price_now) > ATR_MULT_TARGET * atr or
                              (price_now - future_min) > ATR_MULT_TARGET * atr))
        else:
            labels.append(0)
    df["label"] = labels
    return df.dropna()

def extract_features(df):
    f = df[["atr_norm", "bb_width", "rsi", "macd_diff", "adx",
            "volume", "open", "close", "high", "low"]].copy()
    f["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    f["price_above_mid"] = df["close"] - df["bb_middle"]
    f["price_body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    return f.fillna(0)

def train_model(label_func, save_path, label_name):
    all_data, coins_used, coins_skipped, pos_total, neg_total = [], 0, 0, 0, 0
    for coin in COINS:
        print(f"📊 Downloading {coin}...")
        df = fetch_data(coin, limit=300)
        if df is None or df.empty:
            coins_skipped += 1; continue
        df = compute_indicators(df)
        df = label_func(df)
        if df.empty:
            coins_skipped += 1; continue
        X, y = extract_features(df), df["label"].astype(int)
        pos_total += (y == 1).sum(); neg_total += (y == 0).sum()
        all_data.append((X, y)); coins_used += 1
    if not all_data: return False
    X_all = pd.concat([x for x, _ in all_data], ignore_index=True)
    y_all = pd.concat([y for _, y in all_data], ignore_index=True)
    weights = compute_sample_weight(class_weight="balanced", y=y_all)
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8)
    model.fit(X_all, y_all, sample_weight=weights)
    joblib.dump(model, save_path)
    print(f"\n✅ {label_name} Model trained:")
    print(f"   Coins used: {coins_used} / {len(COINS)}")
    print(f"   Examples: {len(X_all)} | Positive: {pos_total} | Negative: {neg_total}")
    print(f"   Skipped coins: {coins_skipped}")
    return True

def retrain_if_needed():
    now = datetime.utcnow()
    if LAST_TRAIN_FILE.exists():
        last_train = datetime.fromisoformat(LAST_TRAIN_FILE.read_text().strip())
        if (now - last_train).days < 7: return
    print("🔄 Retraining both models...")
    train_model(label_breakouts, MODEL_PATH_BREAKOUT, "Breakout")
    train_model(label_imminent, MODEL_PATH_IMMINENT, "Imminent Breakout")
    LAST_TRAIN_FILE.write_text(now.isoformat())

def predict_probability(df, model_path):
    if not model_path.exists(): return None
    model = joblib.load(model_path)
    return model.predict_proba(extract_features(df.iloc[[-1]]))[0][1]

def send_alert(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

def check_breakout(symbol, df, live_price):
    latest, vol_mean, atr = df.iloc[-1], df["volume"].rolling(20).mean().iloc[-1], df["atr"].iloc[-1]
    squeeze = latest["bb_width"] < df["bb_width"].rolling(50).mean().iloc[-1] * 0.8
    volume_surge = latest["volume"] > 1.5 * vol_mean
    trend_strong = latest["adx"] > 25
    tf_bias = multi_tf_confirmation(symbol)
    if squeeze and volume_surge and trend_strong:
        if latest["close"] > df["high"].rolling(20).max().iloc[-2] and tf_bias == "Bullish":
            direction = "Long"
        elif latest["close"] < df["low"].rolling(20).min().iloc[-2] and tf_bias == "Bearish":
            direction = "Short"
        else: return
        prob = predict_probability(df, MODEL_PATH_BREAKOUT)
        if prob and prob >= BREAKOUT_THRESHOLD:
            tp = live_price + 2 * atr if direction == "Long" else live_price - 2 * atr
            sl = live_price - 1.2 * atr if direction == "Long" else live_price + 1.2 * atr
            send_alert(f"🚀 *Breakout Alert* — {symbol.upper()}\nSide: {direction}\nPrice: {format_price(live_price)}\nTP: {format_price(tp)} | SL: {format_price(sl)}\nConfidence: {prob*100:.1f}%")

def check_breakout_imminent(symbol, df, live_price):
    latest, atr = df.iloc[-1], df["atr"].iloc[-1]
    band_press_upper = latest["close"] > latest["bb_middle"] and latest["close"] > (latest["bb_upper"] - 0.25 * atr)
    band_press_lower = latest["close"] < latest["bb_middle"] and latest["close"] < (latest["bb_lower"] + 0.25 * atr)
    accumulation = df["volume"].rolling(3).mean().iloc[-1] > 0.9 * df["volume"].rolling(10).mean().iloc[-1]
    tf_bias = multi_tf_confirmation(symbol)
    if latest["bb_width"] < df["bb_width"].rolling(50).mean().iloc[-1] * 0.85 and accumulation:
        if band_press_upper and tf_bias == "Bullish":
            direction = "Bullish"
        elif band_press_lower and tf_bias == "Bearish":
            direction = "Bearish"
        else: return
        prob = predict_probability(df, MODEL_PATH_IMMINENT)
        if prob and prob >= IMMINENT_THRESHOLD:
            tp = live_price + 1.8 * atr if direction == "Bullish" else live_price - 1.8 * atr
            sl = live_price - 1.0 * atr if direction == "Bullish" else live_price + 1.0 * atr
            send_alert(f"⚠️ *Breakout Imminent* — {symbol.upper()}\nBias: {direction}\nPrice: {format_price(live_price)}\nTP: {format_price(tp)} | SL: {format_price(sl)}\nConfidence: {prob*100:.1f}%")


def run_alerts():
    now = datetime.utcnow()
    for symbol in COINS:
        if symbol in cooldowns and (now - cooldowns[symbol]).total_seconds() < 1800: continue
        df = fetch_data(symbol)
        if df is None or df.empty: continue
        df = compute_indicators(df)
        live_price = df["close"].iloc[-1]
        check_breakout(symbol, df, live_price)
        check_breakout_imminent(symbol, df, live_price)
        cooldowns[symbol] = now

if __name__ == "__main__":
    retrain_if_needed()
    while True:
        run_alerts()
        time.sleep(300)
