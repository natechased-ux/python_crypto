import pandas as pd
import numpy as np
import requests
import joblib
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD
from lightgbm import LGBMClassifier
from datetime import datetime
from pathlib import Path

# === CONFIG ===
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
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def fetch_historical(symbol, granularity=900, limit=300):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    try:
        df = pd.DataFrame(requests.get(url).json(), columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.sort_values("time")
        return df
    except:
        return None

def compute_indicators(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    bb = BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"]).adx()
    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()
    return df

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
        success = 0
        if (future_max - price_now) > ATR_MULT_TARGET * atr:
            success = 1
        elif (price_now - future_min) > ATR_MULT_TARGET * atr:
            success = 1
        labels.append(success)
    df["label"] = labels
    return df.dropna()

def extract_features(df):
    features = df[[
        "atr", "bb_width", "rsi", "macd_diff", "adx",
        "volume", "open", "close", "high", "low"
    ]].copy()
    features["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    features["price_above_mid"] = df["close"] - df["bb_middle"]
    features["price_body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    return features.fillna(0)

def train_breakout_model():
    all_data = []
    for coin in COINS:
        print(f"Downloading {coin}...")
        df = fetch_historical(coin)
        if df is None or df.empty:
            continue
        df = compute_indicators(df)
        df = label_breakouts(df)
        X = extract_features(df)
        y = df["label"].astype(int)
        all_data.append((X, y))
    
    if not all_data:
        print("❌ No training data available.")
        return False
    
    X_all = pd.concat([x for x, _ in all_data], ignore_index=True)
    y_all = pd.concat([y for _, y in all_data], ignore_index=True)

    print(f"Training model on {len(X_all)} samples...")
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_all, y_all)

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved: {MODEL_PATH}")
    return True

if __name__ == "__main__":
    train_breakout_model()
