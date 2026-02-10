import pandas as pd
import numpy as np
import requests
import time
import math
from datetime import datetime, timedelta, timezone
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD

# Config
COINS = ["btc-usd", "eth-usd", "sol-usd"]
BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY = 3600
MAX_CANDLES = 300

# TP/SL ranges for optimization
TP_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]
SL_MULTIPLIERS = [0.8, 1.0, 1.2, 1.5, 2.0]

LOOKAHEAD_HOURS = 24  # Max time to check for TP/SL hit
def fetch_coinbase_candles(pair, start, end, granularity=GRANULARITY):
    """Fetch historical candles from Coinbase API in chunks."""
    all_data = []
    chunk_seconds = granularity * MAX_CANDLES
    current_start = start

    while current_start < end:
        current_end = min(current_start + timedelta(seconds=chunk_seconds), end)
        url = f"{BASE_URL}/products/{pair}/candles"
        params = {
            "granularity": granularity,
            "start": current_start.isoformat(),
            "end": current_end.isoformat()
        }
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"Error fetching {pair}: {e}")
            break

        if not data:
            break

        all_data.extend(data)
        time.sleep(0.4)  # Avoid hitting rate limits
        current_start = current_end

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time")


def add_indicators(df):
    """Add TA indicators for TP/SL modeling."""
    if df.empty:
        return df

    # RSI
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

    # Bollinger Bands
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    # ATR
    df["atr"] = AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    # MACD
    macd = MACD(
        df["close"], window_slow=26, window_fast=12, window_sign=9
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # ADX
    adx = ADXIndicator(
        df["high"], df["low"], df["close"], window=14
    )
    df["adx"] = adx.adx()

    # Volatility %
    df["volatility_pct"] = df["atr"] / df["close"]

    return df
def find_best_tp_sl_for_trade(df, entry_index):
    """
    Given a candle index (entry point), test all TP/SL combinations
    and return the one with the highest profit.
    """
    entry_price = df.iloc[entry_index]["close"]
    atr = df.iloc[entry_index]["atr"]

    # Ensure ATR is valid
    if np.isnan(atr) or atr <= 0:
        return None, None

    best_tp = None
    best_sl = None
    best_profit = -float("inf")

    # Check future candles up to LOOKAHEAD_HOURS
    future_df = df.iloc[entry_index+1 : entry_index+LOOKAHEAD_HOURS+1]
    if future_df.empty:
        return None, None

    for tp_mult in TP_MULTIPLIERS:
        for sl_mult in SL_MULTIPLIERS:
            tp_price = entry_price + tp_mult * atr
            sl_price = entry_price - sl_mult * atr
            profit = 0
            exit_found = False

            # Simulate bar-by-bar
            for _, row in future_df.iterrows():
                if row["high"] >= tp_price:
                    profit = tp_mult * atr
                    exit_found = True
                    break
                elif row["low"] <= sl_price:
                    profit = -sl_mult * atr
                    exit_found = True
                    break

            # If neither hit, close at final candle's price
            if not exit_found:
                profit = (future_df.iloc[-1]["close"] - entry_price) / atr

            # Keep best combo
            if profit > best_profit:
                best_profit = profit
                best_tp = tp_mult
                best_sl = sl_mult

    return best_tp, best_sl
# Configurable: how many months of data to train on
TRAIN_MONTHS = 6  

def build_tp_sl_dataset_from_csv():
    """
    Build TP/SL training dataset using already saved historical CSVs from Phase 1.
    Limits to recent months for faster training.
    """
    all_rows = []
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=30 * TRAIN_MONTHS)

    for coin in COINS:
        csv_file = f"{coin.replace('-', '_')}_historical_dataset.csv"

        try:
            print(f"\nLoading {csv_file}...")
            df = pd.read_csv(csv_file, parse_dates=["time"])
        except FileNotFoundError:
            print(f"⚠ File {csv_file} not found. Skipping {coin.upper()}.")
            continue

        # Filter to recent months only
        df = df[df["time"] >= cutoff_date]

        # Ensure ATR is present and valid
        if "atr" not in df.columns:
            print(f"⚠ ATR not found in {csv_file}, skipping {coin.upper()}.")
            continue

        df.dropna(inplace=True)

        # Loop through each possible trade entry point
        for i in range(len(df) - LOOKAHEAD_HOURS - 1):
            best_tp, best_sl = find_best_tp_sl_for_trade(df, i)
            if best_tp is None or best_sl is None:
                continue

            row = {
                "coin": coin.upper(),
                "rsi": df.iloc[i]["rsi"],
                "bb_high": df.iloc[i]["bb_high"],
                "bb_low": df.iloc[i]["bb_low"],
                "atr": df.iloc[i]["atr"],
                "macd": df.iloc[i]["macd"],
                "macd_signal": df.iloc[i]["macd_signal"],
                "adx": df.iloc[i]["adx"],
                "volatility_pct": df.iloc[i]["volatility_pct"],
                "volume": df.iloc[i]["volume"],
                "best_tp_mult": best_tp,
                "best_sl_mult": best_sl
            }
            all_rows.append(row)

    # Save dataset
    if all_rows:
        dataset = pd.DataFrame(all_rows)
        dataset.to_csv("tp_sl_training_dataset.csv", index=False)
        print(f"\n✅ TP/SL training dataset saved: tp_sl_training_dataset.csv")
    else:
        print("\n⚠ No data collected.")


import joblib
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_tp_sl_models():
    df = pd.read_csv("tp_sl_training_dataset.csv")

    df["coin"] = df["coin"].astype("category").cat.codes
    features = [
        "coin", "rsi", "bb_high", "bb_low", "atr", "macd",
        "macd_signal", "adx", "volatility_pct", "volume"
    ]

    # --- Train TP Model ---
    y_tp = df["best_tp_mult"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], y_tp, test_size=0.2, random_state=42
    )
    tp_model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    tp_model.fit(X_train, y_train)
    y_pred_tp = tp_model.predict(X_test)
    
    print("\nTP Model RMSE:", math.sqrt(mean_squared_error(y_test, y_pred_tp)))


    # --- Train SL Model ---
    y_sl = df["best_sl_mult"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], y_sl, test_size=0.2, random_state=42
    )
    sl_model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    sl_model.fit(X_train, y_train)
    y_pred_sl = sl_model.predict(X_test)
    print("\nTP Model RMSE:", math.sqrt(mean_squared_error(y_test, y_pred_tp)))

    # Save models
    joblib.dump((tp_model, features), "tp_model.pkl")
    joblib.dump((sl_model, features), "sl_model.pkl")
    print("\n✅ TP & SL models saved: tp_model.pkl, sl_model.pkl")


# ====== STEP 3: RUN ALL ======
if __name__ == "__main__":
    build_tp_sl_dataset_from_csv()
    train_tp_sl_models()
