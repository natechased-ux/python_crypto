import requests
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime, timedelta, timezone
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD

# ===== CONFIG =====
COINS = [
    "btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd",
    "sol-usd", "link-usd", "avax-usd", "dot-usd"
]
BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY = 3600  # 1 hour candles
MAX_CANDLES = 300

# Telegram Config
TELEGRAM_TOKEN = "YOUR_TELEGRAM_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

# Load ML Models
ml_model, ml_features = joblib.load("hourly_entry_model.pkl")
tp_model, tp_features = joblib.load("tp_model.pkl")
sl_model, sl_features = joblib.load("sl_model.pkl")
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
        time.sleep(0.4)  # Coinbase rate limit
        current_start = current_end

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time")


def add_indicators(df):
    """Add TA indicators for ML prediction."""
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
def ml_predict_entry(df):
    """
    Predict if the latest candle is a good entry using the trained entry ML model.
    Returns: (prediction, probability)
    """
    latest = df.iloc[-1:][ml_features]
    pred = ml_model.predict(latest)[0]
    prob = ml_model.predict_proba(latest)[0][1]
    return pred, prob


def ml_predict_tp_sl(df, coin):
    """
    Predict optimal TP & SL multipliers using trained regression models.
    Returns: (tp_price, sl_price)
    """
    latest = df.iloc[-1:].copy()

    # Add coin encoding if not already
    if "coin" in latest.columns:
        latest["coin"] = latest["coin"].astype("category").cat.codes
    else:
        latest["coin"] = pd.Series([coin.upper()]).astype("category").cat.codes

    tp_mult = tp_model.predict(latest[tp_features])[0]
    sl_mult = sl_model.predict(latest[sl_features])[0]

    atr = latest["atr"].iloc[0]
    close_price = latest["close"].iloc[0]

    tp_price = close_price + tp_mult * atr
    sl_price = close_price - sl_mult * atr

    return tp_price, sl_price, tp_mult, sl_mult

def send_telegram_message(msg):
    """Send a message to Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram send error: {e}")


def initialize_recent_data():
    """Load ~200 hourly candles at startup."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=200)
    recent_data = {}

    for coin in COINS:
        print(f"📥 Fetching initial recent data for {coin.upper()}...")
        df = fetch_coinbase_candles(coin, start, now)
        if df.empty:
            continue
        df = add_indicators(df)
        df.dropna(inplace=True)
        df["coin"] = coin.upper()
        recent_data[coin] = df
    return recent_data


def update_with_latest_candle(recent_data):
    """Fetch the latest hourly candle and append to in-memory data."""
    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=2)  # ensures we get the last closed candle

    for coin in COINS:
        try:
            latest_df = fetch_coinbase_candles(coin, one_hour_ago, now)
            if latest_df.empty:
                continue

            # Last fully closed candle
            latest_candle = latest_df.iloc[-1:]
            latest_candle = add_indicators(latest_candle)
            latest_candle["coin"] = coin.upper()

            # Append and keep only the last 200 rows
            recent_data[coin] = pd.concat(
                [recent_data[coin], latest_candle]
            ).drop_duplicates(subset=["time"]).tail(200)

        except Exception as e:
            print(f"Error updating {coin.upper()}: {e}")


def run_bot():
    print("\n🚀 Starting Live ML Trading Alert Bot (Hourly) 🚀\n")

    # Step 1: Load ~200 candles per coin
    recent_data = initialize_recent_data()

    while True:
        # Step 2: Append with latest candle
        update_with_latest_candle(recent_data)

        # Step 3: Make predictions
        for coin in COINS:
            try:
                df = recent_data[coin]
                if df.empty:
                    continue

                # ML Entry Prediction
                entry_pred, entry_prob = ml_predict_entry(df)

                if entry_pred == 1:
                    tp_price, sl_price, tp_mult, sl_mult = ml_predict_tp_sl(df, coin)
                    msg = (
                        f"📊 {coin.upper()} ML Trade Signal\n"
                        f"✅ Entry Prob: {entry_prob:.2f}\n"
                        f"🎯 TP: {tp_price:.2f} ({tp_mult:.2f}×ATR)\n"
                        f"🛑 SL: {sl_price:.2f} ({sl_mult:.2f}×ATR)\n"
                        f"ATR: {df.iloc[-1]['atr']:.2f}\n"
                        f"Close: {df.iloc[-1]['close']:.2f}\n"
                        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
                    )
                    send_telegram_message(msg)
                    print(msg)
                else:
                    print(f"{coin.upper()} skipped by ML filter.")

            except Exception as e:
                print(f"Error processing {coin.upper()}: {e}")

        # Step 4: Wait until next hour
        print("\nSleeping until next hourly candle...\n")
        time.sleep(3600)

if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
