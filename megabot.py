import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD

# Config
COINS = ["btc-usd", "eth-usd", "sol-usd"]
BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY = 3600  # 1 hour
MAX_CANDLES = 300
LOOKAHEAD_HOURS = 6  # Label target within next N hours
ATR_THRESHOLD = 1.0  # Profitability threshold in ATR multiples

def fetch_coinbase_candles(pair, start, end, granularity=GRANULARITY):
    """Fetch historical candles from Coinbase API with chunking."""
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
        time.sleep(0.4)  # Respect API rate limits
        current_start = current_end

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time")

def add_indicators(df):
    """Add TA indicators and volatility features."""
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


def create_target(df, lookahead_hours=LOOKAHEAD_HOURS, atr_threshold=ATR_THRESHOLD):
    """
    Label = 1 if future high within lookahead_hours exceeds current close by atr_threshold × ATR.
    Else 0.
    """
    df["future_high"] = df["high"].shift(-lookahead_hours).rolling(lookahead_hours).max()
    df["target"] = (
        (df["future_high"] - df["close"]) > (atr_threshold * df["atr"])
    ).astype(int)
    return df
def build_historical_dataset():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * 2)

    all_data = []

    for coin in COINS:
        print(f"\nFetching {coin.upper()} historical data...")
        df = fetch_coinbase_candles(coin, start, end)

        if df.empty:
            print(f"⚠ No data for {coin.upper()}")
            continue

        # Add indicators
        df = add_indicators(df)
        df.dropna(inplace=True)

        # Create target labels
        df = create_target(df)
        df.dropna(inplace=True)

        # Add coin column
        df["coin"] = coin.upper()

        # Append to combined dataset
        all_data.append(df)

        # Save individual CSV
        df.to_csv(f"{coin.replace('-', '_')}_historical_dataset.csv", index=False)
        print(f"✅ Saved {coin.upper()} dataset to CSV.")

    # Combine all coins
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv("combined_historical_dataset.csv", index=False)
        print("\n📁 Saved combined dataset: combined_historical_dataset.csv")
    else:
        print("\n⚠ No historical data available to save.")
if __name__ == "__main__":
    build_historical_dataset()

