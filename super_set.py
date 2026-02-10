import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta, timezone
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# ======================
# CONFIG
# ======================
COINS = [
    "btc-usd", "eth-usd", "xrp-usd"
]
BASE_URL = "https://api.exchange.coinbase.com"
SAVE_PATH = "datasets_macro_training"
GRANULARITY = 3600  # 1H candles
START_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)  # restrict to Jan 1, 2024

# ======================
# FETCH HISTORICAL CANDLES
# ======================
def fetch_candles(symbol, granularity=GRANULARITY, start_date=START_DATE):
    """Fetch historical Coinbase candles in chunks (max 300)"""
    end = datetime.now(timezone.utc)
    start = start_date
    df = pd.DataFrame()

    while start < end:
        chunk_end = min(start + timedelta(seconds=granularity * 300), end)
        params = {
            "start": start.isoformat(),
            "end": chunk_end.isoformat(),
            "granularity": granularity
        }
        url = f"{BASE_URL}/products/{symbol}/candles"
        r = requests.get(url, params=params)
        time.sleep(0.25)
        try:
            data = r.json()
            if isinstance(data, list) and data:
                chunk = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
                df = pd.concat([df, chunk], ignore_index=True)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        start = chunk_end

    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.drop_duplicates(subset="time", inplace=True)
    df.sort_values("time", inplace=True)
    return df

# ======================
# CALCULATE TECHNICAL FEATURES
# ======================
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



# ======================
# CALCULATE FIBONACCI LEVELS
# ======================
def add_fibonacci_levels(df, lookback=720):  # 30 days of 1h candles
    fib_ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    for ratio in fib_ratios:
        df[f'fib_{ratio}'] = np.nan

    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback:i]
        swing_high = window['high'].max()
        swing_low = window['low'].min()

        for ratio in fib_ratios:
            df.loc[df.index[i], f'fib_{ratio}'] = swing_high - (swing_high - swing_low) * ratio

    return df

import pandas as pd
import numpy as np

LOOKAHEAD_HOURS = 4  # for 1-hour candles, that's 6 future candles

import numpy as np

import numpy as np

def create_labels(df, future_steps=4, threshold=0.002):
    """
    Create trade labels:
    1 → LONG
    2 → SHORT
    0 → NO TRADE
    future_steps: how many candles ahead to check
    threshold: percent change trigger
    """
    labels = []
    closes = df["close"].values

    for i in range(len(df)):
        if i + future_steps >= len(df):
            labels.append(np.nan)
            continue

        future_close = closes[i + future_steps]
        current_close = closes[i]
        change = (future_close - current_close) / current_close

        if change > threshold:
            labels.append(1)  # LONG
        elif change < -threshold:
            labels.append(2)  # SHORT
        else:
            labels.append(0)  # NO TRADE

    df["label"] = labels
    df.dropna(inplace=True)
    df["label"] = df["label"].astype(int)
    return df


def generate_tp_sl_labels(df, lookahead=4, atr_mult_tp=2.0, atr_mult_sl=1.0):
    """
    Generate realistic TP/SL percentage labels for training.
    - lookahead: hours to look ahead from the entry candle
    - atr_mult_tp: ATR multiple for take profit
    - atr_mult_sl: ATR multiple for stop loss
    """

    tp_pcts = np.full(len(df), np.nan)
    sl_pcts = np.full(len(df), np.nan)

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atrs = df['atr'].values

    for i in range(len(df) - lookahead):
        entry_price = closes[i]
        atr = atrs[i]

        tp_level = entry_price * (1 + atr_mult_tp * atr / entry_price)
        sl_level = entry_price * (1 - atr_mult_sl * atr / entry_price)

        tp_hit = False
        sl_hit = False

        # Simulate forward to see which hits first
        for j in range(1, lookahead + 1):
            if highs[i + j] >= tp_level:
                tp_hit = True
                break
            if lows[i + j] <= sl_level:
                sl_hit = True
                break

        # Always store % distances
        tp_pcts[i] = (tp_level - entry_price) / entry_price
        sl_pcts[i] = (sl_level - entry_price) / entry_price

        # If neither hit, store best observed move
        if not tp_hit and not sl_hit:
            tp_pcts[i] = (highs[i+1:i+lookahead+1].max() - entry_price) / entry_price
            sl_pcts[i] = (lows[i+1:i+lookahead+1].min() - entry_price) / entry_price

    df['tp_pct'] = tp_pcts
    df['sl_pct'] = sl_pcts
    return df



def calculate_fib_levels(df, lookback_hours, prefix):
    """
    Calculate Fibonacci retracement levels dynamically based on last major swing
    over a specified lookback period.

    Args:
        df (pd.DataFrame): DataFrame with 'high' and 'low' columns.
        lookback_hours (int): Number of hours for the swing window.
        prefix (str): Prefix for column names (e.g., 'fib_long').

    Returns:
        pd.DataFrame: DataFrame with added Fibonacci level columns.
    """
    # Rolling high/low over lookback
    swing_high = df['high'].rolling(window=lookback_hours, min_periods=1).max()
    swing_low = df['low'].rolling(window=lookback_hours, min_periods=1).min()

    diff = swing_high - swing_low

    # Add Fib retracement levels
    df[f"{prefix}_0"] = swing_low
    df[f"{prefix}_236"] = swing_high - diff * 0.236
    df[f"{prefix}_382"] = swing_high - diff * 0.382
    df[f"{prefix}_5"] = swing_high - diff * 0.5
    df[f"{prefix}_618"] = swing_high - diff * 0.618
    df[f"{prefix}_786"] = swing_high - diff * 0.786
    df[f"{prefix}_1"] = swing_high

    return df



# ======================
# LOAD BTC.D FROM TRADINGVIEW
# ======================
def load_btcd_csv(path):
    btcd = pd.read_csv(path)
    btcd.columns = [c.strip().lower() for c in btcd.columns]
    if "time" in btcd.columns:
        btcd["time"] = pd.to_datetime(btcd["time"], utc=True, errors="coerce")
    elif "timestamp" in btcd.columns:
        btcd["time"] = pd.to_datetime(btcd["timestamp"], utc=True, errors="coerce")

    # Resample 2H → 1H (forward fill)
    btcd = btcd.set_index("time").sort_index()
    if "close" in btcd.columns:
        btcd = btcd[["close"]].rename(columns={"close": "btc_dominance"})
    else:
        raise ValueError("BTC.D CSV missing 'close' column")

    btcd = btcd.resample("1H").ffill().reset_index()
    return btcd

# ======================
# MAIN SCRIPT
# ======================
def main():
    btcd = load_btcd_csv(r"C:\Users\natec\CRYPTOCAP_BTCD120.csv")
    btc_df = fetch_candles("BTC-USD", granularity=GRANULARITY, start_date=START_DATE)
    btc_df = calculate_features(btc_df)
    btc_df['btc_return'] = btc_df['close'].pct_change()
    btc_df = btc_df[['time', 'close', 'btc_return']].rename(columns={'close': 'btc_close'})

  # replace path
    print(f"✅ BTC.D loaded: {btcd.shape[0]} rows")

    for coin in COINS:
        print(f"\n📌 Processing {coin.upper()} ...")
        df = fetch_candles(coin, granularity=GRANULARITY, start_date=START_DATE)
        if df is None or df.empty:
            print(f"❌ No data for {coin}")
            continue

        df = calculate_features(df)
        df = calculate_fib_levels(df, 720, "fib_long")
        df = calculate_fib_levels(df, 360, "fib_med")
        df = calculate_fib_levels(df, 168, "fib_short")

        df = df.merge(btc_df, on='time', how='left')
        df['coin_return'] = df['close'].pct_change()
        df['rel_strength'] = df['coin_return'] - df['btc_return'] 
        df = df.merge(btcd, on="time", how="left")
        
        df.fillna(method="ffill", inplace=True)
        df.dropna(inplace=True)

        df = create_labels(df, future_steps=4, threshold=0.002)
        df = generate_tp_sl_labels(df, lookahead=4, atr_mult_tp=2.0, atr_mult_sl=1.0)


        save_name = f"{SAVE_PATH}/{coin.replace('-', '_').upper()}.csv"
        df.to_csv(save_name, index=False)
        print(f"💾 Saved {save_name} → {df.shape[0]} rows")

if __name__ == "__main__":
    main()
