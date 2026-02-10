import pandas as pd
import numpy as np
import requests
import time

from datetime import datetime, timedelta, timezone
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import os
# ======================
# CONFIG
# ======================
COINS = [
    "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "DOGE-USD", "ADA-USD", "XLM-USD", "SUI-USD",
    "BCH-USD", "LINK-USD", "HBAR-USD", "LTC-USD", "AVAX-USD", "SHIB-USD", "UNI-USD", "DOT-USD",
    "CRO-USD",  "AAVE-USD", "NEAR-USD", "ETC-USD", "ONDO-USD",
    "APT-USD", "ICP-USD", "ALGO-USD", "ARB-USD", "VET-USD",
    "ATOM-USD", "POL-USD", "BONK-USD", "RENDER-USD",
    "FET-USD", "SEI-USD", "FLR-USD", "FIL-USD", "QNT-USD", "LSETH-USD", "INJ-USD",
    "CRV-USD", "STX-USD", "TIA-USD"
]

ATR_PERIOD = 10
LOOKAHEAD_HOURS=8

THRESH=0.02
ATR_TP=2.5
ATR_SL=1.4


BASE_URL = "https://api.exchange.coinbase.com"
SAVE_PATH = "datasets_macro_training3"
os.makedirs(SAVE_PATH, exist_ok=True)  # ✅ Make sure the folder exists


GRANULARITY = 3600  # 1H candles
START_DATE = datetime(2022, 1, 1, tzinfo=timezone.utc)  # restrict to Jan 1, 2024

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
    atr = AverageTrueRange(df["high"], df["low"], df["close"],window=ATR_PERIOD)
    df["atr"] = atr.average_true_range()
    #df['atr'] = ta.ATR(df['high'], df['low'], df['close'], t=ATR_PERIOD)    
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

def create_labels(df, lookahead=LOOKAHEAD_HOURS, threshold=THRESH):
    """
    Assign labels:
    0 = HOLD
    1 = BUY (TP hit first)
    2 = SELL (SL hit first)
    """

    labels = []

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    for i in range(len(df) - lookahead):
        entry_price = closes[i]
        tp_price = entry_price * (1 + threshold)
        sl_price = entry_price * (1 - threshold)

        future_highs = highs[i + 1:i + 1 + lookahead]
        future_lows = lows[i + 1:i + 1 + lookahead]

        label = 0  # HOLD by default
        for h, l in zip(future_highs, future_lows):
            if h >= tp_price:
                label = 1  # BUY
                break
            elif l <= sl_price:
                label = 2  # SELL
                break

        labels.append(label)

    # Pad last rows with NaNs to keep shape consistent
    labels += [np.nan] * lookahead
    df["label"] = labels

    # Drop rows with no label (incomplete lookahead)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    return df




def generate_tp_sl_labels(df, lookahead=LOOKAHEAD_HOURS, atr_mult_tp=ATR_TP, atr_mult_sl=ATR_SL):
    """
    Calculates realized TP and SL % based on actual future price range.
    Aligned with label logic — positive % only if that level was hit first.
    """

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    tp_pct = []
    sl_pct = []

    for i in range(len(df) - lookahead):
        entry = closes[i]
        atr = df["atr"].values[i]

        tp_level = entry * (1 + atr_mult_tp * atr / entry)
        sl_level = entry * (1 - atr_mult_sl * atr / entry)

        tp_hit = False
        sl_hit = False

        future_highs = highs[i + 1:i + 1 + lookahead]
        future_lows = lows[i + 1:i + 1 + lookahead]

        for j, (h, l) in enumerate(zip(future_highs, future_lows)):
            if h >= tp_level:
                tp_hit = True
                tp_pct.append((tp_level - entry) / entry)
                sl_pct.append(0.0)
                break
            elif l <= sl_level:
                sl_hit = True
                sl_pct.append((entry - sl_level) / entry)
                tp_pct.append(0.0)
                break

        if not tp_hit and not sl_hit:
            tp_pct.append((atr_mult_tp * atr) / entry)
            sl_pct.append((atr_mult_sl * atr) / entry)

    # Pad tail
    tail_len = len(df) - len(tp_pct)
    df["tp_pct"] = pd.Series(tp_pct + [np.nan] * tail_len)
    df["sl_pct"] = pd.Series(sl_pct + [np.nan] * tail_len)

    df.dropna(subset=["tp_pct", "sl_pct"], inplace=True)
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
# MAIN SCRIPT
# ======================
def merge_btc_features(df, btc_csv_path=f"{SAVE_PATH}/BTC_USD.csv"):
    btc_df = pd.read_csv(btc_csv_path)
    btc_df['time'] = pd.to_datetime(btc_df['time'], utc=True)

    # Make sure BTC columns are all prefixed (except time)
    btc_df = btc_df.rename(columns={c: f"btc_{c}" for c in btc_df.columns if c != "time"})

    # Drop btc_return if exists
    btc_cols = [c for c in btc_df.columns if c != "btc_return"]
    btc_df = btc_df[btc_cols]

    return pd.merge_asof(
        df.sort_values("time"),
        btc_df.sort_values("time"),
        on="time",
        direction="backward"
    )

def main():
    # 1️⃣ Fetch & save BTC once
    btc_path = f"{SAVE_PATH}/BTC_USD.csv"
    if not os.path.exists(btc_path):
        print("📥 Fetching BTC-USD dataset...")
        btc_df = fetch_candles("BTC-USD", granularity=GRANULARITY, start_date=START_DATE)
        if btc_df is None or btc_df.empty:
            print("❌ No BTC-USD data, aborting.")
            return
        btc_df = calculate_features(btc_df)
        btc_df = calculate_fib_levels(btc_df, 720, "fib_long")
        btc_df = calculate_fib_levels(btc_df, 360, "fib_med")
        btc_df = calculate_fib_levels(btc_df, 168, "fib_short")
        btc_df.ffill(inplace=True)
        btc_df.dropna(inplace=True)
        btc_df = btc_df.rename(columns={c: f"btc_{c}" for c in btc_df.columns if c != "time"})
        btc_df.to_csv(btc_path, index=False)
        print(f"💾 Saved BTC-USD dataset to {btc_path}")
    else:
        print("✅ Using saved BTC-USD dataset")

    # 2️⃣ Process all coins
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
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        # Merge BTC into all non-BTC coins
        if coin != "BTC-USD":
            df = merge_btc_features(df)

        # Special lookahead for BTC
        if coin == "BTC-USD":
            future_steps = LOOKAHEAD_HOURS
        else:
            future_steps = LOOKAHEAD_HOURS

        df = create_labels(df, lookahead=LOOKAHEAD_HOURS, threshold=THRESH)
        df = generate_tp_sl_labels(df, lookahead=LOOKAHEAD_HOURS)


        save_name = f"{SAVE_PATH}/{coin.replace('-', '_').upper()}.csv"
        df.to_csv(save_name, index=False)
        print(f"💾 Saved {save_name} → {df.shape[0]} rows")

if __name__ == "__main__":
    main()
