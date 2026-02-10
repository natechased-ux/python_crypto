import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# === CONFIG ===
COINS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
         "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
         "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd", "avax-usd", "xcn-usd", "uni-usd",
         "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
         "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd", "qnt-usd", "tia-usd", "ip-usd",
         "pnut-usd", "apt-usd", "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "pyth-usd", "sand-usd",
         "morpho-usd", "mana-usd", "coti-usd", "c98-usd", "axs-usd"]
BASE_URL = "https://api.exchange.coinbase.com"
LOOKBACK_DAYS = 900           # ~2.5 years
GRANULARITY = 21600           # 6-hour candles
LOOKAHEAD_CANDLES = 3         # 12 hours forward look

# === FETCH CANDLES ===
def fetch_candles(symbol, granularity=GRANULARITY, lookback_days=LOOKBACK_DAYS):
    """Fetch historical candles from Coinbase in chunks to avoid 300-candle limit."""
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    df = pd.DataFrame()

    while start < end:
        chunk_end = min(start + timedelta(hours=(granularity / 3600) * 300), end)
        params = {
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "granularity": granularity
        }
        url = f"{BASE_URL}/products/{symbol}/candles"
        res = requests.get(url, params=params)
        time.sleep(0.25)
        try:
            data = res.json()
            if isinstance(data, list) and data:
                chunk = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
                df = pd.concat([df, chunk], ignore_index=True)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        start = chunk_end

    if df.empty:
        return None

    # Clean & sort
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.drop_duplicates(subset="time", inplace=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# === FEATURE CALCULATION ===
def calculate_features(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    adx = ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()
    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()

    # EMAs + slopes
    for p in [10, 20, 50, 200]:
        df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()
        df[f"ema{p}_slope"] = df[f"ema{p}"].diff()

    # ATR
    atr = AverageTrueRange(df["high"], df["low"], df["close"])
    df["atr"] = atr.average_true_range()

    # Volume change
    df["vol_change"] = df["volume"].pct_change()

    # Candle body/wick ratio
    df["body_wick_ratio"] = np.where(
        (df["high"] - df["low"]) != 0,
        abs(df["close"] - df["open"]) / (df["high"] - df["low"]),
        0
    )

    # Trend flag
    df["above_ema200"] = (df["close"] > df["ema200"]).astype(int)
    return df

# === LABEL CREATION ===
def label_trades(df, lookahead=LOOKAHEAD_CANDLES):
    labels = []
    mfe_list = []
    mae_list = []

    for i in range(len(df) - lookahead):
        entry = df["close"].iloc[i]
        future_prices = df["close"].iloc[i+1:i+1+lookahead]

        # Max Favorable Excursion (MFE) in each direction
        mfe_long = (future_prices.max() - entry) / entry
        mfe_short = (entry - future_prices.min()) / entry

        # Max Adverse Excursion (MAE) in each direction
        mae_long = (future_prices.min() - entry) / entry
        mae_short = (entry - future_prices.max()) / entry

        # Determine label based on stronger favorable move
        if mfe_long > mfe_short and mfe_long > 0:
            label = 1  # Long setup
            mfe_list.append(mfe_long)
            mae_list.append(abs(mae_long))
        elif mfe_short > mfe_long and mfe_short > 0:
            label = -1  # Short setup
            mfe_list.append(mfe_short)
            mae_list.append(abs(mae_short))
        else:
            label = 0  # No trade signal
            mfe_list.append(0)
            mae_list.append(0)

        labels.append(label)

    # Pad last rows
    labels += [0] * lookahead
    mfe_list += [0] * lookahead
    mae_list += [0] * lookahead

    df["label"] = labels
    df["mfe_pct"] = mfe_list
    df["mae_pct"] = mae_list
    return df

# === MAIN LOOP ===
all_data = []
for coin in COINS:
    print(f"📡 Fetching {coin}...")
    df = fetch_candles(coin)
    if df is None:
        print(f"⚠ No data for {coin}")
        continue
    df = calculate_features(df)
    df = label_trades(df)
    df["symbol"] = coin
    all_data.append(df)

# === SAVE DATASET ===
if all_data:
    dataset = pd.concat(all_data, ignore_index=True)
    dataset.dropna(inplace=True)
    dataset.to_csv("lstm_training_dataset_3candle.csv", index=False)
    print(f"✅ Dataset saved: {dataset.shape[0]} rows, {dataset.shape[1]} columns → lstm_training_dataset_3candle.csv")
else:
    print("⛔ No data generated.")
