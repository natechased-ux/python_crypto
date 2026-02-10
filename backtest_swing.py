import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator

COINS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
         "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
         "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd", "avax-usd", "xcn-usd", "uni-usd",
         "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
         "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd", "qnt-usd", "tia-usd", "ip-usd",
         "pnut-usd", "apt-usd", "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "pyth-usd", "sand-usd",
         "morpho-usd", "mana-usd", "velo-usd", "coti-usd", "c98-usd", "axs-usd"]

BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY = 3600
LOOKBACK_DAYS = 30

TP_MULT = 2.25
SL_MULT = 1
TRADE_SIZE = 1000

def fetch_candles(symbol):
    end = datetime.utcnow()
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = pd.DataFrame()
    while start < end:
        chunk_end = min(start + timedelta(hours=300), end)
        url = f"{BASE_URL}/products/{symbol}/candles"
        params = {
            "start": start.isoformat(),
            "end": chunk_end.isoformat(),
            "granularity": GRANULARITY
        }
        res = requests.get(url, params=params)
        time.sleep(0.2)
        try:
            data = res.json()
            if isinstance(data, list) and data:
                chunk = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
                df = pd.concat([df, chunk])
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        start = chunk_end
    if df.empty:
        return None
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_indicators(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"])
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()
    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()
    return df

def find_swing_high(df, index, window=5):
    if index < window or index + window >= len(df):
        return None
    center = df["high"].iloc[index]
    return all(center > df["high"].iloc[index - i] and center > df["high"].iloc[index + i] for i in range(1, window + 1))

def find_swing_low(df, index, window=5):
    if index < window or index + window >= len(df):
        return None
    center = df["low"].iloc[index]
    return all(center < df["low"].iloc[index - i] and center < df["low"].iloc[index + i] for i in range(1, window + 1))

def backtest(df, symbol):
    trades = []
    swing_highs = []
    swing_lows = []

    for i in range(len(df)):
        if find_swing_high(df, i):
            swing_highs.append((i, df["high"].iloc[i]))
        if find_swing_low(df, i):
            swing_lows.append((i, df["low"].iloc[i]))

    for i in range(50, len(df) - 1):
        row = df.iloc[i]
        if np.isnan(row["adx"]) or np.isnan(row["rsi"]) or np.isnan(row["macd_diff"]):
            continue

        # === LONG ENTRY ===
        if row["adx"] > 20 and row["plus_di"] > row["minus_di"] and row["macd_diff"] > 0 and row["rsi"] > 50:
            recent_swings = [p for j, p in swing_lows if j < i]
            if not recent_swings:
                continue
            sl = recent_swings[-1]
            entry = row["close"]
            risk = entry - sl
            tp = entry + TP_MULT * risk
            sl_new = entry - SL_MULT * risk

            for j in range(i + 1, len(df)):
                high = df["high"].iloc[j]
                low = df["low"].iloc[j]
                if low <= sl_new:
                    trades.append({"symbol": symbol, "side": "long", "entry": entry, "exit": sl_new, "result": "SL", "r": -SL_MULT, "duration": j - i})
                    break
                if high >= tp:
                    trades.append({"symbol": symbol, "side": "long", "entry": entry, "exit": tp, "result": "TP", "r": TP_MULT, "duration": j - i})
                    break

        # === SHORT ENTRY ===
        elif row["adx"] > 20 and row["minus_di"] > row["plus_di"] and row["macd_diff"] < 0 and row["rsi"] < 50:
            recent_swings = [p for j, p in swing_highs if j < i]
            if not recent_swings:
                continue
            sl = recent_swings[-1]
            entry = row["close"]
            risk = sl - entry
            tp = entry - TP_MULT * risk
            sl_new = entry + SL_MULT * risk

            for j in range(i + 1, len(df)):
                high = df["high"].iloc[j]
                low = df["low"].iloc[j]
                if high >= sl_new:
                    trades.append({"symbol": symbol, "side": "short", "entry": entry, "exit": sl_new, "result": "SL", "r": -SL_MULT, "duration": j - i})
                    break
                if low <= tp:
                    trades.append({"symbol": symbol, "side": "short", "entry": entry, "exit": tp, "result": "TP", "r": TP_MULT, "duration": j - i})
                    break
    return trades

# === EXECUTE BACKTEST ===
all_trades = []

for coin in COINS:
    print(f"Processing {coin}...")
    df = fetch_candles(coin)
    if df is None or len(df) < 100:
        print(f"Skipping {coin} (insufficient data)")
        continue
    df = calculate_indicators(df)
    trades = backtest(df, coin)
    all_trades.extend(trades)

# === FINAL OUTPUT ===
df_trades = pd.DataFrame(all_trades)
df_trades["pnl"] = df_trades["r"] * TRADE_SIZE

summary = df_trades.groupby("symbol").agg(
    total_trades=("r", "count"),
    win_rate=("r", lambda x: (x > 0).mean()),
    avg_r=("r", "mean"),
    total_pnl=("pnl", "sum"),
    avg_duration=("duration", "mean")
).sort_values("win_rate", ascending=False)

print("\n===== SUMMARY =====")
print(summary.round(2))

df_trades.to_csv("structure_backtest_optimized_tp_sl.csv", index=False)
