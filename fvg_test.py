import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.volatility import AverageTrueRange
import time

# --- CONFIG ---
symbols = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "doge-usd"]
capital_per_trade = 2500
atr_window = 14
base_url = "https://api.exchange.coinbase.com"

# --- HELPERS ---
def fetch_candles(symbol, days=3, granularity=900):
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    params = {
        "start": start.isoformat() + "Z",  # Add Zulu time identifier
        "end": end.isoformat() + "Z",
        "granularity": granularity
    }
    url = f"{base_url}/products/{symbol}/candles"
    response = requests.get(url, params=params)
    if response.status_code == 200:
        df = pd.DataFrame(response.json(), columns=["time", "low", "high", "open", "close", "volume"])
        df = df.sort_values(by="time").reset_index(drop=True)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df
    else:
        print(f"Failed to fetch data for {symbol} — status code {response.status_code}")
        return None

def detect_fvg(df):
    gaps = []
    for i in range(2, len(df)):
        prev2 = df.iloc[i - 2]
        curr = df.iloc[i]
        # Bullish FVG
        if prev2["low"] > curr["high"]:
            gaps.append({"type": "bullish", "start": curr["high"], "end": prev2["low"], "index": i})
        # Bearish FVG
        elif prev2["high"] < curr["low"]:
            gaps.append({"type": "bearish", "start": prev2["high"], "end": curr["low"], "index": i})
    return gaps

def backtest_fvg(df, gaps):
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=atr_window).average_true_range()
    wins, losses, profit = 0, 0, 0.0

    for gap in gaps:
        i = gap["index"]
        entry_price = df.iloc[i + 1]["close"] if i + 1 < len(df) else None
        if entry_price is None or pd.isna(atr[i]):
            continue

        # TP1 and SL1
        if gap["type"] == "bullish":
            tp = entry_price + atr[i] * 1
            sl = entry_price - atr[i] * 1
            for j in range(i + 1, len(df)):
                low = df.iloc[j]["low"]
                high = df.iloc[j]["high"]
                if low <= sl:
                    losses += 1
                    profit -= capital_per_trade * (entry_price - sl) / entry_price
                    break
                elif high >= tp:
                    wins += 1
                    profit += capital_per_trade * (tp - entry_price) / entry_price
                    break
        elif gap["type"] == "bearish":
            tp = entry_price - atr[i] * 1
            sl = entry_price + atr[i] * 1
            for j in range(i + 1, len(df)):
                high = df.iloc[j]["high"]
                low = df.iloc[j]["low"]
                if high >= sl:
                    losses += 1
                    profit -= capital_per_trade * (sl - entry_price) / entry_price
                    break
                elif low <= tp:
                    wins += 1
                    profit += capital_per_trade * (entry_price - tp) / entry_price
                    break

    return wins, losses, profit

# --- MAIN ---
summary = []
total_wins = total_losses = 0
total_profit = 0.0

for symbol in symbols:
    df = fetch_candles(symbol)
    time.sleep(1)  # Pause to avoid rate limit
    if df is None or len(df) < 20:
        continue
    gaps = detect_fvg(df)
    wins, losses, profit = backtest_fvg(df, gaps)
    total_wins += wins
    total_losses += losses
    total_profit += profit
    summary.append({
        "symbol": symbol,
        "wins": wins,
        "losses": losses,
        "net_profit": round(profit, 2)
    })

if summary:
    summary_df = pd.DataFrame(summary)
    summary_df["total_trades"] = summary_df["wins"] + summary_df["losses"]
    summary_df = summary_df.sort_values(by="net_profit", ascending=False)

    print(summary_df)
    print(f"\nTOTAL WINS: {total_wins}")
    print(f"TOTAL LOSSES: {total_losses}")
    print(f"NET PROFIT: ${round(total_profit, 2)}")
else:
    print("❌ No data fetched or no valid signals detected.")
