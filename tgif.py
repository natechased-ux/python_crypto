import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

COIN_LIST = ["BTC-USD", "ETH-USD", "XRP-USD"]
GRANULARITY_1H = 3600
GRANULARITY_1D = 86400


def fetch_coinbase_candles(symbol, granularity, limit=300):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=granularity * limit)
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "granularity": granularity
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit='s')
        df.sort_values("time", inplace=True)
        return df.reset_index(drop=True)
    else:
        print(f"Error fetching {symbol}: {r.text}")
        return pd.DataFrame()


def get_past_fridays(n=12):
    today = datetime.utcnow().date()
    fridays = []
    current = today - timedelta(days=today.weekday() + 3)
    for _ in range(n):
        fridays.append(current)
        current -= timedelta(weeks=1)
    return fridays


def tgif_zone_check(df_daily, df_hourly, friday_date):
    df_week = df_daily[df_daily["time"].dt.date < friday_date]
    week_df = df_week[df_week["time"].dt.weekday < 4]  # Monday–Thursday only
    if week_df.empty:
        return None

    open_price = week_df.iloc[0]["open"]
    close_price = week_df.iloc[-1]["close"]
    high = week_df["high"].max()
    low = week_df["low"].min()
    direction = "bullish" if close_price > open_price else "bearish"
    
    range_size = high - low
    if direction == "bullish":
        zone_low = low + range_size * 0.20
        zone_high = low + range_size * 0.30
    else:
        zone_high = high - range_size * 0.20
        zone_low = high - range_size * 0.30

    friday_data = df_hourly[df_hourly["time"].dt.date == friday_date]
    for _, row in friday_data.iterrows():
        if zone_low <= row["low"] <= zone_high or zone_low <= row["high"] <= zone_high:
            return {
                "date": friday_date,
                "direction": "LONG" if direction == "bullish" else "SHORT",
                "entry_price": row["close"],
                "zone_low": zone_low,
                "zone_high": zone_high
            }
    return None


results = []
for friday in tqdm(get_past_fridays(12), desc="TGIF Backtest"):
    for symbol in COIN_LIST:
        df_1h = fetch_coinbase_candles(symbol, GRANULARITY_1H, 300)
        df_daily = fetch_coinbase_candles(symbol, GRANULARITY_1D, 300)

        df_1h = df_1h[df_1h["time"].dt.date <= friday]
        df_daily = df_daily[df_daily["time"].dt.date <= friday]

        if len(df_1h) < 50 or len(df_daily) < 10:
            continue

        result = tgif_zone_check(df_daily, df_1h, friday)
        if result:
            result["symbol"] = symbol
            results.append(result)

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("tgif_retracement_trades.csv", index=False)
print("✅ TGIF backtest complete. Results saved to tgif_retracement_trades.csv")
