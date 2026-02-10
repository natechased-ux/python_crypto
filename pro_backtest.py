import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# ===== CONFIG =====
PRODUCT_IDS = ["BTC-USD", "ETH-USD", "SOL-USD"]
START_DAYS = 90
ENTRY_INTERVAL = 300    # 5m
CONFIRM_INTERVAL = 3600 # 1h
ATR_MULT_TP = 1.5
ATR_MULT_SL = 1.0

# ===== INDICATORS =====
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(high, low, close, period=14):
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ===== DATA FETCHER =====
def fetch_candles(product_id, granularity, start, end):
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {
        "granularity": granularity,
        "start": start.isoformat(),
        "end": end.isoformat()
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    # Coinbase returns: [time, low, high, open, close, volume]
    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df

def get_full_history(product_id, granularity, days):
    now = datetime.now(timezone.utc)
    end = now
    all_data = []
    step = timedelta(seconds=granularity * 300)  # ~300 candles per request
    start = now - timedelta(days=days)
    while end > start:
        chunk_start = max(start, end - step)
        df = fetch_candles(product_id, granularity, chunk_start, end)
        if df.empty:
            break
        all_data.append(df)
        end = df["time"].min()
    return pd.concat(all_data).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

# ===== STRATEGY =====
def trend_direction(df):
    if len(df) < 55:
        return None
    fast = ema(df['close'], 21)
    slow = ema(df['close'], 55)
    if fast.iloc[-1] > slow.iloc[-1]:
        return "BULL"
    elif fast.iloc[-1] < slow.iloc[-1]:
        return "BEAR"
    return None

def backtest_coin(symbol, df_5m, df_1h):
    trades = []
    for i in range(55, len(df_5m)):
        # 1h trend confirmation
        t = df_5m.iloc[i]["time"]
        df1h_cut = df_1h[df_1h["time"] <= t]
        if df1h_cut.empty:
            continue
        trend_5m = trend_direction(df_5m.iloc[:i+1])
        trend_1h = trend_direction(df1h_cut)
        if trend_5m != trend_1h or trend_5m is None:
            continue

        # Indicators on 5m
        df5_cut = df_5m.iloc[:i+1].copy()
        df5_cut['ema_fast'] = ema(df5_cut['close'], 21)
        df5_cut['ema_slow'] = ema(df5_cut['close'], 55)
        df5_cut['rsi'] = rsi(df5_cut['close'], 14)
        df5_cut['atr'] = atr(df5_cut['high'], df5_cut['low'], df5_cut['close'], 14)
        df5_cut['vol_ma'] = df5_cut['volume'].rolling(20).mean()
        last = df5_cut.iloc[-1]

        prev_high = df5_cut['high'].iloc[-10:-1].max()
        prev_low = df5_cut['low'].iloc[-10:-1].min()

        long_cond = (
            trend_5m == "BULL" and
            last['rsi'] > 55 and
            last['volume'] > 1.2 * last['vol_ma'] and
            last['close'] > prev_high
        )
        short_cond = (
            trend_5m == "BEAR" and
            last['rsi'] < 45 and
            last['volume'] > 1.2 * last['vol_ma'] and
            last['close'] < prev_low
        )

        if not (long_cond or short_cond):
            continue

        entry_price = last['close']
        atr_val = last['atr']
        if atr_val == 0 or pd.isna(atr_val):
            continue

        if long_cond:
            tp = entry_price + ATR_MULT_TP * atr_val
            sl = entry_price - ATR_MULT_SL * atr_val
            direction = "LONG"
        else:
            tp = entry_price - ATR_MULT_TP * atr_val
            sl = entry_price + ATR_MULT_SL * atr_val
            direction = "SHORT"

        # Simulate forward
        outcome = None
        rr = None
        for j in range(i+1, len(df_5m)):
            high = df_5m.iloc[j]['high']
            low = df_5m.iloc[j]['low']
            if direction == "LONG":
                if high >= tp:
                    outcome = "TP"
                    rr = ATR_MULT_TP / ATR_MULT_SL
                    break
                elif low <= sl:
                    outcome = "SL"
                    rr = -1
                    break
            else:
                if low <= tp:
                    outcome = "TP"
                    rr = ATR_MULT_TP / ATR_MULT_SL
                    break
                elif high >= sl:
                    outcome = "SL"
                    rr = -1
                    break
        if outcome is None:
            outcome = "NONE"
            rr = 0

        trades.append({
            "symbol": symbol,
            "time": t,
            "direction": direction,
            "entry": entry_price,
            "tp": tp,
            "sl": sl,
            "outcome": outcome,
            "rr": rr
        })

    return pd.DataFrame(trades)

# ===== RUN BACKTEST =====
all_results = []
summary_rows = []

for symbol in PRODUCT_IDS:
    print(f"Fetching data for {symbol}...")
    df_5m = get_full_history(symbol, ENTRY_INTERVAL, START_DAYS)
    df_1h = get_full_history(symbol, CONFIRM_INTERVAL, START_DAYS)

    print(f"Backtesting {symbol}...")
    trades_df = backtest_coin(symbol, df_5m, df_1h)
    all_results.append(trades_df)

    total_trades = len(trades_df)
    wins = (trades_df['outcome'] == "TP").sum()
    losses = (trades_df['outcome'] == "SL").sum()
    avg_rr = trades_df['rr'].mean() if total_trades > 0 else 0
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    alerts_per_day = total_trades / START_DAYS

    summary_rows.append({
        "Coin": symbol,
        "Alerts/day": round(alerts_per_day, 2),
        "Win rate %": round(win_rate, 1),
        "Avg R:R": round(avg_rr, 2),
        "Total Trades": total_trades
    })

# Save trade log
all_trades_df = pd.concat(all_results)
all_trades_df.to_csv("trades_log.csv", index=False)

# Print summary
summary_df = pd.DataFrame(summary_rows)
print("\n=== 90-DAY BACKTEST SUMMARY ===")
print(summary_df.to_string(index=False))
print("\nDetailed trade log saved to trades_log.csv")
