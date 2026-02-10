
# === CRYPTO BACKTEST SCRIPT v2 ===
# Simulates golden zone + Stoch RSI strategy with filters

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# Strategy settings
FIB_LEVELS = [0.382, 0.5, 0.618, 0.66, 0.786]
GOLDEN_ZONE = (0.618, 0.66)
PRICE_BUFFER = 0.0025  # 0.25%

USE_EMA_FILTER = True
USE_MTF_CONFIRMATION = False
USE_ENGULFING_FILTER = False
USE_STRONG_CROSS_FILTER = False
USE_VOLATILITY_SUPPRESSION = False

USE_RR_FILTER = False

def fetch_ohlcv(symbol, granularity=3600):
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=granularity * 300)
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    params = {"granularity": granularity, "start": start.isoformat(), "end": end.isoformat()}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch OHLCV for {symbol}")
        return None
    df = pd.DataFrame(response.json(), columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.sort_values("time").reset_index(drop=True)

def calculate_stoch_rsi(df, period=14, smooth_k=3, smooth_d=3):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    min_rsi = rsi.rolling(period).min()
    max_rsi = rsi.rolling(period).max()
    stoch_rsi = 100 * (rsi - min_rsi) / (max_rsi - min_rsi)
    k = stoch_rsi.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_ema(series, period=200):
    return series.ewm(span=period, adjust=False).mean()

def is_bullish_engulfing(df):
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return (prev["close"] < prev["open"]) and (curr["close"] > curr["open"]) and (curr["close"] > prev["open"]) and (curr["open"] < prev["close"])

def is_bearish_engulfing(df):
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return (prev["close"] > prev["open"]) and (curr["close"] < curr["open"]) and (curr["close"] < prev["open"]) and (curr["open"] > prev["close"])

def calculate_trend(df, ema_period=200):
    ema = calculate_ema(df["close"], ema_period)
    return df["close"].iloc[-1] > ema.iloc[-1]

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean().iloc[-1]

def calculate_rr(tp, sl, entry):
    reward = abs(tp - entry)
    risk = abs(entry - sl)
    return reward / risk if risk > 0 else 0

def simulate_trade(df, side, entry_price, sl, tp):
    for i in range(len(df)):
        high = df.iloc[i]["high"]
        low = df.iloc[i]["low"]
        if side == "LONG":
            if low <= sl:
                return "SL", sl
            elif high >= tp:
                return "TP", tp
        else:
            if high >= sl:
                return "SL", sl
            elif low <= tp:
                return "TP", tp
    return "NONE", df.iloc[-1]["close"]

def should_trigger_long(df, price, golden_low, golden_high, k_val, d_val, atr, tp, sl):
    if USE_EMA_FILTER and not calculate_trend(df):
        return False
    if USE_MTF_CONFIRMATION:
        trend_df = df.copy()
        trend_df["ema"] = calculate_ema(df["close"])
        if price < trend_df["ema"].iloc[-1]:
            return False
    if USE_ENGULFING_FILTER and not is_bullish_engulfing(df):
        return False
    if USE_STRONG_CROSS_FILTER and abs(k_val - d_val) > 5:
        return False
    if USE_VOLATILITY_SUPPRESSION and atr / price < 0.002:
        return False
    if USE_RR_FILTER and calculate_rr(tp, sl, price) < 1.5:
        return False
    return True

def should_trigger_short(df, price, golden_low, golden_high, k_val, d_val, atr, tp, sl):
    if USE_EMA_FILTER and calculate_trend(df):
        return False
    if USE_MTF_CONFIRMATION:
        trend_df = df.copy()
        trend_df["ema"] = calculate_ema(df["close"])
        if price > trend_df["ema"].iloc[-1]:
            return False
    if USE_ENGULFING_FILTER and not is_bearish_engulfing(df):
        return False
    if USE_STRONG_CROSS_FILTER and abs(k_val - d_val) > 5:
        return False
    if USE_VOLATILITY_SUPPRESSION and atr / price < 0.002:
        return False
    if USE_RR_FILTER and calculate_rr(tp, sl, price) < 1.5:
        return False
    return True

def run_batch_backtest(symbols):
    results = []
    for symbol in symbols:
        df = fetch_ohlcv(symbol)
        if df is None or len(df) < 100:
            continue
        k, d = calculate_stoch_rsi(df)
        atr = calculate_atr(df)

        recent_high = df["high"].max()
        recent_low = df["low"].min()
        fib_range = recent_high - recent_low
        golden_high = recent_high - fib_range * GOLDEN_ZONE[0]
        golden_low = recent_high - fib_range * GOLDEN_ZONE[1]

        for i in range(75, len(df) - 1):
            row = df.iloc[i]
            price = row["close"]
            k_val, d_val = k.iloc[i], d.iloc[i]
            k_prev, d_prev = k.iloc[i - 1], d.iloc[i - 1]
            crossed = (k_prev < d_prev and k_val > d_val) or (k_prev > d_prev and k_val < d_val)

            in_zone = golden_low <= price <= golden_high
            slightly_below = golden_low * (1 - PRICE_BUFFER) <= price < golden_low
            slightly_above = golden_high < price <= golden_high * (1 + PRICE_BUFFER)

            if crossed:
                if (in_zone or slightly_below) and k_val < 20 and d_val < 20:
                    tp = price + atr
                    sl = golden_low * 0.995
                    if should_trigger_long(df.iloc[:i+1], price, golden_low, golden_high, k_val, d_val, atr, tp, sl):
                        result, exit_price = simulate_trade(df.iloc[i+1:], "LONG", price, sl, tp)
                        rr = calculate_rr(tp, sl, price)
                        results.append([symbol, "LONG", price, tp, sl, result, exit_price, atr, rr])

                elif (in_zone or slightly_above) and k_val > 80 and d_val > 80:
                    tp = price - atr
                    sl = golden_high * 1.005
                    if should_trigger_short(df.iloc[:i+1], price, golden_low, golden_high, k_val, d_val, atr, tp, sl):
                        result, exit_price = simulate_trade(df.iloc[i+1:], "SHORT", price, sl, tp)
                        rr = calculate_rr(tp, sl, price)
                        results.append([symbol, "SHORT", price, tp, sl, result, exit_price, atr, rr])
    return pd.DataFrame(results, columns=["symbol", "direction", "entry", "tp", "sl", "result", "exit_price", "atr", "rr"])

if __name__ == "__main__":
    symbols = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd",
           "syrup-usd","fartcoin-usd", "aero-usd", "link-usd","hbar-usd",
           "aave-usd", "fet-usd", "crv-usd","tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd",
           "bch-usd", "inj-usd", "pepe-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
           "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
           "qnt-usd", "tia-usd", "ip-usd" , "pnut-usd", "apt-usd", "vet-usd",
           "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "velo-usd", "coti-usd",
           "axs-usd"]
    df = run_batch_backtest(symbols)
    df.to_csv("backtest_results.csv", index=False)
    print(df)
