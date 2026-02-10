import pandas as pd
import numpy as np
import requests
import asyncio, aiohttp, joblib, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# =========================
# CONFIG
# =========================
BASE_URL = "https://api.exchange.coinbase.com"
LOOKBACK_MINUTES = 300  # enough to build indicators
MFE_THRESHOLD = 0.005   # match trainer
ATR_PCT_MIN = 0.003
ADX_MIN = 20
COOLDOWN_BARS = 60  # 1 hour on 1m chart

WHITELIST_FILE = "whitelist.csv"
MODEL_DIR = Path(r"C:\Users\natec")
  # directory with _long.pkl and _short.pkl models

REQUEST_DELAY = 0.35

BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"

# =========================
# TELEGRAM ALERT FUNCTION (use your existing one)
# =========================
def send_alert(symbol, signal, entry, tp, sl):
    msg = (
        f"🚨 {signal} {symbol}\n"
        f"Entry: {format_price(entry)}\n"
        f"TP: {format_price(tp)}\n"
        f"SL: {format_price(sl)}\n"
        f"SUPERBOT"
    )
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        )
    except Exception as e:
        print(f"Telegram send error: {e}")

def format_price(price):
    """Format price with appropriate decimals based on its value."""
    if price >= 100:
        return f"{price:,.2f}"   # 2 decimals for large prices
    elif price >= 1:
        return f"{price:,.4f}"   # 4 decimals for mid prices
    else:
        return f"{price:,.6f}"   # 6 decimals for small prices

# =========================
# FETCH FUNCTIONS
# =========================
async def fetch_candles(coin, minutes):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=minutes)
    all_data = []
    async with aiohttp.ClientSession() as session:
        while start_time < end_time:
            chunk_end = min(start_time + timedelta(seconds=60 * 300), end_time)
            await asyncio.sleep(REQUEST_DELAY)
            url = f"{BASE_URL}/products/{coin}/candles"
            params = {
                "granularity": 60,
                "start": start_time.isoformat(),
                "end": chunk_end.isoformat()
            }
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if isinstance(data, dict) and "message" in data:
                    return pd.DataFrame()
                df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
                df["time"] = pd.to_datetime(df["time"], unit="s")
                all_data.append(df.sort_values("time"))
            start_time = chunk_end
    if all_data:
        return pd.concat(all_data).drop_duplicates(subset=["time"]).reset_index(drop=True)
    return pd.DataFrame()

# =========================
# INDICATOR FUNCTIONS (match trainer)
# =========================
def add_indicators(df):
    if df.empty:
        return df
    df = df.copy()

    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    df["atr_norm"] = df["atr"] / df["close"]

    bb = BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]

    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"]).adx()
    df["macd_diff"] = MACD(df["close"]).macd_diff()
    df["rsi"] = RSIIndicator(df["close"]).rsi()

    df["ema20_sim_5m"] = df["close"].rolling(5).mean().ewm(span=20).mean()
    df["ema50_sim_5m"] = df["close"].rolling(5).mean().ewm(span=50).mean()
    df["rsi_sim_5m"] = df["close"].rolling(5).mean().pipe(RSIIndicator).rsi()

    df["ema20_sim_15m"] = df["close"].rolling(15).mean().ewm(span=20).mean()
    df["ema50_sim_15m"] = df["close"].rolling(15).mean().ewm(span=50).mean()
    df["rsi_sim_15m"] = df["close"].rolling(15).mean().pipe(RSIIndicator).rsi()

    return df.fillna(0)

def extract_features(df):
    features = [
        "atr_norm", "bb_width", "rsi", "macd_diff", "adx",
        "volume", "open", "close", "high", "low",
        "volume_ratio", "price_above_mid", "price_body_ratio",
        "ema20", "ema50", "ema200",
        "ema20_sim_5m", "ema50_sim_5m", "rsi_sim_5m",
        "ema20_sim_15m", "ema50_sim_15m", "rsi_sim_15m"
    ]
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["price_above_mid"] = df["close"] - df["bb_middle"]
    df["price_body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    return df[features].fillna(0)

# =========================
# MAIN ALERT LOOP
# =========================
async def alert_loop():
    whitelist = pd.read_csv(WHITELIST_FILE)["coin"].tolist()
    models = {}

    # Load models for each coin
    for coin in whitelist:
        models[coin] = {
            "long": joblib.load(MODEL_DIR / f"lgbm_model_{coin.replace('-', '_')}_long.pkl"),
            "short": joblib.load(MODEL_DIR / f"lgbm_model_{coin.replace('-', '_')}_short.pkl")
        }

    last_alert_time = {
    coin: {"long": datetime.min.replace(tzinfo=timezone.utc), "short": datetime.min.replace(tzinfo=timezone.utc)}
    for coin in whitelist
}

    while True:
        for coin in whitelist:
            df = await fetch_candles(coin, LOOKBACK_MINUTES)
            if df.empty:
                print(f"[{coin}] ⚠️ No data fetched")
                continue
            df = add_indicators(df)
            X = extract_features(df)
            latest_features = X.iloc[-1:].values

            pred_long = models[coin]["long"].predict(latest_features)[0]
            pred_short = models[coin]["short"].predict(latest_features)[0]
            atr_val = df["atr_norm"].iloc[-1]
            adx_val = df["adx"].iloc[-1]

            now = datetime.now(timezone.utc)

            # Debug log for monitoring
            print(
                f"[{coin}] LONG={pred_long:.4f}, SHORT={pred_short:.4f}, "
                f"ATR%={atr_val:.4f}, ADX={adx_val:.2f}"
            )

            # ATR/ADX filter
            if atr_val < ATR_PCT_MIN or adx_val < ADX_MIN:
                continue

            # Direction logic: whichever has the higher prediction above threshold
            if pred_long >= MFE_THRESHOLD or pred_short >= MFE_THRESHOLD:
                if pred_long >= pred_short:
                    direction = "LONG"
                    pred_val = pred_long
                else:
                    direction = "SHORT"
                    pred_val = pred_short
            else:
                continue

            # Cooldown check
            if (now - last_alert_time[coin][direction.lower()]) >= timedelta(minutes=COOLDOWN_BARS):
                entry_price = df["close"].iloc[-1]
                if direction == "LONG":
                    tp = entry_price * (1 + pred_val)
                    sl = entry_price * (1 - MFE_THRESHOLD)
                else:  # SHORT
                    tp = entry_price * (1 - pred_val)
                    sl = entry_price * (1 + MFE_THRESHOLD)

                send_alert(
                    symbol=coin,
                    signal=direction,
                    entry=entry_price,
                    tp=tp,
                    sl=sl
                )

                last_alert_time[coin][direction.lower()] = now

        # Sleep until next minute
        time.sleep(60 - datetime.now().second)

   


if __name__ == "__main__":
    asyncio.run(alert_loop())
