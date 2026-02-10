import requests, pandas as pd, asyncio, aiohttp, joblib
from datetime import datetime, timedelta
from pathlib import Path
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD
from train_superbot import fetch_all_candles, add_indicators, create_labels, train_model

# === CONFIG ===
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"


#TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
#CHAT_ID = "7967738614"

#personal chat id:"7967738614"
#channel chat id:
#"-4916911067"
COINS = [   "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "DOGE-USD", "ADA-USD", "XLM-USD", "SUI-USD",
    "BCH-USD", "LINK-USD", "HBAR-USD", "LTC-USD", "AVAX-USD", "SHIB-USD", "UNI-USD", "DOT-USD",
    "CRO-USD",  "AAVE-USD", "NEAR-USD", "ETC-USD", "ONDO-USD",
    "APT-USD", "ICP-USD", "ALGO-USD", "ARB-USD", "VET-USD",
    "ATOM-USD", "POL-USD", "BONK-USD", "RENDER-USD",
    "FET-USD", "SEI-USD", "FLR-USD", "FIL-USD", "QNT-USD", "LSETH-USD", "INJ-USD",
    "CRV-USD", "STX-USD", "TIA-USD"]

BASE_URL = "https://api.exchange.com"  # Replace with real API endpoint
MODEL_PATH = Path("lgbm_model.pkl")
TIMEFRAMES = ["1m", "5m", "15m", "1h"]


BASE_POSITION_SIZE = 0.01
MAX_DRAWDOWN_DAY = 0.05
tf_seconds_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}

# === Telegram Alerts ===
def send_telegram(msg):
    try:
        requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                     params={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print("Telegram error:", e)

# === Async fetch live candles ===
REQUEST_DELAY = 0.35  # ~3/sec safe
request_queue = asyncio.Queue()

async def request_worker(session):
    while True:
        url, params, coin = await request_queue.get()
        try:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if isinstance(data, dict) and "message" in data:
                    print(f"API error for {coin}: {data}")
                    result = pd.DataFrame()
                else:
                    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    result = df.sort_values("time")
        except Exception as e:
            print(f"Request failed for {coin}: {e}")
            result = pd.DataFrame()
        request_queue.task_done()
        await asyncio.sleep(REQUEST_DELAY)
        return result

async def fetch_candle(session, coin, tf, limit=100):
    url = f"https://api.exchange.coinbase.com/products/{coin}/candles"
    params = {"granularity": tf_seconds_map[tf]}
    await request_queue.put((url, params, coin))
    df = await request_worker(session)
    return df.head(limit)

async def fetch_all_data(coin):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_candle(session, coin, tf) for tf in TIMEFRAMES]
        results = await asyncio.gather(*tasks)
        return {tf: df for tf, df in zip(TIMEFRAMES, results)}


# === Merge features ===
def merge_timeframes(data_dict):
    base_df = data_dict["1m"]
    for tf in TIMEFRAMES[1:]:
        tf_df = data_dict[tf].add_suffix(f"_{tf}")
        base_df = base_df.join(tf_df, rsuffix=f"_{tf}", how="left")
    return base_df.fillna(method="ffill").fillna(0)

# === Decision & Position Sizing ===
def decide_trade(proba):
    buy_prob, hold_prob, sell_prob = proba
    if buy_prob > 0.6:
        return "BUY", buy_prob
    elif sell_prob > 0.6:
        return "SELL", sell_prob
    return "HOLD", hold_prob

def get_position_size(conf):
    return BASE_POSITION_SIZE * max(0, (conf - 0.5) * 2)

# === Retrain all coin models ===
async def retrain_all_models():
    for coin in COINS:
        print(f"🔄 Retraining model for {coin}...")
        merged_df = None
        for tf in TIMEFRAMES:
            df_tf = await fetch_all_candles(coin, tf_seconds_map[tf])
            df_tf = add_indicators(df_tf)
            if merged_df is None:
                merged_df = df_tf
            else:
                merged_df = merged_df.merge(df_tf, on="time", how="left", suffixes=("", f"_{tf}"))
        labeled = create_labels(merged_df)
        if not labeled.empty:
            X = labeled.drop(columns=[
                "time", "low", "high", "open", "close", "volume",
                "future_close", "future_return", "target"
            ]).values
            y = labeled["target"].values
            model_path = Path(f"lgbm_model_{coin.replace('-', '_')}.pkl")
            train_model(X, y, model_path)
    send_telegram("📈 All per-coin models retrained.")

# === Main trading loop ===
async def main_loop():
    models = {}
    for coin in COINS:
        model_path = Path(f"lgbm_model_{coin.replace('-', '_')}.pkl")
        if model_path.exists():
            models[coin] = joblib.load(model_path)

    daily_pnl = 0
    next_retrain = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    while True:
        now = datetime.utcnow()

        # Daily retrain check
        if now >= next_retrain:
            await retrain_all_models()
            for coin in COINS:
                models[coin] = joblib.load(Path(f"lgbm_model_{coin.replace('-', '_')}.pkl"))
            next_retrain += timedelta(days=1)

        for coin in COINS:
            try:
                data_dict = await fetch_all_data(coin)
                data_dict = {tf: add_indicators(df) for tf, df in data_dict.items()}
                merged = merge_timeframes(data_dict)
                X_live = merged.drop(columns=["time", "open", "high", "low", "close", "volume"]).values[-1:]

                if coin in models:
                    proba = models[coin].predict_proba(X_live)[0]
                    action, conf = decide_trade(proba)
                    size = get_position_size(conf)

                    if daily_pnl < -MAX_DRAWDOWN_DAY:
                        send_telegram("🚨 Daily drawdown limit reached. Trading stopped.")
                        return

                    if action in ["BUY", "SELL"]:
                        send_telegram(f"{action} {coin} | Conf: {conf:.2f} | Size: {size:.4f}")
                        # PLACE ORDER HERE

            except Exception as e:
                print(f"Error with {coin}: {e}")

        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main_loop())
