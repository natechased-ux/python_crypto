import pandas as pd
import numpy as np
import asyncio, aiohttp, joblib
from pathlib import Path
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight

# === CONFIG ===
BASE_URL = "https://api.exchange.coinbase.com"
COINS = [   "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "DOGE-USD", "ADA-USD", "XLM-USD", "SUI-USD",
    "BCH-USD", "LINK-USD", "HBAR-USD", "LTC-USD", "AVAX-USD", "SHIB-USD", "UNI-USD", "DOT-USD",
    "CRO-USD",  "AAVE-USD", "NEAR-USD", "ETC-USD", "ONDO-USD",
    "APT-USD", "ICP-USD", "ALGO-USD", "ARB-USD", "VET-USD",
    "ATOM-USD", "POL-USD", "BONK-USD", "RENDER-USD",
    "FET-USD", "SEI-USD", "FLR-USD", "FIL-USD", "QNT-USD", "LSETH-USD", "INJ-USD",
    "CRV-USD", "STX-USD", "TIA-USD"]
TIMEFRAMES = ["1m", "5m", "15m", "1h"]

CANDLE_LIMIT = 300
LOOKBACK_DAYS = 90
tf_seconds_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}

# Rate limit
REQUEST_DELAY = 0.35
async def fetch_candles_chunked(session, coin, tf_sec, start, end):
    await asyncio.sleep(REQUEST_DELAY)
    url = f"{BASE_URL}/products/{coin}/candles"
    params = {"granularity": tf_sec, "start": start.isoformat(), "end": end.isoformat()}
    async with session.get(url, params=params) as resp:
        data = await resp.json()
        if isinstance(data, dict) and "message" in data:
            print(f"API error for {coin}: {data}")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df.sort_values("time")

async def fetch_all_candles(coin, tf_sec):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=LOOKBACK_DAYS)
    all_data = []
    async with aiohttp.ClientSession() as session:
        while start_time < end_time:
            chunk_end = min(start_time + timedelta(seconds=tf_sec*CANDLE_LIMIT), end_time)
            df = await fetch_candles_chunked(session, coin, tf_sec, start_time, chunk_end)
            if not df.empty:
                all_data.append(df)
            start_time = chunk_end
    if all_data:
        return pd.concat(all_data).drop_duplicates(subset=["time"]).reset_index(drop=True)
    return pd.DataFrame()

# === Feature Engineering ===
def add_features(df):
    if df.empty:
        return df
    df = df.copy()
    # Returns & volatility
    for lag in [1,3,5,10,30]:
        df[f"ret_{lag}"] = df["close"].pct_change(lag)
    df["volatility_20"] = df["close"].pct_change().rolling(20).std()
    # Trend
    for span in [10,20,50]:
        df[f"ema_{span}"] = EMAIndicator(df["close"], span).ema_indicator()
        df[f"sma_{span}"] = SMAIndicator(df["close"], span).sma_indicator()
    df["macd"] = MACD(df["close"]).macd()
    df["macd_signal"] = MACD(df["close"]).macd_signal()
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], 14).adx()
    # Momentum
    df["rsi"] = RSIIndicator(df["close"], 14).rsi()
    df["stoch_rsi"] = StochRSIIndicator(df["close"], 14).stochrsi()
    df["williams_r"] = WilliamsRIndicator(df["high"], df["low"], df["close"], 14).williams_r()
    # Volatility
    bb = BollingerBands(df["close"], 20, 2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
    # Volume
    df["obv"] = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    df["cmf"] = ChaikinMoneyFlowIndicator(df["high"], df["low"], df["close"], df["volume"], 20).chaikin_money_flow()
    return df.fillna(0)

# === Multi-timeframe merge ===
async def fetch_all_timeframes(coin):
    dfs = await asyncio.gather(*[fetch_all_candles(coin, tf_seconds_map[tf]) for tf in TIMEFRAMES])
    dfs = [add_features(df).set_index("time") for df in dfs]
    df_1m = dfs[0]
    for i, tf in enumerate(TIMEFRAMES[1:], start=1):
        df_tf = dfs[i].reindex(df_1m.index, method="ffill").add_suffix(f"_{tf}")
        df_1m = df_1m.join(df_tf, how="left")
    return df_1m.reset_index().fillna(0)

# === Volatility-adjusted labels ===
def create_labels(df, horizon=5, vol_mult=0.5, fee=0.001):
    df = df.copy()
    rolling_vol = df["close"].pct_change().rolling(200).std()
    df["threshold"] = rolling_vol * vol_mult
    df["future_return"] = (df["close"].shift(-horizon) - df["close"]) / df["close"] - fee
    df["target"] = 0
    df.loc[df["future_return"] > df["threshold"], "target"] = 1
    df.loc[df["future_return"] < -df["threshold"], "target"] = -1
    return df.dropna()

# === Performance Metrics ===
def evaluate_preds(y_true, y_pred, returns):
    trades = np.where(y_pred != 0)[0]
    if len(trades) == 0:
        return {"win_rate": 0, "profit_factor": 0, "sharpe": 0}
    trade_returns = returns[trades] * y_pred[trades]
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    win_rate = len(wins) / max(1, len(trades))
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    sharpe = trade_returns.mean() / (trade_returns.std() + 1e-9) * np.sqrt(252*24*60)
    return {"win_rate": win_rate, "profit_factor": profit_factor, "sharpe": sharpe}

# === Walk-forward training ===
def walk_forward_train(df, coin, train_days=30, test_days=7):
    results = []
    start_idx = 0
    while True:
        train_end = start_idx + train_days*1440  # minutes
        test_end = train_end + test_days*1440
        if test_end >= len(df):
            break
        train_df = df.iloc[start_idx:train_end]
        test_df = df.iloc[train_end:test_end]
        X_train = train_df.drop(columns=["time", "low", "high", "open", "close", "volume", "future_return", "target", "threshold"])
        y_train = train_df["target"]
        X_test = test_df.drop(columns=["time", "low", "high", "open", "close", "volume", "future_return", "target", "threshold"])
        y_test = test_df["target"]
        model = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8
        )
        weights = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=weights)
        y_pred = model.predict(X_test)
        metrics = evaluate_preds(y_test.values, y_pred, test_df["future_return"].values)
        results.append(metrics)
        start_idx += test_days*1440
    avg_metrics = {k: np.mean([r[k] for r in results]) for k in results[0]}
    return avg_metrics, model

# === Main ===
async def main():
    for coin in COINS:
        print(f"📡 Fetching data for {coin}")
        df = await fetch_all_timeframes(coin)
        df = create_labels(df)
        if df.empty:
            print(f"⚠️ No data for {coin}")
            continue
        metrics, model = walk_forward_train(df, coin)
        print(f"{coin} | WinRate: {metrics['win_rate']:.2f} | PF: {metrics['profit_factor']:.2f} | Sharpe: {metrics['sharpe']:.2f}")
        joblib.dump(model, Path(f"lgbm_model_{coin.replace('-', '_')}.pkl"))

if __name__ == "__main__":
    asyncio.run(main())
