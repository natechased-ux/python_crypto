import pandas as pd
import numpy as np
import asyncio, aiohttp, joblib, optuna
from pathlib import Path
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
import warnings
import argparse
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--diagnostic', action='store_true', help='Run in diagnostic mode (no model saving)')
args = parser.parse_args()
DIAGNOSTIC_MODE = args.diagnostic



# === CONFIG ===
PF_results = []
winrate_results = []
trades_results = []
BASE_URL = "https://api.exchange.coinbase.com"  # Coinbase API
COINS = [    "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "DOGE-USD", "ADA-USD", "XLM-USD", "SUI-USD",
    "BCH-USD", "LINK-USD", "HBAR-USD", "LTC-USD", "AVAX-USD", "SHIB-USD", "UNI-USD", "DOT-USD",
    "CRO-USD",  "AAVE-USD", "NEAR-USD", "ETC-USD", "ONDO-USD",
    "APT-USD", "ICP-USD", "ALGO-USD", "ARB-USD", "VET-USD",
    "ATOM-USD", "POL-USD", "BONK-USD", "RENDER-USD",
    "FET-USD", "SEI-USD", "FLR-USD", "FIL-USD", "QNT-USD", "LSETH-USD", "INJ-USD",
    "CRV-USD", "STX-USD", "TIA-USD"]

TIMEFRAMES = ["1m", "5m", "15m", "1h"]
MODEL_PATH = Path("lgbm_model.pkl")


CANDLE_LIMIT = 300
LOOKBACK_DAYS = 180
tf_seconds_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}

# Triple barrier params
TP_PCT = 0.005     # 0.5%
SL_PCT = 0.003     # 0.3%
MAX_HOLD = 15      # bars
EXEC_COST = 0.0005 # 0.05%

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

async def fetch_all_timeframes(coin):
    dfs = await asyncio.gather(*[fetch_all_candles(coin, tf_seconds_map[tf]) for tf in TIMEFRAMES])
    dfs = [add_features(df).set_index("time") for df in dfs]
    df_1m = dfs[0]
    for i, tf in enumerate(TIMEFRAMES[1:], start=1):
        df_tf = dfs[i].reindex(df_1m.index, method="ffill").add_suffix(f"_{tf}")
        df_1m = df_1m.join(df_tf, how="left")
    return df_1m.reset_index().fillna(0)

def triple_barrier_labeling(df):
    df = df.copy()
    df["target"] = 0
    for i in range(len(df) - MAX_HOLD):
        entry = df.loc[i, "close"]
        tp = entry * (1 + TP_PCT)
        sl = entry * (1 - SL_PCT)
        for j in range(1, MAX_HOLD+1):
            high = df.loc[i+j, "high"]
            low = df.loc[i+j, "low"]
            if high >= tp:
                df.loc[i, "target"] = 1
                break
            elif low <= sl:
                df.loc[i, "target"] = -1
                break
    return df

def evaluate_preds(y_true, y_pred, returns):
    trades = np.where(y_pred != 0)[0]
    if len(trades) == 0:
        return {"win_rate": 0.0, "profit_factor": 0.0}
    trade_returns = returns[trades] * y_pred[trades] - EXEC_COST
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    win_rate = len(wins) / max(1, len(trades))
    if len(losses) == 0:
        profit_factor = np.inf if len(wins) > 0 else 0.0
    else:
        profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 0.0
    if np.isnan(profit_factor) or np.isinf(profit_factor):
        profit_factor = 0.0
    return {"win_rate": win_rate, "profit_factor": profit_factor}

def walk_forward_train(df, params):
    results = []
    start_idx = 0
    train_days, test_days = 30, 7
    while True:
        train_end = start_idx + train_days*1440
        test_end = train_end + test_days*1440
        if test_end >= len(df):
            break
        train_df = df.iloc[start_idx:train_end]
        test_df = df.iloc[train_end:test_end]
        X_train = train_df.drop(columns=["time","low","high","open","close","volume","target"])
        y_train = train_df["target"]
        X_test = test_df.drop(columns=["time","low","high","open","close","volume","target"])
        y_test = test_df["target"]
        model = LGBMClassifier(**params)
        weights = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=weights)
        y_pred = model.predict(X_test)
        metrics = evaluate_preds(y_test.values, y_pred, test_df["close"].pct_change().shift(-1).fillna(0).values)
        results.append(metrics)
        start_idx += test_days*1440
    avg_pf = np.mean([r["profit_factor"] for r in results])
    return avg_pf

def optuna_objective(trial, df):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
    }
    pf = walk_forward_train(df, params)
    if np.isnan(pf) or np.isinf(pf):
        return 0.0
    return pf

async def main():
    for coin in COINS:
        print(f"📡 Fetching data for {coin}")
        df = await fetch_all_timeframes(coin)
        df = triple_barrier_labeling(df)
        df = df[df["target"] != 0] # remove holds
        if df.empty:
            print(f"⚠️ No trades for {coin}")
            continue
        print(f"🔍 Optimizing {coin}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: optuna_objective(trial, df), n_trials=20)
        best_params = study.best_params
        print(f"🏆 Best params for {coin}: {best_params}")
        # Final train on all data
        X = df.drop(columns=["time","low","high","open","close","volume","target"])
        y = df["target"]
        model = LGBMClassifier(**best_params)
        weights = compute_sample_weight("balanced", y)
        model.fit(X, y, sample_weight=weights)
        joblib.dump(model, Path(f"lgbm_model_{coin.replace('-', '_')}.pkl"))
        with open(f"features_{coin.replace('-', '_')}.txt", "w") as f:
            f.write("\n".join(X.columns))

if __name__ == "__main__":
    
if DIAGNOSTIC_MODE:
    total_trades = sum(trades_results)
    avg_pf = np.mean(PF_results) if PF_results else 0
    avg_wr = np.mean(winrate_results) if winrate_results else 0
    deploy_status = "✅ Deploy Live" if avg_pf >= 1.3 and avg_wr >= 50 and total_trades >= 200 else ("⚠️ Caution" if avg_pf >= 1.0 else "❌ Do Not Deploy")
    print(f"\n[Diagnostic Summary] PF={avg_pf:.2f}, WinRate={avg_wr:.1f}%, Trades={total_trades} => {deploy_status}")
    if avg_pf < 1.0:
        print("Reason: Low PF")
    elif avg_wr < 50:
        print("Reason: Low Win Rate")
    elif total_trades < 200:
        print("Reason: Too few trades")

asyncio.run(main())
