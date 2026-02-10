import pandas as pd
import numpy as np
import asyncio, aiohttp, joblib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from lightgbm import LGBMRegressor

# =========================
# CONFIG
# =========================
BASE_URL = "https://api.exchange.coinbase.com"
COINS = [   "ENA-USD", "ETH-USD", "XRP-USD", "SOL-USD", "DOGE-USD", "ADA-USD", "XLM-USD", "SUI-USD",
    "BCH-USD", "LINK-USD", "HBAR-USD", "LTC-USD", "AVAX-USD", "SHIB-USD", "UNI-USD", "DOT-USD",
    "CRO-USD",  "AAVE-USD", "NEAR-USD", "ETC-USD", "ONDO-USD",
    "APT-USD", "ICP-USD", "ALGO-USD", "ARB-USD", "VET-USD",
    "ATOM-USD", "POL-USD", "BONK-USD", "RENDER-USD",
    "FET-USD", "SEI-USD", "ENA-USD", "FIL-USD", "QNT-USD", "LSETH-USD", "INJ-USD",
    "CRV-USD", "STX-USD", "TIA-USD"]

tf_seconds_map = {"1m": 60}
CANDLE_LIMIT = 300
LOOKBACK_DAYS = 60  # faster testing

# MFE/MAE params
MAX_HOLD = 60
ATR_PCT_MIN = 0.004
ADX_MIN = 25
MFE_THRESHOLD = 0.005  # 0.5% move required to count as a win

TRAIN_DAYS = 30
TEST_DAYS = 7
MIN_PF = 3
MIN_WINRATE = 0.4

RESULTS_CSV = "results2.csv"
WHITELIST_CSV = "whitelist2.csv"

LGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)

REQUEST_DELAY = 0.35

# =========================
# FETCH FUNCTIONS
# =========================
async def fetch_candles_chunked(session, coin, tf_sec, start, end):
    await asyncio.sleep(REQUEST_DELAY)
    url = f"{BASE_URL}/products/{coin}/candles"
    params = {"granularity": tf_sec, "start": start.isoformat(), "end": end.isoformat()}
    async with session.get(url, params=params) as resp:
        data = await resp.json()
        if isinstance(data, dict) and "message" in data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df.sort_values("time")

async def fetch_all_candles(coin, tf_sec):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=LOOKBACK_DAYS)
    all_data = []
    async with aiohttp.ClientSession() as session:
        while start_time < end_time:
            chunk_end = min(start_time + timedelta(seconds=tf_sec * CANDLE_LIMIT), end_time)
            df = await fetch_candles_chunked(session, coin, tf_sec, start_time, chunk_end)
            if not df.empty:
                all_data.append(df)
            start_time = chunk_end
    if all_data:
        return pd.concat(all_data).drop_duplicates(subset=["time"]).reset_index(drop=True)
    return pd.DataFrame()

async def fetch_one_minute_data(coin):
    df = await fetch_all_candles(coin, tf_seconds_map["1m"])
    return add_indicators(df).fillna(0)

# =========================
# INDICATORS
# =========================

def apply_cooldown(pred_dirs, cooldown_bars):
    """
    Apply cooldown to trade predictions.
    pred_dirs: array of -1 (short), 0 (no trade), or 1 (long)
    cooldown_bars: number of bars to skip after a trade
    """
    pred_dirs = pred_dirs.copy()
    last_trade_idx = -cooldown_bars - 1

    for i in range(len(pred_dirs)):
        if pred_dirs[i] != 0 and (i - last_trade_idx) <= cooldown_bars:
            pred_dirs[i] = 0
        elif pred_dirs[i] != 0:
            last_trade_idx = i

    return pred_dirs


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
# MFE/MAE LABELING
# =========================
# =========================
# MFE/MAE LABELING (Long + Short)
# =========================
def mfe_mae_labeling(df):
    mfe_long, mae_long = [], []
    mfe_short, mae_short = [], []
    atr_ok_list, adx_ok_list = [], []

    for i in range(len(df)):
        if i + 1 >= len(df):
            mfe_long.append(np.nan)
            mae_long.append(np.nan)
            mfe_short.append(np.nan)
            mae_short.append(np.nan)
            atr_ok_list.append(False)
            adx_ok_list.append(False)
            continue

        atr_ok = df["atr_norm"].iloc[i] >= ATR_PCT_MIN
        adx_ok = df["adx"].iloc[i] >= ADX_MIN
        atr_ok_list.append(atr_ok)
        adx_ok_list.append(adx_ok)

        entry_price = df["close"].iloc[i]
        end_idx = min(i + MAX_HOLD, len(df) - 1)

        future_high = df["high"].iloc[i+1:end_idx+1].max()
        future_low = df["low"].iloc[i+1:end_idx+1].min()

        # Long perspective
        mfe_l = (future_high - entry_price) / entry_price
        mae_l = (future_low - entry_price) / entry_price

        # Short perspective
        mfe_s = (entry_price - future_low) / entry_price
        mae_s = (entry_price - future_high) / entry_price

        mfe_long.append(mfe_l)
        mae_long.append(mae_l)
        mfe_short.append(mfe_s)
        mae_short.append(mae_s)

    df["MFE_long"] = mfe_long
    df["MAE_long"] = mae_long
    df["MFE_short"] = mfe_short
    df["MAE_short"] = mae_short
    df["atr_ok"] = atr_ok_list
    df["adx_ok"] = adx_ok_list

    return df.dropna(subset=["MFE_long", "MFE_short"])



# =========================
# REGRESSION EVALUATION
# =========================
# =========================
# EVALUATION (Long + Short)
# =========================
def evaluate_preds_long_short(y_true_long, y_true_short, pred_long, pred_short, mfe_threshold, cooldown_bars=60):
    trades_idx = []
    trade_dirs = []  # 1 = long, -1 = short
    actual_results = []

    for i in range(len(pred_long)):
        if pred_long[i] >= mfe_threshold or pred_short[i] >= mfe_threshold and pred_long[i] < 0.002:
            if pred_long[i] >= pred_short[i]:
                trade_dirs.append(1)
                actual_results.append(y_true_long[i])
            else:
                trade_dirs.append(-1)
                actual_results.append(y_true_short[i])
        else:
            trade_dirs.append(0)
            actual_results.append(0)

    # Apply cooldown
    trade_dirs = np.array(trade_dirs)
    trade_dirs = apply_cooldown(trade_dirs, cooldown_bars)

    # Rebuild trades_idx and actual_results based on cooldown-adjusted signals
    trades_idx = np.where(trade_dirs != 0)[0]
    actual_results = np.array(actual_results)[trades_idx]

    if len(trades_idx) == 0:
        return {"win_rate": 0, "profit_factor": 0, "sharpe": 0, "trades": 0}

    wins = actual_results[actual_results >= mfe_threshold]
    losses = actual_results[actual_results < mfe_threshold]

    win_rate = len(wins) / max(1, len(trades_idx))
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
    sharpe = (wins.mean() - abs(losses.mean())) / (
        np.std(np.concatenate([wins, losses])) + 1e-9
    ) * np.sqrt(252 * 24 * 60)

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "trades": len(trades_idx)
    }


# =========================
# WALK-FORWARD REGRESSION
# =========================
# =========================
# WALK-FORWARD REGRESSION (Long + Short Models)
# =========================
def walk_forward_train_long_short(df, params, mfe_threshold=MFE_THRESHOLD, cooldown_bars=60):
    results = []
    start_idx = 0
    final_model_long, final_model_short = None, None

    while True:
        train_end = start_idx + TRAIN_DAYS * 1440
        test_end = train_end + TEST_DAYS * 1440
        if test_end >= len(df):
            break

        train_df = df.iloc[start_idx:train_end]
        test_df = df.iloc[train_end:test_end]

        if train_df.empty or test_df.empty:
            start_idx += TEST_DAYS * 1440
            continue

        # Features
        X_train = extract_features(train_df)
        X_test = extract_features(test_df)

        # Train Long model
        y_train_long = train_df["MFE_long"]
        model_long = LGBMRegressor(**params)
        model_long.fit(X_train, y_train_long)

        # Train Short model
        y_train_short = train_df["MFE_short"]
        model_short = LGBMRegressor(**params)
        model_short.fit(X_train, y_train_short)

        # Predict
        pred_long = model_long.predict(X_test)
        pred_short = model_short.predict(X_test)

        metrics = evaluate_preds_long_short(
            test_df["MFE_long"].values,
            test_df["MFE_short"].values,
            pred_long,
            pred_short,
            mfe_threshold,
            cooldown_bars
        )

        results.append(metrics)
        final_model_long, final_model_short = model_long, model_short
        start_idx += TEST_DAYS * 1440

    if not results:
        return {"win_rate": 0, "profit_factor": 0, "sharpe": 0, "trades": 0}, None, None

    avg_metrics = {k: np.mean([r[k] for r in results]) for k in results[0]}
    return avg_metrics, final_model_long, final_model_short


# =========================
# MAIN
# =========================
# =========================
# MAIN TRAINER (Save Long + Short Models)
# =========================
async def main():
    results_list = []
    whitelist = []
    for coin in tqdm(COINS, desc="Coins", position=0):
        try:
            df = await fetch_one_minute_data(coin)
            if df.empty:
                print(f"No data for {coin}")
                continue

            labeled_df = mfe_mae_labeling(df.copy())
            if labeled_df.empty:
                print(f"No labels for {coin}")
                continue

            metrics, model_long, model_short = walk_forward_train_long_short(
                labeled_df, LGB_PARAMS
            )

            results_list.append({"coin": coin, **metrics})

            if metrics["profit_factor"] >= MIN_PF:
                joblib.dump(model_long, Path(f"lgbm_model2_{coin.replace('-', '_')}_long.pkl"))
                joblib.dump(model_short, Path(f"lgbm_model2_{coin.replace('-', '_')}_short.pkl"))
                whitelist.append(coin)

            print(f"{coin} → WinRate={metrics['win_rate']:.2f}, PF={metrics['profit_factor']:.2f}, "
                  f"Sharpe={metrics['sharpe']:.2f}, Trades={metrics['trades']}")

        except Exception as e:
            print(f"⚠️ Skipping {coin} due to error: {e}")
            continue

    pd.DataFrame(results_list).to_csv(RESULTS_CSV, index=False)
    pd.DataFrame({"coin": whitelist}).to_csv(WHITELIST_CSV, index=False)
    print(f"\n=== Results saved to {RESULTS_CSV} ===")
    print(f"=== Whitelist saved to {WHITELIST_CSV} ===")


if __name__ == "__main__":
    asyncio.run(main())
