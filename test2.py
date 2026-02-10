import os
import pandas as pd
import numpy as np
import requests
import torch
import telegram
from datetime import datetime, timezone
from super_set import calculate_features, calculate_fib_levels
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

# === CONFIG ===
COINS = ["BTC-USD", "ETH-USD", "XRP-USD"]
SEQ_LEN = 60
CONF_THRESHOLD = 0.70
CHAT_ID = "7967738614"
BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
MODELS_PATH = "models_entry_1h"
TPSL_MODELS_PATH = "models_tp_sl_1h"
bot = telegram.Bot(token=BOT_TOKEN)

# === EXACT TRAINING FEATURES ===
# 50 features for ENTRY model (full training set)
TRAINING_FEATURES_ENTRY = [
    'low', 'high', 'open', 'close', 'volume', 'rsi', 'adx', 'plus_di', 'minus_di',
    'macd_diff', 'ema10', 'ema10_slope', 'ema20', 'ema20_slope', 'ema50', 'ema50_slope',
    'ema200', 'ema200_slope', 'atr', 'vol_change', 'body_wick_ratio', 'above_ema200',
    'fib_long_0', 'fib_long_236', 'fib_long_382', 'fib_long_5', 'fib_long_618', 'fib_long_786', 'fib_long_1',
    'fib_med_0', 'fib_med_236', 'fib_med_382', 'fib_med_5', 'fib_med_618', 'fib_med_786', 'fib_med_1',
    'fib_short_0', 'fib_short_236', 'fib_short_382', 'fib_short_5', 'fib_short_618', 'fib_short_786', 'fib_short_1',
    'btc_close', 'btc_return', 'coin_return', 'rel_strength', 'btc_dominance'
]

# 48 features for TPSL model (removed btc_dominance & btc_close)
TRAINING_FEATURES_TPSL = [f for f in TRAINING_FEATURES_ENTRY if f not in ("btc_dominance", "btc_close")]


# === MODEL ARCHITECTURE ===
class PriceActionModel(nn.Module):
    def __init__(self, input_size=50, hidden_size=128, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# === MODEL LOADER ===
def load_model(path, output_size=3, input_size=50):
    torch.serialization.add_safe_globals([StandardScaler, np._core.multiarray.scalar])
    checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    model = PriceActionModel(input_size=input_size, hidden_size=128, num_layers=2, output_size=output_size)
    scaler = None
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        scaler = checkpoint.get("scaler", None)
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, scaler



# === FETCH LIVE CANDLES ===
def fetch_live_candles(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    params = {"granularity": 3600}
    r = requests.get(url, params=params)
    data = r.json()
    if isinstance(data, list) and data:
        df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.sort_values("time", inplace=True)
        return df
    return None

from datetime import datetime, timezone

from datetime import datetime, timezone

def fetch_btcd_data():
    try:
        url = "https://api.coingecko.com/api/v3/global"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        btc_dominance = data.get("data", {}).get("market_cap_percentage", {}).get("btc", None)
        if btc_dominance is None:
            print("⚠️ BTC Dominance not found in API response")
            return pd.DataFrame(columns=["time", "btc_dominance"])
        now_utc = datetime.now(timezone.utc)
        return pd.DataFrame([{"time": now_utc, "btc_dominance": btc_dominance}])
    except Exception as e:
        print(f"❌ Error fetching BTC dominance: {e}")
        return pd.DataFrame(columns=["time", "btc_dominance"])


    
# === PREPARE LIVE DATA ===
def prepare_live_data(symbol):
    df = fetch_live_candles(symbol)
    if df is None or df.empty:
        return None

    df = calculate_features(df)
    df = calculate_fib_levels(df, 720, "fib_long")
    df = calculate_fib_levels(df, 360, "fib_med")
    df = calculate_fib_levels(df, 168, "fib_short")

    btc_df = fetch_live_candles("BTC-USD")
    btc_df = calculate_features(btc_df)
    btc_df['btc_return'] = btc_df['close'].pct_change()
    btc_df = btc_df[['time', 'close', 'btc_return']].rename(columns={'close': 'btc_close'})

    df = df.merge(btc_df, on='time', how='left')
    df['coin_return'] = df['close'].pct_change()
    df['rel_strength'] = df['coin_return'] - df['btc_return']

    btcd_df = fetch_btcd_data()
    if not btcd_df.empty:
        latest_btcd = btcd_df["btc_dominance"].iloc[0]
        df["btc_dominance"] = latest_btcd
    else:
        df["btc_dominance"] = np.nan

    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df



# === MAIN PREDICTION LOOP ===
def check_ai_signals():
    print(f"🔍 Checking AI signals at {datetime.utcnow()} UTC")

    for coin in COINS:
        print(f"\n📌 Processing {coin} ...")
        df = prepare_live_data(coin)
        if df is None or df.empty:
            print(f"❌ No data for {coin}")
            continue

        # ENTRY MODEL (50 features)
        entry_model, entry_scaler = load_model(
            f"{MODELS_PATH}/{coin.replace('-', '_')}_entry.pt",
            output_size=3,
            input_size=len(TRAINING_FEATURES_ENTRY)
        )
        latest_data_entry = df[TRAINING_FEATURES_ENTRY].values[-SEQ_LEN:]
        if entry_scaler:
            latest_data_entry = entry_scaler.transform(latest_data_entry)
        X_entry = torch.tensor(latest_data_entry, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            entry_probs = torch.softmax(entry_model(X_entry), dim=1).numpy()[0]

        entry_signal = np.argmax(entry_probs)
        entry_conf = entry_probs[entry_signal]

        if entry_conf >= CONF_THRESHOLD and entry_signal in [1, 2]:
            direction = "BUY" if entry_signal == 1 else "SELL"

            # TPSL MODEL (48 features)
            tpsl_model, tpsl_scaler = load_model(
                f"{TPSL_MODELS_PATH}/{coin.replace('-', '_')}_tp_sl.pt",
                output_size=2,
                input_size=len(TRAINING_FEATURES_TPSL)
            )
            latest_data_tpsl = df[TRAINING_FEATURES_TPSL].values[-SEQ_LEN:]
            if tpsl_scaler:
                latest_data_tpsl = tpsl_scaler.transform(latest_data_tpsl)
            X_tpsl = torch.tensor(latest_data_tpsl, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                tp_pct, sl_pct = tpsl_model(X_tpsl).numpy()[0]

            msg = f"🚨 {coin} {direction} | Conf: {entry_conf:.2f} | TP: {tp_pct:.3f} | SL: {sl_pct:.3f}"
            bot.send_message(chat_id=CHAT_ID, text=msg)
            print(msg)
        else:
            print(f"ℹ No high-confidence entry for {coin} (Conf: {entry_conf:.2f})")



# === RUN ONCE ===
if __name__ == "__main__":
    check_ai_signals()
