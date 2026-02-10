import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import StandardScaler
from torch.serialization import add_safe_globals


# Allow PyTorch to unpickle StandardScaler
add_safe_globals([StandardScaler])

# ==== USER CONFIG ====
coins = [
     "ETH-USD", "XRP-USD", "SOL-USD", "DOGE-USD", "ADA-USD", "XLM-USD", "SUI-USD",
    "BCH-USD", "LINK-USD", "HBAR-USD", "LTC-USD", "AVAX-USD", "SHIB-USD", "UNI-USD", "DOT-USD",
    "CRO-USD",  "AAVE-USD", "NEAR-USD", "ETC-USD", "ONDO-USD",
    "APT-USD", "ICP-USD", "ALGO-USD", "ARB-USD", "VET-USD",
    "ATOM-USD", "POL-USD", "BONK-USD", "RENDER-USD",
    "FET-USD", "SEI-USD", "FLR-USD", "FIL-USD", "QNT-USD", "LSETH-USD", "INJ-USD",
    "CRV-USD", "STX-USD", "TIA-USD"
]
model_dir = "models_lstm2"
train_csv = "datasets_macro_training2/AAVE_USD.csv"
seq_len = 60
strong_threshold = 0.50
minute_delay = 2
needed_hours = 1000
telegram_token = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
chat_id = "-4916911067"

#TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
#CHAT_ID = "7967738614"

#personal chat id:"7967738614"
#channel chat id:
#"-4916911067"

def format_price(price):
    return f"{price:.6f}" if price < 1 else f"{price:.2f}"


# ==== TELEGRAM ====
def send_telegram_message(message: str):
    try:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ Telegram send error: {e}")

# ==== LSTM MODEL ====
class LSTMEntryModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super(LSTMEntryModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_entry = nn.Linear(hidden_dim, 3)  # HOLD, BUY, SELL
        self.fc_tp = nn.Linear(hidden_dim, 1)     # TP %
        self.fc_sl = nn.Linear(hidden_dim, 1)     # SL %

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        entry_out = self.fc_entry(out)
        tp_out = self.fc_tp(out)
        sl_out = self.fc_sl(out)
        return entry_out, tp_out, sl_out

# ==== FETCH CANDLES WITH CHUNKING ====
def fetch_latest_candles(symbol, needed=needed_hours, granularity=3600):
    all_data = []
    end = datetime.now(timezone.utc)
    limit = 300
    hours_per_fetch = limit * (granularity / 3600)

    while len(all_data) < needed:
        start = end - timedelta(hours=hours_per_fetch - 1)
        
        url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
        params = {
            "granularity": granularity,
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        r = requests.get(url, params=params)
        try:
            data = r.json()
        except Exception:
            print(f"❌ JSON parse fail for {symbol}")
            break

        if not isinstance(data, list) or len(data) == 0:
            print(f"⚠ No data returned for {symbol} between {start} and {end}")
            break

        all_data.extend(data)
        end = start  # move backward without overlap

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


    df = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df



# ==== FEATURE ENGINEERING ====
def calculate_fib_levels(df, lookback_hours, prefix):
    df[f"{prefix}_fib_high"] = df["high"].rolling(window=lookback_hours).max()
    df[f"{prefix}_fib_low"] = df["low"].rolling(window=lookback_hours).min()
    df[f"{prefix}_fib_range"] = df[f"{prefix}_fib_high"] - df[f"{prefix}_fib_low"]
    for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
        df[f"{prefix}_fib_{level}"] = df[f"{prefix}_fib_high"] - (df[f"{prefix}_fib_range"] * level)
    return df

def calculate_features(df):
    df["returns"] = df["close"].pct_change()

    for span in [5, 10, 20, 50, 100, 200]:
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()
        df[f"sma_{span}"] = df["close"].rolling(window=span).mean()

    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    df["rsi"] = compute_rsi(df["close"], 14)

    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    df["atr"] = tr.rolling(14).mean()

    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["boll_upper"] = sma20 + (std20 * 2)
    df["boll_lower"] = sma20 - (std20 * 2)

    df = calculate_fib_levels(df, 720, "fib720")
    df = calculate_fib_levels(df, 360, "fib360")
    df = calculate_fib_levels(df, 168, "fib168")

    df.fillna(0, inplace=True)
    return df

def format_price(symbol, price):
    """
    Format price based on coin value.
    """
    if price >= 100:       # Large prices like BTC, ETH
        return f"{price:.2f}"
    elif price >= 1:       # Mid prices like SOL, LTC
        return f"{price:.4f}"
    else:                  # Small prices like DOGE, XRP, SHIB
        return f"{price:.8f}"


def compute_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==== MERGE BTC FEATURES ====
def merge_with_btc(df_coin, df_btc):
    # Ensure datetime UTC
    df_coin["time"] = pd.to_datetime(df_coin["time"], utc=True)
    df_btc["time"] = pd.to_datetime(df_btc["time"], utc=True)

    # Sort both
    df_coin = df_coin.sort_values("time")
    df_btc = df_btc.sort_values("time")

    # Rename BTC columns to avoid conflicts
    df_btc_renamed = df_btc.rename(columns=lambda x: f"btc_{x}" if x != "time" else x)

    # Merge allowing approximate matches
    merged = pd.merge_asof(
        df_coin, df_btc_renamed,
        on="time",
        direction="backward",
        tolerance=pd.Timedelta("1h")  # allow matching within 1 hour
    )

    # Drop rows where BTC data missing
    merged.dropna(subset=["btc_close"], inplace=True)

    return merged


# ==== MODEL LOADING ====
import joblib

def load_model(model_path, scaler_path, input_dim):
    model = LSTMEntryModel(input_dim=input_dim)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler

# ==== PREDICT SIGNAL ====
def predict_signal(df, model, scaler, feature_order, latest_close):
    # Ensure only training features are scaled
    df = df[feature_order]
    X_scaled = scaler.transform(df.values)
    seq_input = X_scaled[-seq_len:]
    seq_input = torch.tensor(seq_input, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        entry_out, tp_out, sl_out = model(seq_input)
        probs = torch.softmax(entry_out, dim=1).cpu().numpy()[0]
        entry_class = torch.argmax(entry_out, dim=1).item()

        # ✅ Define label first
        entry_map = {0: ("HOLD", probs[0]), 1: ("BUY", probs[1]), 2: ("SELL", probs[2])}
        label, confidence = entry_map[entry_class]

        tp_pct = tp_out.item()
        sl_pct = sl_out.item()
       # print(f"TP{tp_pct}")
        if tp_pct<0 or sl_pct<0:
            label == "HOLD"
        # ✅ Apply correct TP/SL logic based on trade direction
        if label == "BUY" and tp_pct>0 and sl_pct>0:
            tp_price = latest_close * (1 + tp_pct)
            sl_price = latest_close * (1 - sl_pct)
        elif label == "SELL" and tp_pct>0 and sl_pct>0:
            tp_price = latest_close * (1 - tp_pct)
            sl_price = latest_close * (1 + sl_pct)
        else:  # HOLD or no trade
            tp_price = latest_close
            sl_price = latest_close
            label == "HOLD"

    strong = confidence >= strong_threshold
    return label, confidence, strong, tp_price, sl_price


# ==== MAIN LOOP ====
def run_live():
    print("🚀 Multi-coin LSTM signal bot started...")
    last_signals = {}
    first_run = True
    # Load features from training CSV
    # Load features from training CSV
    feature_order = list(pd.read_csv(train_csv).drop(
        columns=["label", "tp_pct", "sl_pct"], errors="ignore"
        ).columns)

# Remove btc_return (model trained without it)
    if "btc_return" in feature_order:
        print("🧹 Removing btc_return from feature_order to match trained model")
        feature_order.remove("btc_return")

# Remove time if somehow included

        

# Drop btc_return if it's in the list (model trained without it)
    if "btc_return" in feature_order:
        print("🧹 Removing btc_return from feature_order to match trained model")
        feature_order.remove("btc_return")


    while True:
        now = datetime.now(timezone.utc)
        if not first_run:
            next_run = (now.replace(minute=0, second=0, microsecond=0) 
                        + timedelta(hours=1, minutes=minute_delay))
            wait_seconds = (next_run - datetime.now(timezone.utc)).total_seconds()
            if wait_seconds > 0:
                print(f"⏳ Waiting {int(wait_seconds)}s until next run at {next_run.strftime('%H:%M:%S UTC')}")
                time.sleep(wait_seconds)

        try:
            print("📥 Fetching BTC data...")
            df_btc = fetch_latest_candles("BTC-USD", needed=needed_hours)
            df_btc = calculate_features(df_btc)

            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            message_lines = [f"📊 LSTM Signals — {timestamp}"]

            for symbol in coins:
                try:
                    model_file = os.path.join(model_dir, f"{symbol.lower().replace('-', '_')}_lstm_tp_sl.pth")
                    if not os.path.exists(model_file):
                        print(f"⚠ No model for {symbol}, skipping.")
                        continue

                    df_coin = fetch_latest_candles(symbol, needed=needed_hours)
                    if df_coin.empty:
                        print(f"⚠ No coin data for {symbol}, skipping.")
                        continue
                    df_coin = calculate_features(df_coin)
                    df_merged = merge_with_btc(df_coin, df_btc)


                    if df_merged.empty:
                        print(f"⚠ No merged data for {symbol}, skipping.")
                        continue
                    extra_cols = [c for c in df_merged.columns if c not in feature_order]
                    if extra_cols:
                       # print(f"🧹 Dropping unused extra columns: {extra_cols}")
                        df_merged.drop(columns=extra_cols, inplace=True, errors="ignore")

# Force exact training order
                    # Ensure only numeric columns remain
                    df_merged = df_merged.apply(pd.to_numeric, errors="coerce")
    
                    df_merged = df_merged.reindex(columns=feature_order, fill_value=0)

                    # Ensure we have enough rows for seq_len
                    if len(df_merged) < seq_len:
                        print(f"⚠ Not enough data for {symbol} after filtering ({len(df_merged)} rows), skipping.")
                        continue


                    # === DIAGNOSTIC: Compare live vs training columns ===
                    live_cols = list(df_merged.columns)
                    train_cols = list(feature_order)
                    extra_cols = [c for c in live_cols if c not in train_cols]
                    missing_cols = [c for c in train_cols if c not in live_cols]
                    print(f"🛠 TRAIN features: {len(train_cols)} | LIVE features: {len(live_cols)}")
                    if extra_cols:
                        print(f"   ➕ Extra columns in LIVE: {extra_cols}")
                    if missing_cols:
                        print(f"   ➖ Missing columns in LIVE: {missing_cols}")

                    # === AUTO-FIX: Drop extra columns ===
                    if extra_cols:
                        print(f"🧹 Dropping extra columns: {extra_cols}")
                        df_merged.drop(columns=extra_cols, inplace=True, errors="ignore")

                    # ✅ Match training order exactly
                    df_merged = df_merged.reindex(columns=feature_order, fill_value=0)
                    if "close" in df_merged.columns:
                        latest_close = float(df_merged.iloc[-1]["close"])
                    else:
                        print(f"⚠ No 'close' column found for {symbol}, skipping.")
                        continue
                    # ✅ Always use training feature count
                    model_file = os.path.join(model_dir, f"{symbol.replace('-', '_')}_lstm_tp_sl.pth")
                    scaler_file = os.path.join(model_dir, "scalers", f"{symbol.replace('-', '_')}_scaler.pkl")
                    feature_file = os.path.join(model_dir, "features", f"{symbol.replace('-', '_')}_features.pkl")

                    feature_order = joblib.load(feature_file)
                    model, scaler = load_model(model_file, scaler_file, input_dim=len(feature_order))



                    label, confidence, strong, tp_price, sl_price = predict_signal(df_merged, model, scaler, feature_order, latest_close)
                    # ❌ Skip if TP/SL is invalid for the trade direction
                    if (
                        (label == "BUY" and (tp_price <= latest_close or sl_price >= latest_close)) or
                        (label == "SELL" and (tp_price >= latest_close or sl_price <= latest_close))
                    ):
                        print(f"[SKIPPED] {symbol} | Invalid TP/SL for {label}: TP={tp_price}, SL={sl_price}, Entry={latest_close}")
                        continue

                    current_signal = (label, round(confidence, 3), round(tp_price, 2), round(sl_price, 2))
                    
                    if first_run and label != "HOLD"  and confidence >= 0.70 or (last_signals.get(symbol) != current_signal and label != "HOLD" and confidence >= 0.70):
                        last_signals[symbol] = current_signal

        # Icon for strong signals
                        icon = "✅" if confidence >= 0.80 else "⚠️"

                        

        # Format prices
                        def format_price(p):
                            if p < 0.0001:
                                return f"{p:.8f}"
                            elif p < 0.01:
                                return f"{p:.6f}"
                            elif p < 1:
                                return f"{p:.5f}"
                            elif p < 100:
                                return f"{p:.4f}"
                            else:
                                return f"{p:.1f}"

                        tp_fmt = format_price(tp_price)
                        sl_fmt = format_price(sl_price)
                        entry_price = format_price(df_merged["close"].iloc[-1]) 
                
        # Compose message
                        if strong:
                            line = f"{icon} {label}: {symbol} (p={confidence:.2%}) | Entry: {entry_price} | TP: {tp_fmt} | SL: {sl_fmt}"

                        else:
                            line = f"{symbol}: {label} (p={confidence:.2%}) | Entry: {entry_price} | TP: {tp_fmt} | SL: {sl_fmt}"


                        message_lines.append(line)
                        print(line)
                except Exception as e:
                    print(f"❌ Error for {symbol}: {e}")

            if len(message_lines) > 1:
                send_telegram_message("\n".join(message_lines))

        except Exception as e:
            print(f"❌ Main loop error: {e}")

        first_run = False

if __name__ == "__main__":
    run_live()
