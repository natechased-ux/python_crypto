import torch
import pandas as pd
import numpy as np
import joblib
import os
from model_def import LSTMEntryModel
from sklearn.preprocessing import StandardScaler

# 🔧 CONFIG
COINS = ["AAVE-USD"]  # Add your list here
DATA_DIR = "live_data"
MODEL_DIR = "models_lstm"
SCALER_DIR = "scalers"
FEATURE_DIR = "features"
SEQ_LEN = 60
CONF_THRESH = 0.9

def load_latest_data(coin):
    path = os.path.join(DATA_DIR, f"{coin}.csv")
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    return df

def prepare_features(df, scaler, feature_order):
    df = df.copy()
    X = df[feature_order].values
    X_scaled = scaler.transform(X)
    X_seq = [X_scaled[i:i+SEQ_LEN] for i in range(len(X_scaled) - SEQ_LEN)]
    times = df['time'].iloc[SEQ_LEN:].values
    return np.array(X_seq), times

def make_prediction(model, X_seq):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_seq[-1:], dtype=torch.float32)
        entry_out, tp_out, sl_out = model(inputs)
        probs = torch.softmax(entry_out, dim=1).cpu().numpy()[0]
        pred_label = np.argmax(probs)
        confidence = probs[pred_label]
        return pred_label, confidence, tp_out.item(), sl_out.item()

def label_to_text(label):
    return {0: "HOLD", 1: "BUY", 2: "SELL"}.get(label, "UNKNOWN")

# 🔁 Run for each coin
for coin in COINS:
    print(f"\n🔎 Checking {coin}...")

    df = load_latest_data(coin)

    # Load model, scaler, features
    coin_symbol = coin.replace("-", "_")
    model_path = os.path.join(MODEL_DIR, f"{coin_symbol}_lstm_tp_sl.pth")
    scaler_path = os.path.join(SCALER_DIR, f"{coin_symbol}_scaler.pkl")
    features_path = os.path.join(FEATURE_DIR, f"{coin_symbol}_features.pkl")

    model = LSTMEntryModel(input_dim=len(pd.read_pickle(features_path)))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    scaler = joblib.load(scaler_path)
    feature_order = joblib.load(features_path)

    X_seq, times = prepare_features(df, scaler, feature_order)

    if len(X_seq) == 0:
        print(f"⚠️ Not enough data for {coin}")
        continue

    pred_label, conf, tp, sl = make_prediction(model, X_seq)

    if conf >= CONF_THRESH:
        print(f"🚨 {coin} ALERT: {label_to_text(pred_label)} | Confidence: {conf:.2f} | TP: {tp:.3f} | SL: {sl:.3f}")
    else:
        print(f"❌ No confident signal for {coin} (Conf: {conf:.2f})")
