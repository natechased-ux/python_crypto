import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import StandardScaler
import joblib

# === USER CONFIG ===
coins = ["AAVE-USD"]
model_dir = "models_lstm"
dataset_path = "datasets_macro_training/AAVE_USD.csv"
scaler_path = "models_lstm/scalers/AAVE_USD_scaler.pkl"
seq_len = 60
strong_threshold = 0.50
minute_delay = 2
needed_hours = 1000

telegram_token = "YOUR_TELEGRAM_BOT_TOKEN"
chat_id = "YOUR_CHAT_ID"

# === TELEGRAM ===
def send_telegram_message(message: str):
    try:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ Telegram send error: {e}")

# === LSTM MODEL ===
class LSTMEntryModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_entry = nn.Linear(hidden_dim, 3)
        self.fc_tp = nn.Linear(hidden_dim, 1)
        self.fc_sl = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        entry_out = self.fc_entry(out)
        tp_out = self.fc_tp(out)
        sl_out = self.fc_sl(out)
        return entry_out, tp_out, sl_out

# === MAIN LOOP ===
model_file = os.path.join(model_dir, "AAVE_USD.pt")
scaler = joblib.load(scaler_path)
model = LSTMEntryModel(input_dim=dataset_path.shape[1] - 1)  # we adjust this below
model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
model.eval()

last_signal = None

while True:
    try:
        df = pd.read_csv(dataset_path)
        if df.shape[0] < needed_hours:
            print("⏳ Not enough data yet.")
            time.sleep(minute_delay * 60)
            continue

        df = df.drop(columns=["timestamp"], errors="ignore")
        input_data = scaler.transform(df[-seq_len:])
        x = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            entry_out, tp_out, sl_out = model(x)
            probs = torch.softmax(entry_out, dim=1).numpy()[0]
            entry_pred = np.argmax(probs)
            tp_val = tp_out.item()
            sl_val = sl_out.item()

            label = ["HOLD", "BUY", "SELL"][entry_pred]
            confidence = probs[entry_pred]

            if label != "HOLD" and confidence > strong_threshold and last_signal != label:
                message = f"📡 {label} SIGNAL for AAVE_USD\nConfidence: {confidence:.2f}\nTP: {tp_val:.2%}\nSL: {sl_val:.2%}"
                send_telegram_message(message)
                last_signal = label
            else:
                print(f"⏱️ {datetime.now()} - {label} ({confidence:.2f})")

    except Exception as e:
        print(f"❌ Error in main loop: {e}")

    time.sleep(minute_delay * 60)
