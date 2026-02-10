import os
import pandas as pd
import numpy as np
import glob
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime


# ==============================================
# CONFIGURATION
# ==============================================
ATR_PERIOD = 10
LOOKAHEAD_HOURS = 8
SEQ_LEN = 60
BATCH_SIZE = 64
LR = 1e-3
NUM_EPOCHS = 50
PATIENCE = 5
skip_coins = ["BTC_USD"]


DATA_DIR = "datasets_macro_training"
MODEL_DIR = "models_lstm"
SCALER_DIR = "scalers"
FEATURES_DIR = "features"

# Make sure folders exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)


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

# -----------------------------
# Realistic profit simulation
# -----------------------------
def simulate_profit(pred_labels, closes, highs, lows, tp_preds, sl_preds, lookahead=lookahead):
    """
    pred_labels: np.array of predicted classes (0=HOLD, 1=BUY, 2=SELL)
    closes, highs, lows: np.arrays of price data
    tp_preds, sl_preds: arrays of predicted TP/SL percentages
    lookahead: number of bars to check forward
    """

    profits = []
    active_trade_profits = []
    active_trade_count = 0

    for i in range(len(pred_labels) - lookahead):
        label = pred_labels[i]

        # Skip HOLD trades
        if label == 0:
            continue

        entry_price = closes[i]
        tp_price = entry_price * (1 + tp_preds[i]) if label == 1 else entry_price * (1 - tp_preds[i])
        sl_price = entry_price * (1 - sl_preds[i]) if label == 1 else entry_price * (1 + sl_preds[i])

        trade_profit = None

        for j in range(1, lookahead + 1):
            high_price = highs[i + j]
            low_price = lows[i + j]

            if label == 1:  # BUY trade
                if high_price >= tp_price:
                    trade_profit = tp_preds[i]
                    break
                elif low_price <= sl_price:
                    trade_profit = -sl_preds[i]
                    break
            elif label == 2:  # SELL trade
                if low_price <= tp_price:
                    trade_profit = tp_preds[i]
                    break
                elif high_price >= sl_price:
                    trade_profit = -sl_preds[i]
                    break

        if trade_profit is None:
            # If neither TP nor SL hit, profit = price change at end of lookahead
            if label == 1:
                trade_profit = (closes[i + lookahead] - entry_price) / entry_price
            elif label == 2:
                trade_profit = (entry_price - closes[i + lookahead]) / entry_price

        profits.append(trade_profit)
        active_trade_profits.append(trade_profit)
        active_trade_count += 1

    # Active win rate = only BUY + SELL trades
    if active_trade_count > 0:
        active_win_rate = np.mean(np.array(active_trade_profits) > 0)
    else:
        active_win_rate = 0.0

    val_profit = np.mean(profits) if profits else 0.0
    return val_profit, active_win_rate, active_trade_count


# ==============================================
# TRAINING FUNCTION
# ==============================================
def train_lstm(dataset_path, coin_symbol, model_save_dir, scaler_save_dir, feature_order_save_dir,
               seq_len=60, batch_size=64, lr=1e-3, num_epochs=50, lookahead=lookahead):

    # Load dataset
    df = pd.read_csv(dataset_path)
    df['time'] = pd.to_datetime(df['time'], utc=True)

    # Labels
    if 'entry' in df.columns:
        y_entry = df['entry'].values
    elif 'label' in df.columns:
        y_entry = df['label'].values
    else:
        raise ValueError(f"{coin_symbol} missing both 'entry' and 'label' columns.")

    y_tp = df['tp_pct'].values
    y_sl = df['sl_pct'].values

    # Features
    feature_order = [c for c in df.columns if c not in ['time', 'entry', 'label', 'tp_pct', 'sl_pct']]
    X = df[feature_order].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler and feature order
    atr_suffix = f"ATR{ATR_PERIOD}_LH{LOOKAHEAD_HOURS}"
    joblib.dump(scaler, os.path.join(scaler_save_dir, f"{coin_symbol}_{atr_suffix}_scaler.pkl"))
    joblib.dump(feature_order, os.path.join(feature_order_save_dir, f"{coin_symbol}_{atr_suffix}_features.pkl"))

    # Sequence data
    X_seq, y_entry_seq, y_tp_seq, y_sl_seq, close_seq, high_seq, low_seq = [], [], [], [], [], [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i+seq_len])
        y_entry_seq.append(y_entry[i+seq_len])
        y_tp_seq.append(y_tp[i+seq_len])
        y_sl_seq.append(y_sl[i+seq_len])
        close_seq.append(df.iloc[i+seq_len]['close'])
        high_seq.append(df.iloc[i+seq_len]['high'])
        low_seq.append(df.iloc[i+seq_len]['low'])

    X_seq, y_entry_seq, y_tp_seq, y_sl_seq, close_seq, high_seq, low_seq = map(
        np.array, (X_seq, y_entry_seq, y_tp_seq, y_sl_seq, close_seq, high_seq, low_seq)
    )

    # Train/Val split
    seq_times = df['time'].iloc[seq_len:].reset_index(drop=True)
    split_date = pd.Timestamp("2025-01-01", tz="UTC")
    mask_train = seq_times < split_date
    mask_val = seq_times >= split_date

    X_train, X_val = X_seq[mask_train], X_seq[mask_val]
    y_entry_train, y_entry_val = y_entry_seq[mask_train], y_entry_seq[mask_val]
    y_tp_train, y_tp_val = y_tp_seq[mask_train], y_tp_seq[mask_val]
    y_sl_train, y_sl_val = y_sl_seq[mask_train], y_sl_seq[mask_val]
    close_val = close_seq[mask_val]
    high_val = high_seq[mask_val]
    low_val = low_seq[mask_val]

    print(f"[{coin_symbol}] Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_entry_train, dtype=torch.long),
                      torch.tensor(y_tp_train, dtype=torch.float32),
                      torch.tensor(y_sl_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_entry_val, dtype=torch.long),
                      torch.tensor(y_tp_val, dtype=torch.float32),
                      torch.tensor(y_sl_val, dtype=torch.float32)),
        batch_size=batch_size
    )

    # Model setup
    model = LSTMEntryModel(input_dim=X_seq.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weights = torch.tensor([1.0, 2.0, 2.0])  # HOLD=1, BUY=2, SELL=2
    criterion_entry = nn.CrossEntropyLoss(weight=weights)
    criterion_tp_sl = nn.MSELoss()

    # Training loop
    best_val_profit = -np.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_entry_batch, y_tp_batch, y_sl_batch in train_loader:
            optimizer.zero_grad()
            entry_out, tp_out, sl_out = model(X_batch)
            loss_entry = criterion_entry(entry_out, y_entry_batch)
            loss_tp = criterion_tp_sl(tp_out.squeeze(), y_tp_batch)
            loss_sl = criterion_tp_sl(sl_out.squeeze(), y_sl_batch)
            loss = loss_entry + 0.1 * loss_tp + 0.1 * loss_sl
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds, all_tp_preds, all_sl_preds = [], [], []
        with torch.no_grad():
            for X_batch, _, _, _ in val_loader:
                entry_out, tp_out, sl_out = model(X_batch)
                preds = torch.argmax(entry_out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_tp_preds.extend(tp_out.squeeze().cpu().numpy())
                all_sl_preds.extend(sl_out.squeeze().cpu().numpy())

        # Profit simulation
        val_profit, win_rate, active_trades = simulate_profit(
            np.array(all_preds), close_val, high_val, low_val,
            all_tp_preds, all_sl_preds, lookahead=lookahead
        )

        print(f"[{coin_symbol}] Epoch {epoch+1}/{num_epochs} - Val Profit: {val_profit:.4f} | Win Rate: {win_rate:.2%} | Trades: {active_trades}")

        if val_profit > best_val_profit:
            best_val_profit = val_profit
            patience_counter = 0
            model_path = os.path.join(model_save_dir, f"{coin_symbol}_{atr_suffix}_lstm_tp_sl.pth")
            torch.save(model.state_dict(), model_path)
            print(f"💰 Best Val Profit: {best_val_profit*100:.2f}% — Model saved to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⏹ Early stopping")
                break


# ==============================================
# TRAIN ALL COINS
# ==============================================
import glob
import re

def train_all_datasets(dataset_dir, model_dir, seq_len=60, batch_size=64, lr=1e-3, num_epochs=50):
    files = glob.glob(f"{dataset_dir}/*.csv")
    coin_files = {}

    # Group all files by coin symbol (e.g., ETH_USD)
    for f in files:
        match = re.match(rf"{dataset_dir}/(\w+_\w+)__atr(\d+)__look(\d+)\.csv", f)
        if match:
            coin, atr, lookahead = match.groups()
            coin_files.setdefault(coin, []).append((f, int(atr), int(lookahead)))

    for coin, variations in coin_files.items():
        best_profit = -float('inf')
        best_settings = None

        for dataset_path, atr, lookahead in variations:
            print(f"\n📈 Evaluating {coin} | ATR={atr} | Lookahead={lookahead}")
            val_profit = train_lstm(
                dataset_path, coin, model_dir,
                scaler_save_dir, features_save_dir,
                seq_len=seq_len, batch_size=batch_size, lr=lr,
                num_epochs=num_epochs, lookahead=lookahead  # Pass lookahead to simulate_profit
            )
            if val_profit > best_profit:
                best_profit = val_profit
                best_settings = (atr, lookahead)

        print(f"\n✅ BEST for {coin}: ATR={best_settings[0]}, Lookahead={best_settings[1]} with Profit={best_profit:.2%}")



if __name__ == "__main__":
    train_all_datasets(DATA_DIR, MODEL_DIR, SEQ_LEN, BATCH_SIZE, LR, NUM_EPOCHS)
