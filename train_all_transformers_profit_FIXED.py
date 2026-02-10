
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Transformer Model (unchanged) ---
class TransformerEntryModel(nn.Module):
    def __init__(self, input_dim, seq_len=48, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._generate_positional_encoding(d_model, seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.class_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.tp_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.sl_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _generate_positional_encoding(self, d_model, seq_len):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = self.input_proj(x)
        pe = self.positional_encoding.to(x.device)
        x = x + pe[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        label_logits = self.class_head(x)
        tp_out = self.tp_head(x)
        sl_out = self.sl_head(x)
        return label_logits, tp_out, sl_out

def simulate_profits(pred_labels, closes, highs, lows, tp_preds, sl_preds, lookahead=8):
    profits = []
    active_trade_profits = []
    active_trade_count = 0

    for i in range(len(pred_labels) - lookahead):
        label = pred_labels[i]
        if label == 0:  # HOLD
            continue

        entry_price = closes[i]
        tp_price = entry_price * (1 + tp_preds[i]) if label == 1 else entry_price * (1 - tp_preds[i])
        sl_price = entry_price * (1 - sl_preds[i]) if label == 1 else entry_price * (1 + sl_preds[i])

        trade_profit = None
        for j in range(1, lookahead + 1):
            high_price = highs[i + j]
            low_price = lows[i + j]

            if label == 1:  # BUY
                if high_price >= tp_price:
                    trade_profit = tp_preds[i]
                    break
                elif low_price <= sl_price:
                    trade_profit = -sl_preds[i]
                    break
            elif label == 2:  # SELL
                if low_price <= tp_price:
                    trade_profit = tp_preds[i]
                    break
                elif high_price >= sl_price:
                    trade_profit = -sl_preds[i]
                    break

        if trade_profit is None:
            if label == 1:
                trade_profit = (closes[i + lookahead] - entry_price) / entry_price
            elif label == 2:
                trade_profit = (entry_price - closes[i + lookahead]) / entry_price

        profits.append(trade_profit)
        active_trade_profits.append(trade_profit)
        active_trade_count += 1

    active_win_rate = float(np.mean(np.array(active_trade_profits) > 0)) if active_trade_count > 0 else 0.0
    val_profit = float(np.mean(profits)) if profits else 0.0
    return val_profit, active_win_rate, active_trade_count

def make_windows(df_scaled, feature_cols, seq_len, start_idx, end_idx):
    """Create windows [start_idx, end_idx) in *row index space* after scaling."""
    X, y_cls, y_tp, y_sl = [], [], [], []
    # Ensure we have room for seq_len
    for i in range(start_idx, end_idx - seq_len):
        X.append(df_scaled[feature_cols].iloc[i:i+seq_len].values)
        y_cls.append(df_scaled['label'].iloc[i+seq_len])
        y_tp.append(df_scaled['tp_pct'].iloc[i+seq_len])
        y_sl.append(df_scaled['sl_pct'].iloc[i+seq_len])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y_cls = torch.tensor(y_cls, dtype=torch.long)
    y_tp = torch.tensor(y_tp, dtype=torch.float32).unsqueeze(-1)
    y_sl = torch.tensor(y_sl, dtype=torch.float32).unsqueeze(-1)
    return X, y_cls, y_tp, y_sl

# --- Training Function (fixed splits/scaler) ---
def train_on_file(file_path, model_dir, scaler_dir, features_dir, seq_len=48, epochs=15, batch_size=64):
    coin_symbol = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path)
    df = df.select_dtypes(include=[np.number])  # keep numeric columns (assumes 'open,high,low,close,volume' present)

    # Columns
    feature_cols = [c for c in df.columns if c not in ['label', 'tp_pct', 'sl_pct']]
    feature_order = feature_cols.copy()
    input_dim = len(feature_cols)

    # --- Time-based split BEFORE scaling/windowing ---
    N = len(df)
    if N <= seq_len + 100:
        raise ValueError(f"Not enough rows for {coin_symbol}: {N}")

    split_idx = int(N * 0.85)  # 85% past for train, 15% future for val
    # We want windows to end at < split_idx for train; start at split_idx for val
    # Fit scaler on TRAIN SLICE ONLY
    scaler = StandardScaler()
    train_slice = df.iloc[:split_idx].copy()
    scaler.fit(train_slice[feature_cols])

    # Transform train and val separately with the train-fitted scaler
    df_train = train_slice.copy()
    df_train[feature_cols] = scaler.transform(df_train[feature_cols])

    df_val = df.iloc[split_idx:].copy()
    df_val[feature_cols] = scaler.transform(df_val[feature_cols])

    # Save artifacts
    os.makedirs(scaler_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_dir, f"{coin_symbol}_scaler.pkl"))
    joblib.dump(feature_order, os.path.join(features_dir, f"{coin_symbol}_features.pkl"))

    # --- Build windows WITHIN each slice (prevents cross-split overlap) ---
    X_train, y_cls_train, y_tp_train, y_sl_train = make_windows(df_train, feature_cols, seq_len, 0, len(df_train))
    X_val, y_cls_val, y_tp_val, y_sl_val = make_windows(df_val, feature_cols, seq_len, 0, len(df_val))

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(f"Windowing failed for {coin_symbol}: train={len(X_train)} val={len(X_val)}")

    train_loader = DataLoader(TensorDataset(X_train, y_cls_train, y_tp_train, y_sl_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_cls_val, y_tp_val, y_sl_val), batch_size=batch_size, shuffle=False)

    # --- Model/optim ---
    model = TransformerEntryModel(input_dim=input_dim, seq_len=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_cls = nn.CrossEntropyLoss()
    loss_reg = nn.MSELoss()

    best_val_profit = -1e9
    for epoch in range(epochs):
        model.train()
        loss_cls_total = 0.0
        loss_tp_total = 0.0
        loss_sl_total = 0.0
        for xb, yb_cls, yb_tp, yb_sl in train_loader:
            optimizer.zero_grad()
            out_cls, out_tp, out_sl = model(xb)
            loss_cls_val = loss_cls(out_cls, yb_cls)
            loss_tp_val = loss_reg(out_tp, yb_tp)
            loss_sl_val = loss_reg(out_sl, yb_sl)
            loss = loss_cls_val + loss_tp_val + loss_sl_val
            loss.backward()
            optimizer.step()
            loss_cls_total += float(loss_cls_val.item())
            loss_tp_total += float(loss_tp_val.item())
            loss_sl_total += float(loss_sl_val.item())

        # --- Validation ---
        model.eval()
        val_pred_cls, val_pred_tp, val_pred_sl = [], [], []
        with torch.no_grad():
            for xb, yb_cls, yb_tp, yb_sl in val_loader:
                out_cls, out_tp, out_sl = model(xb)
                preds = torch.argmax(out_cls, dim=1)
                val_pred_cls.extend(preds.tolist())
                val_pred_tp.extend(out_tp.squeeze().tolist())
                val_pred_sl.extend(out_sl.squeeze().tolist())

        # Align closes/highs/lows with VAL windows:
        # For df_val with M rows, we have len(X_val) = M - seq_len windows; labels correspond to indices [seq_len .. M-1]
        closes = df_val["close"].values[seq_len:]
        highs = df_val["high"].values[seq_len:]
        lows = df_val["low"].values[seq_len:]

        val_profit, win_rate, trades = simulate_profits(
            np.array(val_pred_cls),
            closes,
            highs,
            lows,
            np.array(val_pred_tp),
            np.array(val_pred_sl)
        )

        total_loss = loss_cls_total + loss_tp_total + loss_sl_total
        print(f"📈 {coin_symbol} | Epoch {epoch+1}/{epochs} | Loss: {total_loss:.6f} | CLS: {loss_cls_total:.4f} | TP: {loss_tp_total:.4f} | SL: {loss_sl_total:.4f}")
        print(f"💰 Val Profit (avg return): {val_profit*100:.4f}% | Win Rate: {win_rate*100:.2f}% | Trades: {trades}")

        if val_profit > best_val_profit:
            best_val_profit = val_profit
            os.makedirs(model_dir, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "feature_order": feature_order
            }, os.path.join(model_dir, f"{coin_symbol}_transformer_tp_sl.pth"))
            print(f"✅ New best — Val Profit: {best_val_profit*100:.2f}% | model saved")

# --- Entry Point ---
def train_all_models(data_dir="C:/Users/natec/datasets_macro_training5",
                     model_dir="models_transformer_p2_fixed",
                     scaler_dir="models_transformer_p2_fixed/scalers",
                     features_dir="models_transformer_p2_fixed/features"):
    os.makedirs(model_dir, exist_ok=True)
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            path = os.path.join(data_dir, fname)
            try:
                train_on_file(path, model_dir, scaler_dir, features_dir)
            except Exception as e:
                print(f"❌ Failed on {fname}: {e}")

if __name__ == "__main__":
    train_all_models()
