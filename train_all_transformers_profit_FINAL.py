
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# --- Transformer Model ---
class TransformerEntryModel(nn.Module):
    def __init__(self, input_dim, seq_len=48, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerEntryModel, self).__init__()
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

# --- Training Function ---
def train_on_file(file_path, model_dir, scaler_dir, features_dir, seq_len=48, epochs=15, batch_size=64):
    coin_symbol = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path)
    df = df.select_dtypes(include=[np.number])  # Drop non-numeric like datetime

    # Features
    feature_cols = [c for c in df.columns if c not in ['label', 'tp_pct', 'sl_pct']]
    feature_order = feature_cols.copy()
    input_dim = len(feature_cols)

    # Normalize
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    os.makedirs(scaler_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_dir, f"{coin_symbol}_scaler.pkl"))
    joblib.dump(feature_order, os.path.join(features_dir, f"{coin_symbol}_features.pkl"))

    # Prepare sequences
    X, y_cls, y_tp, y_sl = [], [], [], []
    for i in range(len(df) - seq_len):
        X.append(df[feature_cols].iloc[i:i+seq_len].values)
        y_cls.append(df['label'].iloc[i+seq_len])
        y_tp.append(df['tp_pct'].iloc[i+seq_len])
        y_sl.append(df['sl_pct'].iloc[i+seq_len])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y_cls = torch.tensor(y_cls, dtype=torch.long)
    y_tp = torch.tensor(y_tp, dtype=torch.float32).unsqueeze(-1)
    y_sl = torch.tensor(y_sl, dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(X, y_cls, y_tp, y_sl)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
    # --- Validation Split ---
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_cls_train, y_cls_val = y_cls[:split_idx], y_cls[split_idx:]
    y_tp_train, y_tp_val = y_tp[:split_idx], y_tp[split_idx:]
    y_sl_train, y_sl_val = y_sl[:split_idx], y_sl[split_idx:]

    train_loader = DataLoader(TensorDataset(X_train, y_cls_train, y_tp_train, y_sl_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_cls_val, y_tp_val, y_sl_val), batch_size=batch_size)

    # Train model
    model = TransformerEntryModel(input_dim=input_dim, seq_len=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_cls = nn.CrossEntropyLoss()
    loss_reg = nn.MSELoss()

    best_val_profit = -1e9
    for epoch in range(epochs):
        model.train()
        loss_cls_total = 0
        loss_tp_total = 0
        loss_sl_total = 0
        for xb, yb_cls, yb_tp, yb_sl in train_loader:
            optimizer.zero_grad()
            out_cls, out_tp, out_sl = model(xb)
            loss_cls_val = loss_cls(out_cls, yb_cls)
            loss_tp_val = loss_reg(out_tp, yb_tp)
            loss_sl_val = loss_reg(out_sl, yb_sl)
            loss = loss_cls_val + loss_tp_val + loss_sl_val
            loss.backward()
            optimizer.step()
            loss_cls_total += loss_cls_val.item()
            loss_tp_total += loss_tp_val.item()
            loss_sl_total += loss_sl_val.item()

        # --- Validation ---
        model.eval()
        val_pred_cls, val_pred_tp, val_pred_sl = [], [], []
        val_true_cls, val_true_tp, val_true_sl = [], [], []
        with torch.no_grad():
            for xb, yb_cls, yb_tp, yb_sl in val_loader:
                out_cls, out_tp, out_sl = model(xb)
                preds = torch.argmax(out_cls, dim=1)
                val_pred_cls.extend(preds.tolist())
                val_pred_tp.extend(out_tp.squeeze().tolist())
                val_pred_sl.extend(out_sl.squeeze().tolist())
                val_true_cls.extend(yb_cls.tolist())
                val_true_tp.extend(yb_tp.squeeze().tolist())
                val_true_sl.extend(yb_sl.squeeze().tolist())

        closes = df["close"].values[seq_len+split_idx:]
        highs = df["high"].values[seq_len+split_idx:]
        lows = df["low"].values[seq_len+split_idx:]

        val_profit, win_rate, trades = simulate_profits(
            np.array(val_pred_cls),
            closes,
            highs,
            lows,
            np.array(val_pred_tp),
            np.array(val_pred_sl)
        )
        print(f"📈 {coin_symbol} | Epoch {epoch+1}/{epochs} | Loss: {loss_cls_total + loss_tp_total + loss_sl_total:.6f} | CLS: {loss_cls_total:.4f} | TP: {loss_tp_total:.4f} | SL: {loss_sl_total:.4f}")
        print(f"💰 Val Profit: {val_profit:.4f}% | Win Rate: {win_rate:.2f}% | Trades: {trades}")



        if val_profit > best_val_profit:
            best_val_profit = val_profit
            os.makedirs(model_dir, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "feature_order": feature_order
            }, os.path.join(model_dir, f"{coin_symbol}_transformer_tp_sl.pth"))
            print(f"💰 Best Val Profit: {best_val_profit*100:.2f}% — Model saved")

# --- Entry Point ---
def train_all_models(data_dir="C:/Users/natec/datasets_macro_training5",
                     model_dir="models_transformer_p2",
                     scaler_dir="models_transformer_p2/scalers",
                     features_dir="models_transformer_p2/features"):
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
