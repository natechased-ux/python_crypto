import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

ATR_PERIOD=12
LOOKAHEAD_HOURS=4

skip_coins = ["BTC_USD"]
# -----------------------------
# LSTM Model
# -----------------------------
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
def simulate_profit(pred_labels, closes, highs, lows, tp_preds, sl_preds, lookahead=4):
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


# -----------------------------
# Train one coin
# -----------------------------
def train_lstm(df, coin_symbol, model_save_dir, scaler_save_dir, feature_order_save_dir,
               seq_len=60, batch_size=64, lr=1e-3, num_epochs=50):

    if isinstance(df, str):
        df = pd.read_csv(df)

    df['time'] = pd.to_datetime(df['time'], utc=True)

    if 'entry' in df.columns:
        y_entry = df['entry'].values
    elif 'label' in df.columns:
        y_entry = df['label'].values
    else:
        raise ValueError(f"{coin_symbol} missing both 'entry' and 'label' columns.")

    y_tp = df['tp_pct'].values
    y_sl = df['sl_pct'].values

    feature_order = [c for c in df.columns if c not in ['time', 'entry', 'label', 'tp_pct', 'sl_pct']]
    X = df[feature_order].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, os.path.join(scaler_save_dir, f"{coin_symbol}_scaler.pkl"))
    joblib.dump(feature_order, os.path.join(feature_order_save_dir, f"{coin_symbol}_features.pkl"))

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

    model = LSTMEntryModel(input_dim=X_seq.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Compute dynamic class weights
    weights = torch.tensor([1, 2.0, 2.0])
    criterion_entry = nn.CrossEntropyLoss(weight=weights)


    criterion_tp_sl = nn.MSELoss()

    best_val_profit = -np.inf
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0

        for X_batch, y_entry_batch, y_tp_batch, y_sl_batch in train_loader:
            optimizer.zero_grad()
            entry_out, tp_out, sl_out = model(X_batch)
            loss_entry = criterion_entry(entry_out, y_entry_batch)
            loss_tp = criterion_tp_sl(tp_out.squeeze(), y_tp_batch)
            loss_sl = criterion_tp_sl(sl_out.squeeze(), y_sl_batch)
            loss = loss_entry + 0.1 * loss_tp + 0.1 * loss_sl
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(entry_out.data, 1)
            total += y_entry_batch.size(0)
            correct += (predicted == y_entry_batch).sum().item()

        train_acc = correct / total

        confidence_threshold = 0.40  # minimum probability to consider a trade

        model.eval()
        correct, total = 0, 0
        all_preds, all_confidences, all_tp_preds, all_sl_preds = [], [], [], []
    
        with torch.no_grad():
            for X_batch, y_entry_batch, y_tp_batch, y_sl_batch in val_loader:
                entry_out, tp_out, sl_out = model(X_batch)
                probs = torch.softmax(entry_out, dim=1)
                conf_vals, predicted = torch.max(probs, 1)

                total += y_entry_batch.size(0)
                correct += (predicted == y_entry_batch).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_confidences.extend(conf_vals.cpu().numpy())
                all_tp_preds.extend(tp_out.squeeze().cpu().numpy())
                all_sl_preds.extend(sl_out.squeeze().cpu().numpy())

        val_acc = correct / total

# Convert to numpy arrays
        pred_labels = np.array(all_preds)
        pred_conf = np.array(all_confidences)

# Confidence mask
        conf_mask = pred_conf >= confidence_threshold
        pred_labels = pred_labels[conf_mask]
        tp_preds_f = np.array(all_tp_preds)[conf_mask]
        sl_preds_f = np.array(all_sl_preds)[conf_mask]

# Now run simulate_profit only on confident trades
        val_profit, win_rate, active_trades = simulate_profit(
            np.array(pred_labels), close_val[conf_mask], high_val[conf_mask], low_val[conf_mask],
            tp_preds_f, sl_preds_f, lookahead=4
        )

# Distribution stats
        total_preds = len(pred_labels)
        buy_count = np.sum(pred_labels == 1)
        sell_count = np.sum(pred_labels == 2)
        hold_count = np.sum(pred_labels == 0)

        buy_pct = (buy_count / total_preds) * 100 if total_preds else 0
        sell_pct = (sell_count / total_preds) * 100 if total_preds else 0
        hold_pct = (hold_count / total_preds) * 100 if total_preds else 0

        active_trade_count = buy_count + sell_count
        print(f"[{coin_symbol}] Epoch {epoch+1}/{num_epochs} - ")
        print(f"[{coin_symbol}] Val Distribution (conf ≥ {confidence_threshold}): "
              f"BUY={buy_pct:.1f}%, SELL={sell_pct:.1f}%, HOLD={hold_pct:.1f}%")
        print(f"[{coin_symbol}] Active Trades: {active_trade_count} / {total_preds}")
        print(f"[{coin_symbol}] Validation Profit: {val_profit:.4f} | Win Rate: {win_rate:.2%}")



        if val_profit > best_val_profit:
            best_val_profit = val_profit
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_save_dir, f"{coin_symbol}_lstm_tp_sl.pth"))
            print(f"💰 Best Val Profit: {best_val_profit*100:.2f}% — Model saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹ Early stopping")
                break

# -----------------------------
# Train all coins
# -----------------------------
def train_all_datasets(data_dir, model_dir, seq_len=60, batch_size=64, lr=1e-3, num_epochs=50):
    os.makedirs(model_dir, exist_ok=True)
    scaler_dir = os.path.join(model_dir, "scalers")
    features_dir = os.path.join(model_dir, "features")
    os.makedirs(scaler_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            coin_symbol = file.replace(".csv", "")
            
            if coin_symbol in skip_coins:
                print(f"⏭ Skipping {coin_symbol} training")
                continue  # Skip training for BTC or other specified coins
            df = pd.read_csv(os.path.join(data_dir, file))
            if not any(col in df.columns for col in ['entry', 'label']):
                print(f"⚠ Skipping {coin_symbol} — missing label column.")
                continue
            print(f"\n📈 Training {coin_symbol}...")
            train_lstm(df, coin_symbol, model_dir, scaler_dir, features_dir,
                       seq_len=seq_len, batch_size=batch_size, lr=lr, num_epochs=num_epochs)

if __name__ == "__main__":
    train_all_datasets("datasets_macro_training", "models_lstm",
                       seq_len=60, batch_size=64, lr=1e-3, num_epochs=50)
