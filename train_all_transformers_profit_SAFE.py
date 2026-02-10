
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

# -----------------
# Config
# -----------------
DATA_DIR = "C:/Users/natec/datasets_macro_training5"
MODEL_SAVE_DIR = "models_transformer_safe"
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.15  # Remaining goes to test
FEE_BPS = 0.0005  # 5 bps
SLIPPAGE_BPS = 0.0005
FEATURE_SHIFT_1BAR = False
EARLY_STOP_PATIENCE = 5
THRESHOLDS = np.arange(0.50, 0.91, 0.05)

# -----------------
# Dataset
# -----------------
class SequenceDataset(Dataset):
    def __init__(self, X, y_class, y_tp, y_sl):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.y_tp = torch.tensor(y_tp, dtype=torch.float32)
        self.y_sl = torch.tensor(y_sl, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_tp[idx], self.y_sl[idx]

# -----------------
# Model
# -----------------
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_class = nn.Linear(hidden_dim, 3)
        self.fc_tp = nn.Linear(hidden_dim, 1)
        self.fc_sl = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc_class(x), self.fc_tp(x), self.fc_sl(x)

# -----------------
# Utility
# -----------------
def simulate_profits(preds, probs, tp_preds, sl_preds, y_class, y_tp, y_sl, threshold):
    balance = 0
    wins = 0
    trades = 0
    for i in range(len(preds)):
        prob = max(probs[i])
        if prob < threshold or preds[i] == 1:  # Hold class
            continue
        trades += 1
        side = 1 if preds[i] == 2 else -1  # 2 = long, 0 = short
        tp = tp_preds[i]
        sl = sl_preds[i]
        if side == 1:
            if y_tp[i] <= tp:
                wins += 1
                balance += tp - (tp * (FEE_BPS + SLIPPAGE_BPS))
            else:
                balance -= sl + (sl * (FEE_BPS + SLIPPAGE_BPS))
        else:
            if y_tp[i] <= tp:
                wins += 1
                balance += tp - (tp * (FEE_BPS + SLIPPAGE_BPS))
            else:
                balance -= sl + (sl * (FEE_BPS + SLIPPAGE_BPS))
    win_rate = wins / trades if trades > 0 else 0
    return balance, win_rate

# -----------------
# Train function
# -----------------
def train_coin(coin):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{coin}.csv"))
    df = df.sort_values("time").reset_index(drop=True)

    n_total = len(df)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val = int(n_total * VAL_SPLIT)
    n_test = n_total - n_train - n_val

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train+n_val]
    df_test = df.iloc[n_train+n_val:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train.drop(columns=["y_class","y_tp","y_sl","time"]))
    X_val = scaler.transform(df_val.drop(columns=["y_class","y_tp","y_sl","time"]))
    X_test = scaler.transform(df_test.drop(columns=["y_class","y_tp","y_sl","time"]))

    y_train_class = df_train["y_class"].values
    y_train_tp = df_train["y_tp"].values
    y_train_sl = df_train["y_sl"].values
    y_val_class = df_val["y_class"].values
    y_val_tp = df_val["y_tp"].values
    y_val_sl = df_val["y_sl"].values
    y_test_class = df_test["y_class"].values
    y_test_tp = df_test["y_tp"].values
    y_test_sl = df_test["y_sl"].values

    train_ds = SequenceDataset(X_train, y_train_class, y_train_tp, y_train_sl)
    val_ds = SequenceDataset(X_val, y_val_class, y_val_tp, y_val_sl)
    test_ds = SequenceDataset(X_test, y_test_class, y_test_tp, y_test_sl)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = TransformerModel(X_train.shape[1])
    criterion_class = nn.CrossEntropyLoss()
    criterion_tp = nn.MSELoss()
    criterion_sl = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_profit = -np.inf
    best_threshold = 0.5
    patience_counter = 0

    for epoch in range(100):
        model.train()
        for xb, yb_class, yb_tp, yb_sl in train_dl:
            optimizer.zero_grad()
            out_class, out_tp, out_sl = model(xb.unsqueeze(1))
            loss = criterion_class(out_class, yb_class) + criterion_tp(out_tp.squeeze(), yb_tp) + criterion_sl(out_sl.squeeze(), yb_sl)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_probs, all_tp_preds, all_sl_preds = [], [], [], []
        with torch.no_grad():
            for xb, yb_class, yb_tp, yb_sl in val_dl:
                out_class, out_tp, out_sl = model(xb.unsqueeze(1))
                probs = torch.softmax(out_class, dim=1).numpy()
                preds = np.argmax(probs, axis=1)
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_tp_preds.extend(out_tp.squeeze().numpy())
                all_sl_preds.extend(out_sl.squeeze().numpy())

        best_epoch_profit = -np.inf
        best_epoch_threshold = 0.5
        for t in THRESHOLDS:
            profit, _ = simulate_profits(all_preds, all_probs, all_tp_preds, all_sl_preds, y_val_class, y_val_tp, y_val_sl, t)
            if profit > best_epoch_profit:
                best_epoch_profit = profit
                best_epoch_threshold = t

        if best_epoch_profit > best_val_profit:
            best_val_profit = best_epoch_profit
            best_threshold = best_epoch_threshold
            patience_counter = 0
            os.makedirs(os.path.join(MODEL_SAVE_DIR, coin), exist_ok=True)
            torch.save({"model_state": model.state_dict(), "scaler": scaler, "threshold": best_threshold},
                       os.path.join(MODEL_SAVE_DIR, coin, "model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break

    # Test evaluation
    checkpoint = torch.load(os.path.join(MODEL_SAVE_DIR, coin, "model.pth"))
    model.load_state_dict(checkpoint["model_state"])
    scaler = checkpoint["scaler"]
    threshold = checkpoint["threshold"]

    all_preds, all_probs, all_tp_preds, all_sl_preds = [], [], [], []
    with torch.no_grad():
        for xb, yb_class, yb_tp, yb_sl in test_dl:
            out_class, out_tp, out_sl = model(xb.unsqueeze(1))
            probs = torch.softmax(out_class, dim=1).numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_tp_preds.extend(out_tp.squeeze().numpy())
            all_sl_preds.extend(out_sl.squeeze().numpy())

    profit, win_rate = simulate_profits(all_preds, all_probs, all_tp_preds, all_sl_preds, y_test_class, y_test_tp, y_test_sl, threshold)
    print(f"{coin} TEST Profit: {profit:.2f}, Win rate: {win_rate:.2%}, Threshold: {threshold}")

if __name__ == "__main__":
    for coin_file in os.listdir(DATA_DIR):
        if coin_file.endswith(".csv"):
            coin = coin_file.replace(".csv", "")
            train_coin(coin)
