import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import os

# =======================
# Model definition
# =======================
class LSTMTpSlModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(LSTMTpSlModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)  # output: tp_pct, sl_pct

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out


# =======================
# Training function
# =======================
def train_tp_sl(dataset_path, model_save_path, seq_len=60, epochs=50, batch_size=64, lr=0.001):
    # Load data
    df = pd.read_csv(dataset_path)

    # Drop NaN
    df.dropna(subset=["tp_pct", "sl_pct"], inplace=True)

    FEATURES = [c for c in df.columns if c not in ["time", "label", "tp_pct", "sl_pct"]]
    X = df[FEATURES].values
    y = df[["tp_pct", "sl_pct"]].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    # Torch datasets
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_ds = torch.utils.data.TensorDataset(X_val_t, y_val_t)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = LSTMTpSlModel(input_dim=len(FEATURES))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler": scaler,
                "features": FEATURES
            }, model_save_path)
            print(f"✅ Best model saved — Val Loss: {val_loss:.6f}")


# =======================
# CLI
# =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to coin dataset CSV")
    parser.add_argument("--output", required=True, help="Path to save model")
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    train_tp_sl(
        args.dataset,
        args.output,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
