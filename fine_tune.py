import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing

# Allowlist StandardScaler so PyTorch can load your model
torch.serialization.add_safe_globals([sklearn.preprocessing._data.StandardScaler])

# ====================
# Dataset Class
# ====================
class CryptoDataset(Dataset):
    def __init__(self, df, features, seq_len):
        self.features = features
        self.seq_len = seq_len
        self.scaler = StandardScaler()

        # Scale features
        X = self.scaler.fit_transform(df[self.features])

        # Map labels: -1 → 0, 0 → 1, 1 → 2
        y_raw = df["label"].values
        label_map = { -1: 0, 0: 1, 1: 2 }
        y = np.array([label_map[val] for val in y_raw])

        # Build sequences
        self.X = []
        self.y = []
        for i in range(len(df) - seq_len):
            self.X.append(X[i:i+seq_len])
            self.y.append(y[i+seq_len])

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ====================
# Model (same as your trained one)
# ====================
class LSTMEntryModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out

# ====================
# Training Function
# ====================
def fine_tune(base_model_path, dataset_path, output_path, seq_len=60, epochs=15, batch_size=64, lr=1e-4):
    # Load dataset
    df = pd.read_csv(dataset_path)
    features = [c for c in df.columns if c not in ["time", "label", "symbol"]]
    dataset = CryptoDataset(df, features, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load checkpoint (global model)
    checkpoint = torch.load(base_model_path, map_location="cpu", weights_only=False)

    # Create model with matching input size
    input_dim = len(features)
    model = LSTMEntryModel(input_dim=input_dim)

    # Load just the model weights from checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Fine-tuning
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")

    # Save fine-tuned model
    torch.save(model.state_dict(), output_path)
    print(f"✅ Fine-tuned model saved → {output_path}")

# ====================
# CLI Entry Point
# ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Path to global model .pt file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to coin-specific CSV dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save fine-tuned model")
    parser.add_argument("--seq_len", type=int, default=60, help="Sequence length for LSTM")
    parser.add_argument("--epochs", type=int, default=15, help="Fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    fine_tune(
        args.base_model,
        args.dataset,
        args.output,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
