import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# === TP/SL Model Class ===
class LSTMTpSlModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)  # [TP_pct, SL_pct]

# === Fine-tune Function ===
def fine_tune(base_model_path, dataset_path, output_path,
              seq_len=60, epochs=15, batch_size=64, lr=1e-4):

    # === 1. Load base model checkpoint ===
    checkpoint = torch.load(base_model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # === 2. Detect input dimension from saved weights ===
    for k, v in state_dict.items():
        if "weight_ih_l0" in k:
            input_dim_from_model = v.shape[1]  # e.g., 17
            break
    else:
        raise ValueError("Could not detect input dimension from TP/SL model.")

    print(f"📏 Global TP/SL model was trained with {input_dim_from_model} features.")

    # === 3. Load dataset ===
    df = pd.read_csv(dataset_path)
    df.dropna(inplace=True)

    # Keep only numeric columns, drop non-features
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    exclude_cols = ["label", "mfe_pct", "mae_pct"]
    features_all = [c for c in numeric_cols if c not in exclude_cols]

    # === 4. Match feature count ===
    if len(features_all) > input_dim_from_model:
        print(f"⚠ Dataset has {len(features_all)} features, model expects {input_dim_from_model}. Trimming extra columns.")
        FEATURES = features_all[:input_dim_from_model]
    elif len(features_all) < input_dim_from_model:
        raise ValueError(f"Dataset has {len(features_all)} features but model expects {input_dim_from_model}. Missing features.")
    else:
        FEATURES = features_all

    print(f"✅ Using features: {FEATURES}")

    # === 5. Prepare data ===
    TARGETS = ["mfe_pct", "mae_pct"]
    X = df[FEATURES].values
    y = df[TARGETS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    dataset = TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # === 6. Load model ===
    model = LSTMTpSlModel(input_dim=input_dim_from_model)
    model.load_state_dict(state_dict)

    # === 7. Optimizer & loss ===
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # === 8. Fine-tune loop ===
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")

    # === 9. Save fine-tuned model ===
    torch.save(model.state_dict(), output_path)
    print(f"✅ Fine-tuned TP/SL model saved → {output_path}")

# === Main CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Path to global TP/SL model (.pt)")
    parser.add_argument("--dataset", required=True, help="Path to coin dataset CSV")
    parser.add_argument("--output", required=True, help="Path to save fine-tuned model")
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
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
