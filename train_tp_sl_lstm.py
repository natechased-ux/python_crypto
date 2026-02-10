def train_tp_sl_model(
    dataset_path, output_path,
    seq_len=60, epochs=50, batch_size=64, lr=0.001,
    patience=5
):
    import pandas as pd
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    # Load dataset
    df = pd.read_csv(dataset_path)
    features = [c for c in df.columns if c not in ["time", "tp_pct", "sl_pct"]]
    X = df[features].values
    y = df[["tp_pct", "sl_pct"]].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sequence split
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    # Train/val split
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

    # Torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    # Model
    class LSTMTpSlModel(nn.Module):
        def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)

    model = LSTMTpSlModel(input_dim=len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Early stopping vars
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                preds = model(Xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

        # Check early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "scaler": scaler}, output_path)
            print(f"✅ New best model saved (Val Loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("⏹ Early stopping triggered.")
                break
