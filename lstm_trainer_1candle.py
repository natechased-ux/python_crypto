import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# === CONFIG ===
SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
MODEL_PATH = "lstm_crypto_model_1candle.pt"

# === LOAD DATA ===
df = pd.read_csv("lstm_training_dataset_1candle.csv")

FEATURES = [
    'rsi', 'adx', 'plus_di', 'minus_di', 'macd_diff',
    'ema10', 'ema20', 'ema50', 'ema200',
    'ema10_slope', 'ema20_slope', 'ema50_slope', 'ema200_slope',
    'atr', 'vol_change', 'body_wick_ratio', 'above_ema200'
]

label_map = {-1: 0, 0: 1, 1: 2}  # Short, No Trade, Long
df.dropna(inplace=True)
X = df[FEATURES].values
y = df["label"].map(label_map).values

# === SCALE FEATURES ===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === CREATE SEQUENCES ===
def create_sequences(X, y, seq_len=SEQ_LEN):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y, SEQ_LEN)

# === TRAIN-TEST SPLIT ===
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# === TORCH DATASET ===
class CryptoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CryptoDataset(X_train, y_train)
val_dataset = CryptoDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === LSTM MODEL ===
class LSTMTrader(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h[:, -1, :])

model = LSTMTrader(input_size=len(FEATURES))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === TRAIN LOOP ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    print(f"Epoch {epoch+1}/{EPOCHS} - "
          f"Train Loss: {total_loss/len(train_loader):.4f} - "
          f"Val Loss: {val_loss/len(val_loader):.4f} - "
          f"Val Acc: {correct/total:.4f}")

# === SAVE MODEL ===
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler
}, MODEL_PATH)
print(f"✅ Entry Model saved to {MODEL_PATH}")
