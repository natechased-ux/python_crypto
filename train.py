import ccxt
import pandas as pd
import numpy as np
import ta
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
from sklearn.utils import resample

# ==== CONFIG ====
SYMBOL = 'BTC/USD'
TIMEFRAME = '5m'
SEQ_LEN = 30
HORIZON = 12                     # 1 hour ahead
TOTAL_CANDLES = 50000
UP_THRESHOLD = 0.0015             # +0.15% for strong up
DOWN_THRESHOLD = -0.0015          # -0.15% for strong down
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
MODEL_SAVE_PATH = SYMBOL.replace("/", "_") + "_strong_move_classifier.pth"

# ==== FETCH HISTORICAL ====
def fetch_historical(symbol, timeframe, limit, total_candles=20000):
    ex = ccxt.coinbase()
    all_data = []
    since = None
    while len(all_data) < total_candles:
        limit = min(300, total_candles - len(all_data))
        chunk = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
        if not chunk:
            break
        all_data = chunk + all_data  # prepend oldest first
        since = chunk[0][0] - ex.parse_timeframe(timeframe) * 1000
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.drop_duplicates(subset='timestamp').sort_values('timestamp')

# ==== FEATURES ====
def add_features(df):
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['rolling_vol'] = df['return'].rolling(10).std()
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr.average_true_range()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_b'] = bb.bollinger_pband()
    df['bb_width'] = bb.bollinger_wband()
    df['vol_change'] = df['volume'].pct_change()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['oc_range'] = abs(df['close'] - df['open']) / df['close']
    df['wick_ratio'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['daily_high'] = df['high'].rolling(int(24*60 / 5)).max()
    df['daily_low'] = df['low'].rolling(int(24*60 / 5)).min()
    df['daily_pos'] = (df['close'] - df['daily_low']) / (df['daily_high'] - df['daily_low'])
    df = df.dropna()
    return df

# ==== TARGET ====
def add_target(df, horizon, up_th, down_th):
    df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
    df = df.dropna(subset=['future_return'])

    # Label: 0 = No Trade, 1 = Long, 2 = Short
    df['label'] = np.where(df['future_return'] >= up_th, 1,
                    np.where(df['future_return'] <= down_th, 2, 0))
    return df

# ==== BALANCE CLASSES ====
def balance_classes(df):
    classes = df['label'].unique()
    min_size = min(df['label'].value_counts())
    balanced_df = pd.concat([
        resample(df[df['label'] == cls], replace=True, n_samples=min_size, random_state=42)
        for cls in classes
    ])
    return balanced_df.sort_index()

# ==== DATASET ====
class CryptoDataset(Dataset):
    def __init__(self, df, seq_len, feature_cols):
        X_list = []
        y_list = []
        for i in range(len(df) - seq_len):
            X_list.append(df[feature_cols].iloc[i:i+seq_len].values)
            y_list.append(df['label'].iloc[i+seq_len])
        self.X = torch.tensor(np.array(X_list), dtype=torch.float32)
        self.y = torch.tensor(np.array(y_list), dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==== MODEL ====
class TransformerModelClass(nn.Module):
    def __init__(self, seq_len=60, feature_size=5, num_classes=3):
        super().__init__()
        self.embedding = nn.Linear(feature_size, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc_class = nn.Linear(64*seq_len, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.flatten(1)
        return self.fc_class(x)

# ==== MAIN ====
if __name__ == "__main__":
    print(f"Fetching {TOTAL_CANDLES} candles for {SYMBOL} ({TIMEFRAME})...")
    df = fetch_historical(SYMBOL, TIMEFRAME, limit=300, total_candles=TOTAL_CANDLES)
    df = add_features(df)
    df = add_target(df, HORIZON, UP_THRESHOLD, DOWN_THRESHOLD)
    df = balance_classes(df)

    print("Class counts after balancing:", df['label'].value_counts().to_dict())

    exclude_cols = ['timestamp', 'future_return', 'label', 'daily_high', 'daily_low']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    dataset = CryptoDataset(df, SEQ_LEN, feature_cols)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TransformerModelClass(seq_len=SEQ_LEN, feature_size=len(feature_cols), num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            out_class = model(X_batch)
            loss = criterion(out_class, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(out_class, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}, Acc: {correct/total*100:.2f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
