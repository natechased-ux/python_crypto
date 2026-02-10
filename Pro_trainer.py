import ccxt
import pandas as pd
import numpy as np
import ta
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

# ==== Config ====
SYMBOL = 'BTC/USD'
TIMEFRAME = '5m'
SEQ_LEN = 30                  # shorter seq length
HORIZON = 12                  # 12 x 5m = 1 hour ahead
TOP_MOVE_PERCENT = 0.50       # keep top 50% largest moves
TOTAL_CANDLES = 50000         # ~172 days of 5m data
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
LAMBDA_REG = 1.0
MODEL_SAVE_PATH = SYMBOL.replace("/", "_") + "_dual_model_filtered.pth"

# ==== Fetch Historical Data ====
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

# ==== Feature Engineering ====
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

# ==== Auto Threshold Labeling ====
def add_targets_auto_threshold(df, horizon, top_move_percent):
    df = df.copy()
    df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
    df = df.dropna(subset=['future_return'])

    # Show move distribution before filtering
    print("\nFuture Return Distribution (abs values):")
    print(df['future_return'].abs().describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    abs_moves = df['future_return'].abs()
    threshold = abs_moves.quantile(1 - top_move_percent)
    print(f"\nAuto-selected MOVE_THRESHOLD = {threshold:.5f} ({top_move_percent*100:.0f}% largest moves kept)")

    mask_up = df['future_return'] >= threshold
    mask_down = df['future_return'] <= -threshold
    df.loc[mask_up, 'direction'] = 1
    df.loc[mask_down, 'direction'] = 0

    df = df.dropna(subset=['direction'])
    df['direction'] = df['direction'].astype(int)
    return df, threshold

# ==== Dataset ====
class CryptoDataset(Dataset):
    def __init__(self, df, seq_len, feature_cols):
        self.X = []
        self.y_class = df['direction'].values
        self.y_reg = df['future_return'].values
        for i in range(len(df) - seq_len):
            self.X.append(df[feature_cols].iloc[i:i+seq_len].values)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y_class = torch.tensor(self.y_class[seq_len:], dtype=torch.long)
        self.y_reg = torch.tensor(self.y_reg[seq_len:], dtype=torch.float32)

    def __len__(self): return len(self.y_class)
    def __getitem__(self, idx): return self.X[idx], self.y_class[idx], self.y_reg[idx]

# ==== Dual-Head Transformer ====
class TransformerModelDual(nn.Module):
    def __init__(self, seq_len=60, feature_size=5):
        super().__init__()
        self.embedding = nn.Linear(feature_size, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc_class = nn.Linear(64*seq_len, 2)
        self.fc_reg = nn.Linear(64*seq_len, 1)
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.flatten(1)
        class_logits = self.fc_class(x)
        reg_output = self.fc_reg(x)
        return class_logits, reg_output

# ==== Main ====
if __name__ == "__main__":
    print(f"Fetching data for {SYMBOL} ({TIMEFRAME})...")
    df = fetch_historical(SYMBOL, TIMEFRAME, limit=300, total_candles=TOTAL_CANDLES)
    df = add_features(df)
    df, threshold = add_targets_auto_threshold(df, HORIZON, TOP_MOVE_PERCENT)

    # Show class distribution
    up_count = (df['direction'] == 1).sum()
    down_count = (df['direction'] == 0).sum()
    print(f"Class distribution after filtering: Up={up_count}, Down={down_count}")

    exclude_cols = ['timestamp', 'future_return', 'direction', 'daily_high', 'daily_low']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    dataset = CryptoDataset(df, SEQ_LEN, feature_cols)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty after filtering. Try lowering TOP_MOVE_PERCENT or fetching more data.")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Class weights
    weight_up = len(df) / (2 * up_count)
    weight_down = len(df) / (2 * down_count)
    class_weights = torch.tensor([weight_down, weight_up], dtype=torch.float32)

    model = TransformerModelDual(seq_len=SEQ_LEN, feature_size=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    criterion_reg = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = total_class_loss = total_reg_loss = correct = total = 0
        for X_batch, y_class, y_reg in loader:
            optimizer.zero_grad()
            out_class, out_reg = model(X_batch)
            loss_class = criterion_class(out_class, y_class)
            loss_reg = criterion_reg(out_reg.squeeze(), y_reg)
            loss = loss_class + LAMBDA_REG * loss_reg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_class_loss += loss_class.item()
            total_reg_loss += loss_reg.item()
            preds = torch.argmax(out_class, dim=1)
            correct += (preds == y_class).sum().item()
            total += y_class.size(0)

        print(f"Epoch {epoch+1}/{EPOCHS}, "
              f"Total Loss: {total_loss/len(loader):.4f}, "
              f"Class Loss: {total_class_loss/len(loader):.4f}, "
              f"Reg Loss: {total_reg_loss/len(loader):.6f}, "
              f"Accuracy: {correct/total*100:.2f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
