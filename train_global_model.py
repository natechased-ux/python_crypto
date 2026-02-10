import pandas as pd, numpy as np, torch, os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from dataset_window_stream import StreamingWindowDataset
from transformer_global_model import TransformerWithCoin

# === Config
DATA_PATH = "datasets_macro_training5/global_dataset_fixed.csv"
SEQ_LEN = 48
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("checkpoints", exist_ok=True)

# === Load and prepare dataframe
df = pd.read_csv(DATA_PATH)
df = df.rename(columns={'label': 'y_class', 'tp_pct': 'y_tp', 'sl_pct': 'y_sl'})
df = df.dropna().reset_index(drop=True)
df['coin_id'] = df['coin_id'].astype('category')
df['coin_id_code'] = df['coin_id'].cat.codes

non_features = ['time', 'y_class', 'y_tp', 'y_sl', 'coin_id', 'coin_id_code']
feature_cols = [c for c in df.columns if c not in non_features]

scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
coin_count = df['coin_id_code'].nunique()

# === Streaming dataset
dataset = StreamingWindowDataset(df, feature_cols, SEQ_LEN)
n_total = len(dataset)
n_test = int(n_total * 0.15)
n_val = int(n_total * 0.15)
n_train = n_total - n_val - n_test
ds_train, ds_val, ds_test = random_split(dataset, [n_train, n_val, n_test])

train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

# === Model
model = TransformerWithCoin(input_dim=len(feature_cols), coin_count=coin_count).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_cls = nn.CrossEntropyLoss()
loss_reg = nn.MSELoss()

# === Simulate profits
@torch.no_grad()
@torch.no_grad()
def simulate_profits(model, test_loader, df_all, lookahead=8, cost=0.002):
    model.eval()
    profits, wins, count = [], [], 0

    base_dataset = test_loader.dataset.dataset  # unwrap Subset
    subset_indices = test_loader.dataset.indices

    for xb, cb, yb_cls, yb_tp, yb_sl in test_loader:
        xb, cb = xb.to(DEVICE), cb.to(DEVICE)
        out_cls, out_tp, out_sl = model(xb, cb)
        preds_cls = out_cls.softmax(-1).argmax(-1).cpu().numpy()
        preds_tp = out_tp.squeeze().cpu().numpy()
        preds_sl = out_sl.squeeze().cpu().numpy()

        for i in range(len(preds_cls)):
            label = int(preds_cls[i])
            if label == 0:
                continue

            # Map back into StreamingWindowDataset
            real_idx = subset_indices[count + i]
            index = base_dataset.valid_indices[real_idx]

            entry = df_all.iloc[index]["close"]
            future_df = df_all.iloc[index+1 : index+1+lookahead]

            if len(future_df) < lookahead:
                continue

            tp_price = entry * (1 + preds_tp[i]) if label == 1 else entry * (1 - preds_tp[i])
            sl_price = entry * (1 - preds_sl[i]) if label == 1 else entry * (1 + preds_sl[i])

            hit_tp, hit_sl = False, False
            for _, row in future_df.iterrows():
                if label == 1:
                    if row["high"] >= tp_price: hit_tp = True; break
                    if row["low"] <= sl_price: hit_sl = True; break
                else:
                    if row["low"] <= tp_price: hit_tp = True; break
                    if row["high"] >= sl_price: hit_sl = True; break

            if hit_tp:
                ret = preds_tp[i]
            elif hit_sl:
                ret = -preds_sl[i]
            else:
                future_close = df_all.iloc[index + lookahead]["close"]
                ret = ((future_close - entry) / entry) if label == 1 else ((entry - future_close) / entry)

            profits.append(ret - cost)
            wins.append(ret > cost)

        count += len(preds_cls)

    if profits:
        win_rate = np.mean(wins)
        avg_return = np.mean(profits)
        print(f"📈 Simulated {len(profits)} trades")
        print(f"✅ Win Rate: {win_rate:.2%}")
        print(f"💰 Avg Return: {avg_return:.2%}")
        return win_rate, avg_return
    else:
        print("⚠️ No trades triggered during simulation")
        return 0.0, 0.0


# === Training loop with checkpointing
best_loss = float("inf")
best_profit = -float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, cb, yb_cls, yb_tp, yb_sl in train_loader:
        xb, cb = xb.to(DEVICE), cb.to(DEVICE)
        yb_cls, yb_tp, yb_sl = yb_cls.to(DEVICE), yb_tp.to(DEVICE), yb_sl.to(DEVICE)
        out_cls, out_tp, out_sl = model(xb, cb)
        loss = loss_cls(out_cls, yb_cls) + loss_reg(out_tp, yb_tp) + loss_reg(out_sl, yb_sl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    # Save checkpoint each epoch
    ckpt_path = f"checkpoints/model_epoch{epoch+1}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"💾 Saved checkpoint {ckpt_path}")

    # Save best by loss
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_model_loss.pth")
        print(f"🌟 New best model (by loss) saved at epoch {epoch+1}")

    # Evaluate profit every epoch
    print(f"🔍 Running profit simulation at epoch {epoch+1}...")
    win_rate, avg_return = simulate_profits(model, test_loader, df)

    # Save best by profit
    if avg_return > best_profit:
        best_profit = avg_return
        torch.save(model.state_dict(), "best_model_profit.pth")
        print(f"💰 New best model (by profit) saved at epoch {epoch+1}")

print("✅ Training finished")
