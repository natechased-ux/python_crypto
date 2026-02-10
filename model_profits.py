import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split
from transformer_global_model import TransformerWithCoin
from dataset_window_stream import StreamingWindowDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Reload dataset ===
DATA_PATH = "datasets_macro_training5/global_dataset_fixed.csv"
SEQ_LEN = 48
BATCH_SIZE = 256

df = pd.read_csv(DATA_PATH)
df = df.rename(columns={'label': 'y_class','tp_pct':'y_tp','sl_pct':'y_sl'})
df['coin_id'] = df['coin_id'].astype('category')
df['coin_id_code'] = df['coin_id'].cat.codes
non_features = ['time','y_class','y_tp','y_sl','coin_id','coin_id_code']
feature_cols = [c for c in df.columns if c not in non_features]

# === Dataset and test split ===
dataset = StreamingWindowDataset(df, feature_cols, SEQ_LEN)
n_total = len(dataset)
n_test = int(n_total * 0.15)
n_val = int(n_total * 0.15)
n_train = n_total - n_val - n_test
_, _, ds_test = random_split(dataset, [n_train, n_val, n_test])

test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# === Define simulate_profits with trade logging ===
@torch.no_grad()
def simulate_profits(model, test_loader, df_all, lookahead=8, cost=0.002, log_csv=None):
    model.eval()
    profits, wins, count = [], [], 0
    trade_log = []

    base_dataset = test_loader.dataset.dataset  # unwrap Subset
    subset_indices = test_loader.dataset.indices

    for xb, cb, _, _, _ in test_loader:
        xb, cb = xb.to(DEVICE), cb.to(DEVICE)
        out_cls, out_tp, out_sl = model(xb, cb)
        preds_cls = out_cls.softmax(-1).argmax(-1).cpu().numpy()
        preds_tp = out_tp.squeeze().cpu().numpy()
        preds_sl = out_sl.squeeze().cpu().numpy()

        for i in range(len(preds_cls)):
            label = int(preds_cls[i])
            if label == 0:
                continue

            real_idx = subset_indices[count + i]
            index = base_dataset.valid_indices[real_idx]

            entry = df_all.iloc[index]["close"]
            coin_id = df_all.iloc[index]["coin_id"]
            future_df = df_all.iloc[index+1 : index+1+lookahead]

            if len(future_df) < lookahead:
                continue

            tp_price = entry * (1 + preds_tp[i]) if label == 1 else entry * (1 - preds_tp[i])
            sl_price = entry * (1 - preds_sl[i]) if label == 1 else entry * (1 + preds_sl[i])

            outcome, ret = "expiry", 0
            for _, row in future_df.iterrows():
                if label == 1:  # long
                    if row["high"] >= tp_price:
                        ret, outcome = preds_tp[i], "tp"; break
                    if row["low"] <= sl_price:
                        ret, outcome = -preds_sl[i], "sl"; break
                else:  # short
                    if row["low"] <= tp_price:
                        ret, outcome = preds_tp[i], "tp"; break
                    if row["high"] >= sl_price:
                        ret, outcome = -preds_sl[i], "sl"; break
            else:
                future_close = df_all.iloc[index + lookahead]["close"]
                ret = (future_close - entry) / entry if label == 1 else (entry - future_close) / entry

            net_ret = ret - cost
            profits.append(net_ret)
            wins.append(net_ret > 0)

            trade_log.append({
                "index": index,
                "coin_id": coin_id,
                "direction": "long" if label == 1 else "short",
                "entry": entry,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "outcome": outcome,
                "raw_return": ret,
                "net_return": net_ret
            })

        count += len(preds_cls)

    if profits:
        win_rate, avg_return = float(np.mean(wins)), float(np.mean(profits))
        print(f"📈 Simulated {len(profits)} trades")
        print(f"✅ Win Rate: {win_rate:.2%}")
        print(f"💰 Avg Return: {avg_return:.2%}")

        if log_csv:
            pd.DataFrame(trade_log).to_csv(log_csv, index=False)
            print(f"📝 Trade log saved to {log_csv}")

        return win_rate, avg_return
    else:
        print("⚠️ No trades triggered")
        return 0.0, 0.0

# === Load model and evaluate ===
coin_count = df['coin_id_code'].nunique()
model = TransformerWithCoin(input_dim=len(feature_cols), coin_count=coin_count).to(DEVICE)

# load either best-by-profit or best-by-loss
model.load_state_dict(torch.load("best_model_profit.pth", map_location=DEVICE))
model.eval()

# run simulation and save CSV of trades
simulate_profits(model, test_loader, df, log_csv="trade_log_best_profit.csv")
