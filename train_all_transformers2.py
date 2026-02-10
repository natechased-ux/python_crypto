
import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view

# ==================
# Config
# ==================
DATA_DIR = "C:/Users/natec/datasets_macro_training5"
OUT_DIR = "models_transformer_multiscale_safe"

SEQ_LEN_SHORT = 48
SEQ_LEN_LONG  = 192
DOWNSAMPLE    = 4        # 192/4 = 48

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-3

TRAIN_FRAC = 0.75
VAL_FRAC   = 0.15   # remainder is test
EARLY_STOP_PATIENCE = 6
THRESH_GRID = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]
FEE_BPS = 5       # per side
SLIP_BPS = 5      # per side
FEATURE_SHIFT_1BAR = False  # if True, use features from prior bar

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def bps_to_frac(bps): return bps/10000.0

# ==================
# Column helpers
# ==================
def pick_column(df, candidates, required=True, name=""):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column {name}; tried: {candidates}; have: {list(df.columns)}")
    return None

# ==================
# Multi-scale windowing
# ==================
def make_windows_multiscale(df_scaled, feature_cols, seq_s, seq_l, ds):
    """Return X (N, seq_s, F_short+F_long), y_cls, y_tp, y_sl.
       - short window length = seq_s
       - long window length  = seq_l, downsampled by ds to seq_s steps
    """
    arr = df_scaled[feature_cols].to_numpy()
    n, f = arr.shape
    if n < max(seq_s, seq_l):
        raise ValueError(f"Not enough rows for windows: have {n}, need {max(seq_s, seq_l)}")

    # Short windows: (N_short, seq_s, F)
    Ws = sliding_window_view(arr, (seq_s, f))  # shape: (n-seq_s+1, 1, 1, seq_s, f)
    Ws = Ws.reshape(-1, seq_s, f)

    # Long windows: (N_long, seq_l, F) -> downsample along time -> (N_long, seq_s, F)
    Wl = sliding_window_view(arr, (seq_l, f)).reshape(-1, seq_l, f)
    # Downsample evenly from the END so it aligns with the same "current" bar as short
    Wl = Wl[:, ::ds, :]
    if Wl.shape[1] != seq_s:
        # Adjust by slicing last seq_s steps to ensure alignment
        Wl = Wl[:, -seq_s:, :]

    # Align counts: short windows start at index 0..n-seq_s, long at 0..n-seq_l
    # Use the overlapping range where both exist (start at idx = seq_l - seq_s)
    start = seq_l - seq_s
    end = min(Ws.shape[0], Wl.shape[0] - start)
    if end <= 0:
        raise ValueError("Window alignment failed; check seq lengths and downsample.")
    Ws = Ws[start:start+end]
    Wl = Wl[:end]

    # Concat features per timestep: (N, seq_s, F_short+F_long)
    X = np.concatenate([Ws, Wl], axis=2)

    # Targets aligned to last timestep of the short window component
    y_idx = np.arange(start, start+end) + (seq_s - 1)
    y_cls = df_scaled['y_class'].iloc[y_idx].to_numpy()
    y_tp  = df_scaled['y_tp'].iloc[y_idx].to_numpy()
    y_sl  = df_scaled['y_sl'].iloc[y_idx].to_numpy()

    X = torch.tensor(X, dtype=torch.float32)
    y_cls = torch.tensor(y_cls, dtype=torch.long)
    y_tp  = torch.tensor(y_tp, dtype=torch.float32).unsqueeze(-1)
    y_sl  = torch.tensor(y_sl, dtype=torch.float32).unsqueeze(-1)
    return X, y_cls, y_tp, y_sl, y_idx

# ==================
# Profit simulation (with costs)
# ==================
def simulate_profits(labels, closes, highs, lows, tp_preds, sl_preds, lookahead=8):
    profits = []
    outcomes = []
    fee = bps_to_frac(FEE_BPS)
    slip = bps_to_frac(SLIP_BPS)
    cost = 2*(fee+slip)

    n = len(labels)
    max_i = n - lookahead
    for i in range(max_i):
        lab = int(labels[i])
        if lab == 0:  # HOLD
            continue
        entry = closes[i]
        tp_price = entry * (1 + tp_preds[i]) if lab == 1 else entry * (1 - tp_preds[i])
        sl_price = entry * (1 - sl_preds[i]) if lab == 1 else entry * (1 + sl_preds[i])

        hit = None
        for j in range(1, lookahead+1):
            hi, lo = highs[i+j], lows[i+j]
            if lab == 1:
                if hi >= tp_price:
                    hit = ("tp", tp_preds[i]); break
                if lo <= sl_price:
                    hit = ("sl", sl_preds[i]); break
            else: # lab == 2 short
                if lo <= tp_price:
                    hit = ("tp", tp_preds[i]); break
                if hi >= sl_price:
                    hit = ("sl", sl_preds[i]); break
        if hit is None:
            if lab == 1:
                ret = (closes[i+lookahead] - entry)/entry
            else:
                ret = (entry - closes[i+lookahead])/entry
        else:
            ret = hit[1] if hit[0] == "tp" else -hit[1]

        ret -= cost
        profits.append(ret)
        outcomes.append(ret > 0.0)

    avg_ret = float(np.mean(profits)) if profits else 0.0
    win_rate = float(np.mean(outcomes)) if outcomes else 0.0
    return avg_ret, win_rate, len(profits)

# ==================
# Model
# ==================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerEntryModel(nn.Module):
    def __init__(self, input_dim, d_model=D_MODEL, nhead=N_HEADS, num_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.posenc = PositionalEncoding(d_model, max_len=2048)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 3))
        self.tp_head  = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.sl_head  = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.input_proj(x)
        h = self.posenc(h)
        h = self.encoder(h)
        h_last = h[:, -1, :]
        return self.cls_head(h_last), self.tp_head(h_last), self.sl_head(h_last)

# ==================
# Train single coin
# ==================
def train_coin(coin_file):
    coin = coin_file[:-4]
    path = os.path.join(DATA_DIR, coin_file)
    df = pd.read_csv(path)

    # Detect columns
    time_col = None
    for c in ["time","timestamp","date","datetime"]:
        if c in df.columns: time_col=c; break
    y_class_col = "y_class" if "y_class" in df.columns else ("label" if "label" in df.columns else None)
    if y_class_col is None: raise KeyError(f"{coin}: missing y_class/label")
    y_tp_col = "y_tp" if "y_tp" in df.columns else ("tp_pct" if "tp_pct" in df.columns else None)
    y_sl_col = "y_sl" if "y_sl" in df.columns else ("sl_pct" if "sl_pct" in df.columns else None)
    if y_tp_col is None or y_sl_col is None: raise KeyError(f"{coin}: missing y_tp/y_sl or tp_pct/sl_pct")

    # Sort by time if present
    if time_col: df = df.sort_values(time_col).reset_index(drop=True)

    # Numeric only and rename labels to standard
    df = df.select_dtypes(include=[np.number]).copy()
    df.rename(columns={y_class_col:"y_class", y_tp_col:"y_tp", y_sl_col:"y_sl"}, inplace=True)

    # Optional feature shift
    feature_cols = [c for c in df.columns if c not in ["y_class","y_tp","y_sl"]]
    if FEATURE_SHIFT_1BAR:
        df[feature_cols] = df[feature_cols].shift(1)

    # Splits
    N = len(df)
    n_train = int(N*TRAIN_FRAC)
    n_val = int(N*VAL_FRAC)
    n_test = N - n_train - n_val
    if min(n_train, n_val, n_test) <= SEQ_LEN_LONG + 16:
        raise ValueError(f"{coin}: not enough rows for multiscale windows (N={N})")

    df_train = df.iloc[:n_train].copy()
    df_val   = df.iloc[n_train:n_train+n_val].copy()
    df_test  = df.iloc[n_train+n_val:].copy()

    # Scale (fit on train only)
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])
    df_train[feature_cols] = scaler.transform(df_train[feature_cols])
    df_val[feature_cols]   = scaler.transform(df_val[feature_cols])
    df_test[feature_cols]  = scaler.transform(df_test[feature_cols])

    # Windows
    Xtr,ytr_cls,ytr_tp,ytr_sl, idx_tr = make_windows_multiscale(df_train, feature_cols, SEQ_LEN_SHORT, SEQ_LEN_LONG, DOWNSAMPLE)
    Xva,yva_cls,yva_tp,yva_sl, idx_va = make_windows_multiscale(df_val,   feature_cols, SEQ_LEN_SHORT, SEQ_LEN_LONG, DOWNSAMPLE)
    Xte,yte_cls,yte_tp,yte_sl, idx_te = make_windows_multiscale(df_test,  feature_cols, SEQ_LEN_SHORT, SEQ_LEN_LONG, DOWNSAMPLE)

    train_loader = DataLoader(TensorDataset(Xtr,ytr_cls,ytr_tp,ytr_sl), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xva,yva_cls,yva_tp,yva_sl), batch_size=BATCH_SIZE, shuffle=False)

    # Market arrays aligned to y indices (already aligned to last timestep of short window)
    closes_val = df_val["close"].values[idx_va]
    highs_val  = df_val["high"].values[idx_va]
    lows_val   = df_val["low"].values[idx_va]
    closes_t   = df_test["close"].values[idx_te]
    highs_t    = df_test["high"].values[idx_te]
    lows_t     = df_test["low"].values[idx_te]

    # Model
    input_dim = Xtr.shape[2]  # short+long features combined
    model = TransformerEntryModel(input_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_cls = nn.CrossEntropyLoss()
    loss_reg = nn.MSELoss()

    best_val_profit = -1e9
    best_thr = 0.5
    patience = EARLY_STOP_PATIENCE

    for epoch in range(EPOCHS):
        # Train
        model.train()
        for xb, yb_cls, yb_tp, yb_sl in train_loader:
            xb, yb_cls, yb_tp, yb_sl = xb.to(DEVICE), yb_cls.to(DEVICE), yb_tp.to(DEVICE), yb_sl.to(DEVICE)
            opt.zero_grad()
            out_cls, out_tp, out_sl = model(xb)
            l = loss_cls(out_cls, yb_cls) + loss_reg(out_tp, yb_tp) + loss_reg(out_sl, yb_sl)
            l.backward()
            opt.step()

        # Validate with threshold search
        model.eval()
        probs, tp_preds, sl_preds = [], [], []
        with torch.no_grad():
            for xb, _, _, _ in val_loader:
                xb = xb.to(DEVICE)
                lc, tp, sl = model(xb)
                probs.append(torch.softmax(lc, dim=1).cpu())
                tp_preds.append(tp.squeeze().cpu())
                sl_preds.append(sl.squeeze().cpu())
        probs = torch.cat(probs, dim=0)
        tp_preds = torch.cat(tp_preds).numpy()
        sl_preds = torch.cat(sl_preds).numpy()

        # Grid search threshold: turn low-confidence to HOLD(0)
        maxp, argmax = probs.max(dim=1)
        best_epoch_profit, best_epoch_thr, best_epoch_wr, best_epoch_trades = -1e9, None, 0.0, 0
        for thr in THRESH_GRID:
            labels = argmax.clone()
            labels[maxp < thr] = 0
            p, wr, n = simulate_profits(labels.numpy(), closes_val, highs_val, lows_val, tp_preds, sl_preds, lookahead=8)
            if p > best_epoch_profit:
                best_epoch_profit, best_epoch_thr, best_epoch_wr, best_epoch_trades = p, float(thr), wr, n

        print(f"{coin} | Epoch {epoch+1}/{EPOCHS} | Val Profit: {best_epoch_profit*100:.4f}% | WR: {best_epoch_wr*100:.2f}% | Trades: {best_epoch_trades} | Thr: {best_epoch_thr:.2f}")

        if best_epoch_profit > best_val_profit + 1e-12:
            best_val_profit = best_epoch_profit
            best_thr = best_epoch_thr
            patience = EARLY_STOP_PATIENCE

            # Save safe checkpoint
            out_dir_coin = os.path.join(OUT_DIR, coin)
            os.makedirs(out_dir_coin, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(out_dir_coin, "model_state.pth"))
            meta = {
                "threshold": float(best_thr),
                "seq_len_short": int(SEQ_LEN_SHORT),
                "seq_len_long": int(SEQ_LEN_LONG),
                "downsample": int(DOWNSAMPLE),
                "d_model": int(D_MODEL),
                "n_layers": int(N_LAYERS),
                "n_heads": int(N_HEADS),
                "fees_bps": int(FEE_BPS),
                "slip_bps": int(SLIP_BPS),
                "feature_count": int(input_dim)
            }
            with open(os.path.join(out_dir_coin, "meta.json"), "w") as f:
                json.dump(meta, f)
            print(f"✅ Saved best checkpoint — thr={best_thr:.2f}")
        else:
            patience -= 1
            if patience <= 0:
                print(f"{coin} | Early stop at epoch {epoch+1}")
                break

    # Final TEST using best thr
    model.eval()
    Xte = Xte_cpu = None  # release reference
    with torch.no_grad():
        # Rebuild test loader directly (small overhead) to avoid retaining big tensors
        test_dataset = TensorDataset(Xte if 'Xte' in locals() else torch.tensor(np.zeros((0,))), yte_cls, yte_tp, yte_sl)  # placeholder
    # Directly compute outputs without DataLoader for test (we already have tensors)
    with torch.no_grad():
        lc, tp, sl = model(Xte.to(DEVICE))
        probs = torch.softmax(lc, dim=1).cpu()
        tp_preds = tp.squeeze().cpu().numpy()
        sl_preds = sl.squeeze().cpu().numpy()
    maxp, argmax = probs.max(dim=1)
    labels = argmax.clone()
    labels[maxp < best_thr] = 0
    test_profit, test_wr, test_trades = simulate_profits(labels.numpy(), closes_t, highs_t, lows_t, tp_preds, sl_preds, lookahead=8)
    print(f"{coin} TEST | Net Profit: {test_profit*100:.4f}% | WR: {test_wr*100:.2f}% | Trades: {test_trades} | Thr*: {best_thr:.2f}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"): continue
        print(f"-> Training {fname}")
        try:
            train_coin(fname)
        except Exception as e:
            print(f"❌ {fname}: {e}")

if __name__ == "__main__":
    main()
