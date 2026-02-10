import os, json, math, time, requests
import numpy as np, pandas as pd, torch
import torch.nn as nn
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view

# ========= Config =========
DATA_DIR   = r"C:/Users/natec/datasets_macro_training5"
MODELS_DIR = r"models_transformer_multiscale_safe_stable"   # parent folder with one subfolder per coin (e.g., AAVE_USD)
GRANULARITY = 3600         # 1h bars
POLL_SEC    = 60
TRAIN_FRAC  = 0.75
VAL_FRAC    = 0.15

# ========= TA helpers (subset of your dataset_builder) =========
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

BASE_URL = "https://api.exchange.coinbase.com"

def fetch_range(symbol, start_dt, end_dt, granularity=GRANULARITY):
    rows, start = [], start_dt
    while start < end_dt:
        chunk_end = min(start + timedelta(seconds=granularity*300), end_dt)
        r = requests.get(f"{BASE_URL}/products/{symbol}/candles",
                         params={"start": start.isoformat(), "end": chunk_end.isoformat(), "granularity": granularity},
                         timeout=10)
        try:
            data = r.json()
            if isinstance(data, list) and data:
                rows.append(pd.DataFrame(data, columns=["time","low","high","open","close","volume"]))
        except Exception:
            pass
        start = chunk_end
        time.sleep(0.25)
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.sort_values("time", inplace=True)
    df["hour"] = df["time"].dt.hour/23.0
    df["day_of_week"] = df["time"].dt.dayofweek/6.0
    df["day_of_month"] = df["time"].dt.day/31.0
    df.drop_duplicates("time", inplace=True)
    return df

def calc_features(df):
    df = df.copy()
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    adx = ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx(); df["plus_di"] = adx.adx_pos(); df["minus_di"] = adx.adx_neg()
    df["macd_diff"] = MACD(df["close"]).macd_diff()
    for p in [10,20,50,200]:
        ema = df["close"].ewm(span=p, adjust=False).mean()
        df[f"ema{p}"] = ema
        df[f"ema{p}_slope"] = ema.diff()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=10).average_true_range()
    df["vol_change"] = df["volume"].pct_change()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_wick_ratio"] = (df["close"] - df["open"]).abs()/rng
    df["body_wick_ratio"] = df["body_wick_ratio"].fillna(0.0)
    df["above_ema200"] = (df["close"] > df["ema200"]).astype(int)
    return df

def calc_fib(df, lookback):
    swing_high = df["high"].rolling(lookback, min_periods=1).max()
    swing_low  = df["low"].rolling(lookback, min_periods=1).min()
    diff = swing_high - swing_low
    out = pd.DataFrame(index=df.index)
    out[f"fib_{lookback}_0"]   = swing_low
    out[f"fib_{lookback}_236"] = swing_high - diff*0.236
    out[f"fib_{lookback}_382"] = swing_high - diff*0.382
    out[f"fib_{lookback}_5"]   = swing_high - diff*0.5
    out[f"fib_{lookback}_618"] = swing_high - diff*0.618
    out[f"fib_{lookback}_786"] = swing_high - diff*0.786
    out[f"fib_{lookback}_1"]   = swing_high
    return pd.concat([df, out], axis=1)

def merge_btc(df, granularity=GRANULARITY):
    end = df["time"].max()
    start = end - timedelta(days=60)
    btc = fetch_range("BTC-USD", start, end, granularity)
    btc = calc_features(btc)
    for L in [720,360,168]:
        btc = calc_fib(btc, L)
    btc = btc.rename(columns={c:f"btc_{c}" for c in btc.columns if c!="time"})
    return pd.merge_asof(df.sort_values("time"), btc.sort_values("time"), on="time", direction="backward")

# ========= Model =========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerEntryModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.posenc = PositionalEncoding(d_model, max_len=2048)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 3))
        self.tp_head  = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.sl_head  = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        h = self.input_proj(x); h = self.posenc(h); h = self.encoder(h); h_last = h[:, -1, :]
        return self.cls_head(h_last), self.tp_head(h_last), self.sl_head(h_last)

def make_windows_multiscale(df_scaled, feature_cols, seq_s, seq_l, ds):
    arr = df_scaled[feature_cols].to_numpy()
    Ws = sliding_window_view(arr, (seq_s, arr.shape[1])).reshape(-1, seq_s, arr.shape[1])
    Wl = sliding_window_view(arr, (seq_l, arr.shape[1])).reshape(-1, seq_l, arr.shape[1])
    Wl = Wl[:, ::ds, :]
    if Wl.shape[1] != seq_s:
        Wl = Wl[:, -seq_s:, :]
    start = seq_l - seq_s
    end = min(Ws.shape[0], Wl.shape[0] - start)
    if end <= 0:
        raise RuntimeError("Window alignment failed; check seqs and downsample.")
    Ws = Ws[start:start+end]; Wl = Wl[:end]
    X = np.concatenate([Ws, Wl], axis=2)
    y_idx = np.arange(start, start+end) + (seq_s - 1)
    return torch.tensor(X, dtype=torch.float32), y_idx

# ========= Coin context =========
class CoinContext:
    def __init__(self, coin_dir):
        self.coin_dir = coin_dir
        self.coin_tag = os.path.basename(coin_dir)          # e.g., AAVE_USD
        self.coin_sym = self.coin_tag.replace("_", "-")     # e.g., AAVE-USD
        with open(os.path.join(coin_dir, "meta.json"), "r") as f:
            self.meta = json.load(f)
        self.feature_cols = self.meta["feature_names"]
        self.input_dim = self.meta["feature_count"]
        self.seq_s = self.meta["seq_len_short"]
        self.seq_l = self.meta["seq_len_long"]
        self.ds    = self.meta["downsample"]
        self.threshold = self.meta["threshold"]
        self.use_btc = any(c.startswith("btc_") for c in self.feature_cols)

        state = torch.load(os.path.join(coin_dir, "model_state.pth"), map_location="cpu")
        self.model = TransformerEntryModel(self.input_dim, self.meta["d_model"], self.meta["n_heads"], self.meta["n_layers"])
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        # scaler fit on train slice using EXACT training features
        csv_path = os.path.join(DATA_DIR, f"{self.coin_tag}.csv")
        df = pd.read_csv(csv_path)
        df_num = df.select_dtypes(include=[np.number]).copy()
        missing = [c for c in self.feature_cols if c not in df_num.columns]
        if missing:
            raise RuntimeError(f"{self.coin_tag}: dataset missing {len(missing)} training columns (e.g., {missing[:10]}). Rebuild dataset or retrain.")
        X = df_num[self.feature_cols].copy()
        n = len(X); n_train = int(n*TRAIN_FRAC)
        self.scaler = StandardScaler().fit(X.iloc[:n_train])

    def _latest_features(self):
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(hours=self.seq_l + 800)
        df = fetch_range(self.coin_sym, start, now, GRANULARITY)
        if df is None or df.empty:
            raise RuntimeError(f"{self.coin_tag}: no live data fetched")
        df = calc_features(df)
        for L in [720,360,168]:
            df = calc_fib(df, L)
        if self.use_btc:
            df = merge_btc(df)
        df_num = df.select_dtypes(include=[np.number]).copy().reindex(columns=self.feature_cols)
        X_scaled = df_num.copy()
        X_scaled[self.feature_cols] = self.scaler.transform(X_scaled[self.feature_cols])
        X, y_idx = make_windows_multiscale(X_scaled, self.feature_cols, self.seq_s, self.seq_l, self.ds)
        return X[-1:].float(), df.iloc[y_idx[-1]]

    def infer(self):
        X, bar = self._latest_features()
        with torch.no_grad():
            lc, tp, sl = self.model(X)
            probs = torch.softmax(lc, dim=1).numpy()[0]
            pred = int(np.argmax(probs)); maxp = float(np.max(probs))
            if maxp < self.threshold or pred == 0:
                return {"coin": self.coin_sym, "time": str(bar.get("time", "")), "status": "HOLD"}
            side = "LONG" if pred == 1 else "SHORT"
            return {
                "coin": self.coin_sym,
                "time": str(bar.get("time","")),
                "status": "ALERT",
                "side": side,
                "confidence": round(maxp, 4),
                "tp_pct": float(tp.squeeze().numpy()),
                "sl_pct": float(sl.squeeze().numpy()),
            }

# ========= Runner =========
def discover_coins(models_dir):
    coins = []
    for name in os.listdir(models_dir):
        coin_dir = os.path.join(models_dir, name)
        if not os.path.isdir(coin_dir): continue
        if not os.path.exists(os.path.join(coin_dir, "meta.json")): continue
        if not os.path.exists(os.path.join(coin_dir, "model_state.pth")): continue
        coins.append(coin_dir)
    return sorted(coins)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["watch","simulate"], default="watch")
    args = ap.parse_args()

    coin_dirs = discover_coins(MODELS_DIR)
    if not coin_dirs:
        print("No coin models found under:", MODELS_DIR); return
    print("Loaded coins:", [os.path.basename(d) for d in coin_dirs])

    # Build contexts
    contexts = [CoinContext(cd) for cd in coin_dirs]

    if args.mode == "simulate":
        # Walk each coin’s CSV test split and print would-be alerts (quick smoke test)
        for ctx in contexts:
            print(f"\n[SIM] {ctx.coin_tag} using threshold={ctx.threshold:.2f}")
            csv_path = os.path.join(DATA_DIR, f"{ctx.coin_tag}.csv")
            df = pd.read_csv(csv_path)
            df_num = df.select_dtypes(include=[np.number]).copy().reindex(columns=ctx.feature_cols)
            n = len(df_num); n_train = int(n*TRAIN_FRAC); n_val = int(n*VAL_FRAC)
            test_df = df_num.iloc[n_train+n_val:].copy()
            test_df[ctx.feature_cols] = ctx.scaler.transform(test_df[ctx.feature_cols])
            X, y_idx = make_windows_multiscale(test_df, ctx.feature_cols, ctx.seq_s, ctx.seq_l, ctx.ds)
            alerts = 0
            with torch.no_grad():
                lc, tp, sl = ctx.model(X)
                probs = torch.softmax(lc, dim=1).numpy()
                maxp = probs.max(axis=1); arg = probs.argmax(axis=1)
                for i in range(len(X)):
                    if maxp[i] >= ctx.threshold and arg[i] != 0:
                        alerts += 1
                print(f"[SIM] Alerts on test slice: {alerts}")
        return

    # Live watch loop
    print("[WATCH] Starting live alerts… polling every", POLL_SEC, "sec")
    last_times = {}
    while True:
        try:
            for ctx in contexts:
                try:
                    sig = ctx.infer()
                    t = sig.get("time")
                    key = (ctx.coin_tag, t)
                    if last_times.get(ctx.coin_tag) != t:
                        last_times[ctx.coin_tag] = t
                        if sig["status"] == "ALERT":
                            print(f"[ALERT] {ctx.coin_sym} {sig['side']} @ {t} conf={sig['confidence']:.2f} tp={sig['tp_pct']:.3f} sl={sig['sl_pct']:.3f}")
                        else:
                            print(f"[OK] {ctx.coin_sym} HOLD @ {t}")
                except Exception as ce:
                    print(f"[ERR][{ctx.coin_tag}] {ce}")
            time.sleep(POLL_SEC)
        except KeyboardInterrupt:
            print("\nStopping…")
            break

if __name__ == "__main__":
    main()
