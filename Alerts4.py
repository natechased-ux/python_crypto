
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

# Force a live override for probability threshold (None = use per-coin meta threshold)
prob_t = None  # set to 0.80 to force stricter alerts

# ========= TA helpers (subset of your dataset_builder) =========
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

BASE_URL = "https://api.exchange.coinbase.com"

# === Telegram config ===
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"   # e.g. 123456789:AA... 
TELEGRAM_CHAT_ID   = "7967738614"     # your chat id (a number or @channel username)




#TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
#CHAT_ID = "7967738614"

#personal chat id:"7967738614"
#channel chat id:
#"-4916911067"

def send_telegram(text: str):
    """Send a plain-text Telegram message. No-op if vars are missing."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True},
            timeout=10
        )
    except Exception as e:
        print(f"[ERR][TELEGRAM] {e}")

# ---- Utilities ----
def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce common OHLCV variants to ['open','high','low','close','volume'] and ensure numeric."""
    if df is None or df.empty:
        return df
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    for std, candidates in {
        "open":   ["open","o"],
        "high":   ["high","h"],
        "low":    ["low","l"],
        "close":  ["close","c"],
        "volume": ["volume","v","vol"],
    }.items():
        for cand in candidates:
            if cand in cols:
                mapping[cols[cand]] = std
                break
    df = df.rename(columns=mapping)
    required = ["open","high","low","close","volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"normalize_ohlcv_columns: missing {missing}; have {list(df.columns)[:12]}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

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
    df.drop_duplicates("time", inplace=True)
    df = normalize_ohlcv_columns(df)
    # normalized time features (raw)
    df["hour"] = df["time"].dt.hour/23.0
    df["day_of_week"] = df["time"].dt.dayofweek/6.0
    df["day_of_month"] = df["time"].dt.day/31.0
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

def calc_fib_prefixed(df, lookback, prefix):
    """Match dataset_builder naming: fib_long_*, fib_med_*, fib_short_*."""
    swing_high = df["high"].rolling(lookback, min_periods=1).max()
    swing_low  = df["low"].rolling(lookback, min_periods=1).min()
    diff = swing_high - swing_low
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_0"]   = swing_low
    out[f"{prefix}_236"] = swing_high - diff*0.236
    out[f"{prefix}_382"] = swing_high - diff*0.382
    out[f"{prefix}_5"]   = swing_high - diff*0.5
    out[f"{prefix}_618"] = swing_high - diff*0.618
    out[f"{prefix}_786"] = swing_high - diff*0.786
    out[f"{prefix}_1"]   = swing_high
    return pd.concat([df, out], axis=1)

def merge_btc(df, granularity=GRANULARITY):
    """Prefer prebuilt BTC CSV from training so btc_* columns match exactly."""
    btc_csv = os.path.join(DATA_DIR, "BTC_USD.csv")
    end = df["time"].max()

    if os.path.exists(btc_csv):
        btc = pd.read_csv(btc_csv)
        btc["time"] = pd.to_datetime(btc["time"], utc=True)
        btc = btc.sort_values("time")
        btc = btc[btc["time"] <= end]
        return pd.merge_asof(df.sort_values("time"), btc, on="time", direction="backward")

    # Fallback live build (won't include btc_label/tp/sl)
    start = end - timedelta(days=60)
    btc_live = fetch_range("BTC-USD", start, end, granularity)
    if btc_live is None or btc_live.empty:
        return df
    btc_live = normalize_ohlcv_columns(btc_live)
    btc_live = calc_features(btc_live)
    btc_live = calc_fib_prefixed(btc_live, 720, "fib_long")
    btc_live = calc_fib_prefixed(btc_live, 360, "fib_med")
    btc_live = calc_fib_prefixed(btc_live, 168, "fib_short")
    btc_live = btc_live.rename(columns={c: f"btc_{c}" for c in btc_live.columns if c != "time"})
    return pd.merge_asof(df.sort_values("time"), btc_live.sort_values("time"), on="time", direction="backward")

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
    X = np.concatenate([Ws, Wl], axis=2)  # doubles channel dim
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
        self.base_input_dim = self.meta["feature_count"]
        self.input_dim = 2 * self.base_input_dim  # MULTISCALE: short+long concatenated
        self.seq_s = self.meta["seq_len_short"]
        self.seq_l = self.meta["seq_len_long"]
        self.ds    = self.meta["downsample"]
        self.threshold = self.meta["threshold"]
        self.use_btc = any(c.startswith("btc_") for c in self.feature_cols)

        # Load weights and sanity-check expected input size
        state = torch.load(os.path.join(coin_dir, "model_state.pth"), map_location="cpu")
        expected_dim = next(v for k,v in state.items() if k.endswith("input_proj.weight")).shape[1]
        if expected_dim != self.input_dim:
            raise RuntimeError(f"{self.coin_tag}: checkpoint expects input_dim={expected_dim}, but alerts computed {self.input_dim} (2*feature_count). Check meta/weights.")

        # Build model with correct input_dim
        self.model = TransformerEntryModel(self.input_dim, self.meta["d_model"], self.meta["n_heads"], self.meta["n_layers"])
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        # scaler fit on train slice using EXACT training base features (not doubled)
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
        start = now - timedelta(hours=self.seq_l + 800)  # warmup

        # 1) FETCH
        df = fetch_range(self.coin_sym, start, now, GRANULARITY)
        if df is None or df.empty:
            raise RuntimeError(f"{self.coin_tag}: no live data fetched")

        # Normalize immediately
        df = normalize_ohlcv_columns(df)
        for col in ["open","high","low","close","volume"]:
            if col not in df.columns:
                raise RuntimeError(f"{self.coin_tag}: missing {col} right after fetch/normalize")

        # 2) FEATURES + FIBS
        df = calc_features(df)
        df = calc_fib_prefixed(df, 720, "fib_long")
        df = calc_fib_prefixed(df, 360, "fib_med")
        df = calc_fib_prefixed(df, 168, "fib_short")

        # Normalize again to guard against any surprises
        df = normalize_ohlcv_columns(df)
        for col in ["open","high","low","close","volume"]:
            if col not in df.columns:
                raise RuntimeError(f"{self.coin_tag}: missing {col} after features/fibs")

        # 3) BTC MERGE
        if self.use_btc:
            df = merge_btc(df)
            # Re-normalize once more (btc merge shouldn’t touch base OHLCV, but let’s be sure)
            df = normalize_ohlcv_columns(df)
            for col in ["open","high","low","close","volume"]:
                if col not in df.columns:
                    raise RuntimeError(f"{self.coin_tag}: missing {col} after BTC merge")

        # Time features (recompute post-merge)
        if "time" not in df.columns:
            raise RuntimeError(f"{self.coin_tag}: 'time' missing after merges")
        df["hour"] = df["time"].dt.hour / 23.0
        df["day_of_week"] = df["time"].dt.dayofweek / 6.0
        df["day_of_month"] = df["time"].dt.day / 31.0

        # Sanitize
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        # Exact training feature order
        df_num = df.select_dtypes(include=[np.number]).copy()
        missing = [c for c in self.feature_cols if c not in df_num.columns]
        if missing:
            print(f"[DBG4][{self.coin_tag}] numeric cols sample: {list(df_num.columns)[:30]}")
            raise RuntimeError(f"{self.coin_tag}: live frame missing training cols, e.g. {missing[:8]}")

        df_num = df_num.reindex(columns=self.feature_cols).dropna()
        if len(df_num) < (self.seq_l + 5):
            raise RuntimeError(f"{self.coin_tag}: not enough clean rows after dropna() (have {len(df_num)})")

        # Scale + window
        X_scaled = df_num.copy()
        X_scaled[self.feature_cols] = self.scaler.transform(df_num[self.feature_cols])
        if not np.isfinite(X_scaled[self.feature_cols].tail(self.seq_l).to_numpy()).all():
            raise RuntimeError(f"{self.coin_tag}: non-finite values remain in latest window slice")

        X, y_idx = make_windows_multiscale(X_scaled, self.feature_cols, self.seq_s, self.seq_l, self.ds)
        last_idx = y_idx[-1]
        return X[-1:].float(), df.iloc[df.index.get_loc(df_num.index[last_idx])]

    def infer(self):
        X, bar = self._latest_features()
        with torch.no_grad():
            lc, tp, sl = self.model(X)
            # Softmax -> probabilities for [HOLD, LONG, SHORT]
            probs = torch.softmax(lc, dim=1).cpu().numpy()[0]
            p_hold, p_long, p_short = float(probs[0]), float(probs[1]), float(probs[2])
            pred = int(np.argmax(probs))
            maxp = float(np.max(probs))
            bar_ts = str(bar.get("time", ""))
            entry_price = float(bar.get("close", np.nan))

            if maxp < self.threshold or pred == 0:
                return {
                    "coin": self.coin_sym,
                    "time": bar_ts,
                    "status": "HOLD",
                    "prob_hold": p_hold,
                    "prob_long": p_long,
                    "prob_short": p_short,
                }

            side = "LONG" if pred == 1 else "SHORT"
            tp_pct = float(tp.squeeze().cpu().numpy())
            sl_pct = float(sl.squeeze().cpu().numpy())
            return {
                "coin": self.coin_sym,
                "time": bar_ts,
                "status": "ALERT",
                "side": side,
                "confidence": round(maxp, 4),
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "entry": entry_price,
                "prob_hold": p_hold,
                "prob_long": p_long,
                "prob_short": p_short,
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

RUN_ON_HOUR = True
POST_HOUR_RETRIES = 5     # how many times to re-check after the hour
RETRY_DELAY_SEC   = 15    # wait between retries
SKEW_BUFFER_SEC   = 2     # tiny buffer so we don't wake up early

def sleep_until_next_hour():
    now = datetime.now(timezone.utc)
    next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    delta = (next_hour - now).total_seconds()
    if delta > 0:
        time.sleep(delta + SKEW_BUFFER_SEC)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["watch","simulate"], default="watch")
    ap.add_argument("--thr", type=float, default=None, help="Global probability threshold override (e.g., 0.80).")
    args = ap.parse_args()

    coin_dirs = discover_coins(MODELS_DIR)
    if not coin_dirs:
        print("No coin models found under:", MODELS_DIR); return
    print("Loaded coins:", [os.path.basename(d) for d in coin_dirs])

    # Build contexts
    contexts = [CoinContext(cd) for cd in coin_dirs]

    # Apply threshold overrides (CLI > hardcoded prob_t > meta)
    for ctx in contexts:
        if args.thr is not None:
            ctx.threshold = float(args.thr)
        elif prob_t is not None:
            ctx.threshold = float(prob_t)
        print(f"[THR] {ctx.coin_tag}: using threshold {ctx.threshold:.2f}")

    # Immediate run once on startup (latest closed candle)
    print("[STARTUP] Running immediate scan before hour-close loop...")
    for ctx in contexts:
        try:
            sig = ctx.infer()
            bar_time = sig.get("time", "")
            pH = sig.get("prob_hold", 0.0); pL = sig.get("prob_long", 0.0); pS = sig.get("prob_short", 0.0)
            print(f"[{ctx.coin_sym}] Probs -> P(H)={pH:.2f}, P(L)={pL:.2f}, P(S)={pS:.2f}")
            if sig["status"] == "ALERT":
                side  = sig["side"]
                entry = sig.get("entry", float("nan"))
                tp_pct = sig["tp_pct"]; sl_pct = sig["sl_pct"]
                if side == "LONG":
                    tp_price = entry * (1 + tp_pct); sl_price = entry * (1 - sl_pct)
                else:
                    tp_price = entry * (1 - tp_pct); sl_price = entry * (1 + sl_pct)
                msg = (f"[ALERT] {ctx.coin_sym} {side} @ {bar_time} "
                       f"conf={sig['confidence']:.2f} "
                       f"entry={entry:.4f} TP={tp_price:.4f} SL={sl_price:.4f}")
                print(msg)
                # send_telegram(msg)  # uncomment to notify on startup too
            else:
                print(f"[OK] {ctx.coin_sym} HOLD @ {bar_time}")
        except Exception as ce:
            print(f"[ERR][{ctx.coin_tag}] {ce}")

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
                probs = torch.softmax(lc, dim=1).cpu().numpy()
                maxp = probs.max(axis=1); arg = probs.argmax(axis=1)
                for i in range(len(X)):
                    if maxp[i] >= ctx.threshold and arg[i] != 0:
                        alerts += 1
                print(f"[SIM] Alerts on test slice: {alerts}")
        return

    # ===== Hour-close aligned watch loop =====
    print("[WATCH] Hour-close mode — running at the top of each hour.")
    last_times = {}  # coin_tag -> last bar timestamp we alerted on

    while True:
        try:
            if RUN_ON_HOUR:
                sleep_until_next_hour()
            else:
                time.sleep(POLL_SEC)

            # After the hour flips, retry a few times to wait for the new candle to appear
            for attempt in range(POST_HOUR_RETRIES):
                any_output = False

                for ctx in contexts:
                    try:
                        sig = ctx.infer()
                        bar_time = sig.get("time", "")

                        # Only act when we see a NEW bar vs the last one we processed
                        if bar_time and last_times.get(ctx.coin_tag) != bar_time:
                            last_times[ctx.coin_tag] = bar_time
                            any_output = True

                            # Probs for display (present for both HOLD and ALERT)
                            pH = sig.get("prob_hold", 0.0)
                            pL = sig.get("prob_long", 0.0)
                            pS = sig.get("prob_short", 0.0)

                            if sig["status"] == "ALERT":
                                side   = sig["side"]
                                entry  = sig.get("entry", float("nan"))
                                tp_pct = sig["tp_pct"]
                                sl_pct = sig["sl_pct"]

                                # Compute TP/SL prices from entry (tp/sl are FRACTIONS)
                                if side == "LONG":
                                    tp_price = entry * (1 + tp_pct)
                                    sl_price = entry * (1 - sl_pct)
                                else:  # SHORT
                                    tp_price = entry * (1 - tp_pct)
                                    sl_price = entry * (1 + sl_pct)

                                # Console with probabilities
                                print(
                                    f"[ALERT] {ctx.coin_sym} {side} @ {bar_time} "
                                    f"conf={sig['confidence']:.2f} "
                                    f"entry={entry:.4f} tp={tp_price:.4f} sl={sl_price:.4f} "
                                    f"(tp%={tp_pct*100:.2f}%, sl%={sl_pct*100:.2f}%) Thr={ctx.threshold:.2f} "
                                    f"[P(H)={pH:.2f}, P(L)={pL:.2f}, P(S)={pS:.2f}]"
                                )

                                # Telegram
                                tg_text = (
                                    f"{ctx.coin_sym} {side}\n"
                                    f"Time: {bar_time}\n"
                                    f"Confidence: {sig['confidence']:.2f}\n"
                                    f"Entry: {entry:.4f}\n"
                                    f"TP: {tp_price:.4f}  (Δ {tp_pct*100:.2f}%)\n"
                                    f"SL: {sl_price:.4f}  (Δ {sl_pct*100:.2f}%)\n"
                                    f"Thr: {ctx.threshold:.2f}"
                                )
                                send_telegram(tg_text)
                            else:
                                # HOLD with probabilities to console
                                print(
                                    f"[OK] {ctx.coin_sym} HOLD @ {bar_time} "
                                    f"[P(H)={pH:.2f}, P(L)={pL:.2f}, P(S)={pS:.2f}]"
                                )

                    except Exception as ce:
                        print(f"[ERR][{ctx.coin_tag}] {ce}")

                # If we saw new bars for any coin, we can break early; otherwise retry
                if any_output:
                    break
                if attempt < POST_HOUR_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SEC)

        except KeyboardInterrupt:
            print("\nStopping…")
            break

if __name__ == "__main__":
    main()
