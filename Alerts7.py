# Alerts7.py — Live hour-close alerts (Transformer, compat loader, timestamp alignment)

import os
import time
import json
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from collections import OrderedDict

import torch
import torch.nn as nn

# TA indicators (pip install ta)
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# ======================
# Config
# ======================
BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY = 3600  # 1h candles
DATA_DIR = r"C:\Users\natec\datasets_macro_training5"

# Models root: one folder per coin (AAVE_USD, BCH_USD, ...)
MODELS_DIR = r"C:\Users\natec\models_transformer_multiscale_safe_stable2"

RUN_ON_HOUR = True
POST_HOUR_RETRIES = 4
RETRY_DELAY_SEC = 12
POLL_SEC = 60

# Manual threshold override (or use --thr)
prob_t = None

# Only used for --mode simulate
TRAIN_FRAC = 0.75
VAL_FRAC   = 0.15

# Telegram ENV (optional)
TG_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TG_CHAT_ID   = "7967738614"



#TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
#CHAT_ID = "7967738614"

#personal chat id:"7967738614"
#channel chat id:
#"-4916911067"
# ======================
# Utils
# ======================
def utc_top_of_hour(dt=None):
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)

def sleep_until_next_hour():
    now = datetime.now(timezone.utc)
    nxt = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    secs = (nxt - now).total_seconds()
    if secs > 0:
        time.sleep(secs)

def send_telegram(text: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print(f"[WARN][TELEGRAM] {e}")

def strip_merge_suffixes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove *_x/_y from merges; keep base where possible."""
    rename = {}
    drop = []
    for c in list(df.columns):
        if c.endswith("_x"):
            base = c[:-2]
            if base not in df.columns:
                rename[c] = base
            else:
                drop.append(c)
        elif c.endswith("_y"):
            drop.append(c)
    if rename:
        df = df.rename(columns=rename)
    if drop:
        df = df.drop(columns=drop)
    return df

def ensure_base_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee open/high/low/close/volume exist and are numeric."""
    need = ["open", "high", "low", "close", "volume"]
    for base in need:
        if base not in df.columns:
            if f"{base}_x" in df.columns:
                df = df.rename(columns={f"{base}_x": base})
            elif f"{base}_y" in df.columns:
                df = df.rename(columns={f"{base}_y": base})
    missing = [b for b in need if b not in df.columns]
    if missing:
        raise RuntimeError(f"live OHLCV column missing: {missing}")
    for b in need:
        df[b] = pd.to_numeric(df[b], errors="coerce")
    return df

# ======================
# Robust live fetch
# ======================
def fetch_range(symbol, start, end, granularity):
    """
    Coinbase fetch in 300-candle pages, ascending, deduped, trimmed to last CLOSED hour.
    Warns on gaps/staleness.
    """
    rows = []
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(seconds=granularity * 300), end)
        params = {"start": cur.isoformat(), "end": chunk_end.isoformat(), "granularity": granularity}
        url = f"{BASE_URL}/products/{symbol}/candles"
        try:
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            if isinstance(data, list) and data:
                dfc = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
                rows.append(dfc)
        except Exception as e:
            print(f"[WARN][fetch_range] {symbol} page error: {e}")
        time.sleep(0.25)
        cur = chunk_end

    if not rows:
        return None

    df = pd.concat(rows, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)

    cut = utc_top_of_hour()
    df = df[df["time"] <= cut].copy()

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if len(df) >= 2:
        gaps = (df["time"].diff().dt.total_seconds().fillna(granularity) != granularity)
        if gaps.tail(10).any():
            miss_idx = df.index[gaps].tolist()[-5:]
            print(f"[DBG][fetch_range] {symbol} gaps near tail @ idx {miss_idx}, max={df['time'].max()} cut={cut}")

    if (cut - df["time"].max()) > timedelta(hours=2):
        print(f"[WARN][fetch_range] {symbol} latest closed={df['time'].max()} is stale vs cut={cut}")

    return df

# ======================
# Feature builders (match training)
# ======================

def fmt_px(px: float) -> str:
    """Format price with dynamic decimals so micro-priced coins (e.g., BONK) look sane."""
    if px is None or not np.isfinite(px):
        return "nan"
    p = float(px)
    # choose decimals by magnitude (tweak to taste)
    if p >= 100:
        d = 2
    elif p >= 1:
        d = 4
    elif p >= 0.1:
        d = 5
    elif p >= 0.01:
        d = 6
    elif p >= 0.001:
        d = 7
    elif p >= 0.0001:
        d = 8
    else:
        d = 9  # super tiny
    return f"{p:.{d}f}"


def calc_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # time features
    df["hour"] = df["time"].dt.hour / 23.0
    df["day_of_week"] = df["time"].dt.dayofweek / 6.0
    df["day_of_month"] = df["time"].dt.day / 31.0

    # RSI
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    # ADX + DI
    adx = ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()
    # MACD diff
    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()
    # EMAs + slopes
    for p in [10, 20, 50, 200]:
        ema = df["close"].ewm(span=p, adjust=False).mean()
        df[f"ema{p}"] = ema
        df[f"ema{p}_slope"] = ema.diff()
    # ATR
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=10).average_true_range()
    # Volume change
    df["vol_change"] = df["volume"].pct_change()
    # Candle body/wick ratio
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_wick_ratio"] = (df["close"] - df["open"]).abs() / rng
    df["body_wick_ratio"] = df["body_wick_ratio"].fillna(0.0)
    # Above EMA200
    df["above_ema200"] = (df["close"] > df["ema200"]).astype(int)
    return df

def calc_fib_prefixed(df: pd.DataFrame, lookback_hours: int, prefix: str) -> pd.DataFrame:
    df = df.copy()
    swing_high = df["high"].rolling(window=lookback_hours, min_periods=1).max()
    swing_low  = df["low"].rolling(window=lookback_hours, min_periods=1).min()
    diff = (swing_high - swing_low).replace(0, np.nan)
    levels = {
        f"{prefix}_0":   swing_low,
        f"{prefix}_236": swing_high - diff * 0.236,
        f"{prefix}_382": swing_high - diff * 0.382,
        f"{prefix}_5":   swing_high - diff * 0.5,
        f"{prefix}_618": swing_high - diff * 0.618,
        f"{prefix}_786": swing_high - diff * 0.786,
        f"{prefix}_1":   swing_high,
    }
    for k, v in levels.items():
        df[k] = v
    return df

# ======================
# Live BTC cache & merge
# ======================
_LIVE_BTC = {"cut": None, "df": None}

def get_btc_live_for_scan(granularity=GRANULARITY):
    """Fetch/build ~60d of BTC up to current top-of-hour; cache per scan."""
    cut = utc_top_of_hour()
    need = (_LIVE_BTC["cut"] != cut) or (_LIVE_BTC["df"] is None)
    if not need:
        last = _LIVE_BTC["df"]["time"].max()
        if (cut - last) > timedelta(hours=1):
            need = True
    if need:
        start = cut - timedelta(days=60)
        btc = fetch_range("BTC-USD", start, cut, granularity)
        if btc is None or btc.empty:
            print("[WARN] BTC live fetch empty.")
            _LIVE_BTC["df"] = None
            _LIVE_BTC["cut"] = cut
            return None
        btc = ensure_base_ohlcv(btc)
        btc = calc_features(btc)
        btc = calc_fib_prefixed(btc, 720, "fib_long")
        btc = calc_fib_prefixed(btc, 360, "fib_med")
        btc = calc_fib_prefixed(btc, 168, "fib_short")
        btc.replace([np.inf, -np.inf], np.nan, inplace=True)
        btc.ffill(inplace=True); btc.bfill(inplace=True)
        # prefix and keep only time + btc_*
        btc = btc.rename(columns={c: (c if c == "time" else f"btc_{c}") for c in btc.columns})
        btc = btc[[c for c in btc.columns if c == "time" or c.startswith("btc_")]]
        _LIVE_BTC["df"] = btc
        _LIVE_BTC["cut"] = cut
    return _LIVE_BTC["df"]

def merge_btc(df):
    """Merge cached live BTC into this coin's DataFrame; drop btc target cols if present."""
    btc_df = get_btc_live_for_scan()
    if btc_df is None or btc_df.empty:
        return df
    drop_cols = [c for c in ("btc_label", "btc_tp_pct", "btc_sl_pct") if c in btc_df.columns]
    if drop_cols:
        btc_df = btc_df.drop(columns=drop_cols)
    merged = pd.merge_asof(
        df.sort_values("time"),
        btc_df.sort_values("time"),
        on="time",
        direction="backward",
        suffixes=("", "")
    )
    return merged

# ======================
# Window builder (multi-scale)
# ======================
def make_windows_multiscale(df_num_scaled: pd.DataFrame, feature_cols, seq_s: int, seq_l: int, ds: int):
    """
    Build a multi-scale window:
      - short: last seq_s rows (step=1)
      - long:  last seq_l*ds rows sampled every ds
    Returns (X, y_idx) where y_idx is the raw row index per window's label position.
    """
    arr = df_num_scaled[feature_cols].values
    n = len(arr)
    windows = []
    y_idx = []
    total_len = seq_s + seq_l
    for end in range(max(seq_s, seq_l*ds), n):
        short = arr[end-seq_s:end, :]
        long_slice = arr[end - seq_l*ds:end:ds, :]
        if len(long_slice) != seq_l:
            continue
        win = np.concatenate([long_slice, short], axis=0)  # [seq_l+seq_s, feat]
        windows.append(win)
        y_idx.append(end - 1)
    if not windows:
        return torch.zeros(0, total_len, len(feature_cols)), np.array([])
    X = torch.tensor(np.stack(windows), dtype=torch.float32)
    return X, np.array(y_idx, dtype=int)

# ======================
# Compat model builder (adapts to checkpoint naming & input dim)
# ======================
def build_transformer_from_state(state_dict, input_dim_meta: int, total_seq: int,
                                 d_model_default=96, nhead=4, num_layers=1):
    """
    Build a Transformer compatible with the checkpoint:
    - Infer input_dim from state['input_proj.weight'].shape[1]
    - Respect 'cls_head' vs 'class_head' naming
    - Tolerate 'posenc.pe' by mapping it to buffer 'posenc_pe' (no dots allowed)
    """
    w = state_dict.get("input_proj.weight", None)
    if w is None:
        raise RuntimeError("Checkpoint missing input_proj.weight; cannot infer input_dim.")
    in_dim_required = w.shape[1]
    d_model = w.shape[0]

    has_cls_head = any(k.startswith("cls_head.") for k in state_dict.keys())
    head_name = "cls_head" if has_cls_head else "class_head"
    has_posenc = any(k.startswith("posenc.pe") or k.startswith("posenc_pe") for k in state_dict.keys())

    class CompatTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq_len = total_seq
            self.input_proj = nn.Linear(in_dim_required, d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            if has_posenc:
                pe = torch.zeros(total_seq, d_model)
                self.register_buffer("posenc_pe", pe, persistent=False)
            head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 3))
            if head_name == "cls_head":
                self.cls_head = head
            else:
                self.class_head = head
            self.tp_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
            self.sl_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x):
            x = self.input_proj(x)
            x = self.encoder(x)
            x = x[:, -1, :]
            logits = self.cls_head(x) if hasattr(self, "cls_head") else self.class_head(x)
            return logits, self.tp_head(x), self.sl_head(x)

    model = CompatTransformer()
    return model, in_dim_required, head_name

# ======================
# Context per coin
# ======================
class CoinContext:
    def __init__(self, coin_dir: str):
        self.coin_dir = coin_dir
        self.coin_tag = os.path.basename(coin_dir)  # e.g., AAVE_USD
        self.coin_sym = self.coin_tag.replace("_", "-")

        # Load meta
        meta_path = os.path.join(coin_dir, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.feature_cols = meta["feature_names"]
        self.seq_s = int(meta.get("seq_len_short", 48))
        self.seq_l = int(meta.get("seq_len_long", 192))
        self.ds    = int(meta.get("downsample", 4))
        self.threshold = float(meta.get("threshold", 0.55))
        self.fees_bps = int(meta.get("fees_bps", 5))
        self.slip_bps = int(meta.get("slip_bps", 5))

        # Sanity: no target-like cols
        bad = [c for c in self.feature_cols if c.endswith(("_label", "_tp_pct", "_sl_pct"))]
        if bad:
            raise RuntimeError(f"{self.coin_tag}: meta feature list contains target-like cols: {bad}")

        # Optional scaler
        scaler_path = os.path.join(coin_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None
            print(f"[INFO][{self.coin_tag}] No scaler.pkl found — assuming model was trained without an external scaler.")

        # Load state & remap posenc key if needed
        state_path = os.path.join(coin_dir, "model_state.pth")
        if not os.path.exists(state_path):
            state_path = os.path.join(coin_dir, "model.pth")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        state = OrderedDict((k.replace("posenc.pe", "posenc_pe"), v) for k, v in state.items())

        total_seq = self.seq_s + self.seq_l
        d_model_meta = int(meta.get("d_model", 96))
        nhead_meta   = int(meta.get("n_heads", 4))
        nlayers_meta = int(meta.get("n_layers", 1))

        # Build compat model to match the checkpoint
        model, in_dim_required, head_name = build_transformer_from_state(
            state_dict=state,
            input_dim_meta=len(self.feature_cols),
            total_seq=total_seq,
            d_model_default=d_model_meta,
            nhead=nhead_meta,
            num_layers=nlayers_meta
        )

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[INFO][{self.coin_tag}] Missing keys when loading: {missing[:6]}{'...' if len(missing)>6 else ''}")
        if unexpected:
            print(f"[INFO][{self.coin_tag}] Unexpected keys ignored: {unexpected[:6]}{'...' if len(unexpected)>6 else ''}")

        self.model = model.eval()
        self.in_dim_required = in_dim_required

    # Build features for last closed candle and one aligned window (timestamp-based alignment)
    def _latest_features(self):
        cut = utc_top_of_hour()

        # 1) Fetch coin live
        start = cut - timedelta(days=60)
        df = fetch_range(self.coin_sym, start, cut, GRANULARITY)
        if df is None or df.empty:
            raise RuntimeError(f"{self.coin_tag}: live fetch empty")

        df = ensure_base_ohlcv(df)
        df = calc_features(df)
        df = calc_fib_prefixed(df, 720, "fib_long")
        df = calc_fib_prefixed(df, 360, "fib_med")
        df = calc_fib_prefixed(df, 168, "fib_short")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True); df.bfill(inplace=True)

        # 2) Merge BTC live (cached)
        df = merge_btc(df)
        df = strip_merge_suffixes(df)
        df = ensure_base_ohlcv(df)
        df = df.sort_values("time")
        df = df[df["time"] <= cut].copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True); df.bfill(inplace=True)

        # ---- Build numeric frame & keep timestamp for alignment
        df_num = df.select_dtypes(include=[np.number]).copy()
        time_series = df["time"].copy().reindex(df_num.index)

        # Ensure training features exist
        missing = [c for c in self.feature_cols if c not in df_num.columns]
        if missing:
            raise RuntimeError(f"{self.coin_tag}: live frame missing training cols, e.g. {missing[:10]}")
        X_num = df_num.reindex(columns=self.feature_cols).copy()

        # Scale if scaler exists
        if self.scaler is not None:
            X_num[self.feature_cols] = self.scaler.transform(X_num[self.feature_cols])

        # Build windows
        X, y_idx = make_windows_multiscale(X_num, self.feature_cols, self.seq_s, self.seq_l, self.ds)
        if len(X) == 0:
            raise RuntimeError(f"{self.coin_tag}: not enough rows for a window")

        # ---- Pin to the last CLOSED hour candle by TIMESTAMP
        target_time = cut - timedelta(hours=1)
        match_df_idx = df.index[df["time"] == target_time]
        if len(match_df_idx):
            row_idx = int(match_df_idx[-1])
        else:
            lt = df[df["time"] < cut]
            if lt.empty:
                raise RuntimeError(f"{self.coin_tag}: no candle strictly before cut")
            row_idx = int(lt.index.max())
        bar_time = df.loc[row_idx, "time"]

        # refuse stale (>2h old)
        if (cut - bar_time) > timedelta(hours=2):
            tail_times = df["time"].tail(3).tolist()
            raise RuntimeError(f"{self.coin_tag}: latest usable bar is stale ({bar_time}). Tail={tail_times}")

        # --- TIMESTAMP alignment into df_num / y_idx space
        pos_candidates = df_num.index[time_series == bar_time]
        if len(pos_candidates):
            pos = int(pos_candidates[-1])
        else:
            earlier = df_num.index[time_series < bar_time]
            if len(earlier) == 0:
                raise RuntimeError(f"{self.coin_tag}: target bar not present in numeric frame and no earlier rows")
            pos = int(earlier.max())

        # Choose last window whose label position <= pos
        valid = np.where(y_idx <= pos)[0]
        if len(valid) == 0:
            raise RuntimeError(f"{self.coin_tag}: no window aligns to the last closed bar")
        win_i = int(valid[-1])

        # Adapt feature dimension if checkpoint expects 2x
        feat_now = X.shape[-1]
        if hasattr(self, "in_dim_required") and self.in_dim_required != feat_now:
            if self.in_dim_required == 2 * feat_now:
                X = torch.cat([X, X], dim=2)
                print(f"[INFO][{self.coin_tag}] Expanded features {feat_now}→{self.in_dim_required} to match checkpoint.")
            else:
                raise RuntimeError(f"{self.coin_tag}: checkpoint expects in_dim={self.in_dim_required}, have {feat_now}; cannot auto-adapt.")

        entry_px = float(df.loc[row_idx, "close"])
        return X[win_i:win_i+1].float(), {"time": bar_time, "entry": entry_px}

    def infer(self):
        X, barinfo = self._latest_features()
        with torch.no_grad():
            lc, tp, sl = self.model(X)
            probs = torch.softmax(lc, dim=1).cpu().numpy()[0]
            p_hold, p_long, p_short = float(probs[0]), float(probs[1]), float(probs[2])
            pred = int(np.argmax(probs))
            maxp = float(np.max(probs))
            bar_ts = str(barinfo.get("time", ""))
            entry_price = float(barinfo.get("entry", np.nan))

            thr = float(self.threshold)
            if maxp < thr or pred == 0:
                return {
                    "coin": self.coin_sym,
                    "time": bar_ts,
                    "status": "HOLD",
                    "prob_hold": p_hold, "prob_long": p_long, "prob_short": p_short
                }

            side = "LONG" if pred == 1 else "SHORT"
            tp_val = float(tp.squeeze().cpu().numpy())
            sl_val = float(sl.squeeze().cpu().numpy())
            return {
                "coin": self.coin_sym,
                "time": bar_ts,
                "status": "ALERT",
                "side": side,
                "confidence": round(maxp, 4),
                "tp_pct": tp_val,
                "sl_pct": sl_val,
                "entry": entry_price,
                "prob_hold": p_hold, "prob_long": p_long, "prob_short": p_short
            }

# ======================
# Discovery
# ======================
def discover_coins(models_dir):
    """Find coin folders that contain meta.json and either model_state.pth or model.pth."""
    out = []
    if not os.path.isdir(models_dir):
        return out
    for name in os.listdir(models_dir):
        cdir = os.path.join(models_dir, name)
        if not os.path.isdir(cdir):
            continue
        meta_ok = os.path.exists(os.path.join(cdir, "meta.json"))
        w1 = os.path.exists(os.path.join(cdir, "model_state.pth"))
        w2 = os.path.exists(os.path.join(cdir, "model.pth"))
        if meta_ok and (w1 or w2):
            out.append(cdir)
    return sorted(out)

# ======================
# Immediate scan (startup)
# ======================
def _immediate_scan(contexts):
    print("[STARTUP] Running immediate scan before hour-close loop...")
    try:
        _ = get_btc_live_for_scan()
    except Exception as e:
        print(f"[WARN] BTC warm failed (startup): {e}")

    results = {}
    for ctx in contexts:
        try:
            sig = ctx.infer()
            bar_time = sig.get("time", "")
            pH = sig.get("prob_hold", 0.0)
            pL = sig.get("prob_long", 0.0)
            pS = sig.get("prob_short", 0.0)
            if sig["status"] == "ALERT":
                side  = sig["side"]
                entry = float(sig.get("entry", float("nan")))
                tp_pct = float(sig["tp_pct"]); sl_pct = float(sig["sl_pct"])
                if side == "LONG":
                    tp_price = entry * (1 + tp_pct)
                    sl_price = entry * (1 - sl_pct)
                    entry_s  = fmt_px(entry)
                    tp_s     = fmt_px(tp_price)
                    sl_s     = fmt_px(sl_price)
                else:
                    tp_price = entry * (1 - tp_pct)
                    sl_price = entry * (1 + sl_pct)
                    entry_s  = fmt_px(entry)
                    tp_s     = fmt_px(tp_price)
                    sl_s     = fmt_px(sl_price)
                print(f"[ALERT] {ctx.coin_sym} {side} @ {bar_time} conf={sig['confidence']:.2f} "
                      f"entry={entry:.4f} TP={tp_price:.4f} SL={sl_price:.4f} "
                      f"[P(H)={pH:.2f}, P(L)={pL:.2f}, P(S)={pS:.2f}]")
                tg = (
                    f"{ctx.coin_sym} {side}\n"
                    f"Time: {bar_time}\n"
                    f"Confidence: {sig['confidence']:.2f}\n"
                    f"Entry: {entry_s}\n"
                    f"TP: {tp_s}  (Δ {tp_pct*100:.2f}%)\n"
                    f"SL: {sl_s}  (Δ {sl_pct*100:.2f}%)\n"
                    f"Thr: {ctx.threshold:.2f}"
                )
                send_telegram(tg)
            else:
                print(f"[OK] {ctx.coin_sym} HOLD @ {bar_time} "
                      f"[P(H)={pH:.2f}, P(L)={pL:.2f}, P(S)={pS:.2f}]")
            results[ctx.coin_tag] = bar_time
        except Exception as ce:
            print(f"[ERR][{ctx.coin_tag}] {ce}")
    return results

# ======================
# Main
# ======================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["watch","simulate"], default="watch")
    ap.add_argument("--thr", type=float, default=None, help="Global probability threshold override, e.g. 0.80")
    args = ap.parse_args()

    coin_dirs = discover_coins(MODELS_DIR)
    if not coin_dirs:
        print("No coin models found under:", MODELS_DIR)
        return
    print("Loaded coins:", [os.path.basename(d) for d in coin_dirs])

    # Build contexts
    contexts = [CoinContext(cd) for cd in coin_dirs]

    # Threshold overrides
    for ctx in contexts:
        if args.thr is not None:
            ctx.threshold = float(args.thr)
        elif prob_t is not None:
            ctx.threshold = float(prob_t)
        print(f"[THR] {ctx.coin_tag}: using threshold {ctx.threshold:.2f}")

    # Immediate run + seed
    _startup_times = _immediate_scan(contexts)

    if args.mode == "simulate":
        for ctx in contexts:
            print(f"\n[SIM] {ctx.coin_tag} using threshold={ctx.threshold:.2f}")
            csv_path = os.path.join(DATA_DIR, f"{ctx.coin_tag}.csv")
            df = pd.read_csv(csv_path)
            df_num = df.select_dtypes(include=[np.number]).copy().reindex(columns=ctx.feature_cols)
            n = len(df_num); n_train = int(n*TRAIN_FRAC); n_val = int(n*VAL_FRAC)
            test_df = df_num.iloc[n_train+n_val:].copy()
            if ctx.scaler is not None:
                test_df[ctx.feature_cols] = ctx.scaler.transform(test_df[ctx.feature_cols])
            X, y_idx = make_windows_multiscale(test_df, ctx.feature_cols, ctx.seq_s, ctx.seq_l, ctx.ds)

            # Adapt to checkpoint input dim if needed
            if len(X):
                feat_now = X.shape[-1]
                if hasattr(ctx, "in_dim_required") and ctx.in_dim_required != feat_now:
                    if ctx.in_dim_required == 2 * feat_now:
                        X = torch.cat([X, X], dim=2)
                        print(f"[INFO][{ctx.coin_tag}] Expanded features {feat_now}→{ctx.in_dim_required} for simulate.")
                    else:
                        print(f"[WARN][{ctx.coin_tag}] simulate dim mismatch {feat_now} vs {ctx.in_dim_required}; skipping.")
                        continue

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

    # Hour-close loop
    print("[WATCH] Hour-close mode — running at the top of each hour.")
    last_times = {}
    try:
        last_times.update(_startup_times)
    except Exception:
        pass

    while True:
        try:
            if RUN_ON_HOUR:
                sleep_until_next_hour()
            else:
                time.sleep(POLL_SEC)

            # Warm BTC once per hour
            try:
                _ = get_btc_live_for_scan()
            except Exception as e:
                print(f"[WARN] BTC warm failed (hourly): {e}")

            for attempt in range(POST_HOUR_RETRIES):
                any_output = False
                for ctx in contexts:
                    try:
                        sig = ctx.infer()
                        bar_time = sig.get("time", "")
                        if bar_time and last_times.get(ctx.coin_tag) != bar_time:
                            last_times[ctx.coin_tag] = bar_time
                            any_output = True
                            pH = sig.get("prob_hold", 0.0)
                            pL = sig.get("prob_long", 0.0)
                            pS = sig.get("prob_short", 0.0)
                            if sig["status"] == "ALERT":
                                side  = sig["side"]
                                entry = float(sig.get("entry", float("nan")))
                                tp_pct = float(sig["tp_pct"])
                                sl_pct = float(sig["sl_pct"])
                                if side == "LONG":
                                    tp_price = entry * (1 + tp_pct)
                                    sl_price = entry * (1 - sl_pct)
                                    entry_s  = fmt_px(entry)
                                    tp_s     = fmt_px(tp_price)
                                    sl_s     = fmt_px(sl_price)
                                else:
                                    tp_price = entry * (1 - tp_pct)
                                    sl_price = entry * (1 + sl_pct)
                                    entry_s  = fmt_px(entry)
                                    tp_s     = fmt_px(tp_price)
                                    sl_s     = fmt_px(sl_price)
                                print(
                                    f"[ALERT] {ctx.coin_sym} {side} @ {bar_time} "
                                    f"conf={sig['confidence']:.2f} entry={entry:.4f} "
                                    f"tp={tp_price:.4f} sl={sl_price:.4f} "
                                    f"(tp%={tp_pct*100:.2f}%, sl%={sl_pct*100:.2f}%) Thr={ctx.threshold:.2f} "
                                    f"[P(H)={pH:.2f}, P(L)={pL:.2f}, P(S)={pS:.2f}]"
                                )
                                tg_text = (
                                    f"{ctx.coin_sym} {side}\n"
                                    f"Time: {bar_time}\n"
                                    f"Confidence: {sig['confidence']:.2f}\n"
                                    f"Entry: {entry_s}\n"
                                    f"TP: {tp_s}  (Δ {tp_pct*100:.2f}%)\n"
                                    f"SL: {sl_s}  (Δ {sl_pct*100:.2f}%)\n"
                                    f"Thr: {ctx.threshold:.2f}"
                                )
                                send_telegram(tg_text)
                            else:
                                print(
                                    f"[OK] {ctx.coin_sym} HOLD @ {bar_time} "
                                    f"[P(H)={pH:.2f}, P(L)={pL:.2f}, P(S)={pS:.2f}]"
                                )
                    except Exception as ce:
                        print(f"[ERR][{ctx.coin_tag}] {ce}")

                if any_output:
                    break
                if attempt < POST_HOUR_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SEC)

        except KeyboardInterrupt:
            print("\nStopping…")
            break

# ======================
if __name__ == "__main__":
    main()
