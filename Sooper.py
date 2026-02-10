import os, requests, pandas as pd, numpy as np, time, joblib, math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from requests.adapters import HTTPAdapter, Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight

"""
Refactor highlights
-------------------
- Stronger mean‑reversion logic (BB z‑score + RSI + regime filter)
- Safer HTTP (retry/backoff, shared Session)
- Faster scans via ThreadPoolExecutor (tunable)
- Model loads are cached; no reload per coin
- Per‑strategy cooldowns & duplicate‑alert guard
- Cleaner config (env overrides, coin exclusions)
- Tighter labeling (no lookahead leakage on features)
- CSV logging for alerts & outcomes (foundation for ML phase 2)

Keep your original behavior toggleable; defaults are stricter.
"""

# ===================== CONFIG =====================
# (New) Optional per-coin MR threshold overrides will be loaded from JSON
PER_COIN_CFG = Path("per_coin_thresholds.json")
DEFAULT_MR_CFG = {  # used if a coin has no custom row
    "rsi_low": 30,
    "rsi_high": 70,
    "z_low": -1.0,
    "z_high": 1.0,
    "adx_max": 20,
    "tp_mult": 1.2,
    "sl_mult": 1.0,
}

# Telegram creds can be overridden by env vars to avoid hardcoding
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "7967738614")

# Coinbase product universe (start from your list, then drop the underperformers you flagged)
DEFAULT_COINS = [
    "btc-usd","eth-usd","xrp-usd","ltc-usd","ada-usd","doge-usd",
    "sol-usd","wif-usd","ondo-usd","sei-usd","magic-usd","ape-usd",
    "jasmy-usd","wld-usd","syrup-usd","fartcoin-usd","aero-usd",
    "link-usd","hbar-usd","aave-usd","fet-usd","crv-usd","tao-usd",
    "avax-usd","xcn-usd","uni-usd","mkr-usd","toshi-usd","near-usd",
    "algo-usd","trump-usd","bch-usd","inj-usd","pepe-usd","xlm-usd",
    "moodeng-usd","bonk-usd","dot-usd","popcat-usd","arb-usd","icp-usd",
    "qnt-usd","tia-usd","ip-usd","pnut-usd","apt-usd","ena-usd","turbo-usd",
    "bera-usd","pol-usd","mask-usd","pyth-usd","sand-usd","morpho-usd",
    "mana-usd","coti-usd","c98-usd","axs-usd"
]

EXCLUDED_COINS = {
    # From your "remove these" list
    "velo-usd", "syrup-usd"
}

COINS = sorted([c for c in DEFAULT_COINS if c not in EXCLUDED_COINS])

# Candles & lookahead
GRANULARITY = 900  # 15m
MAX_CANDLES = 8000
LOOKAHEAD_CANDLES = 6

# Risk (ATR-based) — can be tuned per strategy
TP_MULT = {
    "Breakout": 2.0,
    "Imminent": 1.8,
    "Trend": 1.5,
    "MeanRev": 1.2,
}
SL_MULT = {
    "Breakout": 1.2,
    "Imminent": 1.0,
    "Trend": 1.0,
    "MeanRev": 1.0,
}

# Thresholds
BREAKOUT_THRESHOLD = 0.75
IMMINENT_THRESHOLD = 0.75
TREND_THRESHOLD = 0.72
MEANREV_THRESHOLD = 0.74
HIST_MIN = 0.72  # historical win rate floor

# Cooldowns (seconds) per strategy, per coin
COOLDOWN_S = {
    "Breakout": 45*60,
    "Imminent": 45*60,
    "Trend": 30*60,
    "MeanRev": 30*60,
}

# Models (persisted)
MODEL_PATHS = {
    "Breakout": Path("breakout_model.pkl"),
    "Imminent": Path("imminent_breakout_model.pkl"),
    "Trend": Path("trend_follow_model.pkl"),
    "MeanRev": Path("mean_reversion_model.pkl"),
}
LAST_TRAIN_FILE = Path("last_train.txt")

# Concurrency
MAX_WORKERS = min(8, max(2, os.cpu_count() or 4))

# Files
ALERT_LOG = Path("live_alert_log.csv")

# ================== HTTP / HELPERS ==================

def _session():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET","POST")
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

HTTP = _session()


def format_price(v: float) -> str:
    return f"${v:.2f}" if v >= 100 else f"${v:.4f}" if v >= 1 else f"${v:.8f}"


def fetch_data(symbol: str, granularity: int = GRANULARITY, limit: int = MAX_CANDLES) -> pd.DataFrame | None:
    """
    Robust, paginated fetch for Coinbase candles. Oldest -> newest.
    """
    try:
        max_per_request = 300
        all_dfs = []
        end_time = datetime.now(timezone.utc)
        while len(all_dfs) * max_per_request < limit:
            start_time = end_time - timedelta(seconds=granularity * max_per_request)
            url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
            params = {
                "granularity": int(granularity),
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            }
            r = HTTP.get(url, params=params, timeout=15)
            if r.status_code != 200:
                if r.status_code == 429:
                    time.sleep(5)
                    continue
                return None
            rows = r.json()
            if not rows:
                break
            df = pd.DataFrame(rows, columns=["time","low","high","open","close","volume"]).astype({"low":float,"high":float,"open":float,"close":float,"volume":float})
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            all_dfs.append(df)
            end_time = start_time
        if not all_dfs:
            return None
        full_df = pd.concat(all_dfs, ignore_index=True).sort_values("time").reset_index(drop=True)
        return full_df.tail(limit)
    except Exception:
        return None


# ================== INDICATORS ==================

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = RSIIndicator(out["close"]).rsi()
    out["atr"] = AverageTrueRange(out["high"], out["low"], out["close"]).average_true_range()
    out["atr_norm"] = out["atr"] / out["close"]
    bb = BollingerBands(out["close"], window=20, window_dev=2)
    out["bb_upper"], out["bb_lower"], out["bb_mid"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["close"]
    out["adx"] = ADXIndicator(out["high"], out["low"], out["close"]).adx()
    out["macd_diff"] = MACD(out["close"]).macd_diff()
    out["ema20"] = out["close"].ewm(span=20).mean()
    out["ema50"] = out["close"].ewm(span=50).mean()
    # BB z-score: distance from mid in sigma units
    out["bb_sigma"] = (out["close"] - out["bb_mid"]) / (out["bb_upper"] - out["bb_mid"] + 1e-9)
    return out


# ================== LABELS ==================
# NOTE: We compute labels using ONLY present features; we don't leak future info into features.

def _label_template(df: pd.DataFrame, cond_func) -> pd.DataFrame:
    labels = [np.nan] * len(df)
    for i in range(len(df) - LOOKAHEAD_CANDLES):
        labels[i] = int(cond_func(df, i))
    out = df.copy()
    out["label"] = labels
    return out.dropna()





def label_breakouts(df: pd.DataFrame) -> pd.DataFrame:
    def cond(d, i):
        row = d.iloc[i]
        atr = row["atr"]; pn = row["close"]
        fmax = d["high"].iloc[i+1:i+LOOKAHEAD_CANDLES+1].max()
        fmin = d["low"].iloc[i+1:i+LOOKAHEAD_CANDLES+1].min()
        return (fmax - pn) > 1.0*atr or (pn - fmin) > 1.0*atr
    return _label_template(df, cond)


def label_imminent(df: pd.DataFrame) -> pd.DataFrame:
    mean50 = df["bb_width"].rolling(50).mean()
    def cond(d, i):
        if d["bb_width"].iloc[i] >= 0.85 * mean50.iat[i]:
            return 0
        row = d.iloc[i]
        atr = row["atr"]; pn = row["close"]
        fmax = d["high"].iloc[i+1:i+LOOKAHEAD_CANDLES+1].max()
        fmin = d["low"].iloc[i+1:i+LOOKAHEAD_CANDLES+1].min()
        return int((fmax - pn) > 1.0*atr or (pn - fmin) > 1.0*atr)
    return _label_template(df, cond)


def label_trend_follow(df: pd.DataFrame) -> pd.DataFrame:
    def cond(d, i):
        row = d.iloc[i]
        bull = row["ema20"] > row["ema50"] and row["adx"] > 20
        bear = row["ema20"] < row["ema50"] and row["adx"] > 20
        atr = row["atr"]; pn = row["close"]
        fmax = d["high"].iloc[i+1:i+LOOKAHEAD_CANDLES+1].max()
        fmin = d["low"].iloc[i+1:i+LOOKAHEAD_CANDLES+1].min()
        return (bull and (fmax - pn) > 1.0*atr) or (bear and (pn - fmin) > 1.0*atr)
    return _label_template(df, cond)


def label_mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
    # ADX<20 (chop), BB z-score extremes
    def cond(d, i):
        row = d.iloc[i]
        adx_ok = row["adx"] < 20
        overbought = (row["rsi"] > 70 and row["bb_sigma"] > 1.0)
        oversold   = (row["rsi"] < 30 and row["bb_sigma"] < -1.0)
        if not (overbought or oversold):
            return 0
        mid = row["bb_mid"]; pn = row["close"]
        fmax = d["high"].iloc[i+1:i+LOOKAHEAD_CANDLES+1].max()
        fmin = d["low"].iloc[i+1:i+LOOKAHEAD_CANDLES+1].min()
        if overbought and adx_ok:
            return int(fmin <= mid or (pn - fmin) >= 0.8 * (pn - mid))
        if oversold and adx_ok:
            return int(fmax >= mid or (fmax - pn) >= 0.8 * (mid - pn))
        return 0
    return _label_template(df, cond)



# ================== FEATURES ==================

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df[["atr_norm","bb_width","rsi","macd_diff","adx","volume","open","close","high","low","bb_sigma"]].copy()
    f["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    f["price_vs_mid"] = df["close"] - df["bb_mid"]
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    f["body_ratio"] = (df["close"] - df["open"]).abs() / (rng + 1e-9)
    return f.fillna(0.0)


# ================== TRAIN / PREDICT ==================

MODELS: dict[str, LGBMClassifier] = {}


def _load_model(name: str) -> LGBMClassifier | None:
    if name in MODELS:
        return MODELS[name]
    p = MODEL_PATHS[name]
    if not p.exists():
        return None
    m = joblib.load(p)
    MODELS[name] = m
    return m


def predict_probability(df_last: pd.DataFrame, name: str) -> float | None:
    m = _load_model(name)
    if m is None:
        return None
    X = extract_features(df_last.iloc[[-1]])
    return float(m.predict_proba(X)[0,1])


def train_model(label_func, save_path: Path, label_name: str, historical_data: dict[str, pd.DataFrame]) -> bool:
    rows = []
    pos = neg = used = skipped = 0
    for coin, df in historical_data.items():
        dl = label_func(df)
        if dl.empty:
            skipped += 1
            continue
        X, y = extract_features(dl), dl["label"].astype(int)
        pos += int((y==1).sum()); neg += int((y==0).sum())
        rows.append((X,y)); used += 1
    if not rows:
        print(f"❌ No data for {label_name} model."); return False
    X_all = pd.concat([x for x,_ in rows], ignore_index=True)
    y_all = pd.concat([y for _,y in rows], ignore_index=True)
    weights = compute_sample_weight(class_weight="balanced", y=y_all)
    model = LGBMClassifier(n_estimators=600, learning_rate=0.04, max_depth=7, subsample=0.8, colsample_bytree=0.8)
    model.fit(X_all, y_all, sample_weight=weights)
    joblib.dump(model, save_path)
    print(f"✅ {label_name} trained | coins {used}/{len(historical_data)} | examples {len(X_all)} | +{pos}/-{neg}")
    return True


def retrain_if_needed():
    now = datetime.now(timezone.utc)
    if LAST_TRAIN_FILE.exists():
        try:
            last = datetime.fromisoformat(LAST_TRAIN_FILE.read_text().strip())
            if (now - last).days < 7:
                return
        except Exception:
            pass
    print("🔄 Fetching history for training…")
    hist: dict[str, pd.DataFrame] = {}
    for coin in COINS:
        df = fetch_data(coin, GRANULARITY, MAX_CANDLES)
        if df is None or df.empty:
            continue
        hist[coin] = compute_indicators(df)
        time.sleep(0.2)
    print("🔄 Training models…")
    train_model(label_breakouts, MODEL_PATHS["Breakout"], "Breakout", hist)
    train_model(label_imminent, MODEL_PATHS["Imminent"], "Imminent", hist)
    train_model(label_trend_follow, MODEL_PATHS["Trend"], "Trend", hist)
    train_model(label_mean_reversion, MODEL_PATHS["MeanRev"], "MeanRev", hist)
    LAST_TRAIN_FILE.write_text(now.isoformat())


# =========== HISTORICAL WIN RATE (similarity) ===========

def historical_win_rate(current_df: pd.DataFrame, label_func, symbol: str) -> float | None:
    try:
        feat_now = extract_features(current_df.iloc[[-1]]).iloc[0]
        hist = fetch_data(symbol, GRANULARITY, 4000)  # ~41 days on 15m
        if hist is None or hist.empty:
            return None
        hist = compute_indicators(hist)
        dl = label_func(hist)
        if dl.empty:
            return None
        Xh = extract_features(dl)
        yh = dl["label"].astype(int)
        mask = (
            (Xh["bb_width"].between(feat_now["bb_width"]*0.85, feat_now["bb_width"]*1.15)) &
            (Xh["rsi"].between(feat_now["rsi"]-5, feat_now["rsi"]+5)) &
            (Xh["adx"].between(feat_now["adx"]-5, feat_now["adx"]+5))
        )
        sim = yh[mask]
        if len(sim) == 0:
            return None
        return float((sim == 1).mean())
    except Exception:
        return None


# =============== MULTI‑TF BIAS (1H) ===============

def multi_tf_bias(symbol: str) -> str | None:
    df1h = fetch_data(symbol, 3600, 200)
    if df1h is None or df1h.empty:
        return None
    d = compute_indicators(df1h).iloc[-1]
    if d["adx"] > 22:
        return "Bullish" if d["close"] > d["bb_mid"] else "Bearish"
    return None


# ================== ALERTING ==================

# (New) per-coin config helpers
_PER_COIN_CACHE: dict[str, dict] = {}

def load_per_coin_cfg() -> dict[str, dict]:
    global _PER_COIN_CACHE
    if _PER_COIN_CACHE:
        return _PER_COIN_CACHE
    if PER_COIN_CFG.exists():
        try:
            import json
            data = json.loads(PER_COIN_CFG.read_text())
            # normalize keys to lowercase symbols
            _PER_COIN_CACHE = {k.lower(): {**DEFAULT_MR_CFG, **v} for k, v in data.items()}
        except Exception:
            _PER_COIN_CACHE = {}
    return _PER_COIN_CACHE


def get_coin_cfg(symbol: str) -> dict:
    data = load_per_coin_cfg()
    return data.get(symbol.lower(), DEFAULT_MR_CFG)



_last_alert_time: dict[tuple[str,str], datetime] = {}
_last_alert_hash: set[str] = set()


def _cooldown_ok(symbol: str, strat: str) -> bool:
    key = (symbol, strat)
    now = datetime.now(timezone.utc)
    tprev = _last_alert_time.get(key)
    if tprev is None:
        return True
    return (now - tprev).total_seconds() >= COOLDOWN_S[strat]


def _mark_alert(symbol: str, strat: str, msg_hash: str):
    _last_alert_time[(symbol, strat)] = datetime.now(timezone.utc)
    _last_alert_hash.add(msg_hash)


def _alert_hash(symbol: str, strat: str, side: str, price: float) -> str:
    return f"{symbol}|{strat}|{side}|{round(price, 6)}"


def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    HTTP.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=15)


def log_alert(row: dict):
    df = pd.DataFrame([row])
    if ALERT_LOG.exists():
        df.to_csv(ALERT_LOG, mode="a", header=False, index=False)
    else:
        df.to_csv(ALERT_LOG, index=False)


def build_and_send(strategy: str, symbol: str, side: str, price: float, atr: float,
                   prob: float | None, hwr: float | None,
                   tp_mult: float | None = None, sl_mult: float | None = None):
    tpm = tp_mult if tp_mult is not None else TP_MULT[strategy]
    slm = sl_mult if sl_mult is not None else SL_MULT[strategy]
    tp = price + tpm*atr if side in ("Long","Bullish") else price - tpm*atr
    sl = price - slm*atr if side in ("Long","Bullish") else price + slm*atr
    conf_pct = 100.0*(prob if prob is not None else 0.0)
    hwr_pct = None if hwr is None else 100.0*hwr
    tier = ("🟢 Strong" if conf_pct>=90 and (hwr_pct is None or hwr_pct>=90) else
            "🟡 Moderate" if conf_pct>=80 and (hwr_pct is None or hwr_pct>=80) else
            "⚪ Borderline")
    hist_str = f"{hwr_pct:.1f}%" if hwr_pct is not None else "N/A"
    msg = (
        f"{tier} *{strategy}* — {symbol.upper()}\n"
        f"Side: {side}\n"
        f"Price: {format_price(price)}\n"
        f"TP: {format_price(tp)} | SL: {format_price(sl)}\n"
        f"Confidence: {conf_pct:.1f}%\n"
        f"Historical win rate (similar): {hist_str}"
    )
    h = _alert_hash(symbol, strategy, side, price)
    if h in _last_alert_hash:
        return
    send_telegram(msg)
    _mark_alert(symbol, strategy, h)
    log_alert({
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "strategy": strategy,
        "side": side,
        "price": price,
        "tp": tp,
        "sl": sl,
        "prob": prob,
        "hist_win_rate": hwr,
    })



# ================== STRATEGIES ==================

def check_breakout(symbol: str, df: pd.DataFrame):
    strat = "Breakout"
    if not _cooldown_ok(symbol, strat):
        return
    d = df.iloc[-1]
    vol_mean = df["volume"].rolling(20).mean().iloc[-1]
    squeeze = d["bb_width"] < 0.8 * df["bb_width"].rolling(50).mean().iloc[-1]
    vol_surge = d["volume"] > 1.6 * vol_mean
    trend = d["adx"] > 25
    bias = multi_tf_bias(symbol)
    if squeeze and vol_surge and trend:
        side = None
        if d["close"] > df["high"].rolling(20).max().iloc[-2] and bias == "Bullish":
            side = "Long"
        elif d["close"] < df["low"].rolling(20).min().iloc[-2] and bias == "Bearish":
            side = "Short"
        if side:
            prob = predict_probability(df, strat)
            hwr = historical_win_rate(df, label_breakouts, symbol)
            if (prob is not None and prob >= BREAKOUT_THRESHOLD) and (hwr is not None and hwr >= HIST_MIN):
                build_and_send(strat, symbol, side, d["close"], d["atr"], prob, hwr)


def check_imminent(symbol: str, df: pd.DataFrame):
    strat = "Imminent"
    if not _cooldown_ok(symbol, strat):
        return
    d = df.iloc[-1]
    atr = d["atr"]
    press_up = d["close"] > d["bb_mid"] and d["close"] > (d["bb_upper"] - 0.25*atr)
    press_dn = d["close"] < d["bb_mid"] and d["close"] < (d["bb_lower"] + 0.25*atr)
    accum = df["volume"].rolling(3).mean().iloc[-1] > 0.9 * df["volume"].rolling(10).mean().iloc[-1]
    bias = multi_tf_bias(symbol)
    if d["bb_width"] < 0.85 * df["bb_width"].rolling(50).mean().iloc[-1] and accum:
        side = "Bullish" if (press_up and bias == "Bullish") else ("Bearish" if (press_dn and bias == "Bearish") else None)
        if side:
            prob = predict_probability(df, strat)
            hwr = historical_win_rate(df, label_imminent, symbol)
            if (prob is not None and prob >= IMMINENT_THRESHOLD) and (hwr is not None and hwr >= HIST_MIN):
                build_and_send(strat, symbol, side, d["close"], atr, prob, hwr)


def check_trend(symbol: str, df: pd.DataFrame):
    strat = "Trend"
    if not _cooldown_ok(symbol, strat):
        return
    d = df.iloc[-1]
    bull_pb = d["ema20"] > d["ema50"] and d["adx"] > 20 and d["rsi"] < 60
    bear_pb = d["ema20"] < d["ema50"] and d["adx"] > 20 and d["rsi"] > 40
    side = "Long" if bull_pb else ("Short" if bear_pb else None)
    if side:
        prob = predict_probability(df, strat)
        hwr = historical_win_rate(df, label_trend_follow, symbol)
        if (prob is not None and prob >= TREND_THRESHOLD) and (hwr is not None and hwr >= HIST_MIN):
            build_and_send(strat, symbol, side, d["close"], d["atr"], prob, hwr)


def check_meanrev(symbol: str, df: pd.DataFrame):
    strat = "MeanRev"
    if not _cooldown_ok(symbol, strat):
        return
    d = df.iloc[-1]
    atr = d["atr"]
    cfg = get_coin_cfg(symbol)
    overbought = (d["rsi"] > cfg["rsi_high"] and d["bb_sigma"] > cfg["z_high"])
    oversold  = (d["rsi"] < cfg["rsi_low"]  and d["bb_sigma"] < cfg["z_low"]) 
    adx_ok = d["adx"] < cfg["adx_max"]
    if adx_ok and (overbought or oversold):
        side = "Short" if overbought else "Long"
        prob = predict_probability(df, strat)
        hwr = historical_win_rate(df, label_mean_reversion, symbol)
        if (prob is not None and prob >= MEANREV_THRESHOLD) and (hwr is not None and hwr >= HIST_MIN):
            build_and_send(strat, symbol, side, d["close"], atr, prob, hwr, tp_mult=cfg["tp_mult"], sl_mult=cfg["sl_mult"]) 


# ================== DRIVER ==================

def scan_symbol(symbol: str):
    df = fetch_data(symbol, GRANULARITY, MAX_CANDLES)
    if df is None or df.empty:
        return
    df = compute_indicators(df)
    check_breakout(symbol, df)
    check_imminent(symbol, df)
    check_trend(symbol, df)
    check_meanrev(symbol, df)


def run_once():
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(scan_symbol, sym): sym for sym in COINS}
        for _ in as_completed(futures):
            pass


def main_loop():
    print("🚀 Starting Refactored Multi‑Strategy Bot…")
    retrain_if_needed()
    while True:
        t0 = time.time()
        run_once()
        dt = max(5, 300 - int(time.time() - t0))  # keep ~5m cadence
        time.sleep(dt)


# ================== OFFLINE TUNER (MR) ==================
# Grid-search per-coin thresholds using your live_alert_log.csv.
# Run once with:  TUNE_MR=1 python superbot_refactor_meanreversion_plus.py

def _infer_outcome_for_alert(row: pd.Series, horizon_candles: int = 96) -> int | None:
    """Return 1 if TP first, 0 if SL first, None if neither within horizon."""
    try:
        sym = row["symbol"]
        ts = datetime.fromisoformat(row["ts"])  # alert time (UTC iso)
        price = float(row["price"]); tp = float(row["tp"]); sl = float(row["sl"])
        df = fetch_data(sym, GRANULARITY, max(400, horizon_candles+200))
        if df is None or df.empty:
            return None
        # find index closest after alert ts
        idx = df.index[df["time"] >= pd.Timestamp(ts, tz="UTC")]
        if len(idx) == 0:
            return None
        i0 = int(idx[0])
        # walk forward window
        highs = df["high"].iloc[i0:i0+horizon_candles].values
        lows  = df["low"].iloc[i0:i0+horizon_candles].values
        for h, l in zip(highs, lows):
            if price <= tp <= h or price >= tp >= l:  # tp touched
                return 1
            if price >= sl >= l or price <= sl <= h:  # sl touched
                return 0
        return None
    except Exception:
        return None


def tune_meanrev(csv_path: str = str(ALERT_LOG), min_trades: int = 20):
    import json
    if not Path(csv_path).exists():
        print(f"No CSV at {csv_path} — nothing to tune."); return
    df = pd.read_csv(csv_path)
    df = df[df["strategy"] == "MeanRev"].copy()
    if df.empty:
        print("No MeanRev rows to tune."); return
    # Ensure outcomes exist; if not, infer
    if "outcome" not in df.columns:
        df["outcome"] = df.apply(_infer_outcome_for_alert, axis=1)
    df = df.dropna(subset=["outcome"]).copy()
    df["outcome"] = df["outcome"].astype(int)

    # Search space (small but effective). You can widen later.
    RSI_LOW  = [25, 28, 30, 32]
    RSI_HIGH = [68, 70, 72, 75]
    Z_LOW    = [-1.25, -1.0, -0.9]
    Z_HIGH   = [0.9, 1.0, 1.25]
    ADX_MAX  = [18, 20, 22]
    TP_MULTS = [1.0, 1.2, 1.4]
    SL_MULTS = [0.8, 1.0, 1.2]

    out_cfg: dict[str, dict] = {}
    for sym, g in df.groupby("symbol"):
        best = None
        for rl in RSI_LOW:
            for rh in RSI_HIGH:
                if rh <= rl: continue
                for zl in Z_LOW:
                    for zh in Z_HIGH:
                        if zh <= abs(zl): pass
                        for ax in ADX_MAX:
                            for tpM in TP_MULTS:
                                for slM in SL_MULTS:
                                    sub = g[(g["rsi_at_alert"].between(rl-2, rh+2, inclusive="both") if "rsi_at_alert" in g else True)]
                                    # If rsi_at_alert not logged, fall back to all rows (coarse tune)
                                    n = len(sub)
                                    if n < min_trades:
                                        continue
                                    win = sub["outcome"].sum()
                                    loss = n - win
                                    win_rate = win / n
                                    # simple expectancy proxy using R: (tpM * WR) - (slM * (1-WR))
                                    expectancy = (tpM * win_rate) - (slM * (1 - win_rate))
                                    score = 0.65*expectancy + 0.35*win_rate
                                    cand = (score, win_rate, n, {"rsi_low": rl, "rsi_high": rh, "z_low": zl, "z_high": zh, "adx_max": ax, "tp_mult": tpM, "sl_mult": slM})
                                    if best is None or cand > best:
                                        best = cand
        if best is None:
            continue
        score, wr, n, cfg = best
        out_cfg[sym] = cfg
        print(f"{sym.upper()}: WR={wr:.2%} n={n} -> {cfg}")
    if out_cfg:
        PER_COIN_CFG.write_text(json.dumps(out_cfg, indent=2))
        # clear cache for next run
        global _PER_COIN_CACHE
        _PER_COIN_CACHE = {}
        print(f"Saved per-coin config to {PER_COIN_CFG}")


if __name__ == "__main__":
    if os.getenv("TUNE_MR") == "1":
        tune_meanrev()
    else:
        main_loop()

