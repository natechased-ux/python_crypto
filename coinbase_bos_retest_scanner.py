import time, json, math, os, sys
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import numpy as np

# ───────────────────────── Config ───────────────────────── #
SYMBOLS = [
    "BTC-USD","ETH-USD","LINK-USD","SUI-USD","XRP-USD",
    # add more coinbase product_ids as needed
]

GRANULARITY_SEC = 3600  # 1h
CANDLES_LIMIT = 500

# Structure detection
PIVOT_LEN = 5                 # swing pivot length
RETEST_BARS = 12              # bars to wait for retest after BOS
RETEST_TOL_PCT = 0.25 / 100.0 # tolerance around BOS level

# Trend filter
USE_EMA50 = True
EMA_FAST = 50
EMA_SLOW = 200

# Stoch RSI
RSI_LEN = 14
K_LEN = 14
D_LEN = 3
STOCH_LONG_K_MAX = 40
STOCH_SHORT_K_MIN = 60

# Risk / targets
ATR_LEN = 14
TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.5
USE_SWING_SLTP = False        # if True: SL at recent swing, TP = 1.5R

# Cooldown & loop
COOLDOWN_MIN = 60             # min minutes between alerts per symbol
POLL_EVERY_SEC = 60           # scan cadence

# Telegram (non-async) — using your saved defaults
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID   = "7967738614"

STATE_FILE = "scanner_state.json"

# ───────────────────── Helpers & Indicators ───────────────────── #
def tg_send(msg: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Telegram error:", e, file=sys.stderr)

def coinbase_candles(product_id: str, granularity: int, limit: int) -> pd.DataFrame:
    """
    Returns DataFrame with columns: time, low, high, open, close, volume (ascending time)
    Endpoint returns in reverse-chronological order => we sort ascending.
    """
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {"granularity": granularity, "limit": limit}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("No candle data")
    df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
    df = df.sort_values("time").reset_index(drop=True)
    # Convert epoch to datetime (UTC)
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def stoch_rsi(close, length=14, k_len=14, d_len=3):
    _r = rsi(close, length)
    min_r = _r.rolling(k_len).min()
    max_r = _r.rolling(k_len).max()
    stoch = 100 * (_r - min_r) / (max_r - min_r)
    k = stoch.rolling(3).mean()
    d = k.rolling(d_len).mean()
    return k, d

def true_range(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df, length=14):
    tr = true_range(df)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def pivothigh(series_high, left=5, right=5):
    # True when current bar (i) is the max over [i-left, i+right]
    ph = (series_high.rolling(left+1, min_periods=left+1).max()
          .shift(0) == series_high) & \
         (series_high.shift(-np.arange(1, right+1)).T
          .le(series_high).all())
    return ph

def pivotlow(series_low, left=5, right=5):
    pl = (series_low.rolling(left+1, min_periods=left+1).min()
          .shift(0) == series_low) & \
         (series_low.shift(-np.arange(1, right+1)).T
          .ge(series_low).all())
    return pl

def last_value_before(idx_series: pd.Series, series: pd.Series):
    """Return last non-na value of series up to index i for each row."""
    return series.where(idx_series).ffill()

def dynamic_price_format(price: float) -> str:
    if price >= 100:
        return f"{price:.2f}"
    if price >= 1:
        return f"{price:.4f}"
    return f"{price:.6f}"

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_FILE)

# ───────────────────── Core Signal Logic ───────────────────── #
def scan_symbol(sym: str, state: dict):
    try:
        df = coinbase_candles(sym, GRANULARITY_SEC, CANDLES_LIMIT)
    except Exception as e:
        print(f"[{sym}] fetch error:", e, file=sys.stderr)
        return

    # Indicators
    df["ema_fast"] = ema(df["close"], EMA_FAST)
    df["ema_slow"] = ema(df["close"], EMA_SLOW)
    df["trend_long"]  = (df["close"] > df["ema_slow"]) & (True if not USE_EMA50 else df["close"] > df["ema_fast"])
    df["trend_short"] = (df["close"] < df["ema_slow"]) & (True if not USE_EMA50 else df["close"] < df["ema_fast"])

    # Swings
    ph = pivothigh(df["high"], PIVOT_LEN, PIVOT_LEN)
    pl = pivotlow(df["low"], PIVOT_LEN, PIVOT_LEN)
    df["swing_high"] = np.where(ph, df["high"], np.nan)
    df["swing_low"]  = np.where(pl, df["low"],  np.nan)
    df["last_swing_high"] = pd.Series(df["swing_high"]).ffill()
    df["last_swing_low"]  = pd.Series(df["swing_low"]).ffill()

    # BOS
    df["bos_up"]   = (df["close"].shift(1) <= df["last_swing_high"].shift(1)) & (df["close"] > df["last_swing_high"])
    df["bos_down"] = (df["close"].shift(1) >= df["last_swing_low"].shift(1))  & (df["close"] < df["last_swing_low"])
    # Record timing/level
    df["bos_level"] = np.where(df["bos_up"], df["last_swing_high"],
                        np.where(df["bos_down"], df["last_swing_low"], np.nan))
    df["bos_up_idx"]   = np.where(df["bos_up"], df.index, np.nan)
    df["bos_down_idx"] = np.where(df["bos_down"], df.index, np.nan)
    df["last_bos_idx"]   = pd.Series(np.where(df["bos_up"] | df["bos_down"], df.index, np.nan)).ffill()
    df["last_bos_is_up"] = pd.Series(np.where(df["bos_up"], True, np.where(df["bos_down"], False, np.nan))).ffill()
    df["last_bos_level"] = pd.Series(np.where(~np.isnan(df["bos_level"]), df["bos_level"], np.nan)).ffill()

    # Retest window
    tol = df["last_bos_level"] * RETEST_TOL_PCT
    in_window = (df.index - df["last_bos_idx"]) \
                .where(~df["last_bos_idx"].isna(), np.nan) \
                .between(1, RETEST_BARS)

    df["retest_long"]  = in_window & (df["last_bos_is_up"] == True)  & ( (df["low"]  - df["last_bos_level"]).abs() <= tol )
    df["retest_short"] = in_window & (df["last_bos_is_up"] == False) & ( (df["high"] - df["last_bos_level"]).abs() <= tol )

    # Stoch RSI
    k, d = stoch_rsi(df["close"], RSI_LEN, K_LEN, D_LEN)
    df["k"] = k
    df["d"] = d
    df["long_conf"]  = (df["k"] > df["d"]) & (df["k"] < STOCH_LONG_K_MAX)
    df["short_conf"] = (df["k"] < df["d"]) & (df["k"] > STOCH_SHORT_K_MIN)

    # ATR
    df["atr"] = atr(df, ATR_LEN)

    # Final signals
    df["long_setup"]  = df["trend_long"]  & df["retest_long"]  & df["long_conf"]
    df["short_setup"] = df["trend_short"] & df["retest_short"] & df["short_conf"]

    i = df.index[-1]
    row = df.loc[i]

    # Cooldown check
    now = datetime.now(timezone.utc)
    sym_state = state.get(sym, {})
    last_alert_iso = sym_state.get("last_alert")
    if last_alert_iso:
        last_alert = datetime.fromisoformat(last_alert_iso)
        if (now - last_alert) < timedelta(minutes=COOLDOWN_MIN):
            return  # still cooling down

    # Compose message if signal
    def fmt(x): return dynamic_price_format(float(x))

    if row["long_setup"]:
        entry = row["close"]
        if USE_SWING_SLTP:
            swing_sl = df["low"].rolling(PIVOT_LEN*2).min().iloc[-1]
            risk = entry - swing_sl
            sl = swing_sl
            tp = entry + risk * 1.5
        else:
            sl = entry - row["atr"] * SL_ATR_MULT
            tp = entry + row["atr"] * TP_ATR_MULT

        msg = (
            f"🟢 LONG Setup — {sym}\n"
            f"TF: 1H | {row['dt']}\n"
            f"Entry: {fmt(entry)}\n"
            f"SL: {fmt(sl)}  |  TP: {fmt(tp)}\n"
            f"BOS level: {fmt(row['last_bos_level'])}\n"
            f"Trend: above EMA{EMA_SLOW}" + (f" & EMA{EMA_FAST}" if USE_EMA50 else "") + "\n"
            f"StochRSI K={row['k']:.1f} D={row['d']:.1f} | ATR={fmt(row['atr'])}"
        )
        tg_send(msg)
        sym_state["last_alert"] = now.isoformat()
        state[sym] = sym_state

    elif row["short_setup"]:
        entry = row["close"]
        if USE_SWING_SLTP:
            swing_sl = df["high"].rolling(PIVOT_LEN*2).max().iloc[-1]
            risk = swing_sl - entry
            sl = swing_sl
            tp = entry - risk * 1.5
        else:
            sl = entry + row["atr"] * SL_ATR_MULT
            tp = entry - row["atr"] * TP_ATR_MULT

        msg = (
            f"🔴 SHORT Setup — {sym}\n"
            f"TF: 1H | {row['dt']}\n"
            f"Entry: {fmt(entry)}\n"
            f"SL: {fmt(sl)}  |  TP: {fmt(tp)}\n"
            f"BOS level: {fmt(row['last_bos_level'])}\n"
            f"Trend: below EMA{EMA_SLOW}" + (f" & EMA{EMA_FAST}" if USE_EMA50 else "") + "\n"
            f"StochRSI K={row['k']:.1f} D={row['d']:.1f} | ATR={fmt(row['atr'])}"
        )
        tg_send(msg)
        sym_state["last_alert"] = now.isoformat()
        state[sym] = sym_state

def main_loop():
    state = load_state()
    print("Starting Coinbase BOS/Retest scanner (1H)…")
    while True:
        for sym in SYMBOLS:
            scan_symbol(sym, state)
            time.sleep(0.3)  # light pacing between symbols
        save_state(state)
        time.sleep(POLL_EVERY_SEC)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nExiting.")
