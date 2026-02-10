#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Alerts — ADX 25 w/ Whale Flow + Telegram + Partial-TP→Trail (strict-lite by default)

What it does
- Pulls 1h Coinbase candles (public REST) and resamples to 6h
- 6h trend/structure: EMA200, MACD hist sign, ADX>=min, RSI filters
- 1h confirmation window (12h default): 
    * strictlite: StochRSI (relaxed) AND (breakout OR pullback)
    * medium    : StochRSI (relaxed) OR  (breakout OR pullback)
- Optional FVG confluence on 6h
- TP/SL from tp_sl_config.json (TP=MFE p60, SL=MAE p70) w/ ATR floors + risk cap
- WhaleWatcher (Coinbase websocket 'matches'): 
    * Detects recent large prints near price; can REQUIRE or only tag
- Telegram notifications
- Partial-TP then trail blueprint in the message (TP1 share then trail N×ATR)

USAGE (no CLI args needed once you fill creds below):
  py live_alerts_telegram_whales.py

Fill these before running:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""
import os, sys, time, math, json as _json, argparse, threading, collections
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
import pandas as pd, numpy as np

try:
    import requests
except Exception:
    print("Please install dependencies: pip install requests websocket-client pandas numpy")
    raise

try:
    import websocket  # websocket-client
except Exception:
    websocket = None

CB_BASE = "https://api.exchange.coinbase.com"

# ---------- HARD-CODED TELEGRAM CREDS (EDIT THESE) ----------
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID   = "7967738614"   # positive for DMs, negative for groups/channels

# ---------- Indicator helpers ----------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atr_w = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / (atr_w + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / (atr_w + 1e-12)
    dx = (100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12))
    return dx.ewm(alpha=1/length, adjust=False).mean()

def stoch_rsi(series: pd.Series, rsi_len=14, stoch_len=14, k=3, d=3) -> Tuple[pd.Series, pd.Series]:
    r = rsi(series, rsi_len)
    min_r = r.rolling(stoch_len).min()
    max_r = r.rolling(stoch_len).max()
    stoch = (r - min_r) / ((max_r - min_r) + 1e-12)
    k_line = stoch.rolling(k).mean() * 100.0
    d_line = k_line.rolling(d).mean()
    return k_line, d_line

def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    fvg_up = df["low"] > df["high"].shift(2)
    fvg_dn = df["high"] < df["low"].shift(2)
    return pd.DataFrame({"fvg_up": fvg_up.fillna(False), "fvg_dn": fvg_dn.fillna(False)}, index=df.index)

# ---------- Data ----------
def fetch_candles(product_id: str, start: pd.Timestamp, end: pd.Timestamp, granularity: int) -> pd.DataFrame:
    step = pd.Timedelta(seconds=granularity * 300)
    frames = []
    headers = {"User-Agent": "hc-live/1.0"}
    t0 = start
    while t0 < end:
        t1 = min(t0 + step, end)
        url = f"{CB_BASE}/products/{product_id}/candles"
        params = {
            "granularity": granularity,
            "start": t0.isoformat().replace("+00:00","Z"),
            "end": t1.isoformat().replace("+00:00","Z"),
        }
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 429:
            time.sleep(1.0)
            r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json() or []
        if data:
            arr = np.array(data, dtype=float)
            df = pd.DataFrame(arr, columns=["time","low","high","open","close","volume"])
            df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
            df = df.sort_values("time").set_index("time")[["open","high","low","close","volume"]]
            frames.append(df)
        time.sleep(0.15)
        t0 = t1
    if not frames:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    return pd.concat(frames).groupby(level=0).last().sort_index()

def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    # use lowercase 'h' to avoid FutureWarning
    o = df["open"].resample(tf).first()
    h = df["high"].resample(tf).max()
    l = df["low"].resample(tf).min()
    c = df["close"].resample(tf).last()
    v = df["volume"].resample(tf).sum()
    return pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":v}).dropna(how="any")

# ---------- TP/SL ----------
def load_config(path: str) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(path):
        print(f"[WARN] config not found at {path}; using GLOBAL defaults if present")
        return {}
    with open(path, "r") as f:
        return _json.load(f)

def compute_levels(symbol: str, side: str, entry: float, cfg: Dict[str, Dict[str, float]],
                   atr_val: Optional[float]=None, tp_floor_atr: float=1.0, sl_floor_atr: float=0.8,
                   risk_cap_pct: float=0.02) -> Tuple[float, float]:
    params = cfg.get(symbol) or cfg.get("GLOBAL") or {"tp_pct": 2.0, "sl_pct": 2.0}
    tp_pct = params["tp_pct"] / 100.0
    sl_pct = params["sl_pct"] / 100.0
    if side == "long":
        tp = entry * (1 + tp_pct); sl = entry * (1 - sl_pct)
    else:
        tp = entry * (1 - tp_pct); sl = entry * (1 + sl_pct)
    if atr_val is not None:
        if side == "long":
            tp = max(tp, entry + tp_floor_atr * atr_val)
            sl = min(sl, entry - sl_floor_atr * atr_val)
        else:
            tp = min(tp, entry - tp_floor_atr * atr_val)
            sl = max(sl, entry + sl_floor_atr * atr_val)
    if abs(entry - sl) / entry > risk_cap_pct:
        adj = risk_cap_pct * entry
        sl = entry - adj if side == "long" else entry + adj
    return tp, sl

# ---------- Signal Logic ----------
@dataclass
class Params:
    adx_min: float = 25.0
    rsi_long_min: float = 55.0
    rsi_short_max: float = 45.0
    ema_len: int = 200
    atr_len: int = 14
    confirm_hours: int = 12
    mode: str = "strictlite"  # medium|strictlite
    use_fvg: bool = True

def compute_htf(ohlc_6h: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = ohlc_6h.copy()
    df["ema200"] = ema(df["close"], p.ema_len)
    _, _, hist = macd(df["close"])
    df["macd_hist"] = hist
    df["adx"] = adx(df)
    df["rsi"] = rsi(df["close"])
    df["uptrend"] = (df["close"] > df["ema200"]) & (df["macd_hist"] > 0)
    df["downtrend"] = (df["close"] < df["ema200"]) & (df["macd_hist"] < 0)
    df["long_ok"] = df["uptrend"] & (df["adx"] >= p.adx_min) & (df["rsi"] >= p.rsi_long_min)
    df["short_ok"] = df["downtrend"] & (df["adx"] >= p.adx_min) & (df["rsi"] <= p.rsi_short_max)
    if p.use_fvg:
        fvg = detect_fvg(df[["open","high","low","close","volume"]])
        df["fvg_up"] = fvg["fvg_up"]; df["fvg_dn"] = fvg["fvg_dn"]
        df["long_ok"] = df["long_ok"] & df["fvg_up"]
        df["short_ok"] = df["short_ok"] & df["fvg_dn"]
    return df

def compute_ltf(ohlc_1h: pd.DataFrame) -> pd.DataFrame:
    ltf = ohlc_1h.copy()
    ltf["atr"] = atr(ltf, 14)
    k, d = stoch_rsi(ltf["close"])
    ltf["stoch_k"], ltf["stoch_d"] = k, d
    # relaxed thresholds
    ltf["stoch_long_ok"]  = (ltf["stoch_k"] > ltf["stoch_d"]) & (ltf["stoch_k"] < 50)
    ltf["stoch_short_ok"] = (ltf["stoch_k"] < ltf["stoch_d"]) & (ltf["stoch_k"] > 50)
    # secondary triggers
    ltf["ema20"] = ema(ltf["close"], 20)
    ltf["ema50"] = ema(ltf["close"], 50)
    ltf["prev_high20"] = ltf["high"].shift(1).rolling(20).max()
    ltf["prev_low20"]  = ltf["low"].shift(1).rolling(20).min()
    ltf["breakout_long"] = ltf["high"] > ltf["prev_high20"]
    ltf["breakout_short"] = ltf["low"] < ltf["prev_low20"]
    ltf["pullback_long"] = (ltf["low"] <= ltf["ema20"]) | (ltf["low"] <= ltf["ema50"])
    ltf["pullback_short"] = (ltf["high"] >= ltf["ema20"]) | (ltf["high"] >= ltf["ema50"])
    return ltf

def find_signal(htf: pd.DataFrame, ltf: pd.DataFrame, p: Params) -> Optional[Dict]:
    if len(htf) < 3 or len(ltf) < 10:
        return None
    last_htf_ts = htf.index[-1]
    row = htf.iloc[-1]
    side = "long" if row.get("long_ok", False) else ("short" if row.get("short_ok", False) else None)
    if side is None:
        return None
    start = last_htf_ts
    end = start + pd.Timedelta(hours=p.confirm_hours)
    window = ltf[(ltf.index > start) & (ltf.index <= end)].copy()
    if window.empty:
        return None
    if side == "long":
        stoch_ok = window["stoch_long_ok"]; breakout_ok = window["breakout_long"]; pullback_ok = window["pullback_long"]
    else:
        stoch_ok = window["stoch_short_ok"]; breakout_ok = window["breakout_short"]; pullback_ok = window["pullback_short"]
    trig = (stoch_ok & (breakout_ok | pullback_ok)) if p.mode == "strictlite" else (stoch_ok | breakout_ok | pullback_ok)
    if not trig.any():
        return None
    trig_time = trig[trig].index[0]
    return {"side": side, "trigger_time": trig_time, "htf_time": last_htf_ts}

# ---------- WhaleWatcher ----------
class WhaleWatcher:
    """Subscribe to Coinbase 'matches' and track recent large prints per symbol."""
    WS_URL = "wss://ws-feed.exchange.coinbase.com"
    def __init__(self, symbols, min_usd=100_000.0, near_price_pct=0.25, max_age_sec=600):
        self.symbols = symbols
        self.min_usd = float(min_usd)
        self.near_pct = float(near_price_pct) / 100.0
        self.max_age = int(max_age_sec)
        self._deques = {s: collections.deque(maxlen=5000) for s in symbols}
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if websocket is None:
            print("[WARN] websocket-client not installed; whale watcher disabled")
            return
        def run():
            ws = None
            while not self._stop.is_set():
                try:
                    ws = websocket.create_connection(self.WS_URL, timeout=30)
                    sub = {
                        "type": "subscribe",
                        "product_ids": self.symbols,
                        "channels": [{"name": "matches", "product_ids": self.symbols}]
                    }
                    ws.send(_json.dumps(sub))
                    while not self._stop.is_set():
                        msg = ws.recv()
                        if not msg: 
                            continue
                        data = _json.loads(msg)
                        if data.get("type") != "match":
                            continue
                        product = data.get("product_id")
                        price = float(data.get("price", 0))
                        size  = float(data.get("size", 0))
                        side  = data.get("side")  # "buy" or "sell"
                        ts    = data.get("time", "")
                        try:
                            t = datetime.fromisoformat(ts.replace("Z","+00:00"))
                        except Exception:
                            t = datetime.now(timezone.utc)
                        notional = price * size
                        if notional >= self.min_usd and product in self._deques:
                            self._deques[product].append({"ts": t, "price": price, "side": side, "usd": notional})
                except Exception as e:
                    try:
                        if ws: ws.close()
                    except Exception:
                        pass
                    time.sleep(1.0)
            try:
                if ws: ws.close()
            except Exception:
                pass
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        print(f"[WHALE] started: min_usd={self.min_usd}, near={self.near_pct*100:.2f}%, age≤{self.max_age}s")

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def side_aligned_flow(self, symbol: str, side: str, ref_price: float) -> bool:
        dq = self._deques.get(symbol)
        if not dq:
            return False
        now = datetime.now(timezone.utc)
        for p in reversed(dq):
            age = (now - p["ts"]).total_seconds()
            if age > self.max_age:
                break
            if abs(p["price"] - ref_price) / ref_price > self.near_pct:
                continue
            if side == "long" and p["side"] == "buy":
                return True
            if side == "short" and p["side"] == "sell":
                return True
        return False

# ---------- Telegram ----------
def telegram_send(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print(f"[WARN] Telegram send failed: {r.status_code} {r.text[:300]}")
        else:
            data = r.json()
            if not data.get("ok", False):
                print(f"[WARN] Telegram API ok=false: {data}")
    except Exception as e:
        print(f"[WARN] Telegram exception: {e}")

# ---------- Main ----------
def format_console(symbol: str, side: str, entry: float, tp1: float, sl: float, rr: float,
                   mode: str, adx_min: float, use_fvg: bool, trail_atr: float, whale_tag: str) -> str:
    return (f"{symbol} {side.upper()} | entry={entry:.6f} | TP1={tp1:.6f} | SL={sl:.6f} | RR~{rr:.2f} | "
            f"mode={mode} adx_min={adx_min} FVG={use_fvg} | trail={trail_atr}xATR after TP1{whale_tag}")

def main():
    ap = argparse.ArgumentParser()
    cooldown_minutes = 120  # change as needed
    last_alert_time = {}

    ap.add_argument("--symbols", nargs="+", default=["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "DOGE-USD", "ADA-USD", "XLM-USD", "SUI-USD",
    "BCH-USD", "LINK-USD", "HBAR-USD", "LTC-USD", "AVAX-USD", "SHIB-USD", "UNI-USD", "DOT-USD",
    "CRO-USD",  "AAVE-USD", "NEAR-USD", "ETC-USD", "ONDO-USD",
    "APT-USD", "ICP-USD", "ALGO-USD", "ARB-USD", "VET-USD",
    "ATOM-USD", "POL-USD", "BONK-USD", "RENDER-USD",
    "FET-USD", "SEI-USD", "FIL-USD", "QNT-USD", "LSETH-USD", "INJ-USD",
    "CRV-USD", "STX-USD", "TIA-USD"])
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--lookback_days", type=int, default=30)
    ap.add_argument("--mode", choices=["medium","strictlite"], default="strictlite")
    ap.add_argument("--adx_min", type=float, default=25.0)
    ap.add_argument("--use_fvg", action="store_true")
    ap.add_argument("--out", type=str, default="alerts.csv")
    ap.add_argument("--config", type=str, default="tp_sl_config.json")
    # Telegram hard-coded defaults: enabled by default
    ap.add_argument("--telegram", action="store_true", default=True)
    ap.add_argument("--telegram_token", type=str, default=TELEGRAM_BOT_TOKEN)
    ap.add_argument("--telegram_chat_id", type=str, default=TELEGRAM_CHAT_ID)
    ap.add_argument("--tp1_share", type=float, default=0.5)
    ap.add_argument("--trail_atr", type=float, default=1.0)
    # Whale
    ap.add_argument("--whale", action="store_true", default=True)
    ap.add_argument("--whale_require", action="store_true", default=False)
    ap.add_argument("--whale_min_usd", type=float, default=100000.0)
    ap.add_argument("--whale_near_price_pct", type=float, default=0.25)
    ap.add_argument("--whale_max_age_sec", type=int, default=600)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Prepare CSV
    if not os.path.exists(args.out):
        pd.DataFrame(columns=[
            "ts","symbol","side","entry","tp1","sl","mode","adx_min","use_fvg",
            "trigger_time","htf_time","tp1_share","trail_atr","whale_aligned"
        ]).to_csv(args.out, index=False)

    # Whale watcher
    whale = None
    if args.whale:
        whale = WhaleWatcher(
            symbols=args.symbols,
            min_usd=args.whale_min_usd,
            near_price_pct=args.whale_near_price_pct,
            max_age_sec=args.whale_max_age_sec
        )
        whale.start()

    # Startup Telegram ping
    if args.telegram and args.telegram_token and args.telegram_chat_id:
        telegram_send(args.telegram_token, args.telegram_chat_id, "✅ Live alerts bot started (with Whale filter).")

    print(f"Polling {args.interval}m | mode={args.mode} | adx_min={args.adx_min} | FVG={args.use_fvg} | Whale={'on' if whale else 'off'}")
    while True:
        loop_start = time.time()
        now = pd.Timestamp.utcnow().floor("T")
        start_hist = now - pd.Timedelta(days=args.lookback_days)
        end_hist = now

        for symbol in args.symbols:
            try:
                ohlc_1h = fetch_candles(symbol, start_hist, end_hist, granularity=3600)
                if len(ohlc_1h) < 400:
                    print(f"[WARN] {symbol} insufficient 1h candles ({len(ohlc_1h)})"); continue
                ohlc_6h = resample_ohlcv(ohlc_1h, "6h")
                if len(ohlc_6h) < 80:
                    print(f"[WARN] {symbol} insufficient 6h candles ({len(ohlc_6h)})"); continue

                p = Params(adx_min=args.adx_min, mode=args.mode, use_fvg=args.use_fvg)
                htf = compute_htf(ohlc_6h, p)
                ltf = compute_ltf(ohlc_1h)

                sig = find_signal(htf, ltf, p)
                if not sig:
                    continue

                trig_time = sig["trigger_time"]
                try:
                    pos = ltf.index.get_loc(trig_time)
                except KeyError:
                    continue
                if pos + 1 >= len(ltf):
                    continue

                entry_time = ltf.index[pos + 1]
                entry = float(ltf["open"].iloc[pos + 1])
                side = sig["side"]
                atr_val = float(ltf["atr"].iloc[pos])

                # Per-symbol cooldown check
                now_ts = pd.Timestamp.utcnow()
                if symbol in last_alert_time:
                    elapsed_min = (now_ts - last_alert_time[symbol]).total_seconds() / 60.0
                    if elapsed_min < cooldown_minutes:
                        continue


                # Whale confluence
                whale_aligned = False
                if whale:
                    whale_aligned = whale.side_aligned_flow(symbol, side, entry)
                    if args.whale_require and not whale_aligned:
                        # skip alert if requiring whale flow
                        continue

                # TP/SL and partial blueprint
                tp, sl = compute_levels(symbol, side, entry, cfg, atr_val=atr_val)
                tp1 = tp
                rr = (tp1 - entry) / (entry - sl) if side == "long" else (entry - tp1) / (sl - entry)

                whale_tag = " | WHALE✅" if whale_aligned else " | WHALE–"
                line = format_console(symbol, side, entry, tp1, sl, rr, args.mode, args.adx_min, args.use_fvg, args.trail_atr, whale_tag)
                print(f"[{pd.Timestamp.utcnow().isoformat()}] {line}")

                # CSV log
                row = {
                    "ts": pd.Timestamp.utcnow().isoformat(),
                    "symbol": symbol, "side": side, "entry": entry, "tp1": tp1, "sl": sl,
                    "mode": args.mode, "adx_min": args.adx_min, "use_fvg": args.use_fvg,
                    "trigger_time": trig_time.isoformat(), "htf_time": sig["htf_time"].isoformat(),
                    "tp1_share": args.tp1_share, "trail_atr": args.trail_atr, "whale_aligned": whale_aligned
                }
                pd.DataFrame([row]).to_csv(args.out, mode="a", header=False, index=False)

                # Telegram push
                if args.telegram and args.telegram_token and args.telegram_chat_id:
                    msg = (
                        f"{symbol} {side.upper()}{whale_tag}"
                        f"\nentry: {entry:.6f}"
                        f"\nTP1 ({int(args.tp1_share*100)}%): {tp1:.6f}"
                        f"\nSL: {sl:.6f}"
                        f"\nTrail after TP1: {args.trail_atr}×ATR"
                        f"\nmode={args.mode} adx_min={args.adx_min} FVG={args.use_fvg}"
                        f"\ntrigger: {trig_time.isoformat()}"
                    )
                    telegram_send(args.telegram_token, args.telegram_chat_id, msg)
                    last_alert_time[symbol] = now_ts


            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")

        elapsed = time.time() - loop_start
        time.sleep(max(5, args.interval*60 - elapsed))

if __name__ == "__main__":
    main()
