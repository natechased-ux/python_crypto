#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Alerts (ADX 25) with Telegram + Partial-TP-then-Trail Blueprint

New in this version
- Telegram notifications (env or flags)
- Partial TP then Trail blueprint in each alert payload:
  * TP1 = per-symbol TP (MFE p60) on 50% (configurable via --tp1_share)
  * On TP1 hit: move SL to breakeven
  * Trail remainder with N×ATR (default 1.0) behind price in trend direction

Environment (or pass flags):
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...

Examples:
  py live_alerts_telegram.py --symbols BTC-USD ETH-USD SOL-USD --interval 5 --mode strictlite
  py live_alerts_telegram.py --use_fvg --telegram --tp1_share 0.5 --trail_atr 1.0
"""
import argparse, json, os, sys, time, math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd, numpy as np

try:
    import requests
except Exception:
    print("Please install requests: pip install requests")
    raise

CB_BASE = "https://api.exchange.coinbase.com"

# -------------------- Indicators --------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
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

# -------------------- Data --------------------
def fetch_candles(product_id: str, start: pd.Timestamp, end: pd.Timestamp, granularity: int) -> pd.DataFrame:
    step = pd.Timedelta(seconds=granularity * 300)
    frames = []
    headers = {"User-Agent": "live-alerts/telegram/1.0"}
    t0 = start
    while t0 < end:
        t1 = min(t0 + step, end)
        url = f"{CB_BASE}/products/{product_id}/candles"
        params = {
            "granularity": granularity,
            "start": t0.isoformat().replace("+00:00","Z"),
            "end": t1.isoformat().replace("+00:00","Z"),
        }
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 429:
            time.sleep(1.0)
            resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json() or []
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
    o = df["open"].resample(tf).first()
    h = df["high"].resample(tf).max()
    l = df["low"].resample(tf).min()
    c = df["close"].resample(tf).last()
    v = df["volume"].resample(tf).sum()
    return pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":v}).dropna(how="any")

# -------------------- TP/SL Config --------------------
def load_config(path: str) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(path):
        print(f"[WARN] config not found at {path}; using GLOBAL defaults if present")
        return {}
    with open(path, "r") as f:
        return json.load(f)

def compute_levels(symbol: str, side: str, entry: float, cfg: Dict[str, Dict[str, float]],
                   atr_val: Optional[float]=None, tp_floor_atr: float=1.0, sl_floor_atr: float=0.8,
                   risk_cap_pct: float=0.02) -> Tuple[float, float]:
    params = cfg.get(symbol) or cfg.get("GLOBAL") or {"tp_pct": 2.0, "sl_pct": 2.0}
    tp_pct = params["tp_pct"] / 100.0
    sl_pct = params["sl_pct"] / 100.0

    if side == "long":
        tp = entry * (1 + tp_pct)
        sl = entry * (1 - sl_pct)
    else:
        tp = entry * (1 - tp_pct)
        sl = entry * (1 + sl_pct)

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

# -------------------- Signal Logic --------------------
@dataclass
class Params:
    adx_min: float = 25.0
    rsi_long_min: float = 55.0
    rsi_short_max: float = 45.0
    ema_len: int = 200
    atr_len: int = 14
    confirm_hours: int = 12
    mode: str = "strictlite"  # "medium" or "strictlite"
    use_fvg: bool = False

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
    ltf["stoch_long_ok"]  = (ltf["stoch_k"] > ltf["stoch_d"]) & (ltf["stoch_k"] < 50)
    ltf["stoch_short_ok"] = (ltf["stoch_k"] < ltf["stoch_d"]) & (ltf["stoch_k"] > 50)
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
        stoch_ok = window["stoch_long_ok"]
        breakout_ok = window["breakout_long"]
        pullback_ok = window["pullback_long"]
    else:
        stoch_ok = window["stoch_short_ok"]
        breakout_ok = window["breakout_short"]
        pullback_ok = window["pullback_short"]

    if p.mode == "strictlite":
        trig = stoch_ok & (breakout_ok | pullback_ok)
    else:
        trig = stoch_ok | breakout_ok | pullback_ok

    if not trig.any():
        return None
    trig_time = trig[trig].index[0]
    return {"side": side, "trigger_time": trig_time, "htf_time": last_htf_ts}

# -------------------- Telegram --------------------
def telegram_send(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print(f"[WARN] Telegram send failed: {r.status_code} {r.text[:120]}")
    except Exception as e:
        print(f"[WARN] Telegram exception: {e}")

# -------------------- Alert Engine --------------------
def format_alert(symbol: str, side: str, entry: float, tp1: float, sl: float, rr: float,
                 mode: str, adx_min: float, use_fvg: bool, trail_atr: float) -> str:
    return (f"{symbol} {side.upper()} | entry={entry:.6f} | TP1={tp1:.6f} | SL={sl:.6f} | "
            f"RR~{rr:.2f} | mode={mode} adx_min={adx_min} FVG={use_fvg} | trail={trail_atr}xATR after TP1")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=[ "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "DOGE-USD", "ADA-USD", "XLM-USD", "SUI-USD",
    "BCH-USD", "LINK-USD", "HBAR-USD", "LTC-USD", "AVAX-USD", "SHIB-USD", "UNI-USD", "DOT-USD",
    "CRO-USD",  "AAVE-USD", "NEAR-USD", "ETC-USD", "ONDO-USD",
    "APT-USD", "ICP-USD", "ALGO-USD", "ARB-USD", "VET-USD",
    "ATOM-USD", "POL-USD", "BONK-USD", "RENDER-USD",
    "FET-USD", "SEI-USD", "FIL-USD", "QNT-USD", "LSETH-USD", "INJ-USD",
    "CRV-USD", "STX-USD", "TIA-USD"])
    ap.add_argument("--interval", type=int, default=5, help="Polling interval in minutes")
    ap.add_argument("--lookback_days", type=int, default=30, help="History to pull on startup")
    ap.add_argument("--mode", choices=["medium","strictlite"], default="strictlite")
    ap.add_argument("--adx_min", type=float, default=25.0)
    ap.add_argument("--use_fvg", action="store_true", default=True)
    ap.add_argument("--out", type=str, default="alerts.csv")
    ap.add_argument("--config", type=str, default="tp_sl_config.json")
    # Telegram
    ap.add_argument("--telegram", action="store_true", default=True, help="Enable Telegram sends")
    ap.add_argument("--telegram_token", type=str, default="8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw")
    ap.add_argument("--telegram_chat_id", type=str, default="7967738614")
    # Partial + trail blueprint
    ap.add_argument("--tp1_share", type=float, default=0.5, help="Fraction to close at TP1 (0..1)")
    ap.add_argument("--trail_atr", type=float, default=1.0, help="ATR multiple for trailing stop after TP1")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Loaded TP/SL config for {len(cfg)} symbols from {args.config}")

    if args.telegram and not (args.telegram_token and args.telegram_chat_id):
        print("[WARN] --telegram set but TELEGRAM_BOT_TOKEN/CHAT_ID missing. Set env or pass flags.")

    # Prepare CSV
    if not os.path.exists(args.out):
        pd.DataFrame(columns=[
            "ts","symbol","side","entry","tp1","sl","mode","adx_min","use_fvg",
            "trigger_time","htf_time","tp1_share","trail_atr"
        ]).to_csv(args.out, index=False)

    print(f"Polling every {args.interval} min | mode={args.mode} | adx_min={args.adx_min} | use_fvg={args.use_fvg}")
    while True:
        loop_start = time.time()
        now = pd.Timestamp.utcnow().floor("T")
        start_hist = now - pd.Timedelta(days=args.lookback_days)
        end_hist = now

        for symbol in args.symbols:
            try:
                ohlc_1h = fetch_candles(symbol, start_hist, end_hist, granularity=3600)
                if len(ohlc_1h) < 400:
                    print(f"[WARN] {symbol} insufficient 1H candles ({len(ohlc_1h)})"); continue
                ohlc_6h = resample_ohlcv(ohlc_1h, "6h")
                if len(ohlc_6h) < 80:
                    print(f"[WARN] {symbol} insufficient 6H candles ({len(ohlc_6h)})"); continue

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

                # Per-symbol TP/SL baseline
                tp, sl = compute_levels(symbol, side, entry, cfg, atr_val=atr_val)

                # Partial TP blueprint
                tp1 = tp  # take args.tp1_share at TP1
                rr = (tp1 - entry) / (entry - sl) if side == "long" else (entry - tp1) / (sl - entry)

                # Compose line
                line = format_alert(symbol, side, entry, tp1, sl, rr, args.mode, args.adx_min, args.use_fvg, args.trail_atr)
                print(f"[{pd.Timestamp.utcnow().isoformat()}] {line}")

                # CSV row
                row = {
                    "ts": pd.Timestamp.utcnow().isoformat(),
                    "symbol": symbol, "side": side, "entry": entry, "tp1": tp1, "sl": sl,
                    "mode": args.mode, "adx_min": args.adx_min, "use_fvg": args.use_fvg,
                    "trigger_time": trig_time.isoformat(), "htf_time": sig["htf_time"].isoformat(),
                    "tp1_share": args.tp1_share, "trail_atr": args.trail_atr
                }
                pd.DataFrame([row]).to_csv(args.out, mode="a", header=False, index=False)

                # Telegram send
                if args.telegram and args.telegram_token and args.telegram_chat_id:
                    msg = (
                        f"{symbol} {side.upper()}"
                        f"\nentry: {entry:.6f}"
                        f"\nTP1 ({int(args.tp1_share*100)}%): {tp1:.6f}"
                        f"\nSL: {sl:.6f}"
                        f"\nTrail after TP1: {args.trail_atr}×ATR"
                        f"\nmode={args.mode} adx_min={args.adx_min} FVG={args.use_fvg}"
                        f"\ntrigger: {trig_time.isoformat()}"
                    )
                    telegram_send(args.telegram_token, args.telegram_chat_id, msg)

            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")

        elapsed = time.time() - loop_start
        time.sleep(max(5, args.interval*60 - elapsed))

if __name__ == "__main__":
    main()
