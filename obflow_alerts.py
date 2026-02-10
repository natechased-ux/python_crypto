#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obflow_alerts_rules_hotreload.py
--------------------------------
Per-symbol tightened-rule Telegram alerts with timed hot-reload.

- Loads setups from JSON (Windows path preferred, env override allowed)
- Computes features (flows/ratios, proximity flags, Stoch RSI) like obflow_tracker
- Evaluates LONG/SHORT rules per symbol
- Sends Telegram alerts with setup-specific TP/SL
- Cooldown per symbol/side
"""

import os, time, json, threading, math, collections
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import numpy as np
import requests
import urllib.parse as _url

# --------------------- Config ---------------------
KRAKEN_BASE = "https://api.kraken.com"

DEFAULT_COINS = [
    "XBT/USD","ETH/USD","SOL/USD","XRP/USD","MATIC/USD","ADA/USD","AVAX/USD","DOGE/USD","DOT/USD",
    "LINK/USD","ATOM/USD","NEAR/USD","ARB/USD","OP/USD","INJ/USD","AAVE/USD","LTC/USD","BCH/USD",
    "ETC/USD","ALGO/USD","FIL/USD","ICP/USD","RNDR/USD","STX/USD","GRT/USD","SEI/USD","HBAR/USD",
    "JUP/USD","TIA/USD","UNI/USD","MKR/USD","FET/USD","CRV/USD","SAND/USD","MANA/USD","AXS/USD",
    "PEPE/USD","XLM/USD"
]
COINS = [s.strip() for s in os.getenv("COINS_CSV", ",".join(DEFAULT_COINS)).split(",") if s.strip()]

TS_ENV = os.getenv("TRACK_SYMBOLS", "").strip()
TRACK_SYMBOLS = [s.strip() for s in TS_ENV.split(",") if s.strip()] if TS_ENV else COINS.copy()

PING_EVERY_SEC = int(os.getenv("PING_EVERY_SEC", "300"))  # 5 minutes
NEAR_PCT = float(os.getenv("NEAR_PCT", "0.0025"))  # 0.25%

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "7967738614")
ALERT_COOLDOWN_MIN = int(os.getenv("ALERT_COOLDOWN_MIN", "90"))

# --------- Setups JSON path (Windows preferred) ---------
SETUPS_JSON_PATH = os.getenv("SETUPS_JSON_PATH", r"C:\Users\natec\price_excluded_setups_tightened_subsetTP.json")
SETUPS_FALLBACK  = os.getenv("SETUPS_JSON_FALLBACK", "/mnt/data/price_excluded_setups_tightened_subsetTP.json")

SETUPS_REFRESH_SEC = int(os.getenv("SETUPS_REFRESH_SEC", "300"))  # hot-reload every 5 minutes

# ------------------ Helpers reused ------------------
def _kr_pair_for_rest(pair: str) -> str:
    return pair.replace("/", "")

def fetch_candles(product_id: str, granularity: int) -> pd.DataFrame:
    """
    Kraken OHLC: /0/public/OHLC?pair=XBTUSD&interval=60
    Normalize to columns: ["time","low","high","open","close","volume"], ascending by time.
    """
    itv_map = {60: 1, 300: 5, 900: 15, 1800: 30, 3600: 60, 21600: 240, 86400: 1440}
    interval = itv_map.get(granularity, 60)

    pair_q = _kr_pair_for_rest(product_id)
    url = f"{KRAKEN_BASE}/0/public/OHLC?pair={_url.quote(pair_q)}&interval={interval}"
    r = requests.get(url, timeout=15); r.raise_for_status()
    res = r.json().get("result", {})
    ohlc = next((v for k, v in res.items() if k != "last"), None)
    if not ohlc:
        return pd.DataFrame(columns=["time","low","high","open","close","volume"])
    df = pd.DataFrame(ohlc)
    if df.shape[1] == 8:
        df.columns = ["time","open","high","low","close","vwap","volume","count"]
    elif df.shape[1] == 7:
        df.columns = ["time","open","high","low","close","volume","count"]
        df["vwap"] = pd.NA
    else:
        while df.shape[1] < 8:
            df[df.shape[1]] = pd.NA
        df = df.iloc[:, :8]
        df.columns = ["time","open","high","low","close","vwap","volume","count"]
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["time","low","high","open","close","volume"]].sort_values("time").reset_index(drop=True)

def fetch_trades(product_id: str, limit: int = 1000) -> pd.DataFrame:
    """
    Kraken Trades: /0/public/Trades?pair=XBTUSD
    Return normalized columns: ["price","volume","time","side","ordertype","misc","notional"]
    side: 'buy' or 'sell', time is UTC ts.
    """
    pair_q = _kr_pair_for_rest(product_id)
    url = f"{KRAKEN_BASE}/0/public/Trades?pair={_url.quote(pair_q)}"
    r = requests.get(url, timeout=12); r.raise_for_status()
    res = r.json().get("result", {})
    arr = next((v for k, v in res.items() if k != "last"), None)
    if not arr:
        return pd.DataFrame(columns=["price","volume","time","side","ordertype","misc","notional"])
    # Trim if too long
    if len(arr) > limit:
        arr = arr[-limit:]
    first = arr[0] if len(arr) else []
    n = len(first)
    if n >= 7:
        cols = ["price","volume","time","side","ordertype","misc","trade_id"]
        df = pd.DataFrame(arr, columns=cols[:n])
    elif n == 6:
        cols = ["price","volume","time","side","ordertype","misc"]
        df = pd.DataFrame(arr, columns=cols)
    else:
        df = pd.DataFrame(arr)
        while df.shape[1] < 6:
            df[df.shape[1]] = pd.NA
        df = df.iloc[:, :6]
        df.columns = ["price","volume","time","side","ordertype","misc"]
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["side"] = df["side"].map({"b":"buy","s":"sell"}).fillna("buy")
    df["notional"] = df["price"] * df["volume"]
    return df.sort_values("time").reset_index(drop=True)

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi_wilder(series: pd.Series, length: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    delta = s.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0.0, pd.NA)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(method="bfill").fillna(50.0)

def stoch_rsi_kd(series_close: pd.Series, rsi_len: int = 14, stoch_len: int = 14, k_len: int = 3, d_len: int = 3) -> Tuple[float,float]:
    rsi = rsi_wilder(series_close, rsi_len)
    low = rsi.rolling(stoch_len, min_periods=max(2, stoch_len//2)).min()
    high = rsi.rolling(stoch_len, min_periods=max(2, stoch_len//2)).max()
    denom = (high - low)
    stoch = ((rsi - low) / denom.where(denom != 0, pd.NA)).fillna(0.5)
    k = stoch.rolling(k_len, min_periods=1).mean() * 100.0
    d = k.rolling(d_len, min_periods=1).mean()
    return float(k.iloc[-1]), float(d.iloc[-1])

def _fib_golden_zone(lo: float, hi: float, tol: float = NEAR_PCT) -> Tuple[float,float]:
    r618 = lo + 0.618 * (hi - lo)
    r660 = lo + 0.660 * (hi - lo)
    zmin, zmax = min(r618, r660), max(r618, r660)
    pad = ((zmin+zmax)/2) * tol
    return zmin - pad, zmax + pad

def _price_near_level(price: float, level: float, pct: float = NEAR_PCT) -> int:
    if not (np.isfinite(price) and np.isfinite(level)):
        return 0
    return int(abs(price - level) / price <= pct)

def fetch_depth_top(pair: str) -> Tuple[Optional[float], Optional[float]]:
    bid_top = ask_top = None
    try:
        pair_q = _kr_pair_for_rest(pair)
        url = f"{KRAKEN_BASE}/0/public/Depth?pair={_url.quote(pair_q)}&count=1"
        r = requests.get(url, timeout=10); r.raise_for_status()
        res = r.json().get("result", {})
        dp = next((v for k, v in res.items()), None) or {}
        bids = dp.get("bids", [])
        asks = dp.get("asks", [])
        if bids:
            bid_top = float(bids[0][0])
        if asks:
            ask_top = float(asks[0][0])
    except Exception:
        pass
    return bid_top, ask_top

def _level_flags_and_stoch(sym: str, price: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    df1h = fetch_candles(sym, 3600)
    df1d = fetch_candles(sym, 86400)
    df15 = fetch_candles(sym, 900)

    for per in (10,20,50,200):
        key = f"near_ema{per}_1h"
        if len(df1h) >= per:
            val = float(ema(df1h["close"], per).iloc[-1])
            out[key] = _price_near_level(price, val)
        else:
            out[key] = 0

    for per in (20,50,100,200):
        key = f"near_ema{per}_1d"
        if len(df1d) >= per:
            val = float(ema(df1d["close"], per).iloc[-1])
            out[key] = _price_near_level(price, val)
        else:
            out[key] = 0

    bid_top, ask_top = fetch_depth_top(sym)
    out["near_top_bid"] = _price_near_level(price, bid_top) if bid_top is not None else 0
    out["near_top_ask"] = _price_near_level(price, ask_top) if ask_top is not None else 0

    # swings (simple, same as tracker)
    def _find_last_swing(df: pd.DataFrame, left=3, right=3) -> Tuple[Optional[float], Optional[float]]:
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)
        if len(highs) < left+right+1:
            return None, None
        idx_end = len(highs) - right - 1
        hi_val = lo_val = None
        for i in range(idx_end - 1, left - 1, -1):
            window_h = highs.iloc[i-left:i+right+1]
            if highs.iloc[i] == window_h.max():
                hi_val = float(highs.iloc[i]); break
        for i in range(idx_end - 1, left - 1, -1):
            window_l = lows.iloc[i-left:i+right+1]
            if lows.iloc[i] == window_l.min():
                lo_val = float(lows.iloc[i]); break
        return hi_val, lo_val

    hi_s, lo_s = (None, None)
    if not df1h.empty:
        hi_s, lo_s = _find_last_swing(df1h)
    out["near_swing_high_1h"] = _price_near_level(price, hi_s) if hi_s is not None else 0
    out["near_swing_low_1h"]  = _price_near_level(price, lo_s) if lo_s is not None else 0

    def _near_fib_flags(df_lohi: pd.DataFrame, lo_bars: int, hi_bars: int, key: str):
        if df_lohi.empty:
            out[f"near_{key}_min"] = 0
            out[f"near_{key}_max"] = 0
            out[f"in_{key}_zone"] = 0
            return
        lo = float(df_lohi["low"].tail(lo_bars).min())
        hi = float(df_lohi["high"].tail(hi_bars).max())
        zmin, zmax = _fib_golden_zone(lo, hi, tol=NEAR_PCT)
        out[f"near_{key}_min"] = _price_near_level(price, zmin)
        out[f"near_{key}_max"] = _price_near_level(price, zmax)
        out[f"in_{key}_zone"] = int(zmin <= price <= zmax)

    if not df1h.empty:
        _near_fib_flags(df1h, 2*24, 2*24, "fib_tiny")
        _near_fib_flags(df1h, 7*24, 7*24, "fib_short")
    df6h = fetch_candles(sym, 21600)
    if not df6h.empty:
        _near_fib_flags(df6h, 14*4, 14*4, "fib_medium")
        _near_fib_flags(df6h, 30*4, 30*4, "fib_long")

    # Stoch RSI
    if len(df1h) >= 30:
        k1h, d1h = stoch_rsi_kd(df1h["close"])
    else:
        k1h = d1h = float("nan")
    if len(df15) >= 30:
        k15, d15 = stoch_rsi_kd(df15["close"])
    else:
        k15 = d15 = float("nan")
    out["stoch_k_1h"], out["stoch_d_1h"] = k1h, d1h
    out["stoch_k_15m"], out["stoch_d_15m"] = k15, d15
    return out

def _calc_flows_for_windows(sym: str, windows_s: Dict[str,int]) -> Dict[str, float]:
    tdf = fetch_trades(sym, limit=2000)
    out = {k: 0.0 for k in windows_s}
    if tdf.empty or "time" not in tdf:
        return out
    now = pd.Timestamp.utcnow()
    for name, sec in windows_s.items():
        cutoff = now - pd.Timedelta(seconds=sec)
        win = tdf[tdf["time"] >= cutoff]
        buys = float(win.loc[win["side"]=="buy","notional"].sum())
        sells = float(win.loc[win["side"]=="sell","notional"].sum())
        out[name] = buys - sells
    return out

def _cumulative_flows(windows_s: Dict[str,int]) -> Dict[str, float]:
    sums = {k: 0.0 for k in windows_s}
    now = pd.Timestamp.utcnow()
    for coin in COINS:
        tdf = fetch_trades(coin, limit=2000)
        if tdf.empty:
            continue
        for name, sec in windows_s.items():
            cutoff = now - pd.Timedelta(seconds=sec)
            win = tdf[tdf["time"] >= cutoff]
            buys = float(win.loc[win["side"]=="buy","notional"].sum())
            sells = float(win.loc[win["side"]=="sell","notional"].sum())
            sums[name] += (buys - sells)
    return sums

def _last_price(sym: str) -> Optional[float]:
    for g in (60, 300, 900):
        df = fetch_candles(sym, g)
        if not df.empty:
            return float(df["close"].iloc[-1])
    return None

# -------------- Setups: load + hot-reload --------------
_SYMBOL_SETUPS: Dict[str, Dict[str, Any]] = {}
_SETUPS_LOCK = threading.Lock()
_SETUPS_MTIME: Optional[float] = None

def _resolve_setups_path() -> Optional[str]:
    # Prefer explicit env, then Windows path, then fallback
    candidates = []
    if os.getenv("SETUPS_JSON_PATH"):
        candidates.append(os.getenv("SETUPS_JSON_PATH"))
    candidates.append(SETUPS_JSON_PATH)
    candidates.append(SETUPS_FALLBACK)
    for p in candidates:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            continue
    return None

def _load_symbol_setups() -> None:
    global _SYMBOL_SETUPS
    path = _resolve_setups_path()
    if not path:
        print("[setups] no setups JSON found")
        _SYMBOL_SETUPS = {}
        return
    try:
        with open(path, "r") as f:
            data = json.load(f)
        _SYMBOL_SETUPS = {rec["symbol"]: {"LONG": rec["LONG"], "SHORT": rec["SHORT"]} for rec in data}
        print(f"[setups] loaded {len(_SYMBOL_SETUPS)} symbols from {path}")
    except Exception as e:
        print("[setups] load failed:", e)
        _SYMBOL_SETUPS = {}

def _reload_symbol_setups_if_needed(force: bool = False) -> None:
    global _SETUPS_MTIME
    path = _resolve_setups_path()
    if not path:
        return
    try:
        mtime = os.path.getmtime(path)
        if force or (_SETUPS_MTIME is None) or (mtime != _SETUPS_MTIME):
            _SETUPS_MTIME = mtime
            _load_symbol_setups()
    except Exception as e:
        print("[setups] reload check failed:", e)

def _setups_hot_reload_loop() -> None:
    while True:
        _reload_symbol_setups_if_needed(force=False)
        time.sleep(SETUPS_REFRESH_SEC)

# -------------- Rule evaluation --------------
import re as _re_gate
_COND_RE = _re_gate.compile(r"(.+?)\s*(<=|>)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def _eval_rule_text(rule_text: str, feature_lookup: Dict[str, float]) -> bool:
    """Evaluate 'feat <= thr AND feat > thr ...' against feature values."""
    if not rule_text or rule_text.strip().lower().startswith("no strong"):
        return False
    parts = [c.strip() for c in rule_text.split("AND") if c.strip()]
    if not parts:
        return False
    for cond in parts:
        m = _COND_RE.match(cond)
        if not m:
            return False
        feat, op, val = m.groups()
        feat = feat.strip()
        if feat not in feature_lookup:
            return False
        try:
            x = float(feature_lookup.get(feat))
            thr = float(val)
        except Exception:
            return False
        if op == "<=" and not (x <= thr): return False
        if op == ">"  and not (x >  thr): return False
    return True

# -------------- Alert plumbing --------------

# -------------- Formatting helpers --------------
def _price_decimals(price: float) -> int:
    if price is None or not isinstance(price, (int, float)):
        return 6
    p = float(price)
    if p >= 1000: return 2
    if p >= 100:  return 3
    if p >= 1:    return 4
    if p >= 0.1:  return 5
    if p >= 0.01: return 6
    if p >= 0.001:return 7
    return 8  # for very small prices like PEPE

def _fmt_price(x: float) -> str:
    d = _price_decimals(x)
    return f"{x:.{d}f}"

_LAST_ALERT_TS: Dict[Tuple[str,str], float] = {}

def _cooldown_ok(symbol: str, side: str) -> bool:
    key = (symbol, side); now = time.time(); last = _LAST_ALERT_TS.get(key, 0.0)
    if now - last >= ALERT_COOLDOWN_MIN * 60:
        _LAST_ALERT_TS[key] = now; return True
    return False

def _send_telegram(text: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("[alert] telegram send failed:", e)

def _maybe_alert(symbol: str, side: str, entry_price: float, tp_pct: float, sl_pct: float, p_win: float = None, ev: float = None) -> None:
    if not _cooldown_ok(symbol, side):
        return
    if not (isinstance(entry_price, (int,float)) and isinstance(tp_pct, (int,float)) and isinstance(sl_pct, (int,float))):
        return
    if entry_price <= 0: return
    if side == "long":
        tp_price = entry_price * (1.0 + tp_pct/100.0)
        sl_price = entry_price * (1.0 - sl_pct/100.0)
    else:
        tp_price = entry_price * (1.0 - tp_pct/100.0)
        sl_price = entry_price * (1.0 + sl_pct/100.0)

    now_pt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    p_txt = f" | p_win={p_win:.1%}" if isinstance(p_win, (int,float)) else ""
    ev_txt = f" | EV={ev:.2f}R" if isinstance(ev, (int,float)) else ""
    msg = (
        f"{symbol} | {side.upper()} | "
        f"entry={_fmt_price(entry_price)} | "
        f"TP={_fmt_price(tp_price)} | "
        f"SL={_fmt_price(sl_price)} | "
        f"TP%={tp_pct:.2f} SL%={sl_pct:.2f}"
        f"{p_txt}{ev_txt} | {now_pt}"
    )
    print("[ALERT]", msg)
    _send_telegram(msg)

# -------------- Main tick --------------
def _one_tick() -> None:
    windows_s = {
        "lookback_5m": 5*60,
        "lookback_15m": 15*60,
        "lookback_30min": 30*60,
        "lookback_1hr": 60*60,
        "lookback_4hr": 4*60*60,
    }
    cum = _cumulative_flows(windows_s)

    for sym in TRACK_SYMBOLS:
        try:
            price = _last_price(sym)
            if price is None:
                print(f"[skip] no price for {sym}")
                continue
            flows = _calc_flows_for_windows(sym, windows_s)
            ratios = {}
            for k in windows_s:
                denom = cum.get(k, 0.0)
                ratios[k + "_ratio"] = (flows.get(k, 0.0) / denom) if abs(denom) > 1e-9 else np.nan

            levels = _level_flags_and_stoch(sym, price)

            row = {
                "symbol": sym,
                "price": round(float(price), 8),
                # flows
                "lookback_5m": round(float(flows["lookback_5m"]), 2),
                "lookback_15m": round(float(flows["lookback_15m"]), 2),
                "lookback_30min": round(float(flows["lookback_30min"]), 2),
                "lookback_1hr": round(float(flows["lookback_1hr"]), 2),
                "lookback_4hr": round(float(flows["lookback_4hr"]), 2),
                "lookback_5m_cum": round(float(cum["lookback_5m"]), 2),
                "lookback_15m_cum": round(float(cum["lookback_15m"]), 2),
                "lookback_30min_cum": round(float(cum["lookback_30min"]), 2),
                "lookback_1hr_cum": round(float(cum["lookback_1hr"]), 2),
                "lookback_4hr_cum": round(float(cum["lookback_4hr"]), 2),
                "lookback_5m_ratio": ratios["lookback_5m_ratio"],
                "lookback_15m_ratio": ratios["lookback_15m_ratio"],
                "lookback_30min_ratio": ratios["lookback_30min_ratio"],
                "lookback_1hr_ratio": ratios["lookback_1hr_ratio"],
                "lookback_4hr_ratio": ratios["lookback_4hr_ratio"],
                # levels & stoch
                **levels
            }

            # ---- Evaluate per-symbol tightened rules ----
            setups = _SYMBOL_SETUPS.get(sym, {})
            long_setup  = setups.get("LONG")
            short_setup = setups.get("SHORT")
            feature_lookup = dict(row)

            candidates = []
            if long_setup:
                long_rule = str(long_setup.get("entry_conditions", "")).strip()
                if _eval_rule_text(long_rule, feature_lookup):
                    candidates.append({
                        "side": "long",
                        "tp_pct": float(long_setup.get("tp_pct", 1.5)),
                        "sl_pct": float(long_setup.get("sl_pct", 1.0)),
                        "ev":     float(long_setup.get("EV_R_units", 0.0)),
                        "p_win":  float(long_setup.get("expected_win_rate", 0.0))
                    })
            if short_setup:
                short_rule = str(short_setup.get("entry_conditions", "")).strip()
                if _eval_rule_text(short_rule, feature_lookup):
                    candidates.append({
                        "side": "short",
                        "tp_pct": float(short_setup.get("tp_pct", 1.5)),
                        "sl_pct": float(short_setup.get("sl_pct", 1.0)),
                        "ev":     float(short_setup.get("EV_R_units", 0.0)),
                        "p_win":  float(short_setup.get("expected_win_rate", 0.0))
                    })

            if candidates:
                candidates.sort(key=lambda c: (c["ev"], abs(c["ev"])), reverse=True)
                pick = candidates[0]
                _maybe_alert(sym, pick["side"], price, pick["tp_pct"], pick["sl_pct"], pick.get("p_win"), pick.get("ev"))

        except Exception as e:
            print("[tick] error for", sym, ":", e)

def main() -> None:
    _load_symbol_setups()
    _reload_symbol_setups_if_needed(force=True)
    threading.Thread(target=_setups_hot_reload_loop, daemon=True).start()
    print(f"[alerts] running. Tracking {len(TRACK_SYMBOLS)} symbols:", ", ".join(TRACK_SYMBOLS))
    while True:
        t0 = time.time()
        try:
            _one_tick()
        except Exception as e:
            print("[main] tick error:", e)
        dt = time.time() - t0
        if dt < PING_EVERY_SEC:
            time.sleep(PING_EVERY_SEC - dt)

if __name__ == "__main__":
    main()
