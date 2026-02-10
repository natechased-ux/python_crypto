#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fib16_data_driven.py — Order-Flow + Level Touch Alerts
(Data-Driven Gates + TP/SL + Time Rules + Autotuner Hot-Apply)
----------------------------------------------------------------
Scans Kraken spot pairs, detects entries when price is at key levels AND order-flow
confluence is strong across multiple lookbacks. Logs trades, tracks MAE/MFE for 2h, and
sends Telegram alerts (non-async) with entry & management suggestions.

Improvements baked in from meta_json analysis:
- Multi-lookback flow (60/300/900/1800/3600); use the strongest (best_lb) for side
  and require long-window confluence.
- Data-driven gates: long-confluence, overshoot (|Δ|/thr), distance-to-level (ATR units).
- Fast TP1 + protective SL sized in ATR; time-based nudges at 15/30/60m.
- Clean CSV with quoted JSON; full meta_json at entry (flow_breakdown, confluence counts,
  delta_strength, distances, score, a_tier).
- **Autotuner hot-apply:** reads tuning_rules.json (output of autotune_from_logs.py) and
  applies per-coin or global gates and TP/SL without restart.

Run:
  pip install pandas requests python-dateutil numpy
  export FLOW_THRESHOLDS_PATH=thresholds_multi_lookback.json
  # optional: tuning rules (if present, will be hot-applied)
  export TUNING_RULES_PATH=tuning_rules.json
  export TELEGRAM_BOT_TOKEN=xxxxxxxx:yyyyyyyyyyyyyyyyyyyyyy
  export TELEGRAM_CHAT_ID=123456789
  python fib16_data_driven.py
"""

from __future__ import annotations
import os, time, json, math, threading, csv, uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np

# ===================== Config =====================
SYMBOLS = [
    "XBT/USD","ETH/USD","SOL/USD","XRP/USD","MATIC/USD","ADA/USD","AVAX/USD","DOGE/USD","DOT/USD",
    "LINK/USD","ATOM/USD","NEAR/USD","ARB/USD","OP/USD","INJ/USD","AAVE/USD","LTC/USD","BCH/USD",
    "ETC/USD","ALGO/USD","FIL/USD","ICP/USD","RNDR/USD","STX/USD","GRT/USD","SEI/USD","SUI/USD",
    "APT/USD","TIA/USD","UNI/USD","MKR/USD","FET/USD","CRV/USD","SAND/USD","MANA/USD","AXS/USD",
    "PEPE/USD","XLM/USD"
]

LOOKBACKS = [60, 300, 900, 1800, 3600]
LEVEL_TOL_PCT = 0.005       # ±0.5% band around level
EMA_BAND_PCT  = 0.005       # ±0.5% band around EMA value
PROFILE_LOOKBACK_H = 6       # trades profile horizon for VPOC/HVNs
PROFILE_BIN_PCT    = 0.001   # ~0.1% bin width for VP
TOP_HVNS           = 3
ATR_LEN = 14

TRACK_WINDOW_MIN = 120
SAMPLE_EVERY_MIN  = 5
COOLDOWN_MIN      = 30

# Data-driven gates (defaults; tuner can override per coin)
GATE_LONG_CONF_MIN   = int(os.getenv("GATE_LONG_CONF_MIN", "2"))
GATE_OVERSHOOT_MIN   = float(os.getenv("GATE_OVERSHOOT_MIN", "1.5"))
DIST_ATR_MIN         = float(os.getenv("DIST_ATR_MIN", "0.10"))
DIST_ATR_MAX         = float(os.getenv("DIST_ATR_MAX", "0.50"))
TP1_ATR              = float(os.getenv("TP1_ATR", "0.12"))
TP2_ATR              = float(os.getenv("TP2_ATR", "0.30"))
SL_ATR               = float(os.getenv("SL_ATR",  "0.35"))
SCORE_CAP            = float(os.getenv("SCORE_CAP", "10.0"))

CSV_PATH   = os.getenv("FLOW_LEVELS_LOG", "live_flow_levels_log.csv")
THR_PATH   = os.getenv("FLOW_THRESHOLDS_PATH", "thresholds_multi_lookback.json")
TUNING_PATH= os.getenv("TUNING_RULES_PATH", "tuning_rules.json")
BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw")
CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "7967738614")

KRAKEN_BASE = "https://api.kraken.com"

# ===================== Telegram =====================

def tg_send(text: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print("[tg]", e)

# ===================== Utils / Fetch =====================

def fmt_px(x: float) -> str:
    if x >= 100: return f"{x:.2f}"
    if x >= 10:  return f"{x:.3f}"
    if x >= 1:   return f"{x:.4f}"
    if x >= 0.1: return f"{x:.5f}"
    return f"{x:.9f}"


def _kr_pair_for_rest(pair: str) -> str:
    return pair.replace("/", "")


def fetch_candles(symbol: str, granularity_s: int) -> pd.DataFrame:
    itv_map = {60:1, 300:5, 900:15, 1800:30, 3600:60, 21600:240, 86400:1440}
    interval = itv_map.get(granularity_s, 60)
    pair_q = _kr_pair_for_rest(symbol)
    url = f"{KRAKEN_BASE}/0/public/OHLC?pair={pair_q}&interval={interval}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    res = r.json().get("result", {})
    arr = next((v for k, v in res.items() if k != "last"), [])
    if not arr:
        return pd.DataFrame(columns=["time","low","high","open","close","volume"])
    df = pd.DataFrame(arr)
    if df.shape[1] == 8:
        df.columns = ["time","open","high","low","close","vwap","volume","count"]
    elif df.shape[1] == 7:
        df.columns = ["time","open","high","low","close","volume","count"]
        df["vwap"] = np.nan
    else:
        while df.shape[1] < 8:
            df[df.shape[1]] = np.nan
        df = df.iloc[:, :8]
        df.columns = ["time","open","high","low","close","vwap","volume","count"]
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("time").reset_index(drop=True)[["time","low","high","open","close","volume"]]


def fetch_trades(symbol: str) -> pd.DataFrame:
    pair_q = _kr_pair_for_rest(symbol)
    url = f"{KRAKEN_BASE}/0/public/Trades?pair={pair_q}"
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    res = r.json().get("result", {})
    arr = next((v for k, v in res.items() if k != "last"), [])
    if not arr:
        return pd.DataFrame(columns=["time","price","volume","side","notional"]) 
    first = arr[0]
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
            df[df.shape[1]] = np.nan
        df = df.iloc[:, :6]
        df.columns = ["price","volume","time","side","ordertype","misc"]
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["side"] = df["side"].map({"b":"buy","s":"sell"}).fillna("buy")
    df["notional"] = df["price"] * df["volume"]
    return df.dropna(subset=["time","price","volume"]).sort_values("time").reset_index(drop=True)

# ===================== Indicators =====================

def ema(series: pd.Series, length: int) -> pd.Series:
    return pd.Series(series, dtype=float).ewm(span=length, adjust=False).mean()

def atr_wilder(df: pd.DataFrame, length: int = 14) -> float:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    tr = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1/length, adjust=False).mean().iloc[-1])

def fib_golden_zone(lo: float, hi: float, tol_pct: float) -> Tuple[float,float]:
    r618 = lo + 0.618*(hi-lo)
    r660 = lo + 0.66*(hi-lo)
    zmin, zmax = min(r618, r660), max(r618, r660)
    pad = ((zmin+zmax)/2.0)*tol_pct
    return zmin - pad, zmax + pad

# ===================== Volume Profile =====================

def build_volume_profile(trades: pd.DataFrame, bin_pct: float) -> Tuple[float, List[Tuple[float,float]]]:
    if trades.empty:
        return np.nan, []
    px = trades["price"].astype(float).to_numpy()
    notional = trades["notional"].astype(float).to_numpy()
    mid = float(np.nanmedian(px))
    width = max(1e-9, bin_pct * mid)
    bins = np.round(px / width) * width
    bucket = {}
    for b, n in zip(bins, notional):
        bucket[b] = bucket.get(b, 0.0) + float(n)
    items = sorted(bucket.items(), key=lambda x: x[1], reverse=True)
    if not items:
        return np.nan, []
    vpoc_price = float(items[0][0])
    hvns = [(float(p), float(v)) for p, v in items[1:1+TOP_HVNS]]
    return vpoc_price, hvns

# ===================== Thresholds (JSON, hot-reload) =====================
THRESH = {}
_THR_MTIME = None

def _maybe_reload_thresholds():
    global THRESH, _THR_MTIME
    try:
        m = os.path.getmtime(THR_PATH)
    except OSError:
        return
    if _THR_MTIME is None or m > _THR_MTIME:
        try:
            with open(THR_PATH, "r", encoding="utf-8") as f:
                THRESH = json.load(f)
            _THR_MTIME = m
            print(f"[thresholds] reloaded {THR_PATH}")
        except Exception as e:
            print("[thresholds] load err:", e)


def threshold_for(symbol: str, lb_s: int, key: str = "delta_usd", default: float = 2500.0) -> float:
    _maybe_reload_thresholds()
    sym_cfg = THRESH.get(symbol, {})
    per_lb = sym_cfg.get("thresholds", {}).get(str(lb_s), {})
    val = per_lb.get(key)
    if val is None:
        base = sym_cfg.get("thresholds", {}).get("60", {}).get(key, default)
        val = float(base) * math.sqrt(lb_s/60.0)
    return float(val)

# ===================== Tuning Rules (hot-apply) =====================
class Tuning:
    def __init__(self, path: str):
        self.path = path
        self.mtime = None
        self.rules = {}
    def reload_if_changed(self):
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            return
        if self.mtime is None or m > self.mtime:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.rules = json.load(f)
                self.mtime = m
                print(f"[tuning] reloaded {self.path}")
            except Exception as e:
                print("[tuning] load err:", e)
    def params(self, symbol: str) -> Dict[str, float]:
        self.reload_if_changed()
        g = (self.rules.get("global", {}) or {}).get("params", {}) if isinstance(self.rules, dict) else {}
        p = (self.rules.get("per_coin", {}) or {}).get(symbol, {}) if isinstance(self.rules, dict) else {}
        if isinstance(p, dict): p = p.get("params", {}) or {}
        out = {}
        for k in ("conf_long_min","overshoot_min","tp1_atr","sl_atr"):
            if k in p: out[k] = p[k]
            elif k in g: out[k] = g[k]
        return out

TUNING = Tuning(TUNING_PATH)

# ===================== Flow/Levels =====================

def flow_deltas(trades: pd.DataFrame, lookbacks: List[int]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str,float]] = {}
    if trades.empty:
        for lb in lookbacks:
            out[lb] = {"buys":0.0, "sells":0.0, "delta":0.0, "n":0}
        return out
    now = trades["time"].max()
    for lb in lookbacks:
        cut = now - pd.Timedelta(seconds=lb)
        win = trades[trades["time"] >= cut]
        buys  = float(win.loc[win["side"]=="buy", "notional"].sum())
        sells = float(win.loc[win["side"]=="sell","notional"].sum())
        out[lb] = {"buys":buys, "sells":sells, "delta":buys-sells, "n": int(len(win))}
    return out


def ema_levels(df1h: pd.DataFrame, df1d: pd.DataFrame) -> List[Tuple[str, Tuple[float,float], float]]:
    levels = []
    if len(df1h) > 200:
        closes = df1h["close"].astype(float)
        for p in [50, 200]:
            val = float(ema(closes, p).iloc[-1])
            levels.append((f"EMA{p}_1H", (val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)), val))
    if len(df1d) > 200:
        closes = df1d["close"].astype(float)
        for p in [20, 50, 100, 200]:
            val = float(ema(closes, p).iloc[-1])
            levels.append((f"EMA{p}_1D", (val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)), val))
    return levels


def fib_levels(df1h: pd.DataFrame) -> List[Tuple[str, Tuple[float,float], float]]:
    levels = []
    if len(df1h) < 48:
        return levels
    lo = df1h["low"].tail(48).min(); hi = df1h["high"].tail(48).max()
    z = fib_golden_zone(lo, hi, LEVEL_TOL_PCT)
    levels.append(("FIB_TINY", z, 0.5*(z[0]+z[1])))
    lo = df1h["low"].tail(24*7).min(); hi = df1h["high"].tail(24*7).max()
    z = fib_golden_zone(lo, hi, LEVEL_TOL_PCT)
    levels.append(("FIB_SHORT", z, 0.5*(z[0]+z[1])))
    return levels


def node_levels(vpoc: float, hvns: List[Tuple[float,float]]) -> List[Tuple[str, Tuple[float,float], float]]:
    out = []
    if not math.isnan(vpoc):
        out.append(("VPOC", (vpoc*(1-LEVEL_TOL_PCT), vpoc*(1+LEVEL_TOL_PCT)), vpoc))
    for i, (p, _v) in enumerate(hvns, start=1):
        out.append((f"HVN{i}", (p*(1-LEVEL_TOL_PCT), p*(1+LEVEL_TOL_PCT)), p))
    return out


def price_in_band(px: float, band: Tuple[float,float]) -> bool:
    lo, hi = band
    return (lo <= px <= hi)

# ===================== Trade State & Logging =====================

@dataclass
class VirtualTrade:
    trade_uuid: str
    symbol: str
    side: str
    level: str
    entry_ts: pd.Timestamp
    entry_px: float
    atr_1h: float
    best_lb: int
    best_delta: float
    threshold_usd: float
    deltas: Dict[int, float]
    levels_at_entry: Dict[str, float]
    meta_at_entry: Dict[str, object]
    score: float
    a_tier: int
    # tuned params used at entry
    conf_long_min: int
    overshoot_min: float
    tp1_atr: float
    sl_atr: float
    # sampling
    track_until: pd.Timestamp
    next_sample_ts: pd.Timestamp
    last_sample_ts: pd.Timestamp
    samples: List[Dict] = field(default_factory=list)
    hit_tp1: int = 0
    mae_pct: float = 0.0
    mfe_pct: float = 0.0

OPEN: Dict[str, VirtualTrade] = {}
LAST_TRADE_TS: Dict[str, float] = {}
LOCK = threading.Lock()


def ensure_csv_header():
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp_utc","symbol","side","level","entry","atr_1h",
                "best_lb","best_delta","threshold_usd",
                "lookbacks_json","levels_json","meta_json",
                "sample_ts","sample_px","ret_pct","mae_pct","mfe_pct"
            ])


def append_sample(tr: VirtualTrade, sample_px: float, ts: pd.Timestamp):
    ret_pct = (sample_px - tr.entry_px)/tr.entry_px*100.0 if tr.side == "LONG" else (tr.entry_px - sample_px)/tr.entry_px*100.0
    tr.mfe_pct = max(tr.mfe_pct, ret_pct)
    tr.mae_pct = min(tr.mae_pct, ret_pct)
    tr.samples.append({"ts": ts, "px": sample_px, "ret_pct": ret_pct, "mae_pct": tr.mae_pct, "mfe_pct": tr.mfe_pct})
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            tr.entry_ts.strftime('%Y-%m-%d %H:%M:%S'), tr.symbol, tr.side, tr.level, f"{tr.entry_px:.8f}", f"{tr.atr_1h:.8f}",
            tr.best_lb, f"{tr.best_delta:.2f}", f"{tr.threshold_usd:.2f}",
            json.dumps({int(lb): {"delta": float(v)} for lb, v in tr.deltas.items()}),
            json.dumps(tr.levels_at_entry),
            json.dumps(tr.meta_at_entry),
            ts.strftime('%Y-%m-%d %H:%M:%S'), f"{sample_px:.8f}", f"{ret_pct:.5f}", f"{tr.mae_pct:.5f}", f"{tr.mfe_pct:.5f}"
        ])

# ===================== Score helper =====================

def _score(delta_strength: float, long_conf: int, dist_atr: float) -> float:
    ds = min(delta_strength, SCORE_CAP)
    return 0.6*ds + 0.3*(1 if long_conf >= 2 else 0) + 0.1*(1 if (DIST_ATR_MIN <= dist_atr <= DIST_ATR_MAX) else 0)

# ===================== Detect & Open =====================

def detect_and_open(symbol: str):
    # Cooldown
    now_ts = time.time()
    if symbol in LAST_TRADE_TS and (now_ts - LAST_TRADE_TS[symbol]) < COOLDOWN_MIN*60:
        return

    # Tuning (per-coin overrides)
    tp = TUNING.params(symbol)
    conf_min = int(tp.get("conf_long_min", GATE_LONG_CONF_MIN))
    ov_min   = float(tp.get("overshoot_min", GATE_OVERSHOOT_MIN))
    tp1_atr  = float(tp.get("tp1_atr", TP1_ATR))
    sl_atr   = float(tp.get("sl_atr", SL_ATR))

    # Fetch context
    df1h = fetch_candles(symbol, 3600)
    df1d = fetch_candles(symbol, 86400)
    if len(df1h) < 60 or len(df1d) < 60:
        return
    last_px = float(df1h["close"].iloc[-1])
    atr1h   = atr_wilder(df1h, ATR_LEN)

    # Trades & flow
    trades = fetch_trades(symbol)
    deltas_full = flow_deltas(trades, LOOKBACKS)

    # Volume nodes (recent)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=PROFILE_LOOKBACK_H)
    prof_trades = trades[trades["time"] >= cutoff] if not trades.empty else pd.DataFrame([])
    vpoc, hvns = build_volume_profile(prof_trades, PROFILE_BIN_PCT)

    # Levels
    fibs  = fib_levels(df1h)
    emas  = ema_levels(df1h, df1d)
    nodes = node_levels(vpoc, hvns)
    levels = fibs + emas + nodes
    if not levels:
        return

    # Which bands are we in?
    hits = [(name, band, mid) for (name, band, mid) in levels if price_in_band(last_px, band)]
    if not hits:
        return

    # Choose most specific hit
    priority = {
        "VPOC": 100, "HVN1": 99, "HVN2": 98, "HVN3": 97,
        "EMA200_1D": 90, "EMA100_1D": 89, "EMA50_1D": 88, "EMA20_1D": 87,
        "EMA200_1H": 80,  "EMA50_1H": 79,
        "FIB_SHORT": 60,  "FIB_TINY": 50,
    }
    hits.sort(key=lambda t: priority.get(t[0], 0), reverse=True)
    level_name, level_band, level_mid = hits[0]

    # Best lookback & side
    best_lb = max(deltas_full.keys(), key=lambda lb: abs(float(deltas_full[lb]["delta"])) if lb in deltas_full else 0.0)
    best_delta = float(deltas_full[best_lb]["delta"]) if best_lb in deltas_full else 0.0
    side = "LONG" if best_delta > 0 else ("SHORT" if best_delta < 0 else None)
    if side is None:
        return

    # Confluence & overshoot
    sign = 1 if best_delta > 0 else -1
    conf_all  = sum(1 for lb in LOOKBACKS if lb in deltas_full and float(deltas_full[lb]["delta"]) * sign > 0)
    conf_long = sum(1 for lb in (900, 1800, 3600) if lb in deltas_full and float(deltas_full[lb]["delta"]) * sign > 0)
    thr_usd   = threshold_for(symbol, best_lb, key="delta_usd")
    overshoot = abs(best_delta) / max(1e-9, thr_usd)

    # Distance to triggered level mid (ATR units)
    dist_atr = abs((last_px - float(level_mid)) / max(1e-9, atr1h))

    # Gates (using tuned params)
    if conf_long < conf_min:
        return
    if overshoot < ov_min:
        return
    if not (DIST_ATR_MIN <= dist_atr <= DIST_ATR_MAX):
        return

    # Meta & entry context
    levels_at_entry = {name: float(mid) for (name, _band, mid) in levels}
    fb = {int(lb): {"buys": float(v["buys"]), "sells": float(v["sells"]), "delta": float(v["delta"]), "n": int(v["n"])} for lb, v in deltas_full.items()}

    score = _score(overshoot, conf_long, dist_atr)
    a_tier = int((conf_long >= 2) and (overshoot >= 1.5) and (DIST_ATR_MIN <= dist_atr <= DIST_ATR_MAX))

    meta = {
        "best_lb_s": int(best_lb),
        "best_delta_usd": float(best_delta),
        "threshold_usd": float(thr_usd),
        "flow_breakdown": fb,
        "sign_confluence_all": int(conf_all),
        "sign_confluence_long": int(conf_long),
        "delta_strength": float(overshoot),
        "dist_atr": {level_name: float(dist_atr)},
        "score": float(score),
        "a_tier": bool(a_tier),
        # tuned params used:
        "used_conf_long_min": int(conf_min),
        "used_overshoot_min": float(ov_min),
        "used_tp1_atr": float(tp1_atr),
        "used_sl_atr": float(sl_atr),
    }

    # Open virtual trade (store tuned params)
    vt = VirtualTrade(
        trade_uuid=str(uuid.uuid4()),
        symbol=symbol,
        side=side,
        level=level_name,
        entry_ts=pd.Timestamp.utcnow(),
        entry_px=last_px,
        atr_1h=atr1h,
        best_lb=int(best_lb),
        best_delta=float(best_delta),
        threshold_usd=float(thr_usd),
        deltas={int(lb): float(v["delta"]) for lb, v in deltas_full.items()},
        levels_at_entry=levels_at_entry,
        meta_at_entry=meta,
        score=float(score),
        a_tier=a_tier,
        conf_long_min=int(conf_min),
        overshoot_min=float(ov_min),
        tp1_atr=float(tp1_atr),
        sl_atr=float(sl_atr),
        track_until=pd.Timestamp.utcnow() + pd.Timedelta(minutes=TRACK_WINDOW_MIN),
        next_sample_ts=pd.Timestamp.utcnow() + pd.Timedelta(minutes=SAMPLE_EVERY_MIN),
        last_sample_ts=pd.Timestamp.utcnow(),
    )

    with LOCK:
        OPEN[symbol] = vt
        LAST_TRADE_TS[symbol] = now_ts

    # Log t0
    ensure_csv_header()
    append_sample(vt, last_px, pd.Timestamp.utcnow())

    # TP1/SL suggestions (for the alert)
    tp1 = last_px + (tp1_atr*atr1h if side == "LONG" else -tp1_atr*atr1h)
    sl  = last_px - (sl_atr*atr1h if side == "LONG" else -sl_atr*atr1h)

    msg = (
        f"<b>{symbol}</b> {side}"

        f"Entry: <code>{fmt_px(last_px)}</code>  | Level: {level_name}"

        f"ATR(1H): {fmt_px(atr1h)}  | best_lb: {best_lb}s  | Δ/Thr: {overshoot:.2f}x  | conf_long: {conf_long}"

        f"TP1≈ {fmt_px(tp1)}  | SL≈ {fmt_px(sl)}  | score: {score:.2f}{'  | A-tier' if a_tier else ''}"
    )
    tg_send(msg)
    print("[OPEN]", symbol, side, "@", fmt_px(last_px), "|", level_name, "| confL", conf_long, "| ov", f"{overshoot:.2f}")

# ===================== Sampler =====================

class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
    def run(self):
        while True:
            try:
                self.tick()
            except Exception as e:
                print("[Sampler]", e)
            time.sleep(SAMPLE_EVERY_MIN * 60)
    def tick(self):
        now = pd.Timestamp.utcnow()
        with LOCK:
            items = list(OPEN.items())
        for sym, tr in items:
            if now < tr.next_sample_ts:
                continue
            df1m = fetch_candles(sym, 60)
            if df1m.empty:
                tr.next_sample_ts = now + pd.Timedelta(minutes=SAMPLE_EVERY_MIN)
                continue
            px = float(df1m["close"].iloc[-1])
            append_sample(tr, px, now)

            # Time-based management nudges (use tr.tp1_atr)
            elapsed_min = (now - tr.entry_ts).total_seconds() / 60.0
            atr_pct = (tr.atr_1h / tr.entry_px) * 100.0
            tp1_pct = (tr.tp1_atr * tr.atr_1h / tr.entry_px) * 100.0

            if tr.hit_tp1 == 0 and tr.mfe_pct >= tp1_pct:
                tr.hit_tp1 = 1
                tg_send(f"{sym} • TP1 reached (≈{tr.tp1_atr:.2f} ATR). Move SL to BE.")

            if 14.9 <= elapsed_min <= 20.0:
                if tr.mfe_pct < 0.05*atr_pct and tr.mae_pct <= -0.10*atr_pct:
                    tg_send(f"{sym} • 15m: weak follow-through (MFE<{0.05:.2f} ATR & MAE>{0.10:.2f} ATR). Consider exit/trim.")

            if 29.9 <= elapsed_min <= 35.0:
                if tr.mfe_pct < 0.10*atr_pct:
                    tg_send(f"{sym} • 30m: still <0.10 ATR MFE. Consider cutting/halving.")

            if 59.9 <= elapsed_min <= 65.0:
                if tr.mfe_pct < tp1_pct:
                    tg_send(f"{sym} • 60m: no TP1 yet. Timeout close suggested.")

            if now >= tr.track_until:
                with LOCK:
                    OPEN.pop(sym, None)
                tg_send(f"{sym} • Tracking ended ({TRACK_WINDOW_MIN}m). Final MFE {tr.mfe_pct:.2f}% | MAE {tr.mae_pct:.2f}%.")
            else:
                tr.last_sample_ts = now
                tr.next_sample_ts = now + pd.Timedelta(minutes=SAMPLE_EVERY_MIN)

# ===================== Main =====================

def main():
    ensure_csv_header()
    sampler = Sampler(); sampler.start()
    last_health = 0.0
    while True:
        t0 = time.time()
        for sym in SYMBOLS:
            try:
                detect_and_open(sym)
            except Exception as e:
                print("[Loop]", sym, e)
        if time.time() - last_health > 120:
            print("[health]", datetime.utcnow().strftime("%H:%M:%S"))
            last_health = time.time()
        dt = time.time() - t0
        time.sleep(max(0.0, 10 - dt))


if __name__ == "__main__":
    main()
