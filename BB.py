"""
Viv (Blockchained Snipers) Daily Bias → Python Telegram Alerts (Hardened v2)
---------------------------------------------------------------------
(For Coinbase: native 4H candles are not available; this build now **synthesizes 4H** from 1H via pandas resampling, so all logic stays truly 4H.)
Includes requested polish:
- Close-only HTF bias (no forming-bar repaint)
- Dynamic symbol discovery (USD quote only) + optional 24h volume filter
- Robust CCXT fetch with retry/backoff (per-call)
- PVWAP (session-reset VWAP) *optional* (metric only, not a gate)
- FVG mitigation flag (reduced weight if mitigated)
- Markdown-safe Telegram
- Dedup + per-coin cooldown + global max-per-pass
- Persistent state (cooldowns + recent alert hashes) on disk
- Weighted confluence scoring → ★ / ★★ / ★★★

Usage
- pip install ccxt pandas numpy requests pytz
- Export TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
- Run: python viv_daily_bias_alerts.py
"""

from __future__ import annotations
import os
import time
import json
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable

import pandas as pd
import numpy as np
import requests
import ccxt
import pytz
from datetime import datetime, timedelta

# ==========================
# ---- Configuration ----
# ==========================

EXCHANGE = "coinbase"  # ccxt id ("bybit", "kraken", etc.)
DEFAULT_SYMBOLS = [
    "BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "ADA/USD", "DOGE/USD", "SHIB/USD",
    "LTC/USD", "BCH/USD", "LINK/USD", "AVAX/USD", "DOT/USD", "MATIC/USD", "ATOM/USD",
    "XLM/USD", "ALGO/USD", "ETC/USD", "FIL/USD", "ICP/USD", "IMX/USD", "HBAR/USD",
    "NEAR/USD", "ARB/USD", "OP/USD", "SUI/USD", "SEI/USD", "INJ/USD", "AAVE/USD",
    "MKR/USD", "UNI/USD", "RUNE/USD", "LDO/USD", "DYDX/USD", "SNX/USD", "STX/USD",
    "GRT/USD", "RPL/USD", "RNDR/USD", "PYTH/USD", "JTO/USD", "JUP/USD", "BONK/USD",
    "WLD/USD", "TIA/USD", "AR/USD", "FTM/USD", "SAND/USD", "APE/USD", "GALA/USD",
]
# Symbols to exclude if present on exchange (keeps your earlier removals safe)
EXCLUDE_SYMBOLS = set([
    "USDT/USD"
])

DISCOVER_SYMBOLS = True           # True = load markets + filter /USD, minus EXCLUDE_SYMBOLS
MIN_USD_VOLUME_24H = 5_000_000    # skip pairs with < $5M 24h volume if ticker provides it (best-effort)
MAX_SYMBOLS = 70                   # cap universe

TIMEFRAMES = {"1d": "1d", "4h": "1h", "1h": "1h", "15m": "15m"}  # fetch 1H and resample → 4H
LOOKBACK_BARS = {"1d": 220, "4h": 480, "1h": 1500, "15m": 1600}
  # 1h raised so resampled 4H has enough bars for 200-EMA

# Bias / filters
EMA_FAST = 20
EMA_SLOW = 50
EMA_TREND = 200

# Volume spike
VOL_LOOKBACK = 96
VOL_PERCENTILE = 80

# Manipulation (sweep)
SWEEP_LOOKBACK = 48
WICK_TOLERANCE = 0.0005

# OTE (ICT 0.618–0.66)
OTE_LOW = 0.618
OTE_HIGH = 0.786
OTE_NEAR = 0.001
#OTE_NEAR = 0.006

VWAP_SOFT_GATE = True
VWAP_SOFT_GATE_PCT = 0.003  # 0.3%


# Alert cadence
POLL_SECONDS = 120
COOLDOWN_MINUTES = 30
GLOBAL_MAX_ALERTS_PER_PASS = 6

# Telegram
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID   = "7967738614"


# PVWAP (session-reset VWAP) — optional metric, not gating
PVWAP_ENABLED = True
# Reset times (HH:MM in tz below). Example: NY open 06:30 PT, London 00:00 PT
PVWAP_RESETS_LOCAL = ["00:00", "06:30"]

TZ = pytz.timezone("America/Los_Angeles")
STATE_PATH = "viv_state.json"

# Scoring weights (sum forms 0–6-ish). You can tune these.
WEIGHTS = {
    "vwap": 1.0,
    "poc": 1.0,
    "ote": 1.0,
    "manip": 1.2,
    "obfvg_unmitigated": 1.0,
    "obfvg_mitigated": 0.5,
    "volume": 1.0,
}

# ==========================
# ---- Utilities ----
# ==========================

def resample_ohlc(df: pd.DataFrame, rule: str = '4H') -> pd.DataFrame:
    """Resample OHLCV to a higher timeframe (e.g., 1H → 4H). Index must be tz‑aware.
    Right‑closed windows; drops incomplete leading windows by default.
    """
    o = df['open'].resample(rule, label='right', closed='right').first()
    h = df['high'].resample(rule, label='right', closed='right').max()
    l = df['low'].resample(rule, label='right', closed='right').min()
    c = df['close'].resample(rule, label='right', closed='right').last()
    v = df['volume'].resample(rule, label='right', closed='right').sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ['open','high','low','close','volume']
    return out.dropna()
# ==========================

def md_escape(s: str) -> str:
    return s.replace("_", "\_").replace("*", "\*").replace("[", "\[").replace("]", "\]")


def backoff(retries: int = 3, base: float = 0.6, factor: float = 1.7) -> Callable:
    def deco(fn):
        def wrapped(*args, **kwargs):
            delay = base
            for i in range(retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if i == retries:
                        raise
                    sleep = delay * (factor ** i) * (1 + 0.2 * random.random())
                    time.sleep(sleep)
            return None
        return wrapped
    return deco


def send_telegram(text: str, parse_mode: str = "Markdown") -> None:
    msg = text
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or "<YOUR_" in TELEGRAM_BOT_TOKEN:
        print("[TELEGRAM] Missing token/chat_id – printing instead:\n" + msg)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": parse_mode, "disable_web_page_preview": True}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}\n{msg}")


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df['close'] * df['volume']).cumsum()
    vol = df['volume'].cumsum()
    return pv / vol


def pvwaps(df: pd.DataFrame, resets_local: List[str]) -> pd.Series:
    if not PVWAP_ENABLED:
        return vwap(df)
    # Build session reset timestamps in UTC
    local_index = df.index.tz_convert(TZ)
    reset_times = set()
    for ts in local_index:
        for hhmm in resets_local:
            h, m = map(int, hhmm.split(":"))
            candidate = ts.replace(hour=h, minute=m, second=0, microsecond=0)
            reset_times.add(candidate)
    # Compute PVWAP: reset cumulative at nearest past reset
    pvwap_vals = []
    pv_sum = 0.0
    vol_sum = 0.0
    last_reset = None
    for ts, row in zip(local_index, df.itertuples()):
        # reset at exact times
        if any(abs((ts - r).total_seconds()) < 60 for r in reset_times):
            pv_sum = 0.0; vol_sum = 0.0; last_reset = ts
        pv_sum += row.close * row.volume
        vol_sum += row.volume
        pvwap_vals.append(pv_sum / max(vol_sum, 1e-12))
    return pd.Series(pvwap_vals, index=df.index)


def poc_volume_profile(df: pd.DataFrame, bins: int = 60) -> float:
    tmp = df.copy()
    q = min(bins, max(5, tmp.shape[0] // 5))
    tmp['bin'] = pd.qcut(tmp['close'], q=q, duplicates='drop')
    vol_profile = tmp.groupby('bin')['volume'].sum()
    if vol_profile.empty:
        return float(tmp['close'].iloc[-1])
    max_bin = vol_profile.idxmax()
    return float(tmp.groupby('bin')['close'].mean().loc[max_bin])


def swing_points(df: pd.DataFrame, left: int = 2, right: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        if highs[i] == max(highs[i - left:i + right + 1]):
            sh[i] = True
        if lows[i] == min(lows[i - left:i + right + 1]):
            sl[i] = True
    return sh, sl


def detect_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    fvg_up = (df['low'].shift(1) > df['high'].shift(2))
    fvg_down = (df['high'].shift(1) < df['low'].shift(2))
    # mitigation: if a later candle's wick touches the gap range
    up_mitigated = pd.Series(False, index=df.index)
    down_mitigated = pd.Series(False, index=df.index)
    for i in range(2, len(df)):
        # UP FVG between high[i-2] and low[i-1]
        if bool(fvg_up.iloc[i]):
            hi = df['high'].iloc[i-2]
            lo = df['low'].iloc[i-1]
            later = df.iloc[i+1:]
            if not later.empty and (later['low'] <= lo).any():
                up_mitigated.iloc[i] = True
        # DOWN FVG between high[i-1] and low[i-2]
        if bool(fvg_down.iloc[i]):
            hi = df['high'].iloc[i-1]
            lo = df['low'].iloc[i-2]
            later = df.iloc[i+1:]
            if not later.empty and (later['high'] >= hi).any():
                down_mitigated.iloc[i] = True
    return pd.DataFrame({
        "fvg_up": fvg_up.fillna(False),
        "fvg_down": fvg_down.fillna(False),
        "mitigated_up": up_mitigated,
        "mitigated_down": down_mitigated,
    })


def detect_simple_ob(df: pd.DataFrame) -> pd.DataFrame:
    sh, sl = swing_points(df)
    ob_bull = np.zeros(len(df), dtype=bool)
    ob_bear = np.zeros(len(df), dtype=bool)
    prev_sw_high = np.nan
    prev_sw_low = np.nan
    for i in range(len(df)):
        if sh[i]: prev_sw_high = df['high'].iloc[i]
        if sl[i]: prev_sw_low = df['low'].iloc[i]
        if not np.isnan(prev_sw_high) and df['close'].iloc[i] > prev_sw_high:
            j = max(0, i - 1)
            while j > 0 and df['close'].iloc[j] < df['open'].iloc[j]:
                ob_bull[j] = True; j -= 1
        if not np.isnan(prev_sw_low) and df['close'].iloc[i] < prev_sw_low:
            j = max(0, i - 1)
            while j > 0 and df['close'].iloc[j] > df['open'].iloc[j]:
                ob_bear[j] = True; j -= 1
    return pd.DataFrame({"ob_bull": ob_bull, "ob_bear": ob_bear})


def volume_spike(df: pd.DataFrame) -> bool:
    vols = df['volume'].tail(VOL_LOOKBACK)
    if len(vols) < 10:
        print("Vol issue")
        return False
    thr = np.percentile(vols.iloc[:-1], VOL_PERCENTILE)
    return float(vols.iloc[-1]) >= float(thr)


def bollinger_width(df: pd.DataFrame, length: int = 20, mult: float = 2.0) -> pd.Series:
    ma = df['close'].rolling(length).mean()
    sd = df['close'].rolling(length).std(ddof=0)
    upper = ma + mult * sd
    lower = ma - mult * sd
    return (upper - lower) / ma

# --- Liquidity equal-high/equal-low clustering ---
from typing import List

def equal_levels(levels: List[float], tol: float = 0.0008) -> List[float]:
    """Cluster nearby price levels within a relative tolerance (default 0.08%)
    and return the averaged 'equal' levels. Requires numpy as np.
    """
    if not levels:
        return []
    levels = sorted(map(float, levels))
    clusters: List[float] = []
    cur: List[float] = [levels[0]]
    for x in levels[1:]:
        base = max(1e-9, float(np.mean(cur)))
        if abs(x - base) / base <= tol:
            cur.append(x)
        else:
            if len(cur) >= 2:
                clusters.append(float(np.mean(cur)))
            cur = [x]
    if len(cur) >= 2:
        clusters.append(float(np.mean(cur)))
    return clusters


def find_range(df15: pd.DataFrame) -> Optional[Tuple[float, float]]:
    width = bollinger_width(df15, 20, 2.0)
    recent = width.tail(96)
    if recent.median() < 0.01:
        hi = float(df15['high'].tail(96).max())
        lo = float(df15['low'].tail(96).min())
        return lo, hi
    return None


def detect_sweep(df15: pd.DataFrame, range_levels: Optional[Tuple[float, float]]) -> Dict[str, bool]:
    res = {"sweep_high": False, "sweep_low": False}
    window = df15.tail(SWEEP_LOOKBACK)
    if range_levels:
        lo, hi = range_levels
    else:
        lo, hi = float(window['low'].min()), float(window['high'].max())
    last = window.iloc[-1]
    if (last['high'] > hi * (1 - WICK_TOLERANCE)) and (last['close'] < hi):
        res["sweep_high"] = True
    if (last['low'] < lo * (1 + WICK_TOLERANCE)) and (last['close'] > lo):
        res["sweep_low"] = True
    return res


def current_bias(df1d: pd.DataFrame, df4h: pd.DataFrame) -> str:
    """
    Bias (stricter):
    - Primary: 1D (close vs 200EMA) AND 4H (EMA20 vs EMA50) must agree.
    - If they don't agree, fallback only if BOTH TFs triple-stack in the same direction.
    """
    d1 = df1d.iloc[:-1].copy() if len(df1d) > 1 else df1d.copy()
    h4 = df4h.iloc[:-1].copy() if len(df4h) > 1 else df4h.copy()
    for df in (d1, h4):
        df[f'ema{EMA_FAST}']   = ema(df['close'], EMA_FAST)
        df[f'ema{EMA_SLOW}']   = ema(df['close'], EMA_SLOW)
        df[f'ema{EMA_TREND}']  = ema(df['close'], EMA_TREND)
    d, h = d1.iloc[-1], h4.iloc[-1]

    day_up   = d['close'] > d[f'ema{EMA_TREND}']
    day_down = d['close'] < d[f'ema{EMA_TREND}']
    h4_up    = h[f'ema{EMA_FAST}'] > h[f'ema{EMA_SLOW}']
    h4_down  = h[f'ema{EMA_FAST}'] < h[f'ema{EMA_SLOW}']

    # Primary agreement
    if day_up and h4_up:   return "long"
    if day_down and h4_down:return "short"

    # Fallback stricter: both TFs must triple-stack in the SAME direction
    d_long  = d[f'ema{EMA_FAST}'] > d[f'ema{EMA_SLOW}'] > d[f'ema{EMA_TREND}']
    h_long  = h[f'ema{EMA_FAST}'] > h[f'ema{EMA_SLOW}'] > h[f'ema{EMA_TREND}']
    d_short = d[f'ema{EMA_FAST}'] < d[f'ema{EMA_SLOW}'] < d[f'ema{EMA_TREND}']
    h_short = h[f'ema{EMA_FAST}'] < h[f'ema{EMA_SLOW}'] < h[f'ema{EMA_TREND}']

    if d_long and h_long:     return "long"
    if d_short and h_short:   return "short"
    return "neutral"




def last_major_swing(df1h: pd.DataFrame) -> Tuple[float, float]:
    sh, sl = swing_points(df1h, left=3, right=3)
    idx = len(df1h) - 2
    last_high_idx = np.where(sh[:idx])[0]
    last_low_idx = np.where(sl[:idx])[0]
    if len(last_high_idx) == 0 or len(last_low_idx) == 0:
        return float(df1h['low'].iloc[-50:-1].min()), float(df1h['high'].iloc[-50:-1].max())
    h_idx = last_high_idx[-1]
    l_before_high = last_low_idx[last_low_idx < h_idx]
    if len(l_before_high):
        l_idx = l_before_high[-1]
        return float(df1h['low'].iloc[l_idx]), float(df1h['high'].iloc[h_idx])
    l_idx = last_low_idx[-1]
    h_before_low = last_high_idx[last_high_idx < l_idx]
    if len(h_before_low):
        h_idx = h_before_low[-1]
        return float(df1h['low'].iloc[l_idx]), float(df1h['high'].iloc[h_idx])
    return float(df1h['low'].iloc[-50:-1].min()), float(df1h['high'].iloc[-50:-1].max())


def ote_zone(swing_low: float, swing_high: float, direction: str) -> Tuple[float, float]:
    rng = swing_high - swing_low
    if rng <= 0:
        return swing_low, swing_high
    if direction == "long":
        z1 = swing_high - OTE_HIGH * rng
        z0 = swing_high - OTE_LOW * rng
        return (min(z0, z1) * (1 - OTE_NEAR), max(z0, z1) * (1 + OTE_NEAR))
    else:
        z0 = swing_low + OTE_LOW * rng
        z1 = swing_low + OTE_HIGH * rng
        return (min(z0, z1) * (1 - OTE_NEAR), max(z0, z1) * (1 + OTE_NEAR))


def price_in_zone(price: float, zone: Tuple[float, float]) -> bool:
    return zone[0] <= price <= zone[1]

# ==========================
# ---- Data Layer (CCXT) ----
# ==========================

class Data:
    def __init__(self, exchange_id: str):
        self.ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})

    @backoff()
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int):
        return self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def candles(self, symbol: str, tf: str, limit: int) -> pd.DataFrame:
        ohlcv = self.fetch_ohlcv(symbol, tf, limit)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"]).assign(
            ts=lambda x: pd.to_datetime(x.ts, unit='ms', utc=True)
        )
        return df.set_index('ts')

    @backoff()
    def load_markets(self):
        return self.ex.load_markets()

    @backoff()
    def fetch_ticker(self, symbol: str):
        return self.ex.fetch_ticker(symbol)

# ==========================
# ---- Persistent state ----
# ==========================

def load_state() -> Dict:
    if not os.path.exists(STATE_PATH):
        return {"cooldowns": {}, "seen": {}}
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"cooldowns": {}, "seen": {}}


def save_state(state: Dict) -> None:
    try:
        with open(STATE_PATH, "w") as f:
            json.dump(state, f)
    except Exception:
        pass

# ==========================
# ---- Alert Logic ----
# ==========================

@dataclass
class Setup:
    symbol: str
    direction: str
    entry_zone: Tuple[float, float]
    stop: float
    tp1: float
    tp2: float
    confluence: List[str]
    price: float
    timestamp: pd.Timestamp


def _starify(score: float) -> str:
    if score >= 4.8: return "★★★"
    if score >= 3.0: return "★★"
    return "★"


class VivScanner:
    def __init__(self, data: Data, state: Dict):
        self.data = data
        self.cooldowns: Dict[str, str] = state.get("cooldowns", {})  # iso strings
        self.seen: Dict[str, float] = state.get("seen", {})  # alert hash → timestamp

    def cooldown_ok(self, symbol: str, ts: pd.Timestamp) -> bool:
        iso = self.cooldowns.get(symbol)
        if not iso:
            return True
        prev = pd.Timestamp(iso)
        return (ts - prev).total_seconds() >= COOLDOWN_MINUTES * 60

    def mark_cooldown(self, symbol: str, ts: pd.Timestamp):
        self.cooldowns[symbol] = ts.isoformat()

    def hash_alert(self, s: Setup) -> str:
        z0, z1 = s.entry_zone
        key = f"{s.symbol}|{s.direction}|{round(z0,6)}|{round(z1,6)}|{s.timestamp.floor('15min').isoformat()}"
        return key

    def seen_ok(self, key: str) -> bool:
        # keep for 6h
        now = pd.Timestamp.utcnow()
        ts = self.seen.get(key)
        return (not ts) or ((now.timestamp() - ts) > 6*3600)

    def mark_seen(self, key: str):
        self.seen[key] = pd.Timestamp.utcnow().timestamp()

    def eligible(self, symbol: str) -> Optional[Setup]:
        try:
            d1 = self.data.candles(symbol, TIMEFRAMES['1d'], LOOKBACK_BARS['1d'])
            h1 = self.data.candles(symbol, TIMEFRAMES['1h'], LOOKBACK_BARS['1h'])
        # Build true 4H from 1H
            h4 = resample_ohlc(h1, '4H')
            m15 = self.data.candles(symbol, TIMEFRAMES['15m'], LOOKBACK_BARS['15m'])
        except Exception as e:
            print(f"[DATA] {symbol} {e}")
            return None
        if min(map(len, [d1, h4, h1, m15])) < 60:
            return None

        bias = current_bias(d1, h4)
        if bias == "neutral":
            return None

    # Range & sweep (manipulation) on 15m
        rng = find_range(m15)
        sweep = detect_sweep(m15, rng)
        has_manip = (sweep['sweep_low'] and bias == 'long') or (sweep['sweep_high'] and bias == 'short')

    # Equal-liquidity levels on 1h
        sh, sl = swing_points(h1, left=2, right=2)
        eq_highs = equal_levels(list(h1['high'][sh].tail(20).values)) if hasattr(sh, 'any') and sh.any() else []
        eq_lows  = equal_levels(list(h1['low'][sl].tail(20).values))  if hasattr(sl, 'any') and sl.any() else []

    # FVG / OB on 1h + 15m
        fvg1h = detect_fvgs(h1).tail(30)
        fvg15 = detect_fvgs(m15).tail(40)
        ob1h  = detect_simple_ob(h1).tail(30)

    # Directional structure (polarity-aware)
        fvg_up_qual   = (fvg1h['fvg_up'].any()   or fvg15['fvg_up'].any())
        fvg_down_qual = (fvg1h['fvg_down'].any() or fvg15['fvg_down'].any())
        ob_bull_qual  = ob1h['ob_bull'].any()
        ob_bear_qual  = ob1h['ob_bear'].any()

        if bias == 'long':
            has_ob_or_fvg = ob_bull_qual or fvg_up_qual
        else:
            has_ob_or_fvg = ob_bear_qual or fvg_down_qual

    # Structure may also be satisfied by directional equal-liquidity
        has_struct = has_ob_or_fvg or ((len(eq_highs) > 0 and bias == 'short') or (len(eq_lows) > 0 and bias == 'long'))

    # Volume (keep as gate for now; you can make it score-only if needed)
        vol_ok = volume_spike(m15)

    # Swing/OTE zone
        swing_lo, swing_hi = last_major_swing(h1)
        if swing_hi < swing_lo:  # ordering guard
            swing_lo, swing_hi = swing_hi, swing_lo
        zone = ote_zone(swing_lo, swing_hi, bias)
        last_price = float(m15['close'].iloc[-1])
        in_zone = price_in_zone(last_price, zone)

    # Near-OTE grace (optional)
        NEAR_OTE_ENABLE = True
        NEAR_OTE_PCT = 0.01  # 1%
        mid = (zone[0] + zone[1]) / 2.0
        dist_to_band = min(abs(last_price - zone[0]), abs(last_price - zone[1])) / max(1e-12, mid)
        in_or_near_zone = in_zone or (NEAR_OTE_ENABLE and dist_to_band <= NEAR_OTE_PCT)

    # VWAP / PVWAP
        m15['pvwap'] = pvwaps(m15, PVWAP_RESETS_LOCAL)
        vwap_val = float(m15['pvwap'].iloc[-1])
        vwap_dir = "long" if last_price > vwap_val else "short"

    # VWAP soft gate: only block if clearly opposite by >= 0.3%
        if VWAP_SOFT_GATE:
            vwap_gap = abs(last_price - vwap_val) / max(1e-12, vwap_val)
            if bias == "long" and last_price < vwap_val and vwap_gap >= VWAP_SOFT_GATE_PCT:
                return None
            if bias == "short" and last_price > vwap_val and vwap_gap >= VWAP_SOFT_GATE_PCT:
                return None

    # POC magnet over 1H window
        poc = poc_volume_profile(h1.tail(240))
        poc_confluence = "POC_above" if poc > last_price else "POC_below"

    # Core Viv gate (polarity-correct, with near-OTE allowed)
        core_ok = in_or_near_zone and has_manip and has_struct and vol_ok
        if not core_ok:
            return None

    # Risk (Viv)
        if bias == 'long':
            rngv = swing_hi - swing_lo
            tp1 = swing_hi - 0.27 * rngv
            tp2 = swing_hi - 0.62 * rngv
            stop = swing_lo
        else:
            rngv = swing_hi - swing_lo
            tp1 = swing_lo + 0.27 * rngv
            tp2 = swing_lo + 0.62 * rngv
            stop = swing_hi

    # Confluence tags (include OTE_near if used)
        confl = [f"bias:{bias}", f"VWAP:{vwap_dir}", poc_confluence]
        if rng: confl.append("range15m")
        if has_manip: confl.append("sweep_low" if bias=='long' else "sweep_high")
        if len(eq_highs) and bias == 'short': confl.append("eq_highs")
        if len(eq_lows)  and bias == 'long':  confl.append("eq_lows")
        if (bias=='long' and (ob_bull_qual or fvg_up_qual)) or (bias=='short' and (ob_bear_qual or fvg_down_qual)):
            confl.append("FVG" if ((bias=='long' and fvg_up_qual) or (bias=='short' and fvg_down_qual)) else "OB")
        if 'fvg1h' in locals() and (fvg1h.get('mitigated_up', pd.Series()).any() or fvg1h.get('mitigated_down', pd.Series()).any()
                                    or fvg15.get('mitigated_up', pd.Series()).any() or fvg15.get('mitigated_down', pd.Series()).any()):
            confl.append("FVG_mitigated")
        if vol_ok: confl.append("volume_spike")
        if in_zone: confl.append("OTE")
        elif NEAR_OTE_ENABLE and dist_to_band <= NEAR_OTE_PCT: confl.append("OTE_near")

    # SL sanity guard to avoid illogical stops
        if bias == 'long' and stop >= last_price:
            stop = min(stop, zone[0] * 0.995)
        elif bias == 'short' and stop <= last_price:
            stop = max(stop, zone[1] * 1.005)

    # Optional: quick WHY line for sanity
        print(f"[WHY] {symbol} bias={bias} vwap_dir={vwap_dir} price={last_price:.4f} pvwap={vwap_val:.4f} "
               f"in_or_near_ote={in_or_near_zone} has_manip={has_manip} has_struct={has_struct}")

        return Setup(
            symbol=symbol,
            direction=bias,
            entry_zone=zone,
            stop=float(stop),
            tp1=float(tp1),
            tp2=float(tp2),
            confluence=confl,
            price=last_price,
            timestamp=m15.index[-1].tz_convert(TZ),
        )

# ==========================
# ---- Output / Main ----
# ==========================

def score_confluence(s: Setup) -> Tuple[float, str]:
    dir_long = s.direction == 'long'
    confl = set(s.confluence)
    vwap_ok = ('VWAP:long' in confl and dir_long) or ('VWAP:short' in confl and not dir_long)
    poc_ok  = ('POC_above' in confl and dir_long) or ('POC_below' in confl and not dir_long)
    ote_ok  = ('OTE' in confl)
    manip_ok = ('sweep_low' in confl and dir_long) or ('sweep_high' in confl and not dir_long)
    ob_unmit = ('OB' in confl)
    fvg_unmit = ('FVG' in confl) and ('FVG_mitigated' not in confl)
    fvg_mit = ('FVG_mitigated' in confl)
    obfvg_score = (WEIGHTS['obfvg_unmitigated'] if (ob_unmit or fvg_unmit) else 0.0)
    if fvg_mit:
        obfvg_score = max(obfvg_score, WEIGHTS['obfvg_mitigated'])
    vol_ok = ('volume_spike' in confl)

    score = (
        WEIGHTS['vwap'] * float(vwap_ok) +
        WEIGHTS['poc'] * float(poc_ok) +
        WEIGHTS['ote'] * float(ote_ok) +
        WEIGHTS['manip'] * float(manip_ok) +
        obfvg_score +
        WEIGHTS['volume'] * float(vol_ok)
    )
    stars = _starify(score)
    return score, stars


def format_alert(s: Setup) -> str:
    score, stars = score_confluence(s)
    z0, z1 = s.entry_zone
    ts = s.timestamp.strftime('%Y-%m-%d %I:%M %p %Z')
    confl_str = ", ".join(sorted(s.confluence))
    lines = [
        f"*Viv-Style Setup* — *{md_escape(s.symbol)}* ({s.direction.upper()}) {stars}",
        f"Score: {score:.2f}",
        f"Time: {ts}",
        f"Price: {s.price:.8f}",
        f"Entry (OTE {OTE_LOW:.3f}-{OTE_HIGH:.2f}): {z0:.8f} → {z1:.8f}",
        f"SL (initial): {s.stop:.8f}  — move to BE after TP1",
        f"TP1 (50% @ -0.27): {s.tp1:.8f}",
        f"TP2 (25% @ -0.62): {s.tp2:.8f}",
        f"Confluence: {md_escape(confl_str)}",
    ]
    return "\n".join(lines)


def discover_symbols(data: Data) -> List[str]:
    if not DISCOVER_SYMBOLS:
        return [s for s in DEFAULT_SYMBOLS if s not in EXCLUDE_SYMBOLS][:MAX_SYMBOLS]
    try:
        markets = data.load_markets()
        usd_pairs = [m for m in markets if m.endswith('/USD')]
        usd_pairs = [s for s in usd_pairs if s not in EXCLUDE_SYMBOLS]
        # Optional: filter by 24h volume when available
        scored = []
        for s in usd_pairs:
            try:
                t = data.fetch_ticker(s)
                # Prefer baseVolume * last price when quoteVolume missing
                quote_vol = t.get('quoteVolume') or (t.get('baseVolume') or 0) * (t.get('last') or 0)
                scored.append((s, float(quote_vol or 0)))
            except Exception:
                scored.append((s, 0.0))
        # sort by volume desc and filter threshold
        scored.sort(key=lambda x: x[1], reverse=True)
        filtered = [s for s, v in scored if v >= MIN_USD_VOLUME_24H]
        res = (filtered or [s for s, _ in scored])[:MAX_SYMBOLS]
        return res
    except Exception as e:
        print(f"[DISCOVER] fallback to defaults: {e}")
        return [s for s in DEFAULT_SYMBOLS if s not in EXCLUDE_SYMBOLS][:MAX_SYMBOLS]


# Drop-off debug counter added to diagnose filtering
# ---------------------------------------------------------------------
# This block prints how many symbols fail each major gate per scan pass.
# Helps tune selectivity by showing which conditions are eliminating setups.

# ... keep entire previous script above unchanged ...

# Drop-off debug counter added to diagnose filtering
# ---------------------------------------------------------------------
# This block prints how many symbols fail each major gate per scan pass.
# Helps tune selectivity by showing which conditions are eliminating setups.

# ... keep entire previous script above unchanged ...

# Drop-off debug counter added to diagnose filtering
# ---------------------------------------------------------------------
# This block prints how many symbols fail each major gate per scan pass.
# Helps tune selectivity by showing which conditions are eliminating setups.

# ... keep entire previous script above unchanged ...

def debug_reasons(data: 'Data', symbol: str) -> str:
    """Standalone diagnostics matching eligible() gate order."""
    try:
        d1 = data.candles(symbol, TIMEFRAMES['1d'], LOOKBACK_BARS['1d'])
        h1 = data.candles(symbol, TIMEFRAMES['1h'], LOOKBACK_BARS['1h'])
        h4 = resample_ohlc(h1, '4H')
        m15 = data.candles(symbol, TIMEFRAMES['15m'], LOOKBACK_BARS['15m'])
    except Exception:
        return "data_error"
    if min(map(len, [d1, h4, h1, m15])) < 60:
        return "data_short"
    bias = current_bias(d1, h4)
    if bias == "neutral":
        return "bias_neutral"
    rng = find_range(m15)
    sweep = detect_sweep(m15, rng)
    has_manip = (sweep['sweep_low'] and bias == 'long') or (sweep['sweep_high'] and bias == 'short')
    fvg1h = detect_fvgs(h1).tail(30)
    fvg15 = detect_fvgs(m15).tail(40)
    ob1h = detect_simple_ob(h1).tail(30)
    sh, sl = swing_points(h1, left=2, right=2)
    eq_highs = list(h1['high'][sh].tail(20).values) if hasattr(sh, 'any') and sh.any() else []
    eq_lows  = list(h1['low'][sl].tail(20).values)  if hasattr(sl, 'any') and sl.any() else []
    fvg_present = ((fvg1h['fvg_up'].any() and bias == 'long') or (fvg1h['fvg_down'].any() and bias == 'short') or
                   (fvg15['fvg_up'].any() and bias == 'long') or (fvg15['fvg_down'].any() and bias == 'short'))
    ob_present = ((ob1h['ob_bull'].any() and bias == 'long') or (ob1h['ob_bear'].any() and bias == 'short'))
    has_ob_or_fvg = ob_present or fvg_present
    has_struct = has_ob_or_fvg or ((len(eq_highs) > 0 and bias == 'short') or (len(eq_lows) > 0 and bias == 'long'))
    swing_lo, swing_hi = last_major_swing(h1)
    zone = ote_zone(swing_lo, swing_hi, bias)
    last_price = float(m15['close'].iloc[-1])
    in_zone = (zone[0] <= last_price <= zone[1])
    vol_ok = volume_spike(m15)
    if not in_zone:    return "not_in_ote"
    if not has_manip:  return "no_manip"
    if not has_struct: return "no_struct"
    if not vol_ok:     return "no_vol"
    return "passed_core"

def run():
    data = Data(EXCHANGE)
    state = load_state()
    scanner = VivScanner(data, state)

    symbols = discover_symbols(data)
    print(f"Universe ({len(symbols)}): {symbols}")

    while True:
        start = pd.Timestamp.now(tz="UTC").tz_convert(TZ)
        any_sent = False
        sent_this_pass = 0

        # Debug counters
        drop = {"bias_neutral":0,"vwap_misaligned":0,"not_in_ote":0,
                "no_manip":0,"no_struct":0,"no_vol":0,"passed_core":0}

        for sym in symbols:
            try:
                setup = scanner.eligible(sym)
            except Exception as e:
                print(f"[ERROR] {sym} {e}")
                continue
            if not setup:
                reason = debug_reasons(data, sym)
                drop.setdefault(reason, 0)
                drop[reason] += 1
                continue
            if not scanner.cooldown_ok(sym, setup.timestamp):
                continue
            key = scanner.hash_alert(setup)
            if not scanner.seen_ok(key):
                continue
            msg = format_alert(setup)
            send_telegram(msg)
            scanner.mark_cooldown(sym, setup.timestamp)
            scanner.mark_seen(key)
            any_sent = True
            sent_this_pass += 1
            drop["passed_core"] += 1
            print(f"[ALERT] {sym} {setup.direction} {setup.timestamp} score={score_confluence(setup)[0]:.2f}")
            if sent_this_pass >= GLOBAL_MAX_ALERTS_PER_PASS:
                break

        # Print debug summary after each full pass
        print(f"[DEBUG] drop-off counts: {drop}")

        save_state({"cooldowns": scanner.cooldowns, "seen": scanner.seen})
        if not any_sent:
            print(f"[{start.strftime('%H:%M:%S')}] No setups this pass.")
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    run()
