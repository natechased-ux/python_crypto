"""
Viv (Blockchained Snipers) HYBRID HTF Scanner (1D sweeps + swings; 4H structure)
-------------------------------------------------------------------------------
Hybrid mode:
- Uses 1D for major sweeps, daily volume confirmation, and swing OTE zones
- Uses 4H for OB/FVG, PVWAP metric, equal-levels and local confluence
- No 15m logic used (no intraday entry gating)
- Resamples 1H -> 4H where native 4H isn't available
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
EXCLUDE_SYMBOLS = set(["USDT/USD", "USD1/USD"])

DISCOVER_SYMBOLS = True
MIN_USD_VOLUME_24H = 100_000
MAX_SYMBOLS = 500

# Timeframes: we fetch 1d and 1h and resample 1h->4h
TIMEFRAMES = {"1d": "1d", "4h": "1h", "1h": "1h"}
LOOKBACK_BARS = {"1d": 360, "4h": 480, "1h": 1500}

# Bias / filters
EMA_FAST = 20
EMA_SLOW = 50
EMA_TREND = 200

# Volume spike (applies to 1D series in Hybrid mode)
VOL_LOOKBACK = 60  # number of daily bars to look back for percentile
VOL_PERCENTILE = 70

# Sweep detection window (in daily bars)
SWEEP_LOOKBACK = 30
WICK_TOLERANCE = 0.0025

# OTE (ICT 0.618–0.786)
OTE_LOW = 0.618
OTE_HIGH = 0.786
OTE_NEAR = 0.02

VWAP_SOFT_GATE = False  # PVWAP kept as metric on 4H
VWAP_SOFT_GATE_PCT = 0.003

# Alert cadence
POLL_SECONDS = 240
COOLDOWN_MINUTES = 60
GLOBAL_MAX_ALERTS_PER_PASS = 8

# Telegram (keep your tokens as env vars ideally)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "") or "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")   or "7967738614"

# PVWAP (session-reset VWAP) — optional metric (computed on 4H)
PVWAP_ENABLED = True
PVWAP_RESETS_LOCAL = ["00:00", "06:30"]

TZ = pytz.timezone("America/Los_Angeles")
STATE_PATH = "viv_state_hybrid.json"

# Scoring weights (tune as desired)
WEIGHTS = {
    "vwap": 1.0,
    "poc": 1.0,
    "ote": 1.0,
    "manip": 1.4,
    "obfvg_unmitigated": 1.0,
    "obfvg_mitigated": 0.5,
    "volume": 1.0,
}

# ==========================
# ---- Utilities ----
# ==========================

def resample_ohlc(df: pd.DataFrame, rule: str = '4H') -> pd.DataFrame:
    """Resample OHLCV to a higher timeframe (e.g., 1H → 4H). Index must be tz-aware."""
    o = df['open'].resample(rule, label='right', closed='right').first()
    h = df['high'].resample(rule, label='right', closed='right').max()
    l = df['low'].resample(rule, label='right', closed='right').min()
    c = df['close'].resample(rule, label='right', closed='right').last()
    v = df['volume'].resample(rule, label='right', closed='right').sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ['open','high','low','close','volume']
    return out.dropna()

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
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
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
    local_index = df.index.tz_convert(TZ)
    reset_times = set()
    for ts in local_index:
        for hhmm in resets_local:
            h, m = map(int, hhmm.split(":"))
            candidate = ts.replace(hour=h, minute=m, second=0, microsecond=0)
            reset_times.add(candidate)
    pvwap_vals = []
    pv_sum = 0.0
    vol_sum = 0.0
    for ts, row in zip(local_index, df.itertuples()):
        if any(abs((ts - r).total_seconds()) < 60 for r in reset_times):
            pv_sum = 0.0; vol_sum = 0.0
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
    up_mitigated = pd.Series(False, index=df.index)
    down_mitigated = pd.Series(False, index=df.index)
    for i in range(2, len(df)):
        if bool(fvg_up.iloc[i]):
            hi = df['high'].iloc[i-2]
            lo = df['low'].iloc[i-1]
            later = df.iloc[i+1:]
            if not later.empty and (later['low'] <= lo).any():
                up_mitigated.iloc[i] = True
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
        return False
    thr = np.percentile(vols.iloc[:-1], VOL_PERCENTILE)
    return float(vols.iloc[-1]) >= float(thr)

def bollinger_width(df: pd.DataFrame, length: int = 20, mult: float = 2.0) -> pd.Series:
    ma = df['close'].rolling(length).mean()
    sd = df['close'].rolling(length).std(ddof=0)
    upper = ma + mult * sd
    lower = ma - mult * sd
    return (upper - lower) / ma

def equal_levels(levels: List[float], tol: float = 0.0008) -> List[float]:
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

def find_range(df: pd.DataFrame) -> Optional[Tuple[float, float]]:
    width = bollinger_width(df, 20, 2.0)
    recent = width.tail(48)
    if recent.median() < 0.01:
        hi = float(df['high'].tail(48).max())
        lo = float(df['low'].tail(48).min())
        return lo, hi
    return None

def detect_sweep(df: pd.DataFrame, range_levels: Optional[Tuple[float, float]]) -> Dict[str, bool]:
    res = {"sweep_high": False, "sweep_low": False}
    window = df.tail(SWEEP_LOOKBACK)
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

    if day_up and h4_up:   return "long"
    if day_down and h4_down:return "short"

    d_long  = d[f'ema{EMA_FAST}'] > d[f'ema{EMA_SLOW}'] > d[f'ema{EMA_TREND}']
    h_long  = h[f'ema{EMA_FAST}'] > h[f'ema{EMA_SLOW}'] > h[f'ema{EMA_TREND}']
    d_short = d[f'ema{EMA_FAST}'] < d[f'ema{EMA_SLOW}'] < d[f'ema{EMA_TREND}']
    h_short = h[f'ema{EMA_FAST}'] < h[f'ema{EMA_SLOW}'] < h[f'ema{EMA_TREND}']

    if d_long and h_long:     return "long"
    if d_short and h_short:   return "short"
    return "neutral"

def last_major_swing(df: pd.DataFrame) -> Tuple[float, float]:
    # Generic swing detector that works for 4H or 1D depending on df passed
    sh, sl = swing_points(df, left=3, right=3)
    idx = len(df) - 2
    last_high_idx = np.where(sh[:idx])[0]
    last_low_idx = np.where(sl[:idx])[0]
    if len(last_high_idx) == 0 or len(last_low_idx) == 0:
        return float(df['low'].iloc[-50:-1].min()), float(df['high'].iloc[-50:-1].max())
    h_idx = last_high_idx[-1]
    l_before_high = last_low_idx[last_low_idx < h_idx]
    if len(l_before_high):
        l_idx = l_before_high[-1]
        return float(df['low'].iloc[l_idx]), float(df['high'].iloc[h_idx])
    l_idx = last_low_idx[-1]
    h_before_low = last_high_idx[last_high_idx < l_idx]
    if len(h_before_low):
        h_idx = h_before_low[-1]
        return float(df['low'].iloc[l_idx]), float(df['high'].iloc[h_idx])
    return float(df['low'].iloc[-50:-1].min()), float(df['high'].iloc[-50:-1].max())

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
        self.cooldowns: Dict[str, str] = state.get("cooldowns", {})
        self.seen: Dict[str, float] = state.get("seen", {})

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
        now = pd.Timestamp.utcnow()
        ts = self.seen.get(key)
        return (not ts) or ((now.timestamp() - ts) > 6*3600)

    def mark_seen(self, key: str):
        self.seen[key] = pd.Timestamp.utcnow().timestamp()

    def eligible(self, symbol: str) -> Optional[Setup]:
        try:
            d1 = self.data.candles(symbol, TIMEFRAMES['1d'], LOOKBACK_BARS['1d'])
            h1 = self.data.candles(symbol, TIMEFRAMES['1h'], LOOKBACK_BARS['1h'])
        except Exception as e:
            print(f"[DATA] {symbol} {e}")
            return None

        # Build 4H from 1H
        try:
            h4 = resample_ohlc(h1, '4H')
        except Exception as e:
            print(f"[RESAMPLE] {symbol} {e}")
            return None

        if min(map(len, [d1, h4, h1])) < 60:
            return None

        # Bias (1D + 4H agreement)
        bias = current_bias(d1, h4)
        if bias == "neutral":
            return None

        # ----- HYBRID HTF structure checks -----
        # Range & sweep ON DAILY (1D) — major liquidity grabs
        rng = find_range(d1)
        sweep = detect_sweep(d1, rng)
        has_manip = (sweep['sweep_low'] and bias == 'long') or (sweep['sweep_high'] and bias == 'short')

        # Equal-liquidity levels on 4H (local)
        sh, sl = swing_points(h4, left=2, right=2)
        eq_highs = equal_levels(list(h4['high'][sh].tail(20).values)) if hasattr(sh, 'any') and sh.any() else []
        eq_lows  = equal_levels(list(h4['low'][sl].tail(20).values))  if hasattr(sl, 'any') and sl.any() else []

        # FVG / OB on 4H (local structure)
        fvg4h = detect_fvgs(h4).tail(30)
        ob4h  = detect_simple_ob(h4).tail(30)

        # Directional structure (polarity-aware using 4H FVG/OB)
        fvg_up_qual   = (fvg4h['fvg_up'].any())
        fvg_down_qual = (fvg4h['fvg_down'].any())
        ob_bull_qual  = ob4h['ob_bull'].any()
        ob_bear_qual  = ob4h['ob_bear'].any()

        if bias == 'long':
            has_ob_or_fvg = ob_bull_qual or fvg_up_qual
        else:
            has_ob_or_fvg = ob_bear_qual or fvg_down_qual

        # Structure may also be satisfied by directional equal-liquidity (4H)
        has_struct = has_ob_or_fvg or ((len(eq_highs) > 0 and bias == 'short') or (len(eq_lows) > 0 and bias == 'long'))

        # Volume spike ON DAILY (1D) for hybrid qualification
        vol_ok = volume_spike(d1)

        # Swing/OTE zone from DAILY swings (1D)
        swing_lo, swing_hi = last_major_swing(d1)
        if swing_hi < swing_lo:
            swing_lo, swing_hi = swing_hi, swing_lo
        zone = ote_zone(swing_lo, swing_hi, bias)

        # Price (use 4H last close as local read but OTE is from 1D zone)
        last_price = float(h4['close'].iloc[-1])
        in_zone = price_in_zone(last_price, zone)

        # Near-OTE grace
        NEAR_OTE_ENABLE = True
        NEAR_OTE_PCT = 0.02
        mid = (zone[0] + zone[1]) / 2.0
        dist_to_band = min(abs(last_price - zone[0]), abs(last_price - zone[1])) / max(1e-12, mid)
        in_or_near_zone = in_zone or (NEAR_OTE_ENABLE and dist_to_band <= NEAR_OTE_PCT)

        # PVWAP metric computed on 4H (not gating) for local confluence
        h4['pvwap'] = pvwaps(h4, PVWAP_RESETS_LOCAL)
        pvwap_val = float(h4['pvwap'].iloc[-1])
        vwap_dir = "long" if last_price > pvwap_val else "short"

        # POC magnet over 4H window (local)
        poc = poc_volume_profile(h4.tail(240))
        poc_confluence = "POC_above" if poc > last_price else "POC_below"

        # High-timeframe trend confirmation: require 4H EMA alignment with bias
        h4_latest = h4.iloc[-1]
        h4_latest_ema_fast = ema(h4['close'], EMA_FAST).iloc[-1]
        h4_latest_ema_slow = ema(h4['close'], EMA_SLOW).iloc[-1]
        h4_latest_ema_trend = ema(h4['close'], EMA_TREND).iloc[-1]
        if bias == "long":
            # require 4H close above EMA_SLOW for local confirmation
            if h4_latest['close'] < h4_latest_ema_slow:
                return None
        else:
            if h4_latest['close'] > h4_latest_ema_slow:
                return None

        # Core HYBRID gate: require DAILY sweep + DAILY OTE proximity + local 4H structure
        core_ok = in_or_near_zone and has_manip 
        if not core_ok:
            return None

        # Risk / targets (derived from DAILY swings)
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

        # Confluence tags
        confl = [f"bias:{bias}", f"PVWAP:{vwap_dir}", poc_confluence, "timeframe:HYBRID(1D/4H)"]
        if rng: confl.append("range1d")
        if has_manip: confl.append("sweep_low" if bias=='long' else "sweep_high")
        if len(eq_highs) and bias == 'short': confl.append("eq_highs_4h")
        if len(eq_lows)  and bias == 'long':  confl.append("eq_lows_4h")
        if (bias=='long' and (ob_bull_qual or fvg_up_qual)) or (bias=='short' and (ob_bear_qual or fvg_down_qual)):
            confl.append("FVG/OB_4h")
        if (fvg4h.get('mitigated_up', pd.Series()).any() or fvg4h.get('mitigated_down', pd.Series()).any()):
            confl.append("FVG_mitigated_4h")
        if vol_ok: confl.append("volume_spike_1d")
        if in_zone: confl.append("OTE_1d")
        elif NEAR_OTE_ENABLE and dist_to_band <= NEAR_OTE_PCT: confl.append("OTE_near_1d")

        # SL sanity guard
        if bias == 'long' and stop >= last_price:
            stop = min(stop, zone[0] * 0.995)
        elif bias == 'short' and stop <= last_price:
            stop = max(stop, zone[1] * 1.005)

        print(f"[WHY] {symbol} bias={bias} pvwap_dir={vwap_dir} price={last_price:.6f} pvwap={pvwap_val:.6f} "
              f"in_or_near_ote={in_or_near_zone} has_manip={has_manip} has_struct={has_struct} vol_ok={vol_ok}")

        return Setup(
            symbol=symbol,
            direction=bias,
            entry_zone=zone,
            stop=float(stop),
            tp1=float(tp1),
            tp2=float(tp2),
            confluence=confl,
            price=last_price,
            timestamp=pd.Timestamp.utcnow().tz_convert(TZ),
        )

# ==========================
# ---- Output / Scoring ----
# ==========================

def score_confluence(s: Setup) -> Tuple[float, str]:
    dir_long = s.direction == 'long'
    confl = set(s.confluence)
    vwap_ok = ('PVWAP:long' in confl and dir_long) or ('PVWAP:short' in confl and not dir_long)
    poc_ok  = ('POC_above' in confl and dir_long) or ('POC_below' in confl and not dir_long)
    ote_ok  = ('OTE_1d' in confl)
    manip_ok = ('sweep_low' in confl and dir_long) or ('sweep_high' in confl and not dir_long)
    ob_unmit = any('OB' in c for c in confl)
    fvg_unmit = ('FVG/OB_4h' in confl) and ('FVG_mitigated_4h' not in confl)
    fvg_mit = ('FVG_mitigated_4h' in confl)
    obfvg_score = (WEIGHTS['obfvg_unmitigated'] if fvg_unmit or ob_unmit else 0.0)
    if fvg_mit:
        obfvg_score = max(obfvg_score, WEIGHTS['obfvg_mitigated'])
    vol_ok = ('volume_spike_1d' in confl)

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
        f"*Viv-Hybrid HTF Setup* — *{md_escape(s.symbol)}* ({s.direction.upper()}) {stars}",
        f"Score: {score:.2f}",
        f"Time: {ts}",
        f"Price (4H close): {s.price:.8f}",
        f"Entry (OTE 1D {OTE_LOW:.3f}-{OTE_HIGH:.2f}): {z0:.8f} → {z1:.8f}",
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
        scored = []
        for s in usd_pairs:
            try:
                t = data.fetch_ticker(s)
                quote_vol = t.get('quoteVolume') or (t.get('baseVolume') or 0) * (t.get('last') or 0)
                scored.append((s, float(quote_vol or 0)))
            except Exception:
                scored.append((s, 0.0))
        scored.sort(key=lambda x: x[1], reverse=True)
        filtered = [s for s, v in scored if v >= MIN_USD_VOLUME_24H]
        res = (filtered or [s for s, _ in scored])[:MAX_SYMBOLS]
        return res
    except Exception as e:
        print(f"[DISCOVER] fallback to defaults: {e}")
        return [s for s in DEFAULT_SYMBOLS if s not in EXCLUDE_SYMBOLS][:MAX_SYMBOLS]

def debug_reasons(data: 'Data', symbol: str) -> str:
    try:
        d1 = data.candles(symbol, TIMEFRAMES['1d'], LOOKBACK_BARS['1d'])
        h1 = data.candles(symbol, TIMEFRAMES['1h'], LOOKBACK_BARS['1h'])
        h4 = resample_ohlc(h1, '4H')
    except Exception:
        return "data_error"
    if min(map(len, [d1, h4, h1])) < 60:
        return "data_short"
    bias = current_bias(d1, h4)
    if bias == "neutral":
        return "bias_neutral"
    rng = find_range(d1)
    sweep = detect_sweep(d1, rng)
    has_manip = (sweep['sweep_low'] and bias == 'long') or (sweep['sweep_high'] and bias == 'short')
    fvg4h = detect_fvgs(h4).tail(30)
    ob4h = detect_simple_ob(h4).tail(30)
    sh, sl = swing_points(h4, left=2, right=2)
    eq_highs = list(h4['high'][sh].tail(20).values) if hasattr(sh, 'any') and sh.any() else []
    eq_lows  = list(h4['low'][sl].tail(20).values)  if hasattr(sl, 'any') and sl.any() else []
    fvg_present = ((fvg4h['fvg_up'].any() and bias == 'long') or (fvg4h['fvg_down'].any() and bias == 'short'))
    ob_present = ((ob4h['ob_bull'].any() and bias == 'long') or (ob4h['ob_bear'].any() and bias == 'short'))
    has_ob_or_fvg = ob_present or fvg_present
    has_struct = has_ob_or_fvg or ((len(eq_highs) > 0 and bias == 'short') or (len(eq_lows) > 0 and bias == 'long'))
    swing_lo, swing_hi = last_major_swing(d1)
    zone = ote_zone(swing_lo, swing_hi, bias)
    last_price = float(h4['close'].iloc[-1])
    in_zone = (zone[0] <= last_price <= zone[1])
    vol_ok = volume_spike(d1)
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

        drop = {"bias_neutral":0,"not_in_ote":0,"no_manip":0,"no_struct":0,"no_vol":0,"data_short":0,"data_error":0,"passed_core":0}

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

        print(f"[DEBUG] drop-off counts: {drop}")

        save_state({"cooldowns": scanner.cooldowns, "seen": scanner.seen})
        if not any_sent:
            print(f"[{start.strftime('%H:%M:%S')}] No setups this pass.")
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    run()
